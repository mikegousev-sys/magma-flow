#!/usr/bin/env python3
"""
Оценка скорости течения жидкости по желобу из видео (в пикселях).

Методы (все работают по одному и тому же повёрнутому ROI, течение идёт к -x):
  piv   - блочное сопоставление фрагментов (NCC: грубо на 1/2 разрешения, затем уточнение)
  phase - 2D фазовая корреляция соседних кадров
  farn  - плотный оптический поток Фарнебека (1/2 разрешения), медиана по текстурным пикселям
  lk    - трекинг точек Лукаса-Канаде с проверкой вперёд-назад
  kymo  - 1D-корреляция профилей вдоль желоба за окно времени, несколько лагов

Подготовка кадров:
  - frames.csv рядом с видео (focus_preview) даёт точное время кадров; пары берутся только
    из соседних кадров камеры. Без него (запись экрана) кадры-дубликаты пропускаются,
    а интервал пары = 1 / (уникальных кадров в секунду в окне);
  - яркость нормируется (мерцание автоэкспозиции);
  - ROI поворачивается вдоль течения (угол оценивается по первым кадрам или --angle);
  - вычитается фон: медиана кадров участка + медленное обновление (неподвижные края желоба
    не тянут скорость к нулю).

Зависимости: pip install opencv-python numpy matplotlib

  python flow_speed.py "C:\\...\\видео"                     # самое длинное видео в папке
  python flow_speed.py video.mkv --roi auto                 # ROI по карте движения
  python flow_speed.py video.mp4 --roi 100,200,600,80 --start 5 --max-seconds 60

Результаты: <видео>_speed/{result.json, summary.txt, frames.csv, windows.csv, profile.csv, plot.png, roi*.png}
"""
import argparse
import os
import sys

import cv2
import numpy as np

from common import (GOOD_SPREAD_PCT, Timing, consensus, flow_affine, gray, mask_zero_shift, norm_brightness,
                    parabola, pyplot, read_frames, robust_stats, window_mean, wrap, write_csv, write_json)

VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv", ".mts", ".webm")
METHODS = ("piv", "phase", "farn", "lk", "kymo")
PAIR_METHODS = METHODS[:4]
# столбцы frames.csv: сдвиги (dx, dy) каждого парного метода
COLS = ["frame", "t", "dt", "texture", "phase_resp", "piv_n", "lk_n",
        "piv_dx", "piv_dy", "phase_dx", "phase_dy", "farn_dx", "farn_dy", "lk_dx", "lk_dy"]
C = {c: i for i, c in enumerate(COLS)}
LK_WIN, LK_LEVELS = (21, 21), 5
LK_CRIT = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)


# ---------------------------------------------------------------- видео и ROI

def pick_video(path):
    if os.path.isfile(path):
        return path
    files = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(VIDEO_EXT)]
    if not files:
        sys.exit(f"В папке нет видео: {path}")

    def duration(f):
        t = Timing(f)
        return t.nframes / t.fps
    durs = sorted(((duration(f), f) for f in files), reverse=True)
    for d, f in durs:
        print(f"  {d:8.1f} с  {os.path.basename(f)}")
    print(f"Самое длинное: {os.path.basename(durs[0][1])}")
    return durs[0][1]


def auto_roi(video, tm, samples=120, dup_thr=1.0):
    """ROI по карте движения: средний |I(t)-I(t-1)| по парам соседних кадров,
    разбросанных по участку; берётся рамка самой большой подвижной области."""
    cap = cv2.VideoCapture(video)
    acc, cnt = None, 0
    for j in np.linspace(max(1, tm.lo + 1), max(1, tm.hi - 1), samples).astype(int):
        k = j
        while k < min(tm.hi, j + 400) and not tm.linked(k):
            k += 1
        f1 = read_frames(cap, [k - 1], lambda f: norm_brightness(gray(f).astype(np.float32)))
        ok, f2 = cap.read()
        if not f1 or not ok:
            continue
        d = np.abs(cv2.GaussianBlur(norm_brightness(gray(f2).astype(np.float32)) - f1[0], (0, 0), 2.0))
        if d.mean() >= dup_thr * 0.1:
            acc = d if acc is None else acc + d
            cnt += 1
    cap.release()
    if not cnt:
        return None, None
    m = cv2.GaussianBlur(acc / cnt, (0, 0), 8)
    mask = (m > max(3 * np.median(m), 0.3 * np.percentile(m, 99.5))).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(mask)
    if n < 2:
        return None, m
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    if st[k, cv2.CC_STAT_AREA] < 0.01 * mask.size:
        return None, m
    return tuple(int(v) for v in st[k, :4]), m


def select_roi(frame):
    print("Выделите участок желоба с текущей жидкостью, Enter - подтвердить")
    r = cv2.selectROI("ROI (Enter)", frame, showCrosshair=False)
    cv2.destroyAllWindows()
    if r[2] == 0 or r[3] == 0:
        sys.exit("ROI не выбран")
    return tuple(int(v) for v in r)


# ---------------------------------------------------------------- методы

def to_u8(hp, scale):
    return np.clip(hp * scale + 128.0, 0, 255).astype(np.uint8)


def _match(img, tpl, x0, y0):
    """NCC-поиск шаблона: (x, y) левого верхнего угла с субпикселем и пик."""
    m = cv2.matchTemplate(img, tpl, cv2.TM_CCOEFF_NORMED)
    _, mx, _, (ix, iy) = cv2.minMaxLoc(m)
    fx = ix + (parabola(m[iy, ix - 1], mx, m[iy, ix + 1]) if 0 < ix < m.shape[1] - 1 else 0.0)
    fy = iy + (parabola(m[iy - 1, ix], mx, m[iy + 1, ix]) if 0 < iy < m.shape[0] - 1 else 0.0)
    return x0 + fx, y0 + fy, mx


class Frame:
    """Кадр после вычитания фона и всё, что методы считают по нему один раз
    (используется и как 'текущий', и на следующей паре как 'предыдущий')."""

    def __init__(self, hp, u8, hann, tile):
        self.hp, self.u8 = hp, u8
        self.F = np.fft.rfft2(hp * hann)                              # float32 -> complex64
        self.half = cv2.pyrDown(u8)
        self.pts = cv2.goodFeaturesToTrack(u8, maxCorners=400, qualityLevel=0.01, minDistance=5)
        # std фрагментов PIV на сетке с шагом tile/2 - через интегральные изображения
        s, s2 = cv2.integral2(u8)
        H, W = u8.shape
        ys, xs = np.arange(0, H - tile + 1, tile // 2), np.arange(0, W - tile + 1, tile // 2)
        Y, X = np.meshgrid(ys, xs, indexing="ij")

        def box(ii):
            return ii[Y + tile, X + tile] - ii[Y, X + tile] - ii[Y + tile, X] + ii[Y, X]
        n = tile * tile
        var = box(s2) / n - (box(s) / n) ** 2
        self.grid = [(y, x) for y, x, v in zip(Y.ravel(), X.ravel(), var.ravel()) if v >= 36]  # std >= 6


class PairMethods:
    def __init__(self, w, h, max_shift, min_shift, tile=64, min_ncc=0.6):
        self.hann = cv2.createHanningWindow((w, h), cv2.CV_32F)
        self.max_shift, self.min_shift = max_shift, int(min_shift)
        self.tile, self.min_ncc = tile, min_ncc

    def frame(self, hp, u8):
        return Frame(hp, u8, self.hann, self.tile)

    def phase(self, p, c):
        """2D фазовая корреляция; субпиксель - парабола по x и y."""
        cp = np.conj(p.F) * c.F
        mag = np.abs(cp)
        cp /= mag + 1e-6 * mag.max()
        r = mask_zero_shift(np.fft.irfft2(cp, s=p.hp.shape), self.min_shift)
        h, w = r.shape
        iy, ix = np.unravel_index(int(np.argmax(r)), r.shape)
        dx = ix + parabola(r[iy, (ix - 1) % w], r[iy, ix], r[iy, (ix + 1) % w])
        dy = iy + parabola(r[(iy - 1) % h, ix], r[iy, ix], r[(iy + 1) % h, ix])
        return wrap(dx, w), wrap(dy, h), float(r[iy, ix])

    def piv(self, p, c, tiles_out=None, pred=None):
        """Сетка фрагментов p ищется в c: грубо на 1/2 разрешения, затем уточнение ±4 px.
        Грубый поиск - в окне ±max_shift, а если известен ожидаемый сдвиг pred=(dx, dy)
        (фазовая корреляция или прошлая пара), то вокруг него с запасом ±70% (в разы быстрее,
        медленные слои у стенки в окно всё равно попадают)."""
        H, W = p.u8.shape
        t, t2 = self.tile, self.tile // 2
        if pred is not None and np.all(np.isfinite(pred)) and np.hypot(*pred) < self.max_shift:
            sp = np.hypot(*pred)
            ox, oy = int(round(pred[0] / 2)), int(round(pred[1] / 2))
            rx, ry = int(max(12, 0.7 * sp) / 2) + 2, int(max(8, 0.3 * sp) / 2) + 2
        else:
            ox = oy = 0
            rx = ry = int(self.max_shift / 2) + 2
        out = []
        for y, x in p.grid:
            xs, ys = x // 2, y // 2
            ax0, ay0 = max(0, xs + ox - rx), max(0, ys + oy - ry)
            ax1, ay1 = min(c.half.shape[1], xs + ox + t2 + rx), min(c.half.shape[0], ys + oy + t2 + ry)
            if ax1 - ax0 <= t2 or ay1 - ay0 <= t2:
                continue
            gx, gy, _ = _match(c.half[ay0:ay1, ax0:ax1], p.half[ys:ys + t2, xs:xs + t2], ax0, ay0)
            ex, ey = int(round(gx * 2)), int(round(gy * 2))
            bx0, by0, bx1, by1 = max(0, ex - 4), max(0, ey - 4), min(W, ex + t + 4), min(H, ey + t + 4)
            if bx1 - bx0 <= t or by1 - by0 <= t:
                continue
            fx, fy, ncc = _match(c.u8[by0:by1, bx0:bx1], p.u8[y:y + t, x:x + t], bx0, by0)
            if ncc >= self.min_ncc:
                out.append((fx - x, fy - y))
                if tiles_out is not None:
                    tiles_out.append((y + t2, fx - x))
        if len(out) < 3:
            return np.nan, np.nan, len(out)
        d = np.median(np.array(out), axis=0)
        return float(d[0]), float(d[1]), len(out)

    @staticmethod
    def farneback(p, c):
        """Farneback на 1/2 разрешения (в 4 раза дешевле, для больших сдвигов не хуже);
        медиана по 30% самых текстурных пикселей текущего кадра."""
        flow = cv2.calcOpticalFlowFarneback(p.half, c.half, None, 0.5, 5, 15, 3, 5, 1.2, 0)
        g = c.half.astype(np.float32)
        gm = cv2.Sobel(g, cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(g, cv2.CV_32F, 0, 1) ** 2
        mask = gm > np.percentile(gm[::2, ::2], 70)
        if mask.sum() < 50:
            return np.nan, np.nan
        return 2 * float(np.median(flow[..., 0][mask])), 2 * float(np.median(flow[..., 1][mask]))

    @staticmethod
    def lk(p, c):
        """Лукас-Канаде с проверкой вперёд-назад; углы посчитаны в Frame (Python-биндинг
        OpenCV не принимает готовые пирамиды, поэтому их строит сам calcOpticalFlowPyrLK)."""
        if p.pts is None or len(p.pts) < 10:
            return np.nan, np.nan, 0
        kw = dict(winSize=LK_WIN, maxLevel=LK_LEVELS, criteria=LK_CRIT)
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(p.u8, c.u8, p.pts, None, **kw)
        back, st2, _ = cv2.calcOpticalFlowPyrLK(c.u8, p.u8, nxt, None, **kw)
        fb = np.linalg.norm((back - p.pts).reshape(-1, 2), axis=1)
        ok = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 0.5)
        if ok.sum() < 10:
            return np.nan, np.nan, int(ok.sum())
        d = np.median((nxt - p.pts).reshape(-1, 2)[ok], axis=0)
        return float(d[0]), float(d[1]), int(ok.sum())


def m_kymo(runs, max_lag_shift, min_shift=0):
    """runs: непрерывные отрезки (T, B, W) профилей вдоль желоба (B полос по высоте ROI).
    Корреляции суммируются по кадрам окна и полосам (кадры без текстуры почти не влияют),
    несколько лагов 1, 2, 4... -> сдвиг за 1 шаг кадров (dx)."""
    runs = [r for r in runs if len(r) >= 2]
    if sum(len(r) - 1 for r in runs) < 3:
        return np.nan
    W = runs[0].shape[2]
    Fs = [np.fft.rfft(r * np.hanning(W), axis=2) for r in runs]
    T = max(len(r) for r in runs)

    def shift_for_lag(k):
        cp = sum((np.conj(F[:-k]) * F[k:]).sum(axis=(0, 1)) for F in Fs if len(F) > k)
        cp /= np.abs(cp) + 1e-9 * np.abs(cp).max()
        acc = mask_zero_shift(np.fft.irfft(cp, n=W), int(min_shift))
        i = int(np.argmax(acc))
        return wrap(i + parabola(acc[(i - 1) % W], acc[i], acc[(i + 1) % W]), W)

    s1 = shift_for_lag(1)
    lags, shifts = [1], [s1]
    k = 2
    while k <= T // 2 and abs(s1) * k < max_lag_shift:
        s = shift_for_lag(k)
        if abs(s - s1 * k) > max(2.0, 0.25 * abs(s1 * k)):
            break                                   # лаг "перескочил" или текстура распалась
        lags.append(k)
        shifts.append(s)
        k *= 2
    lags, shifts = np.array(lags, float), np.array(shifts, float)
    return float((lags * shifts).sum() / (lags ** 2).sum())


# ---------------------------------------------------------------- этапы

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="видеофайл или папка (берётся самое длинное видео)")
    ap.add_argument("--roi", help="x,y,w,h в пикселях кадра, или auto (по карте движения)")
    ap.add_argument("--start", type=float, default=0.0, help="начало анализа, с")
    ap.add_argument("--max-seconds", type=float, default=0.0, help="длительность анализа, с (0 = всё)")
    ap.add_argument("--win", type=float, default=1.0, help="окно усреднения, с")
    ap.add_argument("--bg-tau", type=float, default=2.0, help="постоянная времени обновления фона, с (0 = не вычитать)")
    ap.add_argument("--bands", type=int, default=4, help="число полос по высоте для kymo")
    ap.add_argument("--dup-thr", type=float, default=1.0,
                    help="средняя разница яркости, ниже которой кадр считается дубликатом (запись экрана)")
    ap.add_argument("--display-scale", type=float, default=1.0,
                    help="масштаб отображения камеры в записи экрана (0.91 = 91%%): пересчёт в пиксели камеры")
    ap.add_argument("--angle", type=float,
                    help="направление течения, градусы от горизонтали (0 = налево, + = вверх-влево); иначе авто")
    ap.add_argument("--max-shift", type=float, help="макс. сдвиг за кадр для PIV, px (по умолч. w/3)")
    ap.add_argument("--min-shift", type=float, default=2.0,
                    help="phase/kymo: игнорировать пик корреляции ближе этого сдвига к нулю, px (остатки фона); "
                         "для очень медленного течения (<3 px/кадр) ставьте 0")
    ap.add_argument("--frames-csv", help="метки времени кадров (frames.csv рядом с видео подхватывается сам)")
    ap.add_argument("--out", help="папка результатов")
    return ap.parse_args()


def make_prep(roi, angle):
    x, y, w, h = roi
    A = flow_affine(x + w / 2.0, y + h / 2.0, angle, w, h)

    def prep(f):
        g = cv2.warpAffine(gray(f), A, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP).astype(np.float32)
        return norm_brightness(cv2.GaussianBlur(g, (0, 0), 1.0))
    return prep


def estimate_angle(cap, tm, roi, max_shift, dup_thr):
    """PIV по первым уникальным кадрам без поворота; фон = их временная медиана."""
    prep0 = make_prep(roi, 0.0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, tm.lo)
    uniq, linked, j = [], [], tm.lo - 1
    for _ in range(int(tm.fps * 10)):
        ok, f = cap.read()
        j += 1
        if not ok or len(uniq) >= 40:
            break
        g = prep0(f)
        if tm.has_ts or not uniq or np.abs(g - uniq[-1]).mean() >= dup_thr:
            linked.append(bool(uniq) and tm.linked(j))
            uniq.append(g)
    est = []
    if len(uniq) >= 6:
        bg = np.median(np.array(uniq), axis=0)
        sc = 40.0 / (np.std(uniq[0] - bg) + 1e-3)
        pm = PairMethods(*uniq[0].shape[::-1], max_shift, 0)
        fr = [pm.frame(g - bg, to_u8(g - bg, sc)) for g in uniq]
        for k in range(1, len(fr)):
            if linked[k]:
                dx, dy, n = pm.piv(fr[k - 1], fr[k])
                if n >= 5:
                    est.append((dx, dy))
    if len(est) < 5:
        sys.exit("Не удалось оценить направление течения, задайте --angle")
    e = np.median(np.array(est), axis=0)
    ang = float(np.degrees(np.arctan2(-e[1], -e[0])))
    print(f"Направление течения (авто): {ang:+.1f}°, сдвиг ~{np.hypot(*e):.1f} px/кадр")
    return ang


def track(cap, tm, prep, a, w, h):
    """Основной проход. Возвращает строки frames.csv, профили для kymo, фрагменты PIV для
    профиля поперёк потока и счётчики."""
    band_edges = np.linspace(0, h, a.bands + 1).astype(int)
    pm = PairMethods(w, h, a.max_shift or w / 3, a.min_shift)
    # начальный фон - медиана кадров, равномерно взятых по участку (текстура усредняется)
    smp = read_frames(cap, np.linspace(tm.lo, tm.hi - 1, 41).astype(int), prep)
    bg = np.median(np.array(smp), axis=0).astype(np.float32)
    alpha = 1.0 / (a.bg_tau * tm.fps) if a.bg_tau > 0 else 0.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, tm.lo)
    rows, profiles, tiles = [], [], []
    prev = g_last = scale = t_prev = None
    ndup = nbreak = 0
    for fi in range(tm.lo, tm.hi):
        ok, f = cap.read()
        if not ok:
            break
        g = prep(f)
        # запись экрана: картинка не обновилась - пара дала бы нулевое смещение
        if not tm.has_ts and g_last is not None and np.abs(g - g_last).mean() < a.dup_thr:
            ndup += 1
            continue
        g_last = g
        if alpha > 0:
            cv2.accumulateWeighted(g, bg, alpha)
            hp = g - bg
        else:
            hp = g - g.mean()
        if scale is None:
            scale = 40.0 / (hp.std() + 1e-3)
        t = tm.time(fi)
        linked = prev is not None and tm.linked(fi)
        profiles.append((t, linked, [hp[band_edges[b]:band_edges[b + 1]].mean(axis=0) for b in range(a.bands)]))
        cur = pm.frame(hp, to_u8(hp, scale))
        if linked:
            pdx, pdy, presp = pm.phase(prev, cur)
            # ожидаемый сдвиг для PIV: прошлая пара PIV (надёжнее), иначе фазовая корреляция
            pred = rows[-1][C["piv_dx"]:C["piv_dy"] + 1] if rows and np.isfinite(rows[-1][C["piv_dx"]]) else (pdx, pdy)
            tl = []
            vdx, vdy, vn = pm.piv(prev, cur, tl, pred)
            tiles += [(len(rows), cy, dx) for cy, dx in tl]
            fdx, fdy = pm.farneback(prev, cur)
            ldx, ldy, ln = pm.lk(prev, cur)
            dt = t - t_prev if tm.has_ts else np.nan
            rows.append([fi, t, dt, float(hp.std()), presp, vn, ln, vdx, vdy, pdx, pdy, fdx, fdy, ldx, ldy])
            if len(rows) % int(tm.fps * 5) == 0:
                print(f"  {t:6.1f} с  piv={vdx:7.2f},{vdy:6.2f} lk={ldx:7.2f},{ldy:6.2f} px/кадр")
        elif prev is not None:
            nbreak += 1
        prev, t_prev = cur, t
    return np.array(rows, float), profiles, tiles, ndup, nbreak


def window_speeds(R, profiles, tm, a, w):
    """Скорости по окнам времени, px/с. Без меток времени интервал пары = 1 / (уникальных
    кадров в секунду в её окне), дальше обе ветки считают одинаково: сдвиг / dt."""
    tp = np.array([p[0] for p in profiles])
    lp = np.array([p[1] for p in profiles])
    P = np.array([p[2] for p in profiles], np.float32)
    edges = np.arange(tp[0], tp[-1] + 1e-9, a.win)
    wsel = [((R[:, C["t"]] >= e0) & (R[:, C["t"]] < e0 + a.win), (tp >= e0) & (tp < e0 + a.win))
            for e0 in edges[:-1]]
    rate = np.array([ps.sum() / a.win for _, ps in wsel])
    if not tm.has_ts:
        for (s, _), r in zip(wsel, rate):
            R[s, C["dt"]] = 1.0 / r if r else np.nan
    DT = R[:, C["dt"]]
    resp_ok = R[:, C["phase_resp"]] >= 0.5 * np.median(R[:, C["phase_resp"]])
    # после поворота течение идёт вдоль -x: скорость = -dx / dt
    speed = {m: -R[:, C[f"{m}_dx"]] / DT for m in PAIR_METHODS}
    speed["phase"] = np.where(resp_ok, speed["phase"], np.nan)
    wins = {m: np.full(len(wsel), np.nan) for m in METHODS}
    for k, (s, ps) in enumerate(wsel):
        for m in PAIR_METHODS:
            v = speed[m][s]
            if np.isfinite(v).sum() >= max(3, 0.3 * s.sum()):
                wins[m][k] = window_mean(v)
        idx = np.flatnonzero(ps)
        if len(idx) >= 6 and s.any():
            runs = np.split(P[idx], np.flatnonzero(~lp[idx][1:]) + 1)
            wins["kymo"][k] = -m_kymo(runs, w / 3, a.min_shift) / np.nanmedian(DT[s])
    wt = edges[:-1] + a.win / 2
    return wt, rate, wins, speed, resp_ok


def cross_profile(R, tiles):
    """Профиль скорости поперёк потока: PIV-фрагменты по полосам (y в повёрнутом ROI)."""
    by = {}
    for row, cy, dx in tiles:
        by.setdefault(cy, []).append(-dx / R[row, C["dt"]])
    nmax = max((len(v) for v in by.values()), default=0)
    prof = []
    for cy in sorted(by):
        v = np.array(by[cy])
        v = v[np.isfinite(v)]
        if len(v) >= max(30, 0.5 * nmax):              # краевые полосы с редкой текстурой не берём
            prof.append((int(cy), len(v), window_mean(v), float(np.percentile(v, 25)), float(np.percentile(v, 75))))
    return prof


def main():
    a = parse_args()
    video = pick_video(a.path)
    tm = Timing(video, a.frames_csv, a.start, a.max_seconds)
    cap = cv2.VideoCapture(video)
    frame = (read_frames(cap, [tm.lo]) + [None])[0]
    if frame is None:
        sys.exit("Не удалось прочитать кадр")
    print(f"{os.path.basename(video)}: {frame.shape[1]}x{frame.shape[0]}, {tm.nframes} кадров, "
          f"участок {tm.lo}..{tm.hi}; {tm.describe()}")
    out = a.out or os.path.splitext(video)[0] + "_speed"
    os.makedirs(out, exist_ok=True)

    if a.roi == "auto":
        roi, mmap = auto_roi(video, tm, dup_thr=a.dup_thr)
        if mmap is not None:
            cv2.imwrite(os.path.join(out, "motion_map.png"), cv2.normalize(mmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
        if roi is None:
            sys.exit("Движущаяся область не найдена (потока в кадре нет?)")
    else:
        roi = tuple(map(int, a.roi.split(","))) if a.roi else select_roi(frame)
    x, y, w, h = roi
    print(f"ROI: --roi {x},{y},{w},{h}")
    cv2.imwrite(os.path.join(out, "roi.png"), cv2.rectangle(frame.copy(), (x, y), (x + w, y + h), (0, 0, 255), 2))
    if a.angle is None:
        a.angle = estimate_angle(cap, tm, roi, a.max_shift or w / 3, a.dup_thr)
    prep = make_prep(roi, a.angle)
    cv2.imwrite(os.path.join(out, "roi_rotated.png"), prep(frame).clip(0, 255).astype(np.uint8))

    R, profiles, tiles, ndup, nbreak = track(cap, tm, prep, a, w, h)
    cap.release()
    if len(R) < 10:
        sys.exit("Слишком мало кадров для анализа")
    wt, rate, wins, speed, resp_ok = window_speeds(R, profiles, tm, a, w)
    write_csv(os.path.join(out, "frames.csv"), COLS, R.tolist())
    prof = cross_profile(R, tiles)
    write_csv(os.path.join(out, "profile.csv"), ["y_rot_px", "n", "px_per_s", "p25", "p75"], prof)

    # консенсус методов: по каждому окну и по медианам окон
    wc = [consensus([wins[m][k] for m in METHODS])[:2] for k in range(len(wt))]
    wcons, wspread = np.array([c[0] for c in wc]), np.array([c[1] for c in wc])
    k_cam = 1.0 / a.display_scale
    write_csv(os.path.join(out, "windows.csv"),
              ["t", "unique_fps"] + [f"{m}_px_per_s" for m in METHODS] + [f"{m}_campx_per_s" for m in METHODS]
              + ["consensus_px_per_s", "spread_pct"],
              [[f"{wt[k]:.3f}", f"{rate[k]:.1f}"] + [f"{wins[m][k]:.2f}" for m in METHODS]
               + [f"{wins[m][k] * k_cam:.2f}" for m in METHODS] + [f"{wcons[k]:.2f}", f"{wspread[k]:.1f}"]
               for k in range(len(wt))])
    stats = {m: robust_stats(wins[m]) for m in METHODS}
    cons, spread, keep = consensus([stats[m]["median"] for m in METHODS])
    good_w = np.isfinite(wspread) & (wspread <= GOOD_SPREAD_PCT)
    sc = robust_stats(wcons[good_w])
    resid = float(np.degrees(np.arctan2(-np.nanmedian(R[:, C["piv_dy"]]), -np.nanmedian(R[:, C["piv_dx"]]))))
    pv = np.array([q[2] for q in prof])

    def cam(v):
        return f" = {v * k_cam:.1f} px/с камеры" if k_cam != 1 else ""
    dt_ms = np.nanmedian(R[:, C["dt"]]) * 1000
    lines = [f"Видео: {video}",
             (f"Метки времени: {tm.csv_path}; пар соседних кадров: {len(R)}, разрывов серий: {nbreak}, "
              f"интервал пары (медиана): {dt_ms:.2f} мс") if tm.has_ts else
             (f"FPS записи: {tm.fps:.3f}; пар: {len(R)}, дубликатов пропущено: {ndup}; "
              f"уникальных в секунду (медиана по окнам): {np.median(rate):.1f}"),
             f"ROI: {x},{y},{w},{h}; окно {a.win:g} с",
             f"Направление течения: {a.angle:+.1f}° от горизонтали (0 = налево, + = вверх-влево); "
             f"остаточный угол после поворота (PIV): {resid:+.1f}°", "",
             "Скорость вдоль течения, px/с, статистика по окнам:",
             f"{'метод':6} {'окон':>5} {'медиана':>9} {'MAD':>7} {'CV':>6} {'P10..P90':>16}"]
    lines += [f"{m:6} {s['n']:5d} {s['median']:9.1f} {s['mad']:7.2f} {s['cv'] * 100:5.1f}% "
              f"{s['p10']:7.1f}..{s['p90']:<7.1f}{cam(s['median'])}" for m, s in stats.items()]
    bad = [m for m, k in zip(METHODS, keep) if not k]
    lines += ["", f"Согласованные методы: {', '.join(m for m, k in zip(METHODS, keep) if k)}"
              + (f"; отброшены: {', '.join(bad)}" if bad else ""),
              f"Консенсус (медиана согласованных): {cons:.1f} px/с{cam(cons)}",
              f"Расхождение согласованных методов (max-min): {spread:.1f}%"]
    if prof:
        lines += ["", "Профиль скорости поперёк потока (PIV; y - поперёк течения в повёрнутом ROI):"]
        lines += [f"  y={q[0]:4d}  n={q[1]:6d}  {q[2]:8.1f} px/с  (P25..P75 {q[3]:.0f}..{q[4]:.0f})" for q in prof]
        lines += [f"  максимум по профилю (ядро потока): {pv.max():.1f} px/с{cam(pv.max())}",
                  f"  среднее по видимой ширине потока:  {pv.mean():.1f} px/с{cam(pv.mean())}",
                  f"  неравномерность (мин/макс): {pv.min() / pv.max():.2f}"]
    lines += ["", f"Окна, где согласованные методы расходятся <={GOOD_SPREAD_PCT:g}%: {int(good_w.sum())} из {len(wt)}",
              f"  скорость по ним: медиана {sc['median']:.1f} px/с{cam(sc['median'])}, разброс между окнами "
              f"(MAD) {sc['mad']:.1f} px/с = {sc['cv'] * 100:.1f}%, P10..P90 {sc['p10']:.0f}..{sc['p90']:.0f}",
              "Поперечная составляющая (piv/phase/farn/lk), px/кадр: "
              + " / ".join(f"{np.nanmedian(R[:, C[f'{m}_dy']]):.3f}" for m in PAIR_METHODS) + "  (должна быть ~0)",
              f"Отброшено пар phase (слабый пик): {int((~resp_ok).sum())} из {len(R)}"]
    txt = "\n".join(lines)
    print("\n" + txt)
    with open(os.path.join(out, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    write_json(os.path.join(out, "result.json"), dict(
        video=os.path.abspath(video), roi=[x, y, w, h], angle=a.angle, residual_angle=resid,
        start=a.start, seconds=a.max_seconds, win=a.win, has_timestamps=tm.has_ts, display_scale=a.display_scale,
        pairs=len(R), breaks=nbreak, duplicates=ndup, pair_dt_ms=dt_ms,
        methods={m: stats[m] for m in METHODS}, agreed=[m for m, k in zip(METHODS, keep) if k],
        consensus_px_s=cons, methods_spread_pct=spread, windows=len(wt), good_windows=int(good_w.sum()),
        good_windows_speed=sc, core_px_s=pv.max() if prof else None, width_mean_px_s=pv.mean() if prof else None))

    plt = pyplot()
    if plt:
        fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        for m in PAIR_METHODS:
            ax[0].plot(R[:, C["t"]], speed[m], ".", ms=2, alpha=0.4, label=m)
        ax[0].set_ylabel("по кадрам, px/с")
        ax[0].legend(markerscale=5)
        for m in METHODS:
            ax[1].plot(wt, wins[m], "-o", ms=3, label=m)
        ax[1].plot(wt[good_w], wcons[good_w], "k*", ms=9, label=f"консенсус (<={GOOD_SPREAD_PCT:g}%)")
        ax[1].set_ylabel(f"окно {a.win:g} с, px/с")
        ax[1].legend()
        tex = R[:, C["texture"]]
        ax[2].plot(R[:, C["t"]], tex, lw=0.7, label="текстура (std)")
        ax[2].plot(R[:, C["t"]], R[:, C["phase_resp"]] * tex.max(), lw=0.7, label="phase resp (масшт.)")
        ax[2].set_xlabel("t, с")
        ax[2].legend()
        for q in ax:
            q.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "plot.png"), dpi=120)
    print(f"\nРезультаты: {out}")


if __name__ == "__main__":
    main()
