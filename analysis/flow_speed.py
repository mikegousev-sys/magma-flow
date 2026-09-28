#!/usr/bin/env python3
"""
Оценка скорости течения жидкости по желобу из видео (в пикселях).

Жидкость течёт справа налево => смещение текстуры dx < 0.
Скорость выводится как положительная величина vel = -dx (px/кадр и px/с).

Методы (все работают по одной и той же области ROI):
  phase   - 2D фазовая корреляция соседних кадров
  farn    - плотный оптический поток Фарнебека, медиана dx по текстурным пикселям
  lk      - трекинг точек Лукаса-Канаде с проверкой вперёд-назад, медиана dx
  piv     - блочное сопоставление фрагментов (NCC, грубо на 1/2 разрешения,
            затем уточнение), медиана по фрагментам
  kymo    - 1D-корреляция профилей вдоль желоба, накопленная за окно времени,
            несколько лагов (1, 2, 4 ... кадров) -> наклон v = shift / lag

Подготовка кадров:
  - дубликаты (запись экрана: картинка обновляется реже, чем пишет рекордер)
    пропускаются; скорость в px/с = сдвиг за уникальный кадр * уникальных кадров в секунду;
  - яркость нормируется (мерцание автоэкспозиции);
  - ROI поворачивается так, чтобы течение шло строго справа налево
    (угол оценивается автоматически по первым секундам или задаётся --angle);
  - вычитается медленно обновляемый фон (EMA), чтобы неподвижные детали
    (края желоба, разметка) не тянули скорость к нулю.

Зависимости: pip install opencv-python numpy matplotlib

Использование:
  python flow_speed.py "C:\\...\\Flowmeter\\видео"          # выберет самое длинное видео
  python flow_speed.py video.mp4 --roi 100,200,600,80       # x,y,w,h
  python flow_speed.py video.mp4 --start 5 --max-seconds 60

Без --roi откроется окно выбора области (мышью, затем Enter).
Результаты: <имя видео>_speed/{frames.csv, windows.csv, summary.txt, plot.png, roi.png}
"""
import argparse
import csv
import os
import sys

import cv2
import numpy as np

VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv", ".mts", ".webm")
METHODS = ("piv", "phase", "farn", "lk", "kymo")


# ---------------------------------------------------------------- видео

def video_duration(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return -1.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return n / fps if fps > 0 else -1.0


def pick_video(path):
    if os.path.isfile(path):
        return path
    files = [os.path.join(path, f) for f in os.listdir(path)
             if f.lower().endswith(VIDEO_EXT)]
    if not files:
        sys.exit(f"В папке нет видео: {path}")
    durs = sorted(((video_duration(f), f) for f in files), reverse=True)
    for d, f in durs:
        print(f"  {d:8.1f} с  {os.path.basename(f)}")
    print(f"Самое длинное: {os.path.basename(durs[0][1])}")
    return durs[0][1]


def parse_roi(s):
    x, y, w, h = (int(v) for v in s.split(","))
    return x, y, w, h


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


def _parabola(y0, y1, y2):
    den = y0 - 2 * y1 + y2
    return 0.5 * (y0 - y2) / den if den != 0 else 0.0


def m_phase(prev, cur, win, min_shift=0):
    """2D фазовая корреляция; субпиксель - парабола по x и y
    (cv2.phaseCorrelate с центроидом даёт систематическое смещение ~3%)."""
    A = np.fft.rfft2(prev * win)
    B = np.fft.rfft2(cur * win)
    cp = np.conj(A) * B
    cp /= np.abs(cp) + 1e-6 * np.abs(cp).max()
    c = np.fft.irfft2(cp, s=prev.shape)
    h, w = c.shape
    if min_shift > 0:   # пик нулевого сдвига = остатки неподвижного фона
        r = int(min_shift)
        c = c.copy()
        c[np.r_[0:r + 1, h - r:h][:, None], np.r_[0:r + 1, w - r:w]] = c.min()
    iy, ix = np.unravel_index(int(np.argmax(c)), c.shape)
    dx = ix + _parabola(c[iy, (ix - 1) % w], c[iy, ix], c[iy, (ix + 1) % w])
    dy = iy + _parabola(c[(iy - 1) % h, ix], c[iy, ix], c[(iy + 1) % h, ix])
    if dx > w / 2:
        dx -= w
    if dy > h / 2:
        dy -= h
    return float(dx), float(dy), float(c[iy, ix])


def m_farneback(prev8, cur8, grad_mask):
    flow = cv2.calcOpticalFlowFarneback(prev8, cur8, None, 0.5, 6, 21, 3, 7, 1.5, 0)
    if grad_mask.sum() < 50:
        return np.nan, np.nan
    return float(np.median(flow[..., 0][grad_mask])), float(np.median(flow[..., 1][grad_mask]))


LK_PARAMS = dict(winSize=(21, 21), maxLevel=5,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


def m_lk(prev8, cur8):
    pts = cv2.goodFeaturesToTrack(prev8, maxCorners=400, qualityLevel=0.01, minDistance=5)
    if pts is None or len(pts) < 10:
        return np.nan, np.nan, 0
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev8, cur8, pts, None, **LK_PARAMS)
    back, st2, _ = cv2.calcOpticalFlowPyrLK(cur8, prev8, nxt, None, **LK_PARAMS)
    fb = np.linalg.norm((back - pts).reshape(-1, 2), axis=1)
    ok = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 0.5)
    if ok.sum() < 10:
        return np.nan, np.nan, int(ok.sum())
    d = (nxt - pts).reshape(-1, 2)[ok]
    return float(np.median(d[:, 0])), float(np.median(d[:, 1])), int(ok.sum())


def _match(img, tpl, x0, y0):
    """NCC-поиск шаблона; возвращает (x, y) левого верхнего угла с субпикселем и пик."""
    m = cv2.matchTemplate(img, tpl, cv2.TM_CCOEFF_NORMED)
    _, mx, _, (ix, iy) = cv2.minMaxLoc(m)
    fx = ix + (_parabola(m[iy, ix - 1], mx, m[iy, ix + 1]) if 0 < ix < m.shape[1] - 1 else 0.0)
    fy = iy + (_parabola(m[iy - 1, ix], mx, m[iy + 1, ix]) if 0 < iy < m.shape[0] - 1 else 0.0)
    return x0 + fx, y0 + fy, mx


def m_piv(prev8, cur8, max_shift, tile=64, min_ncc=0.6):
    """Блочное сопоставление: сетка фрагментов prev ищется в cur.
    Грубый поиск на половинном разрешении в окне ±max_shift, затем уточнение ±4 px."""
    H, W = prev8.shape
    p2, c2 = cv2.pyrDown(prev8), cv2.pyrDown(cur8)
    t2, S2 = tile // 2, int(max_shift / 2) + 2
    out = []
    for cy in range(tile // 2, H - tile // 2 + 1, tile // 2):
        for cx in range(tile // 2, W - tile // 2 + 1, tile // 2):
            x, y = cx - tile // 2, cy - tile // 2
            tpl = prev8[y:y + tile, x:x + tile]
            if tpl.std() < 6:
                continue
            xs, ys = x // 2, y // 2
            ax0, ay0 = max(0, xs - S2), max(0, ys - S2)
            ax1, ay1 = min(c2.shape[1], xs + t2 + S2), min(c2.shape[0], ys + t2 + S2)
            if ax1 - ax0 <= t2 or ay1 - ay0 <= t2:
                continue
            gx, gy, _ = _match(c2[ay0:ay1, ax0:ax1], p2[ys:ys + t2, xs:xs + t2], ax0, ay0)
            ex, ey = int(round(gx * 2)), int(round(gy * 2))
            bx0, by0 = max(0, ex - 4), max(0, ey - 4)
            bx1, by1 = min(W, ex + tile + 4), min(H, ey + tile + 4)
            if bx1 - bx0 <= tile or by1 - by0 <= tile:
                continue
            fx, fy, ncc = _match(cur8[by0:by1, bx0:bx1], tpl, bx0, by0)
            if ncc >= min_ncc:
                out.append((fx - x, fy - y))
    if len(out) < 3:
        return np.nan, np.nan, len(out)
    d = np.array(out)
    return float(np.median(d[:, 0])), float(np.median(d[:, 1])), len(out)


def _xcorr_peak(acc, min_shift=0):
    """Пик циклической корреляции с субпиксельным уточнением (парабола)."""
    n = acc.size
    if min_shift > 0:
        r = int(min_shift)
        acc = acc.copy()
        acc[np.r_[0:r + 1, n - r:n]] = acc.min()
    i = int(np.argmax(acc))
    s = i + _parabola(acc[(i - 1) % n], acc[i], acc[(i + 1) % n])
    if s > n / 2:
        s -= n
    return s


def m_kymo(profiles, max_lag_shift, min_shift=0):
    """
    profiles: (T, B, W) - профили яркости вдоль желоба, B полос по высоте ROI.
    Корреляции суммируются по всем кадрам окна и полосам -> устойчиво к
    кадрам без текстуры (они вносят мало энергии).
    Возвращает сдвиг за 1 кадр (dx).
    """
    T, B, W = profiles.shape
    if T < 3:
        return np.nan
    taper = np.hanning(W)
    F = np.fft.rfft(profiles * taper, axis=2)

    def shift_for_lag(k):
        cp = (np.conj(F[:-k]) * F[k:]).sum(axis=(0, 1))
        cp /= np.abs(cp) + 1e-9 * np.abs(cp).max()      # фазовая (нормированная)
        return _xcorr_peak(np.fft.irfft(cp, n=W), min_shift)

    s1 = shift_for_lag(1)
    lags, shifts = [1], [s1]
    k = 2
    while k < T // 2 and abs(s1) * k < max_lag_shift:
        s = shift_for_lag(k)
        if abs(s - s1 * k) > max(2.0, 0.25 * abs(s1 * k)):
            break  # лаг "перескочил" или текстура распалась
        lags.append(k)
        shifts.append(s)
        k *= 2
    lags = np.array(lags, float)
    shifts = np.array(shifts, float)
    return float((lags * shifts).sum() / (lags ** 2).sum())


# ---------------------------------------------------------------- анализ

def window_mean(v):
    """Среднее после отбраковки грубых выбросов (0.5..1.6 x медиана).
    Не медиана: при записи экрана между показанными кадрами проходит то 4, то 5
    кадров камеры, и медиана берёт только самую частую группу (занижение ~4%)."""
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan
    med = np.median(v)
    lo, hi = sorted((0.5 * med, 1.6 * med))
    k = v[(v >= lo) & (v <= hi)]
    return float(k.mean()) if k.size else np.nan


def robust_stats(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return dict(n=0, median=np.nan, mad=np.nan, p10=np.nan, p90=np.nan, cv=np.nan)
    med = float(np.median(v))
    mad = float(1.4826 * np.median(np.abs(v - med)))
    return dict(n=int(v.size), median=med, mad=mad,
                p10=float(np.percentile(v, 10)), p90=float(np.percentile(v, 90)),
                cv=mad / abs(med) if med != 0 else np.nan)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="видеофайл или папка (берётся самое длинное видео)")
    ap.add_argument("--roi", help="x,y,w,h в пикселях исходного кадра")
    ap.add_argument("--start", type=float, default=0.0, help="начало анализа, с")
    ap.add_argument("--max-seconds", type=float, default=0.0, help="длительность анализа, с (0 = всё)")
    ap.add_argument("--win", type=float, default=1.0, help="окно усреднения, с")
    ap.add_argument("--bg-tau", type=float, default=2.0, help="постоянная времени фона, с (0 = не вычитать)")
    ap.add_argument("--bands", type=int, default=4, help="число полос по высоте для kymo")
    ap.add_argument("--dup-thr", type=float, default=1.0,
                    help="средняя разница яркости, ниже которой кадр считается дубликатом "
                         "(запись экрана: браузер обновляет картинку реже, чем пишет рекордер)")
    ap.add_argument("--display-scale", type=float, default=1.0,
                    help="масштаб отображения камеры в записи (0.91 = 91%%); "
                         "скорость дополнительно пересчитывается в пиксели камеры")
    ap.add_argument("--angle", type=float,
                    help="направление течения, градусы от горизонтали (0 = налево, + = вверх-влево); "
                         "по умолчанию оценивается автоматически")
    ap.add_argument("--max-shift", type=float, help="макс. сдвиг за кадр для PIV, px (по умолч. w/3)")
    ap.add_argument("--min-shift", type=float, default=2.0,
                    help="phase/kymo: игнорировать пик корреляции ближе этого сдвига к нулю, px "
                         "(остатки неподвижного фона); 0 = не игнорировать. "
                         "Для очень медленного течения (<3 px/кадр) ставьте 0")
    ap.add_argument("--out", help="папка результатов")
    a = ap.parse_args()

    video = pick_video(a.path)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not fps or fps <= 0:
        sys.exit("Не удалось определить FPS")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(a.start * fps))
    ok, frame = cap.read()
    if not ok:
        sys.exit("Не удалось прочитать кадр")
    print(f"{os.path.basename(video)}: {frame.shape[1]}x{frame.shape[0]}, {fps:.2f} fps, {nframes} кадров")

    roi = parse_roi(a.roi) if a.roi else select_roi(frame)
    x, y, w, h = roi
    print(f"ROI: --roi {x},{y},{w},{h}")

    out = a.out or os.path.splitext(video)[0] + "_speed"
    os.makedirs(out, exist_ok=True)
    vis = frame.copy()
    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imwrite(os.path.join(out, "roi.png"), vis)

    limit = int(a.max_seconds * fps) if a.max_seconds > 0 else None
    alpha = 1.0 / (a.bg_tau * fps) if a.bg_tau > 0 else 0.0
    hann = cv2.createHanningWindow((w, h), cv2.CV_32F)
    band_edges = np.linspace(0, h, a.bands + 1).astype(int)
    max_shift = a.max_shift or w / 3
    cx0, cy0 = x + w / 2.0, y + h / 2.0

    def make_prep(ang_deg):
        # выход: окно w x h с центром в центре ROI, ось x направлена против течения
        t = np.radians(ang_deg)
        e1 = np.array([np.cos(t), np.sin(t)])          # против течения
        e2 = np.array([-e1[1], e1[0]])
        A = np.array([[e1[0], e2[0], cx0 - e1[0] * w / 2 - e2[0] * h / 2],
                      [e1[1], e2[1], cy0 - e1[1] * w / 2 - e2[1] * h / 2]], np.float32)

        def prep(fr):
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            g = cv2.warpAffine(g, A, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
            g = cv2.GaussianBlur(g, (0, 0), 1.0)
            return g * (100.0 / (g.mean() + 1e-3))     # нормировка яркости (автоэкспозиция)
        return prep

    def is_dup(g, g_last):
        return float(np.abs(g - g_last).mean()) < a.dup_thr

    if a.angle is None:
        # предварительный проход: PIV по первым уникальным кадрам без поворота,
        # фон = временная медиана этих кадров (убирает неподвижные края)
        prep0 = make_prep(0.0)
        pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        last, uniq = prep0(frame), []
        for _ in range(int(fps * 10)):
            ok, fr = cap.read()
            if not ok or len(uniq) >= 40:
                break
            g = prep0(fr)
            if not is_dup(g, last):
                uniq.append(g)
                last = g
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        est = []
        if len(uniq) >= 6:
            bg0 = np.median(np.array(uniq), axis=0)
            sc = 40.0 / (np.std(uniq[0] - bg0) + 1e-3)
            u8 = [to_u8(g - bg0, sc) for g in uniq]
            for p8, c8 in zip(u8[:-1], u8[1:]):
                dx, dy, n = m_piv(p8, c8, max_shift)
                if n >= 5:
                    est.append((dx, dy))
        if len(est) < 5:
            sys.exit("Не удалось оценить направление течения, задайте --angle")
        e = np.median(np.array(est), axis=0)
        a.angle = float(np.degrees(np.arctan2(-e[1], -e[0])))
        print(f"Направление течения (авто): {a.angle:+.1f}°, сдвиг ~{np.hypot(*e):.1f} px/уник.кадр")
    prep = make_prep(a.angle)
    cv2.imwrite(os.path.join(out, "roi_rotated.png"), prep(frame).clip(0, 255).astype(np.uint8))

    # Анализируются только "уникальные" кадры: дубликаты (картинка не обновилась)
    # пропускаются, иначе пара "кадр-дубликат" даёт нулевое смещение.
    g_last = prep(frame)
    bg = g_last.copy()
    hp_prev, prev8, scale = None, None, None
    rows, profiles = [], []
    i, ndup, nuniq = 0, 0, 0
    t0 = a.start
    while True:
        ok, fr = cap.read()
        if not ok or (limit is not None and i >= limit):
            break
        i += 1
        g = prep(fr)
        if is_dup(g, g_last):
            ndup += 1
            continue
        g_last = g
        nuniq += 1
        if alpha > 0:
            bg += alpha * (g - bg)
            hp = g - bg
        else:
            hp = g - g.mean()
        t = t0 + i / fps
        if nuniq < a.bg_tau * fps * 0.8:  # фон ещё не установился
            bg += 4 * alpha * (g - bg)
            continue
        if scale is None:
            scale = 40.0 / (hp.std() + 1e-3)
        cur8 = to_u8(hp, scale)
        profiles.append((t, [hp[band_edges[b]:band_edges[b + 1]].mean(axis=0) for b in range(a.bands)]))
        if hp_prev is None:
            hp_prev, prev8 = hp, cur8
            continue

        pdx, pdy, presp = m_phase(hp_prev, hp, hann, a.min_shift)
        vdx, vdy, vn = m_piv(prev8, cur8, max_shift)
        gm = np.hypot(cv2.Sobel(hp, cv2.CV_32F, 1, 0), cv2.Sobel(hp, cv2.CV_32F, 0, 1))
        fdx, fdy = m_farneback(prev8, cur8, gm > np.percentile(gm, 70))
        ldx, ldy, ln = m_lk(prev8, cur8)
        rows.append([i, t, pdx, pdy, presp, fdx, fdy, ldx, ldy, ln, float(hp.std()), vdx, vdy, vn])
        hp_prev, prev8 = hp, cur8
        if len(rows) % int(fps * 5) == 0:
            print(f"  {t:6.1f} с  piv={vdx:7.2f},{vdy:6.2f} lk={ldx:7.2f},{ldy:6.2f} px/кадр")
    cap.release()
    if len(rows) < 10:
        sys.exit("Слишком мало кадров для анализа")

    R = np.array(rows, float)
    cols = ["frame", "t", "phase_dx", "phase_dy", "phase_resp", "farn_dx", "farn_dy",
            "lk_dx", "lk_dy", "lk_n", "texture", "piv_dx", "piv_dy", "piv_n"]
    with open(os.path.join(out, "frames.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(cols)
        wr.writerows(R.tolist())

    # после поворота течение идёт вдоль -x; остаточный угол контролируем по PIV
    resp_thr = 0.5 * float(np.median(R[:, 4]))
    good_p = R[:, 4] >= resp_thr
    u = np.array([-1.0, 0.0])
    resid = float(np.degrees(np.arctan2(-np.nanmedian(R[:, 12]), -np.nanmedian(R[:, 11]))))

    def along(dx, dy):
        return dx * u[0] + dy * u[1]

    def across(dx, dy):
        return -dx * u[1] + dy * u[0]

    per_frame = {
        "piv": along(R[:, 11], R[:, 12]),
        "phase": np.where(good_p, along(R[:, 2], R[:, 3]), np.nan),
        "farn": along(R[:, 5], R[:, 6]),
        "lk": along(R[:, 7], R[:, 8]),
    }
    cross = {
        "piv": across(R[:, 11], R[:, 12]),
        "phase": np.where(good_p, across(R[:, 2], R[:, 3]), np.nan),
        "farn": across(R[:, 5], R[:, 6]),
        "lk": across(R[:, 7], R[:, 8]),
    }

    # окна по времени: скорость = медианный сдвиг за уникальный кадр * число уникальных кадров в секунду
    tp = np.array([p[0] for p in profiles])
    P = np.array([p[1] for p in profiles], np.float32)
    edges = np.arange(tp[0], tp[-1] + 1e-9, a.win)
    wins = {m: [] for m in METHODS}
    wt, wrate = [], []
    for k in range(len(edges) - 1):
        sel = (R[:, 1] >= edges[k]) & (R[:, 1] < edges[k + 1])
        psel = (tp >= edges[k]) & (tp < edges[k + 1])
        n = int(psel.sum())
        rate = n / a.win
        wt.append(edges[k] + a.win / 2)
        wrate.append(rate)
        for m in ("piv", "phase", "farn", "lk"):
            v = per_frame[m][sel]
            v = v[np.isfinite(v)]
            wins[m].append(window_mean(v) * rate if v.size >= max(3, 0.3 * n) else np.nan)
        kdx = m_kymo(P[psel], max_lag_shift=w / 3, min_shift=a.min_shift) if n >= 6 else np.nan
        wins["kymo"].append(-kdx * rate)
    wt = np.array(wt)
    wrate = np.array(wrate)
    for m in METHODS:
        wins[m] = np.array(wins[m], float)
    nwin = len(wt)

    # по каждому окну: консенсус и разброс методов, которые согласны между собой
    wcons, wspread = np.full(len(wt), np.nan), np.full(len(wt), np.nan)
    for k in range(len(wt)):
        v = np.array([wins[m][k] for m in METHODS])
        v = v[np.isfinite(v)]
        if v.size >= 2:
            c0 = np.median(v)
            v = v[np.abs(v - c0) <= 0.25 * abs(c0)] if c0 != 0 else v
            if v.size >= 2:
                wcons[k] = np.median(v)
                wspread[k] = (v.max() - v.min()) / abs(wcons[k]) * 100 if wcons[k] else np.nan
    k_cam = 1.0 / a.display_scale
    with open(os.path.join(out, "windows.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["t", "unique_fps"] + [f"{m}_px_per_s" for m in METHODS]
                    + [f"{m}_campx_per_s" for m in METHODS] + ["consensus_px_per_s", "spread_pct"])
        for k in range(nwin):
            v = [wins[m][k] for m in METHODS]
            wr.writerow([f"{wt[k]:.3f}", f"{wrate[k]:.1f}"] + [f"{q:.2f}" for q in v]
                        + [f"{q * k_cam:.2f}" for q in v] + [f"{wcons[k]:.2f}", f"{wspread[k]:.1f}"])

    lines = [f"Видео: {video}",
             f"FPS записи: {fps:.3f}; кадров: {i}, из них дубликатов {ndup} ({ndup / max(i, 1) * 100:.0f}%); "
             f"уникальных в секунду (медиана по окнам): {np.median(wrate):.1f}",
             f"ROI: {x},{y},{w},{h}; окно {a.win:g} с",
             f"Направление течения: {a.angle:+.1f}° от горизонтали (0 = налево, + = вверх-влево); "
             f"остаточный угол после поворота (PIV): {resid:+.1f}°", ""]
    unit = "px/с" + (f" (камера: x{k_cam:.3f})" if a.display_scale != 1 else "")
    lines += [f"Скорость вдоль течения, {unit}, статистика по окнам:",
              f"{'метод':6} {'окон':>5} {'медиана':>9} {'MAD':>7} {'CV':>6} {'P10..P90':>16}"
              + (f" {'медиана(кам)':>13}" if a.display_scale != 1 else "")]
    meds = {}
    for m in METHODS:
        s = robust_stats(wins[m])
        meds[m] = s["median"]
        lines.append(f"{m:6} {s['n']:5d} {s['median']:9.1f} {s['mad']:7.2f} {s['cv'] * 100:5.1f}% "
                     f"{s['p10']:7.1f}..{s['p90']:<7.1f}"
                     + (f" {s['median'] * k_cam:13.1f}" if a.display_scale != 1 else ""))
    mv = {m: meds[m] for m in METHODS if np.isfinite(meds[m])}
    if mv:
        c0 = float(np.median(list(mv.values())))
        ok_m = [m for m in mv if abs(mv[m] - c0) <= 0.25 * abs(c0)]
        bad = [m for m in METHODS if m not in ok_m]
        v = np.array([mv[m] for m in ok_m])
        cons = float(np.median(v))
        spread = float(v.max() - v.min())
        lines += ["", f"Согласованные методы: {', '.join(ok_m)}"
                  + (f"; отброшены (>25% от медианы или нет данных): {', '.join(bad)}" if bad else ""),
                  f"Консенсус (медиана согласованных): {cons:.1f} px/с"
                  + (f" = {cons * k_cam:.1f} px/с камеры" if a.display_scale != 1 else ""),
                  f"Расхождение согласованных методов (max-min): {spread:.1f} px/с "
                  f"({spread / abs(cons) * 100:.1f}%)"]
    good_w = np.isfinite(wspread) & (wspread <= 10)
    sc = robust_stats(wcons[good_w])
    lines += ["", f"Окна, где согласованные методы расходятся <=10%: {int(good_w.sum())} из {len(wt)}",
              f"  скорость по ним: медиана {sc['median']:.1f} px/с"
              + (f" ({sc['median'] * k_cam:.1f} px/с камеры)" if a.display_scale != 1 else "")
              + f", разброс между окнами (MAD) {sc['mad']:.1f} px/с = {sc['cv'] * 100:.1f}%, "
              f"P10..P90 {sc['p10']:.0f}..{sc['p90']:.0f}"]
    lines.append("Поперечная составляющая (piv/phase/farn/lk), px/уник.кадр: "
                 + " / ".join(f"{np.nanmedian(cross[m]):.3f}" if np.isfinite(cross[m]).any() else "nan"
                              for m in ("piv", "phase", "farn", "lk"))
                 + "  (должна быть ~0)")
    lines.append(f"Отброшено пар phase (слабый пик < {resp_thr:.3f}): {int((~good_p).sum())} из {len(R)}")
    txt = "\n".join(lines)
    print("\n" + txt)
    with open(os.path.join(out, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        rate_med = float(np.median(wrate))
        for m in ("piv", "phase", "farn", "lk"):
            ax[0].plot(R[:, 1], per_frame[m] * rate_med, ".", ms=2, alpha=0.4, label=m)
        ax[0].set_ylabel("по кадрам, px/с")
        ax[0].legend(markerscale=5)
        for m in METHODS:
            ax[1].plot(wt, wins[m], "-o", ms=3, label=m)
        ax[1].plot(wt[good_w], wcons[good_w], "k*", ms=9, label="консенсус (<=10%)")
        ax[1].set_ylabel(f"окно {a.win:g} с, px/с")
        ax[1].legend()
        ax[2].plot(R[:, 1], R[:, 10], lw=0.7, label="текстура (std)")
        ax[2].plot(R[:, 1], R[:, 4] * R[:, 10].max(), lw=0.7, label="phase resp (масшт.)")
        ax[2].set_xlabel("t, с")
        ax[2].legend()
        for q in ax:
            q.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "plot.png"), dpi=120)
    except ImportError:
        print("matplotlib не установлен - график не построен")
    print(f"\nРезультаты: {out}")


if __name__ == "__main__":
    main()
