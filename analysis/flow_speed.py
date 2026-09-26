#!/usr/bin/env python3
"""
Оценка скорости течения жидкости по желобу из видео (в пикселях).

Жидкость течёт справа налево => смещение текстуры dx < 0.
Скорость выводится как положительная величина vel = -dx (px/кадр и px/с).

Методы (все работают по одной и той же области ROI):
  phase   - 2D фазовая корреляция соседних кадров
  farn    - плотный оптический поток Фарнебека, медиана dx по текстурным пикселям
  lk      - трекинг точек Лукаса-Канаде с проверкой вперёд-назад, медиана dx
  kymo    - 1D-корреляция профилей вдоль желоба, накопленная за окно времени,
            несколько лагов (1, 2, 4 ... кадров) -> наклон v = shift / lag

Перед анализом из кадров вычитается медленно обновляемый фон (EMA),
чтобы неподвижные детали (края желоба, блики на стенках) не тянули
скорость к нулю.

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
METHODS = ("phase", "farn", "lk", "kymo")


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


def m_phase(prev, cur, win):
    """2D фазовая корреляция; субпиксель - парабола по x и y
    (cv2.phaseCorrelate с центроидом даёт систематическое смещение ~3%)."""
    A = np.fft.rfft2(prev * win)
    B = np.fft.rfft2(cur * win)
    cp = np.conj(A) * B
    cp /= np.abs(cp) + 1e-6 * np.abs(cp).max()
    c = np.fft.irfft2(cp, s=prev.shape)
    h, w = c.shape
    iy, ix = np.unravel_index(int(np.argmax(c)), c.shape)
    dx = ix + _parabola(c[iy, (ix - 1) % w], c[iy, ix], c[iy, (ix + 1) % w])
    dy = iy + _parabola(c[(iy - 1) % h, ix], c[iy, ix], c[(iy + 1) % h, ix])
    if dx > w / 2:
        dx -= w
    if dy > h / 2:
        dy -= h
    return float(dx), float(dy), float(c[iy, ix])


def m_farneback(prev8, cur8, grad_mask):
    flow = cv2.calcOpticalFlowFarneback(prev8, cur8, None, 0.5, 4, 21, 3, 7, 1.5, 0)
    if grad_mask.sum() < 50:
        return np.nan, np.nan
    return float(np.median(flow[..., 0][grad_mask])), float(np.median(flow[..., 1][grad_mask]))


LK_PARAMS = dict(winSize=(21, 21), maxLevel=4,
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


def _xcorr_peak(acc):
    """Пик циклической корреляции с субпиксельным уточнением (парабола)."""
    n = acc.size
    i = int(np.argmax(acc))
    s = i + _parabola(acc[(i - 1) % n], acc[i], acc[(i + 1) % n])
    if s > n / 2:
        s -= n
    return s


def m_kymo(profiles, max_lag_shift):
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
        return _xcorr_peak(np.fft.irfft(cp, n=W))

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

    win_len = max(3, int(round(a.win * fps)))
    limit = int(a.max_seconds * fps) if a.max_seconds > 0 else None
    alpha = 1.0 / (a.bg_tau * fps) if a.bg_tau > 0 else 0.0
    hann = cv2.createHanningWindow((w, h), cv2.CV_32F)
    band_edges = np.linspace(0, h, a.bands + 1).astype(int)

    def prep(fr):
        g = cv2.cvtColor(fr[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY).astype(np.float32)
        return cv2.GaussianBlur(g, (0, 0), 1.0)

    g = prep(frame)
    bg = g.copy()
    # первые кадры только "прогревают" фон, чтобы не анализировать сырой кадр
    warm = int(min(a.bg_tau * fps, 60)) if alpha > 0 else 0
    for _ in range(warm):
        ok, fr = cap.read()
        if not ok:
            break
        g = prep(fr)
        bg += alpha * 5 * (g - bg)  # ускоренный прогрев
    hp_prev = g - bg if alpha > 0 else g - g.mean()
    scale = 40.0 / (hp_prev.std() + 1e-3)
    prev8 = to_u8(hp_prev, scale)

    rows, prof_buf, kymo_rows = [], [], []
    i = 0
    t0 = a.start
    while True:
        ok, fr = cap.read()
        if not ok or (limit is not None and i >= limit):
            break
        i += 1
        g = prep(fr)
        if alpha > 0:
            bg += alpha * (g - bg)
            hp = g - bg
        else:
            hp = g - g.mean()
        cur8 = to_u8(hp, scale)

        pdx, pdy, presp = m_phase(hp_prev, hp, hann)
        gx = cv2.Sobel(hp, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(hp, cv2.CV_32F, 0, 1)
        gm = np.hypot(gx, gy)
        mask = gm > np.percentile(gm, 70)
        fdx, fdy = m_farneback(prev8, cur8, mask)
        ldx, ldy, ln = m_lk(prev8, cur8)
        tex = float(hp.std())

        rows.append([i, t0 + i / fps, pdx, pdy, presp, fdx, fdy, ldx, ldy, ln, tex])
        prof_buf.append([hp[band_edges[b]:band_edges[b + 1]].mean(axis=0) for b in range(a.bands)])
        if len(prof_buf) == win_len:
            kdx = m_kymo(np.array(prof_buf, np.float32), max_lag_shift=w / 3)
            kymo_rows.append((t0 + (i - win_len / 2) / fps, kdx))
            prof_buf = []

        hp_prev, prev8 = hp, cur8
        if i % int(fps * 5) == 0:
            print(f"  {i / fps:6.1f} с  phase={-pdx:6.2f} farn={-fdx:6.2f} lk={-ldx:6.2f} px/кадр")
    cap.release()
    if not rows:
        sys.exit("Нет кадров для анализа")

    R = np.array(rows, float)
    cols = ["frame", "t", "phase_dx", "phase_dy", "phase_resp", "farn_dx", "farn_dy",
            "lk_dx", "lk_dy", "lk_n", "texture"]
    with open(os.path.join(out, "frames.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(cols)
        wr.writerows(R.tolist())

    # отбраковка кадров: слабый пик фазовой корреляции
    resp_thr = max(0.02, 0.5 * float(np.median(R[:, 4])))
    per_frame = {
        "phase": np.where(R[:, 4] >= resp_thr, -R[:, 2], np.nan),
        "farn": -R[:, 5],
        "lk": -R[:, 7],
    }

    # усреднение по окнам (медиана внутри окна)
    nwin = len(R) // win_len
    wins = {m: [] for m in METHODS}
    wt = []
    for k in range(nwin):
        sl = slice(k * win_len, (k + 1) * win_len)
        wt.append(R[sl, 1].mean())
        for m in ("phase", "farn", "lk"):
            v = per_frame[m][sl]
            v = v[np.isfinite(v)]
            wins[m].append(np.median(v) if v.size >= win_len * 0.3 else np.nan)
        wins["kymo"].append(-kymo_rows[k][1] if k < len(kymo_rows) else np.nan)
    wt = np.array(wt)
    for m in METHODS:
        wins[m] = np.array(wins[m], float)

    with open(os.path.join(out, "windows.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["t"] + [f"{m}_px_per_frame" for m in METHODS] + [f"{m}_px_per_s" for m in METHODS])
        for k in range(nwin):
            v = [wins[m][k] for m in METHODS]
            wr.writerow([f"{wt[k]:.3f}"] + [f"{q:.4f}" for q in v] + [f"{q * fps:.2f}" for q in v])

    # сводка
    lines = [f"Видео: {video}", f"FPS: {fps:.3f}, кадров проанализировано: {len(R)}, окно: {win_len} кадров",
             f"ROI: {x},{y},{w},{h}", "",
             "Скорость (положительная = справа налево), статистика по окнам:",
             f"{'метод':6} {'окон':>5} {'медиана px/кадр':>16} {'px/с':>9} {'разброс(MAD) px/с':>18} "
             f"{'CV':>6} {'P10..P90 px/с':>18}"]
    meds = {}
    for m in METHODS:
        s = robust_stats(wins[m])
        meds[m] = s["median"]
        lines.append(f"{m:6} {s['n']:5d} {s['median']:16.3f} {s['median'] * fps:9.1f} {s['mad'] * fps:18.2f} "
                     f"{s['cv'] * 100:5.1f}% {s['p10'] * fps:8.1f}..{s['p90'] * fps:<8.1f}")
    mv = np.array([meds[m] for m in METHODS if np.isfinite(meds[m])])
    if mv.size:
        cons = float(np.median(mv))
        spread = float(mv.max() - mv.min())
        lines += ["", f"Консенсус (медиана методов): {cons:.3f} px/кадр = {cons * fps:.1f} px/с",
                  f"Расхождение методов (max-min): {spread * fps:.1f} px/с "
                  f"({spread / abs(cons) * 100 if cons else float('nan'):.1f}% от консенсуса)"]
    dys = [np.nanmedian(R[:, 3]), np.nanmedian(R[:, 6]), np.nanmedian(R[:, 8])]
    lines.append(f"Поперечная составляющая dy (phase/farn/lk), px/кадр: "
                 + " / ".join(f"{d:.3f}" for d in dys) + "  (должна быть ~0)")
    lines.append(f"Отброшено кадров phase (слабый пик < {resp_thr:.3f}): "
                 f"{int(np.isnan(per_frame['phase']).sum())} из {len(R)}")
    txt = "\n".join(lines)
    print("\n" + txt)
    with open(os.path.join(out, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        for m in ("phase", "farn", "lk"):
            ax[0].plot(R[:, 1], per_frame[m] * fps, ".", ms=1, alpha=0.3, label=m)
        ax[0].set_ylabel("по кадрам, px/с")
        ax[0].legend(markerscale=10)
        for m in METHODS:
            ax[1].plot(wt, wins[m] * fps, "-o", ms=3, label=m)
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
