#!/usr/bin/env python3
"""
Прототип поточного расчёта скорости по СЕРИЯМ кадров (для Raspberry Pi 5).

На каждую серию (по умолчанию 10 соседних кадров камеры):
  - kymo по B полосам поперёк потока -> профиль скорости, ось (максимум) и скорость на оси;
  - kymo по всей полосе;
  - выборочная проверка: phase и LK на одной-двух парах серии;
  - раз в --piv-every серий: PIV (поиск вокруг скорости kymo).
Фон: медиана первых кадров первых серий, дальше - дешёвое слежение за медианой
(шаг в сторону знака разности). Геометрия (ROI, угол) фиксирована - как после калибровки;
для теста берётся из результата flow_speed той же записи.

Время считается только на расчёт (без декодирования видео): на Pi кадры уже в памяти.

  python series_flow.py <video.mkv> <папка результата flow_speed> [--series 10] [--period 0.1]
"""
import argparse
import os
import time

import cv2
import numpy as np

from common import (GOOD_SPREAD_PCT, Timing, consensus, flow_affine, gray, mask_zero_shift, norm_brightness,
                    parabola, parabola_vertex, pyplot, read_frames, robust_stats, window_mean, wrap, write_csv,
                    write_json)
from flow_rate import read_speed_result
from flow_speed import LK_CRIT, PairMethods, to_u8


def make_prep_scaled(roi, angle, s):
    """Поворот ROI вдоль течения сразу в уменьшенное окно (s = 0.5 -> в 4 раза меньше пикселей)."""
    x, y, w, h = roi
    A = flow_affine(x + w / 2.0, y + h / 2.0, angle, w, h)
    A[:, :2] /= s
    size = (int(round(w * s)), int(round(h * s)))

    def prep(f):
        g = cv2.warpAffine(gray(f), A, size, flags=cv2.INTER_AREA | cv2.WARP_INVERSE_MAP).astype(np.float32)
        return norm_brightness(cv2.GaussianBlur(g, (0, 0), 0.7))
    return prep


def kymo_bands(prof, max_lag_shift, min_shift, snr_exclusion=8):
    """kymo сразу для всех полос и для их суммы. prof: (T, B, W).
    Возвращает сдвиг за шаг кадров для каждой полосы (B,), общий сдвиг, SNR общего пика
    на интервале 1 кадр (как в magma_flow.hpp: (пик - среднее) / СКО вне окрестности пика)
    и число интервалов, вошедших в общую оценку."""
    T, B, W = prof.shape
    F = np.fft.rfft(prof * np.hanning(W).astype(np.float32), axis=2)          # (T, B, Wf)
    snr = [np.nan]

    def peaks(cp):                                                              # cp: (N, Wf)
        cp = cp / (np.abs(cp) + 1e-9 * np.abs(cp).max(axis=1, keepdims=True))
        r = np.fft.irfft(cp, n=W, axis=1)
        out = []
        for row in r:
            row = mask_zero_shift(row, int(min_shift))
            i = int(np.argmax(row))
            out.append(wrap(i + parabola(row[(i - 1) % W], row[i], row[(i + 1) % W]), W))
        far = np.abs((np.arange(W) - i + W // 2) % W - W // 2) > snr_exclusion   # последняя строка = сумма
        if np.isnan(snr[0]) and far.sum() >= 10 and row[far].std() > 0:
            snr[0] = float((row[i] - row[far].mean()) / row[far].std())
        return np.array(out)

    lag_cp = {}

    def shifts(k):
        if k not in lag_cp:
            cp = (np.conj(F[:-k]) * F[k:]).sum(axis=0)                           # (B, Wf)
            lag_cp[k] = peaks(np.vstack([cp, cp.sum(axis=0, keepdims=True)]))    # полосы + все вместе
        return lag_cp[k]

    s1 = shifts(1)
    num, den = s1.copy(), np.ones_like(s1)
    alive = np.isfinite(s1)
    n_lags = 1
    k = 2
    while k <= T // 2 and alive.any():
        ok = alive & (np.abs(s1) * k < max_lag_shift)
        if not ok.any():
            break
        sk = shifts(k)
        ok &= np.abs(sk - s1 * k) <= np.maximum(2.0, 0.25 * np.abs(s1 * k))   # лаг не "перескочил"
        num[ok] += k * sk[ok]
        den[ok] += k * k
        n_lags += bool(ok[B])
        alive = ok
        k *= 2
    d = num / den
    return d[:B], d[B], snr[0], n_lags


def phase_pair(a, b, win, min_shift):
    """Сдвиг по x фазовой корреляцией пары кадров."""
    cp = np.conj(np.fft.rfft2(a * win)) * np.fft.rfft2(b * win)
    mag = np.abs(cp)
    r = mask_zero_shift(np.fft.irfft2(cp / (mag + 1e-6 * mag.max()), s=a.shape), int(min_shift))
    w = r.shape[1]
    iy, ix = np.unravel_index(int(np.argmax(r)), r.shape)
    return wrap(ix + parabola(r[iy, (ix - 1) % w], r[iy, ix], r[iy, (ix + 1) % w]), w)


def lk_pair(a8, b8):
    """Сдвиг по x трекингом Лукаса-Канаде с проверкой вперёд-назад."""
    pts = cv2.goodFeaturesToTrack(a8, maxCorners=200, qualityLevel=0.01, minDistance=4)
    if pts is None or len(pts) < 10:
        return np.nan
    kw = dict(winSize=(15, 15), maxLevel=3, criteria=LK_CRIT)
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(a8, b8, pts, None, **kw)
    back, st2, _ = cv2.calcOpticalFlowPyrLK(b8, a8, nxt, None, **kw)
    ok = (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm((back - pts).reshape(-1, 2), axis=1) < 0.5)
    return float(np.median((nxt - pts).reshape(-1, 2)[ok, 0])) if ok.sum() >= 10 else np.nan


def series_index(tm, n, period):
    """Серии: отрезки из n соседних кадров камеры. В сериях длиннее n берутся первые n кадров;
    непрерывная запись режется на серии, начинающиеся каждые period секунд."""
    out, run = [], [tm.lo]
    for i in range(tm.lo + 1, tm.hi):
        if tm.linked(i):
            run.append(i)
            continue
        out.append(run)
        run = [i]
    out.append(run)
    series = []
    for r in out:
        if len(r) > 3 * n:                       # непрерывный участок -> искусственные серии
            t = np.array([tm.time(i) for i in r])
            starts = np.searchsorted(t, np.arange(t[0], t[-1], period))
            series += [r[s:s + n] for s in starts if s + n <= len(r)]
        elif len(r) >= n:
            series.append(r[:n])
    return series


def band_speeds(hp_stack, bands, dt, min_shift, min_tex=0.3, snr_exclusion=8):
    """kymo по полосам поперёк потока и по всей полосе. hp_stack: (T, h, w).
    Полоса без текстуры (тёмный фон за краем потока) или с неправдоподобной скоростью -> NaN.
    Возвращает скорости полос и общую (px/с), центры полос, SNR, число интервалов, текстуру."""
    T, h, w = hp_stack.shape
    hb = h // bands
    blocks = hp_stack[:, :hb * bands].reshape(T, bands, hb, w)
    d_band, d_all, snr, n_lags = kymo_bands(blocks.mean(axis=2), w / 3, min_shift, snr_exclusion)
    v_band, v_all = -d_band / dt, -d_all / dt
    tex = blocks.std(axis=(0, 2, 3))
    bad = (tex < min_tex * tex.max()) | ~(v_band >= 0.1 * v_all) | (v_band > 1.5 * v_all)
    v_band[bad] = np.nan
    return v_band, v_all, (np.arange(bands) + 0.5) * hb, snr, n_lags, float(tex.max())


def axis_from_profile(centers, v):
    ok = np.isfinite(v)
    if ok.sum() < 3:
        return np.nan, np.nan, True
    c, vv = centers[ok], v[ok]
    i = int(np.argmax(vv))
    if 0 < i < len(vv) - 1:
        y = parabola_vertex(c[i - 1:i + 2], vv[i - 1:i + 2])
        # скорость на оси - значение параболы в вершине, не выше 105% максимума точек
        A = np.polyfit(c[i - 1:i + 2], vv[i - 1:i + 2], 2)
        return float(y), float(min(np.polyval(A, y), 1.05 * vv[i])), False
    return float(c[i]), float(vv[i]), True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("speed_dir", help="результат flow_speed той же записи (ROI и угол)")
    ap.add_argument("--series", type=int, default=10, help="кадров в серии")
    ap.add_argument("--period", type=float, default=0.2, help="период серий для непрерывных записей (бюджет), с")
    ap.add_argument("--bands", type=int, default=8)
    ap.add_argument("--check-pairs", type=int, default=1, help="пар на серию для phase+LK (0 = без проверки)")
    ap.add_argument("--piv-every", type=int, default=10, help="PIV раз в N серий (0 = без PIV)")
    ap.add_argument("--bg-init", type=int, default=20, help="серий для начального фона")
    ap.add_argument("--bg-step", type=float, default=0.5, help="шаг слежения фона за медианой, ед. яркости/кадр")
    ap.add_argument("--min-shift", type=float, default=1.0, help="px уменьшенного кадра")
    ap.add_argument("--scale", type=float, default=0.5, help="масштаб обработки (0.5 = половинное разрешение)")
    ap.add_argument("--win", type=float, default=1.0, help="окно сравнения, с")
    ap.add_argument("--out")
    a = ap.parse_args()

    sp = read_speed_result(a.speed_dir)
    x, y, w, h = sp["roi"]
    prep = make_prep_scaled(sp["roi"], sp["angle"], a.scale)
    k_px = 1.0 / a.scale                                        # px уменьшенного кадра -> px камеры
    tm = Timing(a.video)
    series = series_index(tm, a.series, a.period)
    if len(series) < a.bg_init + 5:
        raise SystemExit(f"Мало серий: {len(series)}")
    out = a.out or os.path.join(a.speed_dir, "series")
    os.makedirs(out, exist_ok=True)
    per = np.median(np.diff([tm.time(s[0]) for s in series]))
    print(f"{a.video}: серий {len(series)} по {a.series} кадров, период ~{per * 1000:.0f} мс, "
          f"ROI {x},{y},{w},{h}, угол {sp['angle']:+.1f}°")

    cap = cv2.VideoCapture(a.video)
    bg = np.median(np.array(read_frames(cap, [s[0] for s in series[:a.bg_init]], prep)), axis=0).astype(np.float32)
    hs, ws = bg.shape
    pm = PairMethods(ws, hs, ws / 3, a.min_shift, tile=32)       # фрагмент 32 px = 64 px камеры
    win = cv2.createHanningWindow((ws, hs), cv2.CV_32F)
    scale = None
    rows, band_rows, timing = [], [], []
    for k, s in enumerate(series):
        frames = read_frames(cap, s)                            # декодирование - вне замера
        t0 = time.perf_counter()
        g = np.stack([prep(f) for f in frames])                 # (T, h, w) float32
        tp = time.perf_counter()
        bg += a.bg_step * np.sign(g - bg).mean(axis=0)          # слежение за медианой
        hp = g - bg
        ts = np.array([tm.time(i) for i in s])
        dt = float(np.median(np.diff(ts)))
        v_band, v_all, centers = band_speeds(hp, a.bands, dt / k_px, a.min_shift)[:3]
        centers = centers * k_px                                # y в px повёрнутого ROI камеры
        y_ax, v_ax, on_edge = axis_from_profile(centers, v_band)
        tk = time.perf_counter()
        if scale is None:
            scale = 40.0 / (hp[0].std() + 1e-3)
        ph, lk = [], []
        for j in np.linspace(1, len(s) - 1, a.check_pairs + 2)[1:-1].astype(int) if a.check_pairs else []:
            ddt = (ts[j] - ts[j - 1]) / k_px
            ph.append(-phase_pair(hp[j - 1], hp[j], win, a.min_shift) / ddt)
            lk.append(-lk_pair(to_u8(hp[j - 1], scale), to_u8(hp[j], scale)) / ddt)
        tc = time.perf_counter()
        v_piv = np.nan
        if a.piv_every and k % a.piv_every == 0:
            j = len(s) // 2
            p, c = pm.frame(hp[j - 1], to_u8(hp[j - 1], scale)), pm.frame(hp[j], to_u8(hp[j], scale))
            ddt = (ts[j] - ts[j - 1]) / k_px
            dx = pm.piv(p, c, pred=(-v_all * ddt, 0.0) if np.isfinite(v_all) else None)[0]
            v_piv = -dx / ddt
        te = time.perf_counter()
        v_ph = float(np.nanmedian(ph)) if ph and np.isfinite(ph).any() else np.nan
        v_lk = float(np.nanmedian(lk)) if lk and np.isfinite(lk).any() else np.nan
        cons, spread, _ = consensus([v_all, v_ph, v_lk, v_piv])
        rows.append([ts[0], v_ax, y_ax, int(on_edge), v_all, v_ph, v_lk, v_piv, cons, spread])
        band_rows.append(v_band)
        timing.append([(tp - t0) * 1e3, (tk - tp) * 1e3, (tc - tk) * 1e3, (te - tc) * 1e3, (te - t0) * 1e3])
    cap.release()

    R, B, T = np.array(rows, float), np.array(band_rows, float), np.array(timing, float)
    cols = ["t", "v_axis", "y_axis", "axis_on_edge", "kymo_all", "phase", "lk", "piv", "consensus", "spread_pct"]
    write_csv(os.path.join(out, "series.csv"), cols + [f"band{b}" for b in range(a.bands)],
              np.column_stack([R, B]).tolist())
    # окна для сравнения с полным анализом (windows.csv flow_speed: окна от первой пары, шаг --win)
    edges = np.arange(R[0, 0], R[-1, 0] + a.win, a.win)
    wrows = []
    for e0 in edges[:-1]:
        m = (R[:, 0] >= e0) & (R[:, 0] < e0 + a.win)
        if m.any():
            wrows.append([e0 + a.win / 2, int(m.sum())] + [window_mean(R[m, i]) for i in (1, 4, 5, 6, 7, 8)])
    write_csv(os.path.join(out, "windows.csv"), ["t", "series", "v_axis", "kymo_all", "phase", "lk", "piv", "consensus"], wrows)

    names = ("этап подготовки (поворот)", "kymo 8 полос + вся полоса", f"проверка phase+LK ({a.check_pairs} пар)",
             f"PIV (раз в {a.piv_every})", "всего на серию")
    st = {n: (np.median(T[:, i]), np.percentile(T[:, i], 95), T[:, i].max()) for i, n in enumerate(names)}
    rs = {c: robust_stats(R[:, cols.index(c)]) for c in ("v_axis", "kymo_all", "phase", "lk", "piv", "consensus")}
    good = np.isfinite(R[:, 9]) & (R[:, 9] <= GOOD_SPREAD_PCT)
    lines = [f"Видео: {a.video}", f"Серий: {len(R)} по {a.series} кадров, период ~{per * 1000:.0f} мс", "",
             "Время расчёта на серию, мс (медиана / P95 / макс), без декодирования видео:"]
    lines += [f"  {n:32s} {v[0]:7.1f} / {v[1]:7.1f} / {v[2]:7.1f}" for n, v in st.items()]
    lines += ["", "Скорость по сериям, px/с (медиана, CV по сериям):"]
    lines += [f"  {c:10s} {s['median']:8.1f}   CV {s['cv'] * 100:5.1f}%   n={s['n']}" for c, s in rs.items()]
    lines += [f"  ось на краю профиля в {int(R[:, 3].sum())} из {len(R)} серий",
              f"  серий, где kymo/phase/LK(/PIV) согласны в пределах {GOOD_SPREAD_PCT:g}%: {int(good.sum())} из {len(R)}"]
    lines += ["", "Профиль kymo по полосам (медиана по сериям), px/с:",
              "  " + "  ".join(f"y={c:.0f}: {np.nanmedian(B[:, i]):.0f}" for i, c in enumerate(centers))]
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(out, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    write_json(os.path.join(out, "result.json"), dict(
        series=len(R), series_len=a.series, period_s=per, timing_ms={n: v for n, v in st.items()},
        speed={c: s for c, s in rs.items()}, axis_on_edge=int(R[:, 3].sum()), agreed_series=int(good.sum())))

    plt = pyplot()
    if plt:
        fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        for i, lab in ((1, "kymo: ось"), (4, "kymo: вся полоса"), (5, "phase"), (6, "LK"), (7, "PIV")):
            ax[0].plot(R[:, 0], R[:, i], "." if i != 7 else "o", ms=3 if i != 7 else 5, alpha=0.6, label=lab)
        ax[0].set_ylabel("по сериям, px/с")
        ax[0].legend(ncol=5, fontsize=8)
        ax[1].plot(R[:, 0], T[:, 4], ".", ms=3)
        ax[1].axhline(a.period * 1000, color="r", lw=1, label=f"бюджет {a.period * 1000:.0f} мс")
        ax[1].set_ylabel("время на серию, мс")
        ax[1].set_xlabel("t, с")
        ax[1].legend()
        for q in ax:
            q.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "plot.png"), dpi=110)


if __name__ == "__main__":
    main()
