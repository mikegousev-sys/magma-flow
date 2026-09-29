#!/usr/bin/env python3
"""
Оценка уровня и расхода шлака по результатам flow_speed.py.

Что берётся из видео:
  - профиль скорости поверхности поперёк потока (profile.csv от flow_speed.py);
  - ось потока = максимум этого профиля (вершина параболы по 3 точкам);
  - край корки (A) = переход "фон -> светящийся поток" на медиане кадров, по каждому
    столбцу вдоль течения; берётся медиана по столбцам.
Геометрия:
  - полуширина свободной поверхности b = |ось - край| * scale_normal + crust;
  - уровень z находится по профилю желоба (правая половина), где полуширина = b;
  - площадь сечения и расход считаются по полному профилю (обе половины):
        Q = k * интеграл( u_s(|x|) * глубина(x) dx ),
    u_s - скорость поверхности на расстоянии |x| от оси (профиль симметричен),
    под коркой и до стенки скорость линейно спадает до 0,
    k - отношение средней по глубине скорости к поверхностной
        (ламинарное течение ~0.67, турбулентное ~0.85-0.9).

Масштабы (мм/px камеры) по калибровочной сетке 18 мм:
  вдоль потока 0.548, поперёк (расстояние между линиями вдоль потока) 0.638.

  python flow_rate.py <папка результата flow_speed> <video.mkv>
"""
import argparse
import csv
import os
import re
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def read_params(speed_dir):
    txt = open(os.path.join(speed_dir, "summary.txt"), encoding="utf-8").read()
    roi = tuple(int(v) for v in re.search(r"ROI: (\d+),(\d+),(\d+),(\d+)", txt).groups())
    ang = float(re.search(r"Направление течения: ([-+]?\d+\.?\d*)°", txt).group(1))
    return roi, ang


def read_profile(speed_dir):
    rows = list(csv.DictReader(open(os.path.join(speed_dir, "profile.csv"))))
    y = np.array([float(r["y_rot_px"]) for r in rows])
    v = np.array([float(r["px_per_s"]) for r in rows])
    return y, v


def gutter(path):
    rows = [r for r in csv.reader(open(path)) if r and not r[0].startswith(("#", "x"))]
    P = np.array([[float(a), float(b)] for a, b in rows])
    return P[np.argsort(P[:, 0])]


def half_width(P, z, side):
    """Полуширина сечения на уровне z (от оси до стенки) для side=+1 (правая) / -1 (левая)."""
    s = P[P[:, 0] * side > 0]
    s = s[np.argsort(s[:, 1])]
    # на каждом уровне берём самую дальнюю от оси точку профиля
    zz = np.maximum.accumulate(s[:, 1])
    return float(np.interp(z, zz, np.abs(s[:, 0])))


def level_from_half_width(P, b, side):
    zs = np.linspace(0, P[:, 1].max(), 2000)
    hw = np.array([half_width(P, z, side) for z in zs])
    hw = np.maximum.accumulate(hw)
    if b > hw[-1]:
        return np.nan
    return float(np.interp(b, hw, zs))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("speed_dir")
    ap.add_argument("video")
    ap.add_argument("--scale-along", type=float, default=0.548, help="мм/px вдоль потока")
    ap.add_argument("--scale-normal", type=float, default=0.638, help="мм/px поперёк потока")
    ap.add_argument("--crust", type=float, default=25.0, help="на сколько корка заходит на жидкость, мм")
    ap.add_argument("--profile", default=os.path.join(HERE, "gutter_profile_BB.csv"))
    ap.add_argument("--side", type=int, default=1, help="+1: камера видит правую половину чертежа")
    ap.add_argument("--k", default="0.67,0.85", help="отношение средней скорости к поверхностной (мин,макс)")
    ap.add_argument("--rho", type=float, default=3400.0, help="плотность шлака, кг/м3")
    ap.add_argument("--start", type=float, default=0.0, help="начало участка, с (как в flow_speed)")
    ap.add_argument("--seconds", type=float, default=0.0, help="длительность участка, с (0 = до конца)")
    a = ap.parse_args()

    (x, y, w, h), ang = read_params(a.speed_dir)
    yp, vp = read_profile(a.speed_dir)
    n_prof = yp - h / 2.0                        # смещение поперёк потока от центра ROI, px

    # ось потока: вершина параболы вокруг максимума профиля скорости
    i = int(np.argmax(vp))
    if 0 < i < len(vp) - 1:
        y0, y1, y2 = vp[i - 1:i + 2]
        den = y0 - 2 * y1 + y2
        off = 0.5 * (y0 - y2) / den if den != 0 else 0.0
        n_axis = n_prof[i] + off * (n_prof[1] - n_prof[0])
        axis_note = ""
    else:
        n_axis = n_prof[i]
        axis_note = " (максимум на краю профиля - ось может быть дальше, уровень занижен)"
    edge_sign = -1.0 if n_prof[np.argmin(vp)] < n_axis else 1.0   # в какую сторону от оси край

    # край корки: медиана кадров в повёрнутой системе (Y вдоль e2, как в flow_speed)
    t = np.radians(ang)
    e1 = np.array([np.cos(t), np.sin(t)])
    e2 = np.array([-e1[1], e1[0]])
    cx0, cy0 = x + w / 2.0, y + h / 2.0
    W = H = 1600
    A = np.array([[e1[0], e2[0], cx0 - e1[0] * W / 2 - e2[0] * H / 2],
                  [e1[1], e2[1], cy0 - e1[1] * W / 2 - e2[1] * H / 2]], np.float32)
    cap = cv2.VideoCapture(a.video)
    nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    lo, hi = 0, nfr
    fcsv = os.path.join(os.path.dirname(os.path.abspath(a.video)), "frames.csv")
    if os.path.isfile(fcsv):
        ts = np.array([float(r["time_rel_s"]) for r in csv.DictReader(open(fcsv))])
        lo = int(np.searchsorted(ts, ts[0] + a.start))
        hi = int(np.searchsorted(ts, ts[0] + a.start + a.seconds)) if a.seconds > 0 else len(ts)
        hi = min(hi, nfr)
    frames, valid = [], None
    for j in np.linspace(lo, hi - 1, 60).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, j)
        ok, f = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if valid is None:
            valid = cv2.warpAffine(np.ones_like(g), A, (W, H),
                                   flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP) > 0
        frames.append(cv2.warpAffine(g, A, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP))
    cap.release()
    med = np.median(np.array(frames), axis=0)
    Y_axis = int(round(H / 2 + n_axis))
    edges = []
    for X in range(0, W, 4):
        col_ok = valid[:, X]
        # идём от оси в сторону края, пока не выйдем на фон
        ys = np.arange(Y_axis, 0 if edge_sign < 0 else H - 1, int(edge_sign))
        ys = ys[col_ok[ys]]
        if len(ys) < 40 or not col_ok[Y_axis]:
            continue
        prof = cv2.GaussianBlur(med[ys, X][:, None], (1, 9), 0)[:, 0]
        bgl = np.median(prof[-15:])                 # фон за краем
        top = np.percentile(prof[:max(5, len(prof) // 3)], 90)
        if top - bgl < 5:
            continue
        above = np.flatnonzero(prof >= (top + bgl) / 2)
        # ищем со стороны фона: самая дальняя от оси точка выше порога
        # (тёмные пятна текстуры внутри потока так не мешают)
        if not len(above) or len(ys) - above[-1] < 15:
            continue                                # фон в столбце не виден (край вне кадра)
        edges.append(ys[above[-1]] - H / 2)
    if len(edges) < 10:
        sys.exit("Край корки не найден")
    edges = np.array(edges)
    n_edge = float(np.median(edges))
    d_px = abs(n_axis - n_edge)
    b = d_px * a.scale_normal + a.crust

    P = gutter(a.profile)
    z = level_from_half_width(P, b, a.side)
    xs = np.linspace(P[:, 0].min(), P[:, 0].max(), 4000)
    zb = np.interp(xs, P[:, 0], P[:, 1])
    dep = np.clip(z - zb, 0, None) / 1000.0                    # м
    area = float(np.trapezoid(dep, xs / 1000.0))
    wet = dep > 0
    bL, bR = -xs[wet].min(), xs[wet].max()

    # скорость поверхности как функция расстояния от оси, м/с
    r_prof = np.abs(n_prof - n_axis) * a.scale_normal          # мм
    u_prof = vp * a.scale_along / 1000.0
    order = np.argsort(r_prof)
    r_s, u_s = r_prof[order], u_prof[order]
    r_vis = abs(n_edge - n_axis) * a.scale_normal

    def u_at(r, bw):
        r = np.abs(r)
        u = np.interp(r, r_s, u_s)
        # за последней точкой профиля - линейный спад до 0 у стенки
        last_r, last_u = r_s[-1], u_s[-1]
        tail = r > last_r
        u[tail] = last_u * np.clip((bw - r[tail]) / max(bw - last_r, 1e-6), 0, 1)
        return u

    u_x = np.where(xs >= 0, u_at(xs, bR), u_at(xs, bL))
    q_surf = float(np.trapezoid(u_x * dep, xs / 1000.0))       # м3/с при k = 1
    kmin, kmax = (float(v) for v in a.k.split(","))
    u_mean_s = q_surf / area if area > 0 else np.nan

    lines = [
        f"Результат скорости: {a.speed_dir}",
        f"ROI {x},{y},{w},{h}, направление {ang:+.1f}°",
        f"Ось потока (максимум скорости): n = {n_axis:+.1f} px{axis_note}",
        f"Край корки A: n = {n_edge:+.1f} px (разброс по длине P10..P90 {np.percentile(edges, 10):+.0f}..{np.percentile(edges, 90):+.0f})",
        f"Видимая полуширина поверхности: {d_px:.0f} px = {d_px * a.scale_normal:.0f} мм; с коркой {a.crust:g} мм: b = {b:.0f} мм",
        f"Уровень (от низшей точки профиля Б-Б): {z:.0f} мм; смачиваемая ширина {bL:.0f} + {bR:.0f} мм",
        f"Площадь живого сечения: {area * 1e4:.0f} см²",
        f"Скорость поверхности на оси: {u_prof.max():.2f} м/с; видимый край на {r_vis:.0f} мм от оси",
        f"Средняя по сечению скорость поверхности (взвешенная по глубине): {u_mean_s:.2f} м/с",
        f"Расход при k = {kmin:g}..{kmax:g}: {q_surf * kmin * 1000:.1f}..{q_surf * kmax * 1000:.1f} л/с"
        f" = {q_surf * kmin * a.rho * 3.6:.0f}..{q_surf * kmax * a.rho * 3.6:.0f} т/ч (ρ = {a.rho:g} кг/м³)",
    ]
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(a.speed_dir, "flow_rate.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    vis = cv2.cvtColor(cv2.normalize(med, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for nn, col in ((n_axis, (0, 200, 0)), (n_edge, (0, 0, 255))):
        cv2.line(vis, (0, int(H / 2 + nn)), (W - 1, int(H / 2 + nn)), col, 2)
    cv2.imwrite(os.path.join(a.speed_dir, "level_check.png"), vis[::2, ::2])


if __name__ == "__main__":
    main()
