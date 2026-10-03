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

from common import Timing, flow_affine, gray, parabola_vertex, read_frames, read_json, write_json

HERE = os.path.dirname(os.path.abspath(__file__))


def read_speed_result(speed_dir):
    """result.json от flow_speed; для старых результатов - ROI и угол из summary.txt."""
    p = os.path.join(speed_dir, "result.json")
    if os.path.isfile(p):
        return read_json(p)
    txt = open(os.path.join(speed_dir, "summary.txt"), encoding="utf-8").read()
    return dict(roi=[int(v) for v in re.search(r"ROI: (\d+),(\d+),(\d+),(\d+)", txt).groups()],
                angle=float(re.search(r"Направление течения: ([-+]?\d+\.?\d*)°", txt).group(1)))


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
    """Полуширина сечения на уровне z (скаляр или массив; от оси до стенки)
    для side=+1 (правая) / -1 (левая)."""
    s = P[P[:, 0] * side > 0]
    s = s[np.argsort(s[:, 1])]
    return np.interp(z, s[:, 1], np.abs(s[:, 0]))


def level_from_half_width(P, b, side):
    zs = np.linspace(0, P[:, 1].max(), 2000)
    # на каждом уровне - самая дальняя от оси точка профиля ниже него
    hw = np.maximum.accumulate(half_width(P, zs, side))
    if b > hw[-1]:
        return np.nan
    return float(np.interp(b, hw, zs))


def crust_edges(med, valid, y_axis, sign, step=4):
    """Край корки A по каждому столбцу X (вдоль течения) изображения med в повёрнутой системе:
    идём от оси (строка y_axis) в сторону sign (+1/-1) и берём самую дальнюю от оси точку
    ярче середины между потоком и фоном (тёмные пятна текстуры внутри потока не мешают).
    Возвращает координаты Y края для столбцов, где фон за краем виден."""
    H, W = med.shape
    edges = []
    if not 0 <= y_axis < H:
        return np.array(edges)
    for X in range(0, W, step):
        col_ok = valid[:, X]
        ys = np.arange(y_axis, 0 if sign < 0 else H - 1, int(sign))
        ys = ys[col_ok[ys]]
        if len(ys) < 40 or not col_ok[y_axis]:
            continue
        prof = cv2.GaussianBlur(med[ys, X].astype(np.float32)[:, None], (1, 9), 0)[:, 0]
        bgl = np.median(prof[-15:])                              # фон за краем
        top = np.percentile(prof[:max(5, len(prof) // 3)], 90)
        if top - bgl < 5:
            continue
        above = np.flatnonzero(prof >= (top + bgl) / 2)
        if len(above) and len(ys) - above[-1] >= 15:              # иначе фон в столбце не виден
            edges.append(ys[above[-1]])
    return np.array(edges, float)


def section(P, z, n=4000):
    """Живое сечение при уровне z, мм: x (мм), глубина (м), площадь (м²), смоченные полуширины (мм)."""
    xs = np.linspace(P[:, 0].min(), P[:, 0].max(), n)
    dep = np.clip(z - np.interp(xs, P[:, 0], P[:, 1]), 0, None) / 1000.0
    wet = dep > 0
    bL, bR = (-xs[wet].min(), xs[wet].max()) if wet.any() else (0.0, 0.0)
    return xs, dep, float(np.trapezoid(dep, xs / 1000.0)), bL, bR


def surface_flow(xs, dep, r_s, u_s, bL, bR):
    """Расход при k = 1, м³/с: интеграл скорости поверхности u_s(|x|) на глубину. За последней
    точкой профиля скорости (под коркой) - линейный спад до 0 у стенки."""
    def u_at(r, bw):
        r = np.abs(r)
        u = np.interp(r, r_s, u_s)
        tail = r > r_s[-1]
        u[tail] = u_s[-1] * np.clip((bw - r[tail]) / max(bw - r_s[-1], 1e-6), 0, 1)
        return u
    u_x = np.where(xs >= 0, u_at(xs, bR), u_at(xs, bL))
    return float(np.trapezoid(u_x * dep, xs / 1000.0))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("speed_dir")
    ap.add_argument("video")
    ap.add_argument("--scale-along", type=float, default=0.548, help="мм/px вдоль потока")
    ap.add_argument("--scale-normal", type=float, default=0.638, help="мм/px поперёк потока")
    ap.add_argument("--crust", type=float, default=15.0, help="на сколько корка заходит на жидкость, мм")
    ap.add_argument("--profile", default=os.path.join(HERE, "gutter_profile_R150.csv"))
    ap.add_argument("--side", type=int, default=1, help="+1: камера видит правую половину чертежа")
    ap.add_argument("--k", default="0.67,0.85", help="отношение средней скорости к поверхностной (мин,макс)")
    ap.add_argument("--rho", type=float, default=3500.0, help="плотность шлака, кг/м3")
    ap.add_argument("--start", type=float, help="начало участка, с (по умолчанию - как в result.json)")
    ap.add_argument("--seconds", type=float, help="длительность участка, с (по умолчанию - как в result.json)")
    a = ap.parse_args()

    sp = read_speed_result(a.speed_dir)
    (x, y, w, h), ang = sp["roi"], sp["angle"]
    start = a.start if a.start is not None else sp.get("start", 0.0)
    seconds = a.seconds if a.seconds is not None else sp.get("seconds", 0.0)
    yp, vp = read_profile(a.speed_dir)
    n_prof = yp - h / 2.0                        # смещение поперёк потока от центра ROI, px

    # ось потока: вершина параболы вокруг максимума профиля скорости
    i = int(np.argmax(vp))
    if 0 < i < len(vp) - 1:
        # вершина параболы по 3 точкам; шаг может быть неравномерным (полосы с редкой
        # текстурой flow_speed выбрасывает из profile.csv)
        n_axis = parabola_vertex(n_prof[i - 1:i + 2], vp[i - 1:i + 2])
        axis_note = ""
    else:
        n_axis = n_prof[i]
        axis_note = " (максимум на краю профиля - ось может быть дальше, уровень занижен)"
    edge_sign = -1.0 if n_prof[np.argmin(vp)] < n_axis else 1.0   # в какую сторону от оси край

    # край корки: медиана кадров в повёрнутой системе (Y вдоль e2, как в flow_speed)
    W = H = 1600
    A = flow_affine(x + w / 2.0, y + h / 2.0, ang, W, H)
    tm = Timing(a.video, None, start, seconds)            # участок - так же, как в flow_speed
    cap = cv2.VideoCapture(a.video)
    src_hw = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frames = read_frames(cap, np.unique(np.linspace(tm.lo, tm.hi - 1, 60).astype(int)),
                         lambda f: cv2.warpAffine(gray(f), A, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP))
    cap.release()
    if not frames:
        sys.exit(f"Не удалось прочитать кадры {tm.lo}..{tm.hi - 1} из {a.video}")
    valid = cv2.warpAffine(np.ones(src_hw, np.uint8), A, (W, H), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP) > 0
    med = np.median(np.array(frames), axis=0)         # uint8: 60 кадров 1600x1600 без float-копий
    edges = crust_edges(med, valid, int(round(H / 2 + n_axis)), edge_sign) - H / 2
    if len(edges) < 10:
        sys.exit("Край корки не найден")
    n_edge = float(np.median(edges))
    d_px = abs(n_axis - n_edge)
    b = d_px * a.scale_normal + a.crust

    # скорость поверхности как функция расстояния от оси, м/с (от уровня не зависит)
    r_prof = np.abs(n_prof - n_axis) * a.scale_normal          # мм
    u_prof = vp * a.scale_along / 1000.0
    order = np.argsort(r_prof)
    r_s, u_s = r_prof[order], u_prof[order]
    r_vis = abs(n_edge - n_axis) * a.scale_normal
    P = gutter(a.profile)
    prof_name = os.path.splitext(os.path.basename(a.profile))[0]

    head = [
        f"Результат скорости: {a.speed_dir}",
        f"ROI {x},{y},{w},{h}, направление {ang:+.1f}°",
        f"Ось потока (максимум скорости): n = {n_axis:+.1f} px{axis_note}",
        f"Край корки A: n = {n_edge:+.1f} px (разброс по длине P10..P90 {np.percentile(edges, 10):+.0f}..{np.percentile(edges, 90):+.0f})",
        f"Видимая полуширина поверхности: {d_px:.0f} px = {d_px * a.scale_normal:.0f} мм; с коркой {a.crust:g} мм: b = {b:.0f} мм",
    ]
    speed_line = f"Скорость поверхности на оси: {u_prof.max():.2f} м/с; видимый край на {r_vis:.0f} мм от оси"

    res = dict(video=os.path.abspath(a.video), start=start, seconds=seconds, profile=prof_name,
               axis_n=n_axis, axis_on_edge=bool(axis_note), edge_n=n_edge,
               edge_p10=np.percentile(edges, 10), edge_p90=np.percentile(edges, 90),
               visible_half_mm=r_vis, crust_mm=a.crust, b_mm=b, v_axis_m_s=u_prof.max(),
               **{k: sp.get(k) for k in ("pairs", "windows", "good_windows", "consensus_px_s",
                                         "methods_spread_pct", "core_px_s")})

    def write_out(lines, **extra):
        txt = "\n".join(lines)
        print(txt)
        with open(os.path.join(a.speed_dir, "flow_rate.txt"), "w", encoding="utf-8") as f:
            f.write(txt + "\n")
        write_json(os.path.join(a.speed_dir, "flow_rate.json"), {**res, **extra})
        vis = cv2.cvtColor(cv2.normalize(med, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        for nn, col in ((n_axis, (0, 200, 0)), (n_edge, (0, 0, 255))):
            cv2.line(vis, (0, int(H / 2 + nn)), (W - 1, int(H / 2 + nn)), col, 2)
        cv2.imwrite(os.path.join(a.speed_dir, "level_check.png"), vis[::2, ::2])

    z = level_from_half_width(P, b, a.side)
    if not np.isfinite(z):
        # скорость, ось и край всё равно пишем: по ним collect_metrics ставит оценку
        write_out(head + [f"Уровень: не определён - b больше максимальной полуширины профиля {prof_name} "
                          f"({np.abs(P[:, 0]).max():.0f} мм); край или ось найдены неверно",
                          speed_line], level_mm=None)
        sys.exit(1)
    xs, dep, area, bL, bR = section(P, z)
    q_surf = surface_flow(xs, dep, r_s, u_s, bL, bR)           # м3/с при k = 1
    kmin, kmax = (float(v) for v in a.k.split(","))
    u_mean_s = q_surf / area if area > 0 else np.nan

    write_out(head + [
        f"Уровень (от низшей точки профиля {prof_name}): {z:.0f} мм; смачиваемая ширина {bL:.0f} + {bR:.0f} мм",
        f"Площадь живого сечения: {area * 1e4:.0f} см²",
        speed_line,
        f"Средняя по сечению скорость поверхности (взвешенная по глубине): {u_mean_s:.2f} м/с",
        f"Расход при k = {kmin:g}..{kmax:g}: {q_surf * kmin * 1000:.1f}..{q_surf * kmax * 1000:.1f} л/с"
        f" = {q_surf * kmin * a.rho * 3.6:.0f}..{q_surf * kmax * a.rho * 3.6:.0f} т/ч (ρ = {a.rho:g} кг/м³)",
    ], level_mm=z, area_cm2=area * 1e4, wet_left_mm=bL, wet_right_mm=bR, u_surface_mean_m_s=u_mean_s,
       k=[kmin, kmax], rho=a.rho, q_l_s=[q_surf * kmin * 1000, q_surf * kmax * 1000],
       t_h=[q_surf * kmin * a.rho * 3.6, q_surf * kmax * a.rho * 3.6])

if __name__ == "__main__":
    main()
