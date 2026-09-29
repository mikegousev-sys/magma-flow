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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("speed_dir")
    ap.add_argument("video")
    ap.add_argument("--scale-along", type=float, default=0.548, help="мм/px вдоль потока")
    ap.add_argument("--scale-normal", type=float, default=0.638, help="мм/px поперёк потока")
    ap.add_argument("--crust", type=float, default=25.0, help="на сколько корка заходит на жидкость, мм")
    ap.add_argument("--profile", default=os.path.join(HERE, "gutter_profile_R150.csv"))
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
        # вершина параболы по 3 точкам; шаг может быть неравномерным (полосы с редкой
        # текстурой flow_speed выбрасывает из profile.csv)
        (x0, x1, x2), (y0, y1, y2) = n_prof[i - 1:i + 2], vp[i - 1:i + 2]
        num_ = (x1 - x0) ** 2 * (y1 - y2) - (x1 - x2) ** 2 * (y1 - y0)
        den = (x1 - x0) * (y1 - y2) - (x1 - x2) * (y1 - y0)
        n_axis = x1 - 0.5 * num_ / den if den != 0 else x1
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
    # участок [lo, hi) - так же, как в flow_speed: по меткам времени или по FPS
    fcsv = os.path.join(os.path.dirname(os.path.abspath(a.video)), "frames.csv")
    if os.path.isfile(fcsv):
        with open(fcsv, newline="") as fh:
            ts = np.array([float(r["time_rel_s"]) for r in csv.DictReader(fh)])
        lo = int(np.searchsorted(ts, ts[0] + a.start))
        hi = int(np.searchsorted(ts, ts[0] + a.start + a.seconds)) if a.seconds > 0 else len(ts)
    else:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        if fps <= 0 and (a.start > 0 or a.seconds > 0):
            sys.exit("Не удалось определить FPS для --start/--seconds")
        lo = int(a.start * fps)
        hi = int((a.start + a.seconds) * fps) if a.seconds > 0 else nfr
    hi = min(hi, nfr)
    if hi <= lo:
        sys.exit(f"Пустой участок кадров {lo}..{hi} (в видео {nfr})")
    frames, valid = [], None
    for j in np.unique(np.linspace(lo, hi - 1, 60).astype(int)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, j)
        ok, f = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)          # uint8: 60 кадров 1600x1600 без float-копий
        if valid is None:
            valid = cv2.warpAffine(np.ones_like(g), A, (W, H),
                                   flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP) > 0
        frames.append(cv2.warpAffine(g, A, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP))
    cap.release()
    if not frames:
        sys.exit(f"Не удалось прочитать кадры {lo}..{hi - 1} из {a.video}")
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

    # скорость поверхности как функция расстояния от оси, м/с (от уровня не зависит)
    r_prof = np.abs(n_prof - n_axis) * a.scale_normal          # мм
    u_prof = vp * a.scale_along / 1000.0
    order = np.argsort(r_prof)
    r_s, u_s = r_prof[order], u_prof[order]
    r_vis = abs(n_edge - n_axis) * a.scale_normal

    head = [
        f"Результат скорости: {a.speed_dir}",
        f"ROI {x},{y},{w},{h}, направление {ang:+.1f}°",
        f"Ось потока (максимум скорости): n = {n_axis:+.1f} px{axis_note}",
        f"Край корки A: n = {n_edge:+.1f} px (разброс по длине P10..P90 {np.percentile(edges, 10):+.0f}..{np.percentile(edges, 90):+.0f})",
        f"Видимая полуширина поверхности: {d_px:.0f} px = {d_px * a.scale_normal:.0f} мм; с коркой {a.crust:g} мм: b = {b:.0f} мм",
    ]
    speed_line = f"Скорость поверхности на оси: {u_prof.max():.2f} м/с; видимый край на {r_vis:.0f} мм от оси"

    def write_out(lines):
        txt = "\n".join(lines)
        print(txt)
        with open(os.path.join(a.speed_dir, "flow_rate.txt"), "w", encoding="utf-8") as f:
            f.write(txt + "\n")
        vis = cv2.cvtColor(cv2.normalize(med, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        for nn, col in ((n_axis, (0, 200, 0)), (n_edge, (0, 0, 255))):
            cv2.line(vis, (0, int(H / 2 + nn)), (W - 1, int(H / 2 + nn)), col, 2)
        cv2.imwrite(os.path.join(a.speed_dir, "level_check.png"), vis[::2, ::2])

    P = gutter(a.profile)
    prof_name = os.path.splitext(os.path.basename(a.profile))[0]
    z = level_from_half_width(P, b, a.side)
    if not np.isfinite(z):
        # скорость, ось и край всё равно пишем: по ним collect_metrics ставит оценку
        write_out(head + [f"Уровень: не определён - b больше максимальной полуширины профиля {prof_name} "
                          f"({np.abs(P[:, 0]).max():.0f} мм); край или ось найдены неверно",
                          speed_line])
        sys.exit(1)
    xs = np.linspace(P[:, 0].min(), P[:, 0].max(), 4000)
    zb = np.interp(xs, P[:, 0], P[:, 1])
    dep = np.clip(z - zb, 0, None) / 1000.0                    # м
    area = float(np.trapezoid(dep, xs / 1000.0))
    wet = dep > 0
    bL, bR = (-xs[wet].min(), xs[wet].max()) if wet.any() else (0.0, 0.0)

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

    write_out(head + [
        f"Уровень (от низшей точки профиля {prof_name}): {z:.0f} мм; смачиваемая ширина {bL:.0f} + {bR:.0f} мм",
        f"Площадь живого сечения: {area * 1e4:.0f} см²",
        speed_line,
        f"Средняя по сечению скорость поверхности (взвешенная по глубине): {u_mean_s:.2f} м/с",
        f"Расход при k = {kmin:g}..{kmax:g}: {q_surf * kmin * 1000:.1f}..{q_surf * kmax * 1000:.1f} л/с"
        f" = {q_surf * kmin * a.rho * 3.6:.0f}..{q_surf * kmax * a.rho * 3.6:.0f} т/ч (ρ = {a.rho:g} кг/м³)",
    ])

if __name__ == "__main__":
    main()
