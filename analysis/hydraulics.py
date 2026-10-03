#!/usr/bin/env python3
"""Гидродинамика расплава в жёлобе по результатам flow_speed/flow_rate.

Что считается для каждой записи:
  - сечение по уровню (профиль R150 + стенки 22.5°, при гарнисаже толщиной delta - профиль,
    смещённый внутрь на delta), площадь A, смоченный периметр P, R_h = A/P, ширина зеркала B;
  - ламинарное течение в ЭТОМ сечении: численное решение лапласиана скорости
        mu * (u_xx + u_zz) = -rho * g * sin(theta),  u = 0 на стенке, du/dz = 0 на зеркале;
    отсюда точный коэффициент k_lam = Q / интеграл(u_s(x) * d(x) dx), отношение средней
    скорости к поверхностной на оси и форма поверхностного профиля u_s(x) / u_s(0);
  - эквивалентная вязкость: при какой mu равномерное ламинарное течение под уклоном дало бы
    измеренную скорость; для турбулентного варианта - эквивалентный коэффициент трения Дарси;
  - числа Re и Fr;
  - сравнение формы измеренного профиля u_s(r) с ламинарной (по решению) и турбулентной
    (u_s ~ d^(2/3), Маннинг) моделями: модели сглаживаются окном PIV (64 px) как измерение.

  python3 hydraulics.py <speed_dir> [<speed_dir> ...] --scale-normal 0.65 --scale-along 0.537
"""
import argparse
import json
import math
import os

import numpy as np

R, WALL_DEG, H_MAX = 150.0, 22.5, 135.0
G, SLOPE_DEG = 9.81, 6.35


def bottom(x, delta=0.0):
    """Высота дна (мм) профиля R150 + стенки 22.5° на расстоянии |x| от оси; при гарнисаже
    delta - внутренняя поверхность гарнисажа (смещение профиля по нормали на delta)."""
    a = math.radians(WALL_DEG)
    r = R - delta
    x = np.abs(np.asarray(x, float))
    xt = r * math.cos(a)                       # точка касания дуги и стенки
    arc = R - np.sqrt(np.clip(r * r - np.minimum(x, xt) ** 2, 0, None))
    wall = (R - r * math.sin(a)) + (x - xt) / math.tan(a)
    return np.where(x <= xt, arc, wall)


def half_width(z, delta=0.0):
    xs = np.linspace(0, 200, 20001)
    return float(np.interp(z, bottom(xs, delta), xs))


def level_from_b(b, delta=0.0):
    zs = np.linspace(delta + 0.01, H_MAX + 60, 4000)
    hw = np.array([half_width(z, delta) for z in zs])
    return float(np.interp(b, hw, zs)) if b <= hw[-1] else float("nan")


def laminar(z, delta=0.0, hpx=0.5):
    """phi: решение phi_xx + phi_zz = -1 в сечении (мм), phi = 0 на стенке, dphi/dz = 0 на зеркале.
    Скорость u = rho g S / mu * phi (phi в мм² -> м² множителем 1e-6). CG по маске."""
    b = half_width(z, delta)
    nx = int(math.ceil(b / hpx)) + 2
    nz = int(math.ceil((z - bottom(0, delta)) / hpx)) + 2
    xs = (np.arange(-nx, nx + 1)) * hpx
    zs = bottom(0, delta) + (np.arange(nz) + 0.5) * hpx       # центры ячеек снизу вверх
    zs = zs[zs < z]
    X, Z = np.meshgrid(xs, zs)                                 # строки - z, последняя строка у зеркала
    inside = Z > bottom(X, delta)
    idx = -np.ones(inside.shape, int)
    idx[inside] = np.arange(inside.sum())
    n = inside.sum()

    def apply(v):
        F = np.zeros(inside.shape)
        F[inside] = v
        P = np.pad(F, 1)                                      # вне области - 0 (стенка, Дирихле)
        P[-1, 1:-1] = F[-1]                                   # над зеркалом - зеркальная ячейка (Нейман)
        lap = (P[1:-1, :-2] + P[1:-1, 2:] + P[:-2, 1:-1] + P[2:, 1:-1] - 4 * F) / hpx ** 2
        return -lap[inside]

    rhs = np.ones(n)
    v = np.zeros(n)
    r = rhs - apply(v)
    p = r.copy()
    rr = r @ r
    for _ in range(20000):
        Ap = apply(p)
        al = rr / (p @ Ap)
        v += al * p
        r -= al * Ap
        rn = r @ r
        if math.sqrt(rn) < 1e-9 * math.sqrt(n):
            break
        p = r + rn / rr * p
        rr = rn
    F = np.zeros(inside.shape)
    F[inside] = v
    area = inside.sum() * hpx ** 2
    flux = F.sum() * hpx ** 2                                 # интеграл phi dA, мм⁴
    phi_s = F[-1]                                             # верхний ряд ~ зеркало (на hpx/2 ниже)
    d = np.clip(z - bottom(xs, delta), 0, None)
    return dict(xs=xs, phi_s=phi_s, d=d, area=area, flux=flux,
                k_lam=flux / float(np.sum(phi_s * d) * hpx),
                mean_over_axis=flux / area / phi_s.max(), phi_mean=flux / area)


def wet_perimeter(z, delta=0.0):
    xs = np.linspace(-half_width(z, delta), half_width(z, delta), 4001)
    y = bottom(xs, delta)
    return float(np.sum(np.hypot(np.diff(xs), np.diff(y))))


def box_smooth(r, f, width):
    """Сглаживание функции f(r) окном width (как плитка PIV) на сетке r."""
    out = np.empty_like(f)
    for i, ri in enumerate(r):
        m = np.abs(r - ri) <= width / 2
        out[i] = f[m].mean()
    return out


def colebrook_eps(f, Re, Dh):
    """Шероховатость eps (мм), при которой формула Колбрука даёт коэффициент f."""
    s = 1 / math.sqrt(f)
    x = 10 ** (-s / 2.0) - 2.51 / (Re * math.sqrt(f))
    return 3.7 * Dh * x if x > 0 else float("nan")


def analyse(speed_dir, scale_along, scale_normal, crust, rho, deltas, mus, tile_px=64):
    fr = json.load(open(os.path.join(speed_dir, "flow_rate.json")))
    sp = json.load(open(os.path.join(speed_dir, "result.json")))
    h_roi = sp["roi"][3]
    rows = [l.strip().split(",") for l in open(os.path.join(speed_dir, "profile.csv")).readlines()[1:]]
    y = np.array([float(r[0]) for r in rows])
    v = np.array([float(r[2]) for r in rows])
    n = y - h_roi / 2.0
    n_axis, n_edge = fr["axis_n"], fr["edge_n"]
    side = np.sign(n_edge - n_axis)
    r_mm = (n - n_axis) * side * scale_normal                 # >0 - к краю корки
    u = v * scale_along / 1000.0
    b_vis = abs(n_edge - n_axis) * scale_normal
    b = b_vis + crust
    S = math.sin(math.radians(SLOPE_DEG))
    out = dict(record=os.path.basename(speed_dir.rstrip("/")), b_mm=b, b_visible_mm=b_vis,
               u_axis=float(u.max()), cases=[])
    # измеренный поверхностный профиль: точки между осью и видимым краем (+ под коркой - линейно к 0)
    m = (r_mm >= -20) & (r_mm <= b_vis + tile_px * scale_normal / 2)
    rr, uu = np.abs(r_mm[m]), u[m]
    o = np.argsort(rr)
    rr, uu = rr[o], uu[o]
    for delta in deltas:
        z = level_from_b(b, delta)
        if not np.isfinite(z):
            continue
        L = laminar(z, delta)
        xs, d = L["xs"], L["d"]
        bw = half_width(z, delta)
        us = np.interp(np.abs(xs), rr, uu)
        tail = np.abs(xs) > rr[-1]
        us[tail] = uu[-1] * np.clip((bw - np.abs(xs[tail])) / max(bw - rr[-1], 1e-6), 0, 1)
        hx = xs[1] - xs[0]
        q_surf = float(np.sum(us * d) * hx) * 1e-6           # м³/с при k = 1
        A = L["area"] * 1e-6
        P = wet_perimeter(z, delta) * 1e-3
        Rh = A / P
        B = 2 * bw * 1e-3
        Dhyd = A / B
        case = dict(delta_mm=delta, level_mm=z, depth_mm=z - float(bottom(0, delta)), A_cm2=A * 1e4,
                    P_mm=P * 1e3, Rh_mm=Rh * 1e3, B_mm=B * 1e3, k_lam=L["k_lam"],
                    lam_mean_over_axis=L["mean_over_axis"], q_surf_l_s=q_surf * 1e3, regimes={})
        for name, k in (("ламинарный", L["k_lam"]), ("турбулентный", 0.87)):
            U = k * q_surf / A
            Fr = U / math.sqrt(G * Dhyd)
            mu_eq = rho * G * S * L["phi_mean"] * 1e-6 / U       # Па·с, равномерный ламинарный
            f_eq = 8 * G * Rh * S / U ** 2
            reg = dict(k=k, U_mean=U, Q_l_s=k * q_surf * 1e3, Q_t_h=k * q_surf * rho * 3.6, Fr=Fr,
                       mu_eq_Pa_s=mu_eq, f_darcy_eq=f_eq,
                       Re={str(mu): rho * U * 4 * Rh / mu for mu in mus})
            reg["eps_mm"] = {str(mu): colebrook_eps(f_eq, reg["Re"][str(mu)], 4 * Rh * 1e3) for mu in mus}
            case["regimes"][name] = reg
        # форма профиля: модель на тех же r, сглаженная окном плитки PIV
        r_fine = np.linspace(0, bw, 400)
        lam = np.interp(r_fine, np.abs(xs[xs >= 0]), L["phi_s"][xs >= 0])
        tur = np.clip(z - bottom(r_fine, delta), 0, None) ** (2 / 3)
        w = tile_px * scale_normal
        lam_s, tur_s = box_smooth(r_fine, lam, w), box_smooth(r_fine, tur, w)
        meas = uu / uu.max()
        mvis = rr <= b_vis
        shape = dict(r_mm=rr[mvis].round(1).tolist(), measured=meas[mvis].round(3).tolist(),
                     laminar=(np.interp(rr[mvis], r_fine, lam_s) / lam_s.max()).round(3).tolist(),
                     turbulent=(np.interp(rr[mvis], r_fine, tur_s) / tur_s.max()).round(3).tolist())
        for key in ("laminar", "turbulent"):
            e = np.array(shape[key]) - np.array(shape["measured"])
            shape["rms_" + key] = float(np.sqrt(np.mean(e ** 2)))
        case["shape"] = shape
        out["cases"].append(case)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("speed_dirs", nargs="+")
    ap.add_argument("--scale-along", type=float, default=0.537)
    ap.add_argument("--scale-normal", type=float, default=0.65)
    ap.add_argument("--crust", type=float, default=15.0)
    ap.add_argument("--rho", type=float, default=3500.0)
    ap.add_argument("--delta", type=float, nargs="+", default=[0.0, 10.0, 20.0], help="толщина гарнисажа, мм")
    ap.add_argument("--mu", type=float, nargs="+", default=[0.2, 0.5, 2.0], help="вязкость для Re, Па·с")
    ap.add_argument("--out")
    a = ap.parse_args()
    res = [analyse(d, a.scale_along, a.scale_normal, a.crust, a.rho, a.delta, a.mu) for d in a.speed_dirs]
    for r in res:
        print(f"\n== {r['record']}: u_ось {r['u_axis']:.2f} м/с, b {r['b_mm']:.0f} мм (видимая {r['b_visible_mm']:.0f})")
        for c in r["cases"]:
            print(f"  гарнисаж {c['delta_mm']:.0f} мм: уровень от меди {c['level_mm']:.0f} мм, глубина {c['depth_mm']:.0f} мм, "
                  f"A {c['A_cm2']:.0f} см², R_h {c['Rh_mm']:.0f} мм, B {c['B_mm']:.0f} мм, k_lam {c['k_lam']:.3f} "
                  f"(U/u_ось {c['lam_mean_over_axis']:.3f}), A·u_ось {c['A_cm2'] * 1e-4 * r['u_axis'] * 1e3:.1f} л/с")
            for name, g in c["regimes"].items():
                re = ", ".join(f"{k}: {v:.0f}" for k, v in g["Re"].items())
                print(f"    {name:12s} k {g['k']:.2f}: U {g['U_mean']:.2f} м/с, Q {g['Q_l_s']:.1f} л/с = {g['Q_t_h']:.0f} т/ч, "
                      f"Fr {g['Fr']:.2f}, mu_экв {g['mu_eq_Pa_s']:.2f} Па·с, f_экв {g['f_darcy_eq']:.3f}, Re(mu) {re}")
            s = c["shape"]
            print(f"    форма u_s(r)/u_ось: r {s['r_mm']}\n      измерено {s['measured']}\n      ламинар  {s['laminar']} "
                  f"(СКО {s['rms_laminar']:.3f})\n      турбул.  {s['turbulent']} (СКО {s['rms_turbulent']:.3f})")
    if a.out:
        json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
