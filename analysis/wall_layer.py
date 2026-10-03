#!/usr/bin/env python3
"""Ламинарное течение с холодным вязким слоем у медной водоохлаждаемой стенки.

Вязкость растёт к стенке: mu(s) = mu_c * (1 + (m - 1) * exp(-s / s0)), s - расстояние до стенки,
m = mu_стенки / mu_ядра, s0 - толщина слоя. Решается div(mu grad u) = -rho g sin(theta) в сечении
(u = 0 на стенке, du/dz = 0 на зеркале). Отсюда - форма поверхностного профиля u_s(r)/u_ось
(сравнивается с измеренной), коэффициенты k = Q / интеграл(u_s d dx) и C = Q / (A u_ось),
и эквивалентная (по сопротивлению) вязкость.

  python3 wall_layer.py <speed_dir> --crust 15 --scale-normal 0.65 --scale-along 0.537
"""
import argparse
import json
import math
import os

import cv2
import numpy as np

import hydraulics as hy


def solve(z, m, s0, hpx=0.5):
    """phi: div(nu grad phi) = -1, nu = 1 в ядре (единицы mu_c). Возвращает решение на сетке."""
    b = hy.half_width(z)
    nx = int(math.ceil(b / hpx)) + 2
    xs = np.arange(-nx, nx + 1) * hpx
    zs = (np.arange(int(math.ceil(z / hpx)) + 2) + 0.5) * hpx
    zs = zs[zs < z]
    X, Z = np.meshgrid(xs, zs)
    inside = Z > hy.bottom(X)
    # расстояние до стенки (зеркало стенкой не считается: отражаем область вверх)
    mir = np.vstack([inside, inside[::-1]]).astype(np.uint8)
    dist = cv2.distanceTransform(mir, cv2.DIST_L2, 5)[: inside.shape[0]] * hpx
    nu = np.where(inside, 1.0 + (m - 1.0) * np.exp(-dist / s0), 1.0 + (m - 1.0))
    # вязкость на гранях - среднее гармоническое
    def face(a, b_):
        return 2 * a * b_ / (a + b_)
    nuP = np.pad(nu, 1, mode="edge")
    nE, nW = face(nu, nuP[1:-1, 2:]), face(nu, nuP[1:-1, :-2])
    nN, nS = face(nu, nuP[2:, 1:-1]), face(nu, nuP[:-2, 1:-1])
    nN[-1] = 0.0                                              # зеркало: поток через него 0
    n = inside.sum()

    def apply(v):
        F = np.zeros(inside.shape)
        F[inside] = v
        P = np.pad(F, 1)                                      # вне области 0 (Дирихле)
        out = (nE * (F - P[1:-1, 2:]) + nW * (F - P[1:-1, :-2]) +
               nN * (F - P[2:, 1:-1]) + nS * (F - P[:-2, 1:-1])) / hpx ** 2
        return out[inside]

    v = np.zeros(n)
    r = np.ones(n)
    p = r.copy()
    rr = r @ r
    diag = (nE + nW + nN + nS)[inside] / hpx ** 2             # предобусловливатель Якоби
    zr = r / diag
    p = zr.copy()
    rz = r @ zr
    for _ in range(40000):
        Ap = apply(p)
        al = rz / (p @ Ap)
        v += al * p
        r -= al * Ap
        if math.sqrt(r @ r) < 1e-9 * math.sqrt(n):
            break
        zr = r / diag
        rz_new = r @ zr
        p = zr + rz_new / rz * p
        rz = rz_new
    F = np.zeros(inside.shape)
    F[inside] = v
    area = inside.sum() * hpx ** 2
    flux = F.sum() * hpx ** 2
    phi_s = F[-1]
    d = np.clip(z - hy.bottom(xs), 0, None)
    return dict(xs=xs, phi_s=phi_s, area=area, flux=flux, k=flux / float(np.sum(phi_s * d) * hpx),
                C=flux / area / phi_s.max(), phi_mean=flux / area)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("speed_dir")
    ap.add_argument("--scale-along", type=float, default=0.537)
    ap.add_argument("--scale-normal", type=float, default=0.65)
    ap.add_argument("--crust", type=float, default=15.0)
    ap.add_argument("--rho", type=float, default=3500.0)
    ap.add_argument("--m", type=float, nargs="+", default=[1, 10, 30, 100])
    ap.add_argument("--s0", type=float, nargs="+", default=[1, 2, 5, 10])
    ap.add_argument("--out")
    a = ap.parse_args()

    fr = json.load(open(os.path.join(a.speed_dir, "flow_rate.json")))
    sp = json.load(open(os.path.join(a.speed_dir, "result.json")))
    rows = [l.strip().split(",") for l in open(os.path.join(a.speed_dir, "profile.csv")).readlines()[1:]]
    y = np.array([float(r[0]) for r in rows])
    v = np.array([float(r[2]) for r in rows]) * a.scale_along / 1000.0
    n = y - sp["roi"][3] / 2.0
    side = np.sign(fr["edge_n"] - fr["axis_n"])
    r_mm = (n - fr["axis_n"]) * side * a.scale_normal
    b_vis = abs(fr["edge_n"] - fr["axis_n"]) * a.scale_normal
    z = hy.level_from_b(b_vis + a.crust)
    m = (r_mm >= -20) & (r_mm <= b_vis)
    rr, uu = np.abs(r_mm[m]), v[m]
    o = np.argsort(rr)
    rr, uu = rr[o], uu[o] / v.max()
    w = 64 * a.scale_normal
    S = math.sin(math.radians(hy.SLOPE_DEG))
    print(f"{os.path.basename(a.speed_dir.rstrip('/'))}: корка {a.crust:g} мм, уровень {z:.0f} мм, "
          f"u_ось {v.max():.2f} м/с; измерено u_s/u_ось при r = {rr.round(0).tolist()}: {uu.round(3).tolist()}")
    res = []
    for mm in a.m:
        for s0 in (a.s0 if mm > 1 else [1.0]):
            L = solve(z, mm, s0)
            r_f = np.linspace(0, hy.half_width(z), 400)
            prof = np.interp(r_f, np.abs(L["xs"][L["xs"] >= 0]), L["phi_s"][L["xs"] >= 0])
            sm = hy.box_smooth(r_f, prof, w)
            model = np.interp(rr, r_f, sm) / sm.max()
            rms = float(np.sqrt(np.mean((model - uu) ** 2)))
            # вязкость ядра, при которой равновесная скорость на оси = измеренной
            mu_c = a.rho * hy.G * S * L["phi_s"].max() * 1e-6 / v.max()
            res.append(dict(m=mm, s0=s0, k=L["k"], C=L["C"], rms=rms, mu_core=mu_c, mu_wall=mu_c * mm,
                            model=model.round(3).tolist()))
            print(f"  m {mm:5g}  s0 {s0:4g} мм: k {L['k']:.3f}  C {L['C']:.3f}  СКО формы {rms:.3f}  "
                  f"mu ядра {mu_c:.2f} / у стенки {mu_c * mm:.1f} Па·с  модель {model.round(2).tolist()}")
    if a.out:
        json.dump(dict(level_mm=z, r=rr.tolist(), measured=uu.tolist(), fits=res), open(a.out, "w"),
                  ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
