#!/usr/bin/env python3
"""Калибровка камеры по меловым меткам на стенке жёлоба (разметка 29.09.2026).

Метки — путь по стенке от середины дна, см, шаг 5 см. Профиль «дуга R150 до касания +
стенки 22.5° от вертикали» известен до 135 мм (путь 22.3 см); выше — нерабочая зона
неизвестной формы, поэтому в подгонку идут только метки 10, 15, 20 см.

Модель (как CameraCalibration::ProjectAcross в magma_level.hpp): поперечная координата
кадра p = k·(x·sin β + h·cos β) + c, где x — горизонталь поперёк, h — высота, k — px/мм
в плоскости, перпендикулярной лучу зрения (вдоль потока ракурс не сжимает: масштаб
вдоль потока = 1/k мм/px, поперёк на поверхности = 1/(k·sin β)).

Метка ищется как пик яркости в столбце усреднённого кадра записи разметки; столбцы
задаются вручную (ряды черт на двух секциях жёлоба у стыка).

  python3 calib_marks.py video.mkv --cols 840 870 900 --cols 1180 1210 1240 --out calib.json
"""
import argparse
import json
import math

import cv2
import numpy as np

R, WALL = 150.0, math.radians(22.5)
S_T = R * (math.pi / 2 - WALL)                    # путь по дуге до касания, мм
X_T, H_T = R * math.cos(WALL), R * (1 - math.sin(WALL))
H_MAX = 135.0


def wall_point(s_mm):
    """Точка стенки (x, h), мм, по пути s от середины дна."""
    if s_mm <= S_T:
        t = s_mm / R
        return R * math.sin(t), R * (1 - math.cos(t))
    d = s_mm - S_T
    return X_T + d * math.sin(WALL), H_T + d * math.cos(WALL)


def mean_frame(video, step=5):
    cap = cv2.VideoCapture(video)
    acc, n, i = None, 0, 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i % step == 0:
            _, f = cap.retrieve()
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float64)
            acc = g if acc is None else acc + g
            n += 1
        i += 1
    return acc / n


def peaks(img, x0, half=12, min_contrast=4.0):
    col = img[:, x0 - half:x0 + half].mean(1)
    col = cv2.GaussianBlur(col.astype(np.float32)[:, None], (1, 7), 0)[:, 0]
    d = col - cv2.blur(col[:, None], (1, 61))[:, 0]
    out = []
    for y in range(6, len(d) - 7):
        if d[y] == d[y - 6:y + 7].max() and d[y] > min_contrast:
            a, b, c = d[y - 1], d[y], d[y + 1]
            den = a - 2 * b + c
            out.append((y + (0.5 * (a - c) / den if abs(den) > 1e-9 else 0.0), float(d[y])))
    return out


def fit(rows):
    """rows: [(s_мм, y_px, ряд)]; p = -y (метки выше по стенке - выше в кадре после поворота
    или ниже - знак k это покажет). Сдвиг c свой у каждого ряда. Перебор β, МНК по k, c."""
    best = None
    ids = sorted({r[2] for r in rows})
    for beta in np.arange(20.0, 85.0, 0.05):
        b = math.radians(beta)
        A, y = [], []
        for s, yp, rid in rows:
            x, h = wall_point(s)
            A.append([x * math.sin(b) + h * math.cos(b)] + [1.0 if rid == j else 0.0 for j in ids])
            y.append(yp)
        A, y = np.array(A), np.array(y)
        sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        res = float(np.sqrt(np.mean((A @ sol - y) ** 2)))
        if best is None or res < best[0]:
            best = (res, beta, sol[0])
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--cols", type=int, nargs="+", action="append", required=True,
                    help="столбцы одного ряда черт (повторить для каждого ряда)")
    ap.add_argument("--marks", type=float, nargs="+", default=[10, 15, 20, 25, 30],
                    help="метки ряда снизу вверх по стенке, см (как они идут в кадре)")
    ap.add_argument("--use", type=float, nargs="+", default=[10, 15, 20],
                    help="метки для подгонки (в пределах профиля до 135 мм)")
    ap.add_argument("--out")
    a = ap.parse_args()

    img = mean_frame(a.video)
    rows, detail = [], []
    for rid, cols in enumerate(a.cols):
        for x0 in cols:
            pk = peaks(img, x0)
            # черты - самые контрастные пики; берём len(marks) сильнейших и сортируем по y
            pk = sorted(sorted(pk, key=lambda p: -p[1])[:len(a.marks)])
            ys = [p[0] for p in pk][::-1]                  # снизу вверх по стенке = y убывает
            detail.append(dict(row=rid, col=x0, y=[round(v, 1) for v in ys]))
            if len(ys) != len(a.marks):
                continue
            for s_cm, yp in zip(a.marks, ys):
                if s_cm in a.use:
                    rows.append((s_cm * 10.0, yp, rid))
    res, beta, k = fit(rows)
    k = abs(k)
    # чувствительность: β по отдельным рядам/столбцам
    per = []
    for d in detail:
        if len(d["y"]) == len(a.marks):
            r = [(s * 10.0, yp, 0) for s, yp in zip(a.marks, d["y"]) if s in a.use]
            rr, bb, kk = fit(r)
            per.append(dict(col=d["col"], beta=round(bb, 1), mm_per_px=round(1 / abs(kk), 4)))
    betas = [p["beta"] for p in per]
    out = dict(beta_deg=round(beta, 2), beta_spread_deg=round(float(np.std(betas)), 2) if betas else None,
               mm_per_px_along=round(1 / k, 4), mm_per_px_across_surface=round(1 / (k * math.sin(math.radians(beta))), 4),
               residual_px=round(res, 2), used_marks_cm=a.use, points=len(rows), per_column=per, peaks=detail,
               profile="R150 + стенки 22.5°, метки = путь по стенке от середины дна")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    if a.out:
        json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
