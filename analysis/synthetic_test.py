#!/usr/bin/env python3
"""
Синтетическая проверка flow_speed.py: генерирует видео с известной скоростью.
Текстура едет справа налево со скоростью V px/кадр, её контраст то пропадает,
то появляется; есть неподвижные детали (края желоба) и шум.

  python synthetic_test.py out.mp4 --v 3.7
  python flow_speed.py out.mp4 --roi 40,90,560,100
"""
import argparse

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--v", type=float, default=3.7, help="скорость, px/кадр (справа налево)")
    ap.add_argument("--fps", type=float, default=60)
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dup-every", type=int, default=0,
                    help="каждый N-й кадр - повтор предыдущего (как в записи экрана)")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    W, H = 640, 280
    n = int(a.fps * a.seconds)
    L = W + int(abs(a.v) * n) + 200
    tex = cv2.GaussianBlur(rng.normal(0, 1, (100, L)).astype(np.float32), (0, 0), 2.5)
    # "пятна пены" - медленно меняющаяся по длине огибающая контраста
    env = np.clip(np.sin(np.linspace(0, 40, L)) * 0.8 + rng.normal(0, 0.3, L), 0, None).astype(np.float32)
    tex = tex / tex.std() * env

    static = np.full((H, W), 90, np.float32)
    static[80:85] = 220
    static[195:200] = 220
    for cx in range(0, W, 97):                      # неподвижные метки
        cv2.circle(static, (cx, 140), 6, 200, -1)

    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W, H))
    uniq, img = 0, None
    for i in range(n):
        if a.dup_every and img is not None and i % a.dup_every == a.dup_every - 1:
            vw.write(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
            continue
        off = a.v * uniq
        uniq += 1
        i0 = int(np.floor(off))
        fr = off - i0
        seg = (1 - fr) * tex[:, i0:i0 + W] + fr * tex[:, i0 + 1:i0 + 1 + W]  # сдвиг влево
        img = static.copy()
        img[90:190] += 35 * seg
        img += rng.normal(0, 4, img.shape).astype(np.float32)
        img = np.clip(img, 0, 255).astype(np.uint8)
        vw.write(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    vw.release()
    print(f"{a.out}: {n} кадров ({uniq} уникальных), истинная скорость "
          f"{a.v} px/уник.кадр = {a.v * uniq / n * a.fps:.1f} px/с")


if __name__ == "__main__":
    main()
