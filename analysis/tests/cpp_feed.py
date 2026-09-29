#!/usr/bin/env python3
"""Кормит cpp_replay кадрами записи: серии по --burst соседних кадров, раз в --interval секунд
(0 = серии подряд). Формат на stdout: 'B' uint32 n, затем n раз: uint32 w, uint32 h, double t, w*h байт."""
import argparse
import os
import struct
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import Timing, gray  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--burst", type=int, default=120)
ap.add_argument("--interval", type=float, default=0.0)
ap.add_argument("--max-bursts", type=int, default=0)
a = ap.parse_args()

tm = Timing(a.video)
out = sys.stdout.buffer
cap = cv2.VideoCapture(a.video)
i, bursts, t_next = tm.lo, 0, tm.time(tm.lo)
while i + a.burst <= tm.hi and (not a.max_bursts or bursts < a.max_bursts):
    if tm.time(i) < t_next:
        i += 1
        continue
    idx = list(range(i, i + a.burst))
    if not all(tm.linked(j) for j in idx[1:]):            # серия должна быть из соседних кадров камеры
        i += 1
        continue
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    out.write(b"B" + struct.pack("<I", a.burst))
    for j in idx:
        ok, f = cap.read()
        g = np.ascontiguousarray(gray(f))
        out.write(struct.pack("<IId", g.shape[1], g.shape[0], tm.time(j)) + g.tobytes())
    bursts += 1
    t_next = tm.time(i) + max(a.interval, (a.burst - 1) / tm.fps)
    i += a.burst
out.flush()
