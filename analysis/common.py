"""Общие функции для flow_speed.py, flow_rate.py и collect_metrics.py."""
import csv
import json
import os
import sys

import cv2
import numpy as np

GOOD_SPREAD_PCT = 10.0   # окно "надёжно", если согласованные методы расходятся не больше, %
AGREE_TOL = 0.25         # метод согласован, если отличается от медианы методов не больше чем на 25%
TRIM_LO, TRIM_HI = 0.5, 1.6   # отбраковка выбросов в окне: доля от медианы


# ---------------------------------------------------------------- время кадров

class Timing:
    """Время кадров и участок анализа.

    С frames.csv (focus_preview): t = time_rel_s, пара (i-1, i) связана, если кадры
    идут подряд у драйвера (с учётом every_n). Без него: t = i / fps, связаны все.
    """

    def __init__(self, video, frames_csv=None, start=0.0, seconds=0.0):
        cap = cv2.VideoCapture(video)
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        self.nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        path = frames_csv or os.path.join(os.path.dirname(os.path.abspath(video)), "frames.csv")
        self.ts = self.linked_ok = None
        self.csv_path = path if os.path.isfile(path) else None
        if self.csv_path:
            with open(path, newline="") as f:
                rd = list(csv.DictReader(f))
            self.ts = np.array([float(r["time_rel_s"]) for r in rd])
            dseq = np.diff([int(r["driver_seq"]) for r in rd])
            self.step = int(np.median(dseq)) if dseq.size else 1      # every_n: 2, 5 ...
            self.linked_ok = dseq <= self.step
            self.fps = len(self.ts) / max(self.ts[-1] - self.ts[0], 1e-6)
            lo = np.searchsorted(self.ts, self.ts[0] + start)
            hi = np.searchsorted(self.ts, self.ts[0] + start + seconds) if seconds > 0 else len(self.ts)
        else:
            if self.fps <= 0:
                sys.exit("Не удалось определить FPS")
            lo = start * self.fps
            hi = (start + seconds) * self.fps if seconds > 0 else self.nframes
        self.lo, self.hi = int(lo), int(min(hi, self.nframes))
        if self.hi <= self.lo:
            sys.exit(f"Пустой участок кадров {self.lo}..{self.hi} (в видео {self.nframes})")

    @property
    def has_ts(self):
        return self.ts is not None

    def time(self, i):
        return self.ts[i] if self.has_ts else i / self.fps

    def linked(self, i):
        """Пара (i-1, i) - соседние кадры камеры (без разрыва серии)."""
        return not self.has_ts or (0 < i < len(self.ts) and bool(self.linked_ok[i - 1]))

    def describe(self):
        if not self.has_ts:
            return f"{self.fps:.2f} fps по заголовку видео"
        return (f"метки времени {self.csv_path}: кадров {len(self.ts)}, шаг драйвера {self.step}, "
                f"интервал {np.median(np.diff(self.ts)) * 1000:.2f} мс, разрывов {int((~self.linked_ok).sum())}, "
                f"в среднем {self.fps:.1f} кадр/с")


def read_frames(cap, idxs, fn=lambda f: f, max_skip=30):
    """Прочитать кадры с номерами idxs (по возрастанию), вернуть [fn(frame)] для прочитанных.
    Близкие кадры читаются подряд (grab), дальние - перемоткой: перемотка в FFV1/mp4
    стоит как декодирование ~20 кадров."""
    out, pos = [], None
    for j in sorted(int(i) for i in idxs):
        if pos is None or not 0 <= j - pos <= max_skip:
            cap.set(cv2.CAP_PROP_POS_FRAMES, j)
        else:
            for _ in range(j - pos):
                cap.grab()
        ok, f = cap.read()
        pos = j + 1
        if ok:
            out.append(fn(f))
    return out


def gray(f):
    return cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)


def norm_brightness(g):
    """Нормировка средней яркости к 100 (мерцание автоэкспозиции)."""
    return g * np.float32(100.0 / (g.mean() + 1e-3))


# ---------------------------------------------------------------- геометрия

def flow_affine(cx, cy, angle_deg, out_w, out_h):
    """Матрица для warpAffine(..., WARP_INVERSE_MAP): окно out_w x out_h с центром (cx, cy),
    ось x направлена против течения (течение идёт к -x), ось y - поперёк (e2)."""
    t = np.radians(angle_deg)
    e1 = np.array([np.cos(t), np.sin(t)])
    e2 = np.array([-e1[1], e1[0]])
    return np.array([[e1[0], e2[0], cx - e1[0] * out_w / 2 - e2[0] * out_h / 2],
                     [e1[1], e2[1], cy - e1[1] * out_w / 2 - e2[1] * out_h / 2]], np.float32)


# ---------------------------------------------------------------- пики

def parabola(y0, y1, y2):
    """Смещение вершины параболы через 3 равноотстоящие точки от средней, в шагах."""
    den = y0 - 2 * y1 + y2
    return 0.5 * (y0 - y2) / den if den != 0 else 0.0


def parabola_vertex(x, y):
    """Вершина параболы через 3 точки с произвольным шагом по x."""
    (x0, x1, x2), (y0, y1, y2) = x, y
    num = (x1 - x0) ** 2 * (y1 - y2) - (x1 - x2) ** 2 * (y1 - y0)
    den = (x1 - x0) * (y1 - y2) - (x1 - x2) * (y1 - y0)
    return x1 - 0.5 * num / den if den != 0 else x1


def mask_zero_shift(c, r):
    """Подавить пик циклической корреляции у нулевого сдвига (|s| <= r) по всем осям:
    это остатки неподвижного фона."""
    if r <= 0:
        return c
    c = c.copy()
    idx = tuple(np.r_[0:r + 1, n - r:n] for n in c.shape)
    c[np.ix_(*idx)] = c.min()
    return c


def wrap(s, n):
    return s - n if s > n / 2 else s


# ---------------------------------------------------------------- статистика

def window_mean(v):
    """Среднее после отбраковки грубых выбросов (TRIM_LO..TRIM_HI x медиана).
    Не медиана: при записи экрана между показанными кадрами проходит то 4, то 5
    кадров камеры, и медиана берёт только самую частую группу (занижение ~4%)."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan
    lo, hi = sorted((TRIM_LO * np.median(v), TRIM_HI * np.median(v)))
    k = v[(v >= lo) & (v <= hi)]
    return float(k.mean()) if k.size else np.nan


def robust_stats(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return dict(n=0, median=np.nan, mad=np.nan, p10=np.nan, p90=np.nan, cv=np.nan)
    med = float(np.median(v))
    mad = float(1.4826 * np.median(np.abs(v - med)))
    return dict(n=int(v.size), median=med, mad=mad,
                p10=float(np.percentile(v, 10)), p90=float(np.percentile(v, 90)),
                cv=mad / abs(med) if med else np.nan)


def consensus(vals):
    """Медиана методов, согласованных между собой (в пределах AGREE_TOL от общей медианы).
    Возвращает (консенсус, разброс согласованных в %, маска согласованных)."""
    v = np.asarray(vals, float)
    ok = np.isfinite(v)
    if ok.sum() < 2:
        return np.nan, np.nan, np.zeros_like(ok)
    c0 = np.median(v[ok])
    keep = ok & (np.abs(v - c0) <= AGREE_TOL * abs(c0)) if c0 else ok
    if keep.sum() < 2:
        return np.nan, np.nan, keep
    c = float(np.median(v[keep]))
    return c, float(np.ptp(v[keep]) / abs(c) * 100) if c else np.nan, keep


# ---------------------------------------------------------------- файлы

def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def write_json(path, obj):
    def clean(x):
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, (np.floating, float)):
            return None if not np.isfinite(x) else float(x)
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.bool_):
            return bool(x)
        return x
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean(obj), f, ensure_ascii=False, indent=1)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def pyplot():
    """matplotlib.pyplot в режиме без окна, или None, если matplotlib не установлен."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("matplotlib не установлен - график не построен")
        return None
