#!/usr/bin/env python3
"""
Поточный расчёт скорости и уровня по СЕРИЯМ кадров, совместимый с C++ magma_vision.

Совместимость с code/:
  - настройки: тот же формат и те же ключи, что code/magma_vision.conf (schedule.*, flow.strip.*,
    calibration.*, flow.*, geometry.*, output.*); свои параметры - series.* (C++ их игнорирует).
    Файл перечитывается на ходу;
  - вывод: NDJSON-строки того же вида, что FormatVisionReport/FormatVisionWarning
    (magma_vision_protocol.hpp): те же поля, порядок и форматы чисел, так что строки разбирает
    VisionLineParser и читает magma_vision_client. Подробности (скорость на оси, профиль по
    полосам, методы, расход) - в дополнительных блоках speed_detail и flow_rate в конце строки;
  - рассылка: TCP output.host:output.port, по строке на отчёт (как LineBroadcaster).

На каждую серию (schedule.burst_frames соседних кадров, раз в schedule.burst_interval_s):
  kymo по series.bands полосам -> профиль, ось, скорость на оси; kymo по всей полосе;
  phase + LK на series.check_pairs парах; раз в series.piv_every серий - PIV.
Раз в series.level_every_s: край корки по медианному фону окна поперёк потока + ось ->
  полуширина поверхности -> уровень по профилю жёлоба -> площадь и расход.

Сейчас источник кадров - запись focus_preview (video.mkv + frames.csv), режим воспроизведения;
на Pi тот же SeriesProcessor получает кадры прямо с камеры.

  python vision_series.py <video.mkv> [--conf magma_vision_series.conf] [--realtime] [--serve]
"""
import argparse
import datetime as dt
import json
import os
import socket
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np

from common import Timing, consensus, flow_affine, gray, read_frames, read_json, robust_stats, write_json
from flow_rate import crust_edges, gutter, level_from_half_width, section, surface_flow
from flow_speed import PairMethods, to_u8
from series_flow import axis_from_profile, band_speeds, lk_pair, make_prep_scaled, phase_pair, series_index

HERE = os.path.dirname(os.path.abspath(__file__))


TZ = None      # часовой пояс меток времени; при воспроизведении - пояс записи (из meta.json)


def iso(ts):
    """Метка времени как VisionIsoTimestamp(): 2026-09-10T14:03:11.204 (местное время)."""
    d = dt.datetime.fromtimestamp(ts, TZ)
    return d.strftime("%Y-%m-%dT%H:%M:%S") + f".{d.microsecond // 1000:03d}"


# ---------------------------------------------------------------- настройки

class Conf:
    """«ключ = значение», '#' - комментарий, повтор ключа = список (как magma_config.hpp)."""

    def __init__(self, path):
        self.path, self.mtime, self.values, self.version = path, None, {}, ""
        self.load()

    def load(self):
        vals = {}
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                vals.setdefault(k.strip(), []).append(v.split(" #")[0].strip())
        self.values, self.mtime, self.version = vals, os.path.getmtime(self.path), iso(time.time())

    def reload_if_changed(self):
        try:
            if os.path.getmtime(self.path) != self.mtime:
                self.load()
                return True
        except OSError:
            pass
        return False

    def get(self, key, default, cast=float):
        v = self.values.get(key)
        return cast(v[-1]) if v and v[-1] != "" else default


# ---------------------------------------------------------------- вывод

def _num(v, fmt):
    return fmt % (v if v is not None and np.isfinite(v) else 0.0)


def _clean(x):
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_clean(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else round(float(x), 4)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def format_report(r, ts):
    """Строка как FormatVisionReport(); r - словарь с полями VisionReport + доп. блоки."""
    b = lambda v: "true" if v else "false"                                   # noqa: E731
    j = f'{{"type":"report","ts":"{iso(ts)}",'
    j += f'"speed_camera":{{"valid":{b(r["speed_valid"])}'
    if r["speed_valid"]:
        j += "," + '"value":' + _num(r["speed_ms"], "%.4f") + ',"unit":"m/s"'
    j += ',"snr":' + _num(r["speed_snr"], "%.2f")
    j += f',"profiles":{r["speed_profiles"]},"intervals":{r["speed_intervals"]}'
    j += ',"shift_px_per_frame":' + _num(r["speed_shift_px_per_frame"], "%.2f") + "},"
    j += f'"level":{{"valid":{b(r["level_valid"])}'
    if r["level_valid"]:
        j += ',"value":' + _num(r["level_mm"], "%.2f") + ',"unit":"mm"'
    j += ',"sigma":' + _num(r["level_sigma_mm"], "%.3f") + f',"state":"{r["level_state"]}"'
    j += ',"disagreement":' + _num(r["level_disagreement_mm"], "%.3f")
    j += ',"sources":[' + ",".join(f'"{s}"' for s in r["level_sources"]) + f'],"frames":{r["level_frames"]}}},'
    j += f'"camera":{{"frames":{r["camera_frames"]},"fps":' + _num(r["camera_fps"], "%.1f")
    j += ',"exposure_us":' + _num(r["camera_exposure_us"], "%.0f") + ',"gain_db":' + _num(r["camera_gain_db"], "%.1f")
    j += f',"timeouts":{r["camera_timeouts"]},"errors":{r["camera_errors"]}}},'
    j += f'"calibration":{{"valid":{b(r["calibration_valid"])},"beta_deg":' + _num(r["calibration_beta_deg"], "%.2f")
    j += ',"scale_mm_per_px":' + _num(r["calibration_scale_mm_per_px"], "%.4f")
    j += ',"residual_px":' + _num(r["calibration_residual_px"], "%.3f")
    j += f',"marks":{r["calibration_marks"]},"manual":{b(r["calibration_manual"])}}},'
    for key in ("speed_detail", "flow_rate"):                             # доп. блоки - после стандартных
        j += f'"{key}":' + json.dumps(_clean(r[key]), ensure_ascii=False, separators=(",", ":")) + ","
    j += f'"config_version":"{r["config_version"]}"}}'
    return j


def format_warning(text, ts):
    return json.dumps({"type": "warning", "ts": iso(ts), "text": text}, ensure_ascii=False, separators=(",", ":"))


class Broadcaster:
    """TCP-рассылка строк всем подключённым клиентам (как LineBroadcaster)."""

    def __init__(self, host, port):
        self.clients, self.lock = [], threading.Lock()
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((host, port))
        self.server.listen(8)
        print(f"[BROADCAST] Слушаю {host}:{port}", file=sys.stderr)
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while True:
            c, peer = self.server.accept()
            print(f"[BROADCAST] Клиент подключён: {peer[0]}", file=sys.stderr)
            with self.lock:
                self.clients.append(c)

    def broadcast(self, line):
        data = (line + "\n").encode()
        with self.lock:
            for c in list(self.clients):
                try:
                    c.sendall(data)
                except OSError:
                    self.clients.remove(c)


# ---------------------------------------------------------------- расчёт

class SeriesProcessor:
    """Всё состояние поточного расчёта: геометрия, фон, история профилей, последний уровень."""

    def __init__(self, conf, frame_shape):
        self.frame_shape = frame_shape
        self.apply(conf)

    def apply(self, conf):
        c = conf.get
        self.conf = conf
        s = self.s = c("series.scale", 0.5)
        cx, cy = c("flow.strip.center_x_px", 640.0), c("flow.strip.center_y_px", 512.0)
        L, Wd = c("flow.strip.length_px", 900.0), c("flow.strip.average_px", 60.0)
        self.angle = c("flow.strip.angle_deg", 0.0) + 180.0              # наша система: ось x против течения
        self.Wd, self.Hl = Wd, c("series.level_height_px", 800.0)
        self.prep = make_prep_scaled((cx - L / 2, cy - Wd / 2, L, Wd), self.angle, s)
        # окно поиска края корки: та же ось вдоль потока, выше поперёк; яркость без нормировки
        Al = flow_affine(cx, cy, self.angle, L, self.Hl)
        Al[:, :2] /= s
        self.size_l = (int(round(L * s)), int(round(self.Hl * s)))
        self.prep_l = lambda f: cv2.warpAffine(gray(f), Al, self.size_l,
                                               flags=cv2.INTER_AREA | cv2.WARP_INVERSE_MAP).astype(np.float32)
        self.valid_l = cv2.warpAffine(np.ones(self.frame_shape, np.uint8), Al, self.size_l,
                                      flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP) > 0
        self.bands = int(c("series.bands", 8))
        self.check_pairs, self.piv_every = int(c("series.check_pairs", 1)), int(c("series.piv_every", 10))
        self.bg_step, self.min_shift = c("series.bg_step", 0.5), c("series.min_shift", 1.0)
        self.mm = c("calibration.scale_mm_per_px", 0.4135)
        self.mm_across = c("series.scale_across_mm_per_px", self.mm)
        self.vmin, self.vmax = c("flow.min_speed_ms", 0.4), c("flow.max_speed_ms", 5.0)
        self.min_snr, self.snr_excl = c("flow.min_snr", 4.0), int(c("flow.snr_exclusion", 8))
        self.agree_pct = c("series.agree_pct", 10.0)
        self.budget_ms = c("schedule.burst_interval_s", 0.2) * 1000
        self.level_every = c("series.level_every_s", 1.0)
        self.crust, self.side = c("series.crust_mm", 25.0), int(c("series.profile_side", 1))
        self.k = [float(v) for v in c("series.k", "0.67,0.85", str).split(",")]
        self.rho = c("series.rho", 3400.0)
        prof = c("series.gutter_profile", "gutter_profile_R150.csv", str)
        self.P = gutter(prof if os.path.isabs(prof) else os.path.join(HERE, prof))
        self.level_window = c("series.level_window_s", 5.0)
        self.level_median_n = int(c("series.level_median_n", 5))
        self.bg = self.bg_l = self.scale = None
        self.warm, self.warm_l, self.n = [], [], 0
        # профили полос за последние level_window_s: у оси профиль почти плоский, и максимум
        # за одну секунду прыгает на десятки пикселей - ось для уровня берётся по длинному окну
        self.history = deque()
        self.levels = deque(maxlen=self.level_median_n)          # последние пересчёты уровня
        self.level = dict(state="unknown", valid=False)
        self.t_level = None

    def warming(self):
        return self.bg is None

    def process(self, frames, ts):
        """Одна серия: кадры (BGR) и их времена. Возвращает (отчёт или None во время прогрева, мс)."""
        t0 = time.perf_counter()
        g = np.stack([self.prep(f) for f in frames])
        mid = self.prep_l(frames[len(frames) // 2])
        if self.bg is None:                                       # прогрев: копим кадры для медианы фона
            self.warm.append(g[0])
            self.warm_l.append(mid)
            if len(self.warm) >= int(self.conf.get("series.bg_init", 20)):
                self.bg = np.median(np.array(self.warm), axis=0).astype(np.float32)
                self.bg_l = np.median(np.array(self.warm_l), axis=0).astype(np.float32)
                self.warm, self.warm_l = [], []
            return None, (time.perf_counter() - t0) * 1e3
        self.bg += self.bg_step * np.sign(g - self.bg).mean(axis=0)
        self.bg_l += self.bg_step * np.sign(mid - self.bg_l)
        hp = g - self.bg
        k_px = 1.0 / self.s
        dt_ = float(np.median(np.diff(ts)))
        v_band, v_all, centers, snr, n_lags, tex = band_speeds(hp, self.bands, dt_ / k_px, self.min_shift,
                                                               snr_exclusion=self.snr_excl)
        centers = centers * k_px
        y_ax, v_ax, on_edge = axis_from_profile(centers, v_band)
        if self.scale is None:
            self.scale = 40.0 / (hp[0].std() + 1e-3)
        ph, lk = [], []
        for j in (np.linspace(1, len(frames) - 1, self.check_pairs + 2)[1:-1].astype(int) if self.check_pairs else []):
            ddt = (ts[j] - ts[j - 1]) / k_px
            ph.append(-phase_pair(hp[j - 1], hp[j], self.win(hp), self.min_shift) / ddt)
            lk.append(-lk_pair(to_u8(hp[j - 1], self.scale), to_u8(hp[j], self.scale)) / ddt)
        v_piv = np.nan
        if self.piv_every and self.n % self.piv_every == 0:
            j = len(frames) // 2
            ddt = (ts[j] - ts[j - 1]) / k_px
            hs, ws = hp.shape[1:]
            pm = PairMethods(ws, hs, ws / 3, self.min_shift, tile=32)
            p, c = pm.frame(hp[j - 1], to_u8(hp[j - 1], self.scale)), pm.frame(hp[j], to_u8(hp[j], self.scale))
            v_piv = -pm.piv(p, c, pred=(-v_all * ddt, 0.0) if np.isfinite(v_all) else None)[0] / ddt
        self.n += 1
        ms = lambda v: v * self.mm / 1000.0                                      # noqa: E731  px/с -> м/с
        v_ph = float(np.nanmedian(ph)) if np.isfinite(ph).any() else np.nan
        v_lk = float(np.nanmedian(lk)) if np.isfinite(lk).any() else np.nan
        cons, spread, keep = consensus([v_all, v_ph, v_lk, v_piv])
        speed = ms(cons)
        speed_valid = bool(np.isfinite(speed) and self.vmin <= speed <= self.vmax
                           and np.isfinite(snr) and snr >= self.min_snr)
        self.history.append((ts[0], v_band, centers))
        while self.history and ts[0] - self.history[0][0] > self.level_window:
            self.history.popleft()
        if self.t_level is None or ts[0] - self.t_level >= self.level_every:
            self.update_level(speed_valid)
            self.t_level = ts[0]
        lvl = self.level
        comp_ms = (time.perf_counter() - t0) * 1e3
        report = dict(
            speed_valid=speed_valid, speed_ms=speed, speed_snr=snr, speed_profiles=len(frames),
            speed_intervals=n_lags, speed_shift_px_per_frame=v_all * dt_ if np.isfinite(v_all) else np.nan,
            level_valid=lvl["valid"], level_mm=lvl.get("level_mm"), level_sigma_mm=lvl.get("sigma_mm"),
            level_state=lvl["state"], level_disagreement_mm=0.0,
            level_sources=["axis_kymo", "crust_edge"] if lvl["valid"] else [],
            level_frames=lvl.get("frames", 0),
            speed_detail=dict(axis_ms=ms(v_ax), axis_y_px=y_ax, axis_on_edge=bool(on_edge),
                              bands_ms=[ms(v) for v in v_band], band_y_px=centers,
                              methods_ms=dict(kymo=ms(v_all), phase=ms(v_ph), lk=ms(v_lk), piv=ms(v_piv)),
                              spread_pct=spread, agree=bool(np.isfinite(spread) and spread <= self.agree_pct),
                              texture=tex, compute_ms=comp_ms, budget_ms=self.budget_ms),
            flow_rate={k: lvl.get(k) for k in ("level_mm", "b_mm", "edge_ragged_px", "area_cm2", "q_l_s",
                                               "t_h", "k", "rho")} | dict(valid=lvl["valid"]))
        return report, comp_ms

    def win(self, hp):
        if getattr(self, "_win", None) is None or self._win.shape != hp.shape[1:]:
            self._win = cv2.createHanningWindow(hp.shape[1:][::-1], cv2.CV_32F)
        return self._win

    def update_level(self, speed_valid):
        """Уровень по накопленным профилям полос и медианному фону окна поперёк потока."""
        V = np.array([h[1] for h in self.history])
        centers = self.history[-1][2]
        with np.errstate(all="ignore"):
            vb = np.nanmedian(V, axis=0) if np.isfinite(V).any() else V[0]
        y_ax, v_ax, on_edge = axis_from_profile(centers, vb)
        fin = np.isfinite(vb)
        L = dict(valid=False, state="unknown", frames=len(V))
        if not speed_valid or fin.sum() < 3:
            L["state"] = "empty" if not speed_valid else "unknown"
            self.level = L
            return
        sign = -1.0 if centers[fin][np.argmin(vb[fin])] < y_ax else 1.0
        y_c = (self.Hl / 2 + (y_ax - self.Wd / 2)) * self.s             # ось в окне края (уменьшенном)
        edges = crust_edges(self.bg_l, self.valid_l, int(round(y_c)), sign, step=2)
        if len(edges) < 10:
            self.level = L
            return
        d = abs(y_c - np.median(edges)) / self.s
        ragged = (np.percentile(edges, 90) - np.percentile(edges, 10)) / self.s
        b = d * self.mm_across + self.crust
        z = level_from_half_width(self.P, b, self.side)
        if not np.isfinite(z):
            L["state"] = "unreliable"
            self.level = L
            return
        db = ragged / 2.56 * self.mm_across                           # P10..P90 -> 1 сигма
        zs = [level_from_half_width(self.P, b + e, self.side) for e in (-db, db)]
        sigma = abs(zs[1] - zs[0]) / 2 if all(np.isfinite(zs)) else np.nan
        xs, dep, area, bL, bR = section(self.P, z)
        r = np.abs(centers[fin] - y_ax) * self.mm_across
        o = np.argsort(r)
        q = surface_flow(xs, dep, r[o], vb[fin][o] * self.mm / 1000.0, bL, bR)
        L.update(valid=not on_edge, state="unreliable" if on_edge else "measured", level_mm=z, sigma_mm=sigma,
                 b_mm=b, edge_ragged_px=ragged, area_cm2=area * 1e4, k=self.k, rho=self.rho,
                 q_l_s=[q * k * 1000 for k in self.k], t_h=[q * k * self.rho * 3.6 for k in self.k])
        # в отчёт - медиана последних пересчётов (как CombineLevelObservations в C++: медиана,
        # остальные поля - от пересчёта, ближайшего к ней)
        self.levels.append(L)
        ok = [x for x in self.levels if x["valid"]]
        if ok:
            med = float(np.median([x["level_mm"] for x in ok]))
            L = dict(min(ok, key=lambda x: abs(x["level_mm"] - med)), level_mm=med, frames=len(V))
        self.level = L


# ---------------------------------------------------------------- воспроизведение записи

def strip_from_result(conf, speed_dir):
    """Подменить полосу flow.strip.* на ROI и угол из результата flow_speed (для записей,
    где камера стояла иначе, чем в конфиге)."""
    sp = read_json(os.path.join(speed_dir, "result.json")) if os.path.isfile(os.path.join(speed_dir, "result.json")) \
        else __import__("flow_rate").read_speed_result(speed_dir)
    x, y, w, h = sp["roi"]
    a = (sp["angle"] + 180.0 + 180.0) % 360.0 - 180.0
    for k, v in (("center_x_px", x + w / 2), ("center_y_px", y + h / 2), ("angle_deg", a),
                 ("length_px", w), ("average_px", h)):
        conf.values[f"flow.strip.{k}"] = [f"{v:.2f}"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--conf", default=os.path.join(HERE, "magma_vision_series.conf"))
    ap.add_argument("--geometry-from", help="папка результата flow_speed: взять полосу оттуда")
    ap.add_argument("--realtime", action="store_true", help="выдавать серии в темпе записи")
    ap.add_argument("--serve", action="store_true", help="рассылать строки по TCP (output.host:port)")
    ap.add_argument("--out", help="файл NDJSON")
    ap.add_argument("--max-series", type=int, default=0)
    a = ap.parse_args()

    conf = Conf(a.conf)
    if a.geometry_from:
        strip_from_result(conf, a.geometry_from)
    tm = Timing(a.video)
    n = int(conf.get("schedule.burst_frames", 10))
    series = series_index(tm, n, conf.get("schedule.burst_interval_s", 0.2))
    if a.max_series:
        series = series[:a.max_series]
    meta_p = os.path.join(os.path.dirname(os.path.abspath(a.video)), "meta.json")
    meta = read_json(meta_p) if os.path.isfile(meta_p) else {}
    global TZ
    t_start = time.time()
    if meta.get("start"):
        d0 = dt.datetime.strptime(meta["start"], "%Y-%m-%dT%H:%M:%S%z")
        t_start, TZ = d0.timestamp(), d0.tzinfo
    cam = {}
    if tm.csv_path:
        import csv
        with open(tm.csv_path, newline="") as f:
            cam = {i: r for i, r in enumerate(csv.DictReader(f))}
    cap = cv2.VideoCapture(a.video)
    first = read_frames(cap, [series[0][0]])[0]
    proc = SeriesProcessor(conf, first.shape[:2])
    bc = Broadcaster(conf.get("output.host", "0.0.0.0", str), int(conf.get("output.port", 9101))) if a.serve else None
    print_json = conf.get("output.print_json", "true", str).lower() == "true"
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    out = open(a.out, "w", encoding="utf-8") if a.out else None
    timings, reports, last_warn = [], [], -1e9
    wall0, rec0 = time.time(), tm.time(series[0][0])

    def emit(line):
        if print_json:
            print(line, flush=True)
        if out:
            out.write(line + "\n")
        if bc:
            bc.broadcast(line)

    for s in series:
        if conf.reload_if_changed():
            if a.geometry_from:
                strip_from_result(conf, a.geometry_from)
            proc.apply(conf)
            emit(format_warning("Файл настроек изменился - применён", time.time()))
        frames = read_frames(cap, s)
        if len(frames) < n:
            continue
        ts = np.array([tm.time(i) for i in s])
        if a.realtime:
            time.sleep(max(0.0, (ts[-1] - rec0) - (time.time() - wall0)))
        report, ms = proc.process(frames, ts)
        timings.append(ms)
        if report is None:
            continue
        r0 = cam.get(s[0], {})
        report.update(camera_frames=len(frames), camera_fps=(len(ts) - 1) / (ts[-1] - ts[0]),
                      camera_exposure_us=float(r0.get("exposure_us", 0) or 0),
                      camera_gain_db=float(r0.get("gain_db", 0) or 0), camera_timeouts=0, camera_errors=0,
                      calibration_valid=True, calibration_beta_deg=conf.get("calibration.beta_deg", 0.0),
                      calibration_scale_mm_per_px=proc.mm, calibration_residual_px=0.0, calibration_marks=0,
                      calibration_manual=True, config_version=conf.version)
        t_abs = t_start + ts[0]
        emit(format_report(report, t_abs))
        reports.append(report)
        if ms > proc.budget_ms and t_abs - last_warn > 10:
            emit(format_warning(f"Расчёт серии {ms:.0f} мс - дольше бюджета {proc.budget_ms:.0f} мс", t_abs))
            last_warn = t_abs
    cap.release()
    if out:
        out.close()

    T = np.array(timings[int(conf.get("series.bg_init", 20)):])
    sp = [r["speed_ms"] for r in reports if r["speed_valid"]]
    ax = [r["speed_detail"]["axis_ms"] for r in reports if r["speed_valid"]]
    lv = [r["level_mm"] for r in reports if r["level_valid"]]
    th = [np.mean(r["flow_rate"]["t_h"]) for r in reports if r["level_valid"]]
    summ = dict(series=len(series), reports=len(reports), speed_valid=len(sp),
                compute_ms=dict(median=float(np.median(T)), p95=float(np.percentile(T, 95)), max=float(T.max()),
                                over_budget=int((T > proc.budget_ms).sum()), budget=proc.budget_ms),
                speed_ms=robust_stats(sp), axis_ms=robust_stats(ax), level_mm=robust_stats(lv), t_h=robust_stats(th),
                agree=int(sum(r["speed_detail"]["agree"] for r in reports)))
    print(json.dumps(_clean(summ), ensure_ascii=False), file=sys.stderr)
    if a.out:
        write_json(os.path.splitext(a.out)[0] + "_summary.json", summ)


if __name__ == "__main__":
    main()
