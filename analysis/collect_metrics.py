#!/usr/bin/env python3
"""
Сводка метрик по результатам flow_speed.py + flow_rate.py.

  python collect_metrics.py <папка результата> [<ещё папка или glob> ...] --out metrics

Каждый результат - папка с flow_rate.json (flow_rate.py) и windows.csv (flow_speed.py).
Время начала записи и режим берутся из meta.json рядом с видео (focus_preview).
Выход: metrics.csv, metrics.md, metrics.png.
"""
import argparse
import csv
import datetime as dt
import glob
import os

import numpy as np

from common import GOOD_SPREAD_PCT, pyplot, read_json, robust_stats, write_csv

V_AXIS_MIN_HALF_MM = 70      # видимая полуширина меньше - ось потока, скорее всего, не найдена
RAGGED_EDGE_PX = 80          # разброс края корки по длине (P90-P10) больше - "рваный край"
METHODS_BAD_PCT = 20         # медианное расхождение методов по окнам больше - "методы расходятся"


def f(x):
    return np.nan if x is None else x


def collect(res_dir):
    fr = read_json(os.path.join(res_dir, "flow_rate.json"))
    meta_p = os.path.join(os.path.dirname(fr["video"]), "meta.json")
    meta = read_json(meta_p) if os.path.isfile(meta_p) else {}
    t0 = (dt.datetime.strptime(meta["start"], "%Y-%m-%dT%H:%M:%S%z") + dt.timedelta(seconds=fr["start"] or 0)
          if meta.get("start") else None)
    with open(os.path.join(res_dir, "windows.csv"), newline="") as fh:
        wins = list(csv.DictReader(fh))
    cons = np.array([float(w["consensus_px_per_s"]) for w in wins])
    spread = np.array([float(w["spread_pct"]) for w in wins])
    rec = os.path.basename(os.path.dirname(fr["video"]))
    r = {
        "запись": rec[4:19] if rec.startswith("rec_") else rec,
        "отрезок": f'{fr["start"]:g}–{fr["start"] + fr["seconds"]:g} с' if fr.get("seconds") else "",
        "время": t0.strftime("%H:%M:%S") if t0 else "",
        "режим": meta.get("mode") or meta.get("rate_mode") or "",
        "кадр": f'{meta.get("width", "")}x{meta.get("height", "")}',
        "fps": meta.get("fps_nominal", ""),
        "пар_кадров": fr.get("pairs"),
        "окон_всего": len(wins),
        "окон_согласных": int((spread <= GOOD_SPREAD_PCT).sum()),
        "v_ось_м_с": f(fr.get("v_axis_m_s")),
        "v_ядро_px_с": f(fr.get("core_px_s")),
        "v_консенсус_px_с": robust_stats(cons)["median"],
        "v_разброс_окон_%": robust_stats(cons)["cv"] * 100 if np.isfinite(cons).sum() >= 3 else np.nan,
        "расхождение_методов_%": f(fr.get("methods_spread_pct")),
        "расхождение_методов_окна_%": float(np.nanmedian(spread)) if np.isfinite(spread).any() else np.nan,
        "уровень_мм": f(fr.get("level_mm")),
        "площадь_см2": f(fr.get("area_cm2")),
        "Q_мин_л_с": f((fr.get("q_l_s") or [None])[0]),
        "Q_макс_л_с": f((fr.get("q_l_s") or [None, None])[1]),
        "т_ч_мин": f((fr.get("t_h") or [None])[0]),
        "т_ч_макс": f((fr.get("t_h") or [None, None])[1]),
        "край_P10": fr["edge_p10"],
        "край_P90": fr["edge_p90"],
        "ось_на_краю": fr["axis_on_edge"],
        "видимая_полуширина_мм": fr["visible_half_mm"],
    }
    flags = [msg for cond, msg in (
        (r["ось_на_краю"], "ось вне профиля"),
        (r["край_P90"] - r["край_P10"] > RAGGED_EDGE_PX, "рваный край"),
        (r["видимая_полуширина_мм"] < V_AXIS_MIN_HALF_MM, f"ось не найдена (видимая полуширина < {V_AXIS_MIN_HALF_MM} мм)"),
        (r["расхождение_методов_окна_%"] > METHODS_BAD_PCT, f"методы расходятся >{METHODS_BAD_PCT}%"),
        (not np.isfinite(r["уровень_мм"]), "уровень не определён")) if cond]
    r["оценка"] = "; ".join(flags) or "надёжно"
    r["_t"] = t0
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default="metrics")
    a = ap.parse_args()
    dirs = [p for d in a.dirs for p in (sorted(glob.glob(d)) if any(c in d for c in "*?[") else [d])]
    res = [collect(p) for p in dirs if os.path.isfile(os.path.join(p, "flow_rate.json"))]
    if not res:
        raise SystemExit("Не найдено ни одной папки с flow_rate.json")
    res.sort(key=lambda r: (r["_t"] is None, r["_t"]))
    os.makedirs(a.out, exist_ok=True)
    keys = [k for k in res[0] if not k.startswith("_")]
    write_csv(os.path.join(a.out, "metrics.csv"), keys,
              [[f"{r[k]:.3g}" if isinstance(r[k], float) else r[k] for k in keys] for r in res])

    def fmt(v, spec):
        return format(v, spec) if isinstance(v, (int, float)) and np.isfinite(v) else "—"
    lines = ["| время | запись | отрезок | v на оси, м/с | разброс v, % | уровень, мм | площадь, см² | расход, л/с | т/ч | оценка |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r['время']} | {r['запись']} | {r['отрезок']} | {fmt(r['v_ось_м_с'], '.2f')} | "
              f"{fmt(r['v_разброс_окон_%'], '.1f')} | {fmt(r['уровень_мм'], '.0f')} | {fmt(r['площадь_см2'], '.0f')} | "
              f"{fmt(r['Q_мин_л_с'], '.0f')}–{fmt(r['Q_макс_л_с'], '.0f')} | "
              f"{fmt(r['т_ч_мин'], '.0f')}–{fmt(r['т_ч_макс'], '.0f')} | {r['оценка']} |" for r in res]
    with open(os.path.join(a.out, "metrics.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    ok = [r for r in res if r["_t"] is not None]
    plt = pyplot() if ok else None
    if not ok:
        print("Нет времени начала записей (meta.json) - график не построен")
    if not plt:
        return
    t = np.array([r["_t"] for r in ok])
    good = np.array([r["оценка"] == "надёжно" for r in ok], bool)
    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for q, (key, lab) in zip(ax, (("v_ось_м_с", "скорость на оси, м/с"), ("уровень_мм", "уровень, мм"))):
        v = np.array([r[key] for r in ok], float)
        q.plot(t[good], v[good], "o", color="#2a6fdb", label="надёжно")
        q.plot(t[~good], v[~good], "x", color="#999999", label="с оговорками")
        q.set_ylabel(lab)
    ax[0].legend()
    for r, g in zip(ok, good):
        ax[2].plot([r["_t"]] * 2, [r["т_ч_мин"], r["т_ч_макс"]], "-", lw=4, color="#2a6fdb" if g else "#bbbbbb")
    ax[2].set_ylabel("расход, т/ч (диапазон k)")
    for q in ax:
        q.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "metrics.png"), dpi=120)


if __name__ == "__main__":
    main()
