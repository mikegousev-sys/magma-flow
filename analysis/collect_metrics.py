#!/usr/bin/env python3
"""
Сводка метрик по результатам flow_speed.py + flow_rate.py.

  python collect_metrics.py <папка с результатами> [<ещё папка> ...] --data <папка с записями> --out metrics

Каждый результат - папка с summary.txt, profile.csv, windows.csv, flow_rate.txt.
Имя папки: <запись> или <запись>_<начало, с> (анализ по отрезкам).
Для записи берутся meta.json (время начала, режим, частота) из <data>/<запись>/.
Выход: metrics.csv, metrics.md, metrics.png.
"""
import argparse
import csv
import datetime as dt
import glob
import json
import os
import re

import numpy as np


def num(pat, txt, cast=float, default=np.nan):
    m = re.search(pat, txt)
    return cast(m.group(1)) if m else default


def collect(res_dir, data_dir):
    name = os.path.basename(res_dir.rstrip("/"))
    m = re.match(r"(rec_\d{8}-\d{6}_[a-z0-9_]+?)(?:_(\d+))?$", name)
    # имя самой записи тоже может кончаться на _<число> (режим every_5 ...):
    # отрезком считаем суффикс, только если папки записи с полным именем нет
    if m and m.group(2) and not os.path.isdir(os.path.join(data_dir, name)):
        rec, seg = m.group(1), int(m.group(2))
    else:
        rec, seg = name, None
    meta_p = os.path.join(data_dir, rec, "meta.json")
    meta = json.load(open(meta_p)) if os.path.isfile(meta_p) else {}
    summ = open(os.path.join(res_dir, "summary.txt"), encoding="utf-8").read()
    fr_p = os.path.join(res_dir, "flow_rate.txt")
    fr = open(fr_p, encoding="utf-8").read() if os.path.isfile(fr_p) else ""

    start = meta.get("start")
    t0 = dt.datetime.strptime(start, "%Y-%m-%dT%H:%M:%S%z") + dt.timedelta(seconds=seg or 0) if start else None
    wins = list(csv.DictReader(open(os.path.join(res_dir, "windows.csv"))))
    cons = np.array([float(w["consensus_px_per_s"]) for w in wins], float)
    spread = np.array([float(w["spread_pct"]) for w in wins], float)
    good = np.isfinite(spread) & (spread <= 10)
    cg = cons[np.isfinite(cons)]                 # консенсус методов по всем окнам
    r = {
        "запись": rec[4:19],
        "отрезок_с": seg,
        "время": t0.strftime("%H:%M:%S") if t0 else "",
        "режим": meta.get("mode") or meta.get("rate_mode") or "",
        "кадр": f'{meta.get("width", "")}x{meta.get("height", "")}',
        "fps": meta.get("fps_nominal", ""),
        "пар_кадров": num(r"пар соседних кадров: (\d+)", summ, int, 0),
        "окон_всего": len(wins),
        "окон_согласных": int(good.sum()),
        "v_ось_м_с": num(r"Скорость поверхности на оси: ([\d.]+)", fr),
        "v_ядро_px_с": num(r"ядро потока\): ([\d.]+)", summ),
        "v_консенсус_px_с": float(np.median(cg)) if cg.size else np.nan,
        "v_разброс_окон_%": float(1.4826 * np.median(np.abs(cg - np.median(cg))) / np.median(cg) * 100)
        if cg.size >= 3 else np.nan,
        "расхождение_методов_%": num(r"Расхождение согласованных методов \(max-min\): [\d.]+ px/с \(([\d.]+)%\)", summ),
        "уровень_мм": num(r"Уровень \(от низшей точки профиля [^)]*\): ([\d.]+)", fr),
        "площадь_см2": num(r"Площадь живого сечения: ([\d.]+)", fr),
        "Q_мин_л_с": num(r"Расход при k = [\d.]+\.\.[\d.]+: ([\d.]+)\.\.", fr),
        "Q_макс_л_с": num(r"Расход при k = [\d.]+\.\.[\d.]+: [\d.]+\.\.([\d.]+) л/с", fr),
        "т_ч_мин": num(r"л/с = (\d+)\.\.", fr),
        "т_ч_макс": num(r"л/с = \d+\.\.(\d+) т/ч", fr),
        "край_P10": num(r"P10\.\.P90 ([-+]\d+)\.\.", fr),
        "край_P90": num(r"P10\.\.P90 [-+]\d+\.\.([-+]\d+)", fr),
        "ось_на_краю": "максимум на краю профиля" in fr,
        "видимая_полуширина_мм": num(r"видимый край на ([\d.]+) мм от оси", fr),
        "расхождение_методов_окна_%": float(np.nanmedian(spread)) if np.isfinite(spread).any() else np.nan,
    }
    flags = []
    if r["ось_на_краю"]:
        flags.append("ось вне профиля")
    if np.isfinite(r["край_P10"]) and r["край_P90"] - r["край_P10"] > 80:
        flags.append("рваный край")
    if np.isfinite(r["видимая_полуширина_мм"]) and r["видимая_полуширина_мм"] < 70:
        flags.append("ось не найдена (видимая полуширина < 70 мм)")
    if np.isfinite(r["расхождение_методов_окна_%"]) and r["расхождение_методов_окна_%"] > 20:
        flags.append("методы расходятся >20%")
    if not fr:
        flags.append("нет расчёта уровня")
    elif not np.isfinite(r["уровень_мм"]):
        flags.append("уровень не определён")
    r["оценка"] = "надёжно" if not flags else "; ".join(flags)
    r["_t"] = t0
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="metrics")
    ap.add_argument("--seg-len", type=float, default=30.0, help="длина отрезка анализа, с (как --max-seconds)")
    a = ap.parse_args()
    res = []
    for d in a.dirs:
        for p in sorted(glob.glob(d)) if any(c in d for c in "*?[") else [d]:
            if os.path.isfile(os.path.join(p, "summary.txt")):
                res.append(collect(p, a.data))
    if not res:
        raise SystemExit("Не найдено ни одной папки с summary.txt")
    res.sort(key=lambda r: (r["_t"] is None, r["_t"]))
    os.makedirs(a.out, exist_ok=True)
    keys = [k for k in res[0] if not k.startswith("_")]
    with open(os.path.join(a.out, "metrics.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for r in res:
            w.writerow([f"{r[k]:.3g}" if isinstance(r[k], float) else r[k] for k in keys])

    def f(v, fmt):
        return fmt.format(v) if isinstance(v, (int, float)) and np.isfinite(v) else "—"
    lines = ["| время | запись | отрезок | v на оси, м/с | разброс v, % | уровень, мм | площадь, см² | расход, л/с | т/ч | оценка |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in res:
        seg = "" if r["отрезок_с"] is None else f"{r['отрезок_с']}–{r['отрезок_с'] + a.seg_len:g} с"
        lines.append(f"| {r['время']} | {r['запись']} | {seg} | {f(r['v_ось_м_с'], '{:.2f}')} | "
                     f"{f(r['v_разброс_окон_%'], '{:.1f}')} | {f(r['уровень_мм'], '{:.0f}')} | "
                     f"{f(r['площадь_см2'], '{:.0f}')} | {f(r['Q_мин_л_с'], '{:.0f}')}–{f(r['Q_макс_л_с'], '{:.0f}')} | "
                     f"{f(r['т_ч_мин'], '{:.0f}')}–{f(r['т_ч_макс'], '{:.0f}')} | {r['оценка']} |")
    open(os.path.join(a.out, "metrics.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    ok = [r for r in res if r["_t"] is not None]
    if not ok:
        print("Нет времени начала записей (meta.json) - график не построен")
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t = [r["_t"] for r in ok]
        good = np.array([r["оценка"] == "надёжно" for r in ok], bool)
        fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
        for i, (key, lab) in enumerate((("v_ось_м_с", "скорость на оси, м/с"), ("уровень_мм", "уровень, мм"))):
            v = np.array([r[key] for r in ok], float)
            ax[i].plot(np.array(t)[good], v[good], "o", color="#2a6fdb", label="надёжно")
            ax[i].plot(np.array(t)[~good], v[~good], "x", color="#999999", label="с оговорками")
            ax[i].set_ylabel(lab)
            ax[i].grid(alpha=0.3)
        ax[0].legend()
        qa = np.array([r["т_ч_мин"] for r in ok], float)
        qb = np.array([r["т_ч_макс"] for r in ok], float)
        for tt, lo, hi, g in zip(t, qa, qb, good):
            ax[2].plot([tt, tt], [lo, hi], "-", lw=4, color="#2a6fdb" if g else "#bbbbbb")
        ax[2].set_ylabel("расход, т/ч (k 0.67–0.85)")
        ax[2].grid(alpha=0.3)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "metrics.png"), dpi=120)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
