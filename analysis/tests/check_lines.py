#!/usr/bin/env python3
"""Готовит NDJSON от vision_series.py для vision_protocol_check.cpp: перед каждой строкой
ставит ожидаемые значения (скорость, уровень, их валидность), разобранные стандартным json."""
import json
import sys

for line in open(sys.argv[1], encoding="utf-8"):
    line = line.rstrip("\n")
    d = json.loads(line)                    # заодно проверка, что строка - корректный JSON
    if d["type"] != "report":
        print(f"0 0 0 0\t{line}")
        continue
    s, lv = d["speed_camera"], d["level"]
    print(f'{s.get("value", 0)} {lv.get("value", 0)} {int(s["valid"])} {int(lv["valid"])}\t{line}')
