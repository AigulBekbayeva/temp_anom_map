#!/usr/bin/env python3
"""
Готовит данные для интерактивной карты на GitHub Pages.

Читает кэш, созданный july_anomaly_openmeteo.py (./data/om_*.npz),
считает аномалии относительно нормы и пишет docs/data/anomalies.json.

Шаг сетки, месяц и модель определяются из имён файлов кэша автоматически,
так что менять константы в двух местах не нужно. Если в кэше лежит
несколько вариантов (например, 1.0° и 0.5°), берётся тот, где больше лет;
явно задать нужный можно константами ниже.

Запуск из корня репозитория:
    python src/export_web.py
"""

from __future__ import annotations

import json
import re
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

# ── границы региона: должны совпадать с july_anomaly_openmeteo.py ─────────
LON_MIN, LON_MAX = 46.0, 88.0
LAT_MIN, LAT_MAX = 35.0, 56.0

BASE_START, BASE_END = 1991, 2020

# None = определить из имён файлов кэша
GRID_STEP: float | None = None
MONTH: int | None = None
MODEL: str | None = None

DATADIR = Path("data")
OUT = Path("docs/data/anomalies.json")
DECIMALS = 1
# ──────────────────────────────────────────────────────────────────────────

NAME_RE = re.compile(r"^om_(?P<model>.+?)_m(?P<month>\d{2})_(?P<year>\d{4})_(?P<step>[\d.]+)deg\.npz$")


def find_cache() -> tuple[str, int, float, dict[int, Path]]:
    groups: dict[tuple[str, int, float], dict[int, Path]] = defaultdict(dict)
    for p in sorted(DATADIR.glob("om_*.npz")):
        m = NAME_RE.match(p.name)
        if not m:
            continue
        key = (m["model"], int(m["month"]), float(m["step"]))
        groups[key][int(m["year"])] = p

    if MODEL or MONTH or GRID_STEP:
        groups = {
            k: v for k, v in groups.items()
            if (MODEL is None or k[0] == MODEL)
            and (MONTH is None or k[1] == MONTH)
            and (GRID_STEP is None or abs(k[2] - GRID_STEP) < 1e-9)
        }

    if not groups:
        raise SystemExit(
            f"Не найден кэш в {DATADIR}/ (ожидаются файлы вида om_era5_m07_2026_1.0deg.npz). "
            "Сначала запустите src/july_anomaly_openmeteo.py"
        )

    if len(groups) > 1:
        print("В кэше несколько наборов:")
        for (mo, mm, st), files in groups.items():
            print(f"  {mo}, месяц {mm:02d}, шаг {st}° — {len(files)} лет")

    (model, month, step), files = max(groups.items(), key=lambda kv: len(kv[1]))
    return model, month, step, files


def main() -> None:
    model, month, step, files = find_cache()
    years = sorted(files)
    temps = np.stack([np.load(files[y])["t"] for y in years])
    print(f"выбрано: {model}, месяц {month:02d}, шаг {step}° — "
          f"{len(years)} лет ({years[0]}–{years[-1]}), сетка {temps.shape[1:]}")

    lats = np.arange(LAT_MIN, LAT_MAX + 1e-9, step)
    lons = np.arange(LON_MIN, LON_MAX + 1e-9, step)
    if temps.shape[1:] != (lats.size, lons.size):
        raise SystemExit(
            f"Сетка в кэше {temps.shape[1:]} не сходится с границами региона "
            f"({lats.size}×{lons.size}). Проверьте LAT_MIN/LAT_MAX/LON_MIN/LON_MAX."
        )

    years_arr = np.array(years)
    base_mask = (years_arr >= BASE_START) & (years_arr <= BASE_END)
    n_base = int(base_mask.sum())
    if n_base < 20:
        print(f"ВНИМАНИЕ: в норму попало {n_base} лет — этого мало для устойчивой нормы")

    with warnings.catch_warnings():          # пустые ячейки дают Mean of empty slice
        warnings.simplefilter("ignore", RuntimeWarning)
        clim = np.nanmean(temps[base_mask], axis=0)
    anom = temps - clim

    w = np.cos(np.deg2rad(lats))[:, None]
    region_mean, values = {}, {}
    for i, y in enumerate(years):
        v = anom[i]
        ok = ~np.isnan(v)
        region_mean[str(y)] = round(
            float(np.sum(np.where(ok, v, 0) * w) / np.sum(np.where(ok, 1, 0) * w)), 2
        )
        flat = np.round(v, DECIMALS).ravel()
        values[str(y)] = [None if np.isnan(x) else float(x) for x in flat]

    payload = {
        "meta": {
            "month": month, "model": model, "step": step,
            "base": [BASE_START, BASE_END],
            "lat_min": LAT_MIN, "lat_max": LAT_MAX,
            "lon_min": LON_MIN, "lon_max": LON_MAX,
            "nlat": int(lats.size), "nlon": int(lons.size),
            "source": "ERA5 (Copernicus C3S) via Open-Meteo Archive API",
        },
        "years": years,
        "region_mean": region_mean,
        # порядок: строки с юга на север, внутри строки — с запада на восток
        "values": values,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"[out] {OUT}  ({OUT.stat().st_size / 1e6:.1f} МБ)")

    hottest = max(region_mean, key=lambda k: region_mean[k])
    print(f"самый тёплый год: {hottest} ({region_mean[hottest]:+.2f} °C)")


if __name__ == "__main__":
    main()
