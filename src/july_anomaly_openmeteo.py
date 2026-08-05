#!/usr/bin/env python3
"""
Карта аномалий июльской температуры для Казахстана и Центральной Азии.
Данные: ERA5 через Open-Meteo Historical Weather API (без CDS, без лицензий).

Логика:
  1. Строим регулярную сетку точек по региону.
  2. Для каждого года качаем суточные средние температуры только за 1-31 июля
     (пачками по 100 точек в одном запросе) -> среднее за месяц в каждой точке.
  3. Норма = среднее по базовому периоду (1991-2020).
  4. Аномалия = июль целевого года - норма.
  5. Карта + ряд аномалий по годам + NetCDF/CSV.

Кэш: каждый год сохраняется в ./data/*.npz, повторный запуск ничего не перекачивает.

Установка:
    pip install requests numpy xarray netcdf4 matplotlib cartopy

Документация API: https://open-meteo.com/en/docs/historical-weather-api
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import requests
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import DownloadWarning

# Natural Earth скачивается один раз и кэшируется — не засоряем лог
warnings.filterwarnings("ignore", category=DownloadWarning)

# ============================== КОНФИГУРАЦИЯ ==============================

# Регион: Казахстан + Средняя Азия
LON_MIN, LON_MAX = 46.0, 88.0
LAT_MIN, LAT_MAX = 35.0, 56.0
GRID_STEP = 1          # шаг сетки в градусах (1.0 -> в 4 раза меньше запросов)

MONTH = 7                # июль
BASE_START, BASE_END = 1991, 2020   # норма ВМО
TARGET_YEAR = 2026       # год, для которого рисуем карту

MODEL = "era5"           # "era5" (0.25 град.) | "era5_land" (0.1 град., только суша)
                         # | "era5_seamless" (ERA5 + оперативные данные на хвосте)

# Ключ платного тарифа Open-Meteo. Оставьте None для бесплатного API.
API_KEY = None

BATCH = 100              # координат в одном запросе
PAUSE = 0.25             # пауза между запросами, сек
RETRIES = 5

DATADIR = Path("data")
OUTDIR = Path("output")

SHOW_CITIES = True
CITIES = {
    "Астана": (71.43, 51.13),
    "Алматы": (76.89, 43.24),
    "Караганда": (73.10, 49.81),
    "Актобе": (57.17, 50.28),
    "Атырау": (51.92, 47.09),
    "Ташкент": (69.24, 41.30),
    "Бишкек": (74.60, 42.87),
    "Душанбе": (68.79, 38.56),
    "Ашхабад": (58.38, 37.95),
}

MONTH_RU = {
    1: "январь", 2: "февраль", 3: "март", 4: "апрель", 5: "май", 6: "июнь",
    7: "июль", 8: "август", 9: "сентябрь", 10: "октябрь", 11: "ноябрь", 12: "декабрь",
}

# ==========================================================================

BASE_URL = (
    "https://customer-archive-api.open-meteo.com/v1/archive"
    if API_KEY
    else "https://archive-api.open-meteo.com/v1/archive"
)

SESSION = requests.Session()


def make_grid() -> tuple[np.ndarray, np.ndarray]:
    lats = np.arange(LAT_MIN, LAT_MAX + 1e-9, GRID_STEP)
    lons = np.arange(LON_MIN, LON_MAX + 1e-9, GRID_STEP)
    return lats, lons


def month_bounds(year: int) -> tuple[str, str]:
    last = [31, 29 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 28,
            31, 30, 31, 30, 31, 31, 30, 31, 30, 31][MONTH - 1]
    return f"{year}-{MONTH:02d}-01", f"{year}-{MONTH:02d}-{last:02d}"


def request_batch(lat_chunk, lon_chunk, start: str, end: str) -> list[float]:
    """Один запрос на пачку координат. Возвращает средние за месяц по каждой точке."""
    params = {
        "latitude": ",".join(f"{v:.4f}" for v in lat_chunk),
        "longitude": ",".join(f"{v:.4f}" for v in lon_chunk),
        "start_date": start,
        "end_date": end,
        "daily": "temperature_2m_mean",
        "timezone": "GMT",
        "models": MODEL,
        "cell_selection": "nearest",   # без привязки к ближайшей суше
    }
    if API_KEY:
        params["apikey"] = API_KEY

    for attempt in range(RETRIES):
        try:
            r = SESSION.get(BASE_URL, params=params, timeout=120)
            if r.status_code == 429:
                wait = 30 * (attempt + 1)
                print(f"    429 (лимит), жду {wait} c...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            js = r.json()
            break
        except requests.RequestException as e:
            if attempt == RETRIES - 1:
                raise
            print(f"    ошибка {e}, повтор {attempt + 1}/{RETRIES}")
            time.sleep(5 * (attempt + 1))
    else:
        raise RuntimeError("не удалось получить данные")

    if isinstance(js, dict):          # одна точка -> объект, а не список
        js = [js]
    if len(js) != len(lat_chunk):
        raise RuntimeError(f"ожидалось {len(lat_chunk)} точек, получено {len(js)}")

    out = []
    for item in js:
        vals = item.get("daily", {}).get("temperature_2m_mean", [])
        arr = np.array([np.nan if v is None else v for v in vals], dtype=float)
        out.append(np.nanmean(arr) if arr.size and not np.all(np.isnan(arr)) else np.nan)
    return out


def fetch_year(year: int, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Средняя температура месяца по сетке за один год. С кэшем на диске."""
    cache = DATADIR / f"om_{MODEL}_m{MONTH:02d}_{year}_{GRID_STEP}deg.npz"
    if cache.exists():
        return np.load(cache)["t"]

    grid_lat, grid_lon = np.meshgrid(lats, lons, indexing="ij")
    flat_lat, flat_lon = grid_lat.ravel(), grid_lon.ravel()
    start, end = month_bounds(year)

    n = flat_lat.size
    values = np.full(n, np.nan)
    print(f"[{year}] {n} точек, {int(np.ceil(n / BATCH))} запросов")

    for i in range(0, n, BATCH):
        sl = slice(i, min(i + BATCH, n))
        values[sl] = request_batch(flat_lat[sl], flat_lon[sl], start, end)
        print(f"    {min(i + BATCH, n)}/{n}", end="\r")
        time.sleep(PAUSE)
    print(f"    готово: среднее {np.nanmean(values):.2f} °C" + " " * 20)

    field = values.reshape(grid_lat.shape)
    DATADIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, t=field)
    return field


def build_dataarray() -> xr.DataArray:
    lats, lons = make_grid()
    years = list(range(BASE_START, max(BASE_END, TARGET_YEAR) + 1))

    stack = []
    for y in years:
        stack.append(fetch_year(y, lats, lons))

    da = xr.DataArray(
        np.stack(stack),
        dims=("year", "latitude", "longitude"),
        coords={"year": years, "latitude": lats, "longitude": lons},
        name="t2m",
        attrs={"units": "degC", "source": f"Open-Meteo Archive API, {MODEL}"},
    )
    return da


def plot_map(anom: xr.DataArray, path: Path) -> None:
    vmax = float(np.nanpercentile(np.abs(anom.values), 99))
    vmax = max(0.5, np.ceil(vmax * 4) / 4)
    levels = np.linspace(-vmax, vmax, 17)
    norm = mcolors.BoundaryNorm(levels, ncolors=256)

    proj = ccrs.LambertConformal(
        central_longitude=(LON_MIN + LON_MAX) / 2,
        central_latitude=(LAT_MIN + LAT_MAX) / 2,
        standard_parallels=(LAT_MIN + 5, LAT_MAX - 5),
    )

    fig = plt.figure(figsize=(13, 8))
    ax = plt.axes(projection=proj)
    ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())

    im = ax.pcolormesh(
        anom.longitude, anom.latitude, anom.values,
        cmap="RdBu_r", norm=norm, shading="auto", transform=ccrs.PlateCarree(),
    )

    ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.7, edgecolor="black")
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), lw=0.6, edgecolor="black")
    ax.add_feature(cfeature.LAKES.with_scale("50m"), facecolor="none",
                   edgecolor="black", lw=0.4)

    gl = ax.gridlines(draw_labels=True, lw=0.3, color="gray", alpha=0.5, linestyle=":")
    gl.top_labels = gl.right_labels = False

    if SHOW_CITIES:
        for name, (lon, lat) in CITIES.items():
            ax.plot(lon, lat, "o", ms=3.5, color="black",
                    transform=ccrs.PlateCarree(), zorder=5)
            ax.text(lon + 0.35, lat + 0.2, name, fontsize=8,
                    transform=ccrs.PlateCarree(), zorder=5)

    cb = fig.colorbar(im, ax=ax, orientation="vertical", pad=0.03, shrink=0.8,
                      ticks=levels[::2], extend="both")
    cb.set_label("Аномалия температуры на 2 м, °C")

    ax.set_title(
        f"Аномалия температуры, {MONTH_RU[MONTH]} {TARGET_YEAR} г.\n"
        f"относительно нормы {BASE_START}-{BASE_END}. "
        f"ERA5 через Open-Meteo",
        fontsize=13,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"[out] {path}")
    plt.close(fig)


def plot_series(series: xr.DataArray, path: Path) -> None:
    years, vals = series.year.values, series.values
    colors = np.where(vals >= 0, "#c0392b", "#2471a3")

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.bar(years, vals, color=colors, width=0.85)

    if len(vals) >= 10:
        ax.plot(years, series.rolling(year=10, center=True).mean().values,
                color="black", lw=1.8, label="10-летнее скользящее")
        ax.legend(frameon=False)

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Год")
    ax.set_ylabel("Аномалия, °C")
    ax.set_title(
        f"Аномалия температуры ({MONTH_RU[MONTH]}), среднее по региону. "
        f"Норма {BASE_START}-{BASE_END}"
    )
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"[out] {path}")
    plt.close(fig)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    da = build_dataarray()

    clim = da.sel(year=slice(BASE_START, BASE_END)).mean("year")
    anom_all = da - clim
    anom_map = anom_all.sel(year=TARGET_YEAR)

    w = np.cos(np.deg2rad(da.latitude))
    series = anom_all.weighted(w).mean(("latitude", "longitude"))

    print(f"\n=== {MONTH_RU[MONTH]} {TARGET_YEAR} ===")
    print(f"Средняя аномалия по региону: {float(series.sel(year=TARGET_YEAR)):+.2f} °C")
    print(f"Максимум: {float(anom_map.max()):+.2f} °C, "
          f"минимум: {float(anom_map.min()):+.2f} °C")

    rank = int((series >= series.sel(year=TARGET_YEAR)).sum())
    print(f"Место по теплу среди {series.year.size} лет: {rank}")
    print(f"\nСамые тёплые годы ({MONTH_RU[MONTH]}):")
    top = series.sortby(series, ascending=False)[:5]
    for y, v in zip(top.year.values, top.values):
        print(f"  {int(y)}: {float(v):+.2f} °C")

    plot_map(anom_map, OUTDIR / f"anomaly_{MONTH:02d}_{TARGET_YEAR}.png")
    plot_series(series, OUTDIR / f"anomaly_{MONTH:02d}_series.png")

    anom_map.to_netcdf(OUTDIR / f"anomaly_{MONTH:02d}_{TARGET_YEAR}.nc")
    series.to_dataframe(name="anomaly_C").to_csv(OUTDIR / f"anomaly_{MONTH:02d}_series.csv")
    print(f"[out] {OUTDIR}/anomaly_{MONTH:02d}_{TARGET_YEAR}.nc")


if __name__ == "__main__":
    main()
