"""
Step 2 of the article 3 pipeline: matching of satellite and weather data to the PS131 frames.

Input:  ps131_index\\ps131_frames.csv  (from build_ps131_index.py: time, GPS, frame metrics)
Output: ps131_index\\ps131_collocated.csv - one row per frame:
         frame + quality metrics
       + AMSR2 ASI (Univ. Bremen, 6.25 km, daily) ice concentration:
             sic_px (pixel), sic_3x3 (~19 km), sic_7x7 (~44 km), ice_class (0..3 as in C_ice)
       + ERA5 (nearest hour, 0.25°): u10, v10, wind, t2m, d2m, dpd (= t2m-d2m), tcc, lcc, tp, msl, siconc
       + fog_proxy (dpd < 1 K and lcc > 0.8)
       ps131_index\\collocation_report.txt and .png - relation of frame quality to weather and ice
       (first check of the claim "CV degrades in Arctic weather").

Sources without registration: AMSR2 Bremen (HTTP). ERA5 - via a Copernicus CDS key (~/.cdsapirc).

RUN:
    python collocate_ps131.py                  # download what is missing and match
    python collocate_ps131.py --skip-era5      # AMSR2 only
    python collocate_ps131.py --no-download    # only match the already downloaded data

Data folders (next to the script):
    06_amsr2_bremen_2022   07_era5_ps131
"""

import sys
import os
import io
import math
import zipfile
import argparse
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
AMSR_DIR = BASE / "06_amsr2_bremen_2022"
ERA5_DIR = BASE / "07_era5_ps131"
INDEX_DIR = Path(__file__).resolve().parent / "ps131_index"

AMSR_URL = ("https://data.seaice.uni-bremen.de/amsr2/asi_daygrid_swath/n6250/"
            "{y}/{mon}/Arctic/asi-AMSR2-n6250-{ymd}-v5.4.{ext}")
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]

# NSIDC Polar Stereographic North (EPSG:3411), Bremen n6250 grid: 1792 x 1216
PS_A, PS_E = 6378273.0, 0.081816153
PS_LATC, PS_LON0 = 70.0, -45.0
GRID_X0, GRID_Y0, CELL = -3850000.0, 5850000.0, 6250.0
GRID_SHAPE = (1792, 1216)

ERA5_AREA = [84, -30, 69, 15]  # N, W, S, E - covers the whole PS131 track with a margin
ERA5_VARS = ["10m_u_component_of_wind", "10m_v_component_of_wind", "2m_temperature",
             "2m_dewpoint_temperature", "total_cloud_cover", "low_cloud_cover",
             "total_precipitation", "mean_sea_level_pressure", "sea_ice_cover"]

ICE_THR = (0.15, 0.40, 0.70)  # the same thresholds as in C_ice (arctic_ice_comparison.py)


# ============================================================
# Network: direct, otherwise via the proxy in the environment variable PS131_PROXY
# ============================================================
def make_session():
    import requests
    s = requests.Session()
    s.headers["User-Agent"] = "Mozilla/5.0 (research download)"
    test = "https://data.seaice.uni-bremen.de/amsr2/"
    try:
        if s.get(test, timeout=10).status_code < 500:
            print("[net] Bremen reachable directly")
            return s
    except Exception:
        pass
    proxy = os.environ.get("PS131_PROXY")
    if not proxy:
        return s
    s.proxies = {"http": proxy, "https": proxy}
    try:
        if s.get(test, timeout=10).status_code < 500:
            print("[net] Bremen reachable via the proxy")
            return s
    except Exception as e:
        print(f"[net] not reachable via the proxy either: {e}")
    return s


# ============================================================
# AMSR2 Bremen
# ============================================================
def amsr_fmt():
    try:
        import pyhdf  # noqa: F401
        return "hdf"
    except ImportError:
        print("[amsr] pyhdf not found - using GeoTIFF instead of HDF")
        return "tif"


def download_amsr(dates, fmt):
    AMSR_DIR.mkdir(parents=True, exist_ok=True)
    sess = None
    for d in dates:
        fn = AMSR_DIR / f"asi-AMSR2-n6250-{d:%Y%m%d}-v5.4.{fmt}"
        if fn.exists() and fn.stat().st_size > 10_000:
            continue
        sess = sess or make_session()
        url = AMSR_URL.format(y=d.year, mon=MONTHS[d.month - 1], ymd=f"{d:%Y%m%d}", ext=fmt)
        try:
            r = sess.get(url, timeout=120)
            if r.status_code != 200:
                print(f"[amsr] {fn.name}: HTTP {r.status_code}")
                continue
            fn.write_bytes(r.content)
            print(f"[amsr] {fn.name} ({len(r.content) / 1e6:.1f} MB)")
        except Exception as e:
            print(f"[amsr] {fn.name}: {e}")


def read_amsr(path):
    """Returns SIC as a fraction (0..1), NaN = land/no data."""
    if path.suffix == ".hdf":
        from pyhdf.SD import SD, SDC
        sd = SD(str(path), SDC.READ)
        a = sd.select("ASI Ice Concentration")[:].astype(np.float32)
        sd.end()
    else:
        from PIL import Image
        a = np.asarray(Image.open(path)).astype(np.float32)
    a[(a < 0) | (a > 100)] = np.nan
    if a.shape != GRID_SHAPE:
        print(f"[amsr] [warning] shape {a.shape} != expected {GRID_SHAPE} - check the grid")
    return a / 100.0


def latlon_to_rowcol(lat, lon):
    """NSIDC polar stereographic north (Snyder 1987), vectorised."""
    lat = np.radians(np.asarray(lat, dtype=np.float64))
    dlon = np.radians(np.asarray(lon, dtype=np.float64) - PS_LON0)
    e = PS_E

    def t_of(phi):
        es = e * np.sin(phi)
        return np.tan(np.pi / 4 - phi / 2) / ((1 - es) / (1 + es)) ** (e / 2)

    latc = np.radians(PS_LATC)
    mc = np.cos(latc) / np.sqrt(1 - (e * np.sin(latc)) ** 2)
    rho = PS_A * mc * t_of(lat) / t_of(latc)
    x = rho * np.sin(dlon)
    y = -rho * np.cos(dlon)
    col = np.floor((x - GRID_X0) / CELL).astype(int)
    row = np.floor((GRID_Y0 - y) / CELL).astype(int)
    return row, col



GRID_URL = "https://data.seaice.uni-bremen.de/grid_coordinates/n6250/LongitudeLatitudeGrid-n6250-Arctic.hdf"
_GRID = {}


def load_grid():
    """Official Bremen n6250 coordinate grid (lon/lat of every pixel) + KD-tree.
    Matching on it does not depend on how the array is stored (flipped/transposed)."""
    if "tree" in _GRID:
        return _GRID
    p = AMSR_DIR / "LongitudeLatitudeGrid-n6250-Arctic.hdf"
    if not p.exists():
        try:
            sess = make_session()
            r = sess.get(GRID_URL, timeout=120)
            if r.status_code == 200:
                AMSR_DIR.mkdir(parents=True, exist_ok=True)
                p.write_bytes(r.content)
                print(f"[grid] downloaded the coordinate grid {p.name} ({len(r.content) / 1e6:.1f} MB)")
            else:
                print(f"[grid] grid not available: HTTP {r.status_code}")
        except Exception as e:
            print(f"[grid] grid not available: {e}")
    if not p.exists():
        return None
    try:
        from pyhdf.SD import SD, SDC
        from scipy.spatial import cKDTree
    except ImportError as e:
        print(f"[grid] no pyhdf/scipy ({e}) - using the projection formula")
        return None
    sd = SD(str(p), SDC.READ)
    names = list(sd.datasets().keys())
    lon_name = next(n for n in names if "lon" in n.lower())
    lat_name = next(n for n in names if "lat" in n.lower())
    lon = sd.select(lon_name)[:].astype(np.float64)
    lat = sd.select(lat_name)[:].astype(np.float64)
    sd.end()
    lon = np.where(lon > 180, lon - 360, lon)
    print(f"[grid] {lat_name}/{lon_name}: shape {lat.shape}, latitude {np.nanmin(lat):.1f}..{np.nanmax(lat):.1f}")
    xyz = _to_xyz(lat.ravel(), lon.ravel())
    _GRID.update(lat=lat, lon=lon, shape=lat.shape, tree=cKDTree(xyz))
    return _GRID


def _to_xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def rowcol_from_grid(grid, lat, lon):
    d, idx = grid["tree"].query(_to_xyz(np.asarray(lat, float), np.asarray(lon, float)))
    r, c = np.unravel_index(idx, grid["shape"])
    return r, c, d * 6371.0          # chord ~ arc at small distances, km


def window_mean(a, r, c, k):
    h = k // 2
    out = np.full(len(r), np.nan, dtype=np.float32)
    for i, (ri, ci) in enumerate(zip(r, c)):
        if 0 <= ri < a.shape[0] and 0 <= ci < a.shape[1]:
            w = a[max(0, ri - h): ri + h + 1, max(0, ci - h): ci + h + 1]
            if np.isfinite(w).any():
                out[i] = np.nanmean(w)
    return out


def collocate_amsr(df, fmt):
    for col in ("sic_px", "sic_3x3", "sic_7x7", "amsr_pix_dist_km"):
        df[col] = np.nan
    grid = load_grid()
    if grid is None:
        print("[amsr] [warning] matching by the projection formula, without a check against the Bremen grid")
    ok = df["lat"].notna()
    diag_done = False
    for day, g in df[ok].groupby(df.loc[ok, "time"].dt.floor("D")):
        p = AMSR_DIR / f"asi-AMSR2-n6250-{day:%Y%m%d}-v5.4.{fmt}"
        if not p.exists():
            continue
        a = read_amsr(p)
        if grid is not None:
            if a.shape != grid["shape"]:
                print(f"[amsr] [error] data shape {a.shape} != grid shape {grid['shape']}")
                return df
            r, c, dist = rowcol_from_grid(grid, g["lat"].values, g["lon"].values)
            df.loc[g.index, "amsr_pix_dist_km"] = dist
            if not diag_done:
                rp, cp = latlon_to_rowcol(g["lat"].values, g["lon"].values)
                same = np.mean((rp == r) & (cp == c))
                print(f"[check] {day:%Y-%m-%d}: the projection formula agreed with the Bremen grid for {same:.0%} of frames "
                      f"(if ~0% - the earlier AMSR2 matching was wrong)")
                diag_done = True
        else:
            r, c = latlon_to_rowcol(g["lat"].values, g["lon"].values)
        inside = (r >= 0) & (r < a.shape[0]) & (c >= 0) & (c < a.shape[1])
        px = np.full(len(r), np.nan, dtype=np.float32)
        px[inside] = a[r[inside], c[inside]]
        cells = pd.DataFrame({"r": r, "c": c})
        uniq = cells.drop_duplicates()
        m3 = dict(zip(map(tuple, uniq.values), window_mean(a, uniq.r.values, uniq.c.values, 3)))
        m7 = dict(zip(map(tuple, uniq.values), window_mean(a, uniq.r.values, uniq.c.values, 7)))
        keys = list(map(tuple, cells.values))
        df.loc[g.index, "sic_px"] = px
        df.loc[g.index, "sic_3x3"] = [m3[k] for k in keys]
        df.loc[g.index, "sic_7x7"] = [m7[k] for k in keys]
    if df["amsr_pix_dist_km"].notna().any():
        d = df["amsr_pix_dist_km"]
        print(f"[check] distance frame -> AMSR2 pixel centre: median {d.median():.2f} km, "
              f"max {d.max():.2f} km (normal for the 6.25 km grid: <= 4.5 km)")
    s = df["sic_3x3"]
    df["ice_class"] = np.select([s < ICE_THR[0], s < ICE_THR[1], s < ICE_THR[2], s >= ICE_THR[2]],
                                [0, 1, 2, 3], default=-1)
    track_plot(df, fmt, grid)
    return df


def track_plot(df, fmt, grid):
    """Check image: SIC map of the day + ship track. If the track is in the pack but the colour is 'water', something is wrong."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    day = pd.Timestamp("2022-07-25")
    p = AMSR_DIR / f"asi-AMSR2-n6250-{day:%Y%m%d}-v5.4.{fmt}"
    if not p.exists() or grid is None:
        return
    a = read_amsr(p)
    sel = (grid["lat"] > 70) & (grid["lon"] > -30) & (grid["lon"] < 20)
    rows, cols = np.where(sel)
    r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.pcolormesh(grid["lon"][r0:r1, c0:c1], grid["lat"][r0:r1, c0:c1], a[r0:r1, c0:c1],
                       cmap="Blues_r", vmin=0, vmax=1, shading="auto")
    t = df.dropna(subset=["lat"]).iloc[::60]
    ax.scatter(t["lon"], t["lat"], c="red", s=3, label="PS131 track (whole cruise)")
    dd = df[(df["time"].dt.floor("D") == day) & df["lat"].notna()]
    ax.scatter(dd["lon"], dd["lat"], c="yellow", s=12, label=f"position {day:%d.%m}")
    ax.set_xlim(-30, 20)
    ax.set_ylim(69, 84)
    ax.legend(loc="lower left")
    fig.colorbar(im, label=f"AMSR2 ASI SIC {day:%Y-%m-%d}")
    ax.set_title("Matching check: on 25.07 the ship should be in the pack")
    fig.tight_layout()
    fig.savefig(INDEX_DIR / "amsr_track_check.png", dpi=120)
    plt.close(fig)
    print("[check] check map: amsr_track_check.png")


# ============================================================
# ERA5
# ============================================================
def download_era5(months):
    ERA5_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import cdsapi
    except ImportError:
        print("[era5] no cdsapi: pip install cdsapi")
        return
    c = cdsapi.Client()
    for (y, m), days in months.items():
        out = ERA5_DIR / f"era5_ps131_{y}{m:02d}.download"
        if any(ERA5_DIR.glob(f"era5_ps131_{y}{m:02d}*.nc")) or out.exists():
            continue
        req = {
            "product_type": ["reanalysis"],
            "variable": ERA5_VARS,
            "year": [str(y)], "month": [f"{m:02d}"],
            "day": [f"{d:02d}" for d in sorted(days)],
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": ERA5_AREA,
            "data_format": "netcdf",
            "download_format": "unarchived",
        }
        print(f"[era5] request {y}-{m:02d}: {len(days)} days ...")
        c.retrieve("reanalysis-era5-single-levels", req, str(out))
        unpack_era5(out, y, m)


def unpack_era5(path, y, m):
    """The new CDS may return a zip with several .nc (instant / accum) - unpack them."""
    raw = path.read_bytes()
    if raw[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            for i, name in enumerate(n for n in z.namelist() if n.endswith(".nc")):
                (ERA5_DIR / f"era5_ps131_{y}{m:02d}_{i}.nc").write_bytes(z.read(name))
    else:
        (ERA5_DIR / f"era5_ps131_{y}{m:02d}.nc").write_bytes(raw)
    path.unlink()


def h5_var(f, name):
    d = f[name]
    a = d[()].astype(np.float64)
    fill = d.attrs.get("_FillValue")
    if fill is not None:
        a[a == np.ravel(fill)[0]] = np.nan
    sf, off = d.attrs.get("scale_factor"), d.attrs.get("add_offset")
    if sf is not None:
        a = a * float(np.ravel(sf)[0])
    if off is not None:
        a = a + float(np.ravel(off)[0])
    return a


def load_era5():
    import h5py
    fields, coords = {}, None
    for p in sorted(ERA5_DIR.glob("era5_ps131_*.nc")):
        with h5py.File(p, "r") as f:
            tname = "valid_time" if "valid_time" in f else "time"
            t = f[tname][()].astype(np.int64)
            units = f[tname].attrs.get("units", b"seconds since 1970-01-01")
            units = units.decode() if isinstance(units, bytes) else str(units)
            base = pd.Timestamp(units.split("since")[1].strip()[:19])
            step = pd.Timedelta(1, unit="h" if units.startswith("hours") else "s")
            times = np.asarray(base + t * step, dtype="datetime64[ns]")
            lat, lon = f["latitude"][()], f["longitude"][()]
            coords = (lat, lon)
            for v in ("u10", "v10", "t2m", "d2m", "tcc", "lcc", "tp", "msl", "siconc"):
                if v in f:
                    fields.setdefault(v, []).append((times, h5_var(f, v)))
    out = {}
    for v, parts in fields.items():
        times = np.concatenate([p[0] for p in parts])
        data = np.concatenate([p[1] for p in parts], axis=0)
        order = np.argsort(times)
        out[v] = (times[order], data[order])
    return out, coords


def collocate_era5(df, era, coords):
    lat, lon = coords
    ok = df["lat"].notna().values
    ilat = np.abs(lat[None, :] - df["lat"].values[ok, None]).argmin(axis=1)
    ilon = np.abs(lon[None, :] - df["lon"].values[ok, None]).argmin(axis=1)
    tq = df.loc[ok, "time"].dt.round("h").values.astype("datetime64[ns]")
    for v, (times, data) in era.items():
        it = np.searchsorted(times, tq).clip(0, len(times) - 1)
        good = times[it] == tq
        vals = np.full(ok.sum(), np.nan)
        vals[good] = data[it[good], ilat[good], ilon[good]]
        df[v] = np.nan
        df.loc[ok, v] = vals
    if "u10" in df and "v10" in df:
        df["wind"] = np.hypot(df["u10"], df["v10"])
    if "t2m" in df and "d2m" in df:
        df["dpd"] = df["t2m"] - df["d2m"]
        df["fog_proxy"] = ((df["dpd"] < 1.0) & (df.get("lcc", 0) > 0.8)).astype(int)
    return df


# ============================================================
# GPS: fill episodes without GPS by interpolation in time
# ============================================================
def fill_gps(df):
    df["gps_source"] = np.where(df["lat"].notna(), "exif", "")
    g = df[df["lat"].notna()].sort_values("time")
    miss = df["lat"].isna()
    if miss.any() and len(g) > 1:
        t_all = g["time"].astype("int64").values
        t_q = df.loc[miss, "time"].astype("int64").values
        df.loc[miss, "lat"] = np.interp(t_q, t_all, g["lat"].values)
        # longitude near 0 changes sign - interpolate as is (the track does not cross 180)
        df.loc[miss, "lon"] = np.interp(t_q, t_all, g["lon"].values)
        df.loc[miss, "gps_source"] = "interp"
        eps = [str(e) for e in df.loc[miss, "episode"].unique()]
        print(f"[gps] {miss.sum()} frames without GPS (episodes {eps}) - coordinates interpolated "
              f"in time between neighbouring episodes, gps_source=interp")
    return df


# ============================================================
# Report: relation of frame quality to weather/ice
# ============================================================
def report(df):
    lines = []
    out = lambda s="": (print(s), lines.append(s))
    out("=== PS131 MATCHING ===")
    out(f"Frames: {len(df)}; with SIC: {df['sic_3x3'].notna().sum()}; "
        f"with ERA5: {df['t2m'].notna().sum() if 't2m' in df else 0}")
    if df["sic_3x3"].notna().any():
        out("AMSR2 ice classes (3x3, C_ice thresholds 0.15/0.40/0.70), fraction of frames:")
        vc = df.loc[df["ice_class"] >= 0, "ice_class"].value_counts(normalize=True).sort_index()
        names = {0: "water <15%", 1: "very open 15-40%", 2: "close 40-70%", 3: "heavy >70%"}
        for k, v in vc.items():
            out(f"  {names[k]:16s} {v:.1%}")
    ep = df.groupby("episode").agg(
        start=("time", "min"), sharp=("sharpness", "median"), bright=("brightness", "mean"),
        sic=("sic_3x3", "mean"),
        **({"dpd": ("dpd", "mean"), "lcc": ("lcc", "mean"), "wind": ("wind", "mean"),
            "tp_mm": ("tp", lambda s: 1000 * s.mean())} if "dpd" in df else {}))
    ep.to_csv(INDEX_DIR / "ps131_episodes_collocated.csv", float_format="%.4f")
    out("\nSpearman by EPISODE (n={}), log10(sharpness) vs:".format(len(ep)))
    ls = np.log10(ep["sharp"].clip(lower=1e-3))
    for c in [c for c in ("sic", "dpd", "lcc", "wind", "tp_mm") if c in ep]:
        rho = ls.corr(ep[c], method="spearman")
        out(f"  {c:6s} rho = {rho:+.2f}")
    out("(episodes, not frames: the 720 frames of an episode are strongly correlated, "
        "frame-level p-values would be inflated)")
    if "fog_proxy" in df:
        f = df.groupby("fog_proxy")["sharpness"].median()
        out(f"\nMedian sharpness: fog_proxy=0 -> {f.get(0, np.nan):.1f}, fog_proxy=1 -> {f.get(1, np.nan):.1f}; "
            f"frames with fog_proxy: {int(df['fog_proxy'].sum())}")
    (INDEX_DIR / "collocation_report.txt").write_text("\n".join(lines), encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cols = [c for c in ("sic", "dpd", "lcc", "wind") if c in ep]
        if cols:
            fig, axes = plt.subplots(1, len(cols), figsize=(4 * len(cols), 3.6))
            axes = np.atleast_1d(axes)
            labels = {"sic": "SIC AMSR2 (3x3)", "dpd": "t2m - d2m, K", "lcc": "low cloud cover",
                      "wind": "wind 10 m, m/s"}
            for ax, c in zip(axes, cols):
                ax.scatter(ep[c], ls, s=18, color="#4C72B0")
                ax.set_xlabel(labels[c])
                ax.set_ylabel("log10(median sharpness)")
            fig.suptitle("PS131: frame quality vs ice/weather (point = hourly episode)")
            fig.tight_layout()
            fig.savefig(INDEX_DIR / "collocation_report.png", dpi=120)
            plt.close(fig)
            out("Plot: collocation_report.png")
    except Exception as e:
        out(f"[plot skipped: {e}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--skip-era5", action="store_true")
    args = ap.parse_args()

    src = INDEX_DIR / "ps131_frames.csv"
    df = pd.read_csv(src)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df[df["time"].notna()].copy()
    df = fill_gps(df)

    days = sorted(df["time"].dt.floor("D").unique())
    days = [pd.Timestamp(d).to_pydatetime() for d in days]
    fmt = amsr_fmt()
    months = {}
    for d in days:
        for dd in (d, d + timedelta(days=1)):
            months.setdefault((dd.year, dd.month), set()).add(dd.day)

    if not args.no_download:
        print(f"[amsr] days needed: {len(days)} ({days[0]:%Y-%m-%d} .. {days[-1]:%Y-%m-%d})")
        download_amsr(days, fmt)
        if not args.skip_era5:
            download_era5(months)

    df = collocate_amsr(df, fmt)
    if not args.skip_era5 and any(ERA5_DIR.glob("era5_ps131_*.nc")):
        era, coords = load_era5()
        df = collocate_era5(df, era, coords)
    elif not args.skip_era5:
        print("[era5] no ERA5 files - skipping (or run without --no-download)")

    out = INDEX_DIR / "ps131_collocated.csv"
    df.to_csv(out, index=False, float_format="%.5g")
    print(f"\nSaved: {out}")
    report(df)


if __name__ == "__main__":
    main()
