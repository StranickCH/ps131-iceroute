"""
Coverage of the PS131 episodes by DMI ice charts (article 3, alternative reference for the routing).

For each episode (time, ship position - as in route_ps131_v3.py), among the DMI charts within +-MAX_DT h
the chart with the largest fraction of filled cells in the 100 x 100 km window around the ship (1 km chart grid)
is chosen; with a tie - the nearest in time. The concentration windows and a summary are saved:
  - fraction of filled cells in the window, time lag;
  - DMI ice class at the ship position (3x3 mode) vs the bridge label and the AMSR2 class (independent check of the labels).

RUN:     python check_dmi_coverage_ps131.py --oof runs\\fusion_20261005_1154\\fusion_oof.csv
OUTPUT:  08_icecharts_2022\\dmi_windows_ps131.npz , dmi_coverage.csv , dmi_coverage.txt
"""
import re
import time
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
R_EARTH, LON0 = 6371000.0, -45.0          # sphere and central meridian of the DMI Greenland grid (proj4 in the files)
HALF = 50                                 # window of 100 x 100 cells of 1 km


def ice_class(conc):
    """DMI concentration, % -> classes of this study: <=3/10, 4-8/10, >=9/10 (NaN stays NaN)."""
    return np.where(np.isnan(conc), np.nan, np.where(conc <= 30, 0, np.where(conc <= 80, 1, 2)))


def to_xy(lat, lon):
    rho = 2 * R_EARTH * np.tan(np.pi / 4 - np.radians(lat) / 2)
    d = np.radians(lon - LON0)
    return rho * np.sin(d), -rho * np.cos(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", required=True)
    ap.add_argument("--charts", default=str(HERE / "08_icecharts_2022"))
    ap.add_argument("--max-dt", type=float, default=48.0, help="h: allowed time gap chart - episode")
    args = ap.parse_args()
    t0 = time.time()
    say = lambda m: print(f"[{datetime.now():%H:%M:%S} | +{time.time() - t0:5.0f} s] {m}", flush=True)

    oof = pd.read_csv(args.oof, low_memory=False)
    oof["time"] = pd.to_datetime(oof["time"])
    eps = (oof[oof["vis_label"] >= 0].groupby("episode")
           .agg(time=("time", "min"), lat=("lat", "median"), lon=("lon", "median"),
                ice=("ice_label", lambda s: int(s[s >= 0].mode().iat[0]) if (s >= 0).any() else -1),
                amsr=("pred_amsr", lambda s: int(s.mode().iat[0]))).reset_index())
    say(f"episodes {len(eps)}")

    charts = []
    for sub in ("dmi_regional", "dmi_overview"):
        for p in sorted((Path(args.charts) / sub).glob("*.nc")):
            m = re.search(r"(\d{12})\.nc$", p.name)
            if m:
                charts.append((sub, p, datetime.strptime(m.group(1), "%Y%m%d%H%M")))
    say(f"charts {len(charts)}")

    grid = {}
    rows, windows = [], {}
    for _, e in eps.iterrows():
        x, y = to_xy(e["lat"], e["lon"])
        best = None
        for sub, p, tc in charts:
            dt = abs((tc - e["time"].to_pydatetime()).total_seconds()) / 3600
            if dt > args.max_dt:
                continue
            if p not in grid:
                with xr.open_dataset(p) as ds:
                    grid[p] = (ds["xc"].values, ds["yc"].values)
            xc, yc = grid[p]
            j, i = int(np.argmin(np.abs(xc - x))), int(np.argmin(np.abs(yc - y)))
            if abs(xc[j] - x) > 1500 or abs(yc[i] - y) > 1500 or i < HALF or j < HALF \
                    or i + HALF > len(yc) or j + HALF > len(xc):
                continue
            with xr.open_dataset(p) as ds:
                w = ds["ice_concentration"].isel(time=0, yc=slice(i - HALF, i + HALF),
                                                 xc=slice(j - HALF, j + HALF)).values.astype(float)
            w[w < 0] = np.nan
            frac = float(np.isfinite(w).mean())
            key = (frac, -dt)
            if best is None or key > best[0]:
                best = (key, sub, p.name, dt, w)
        r = {"episode": e["episode"], "time": e["time"], "lat": e["lat"], "lon": e["lon"],
             "ice_bridge": e["ice"], "ice_amsr": e["amsr"]}
        if best is None:
            r.update(chart="", set="", dt_h=np.nan, valid_frac=0.0, ice_dmi=-1)
        else:
            (frac, _), sub, name, dt, w = best
            c = ice_class(w[HALF - 1:HALF + 2, HALF - 1:HALF + 2])
            c = c[np.isfinite(c)]
            r.update(chart=name, set=sub, dt_h=dt, valid_frac=frac,
                     ice_dmi=int(pd.Series(c).mode().iat[0]) if len(c) else -1)
            windows[str(e["episode"])] = w.astype(np.float32)
        rows.append(r)
        say(f"  {e['episode']}: {r['chart'] or 'NO CHART'}  coverage {r['valid_frac']:.0%}  "
            f"lag {r['dt_h']:.1f} h  ice DMI {r['ice_dmi']} / bridge {r['ice_bridge']} / AMSR2 {r['ice_amsr']}")

    df = pd.DataFrame(rows)
    out = Path(args.charts)
    df.to_csv(out / "dmi_coverage.csv", index=False)
    np.savez_compressed(out / "dmi_windows_ps131.npz", **windows)
    lines = [f"Episodes {len(df)}; with a chart (+-{args.max_dt:.0f} h): {int((df['valid_frac'] > 0).sum())}"]
    for thr in (0.5, 0.7, 0.9):
        lines.append(f"  window coverage >= {thr:.0%}: {int((df['valid_frac'] >= thr).sum())} episodes")
    m = df[(df["ice_dmi"] >= 0) & (df["ice_bridge"] >= 0)]
    if len(m):
        lines.append(f"\nIce class at the ship position, {len(m)} episodes:")
        lines.append(f"  DMI = bridge: {(m['ice_dmi'] == m['ice_bridge']).mean():.0%}   "
                     f"AMSR2 = bridge: {(m['ice_amsr'] == m['ice_bridge']).mean():.0%}   "
                     f"DMI = AMSR2: {(m['ice_dmi'] == m['ice_amsr']).mean():.0%}")
        lines.append(pd.crosstab(m["ice_bridge"], m["ice_dmi"], rownames=["bridge"], colnames=["DMI"]).to_string())
    (out / "dmi_coverage.txt").write_text("\n".join(lines), encoding="utf-8")
    for l in lines:
        say(l)
    say(f"DONE: {out}")


if __name__ == "__main__":
    main()
