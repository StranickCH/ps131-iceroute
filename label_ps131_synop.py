"""
Step 3 of the article 3 pipeline: labels from the Polarstern observers for the PS131 frames.

Source: Schmithüsen, Rohleder, Suter (2025): Meteorological observations during
POLARSTERN cruise PS131. PANGAEA, https://doi.org/10.1594/PANGAEA.987909 (CC-BY-4.0)
Synoptic reports every 3 h; full manual observations (visibility, weather, ice) are
usually at 06/09/12/15/18 UTC, at night some fields are empty (automatic reports).

For each frame the nearest NON-EMPTY value of each field within ±3 h is taken
(dt in minutes is kept so that the filter can be tightened).

New columns:
  vis_km, vis_class   - visibility (WMO code 4377) -> km; class: <1 km / 1-4 / 4-10 / >10 km
  ww, ww_class        - present weather (WMO 4677): clear / haze_mist / fog / fog_distant / precip / recent / other
  ci, ice_obs_class   - ice from the bridge (WMO 0639), reduced to the C_ice classes:
                          0 water (<1/10), 1 very open (<=3/10), 2 close (4-8/10), 3 heavy (>=9/10)
  S_i                 - stage of ice development (code)
  N, Nh               - cloud cover (oktas), TTT/TdTdTd/ff - temperature, dew point, wind

Report (synop_report.txt): frame quality vs visibility/weather (by episode) and
agreement of AMSR2 (satellite) ice with the bridge observations.

RUN:
    python label_ps131_synop.py                               # looks for PS131_meteorology.tab in the script folder
    python label_ps131_synop.py --tab pangaea\\PS131_meteorology.tab
Input:  ps131_index\\ps131_collocated.csv (or ps131_frames.csv if there is no collocated file)
Output: ps131_index\\ps131_labeled.csv, synop_report.txt
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

INDEX_DIR = Path(__file__).resolve().parent / "ps131_index"
SEARCH_ROOT = Path(__file__).resolve().parent
TOL = pd.Timedelta(hours=3)


# ------------------------------------------------------------
# WMO codes
# ------------------------------------------------------------
def vv_to_km(code):
    """WMO code table 4377 (horizontal visibility)."""
    if pd.isna(code):
        return np.nan
    c = int(code)
    if 0 <= c <= 50:
        return c * 0.1
    if 56 <= c <= 80:
        return float(c - 50)
    if 81 <= c <= 88:
        return 30.0 + (c - 80) * 5.0
    if c == 89:
        return 75.0
    sea = {90: 0.025, 91: 0.05, 92: 0.2, 93: 0.5, 94: 1.0, 95: 2.0, 96: 4.0, 97: 10.0, 98: 20.0, 99: 50.0}
    return sea.get(c, np.nan)


def vis_class(km):
    if pd.isna(km):
        return ""
    return "<1km" if km < 1 else "1-4km" if km < 4 else "4-10km" if km < 10 else ">10km"


def ww_class(code):
    """WMO code table 4677 (present weather), aggregated."""
    if pd.isna(code):
        return ""
    c = int(code)
    if c <= 3:
        return "clear"
    if 4 <= c <= 12:
        return "haze_mist"          # haze, mist, shallow fog
    if c == 40:
        return "fog_distant"        # fog at a distance, not at the ship
    if 41 <= c <= 49:
        return "fog"
    if 50 <= c <= 99:
        return "precip"             # drizzle, rain, snow, showers
    if 20 <= c <= 29:
        return "recent"             # phenomena during the preceding hour, not now
    return "other"


def ci_to_class(code):
    """WMO code table 0639 (concentration or arrangement of sea ice) -> C_ice classes."""
    if pd.isna(code):
        return -1
    c = int(code)
    return {0: 0, 1: 0, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 3, 8: 3, 9: 3}.get(c, -1)


# ------------------------------------------------------------
def read_pangaea_tab(path):
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    end = next(i for i, l in enumerate(lines) if l.strip() == "*/")
    df = pd.read_csv(path, sep="\t", skiprows=end + 1, encoding="utf-8", encoding_errors="replace")
    df.columns = [c.split(" [")[0].strip() for c in df.columns]
    df["Date/Time"] = pd.to_datetime(df["Date/Time"])
    return df.sort_values("Date/Time").reset_index(drop=True)


def nearest_nonnull(frame_times, syn, col):
    s = syn[["Date/Time", col]].dropna()
    if s.empty:
        return np.full(len(frame_times), np.nan), np.full(len(frame_times), np.nan)
    t = s["Date/Time"].values.astype("datetime64[ns]")
    q = frame_times.values.astype("datetime64[ns]")
    i = np.searchsorted(t, q).clip(1, len(t) - 1)
    left, right = t[i - 1], t[i]
    pick = np.where(np.abs(q - left) <= np.abs(right - q), i - 1, i)
    dt = np.abs(q - t[pick])
    vals = s[col].values[pick].astype(float)
    bad = dt > TOL.to_timedelta64()
    vals[bad] = np.nan
    dt_min = dt.astype("timedelta64[s]").astype(float) / 60
    dt_min[bad] = np.nan
    return vals, dt_min


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tab", default=None, help="path to PS131_meteorology.tab")
    args = ap.parse_args()

    tab = Path(args.tab) if args.tab else next(SEARCH_ROOT.rglob("PS131_meteorology.tab"), None)
    if tab is None or not tab.exists():
        print(f"PS131_meteorology.tab not found under {SEARCH_ROOT}. Give the path: --tab ...")
        return
    syn = read_pangaea_tab(tab)
    print(f"Synoptic data: {tab}\n  reports: {len(syn)}, {syn['Date/Time'].min()} .. {syn['Date/Time'].max()}")
    for c in ("VV", "ww", "ci", "N"):
        if c in syn:
            print(f"  {c}: non-empty {syn[c].notna().sum()}")

    src = INDEX_DIR / "ps131_collocated.csv"
    if not src.exists():
        src = INDEX_DIR / "ps131_frames.csv"
    df = pd.read_csv(src, low_memory=False)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df[df["time"].notna()].copy()
    print(f"Frames: {src.name}, {len(df)}")

    for col, out in (("VV", "vv_code"), ("ww", "ww"), ("ci", "ci"), ("S_i", "S_i"), ("N", "N"),
                     ("Nh", "Nh"), ("TTT", "TTT"), ("TdTdTd", "TdTdTd"), ("ff", "ff")):
        if col in syn:
            v, dt = nearest_nonnull(df["time"], syn, col)
            df[out] = v
            if col in ("VV", "ww", "ci"):
                df[f"dt_{out}_min"] = dt
    df["vis_km"] = df["vv_code"].map(vv_to_km) if "vv_code" in df else np.nan
    df["vis_class"] = df["vis_km"].map(vis_class)
    df["ww_class"] = df["ww"].map(ww_class) if "ww" in df else ""
    df["ice_obs_class"] = df["ci"].map(ci_to_class) if "ci" in df else -1

    out = INDEX_DIR / "ps131_labeled.csv"
    df.to_csv(out, index=False, float_format="%.5g")
    print(f"Saved: {out}")
    report(df)


def report(df):
    lines = []
    say = lambda s="": (print(s), lines.append(s))
    say("\n=== LABELS FROM THE OBSERVERS (Polarstern synoptic reports) ===")
    for col, name in (("vis_class", "Visibility"), ("ww_class", "Weather (ww)")):
        vc = df.loc[df[col] != "", col].value_counts()
        say(f"{name}, frames: " + ", ".join(f"{k}: {v}" for k, v in vc.items()))
    ice_names = {0: "water", 1: "very open", 2: "close", 3: "heavy"}
    vc = df.loc[df["ice_obs_class"] >= 0, "ice_obs_class"].value_counts().sort_index()
    say("Ice from the bridge (C_ice classes), frames: " + ", ".join(f"{ice_names[k]}: {v}" for k, v in vc.items()))

    ep = df.groupby("episode").agg(sharp=("sharpness", "median"), vis=("vis_km", "median"),
                                   ww=("ww_class", lambda s: s.mode().iat[0] if len(s.mode()) else ""))
    ls = np.log10(ep["sharp"].clip(lower=1e-3))
    if ep["vis"].notna().sum() > 5:
        rho = ls.corr(np.log10(ep["vis"]), method="spearman")
        say(f"\nSpearman by episode: log10(sharpness) vs log10(visibility): rho = {rho:+.2f} "
            f"(n = {ep['vis'].notna().sum()})")
    g = ep.groupby("ww")["sharp"].agg(["count", "median"])
    say("Median sharpness by weather (episodes):")
    for k, r in g.iterrows():
        if k:
            say(f"  {k:10s} n={int(r['count']):3d}  sharpness={r['median']:.1f}")

    if "ice_class" in df:
        m = df[(df["ice_class"] >= 0) & (df["ice_obs_class"] >= 0)]
        if len(m):
            ct = pd.crosstab(m["ice_obs_class"].map(ice_names), m["ice_class"].map(ice_names),
                             rownames=["bridge"], colnames=["AMSR2"])
            agree = (m["ice_obs_class"] == m["ice_class"]).mean()
            say(f"\nIce: bridge (rows) vs AMSR2 3x3 (columns), frames {len(m)}, class agreement {agree:.1%}")
            say(ct.to_string())
            under = (m["ice_class"] < m["ice_obs_class"]).mean()
            say(f"AMSR2 underestimates the ice class relative to the bridge: {under:.1%} of frames")
    say("\nNOTE: the code mapping follows WMO-No. 306 code tables 4377/4677/0639 (see the supplementary material).")
    (INDEX_DIR / "synop_report.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
