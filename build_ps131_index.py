"""
Index of the PS131 ATWAICE frames for article 3 (step 1 of the pipeline).

For every GoPro HERO9 frame (5184x3888, 5 s step) collects into one table:
  * path, episode, time (from the file name and EXIF)
  * coordinates from EXIF GPS (lat, lon, alt) -> later used to match AMSR2 / ERA5
  * cheap frame quality metrics (on a 1/8 downscaled copy, via JPEG draft):
      brightness  - mean brightness (0..255)
      contrast    - std of brightness
      sharpness   - variance of the Laplacian (low = fog / blur / drops on the lens)
      white_frac  - fraction of almost white pixels (ice, snow, fog, overexposure)
      sat_mean    - mean saturation (fog/cloud -> low)
      top_bottom  - brightness difference top/bottom of the frame (poor visibility -> the horizon "dissolves")
    These are not labels but proxies of weather/visibility; they are used as frame-quality
    features and to describe the imaging conditions.

Runs in several processes and can continue from where it stopped
(already processed paths are skipped).

RUN:
    python build_ps131_index.py
    python build_ps131_index.py --root D:\\data\\03_camera_PS131_ATWAICE_full --workers 6

Result (in the folder ps131_index next to the script):
    ps131_frames.csv     - one row per frame
    ps131_episodes.csv   - summary by episode (time, coordinates, mean metrics)
    ps131_track.png      - ship track, colour = date
    ps131_quality.png    - histograms of the quality metrics
    ps131_summary.txt    - text summary
"""

import os
import re
import csv
import sys
import math
import argparse
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool

import numpy as np
from PIL import Image

DEFAULT_ROOT = str(Path(__file__).resolve().parent / "03_camera_PS131_ATWAICE_full")
OUT_DIR = Path(__file__).resolve().parent / "ps131_index"
FIELDS = ["path", "episode", "time", "exif_time", "lat", "lon", "alt",
          "brightness", "contrast", "sharpness", "white_frac", "sat_mean", "top_bottom", "error"]
TS_RE = re.compile(r"(20\d{6})(\d{6})")
GPS_IFD = 0x8825


def dms_to_deg(dms, ref):
    try:
        d, m, s = (float(x) for x in dms)
    except Exception:
        return None
    val = d + m / 60 + s / 3600
    return -val if str(ref).upper().startswith(("S", "W")) else val


def process(path_str):
    p = Path(path_str)
    row = {k: "" for k in FIELDS}
    row["path"] = path_str
    row["episode"] = p.parent.name
    m = TS_RE.search(p.name)
    if m:
        try:
            row["time"] = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").isoformat()
        except ValueError:
            pass
    try:
        with Image.open(p) as im:
            ex = im.getexif()
            dt = ex.get(306)  # DateTime
            if dt:
                row["exif_time"] = str(dt).replace(":", "-", 2).replace(" ", "T")
            gps = ex.get_ifd(GPS_IFD) if hasattr(ex, "get_ifd") else {}
            if gps:
                lat = dms_to_deg(gps.get(2), gps.get(1, "N"))
                lon = dms_to_deg(gps.get(4), gps.get(3, "E"))
                row["lat"] = f"{lat:.6f}" if lat is not None else ""
                row["lon"] = f"{lon:.6f}" if lon is not None else ""
                if gps.get(6) is not None:
                    try:
                        row["alt"] = f"{float(gps.get(6)):.1f}"
                    except Exception:
                        pass
            im.draft("RGB", (im.width // 8, im.height // 8))  # fast JPEG decode at 1/8
            im = im.convert("RGB")
            if max(im.size) > 700:
                im.thumbnail((700, 700))
            rgb = np.asarray(im, dtype=np.float32)
        gray = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        lap = (-4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1]
               + gray[1:-1, :-2] + gray[1:-1, 2:])
        mx, mn = rgb.max(axis=2), rgb.min(axis=2)
        sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
        h = gray.shape[0]
        row.update({
            "brightness": f"{gray.mean():.2f}",
            "contrast": f"{gray.std():.2f}",
            "sharpness": f"{lap.var():.2f}",
            "white_frac": f"{(gray > 220).mean():.4f}",
            "sat_mean": f"{sat.mean():.4f}",
            "top_bottom": f"{gray[: h // 2].mean() - gray[h // 2:].mean():.2f}",
        })
    except Exception as e:
        row["error"] = str(e)[:200]
    return row


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def summarize(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path)
    lines = []
    out = lambda s="": (print(s), lines.append(s))
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    n_err = df["error"].notna().sum()
    out(f"Frames: {len(df)}, read errors: {n_err}, with GPS: {df['lat'].notna().sum()}")
    out(f"Period: {df['time'].min()} .. {df['time'].max()}")
    g = df.dropna(subset=["lat", "lon"]).sort_values("time")
    if len(g):
        out(f"Latitude: {g['lat'].min():.3f} .. {g['lat'].max():.3f}; longitude: {g['lon'].min():.3f} .. {g['lon'].max():.3f}")
        sub = g.iloc[::12]  # once per minute
        dist = sum(haversine_km(a.lat, a.lon, b.lat, b.lon)
                   for a, b in zip(sub.itertuples(), sub.iloc[1:].itertuples()))
        out(f"Track length (estimate from one-minute points): {dist:.0f} km")
        bad = (g["lat"].abs() < 1) & (g["lon"].abs() < 1)
        if bad.any():
            out(f"[warning] {bad.sum()} frames with GPS near (0,0) - no fix, discard")
    for col in ["brightness", "contrast", "sharpness", "white_frac", "sat_mean", "top_bottom"]:
        s = df[col].dropna()
        if len(s):
            q = s.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).round(3).tolist()
            out(f"{col:11s} p5/p25/p50/p75/p95: {q}")
    s = df["sharpness"].dropna()
    if len(s):
        thr = s.quantile(0.10)
        out(f"Candidates for 'poor visibility' (sharpness < p10={thr:.1f}): {(df['sharpness'] < thr).sum()} frames")

    ep = df.groupby("episode").agg(
        start=("time", "min"), end=("time", "max"), n=("path", "size"),
        lat=("lat", "mean"), lon=("lon", "mean"),
        brightness=("brightness", "mean"), sharpness=("sharpness", "median"),
        white_frac=("white_frac", "mean"), sat_mean=("sat_mean", "mean"))
    ep.to_csv(OUT_DIR / "ps131_episodes.csv", float_format="%.4f")
    out(f"Summary of {len(ep)} episodes: ps131_episodes.csv")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if len(g):
            fig, ax = plt.subplots(figsize=(7, 6))
            t = (g["time"] - g["time"].min()).dt.total_seconds() / 86400
            sc = ax.scatter(g["lon"], g["lat"], c=t, s=1, cmap="viridis")
            fig.colorbar(sc, label=f"days from {g['time'].min():%Y-%m-%d}")
            ax.set_xlabel("longitude")
            ax.set_ylabel("latitude")
            ax.set_title("PS131: track from the GoPro frame GPS")
            fig.tight_layout()
            fig.savefig(OUT_DIR / "ps131_track.png", dpi=130)
            plt.close(fig)
        cols = ["brightness", "contrast", "sharpness", "white_frac", "sat_mean", "top_bottom"]
        fig, axes = plt.subplots(2, 3, figsize=(11, 6))
        for ax, c in zip(axes.ravel(), cols):
            v = df[c].dropna()
            if c == "sharpness":
                v = np.log10(v.clip(lower=1e-3))
                ax.set_title("log10(sharpness)")
            else:
                ax.set_title(c)
            ax.hist(v, bins=60, color="#4C72B0")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "ps131_quality.png", dpi=110)
        plt.close(fig)
        out("Plots: ps131_track.png, ps131_quality.png")
    except Exception as e:
        out(f"[plots skipped: {e}]")
    (OUT_DIR / "ps131_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--limit", type=int, default=0, help="only N frames (for a check)")
    args = ap.parse_args()

    root = Path(args.root)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "ps131_frames.csv"

    files = sorted(str(p) for p in root.rglob("*")
                   if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg"))
    if args.limit:
        files = files[: args.limit]
    done = set()
    if csv_path.exists():
        with open(csv_path, encoding="utf-8", newline="") as f:
            done = {r["path"] for r in csv.DictReader(f)}
    todo = [f for f in files if f not in done]
    print(f"Frames found: {len(files)}, already in the index: {len(done)}, to process: {len(todo)}; "
          f"processes: {args.workers}")

    if todo:
        new = not csv_path.exists()
        with open(csv_path, "a", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            with Pool(args.workers) as pool:
                for i, row in enumerate(pool.imap_unordered(process, todo, chunksize=16), 1):
                    w.writerow(row)
                    if i % 500 == 0:
                        f.flush()
                        print(f"  {i}/{len(todo)}", flush=True)
    print("\n=== SUMMARY ===")
    summarize(csv_path)
    print(f"\nDONE: {OUT_DIR}")


if __name__ == "__main__":
    main()
