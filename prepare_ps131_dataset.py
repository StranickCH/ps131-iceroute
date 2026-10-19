"""
Step 4 of the article 3 pipeline: preparation of PS131 for training (NO training).

1) LABELS (taken from ps131_index\\ps131_labeled.csv), 3 classes each (see VIS_MAP / ICE_MAP below):
     vis_label  0: <1 km, 1: 1-10 km, 2: >10 km                (visibility, Polarstern observers)
     ice_label  0: <=3/10, 1: 4-8/10, 2: >=9/10                (ice from the bridge, WMO code 0639)
   A label is set only if the synoptic report is no further than --max-dt minutes away (default 90),
   otherwise -1 (the frame does not take part in training for this task). AMSR2/ERA5 are kept as features.

2) train/val/test SPLIT BY EPISODE (~70/15/15 by frames). Out of several thousand random splits
   the one where the visibility and ice distributions of the three parts are closest
   to each other is chosen. Frames of one hourly episode never go to different parts.

3) CACHE of downscaled 1024x768 frames (--cache). The heavy part: ~40 thousand 5184x3888 JPEGs,
   about 20-40 minutes on 6-8 processes, ~5-6 GB on disk. The processes run with lowered
   priority, but the computer will still be busy.
   Can be continued after a stop (finished files are skipped).

RUN:
    python prepare_ps131_dataset.py                 # labels + split only (seconds)
    python prepare_ps131_dataset.py --cache         # + frame cache (slow, CPU)
    python prepare_ps131_dataset.py --cache --workers 4 --stride 2   # every 2nd frame (10 s step)

Result: folder ps131_dataset\\ next to the script:
    splits.csv, labels_train.csv, labels_val.csv, labels_test.csv, dataset_report.txt
Cache: 03_camera_PS131_cache1024\\<episode>\\<frame>.jpg
"""

import os
import argparse
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
LABELED = HERE / "ps131_index" / "ps131_labeled.csv"
OUT = HERE / "ps131_dataset"
CACHE_ROOT = HERE / "03_camera_PS131_cache1024"
CACHE_SIZE = (1024, 768)

# 3 classes: the rare "4-10 km" (2 episodes) and "very open ice" (2 episodes) are merged with their neighbours,
# otherwise they fall entirely into one part of the split and the model either never sees them or is never tested on them.
VIS_MAP = {"<1km": 0, "1-4km": 1, "4-10km": 1, ">10km": 2}
ICE_MAP = {0: 0, 1: 0, 2: 1, 3: 2}
VIS_NAMES = {0: "poor <1km (fog)", 1: "moderate 1-10km", 2: "good >10km"}
ICE_NAMES = {0: "open/very open <=3/10", 1: "close 4-8/10", 2: "heavy >=9/10"}
NCLS = 3
KEEP = ["path", "episode", "time", "lat", "lon", "gps_source",
        "brightness", "contrast", "sharpness", "white_frac", "sat_mean", "top_bottom",
        "sic_px", "sic_3x3", "sic_7x7", "ice_class",
        "u10", "v10", "wind", "t2m", "d2m", "dpd", "tcc", "lcc", "tp", "msl",
        "vis_km", "ww", "ww_class", "ci", "S_i", "N", "Nh", "vis_label", "ice_label"]


# ------------------------------------------------------------
def make_labels(df, max_dt):
    df["vis_label"] = df["vis_class"].map(VIS_MAP).fillna(-1).astype(int)
    df["ice_label"] = df["ice_obs_class"].map(ICE_MAP).fillna(-1).astype(int)
    if "dt_vv_code_min" in df:
        df.loc[~(df["dt_vv_code_min"] <= max_dt), "vis_label"] = -1
    if "dt_ci_min" in df:
        df.loc[~(df["dt_ci_min"] <= max_dt), "ice_label"] = -1
    return df


def dist(sub, col, k=NCLS):
    v = sub.loc[sub[col] >= 0, col].value_counts(normalize=True)
    return np.array([v.get(i, 0.0) for i in range(k)])


def split_episodes(df, n_trials, seed, frac=(0.70, 0.15, 0.15)):
    rng = np.random.default_rng(seed)
    eps = df.groupby("episode").size()
    names = eps.index.values
    total = eps.sum()
    best, best_cost = None, np.inf
    ref_vis, ref_ice = dist(df, "vis_label"), dist(df, "ice_label")
    for _ in range(n_trials):
        order = rng.permutation(names)
        cum = np.cumsum(eps[order].values) / total
        split = np.where(cum <= frac[0], "train", np.where(cum <= frac[0] + frac[1], "val", "test"))
        assign = dict(zip(order, split))
        s = df["episode"].map(assign)
        cost = 0.0
        for part in ("train", "val", "test"):
            sub = df[s == part]
            dv, di = dist(sub, "vis_label"), dist(sub, "ice_label")
            # hard condition: every class of both tasks is present in every part
            if (dv == 0).any() or (di == 0).any():
                cost = np.inf
                break
            if part != "train":
                cost += np.abs(dv - ref_vis).sum() + np.abs(di - ref_ice).sum()
        if cost < best_cost:
            best, best_cost = assign, cost
    return best, best_cost


# ------------------------------------------------------------
def cache_one(args):
    src, dst = args
    dst = Path(dst)
    if dst.exists():
        return 0
    try:
        from PIL import Image
        dst.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as im:
            im.draft("RGB", (im.width // 4, im.height // 4))
            im = im.convert("RGB").resize(CACHE_SIZE, Image.LANCZOS)
            tmp = dst.with_suffix(".tmp.jpg")
            im.save(tmp, quality=90)
            os.replace(tmp, dst)
        return 1
    except Exception as e:
        return f"{src}: {e}"


def lower_priority():
    try:
        import psutil
        p = psutil.Process()
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
    except Exception:
        pass


def build_cache(df, workers, stride):
    jobs = []
    for ep, g in df.sort_values("time").groupby("episode"):
        for p in g["path"].values[::stride]:
            jobs.append((p, str(CACHE_ROOT / str(ep) / Path(p).name)))
    todo = [j for j in jobs if not Path(j[1]).exists()]
    print(f"[cache] frames to cache: {len(jobs)} (stride {stride}), already done: {len(jobs) - len(todo)}, "
          f"left: {len(todo)}; processes: {workers}")
    errors = []
    with Pool(workers, initializer=lower_priority) as pool:
        for i, r in enumerate(pool.imap_unordered(cache_one, todo, chunksize=8), 1):
            if isinstance(r, str):
                errors.append(r)
            if i % 1000 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
    if errors:
        print(f"[cache] errors: {len(errors)}, e.g.: {errors[:3]}")
    done = {src: dst for src, dst in jobs if Path(dst).exists()}
    return done


# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-dt", type=float, default=90, help="max. distance of the report from the frame, min")
    ap.add_argument("--trials", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache", action="store_true", help="build the 1024x768 cache (slow)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--stride", type=int, default=1, help="every N-th frame into the cache")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(LABELED, low_memory=False)
    df["time"] = pd.to_datetime(df["time"])
    df = make_labels(df, args.max_dt)

    assign, cost = split_episodes(df, args.trials, args.seed)
    if assign is None:
        print("[error] no split found where all classes are present in all parts. Increase --trials")
        return
    df["split"] = df["episode"].map(assign)
    pd.Series(assign, name="split").rename_axis("episode").to_csv(OUT / "splits.csv")

    if args.cache:
        done = build_cache(df, args.workers, args.stride)
        df["cache_path"] = df["path"].map(done).fillna("")
    else:
        df["cache_path"] = [str(CACHE_ROOT / str(e) / Path(p).name) for e, p in zip(df["episode"], df["path"])]
        df.loc[~df["cache_path"].map(lambda x: Path(x).exists()), "cache_path"] = ""

    cols = [c for c in KEEP if c in df] + ["cache_path", "split"]
    for part in ("train", "val", "test"):
        df.loc[df["split"] == part, cols].to_csv(OUT / f"labels_{part}.csv", index=False, float_format="%.5g")

    lines = []
    say = lambda s="": (print(s), lines.append(s))
    say(f"Split by episode (best of {args.trials}, distribution mismatch {cost:.3f}):")
    for part in ("train", "val", "test"):
        sub = df[df["split"] == part]
        say(f"\n{part.upper()}: episodes {sub['episode'].nunique()}, frames {len(sub)} "
            f"({len(sub) / len(df):.0%}); with a visibility label {int((sub['vis_label'] >= 0).sum())}, "
            f"with an ice label {int((sub['ice_label'] >= 0).sum())}; in the cache {int((sub['cache_path'] != '').sum())}")
        dv = sub.loc[sub["vis_label"] >= 0, "vis_label"].value_counts(normalize=True).sort_index()
        di = sub.loc[sub["ice_label"] >= 0, "ice_label"].value_counts(normalize=True).sort_index()
        say("  visibility: " + ", ".join(f"{VIS_NAMES[k]} {v:.0%}" for k, v in dv.items()))
        say("  ice:        " + ", ".join(f"{ICE_NAMES[k]} {v:.0%}" for k, v in di.items()))
        say("  episodes: " + " ".join(sorted(map(str, sub["episode"].unique()))))
    say(f"\nLabels only when the report is no further than {args.max_dt:.0f} min from the frame.")
    (OUT / "dataset_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nDONE: {OUT}")
    if not args.cache:
        print("The frame cache was not built. To build it (20-40 min): --cache")


if __name__ == "__main__":
    main()
