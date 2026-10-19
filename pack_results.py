# -*- coding: utf-8 -*-
"""Packs the results behind the paper into paper_results.zip (for Zenodo).

Takes the latest run of every tag used in the paper, the matched frame table with labels,
the out-of-fold predictions, the DMI chart windows and the text summaries. Absolute Windows
paths to the camera frames in CSV files are made relative (03_camera_PS131_ATWAICE_full\\...).
Camera images, AMSR2, ERA5 and DMI source files are not included (see README.md, Data).

Run:  python pack_results.py [folder with runs, ps131_index, 08_icecharts_2022] [output zip]
"""
import glob
import os
import re
import sys
import zipfile

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = sys.argv[2] if len(sys.argv) > 2 else "paper_results.zip"

ROUTE_TAGS = ["main32", "lab60", "timesplit32", "timesplit_matrix", "matrix", "floe3", "floe12", "thr25_75", "thr45_90",
              "cost148", "cost1212", "cost124", "range2", "block1", "block6", "cvar050", "cvar090", "donor32",
              "dmi70", "dmi90", "dmi70water", "dmi70heavy", "dmi70amsr", "dmi50amsr", "dmi70donor"]
OTHER_RUNS = ["fusion_2*", "cv_probe_2*", "cv_finetune_2*", "cvrep_2*", "checks_2*"]
FILES = ["ps131_index/ps131_labeled.csv", "ps131_index/ps131_episodes_collocated.csv",
         "ps131_index/synop_report.txt", "ps131_index/collocation_report.txt", "ps131_index/ps131_summary.txt",
         "08_icecharts_2022/dmi_coverage.csv", "08_icecharts_2022/dmi_coverage.txt",
         "08_icecharts_2022/dmi_windows_ps131.npz",
         "drl_seeds_summary.txt", "reviewer_analyses_ps131.txt"]
ABS = re.compile(r"[A-Za-z]:\\(?:[^\\,\r\n]+\\)*?(?=03_camera_PS131)")


def latest(pattern):
    c = [d for d in sorted(glob.glob(os.path.join(ROOT, "runs", pattern))) if os.path.isdir(d)]
    return c[-1] if c else None


def main():
    dirs = [latest(f"routev3_{t}_2*") for t in ROUTE_TAGS] + [latest(p) for p in OTHER_RUNS]
    missing = [t for t, d in zip(ROUTE_TAGS + OTHER_RUNS, dirs) if d is None]
    if missing:
        print("not found:", missing)
    seeds = {}
    for d in sorted(glob.glob(os.path.join(ROOT, "runs", "drlv3_*"))):
        b = os.path.basename(d)
        if b.startswith("drlv3_smoke") or not os.path.exists(os.path.join(d, "test_results.csv")):
            continue
        seeds[b.split("_")[1] if b.split("_")[1].startswith("s") else "s0"] = d   # drlv3_<date> = seed 0
    dirs += list(seeds.values())
    for fig in ("article3_figs_en", "supplement_en"):
        if os.path.isdir(os.path.join(ROOT, "runs", fig)):
            dirs.append(os.path.join(ROOT, "runs", fig))
    files = [os.path.join(ROOT, f) for f in FILES if os.path.exists(os.path.join(ROOT, f))]
    for d in filter(None, dirs):
        files += [p for p in glob.glob(os.path.join(d, "*")) if os.path.isfile(p)]

    n_bytes = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in files:
            arc = os.path.relpath(p, ROOT).replace("\\", "/")
            if p.endswith(".csv"):
                with open(p, encoding="utf-8") as fh:
                    z.writestr(arc, ABS.sub("", fh.read()))
            else:
                z.write(p, arc)
            n_bytes += os.path.getsize(p)
            print(arc)
    print(f"\n{len(files)} files, {n_bytes / 1e6:.0f} MB -> {OUT} ({os.path.getsize(OUT) / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
