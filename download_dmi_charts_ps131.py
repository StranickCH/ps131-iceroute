"""
DMI (Greenland) and MET Norway (Svalbard) ice charts for the dates of cruise PS131 - alternative reference for
the routing simulation (article 3). Source: Copernicus Marine, product
SEAICE_ARC_SEAICE_L4_NRT_OBSERVATIONS_011_002 (1 km grid, WMO concentration, DMI also SIGRID-3 ice type).

ONCE:   pip install copernicusmarine      then      copernicusmarine login   (marine.copernicus.eu account)
RUN:    python download_dmi_charts_ps131.py --dry-run     # only the list and the number of files (seconds)
        python download_dmi_charts_ps131.py               # download
OUTPUT: 08_icecharts_2022\\<dataset>\\*.nc , download_log.txt
"""
import time
import argparse
from pathlib import Path
from datetime import datetime

HERE = Path(__file__).resolve().parent
DATASETS = {
    "dmi_regional": "cmems_obs-si_arc_physic_nrt_1km-grl_P1D-irr",
    "dmi_overview": "cmems_obs-si_arc_physic_nrt_1km-grl_P1WT3D-m",
    "metno_svalbard": "cmems_obs_si_arc_phy_nrt_1km_svb_P1D-irr",
}
# 10.07.2022 - 12.08.2022 (cruise 11.07-11.08 plus one day on each side)
DATE_RE = r".*2022(07(1\d|2\d|3[01])|08(0\d|1[0-2]))\d*.*\.nc$"


def log(msg, f):
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    f.write(line + "\n")
    f.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=str(HERE / "08_icecharts_2022"))
    args = ap.parse_args()
    import copernicusmarine as cm

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "download_log.txt", "a", encoding="utf-8") as f:
        log(f"copernicusmarine {getattr(cm, '__version__', '?')} | dry-run={args.dry_run}", f)
        for name, ds in DATASETS.items():
            t0 = time.time()
            try:
                r = cm.get(dataset_id=ds, regex=DATE_RE, output_directory=out / name, no_directories=True,
                           skip_existing=True, dry_run=args.dry_run, disable_progress_bar=True)
                files = r.files or []
                log(f"{name:15s} {ds}: files {r.number_of_files_to_download or len(files)}, "
                    f"~{(r.total_size or 0):.0f} MB, {time.time() - t0:.0f} s", f)
                for x in files[:3]:
                    log(f"    example: {x.filename}", f)
            except Exception as e:  # a dataset may not hold the year 2022 - then just report it
                log(f"{name:15s} {ds}: ERROR {type(e).__name__}: {e}", f)
        log("DONE", f)


if __name__ == "__main__":
    main()
