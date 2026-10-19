"""
Download of the PS131 ATWAICE shipborne camera frames (PANGAEA, doi:10.1594/PANGAEA.967580) and of the PS131
synoptic observations (PANGAEA, doi:10.1594/PANGAEA.987909) for article 3.

The paper uses 56 of the 665 hourly archives (about evenly spread over the cruise, 11.07-11.08.2022); their list is
data_manifest/ps131_camera_archives.txt. If this file exists, exactly these archives are downloaded; otherwise
PS131_NUM_FILES archives are chosen evenly over the PANGAEA list. Volume ~100-150 GB.

RUN:     python download_ps131_full.py
Proxy (optional): environment variable PS131_PROXY, e.g. socks5h://127.0.0.1:1080
OUTPUT:  03_camera_PS131_ATWAICE_full\\<archive>\\*.jpg , pangaea\\PS131_meteorology.tab , logs\\download_log.txt
"""

import sys
import subprocess
import zipfile
import time
from pathlib import Path

# ============================================================
# SETTINGS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
CAMERA_DIR = BASE_DIR / "03_camera_PS131_ATWAICE_full"
LOGS_DIR = BASE_DIR / "logs"

PS131_NUM_FILES = 60  # used only if data_manifest/ps131_camera_archives.txt is missing
MANIFEST = BASE_DIR / "data_manifest" / "ps131_camera_archives.txt"
METEO_DATASET_ID = "987909"
METEO_TAB = BASE_DIR / "pangaea" / "PS131_meteorology.tab"
PS131_DATASET_ID = "967580"

# ============================================================
# DEPENDENCIES
# ============================================================
def ensure_packages():
    required = ["requests", "tqdm"]
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            print(f"[setup] Installing {pkg}...")
            subprocess.run([sys.executable, "-m", "pip", "install", pkg, "--quiet"], check=True)

ensure_packages()
import requests
from tqdm import tqdm

# ============================================================
# PROXY (optional): environment variable PS131_PROXY
# ============================================================
import os
PROXY_URL = os.environ.get("PS131_PROXY")
PROXIES = {"http": PROXY_URL, "https": PROXY_URL} if PROXY_URL else None
BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Referer": f"https://doi.pangaea.de/10.1594/PANGAEA.967580",
}


# ============================================================
# UTILITIES
# ============================================================
def log(msg):
    print(msg)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOGS_DIR / "download_log.txt", "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def download_file(url, dest_path, chunk_size=1024 * 1024, retries=6):
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists():
        log(f"[skip] Already exists: {dest_path.name}")
        return True

    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    for attempt in range(1, retries + 1):
        resume_byte_pos = tmp_path.stat().st_size if tmp_path.exists() else 0
        headers = dict(BROWSER_HEADERS)
        if resume_byte_pos:
            headers["Range"] = f"bytes={resume_byte_pos}-"

        try:
            with requests.get(url, headers=headers, stream=True, timeout=60, proxies=PROXIES) as r:
                if r.status_code == 503:
                    wait_s = 75
                    log(f"[info] File is being staged from tape (attempt {attempt}/{retries}), waiting {wait_s} s: {dest_path.name}")
                    time.sleep(wait_s)
                    continue
                if r.status_code not in (200, 206):
                    log(f"[error] {url} -> HTTP {r.status_code}")
                    return False
                total = int(r.headers.get("content-length", 0)) + resume_byte_pos
                mode = "ab" if resume_byte_pos else "wb"
                with open(tmp_path, mode) as f, tqdm(
                    total=total, initial=resume_byte_pos, unit="B", unit_scale=True,
                    desc=dest_path.name[:40]
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            tmp_path.rename(dest_path)
            log(f"[ok] Downloaded: {dest_path}")
            return True
        except Exception as e:
            log(f"[error] Could not download {url}: {e}")
            if attempt < retries:
                time.sleep(5 * attempt)
                continue
            return False
    log(f"[error] Out of attempts for {dest_path.name} (the server keeps returning 503)")
    return False


def unzip_all(folder):
    folder = Path(folder)
    for zip_path in folder.glob("*.zip"):
        extract_to = folder / zip_path.stem
        if extract_to.exists():
            continue
        log(f"[unzip] {zip_path.name}")
        try:
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(extract_to)
        except Exception as e:
            log(f"[error] Could not unpack {zip_path.name}: {e}")


def get_all_filenames():
    """Gets the full list of available hourly archives from PANGAEA."""
    r = requests.get(
        f"https://doi.pangaea.de/10.1594/PANGAEA.{PS131_DATASET_ID}?format=textfile",
        timeout=30,
        proxies=PROXIES,
        headers=BROWSER_HEADERS,
    )
    filenames = []
    if r.status_code == 200:
        for line in r.text.splitlines():
            parts = line.split("\t")
            for p in parts:
                p = p.strip()
                if p.endswith(".zip") and p[0].isdigit():
                    filenames.append(p)
    return sorted(set(filenames))


def pick_spread_subset(filenames, n):
    """Chooses n files evenly spread over the whole list
    (for a variety of conditions rather than one continuous period)."""
    if len(filenames) <= n:
        return filenames
    step = len(filenames) / n
    return [filenames[int(i * step)] for i in range(n)]


# ============================================================
# PS131 ATWAICE camera archives
# ============================================================
def download_ps131_full():
    log("=== PS131 ATWAICE camera archives ===")
    CAMERA_DIR.mkdir(parents=True, exist_ok=True)

    base_url = f"https://download.pangaea.de/dataset/{PS131_DATASET_ID}/files"

    try:
        all_files = get_all_filenames()
        log(f"[info] Archives available: {len(all_files)}")
    except Exception as e:
        log(f"[error] Could not get the file list: {e}")
        return

    if not all_files:
        log("[error] The file list is empty, stopping")
        return

    if MANIFEST.exists():
        selected = [l.strip() for l in MANIFEST.read_text().splitlines() if l.strip()]
        missing = sorted(set(selected) - set(all_files))
        if missing:
            log(f"[warn] Not in the PANGAEA list: {missing}")
        log(f"[info] Archive list from {MANIFEST.name}: {len(selected)}")
    else:
        selected = pick_spread_subset(all_files, PS131_NUM_FILES)
        log(f"[info] Selected {len(selected)} files evenly spread over the whole cruise")

    for fname in selected:
        file_url = f"{base_url}/{fname}"
        download_file(file_url, CAMERA_DIR / fname)

    unzip_all(CAMERA_DIR)


def download_meteo():
    """PS131 synoptic observations (PANGAEA .tab text format) - labels for label_ps131_synop.py."""
    if METEO_TAB.exists():
        log(f"[skip] Already exists: {METEO_TAB}")
        return
    METEO_TAB.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(f"https://doi.pangaea.de/10.1594/PANGAEA.{METEO_DATASET_ID}?format=textfile",
                     timeout=120, proxies=PROXIES, headers=BROWSER_HEADERS)
    if r.status_code != 200:
        log(f"[error] PANGAEA {METEO_DATASET_ID}: HTTP {r.status_code}")
        return
    METEO_TAB.write_bytes(r.content)
    log(f"[ok] {METEO_TAB} ({len(r.content) / 1e6:.1f} MB)")


def main():
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    log("=" * 60)
    log("Download of PS131: camera frames and synoptic observations")
    log(f"Target folder: {BASE_DIR}")
    log("=" * 60)

    download_meteo()
    download_ps131_full()

    log("\n" + "=" * 60)
    log("DONE")
    log("=" * 60)
    log(f"Data in the folder: {CAMERA_DIR}")


if __name__ == "__main__":
    main()
