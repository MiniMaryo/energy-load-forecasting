"""Download UCI Electricity Load Diagrams dataset."""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ZIP_URL = "https://archive.ics.uci.edu/static/public/321/electricity+load+diagrams+20112014.zip"
ZIP_PATH = DATA_DIR / "LD2011_2014.zip"
TXT_PATH = DATA_DIR / "LD2011_2014.txt"


def download(url: str, dest: Path, chunk_size: int = 1 << 20) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest}")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = 100 * downloaded / total
                        print(f"\r  {pct:5.1f}% ({downloaded // (1<<20)} MB)", end="", flush=True)
    print("\nDone.")


def extract(zip_path: Path, dest_dir: Path) -> Path:
    print(f"Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    candidates = list(dest_dir.glob("**/LD2011_2014.txt"))
    if not candidates:
        raise FileNotFoundError("LD2011_2014.txt not found in archive")
    txt = candidates[0]
    if txt != TXT_PATH:
        txt.replace(TXT_PATH)
    print(f"Data ready: {TXT_PATH}")
    return TXT_PATH


def main() -> int:
    if TXT_PATH.exists():
        print(f"Already exists: {TXT_PATH}")
        return 0
    if not ZIP_PATH.exists():
        download(ZIP_URL, ZIP_PATH)
    extract(ZIP_PATH, DATA_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
