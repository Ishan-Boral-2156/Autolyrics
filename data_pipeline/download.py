"""AutoLyrics — Dataset downloading utilities."""

from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path

import requests
from core.logging import get_logger
from tqdm import tqdm

logger = get_logger(__name__)

# Known download metadata (URLs may require manual download due to licensing)
_DATASET_INFO = {
    "nus48e": {
        "description": "NUS Sung and Spoken Lyrics Corpus (48 English songs)",
        "url": None,  # Requires manual download from NUS website
        "instructions": (
            "Download from https://smcnus.comp.nus.edu.sg/nus-48e-sung-and-spoken-lyrics-corpus/\n"
            "Extract to: data/raw/nus48e/"
        ),
    },
    "dali": {
        "description": "DALI — Dataset of Aligned Lyrics and Melodies (5000+ songs)",
        "url": None,
        "instructions": (
            "1. Request access at https://github.com/gMusic/DALI\n"
            "2. pip install DALI-dataset\n"
            "3. Use the DALI Python API to export annotations to data/raw/dali/annotations/\n"
            "4. Place corresponding audio files in data/raw/dali/audio/"
        ),
    },
    "jamendo": {
        "description": "Jamendo Lyrics Dataset — polyphonic music with lyric annotations",
        "url": "https://github.com/f90/jamendolyrics/archive/refs/heads/master.zip",
        "instructions": "Auto-downloadable from GitHub.",
    },
}


def download_file(url: str, dest: Path, chunk_size: int = 8192) -> Path:
    """Download a file from *url* to *dest* with a progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with (
        open(dest, "wb") as f,
        tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as pbar,
    ):
        for chunk in resp.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            pbar.update(len(chunk))
    return dest


def extract_archive(archive_path: Path, dest_dir: Path) -> None:
    """Extract a .zip or .tar.gz archive."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(dest_dir)
    elif archive_path.name.endswith(".tar.gz") or archive_path.name.endswith(".tgz"):
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(dest_dir)
    else:
        logger.warning("Unknown archive format: %s", archive_path)


def download_dataset(name: str, output_dir: str | Path = "./data/raw") -> None:
    """Download and extract a dataset by name."""
    output_dir = Path(output_dir)
    info = _DATASET_INFO.get(name.lower())
    if info is None:
        logger.error("Unknown dataset: %s. Available: %s", name, list(_DATASET_INFO))
        return

    dest = output_dir / name
    if dest.exists() and any(dest.iterdir()):
        logger.info("Dataset %s already exists at %s, skipping.", name, dest)
        return

    url = info.get("url")
    if url is None:
        logger.info("Dataset %s requires manual download:", name)
        logger.info(info["instructions"])
        return

    logger.info("Downloading %s from %s", name, url)
    archive_name = url.split("/")[-1]
    archive_path = output_dir / archive_name
    download_file(url, archive_path)
    extract_archive(archive_path, dest)
    archive_path.unlink(missing_ok=True)
    logger.info("Dataset %s extracted to %s", name, dest)


def download_all(output_dir: str | Path = "./data/raw") -> None:
    """Download all known datasets."""
    for name in _DATASET_INFO:
        download_dataset(name, output_dir)


def download_hf_dataset(dataset_id: str, cache_dir: str = "./.cache/autolyrics/hf") -> None:
    """Pre-download a HuggingFace dataset."""
    from datasets import load_dataset

    logger.info("Downloading HF dataset: %s", dataset_id)
    load_dataset(dataset_id, cache_dir=cache_dir, trust_remote_code=True)
    logger.info("HF dataset %s cached at %s", dataset_id, cache_dir)
