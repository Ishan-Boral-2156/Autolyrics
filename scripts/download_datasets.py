#!/usr/bin/env python
"""AutoLyrics — Download datasets."""

from __future__ import annotations

import argparse

from core.logging import get_logger, setup_logging
from data_pipeline.download import download_all, download_dataset, download_hf_dataset

setup_logging()
logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download AutoLyrics datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help="Dataset to download: nus48e | dali | jamendo | hf | all",
    )
    parser.add_argument("--output", type=str, default="./data/raw", help="Output directory")
    parser.add_argument(
        "--hf-dataset-id", type=str, default=None, help="HF dataset ID (for --dataset hf)"
    )
    args = parser.parse_args()

    if args.dataset == "all":
        download_all(args.output)
        logger.info("All datasets processed.")
    elif args.dataset == "hf":
        dataset_id = args.hf_dataset_id or "jdmoon/SingingDataset"
        download_hf_dataset(dataset_id)
    else:
        download_dataset(args.dataset, args.output)

    logger.info("Done.")


if __name__ == "__main__":
    main()
