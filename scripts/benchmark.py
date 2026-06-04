#!/usr/bin/env python
"""AutoLyrics — Benchmark inference latency and VRAM usage."""

from __future__ import annotations

import argparse
from pathlib import Path

from core.config import config_to_dict, load_config
from core.device import resolve_device, resolve_dtype
from core.io import write_json
from core.logging import get_logger, setup_logging
from core.seed import seed_everything

setup_logging()
logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark AutoLyrics inference")
    parser.add_argument("--model-config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--audio-duration", type=float, default=10.0)
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--output", type=str, default="reports/benchmark.json")
    args = parser.parse_args()

    seed_everything(42)

    model_name = "openai/whisper-small"
    if args.model_config:
        cfg = load_config(args.model_config)
        cfg_dict = config_to_dict(cfg)
        model_name = cfg_dict.get("model", {}).get("hf_name", model_name)

    device = resolve_device("auto")

    from models.lora_config import load_lora_checkpoint
    from models.whisper_model import load_whisper_model, load_whisper_processor

    processor = load_whisper_processor(model_name)
    model = load_whisper_model(
        model_name, device=device, use_cache=True, dtype=resolve_dtype("float16")
    )

    if args.checkpoint and Path(args.checkpoint).exists():
        model = load_lora_checkpoint(model, args.checkpoint)
        model = model.to(device)

    model.eval()

    from evaluation.benchmarks import LatencyBenchmark

    benchmark = LatencyBenchmark(model=model, processor=processor, device=device)
    results = benchmark.run(
        audio_duration_s=args.audio_duration,
        batch_sizes=args.batch_sizes,
        num_runs=args.num_runs,
    )

    write_json(results, args.output)
    logger.info("Benchmark results saved to %s", args.output)


if __name__ == "__main__":
    main()
