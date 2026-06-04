#!/usr/bin/env python
"""AutoLyrics — Evaluate a model (baseline or fine-tuned) on a test set."""

from __future__ import annotations

import argparse
from pathlib import Path

from core.config import config_to_dict, load_config
from core.device import resolve_device, resolve_dtype
from core.logging import get_logger, setup_logging
from core.seed import seed_everything

setup_logging()
logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AutoLyrics model")
    parser.add_argument("--model-config", type=str, default=None, help="Model config YAML")
    parser.add_argument("--data-config", type=str, required=True, help="Data config YAML")
    parser.add_argument("--checkpoint", type=str, default=None, help="Fine-tuned checkpoint path")
    parser.add_argument("--output", type=str, default="reports/eval_results.json")
    parser.add_argument("--overrides", nargs="*", default=None)
    args = parser.parse_args()

    configs = [args.data_config]
    if args.model_config:
        configs.insert(0, args.model_config)
    cfg = load_config(*configs, overrides=args.overrides)
    cfg_dict = config_to_dict(cfg)

    seed_everything(cfg_dict.get("project", {}).get("seed", 42))

    model_cfg = cfg_dict.get("model", {})
    data_cfg = cfg_dict.get("data", cfg_dict)
    model_name = model_cfg.get("hf_name", "openai/whisper-small")
    device = resolve_device(cfg_dict.get("device", {}).get("type", "auto"))

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

    from data_pipeline.dataset import SingingDataset
    from data_pipeline.loaders import get_loader

    loader = get_loader(data_cfg.get("loader", data_cfg.get("name", "nus48e")), data_cfg)
    splits = loader.load_splits()
    test_dataset = SingingDataset(
        clips=splits.get("test", splits.get("val", [])),
        processor=processor,
    )

    logger.info("Evaluating on %d test samples", len(test_dataset))

    from evaluation.evaluator import Evaluator

    gen_kwargs = model_cfg.get(
        "generation",
        {
            "max_new_tokens": 225,
            "num_beams": 5,
            "no_repeat_ngram_size": 3,
            "length_penalty": 1.0,
        },
    )

    evaluator = Evaluator(
        model=model, processor=processor, device=device, generation_kwargs=gen_kwargs
    )
    results = evaluator.evaluate_dataset(test_dataset, output_path=args.output)

    agg = results.get("aggregate", {})
    logger.info("Final WER: %.2f%% | CER: %.2f%%", agg.get("wer", 0), agg.get("cer", 0))


if __name__ == "__main__":
    main()
