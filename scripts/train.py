#!/usr/bin/env python
"""AutoLyrics — Training script for LoRA / QLoRA / full fine-tuning."""

from __future__ import annotations

import argparse
from pathlib import Path

from core.config import config_to_dict, load_config, save_config
from core.device import resolve_device, resolve_dtype
from core.io import ensure_dir, save_run_state
from core.logging import get_logger, setup_logging
from core.seed import seed_everything

setup_logging()
logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="AutoLyrics model training")
    parser.add_argument("--model-config", type=str, required=True)
    parser.add_argument("--data-config", type=str, required=True)
    parser.add_argument("--training-config", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--overrides", nargs="*", default=None)
    args = parser.parse_args()

    # Load merged config
    cfg = load_config(
        args.model_config, args.data_config, args.training_config, overrides=args.overrides
    )
    cfg_dict = config_to_dict(cfg)

    # Extract sections
    model_cfg = cfg_dict.get("model", {})
    data_cfg = cfg_dict.get("data", cfg_dict)
    training_cfg = cfg_dict.get("training", {})
    lora_cfg = cfg_dict.get("lora", {})
    quant_cfg = cfg_dict.get("quantization", {})
    project_cfg = cfg_dict.get("project", {})

    # Output directory
    output_dir = args.output_dir or training_cfg.get("output_dir", "./runs/default")
    training_cfg["output_dir"] = output_dir
    ensure_dir(output_dir)

    # Seed
    seed = project_cfg.get("seed", 42)
    seed_everything(seed)

    # Save config and run state
    save_config(cfg, Path(output_dir) / "config.yaml")
    save_run_state(output_dir, config=cfg_dict, seed=seed)

    # Device
    device = resolve_device(cfg_dict.get("device", {}).get("type", "auto"))
    logger.info("Device: %s", device)

    # Load processor
    from models.lora_config import apply_lora
    from models.quantization import build_quantization_from_cfg
    from models.whisper_model import (
        load_whisper_model,
        load_whisper_processor,
        prepare_model_for_training,
    )

    model_name = model_cfg.get("hf_name", "openai/whisper-small")
    processor = load_whisper_processor(
        model_name,
        language=model_cfg.get("language", "en"),
        task=model_cfg.get("task", "transcribe"),
    )

    # Quantization
    quant_config = build_quantization_from_cfg(cfg_dict)

    # Load model
    model = load_whisper_model(
        model_name,
        device=device,
        dtype=resolve_dtype(cfg_dict.get("device", {}).get("precision", "bf16")),
        attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
        quantization_config=quant_config,
    )
    model = prepare_model_for_training(model)

    # Apply LoRA if strategy is lora/qlora
    strategy = training_cfg.get("strategy", "lora")
    if strategy in ("lora", "qlora") and lora_cfg.get("r", 0) > 0:
        model = apply_lora(model, lora_cfg)

    # Load data
    from data_pipeline.augmentation import build_augmentor
    from data_pipeline.collator import WhisperDataCollator
    from data_pipeline.dataset import SingingDataset
    from data_pipeline.loaders import get_loader

    loader = get_loader(data_cfg.get("loader", data_cfg.get("name", "nus48e")), data_cfg)
    splits = loader.load_splits()

    aug_cfg = data_cfg.get("augmentation", {})
    augmentor = build_augmentor(aug_cfg)

    train_dataset = SingingDataset(
        clips=splits["train"],
        processor=processor,
        max_duration_s=cfg_dict.get("audio", {}).get("max_duration_s", 30.0),
        augment_fn=augmentor,
    )
    eval_dataset = SingingDataset(
        clips=splits["val"],
        processor=processor,
        max_duration_s=cfg_dict.get("audio", {}).get("max_duration_s", 30.0),
    )
    collator = WhisperDataCollator(processor=processor)

    logger.info("Train samples: %d | Val samples: %d", len(train_dataset), len(eval_dataset))

    # Build trainer
    from training.callbacks import build_callbacks
    from training.trainer import AutoLyricsTrainer

    callbacks = build_callbacks(training_cfg)
    trainer = AutoLyricsTrainer(
        model=model,
        processor=processor,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        training_cfg=training_cfg,
        callbacks=callbacks,
    )

    # Train
    metrics = trainer.train()
    logger.info("Training finished. Final metrics: %s", metrics)


if __name__ == "__main__":
    main()
