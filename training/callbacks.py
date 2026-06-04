"""AutoLyrics — Training callbacks: early stopping, VRAM logging, checkpointing."""

from __future__ import annotations

from typing import Any

from core.device import get_vram_info
from core.logging import get_logger
from transformers import (
    EarlyStoppingCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

logger = get_logger(__name__)


class VRAMLoggingCallback(TrainerCallback):
    """Log GPU VRAM usage at each logging step."""

    def on_log(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any
    ) -> None:
        vram = get_vram_info()
        if vram:
            logger.info(
                "VRAM: %.2f GB allocated / %.2f GB total (%.1f%% used) on %s",
                vram["allocated_gb"],
                vram["total_gb"],
                (vram["allocated_gb"] / vram["total_gb"]) * 100 if vram["total_gb"] > 0 else 0,
                vram.get("device_name", "unknown"),
            )


class BestModelLogCallback(TrainerCallback):
    """Log whenever a new best model is found."""

    def __init__(self) -> None:
        self._best_metric: float | None = None

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if metrics is None:
            return
        metric_name = args.metric_for_best_model or "wer"
        key = f"eval_{metric_name}"
        val = metrics.get(key)
        if val is None:
            return
        if self._best_metric is None or val < self._best_metric:
            self._best_metric = val
            logger.info("New best %s: %.4f (step %d)", metric_name, val, state.global_step)


class TrainingProgressCallback(TrainerCallback):
    """Log high-level training progress at epoch boundaries."""

    def on_epoch_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any
    ) -> None:
        epoch = state.epoch or 0
        logger.info(
            "Epoch %.0f/%d complete | Global step: %d",
            epoch,
            args.num_train_epochs,
            state.global_step,
        )


def build_callbacks(training_cfg: dict[str, Any]) -> list[TrainerCallback]:
    """Build a list of callbacks from training config."""
    callbacks: list[TrainerCallback] = [
        VRAMLoggingCallback(),
        BestModelLogCallback(),
        TrainingProgressCallback(),
    ]

    es_cfg = training_cfg.get("early_stopping", {})
    if es_cfg.get("enabled", False):
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=es_cfg.get("patience", 4),
                early_stopping_threshold=es_cfg.get("threshold", 0.001),
            )
        )
        logger.info(
            "Early stopping enabled: patience=%d, threshold=%.4f",
            es_cfg.get("patience", 4),
            es_cfg.get("threshold", 0.001),
        )

    return callbacks
