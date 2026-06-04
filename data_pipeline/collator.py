"""AutoLyrics — Whisper-compatible data collator for seq2seq training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class WhisperDataCollator:
    """Data collator that pads input features and labels for Whisper seq2seq.

    - Input features are padded to the longest in the batch.
    - Labels are padded with ``-100`` so the loss ignores pad positions.
    """

    processor: Any  # WhisperProcessor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Remove BOS token if the tokenizer prepended one
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().item():
            labels = labels[:, 1:]

        return {
            "input_features": batch["input_features"],
            "labels": labels,
        }
