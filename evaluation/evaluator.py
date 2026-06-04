"""AutoLyrics — End-to-end evaluation loop."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from core.io import write_json
from core.logging import get_logger
from tqdm import tqdm

from evaluation.metrics import compute_per_sample_metrics, compute_wer_cer

logger = get_logger(__name__)


class Evaluator:
    """Run inference on a test set and compute aggregate + per-sample metrics."""

    def __init__(
        self,
        model: Any,
        processor: Any,
        device: torch.device | None = None,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.processor = processor
        self.device = device or torch.device("cpu")
        self.generation_kwargs = generation_kwargs or {
            "max_new_tokens": 225,
            "num_beams": 5,
            "no_repeat_ngram_size": 3,
            "length_penalty": 1.0,
        }
        self.model.eval()

    @torch.no_grad()
    def evaluate_dataset(
        self,
        dataset: Any,
        output_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Run evaluation over the full dataset.

        Args:
            dataset: A SingingDataset or similar with ``__getitem__`` returning dicts.
            output_path: If set, save results JSON here.

        Returns:
            Dict with aggregate metrics and per-sample details.
        """
        predictions: list[str] = []
        references: list[str] = []
        latencies: list[float] = []

        for i in tqdm(range(len(dataset)), desc="Evaluating"):
            sample = dataset[i]
            input_features = sample["input_features"]
            if input_features.dim() == 2:
                input_features = input_features.unsqueeze(0)
            input_features = input_features.to(self.device)

            start = time.perf_counter()
            predicted_ids = self.model.generate(input_features, **self.generation_kwargs)
            elapsed = time.perf_counter() - start
            latencies.append(elapsed)

            decoded = self.processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
            predictions.append(decoded[0].strip())
            references.append(sample.get("text", ""))

        agg = compute_wer_cer(predictions, references)
        per_sample = compute_per_sample_metrics(predictions, references)

        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        results = {
            "aggregate": agg,
            "avg_latency_s": round(avg_latency, 4),
            "num_samples": len(dataset),
            "per_sample": per_sample,
        }

        if output_path:
            write_json(results, output_path)
            logger.info("Results saved to %s", output_path)

        logger.info(
            "WER: %.2f%% | CER: %.2f%% | Avg latency: %.3fs", agg["wer"], agg["cer"], avg_latency
        )
        return results
