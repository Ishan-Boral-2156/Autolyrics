"""AutoLyrics — Latency and VRAM benchmarking."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from core.device import empty_cache, get_vram_info
from core.logging import get_logger

logger = get_logger(__name__)


class LatencyBenchmark:
    """Measure inference latency and VRAM usage across batch sizes."""

    def __init__(self, model: Any, processor: Any, device: torch.device | None = None) -> None:
        self.model = model
        self.processor = processor
        self.device = device or torch.device("cpu")
        self.model.eval()

    @torch.no_grad()
    def run(
        self,
        audio_duration_s: float = 10.0,
        batch_sizes: list[int] | None = None,
        num_runs: int = 5,
        warmup_runs: int = 2,
    ) -> dict[str, Any]:
        if batch_sizes is None:
            batch_sizes = [1, 4, 8]

        sample_len = int(audio_duration_s * 16_000)
        dummy_audio = np.random.randn(sample_len).astype(np.float32)

        results: dict[str, Any] = {"audio_duration_s": audio_duration_s, "batch_results": []}

        for bs in batch_sizes:
            empty_cache()
            inputs = self.processor.feature_extractor(
                [dummy_audio] * bs, sampling_rate=16_000, return_tensors="pt"
            )
            input_features = inputs.input_features.to(self.device)

            # Warmup
            for _ in range(warmup_runs):
                self.model.generate(input_features, max_new_tokens=50)

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            latencies = []
            for _ in range(num_runs):
                start = time.perf_counter()
                self.model.generate(input_features, max_new_tokens=225)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                latencies.append(time.perf_counter() - start)

            vram = get_vram_info()
            avg_lat = np.mean(latencies)
            rtf = avg_lat / (audio_duration_s * bs)

            entry = {
                "batch_size": bs,
                "avg_latency_s": round(float(avg_lat), 4),
                "std_latency_s": round(float(np.std(latencies)), 4),
                "rtf": round(float(rtf), 4),
                "vram_allocated_gb": vram.get("allocated_gb", 0),
                "vram_total_gb": vram.get("total_gb", 0),
            }
            results["batch_results"].append(entry)
            logger.info(
                "BS=%d | Latency=%.3fs | RTF=%.3f | VRAM=%.2fGB",
                bs,
                avg_lat,
                rtf,
                vram.get("allocated_gb", 0),
            )

        return results
