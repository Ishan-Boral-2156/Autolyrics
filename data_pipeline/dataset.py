"""AutoLyrics — Unified dataset schema and torch Dataset implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch.utils.data import Dataset


@dataclass
class SingingClip:
    """Unified record for a singing audio segment across all dataset sources."""

    audio_path: str
    text: str
    start: float = 0.0
    end: float = 0.0
    dataset_name: str = ""
    split: str = "train"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end - self.start if self.end > self.start else 0.0


class SingingDataset(Dataset):
    """PyTorch Dataset wrapping a list of :class:`SingingClip` objects.

    Each ``__getitem__`` returns a dict with keys consumed by the Whisper
    feature extractor and tokenizer (populated by the preprocessing step).
    """

    def __init__(
        self,
        clips: list[SingingClip],
        processor: Any = None,
        max_duration_s: float = 30.0,
        augment_fn: Any = None,
        sample_rate: int = 16_000,
    ) -> None:
        self.clips = clips
        self.processor = processor
        self.max_duration_s = max_duration_s
        self.augment_fn = augment_fn
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        clip = self.clips[idx]
        from core.audio import load_audio, load_audio_segment, normalize_audio

        if clip.start > 0 or clip.end > 0:
            waveform = load_audio_segment(clip.audio_path, clip.start, clip.end, self.sample_rate)
        else:
            waveform, _ = load_audio(clip.audio_path, self.sample_rate)

        waveform = normalize_audio(waveform, method="peak")

        max_samples = int(self.max_duration_s * self.sample_rate)
        if waveform.shape[-1] > max_samples:
            waveform = waveform[..., :max_samples]

        audio_np = waveform.squeeze(0).numpy()

        if self.augment_fn is not None:
            audio_np = self.augment_fn(audio_np, sample_rate=self.sample_rate)

        if self.processor is not None:
            inputs = self.processor.feature_extractor(
                audio_np, sampling_rate=self.sample_rate, return_tensors="pt"
            )
            input_features = inputs.input_features.squeeze(0)
            labels = self.processor.tokenizer(clip.text).input_ids
        else:
            input_features = torch.from_numpy(audio_np)
            labels = []

        return {
            "input_features": input_features,
            "labels": labels,
            "input_length": waveform.shape[-1],
            "text": clip.text,
            "audio_path": clip.audio_path,
        }
