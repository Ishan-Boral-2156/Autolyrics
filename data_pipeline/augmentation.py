"""AutoLyrics — Audio augmentation pipeline for singing data."""

from __future__ import annotations

import random
from typing import Any

import numpy as np


class SingingAugmentor:
    """Augmentation pipeline for singing audio with configurable transforms.

    Supports pitch shifting, time stretching, noise injection, and SpecAugment.
    All augmentations operate on numpy arrays at the waveform level.
    """

    def __init__(
        self,
        p_apply: float = 0.5,
        pitch_shift_semitones: tuple[float, float] = (-2.0, 2.0),
        time_stretch_factor: tuple[float, float] = (0.9, 1.1),
        add_noise_snr_db: tuple[float, float] = (10.0, 30.0),
        enabled: bool = True,
    ) -> None:
        self.p_apply = p_apply
        self.pitch_shift_semitones = pitch_shift_semitones
        self.time_stretch_factor = time_stretch_factor
        self.add_noise_snr_db = add_noise_snr_db
        self.enabled = enabled

    def __call__(self, audio: np.ndarray, sample_rate: int = 16_000) -> np.ndarray:
        if not self.enabled or random.random() > self.p_apply:
            return audio
        transforms = [self._pitch_shift, self._time_stretch, self._add_noise]
        chosen = random.choice(transforms)
        return chosen(audio, sample_rate)

    def _pitch_shift(self, audio: np.ndarray, sr: int) -> np.ndarray:
        try:
            import librosa

            semitones = random.uniform(*self.pitch_shift_semitones)
            return librosa.effects.pitch_shift(audio, sr=sr, n_steps=semitones)
        except ImportError:
            return audio

    def _time_stretch(self, audio: np.ndarray, sr: int) -> np.ndarray:
        try:
            import librosa

            factor = random.uniform(*self.time_stretch_factor)
            stretched = librosa.effects.time_stretch(audio, rate=factor)
            return stretched
        except ImportError:
            return audio

    def _add_noise(self, audio: np.ndarray, sr: int) -> np.ndarray:
        snr_db = random.uniform(*self.add_noise_snr_db)
        rms_signal = np.sqrt(np.mean(audio**2))
        if rms_signal == 0:
            return audio
        rms_noise = rms_signal / (10 ** (snr_db / 20.0))
        noise = np.random.normal(0, rms_noise, audio.shape)
        return (audio + noise).astype(audio.dtype)


class SpecAugment:
    """SpecAugment: frequency and time masking on spectrograms."""

    def __init__(
        self,
        time_mask_param: int = 40,
        freq_mask_param: int = 20,
        n_time_masks: int = 2,
        n_freq_masks: int = 2,
        enabled: bool = True,
    ) -> None:
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.n_time_masks = n_time_masks
        self.n_freq_masks = n_freq_masks
        self.enabled = enabled

    def __call__(self, spectrogram: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return spectrogram
        spec = spectrogram.copy()
        n_freq, n_time = spec.shape

        for _ in range(self.n_freq_masks):
            f = random.randint(0, min(self.freq_mask_param, n_freq - 1))
            f0 = random.randint(0, n_freq - f)
            spec[f0 : f0 + f, :] = 0.0

        for _ in range(self.n_time_masks):
            t = random.randint(0, min(self.time_mask_param, n_time - 1))
            t0 = random.randint(0, n_time - t)
            spec[:, t0 : t0 + t] = 0.0

        return spec


def build_augmentor(aug_cfg: dict[str, Any]) -> SingingAugmentor | None:
    """Build a :class:`SingingAugmentor` from a config dict."""
    if not aug_cfg.get("enabled", False):
        return None
    return SingingAugmentor(
        p_apply=aug_cfg.get("p_apply", 0.5),
        pitch_shift_semitones=tuple(aug_cfg.get("pitch_shift_semitones", [-2.0, 2.0])),
        time_stretch_factor=tuple(aug_cfg.get("time_stretch_factor", [0.9, 1.1])),
        add_noise_snr_db=tuple(aug_cfg.get("add_noise_snr_db", [10.0, 30.0])),
        enabled=True,
    )
