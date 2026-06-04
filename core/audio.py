"""AutoLyrics — Audio loading, resampling, normalisation, and feature extraction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

from core.constants import HOP_LENGTH, N_FFT, N_MELS, SAMPLE_RATE


def load_audio(
    path: str | Path, target_sr: int = SAMPLE_RATE, mono: bool = True
) -> tuple[torch.Tensor, int]:
    """Load audio, resample, optionally mix to mono. Returns (waveform, sr)."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data.T)
    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
        sr = target_sr
    return waveform, sr


def load_audio_segment(
    path: str | Path, start_s: float, end_s: float, target_sr: int = SAMPLE_RATE
) -> torch.Tensor:
    """Load a time-bounded segment of audio."""
    info = sf.info(str(path))
    sr = info.samplerate
    start_frame = int(start_s * sr)
    frames = int((end_s - start_s) * sr)
    data, _ = sf.read(str(path), start=start_frame, frames=frames, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data.T)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
    return waveform


def normalize_audio(
    waveform: torch.Tensor, method: str = "peak", target_db: float = -3.0
) -> torch.Tensor:
    """Normalize waveform amplitude. method: peak | rms | none."""
    if method == "none":
        return waveform
    if method == "peak":
        peak = waveform.abs().max()
        return waveform / peak if peak > 0 else waveform
    if method == "rms":
        rms = waveform.pow(2).mean().sqrt()
        if rms > 0:
            return waveform * (10 ** (target_db / 20.0) / rms)
    return waveform


def trim_silence(waveform: torch.Tensor, top_db: float = 25.0) -> torch.Tensor:
    """Trim leading/trailing silence using librosa."""
    try:
        import librosa

        trimmed, _ = librosa.effects.trim(waveform.squeeze(0).numpy(), top_db=top_db)
        return torch.from_numpy(trimmed).unsqueeze(0)
    except ImportError:
        return waveform


def compute_log_mel_spectrogram(
    waveform: torch.Tensor,
    sr: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
) -> torch.Tensor:
    """Log-Mel spectrogram matching Whisper feature extraction."""
    if waveform.dim() == 2:
        waveform = waveform.squeeze(0)
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=2.0
    )(waveform)
    log_mel = torch.clamp(mel, min=1e-10).log10()
    log_mel = torch.maximum(log_mel, log_mel.max() - 8.0)
    return (log_mel + 4.0) / 4.0


def pad_or_trim(waveform: torch.Tensor, length: int) -> torch.Tensor:
    """Pad or trim waveform to exactly *length* samples."""
    if waveform.shape[-1] > length:
        return waveform[..., :length]
    if waveform.shape[-1] < length:
        return torch.nn.functional.pad(waveform, (0, length - waveform.shape[-1]))
    return waveform


def get_audio_duration(path: str | Path) -> float:
    """Return duration in seconds."""
    info = sf.info(str(path))
    return info.frames / info.samplerate


def audio_to_numpy(waveform: torch.Tensor) -> np.ndarray:
    """Convert torch waveform to 1-D numpy array."""
    return waveform.squeeze().cpu().numpy()
