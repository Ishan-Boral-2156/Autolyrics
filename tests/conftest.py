"""AutoLyrics — Shared test fixtures."""

from __future__ import annotations

import numpy as np
import pytest
import torch


@pytest.fixture
def sample_audio_np() -> np.ndarray:
    """1-second mono audio at 16kHz."""
    rng = np.random.default_rng(42)
    return rng.standard_normal(16_000).astype(np.float32)


@pytest.fixture
def sample_audio_tensor() -> torch.Tensor:
    """1-second mono audio tensor at 16kHz, shape (1, 16000)."""
    torch.manual_seed(42)
    return torch.randn(1, 16_000)


@pytest.fixture
def sample_predictions() -> list[str]:
    return [
        "hello world",
        "this is a test",
        "singing in the rain",
        "you are my sunshine",
    ]


@pytest.fixture
def sample_references() -> list[str]:
    return [
        "hello world",
        "this is the test",
        "singing in the rain",
        "you are my sun shine",
    ]


@pytest.fixture
def tmp_audio_file(tmp_path, sample_audio_np) -> str:
    """Write a temporary WAV file and return its path."""
    import soundfile as sf

    path = tmp_path / "test_audio.wav"
    sf.write(str(path), sample_audio_np, 16_000)
    return str(path)
