"""AutoLyrics — Whisper model loading with SDPA attention and cache control."""

from __future__ import annotations

from typing import Any

import torch
from core.logging import get_logger
from transformers import (
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)

logger = get_logger(__name__)


def load_whisper_processor(
    model_name: str = "openai/whisper-small",
    language: str = "en",
    task: str = "transcribe",
) -> WhisperProcessor:
    """Load and configure a WhisperProcessor (feature extractor + tokenizer)."""
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    tokenizer = WhisperTokenizer.from_pretrained(model_name, language=language, task=task)
    processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    return processor


def load_whisper_model(
    model_name: str = "openai/whisper-small",
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    attn_implementation: str = "sdpa",
    use_cache: bool = False,
    quantization_config: Any = None,
) -> WhisperForConditionalGeneration:
    """Load a Whisper model with optional quantization and attention implementation.

    Args:
        model_name: HuggingFace model identifier.
        device: Target device. If quantization is used, set to ``"auto"`` internally.
        dtype: Compute dtype (e.g. ``torch.float16``).
        attn_implementation: ``"sdpa"`` for memory-efficient attention.
        use_cache: Disable during training, enable for inference.
        quantization_config: Optional BitsAndBytesConfig for 4/8-bit loading.

    Returns:
        A :class:`WhisperForConditionalGeneration` model.
    """
    load_kwargs: dict[str, Any] = {
        "attn_implementation": attn_implementation,
    }

    if quantization_config is not None:
        load_kwargs["quantization_config"] = quantization_config
        load_kwargs["device_map"] = "auto"
    elif device is not None:
        load_kwargs["device_map"] = None

    if dtype is not None and quantization_config is None:
        load_kwargs["torch_dtype"] = dtype

    logger.info("Loading Whisper model: %s (attn=%s)", model_name, attn_implementation)

    try:
        model = WhisperForConditionalGeneration.from_pretrained(model_name, **load_kwargs)
    except Exception:
        # Fall back without SDPA if the model doesn't support it
        load_kwargs.pop("attn_implementation", None)
        logger.warning("SDPA not supported, falling back to default attention.")
        model = WhisperForConditionalGeneration.from_pretrained(model_name, **load_kwargs)

    model.config.use_cache = use_cache
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    if device is not None and quantization_config is None:
        model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "Model loaded: %s | Total params: %.1fM | Trainable: %.1fM",
        model_name,
        total / 1e6,
        trainable / 1e6,
    )
    return model


def prepare_model_for_training(
    model: WhisperForConditionalGeneration,
    freeze_encoder: bool = False,
) -> WhisperForConditionalGeneration:
    """Prepare a Whisper model for training (disable cache, optional encoder freeze)."""
    model.config.use_cache = False
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    if freeze_encoder:
        for param in model.model.encoder.parameters():
            param.requires_grad = False
        logger.info("Encoder parameters frozen.")

    model.train()
    return model
