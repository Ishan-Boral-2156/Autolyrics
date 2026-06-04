"""AutoLyrics — BitsAndBytes quantization configuration."""

from __future__ import annotations

from typing import Any

import torch
from core.logging import get_logger

logger = get_logger(__name__)


def build_quantization_config(
    enabled: bool = False,
    bits: int = 4,
    bnb_4bit_compute_dtype: str = "bfloat16",
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_use_double_quant: bool = True,
) -> Any | None:
    """Build a BitsAndBytesConfig for 4/8-bit model loading.

    Returns ``None`` if quantization is disabled or bitsandbytes is unavailable.
    """
    if not enabled:
        return None

    try:
        from transformers import BitsAndBytesConfig
    except ImportError:
        logger.warning("BitsAndBytesConfig not available; skipping quantization.")
        return None

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
    }
    compute_dtype = dtype_map.get(bnb_4bit_compute_dtype, torch.bfloat16)

    if bits == 4:
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
        )
        logger.info(
            "4-bit quantization enabled (type=%s, double_quant=%s)",
            bnb_4bit_quant_type,
            bnb_4bit_use_double_quant,
        )
    elif bits == 8:
        config = BitsAndBytesConfig(load_in_8bit=True)
        logger.info("8-bit quantization enabled.")
    else:
        logger.warning("Unsupported quantization bits=%d; skipping.", bits)
        return None

    return config


def build_quantization_from_cfg(cfg: dict[str, Any]) -> Any | None:
    """Build quantization config from a YAML config ``quantization`` section."""
    q = cfg.get("quantization", {})
    return build_quantization_config(
        enabled=q.get("enabled", False),
        bits=q.get("bits", 4),
        bnb_4bit_compute_dtype=q.get("bnb_4bit_compute_dtype", "bfloat16"),
        bnb_4bit_quant_type=q.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=q.get("bnb_4bit_use_double_quant", True),
    )
