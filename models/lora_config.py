"""AutoLyrics — PEFT LoRA configuration builder."""

from __future__ import annotations

from typing import Any

from core.logging import get_logger
from peft import LoraConfig, PeftModel, TaskType, get_peft_model

logger = get_logger(__name__)


def build_lora_config(
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    bias: str = "none",
    target_modules: list[str] | None = None,
    modules_to_save: list[str] | None = None,
    task_type: str = "SEQ_2_SEQ_LM",
    apply_to_encoder: bool = False,
    apply_to_decoder: bool = True,
) -> LoraConfig:
    """Build a PEFT LoraConfig for Whisper.

    Args:
        r: LoRA rank.
        alpha: LoRA alpha scaling.
        dropout: LoRA dropout.
        target_modules: Attention projection layers to adapt.
        apply_to_encoder: Include encoder modules.
        apply_to_decoder: Include decoder modules.

    Returns:
        A :class:`LoraConfig` instance.
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "k_proj", "out_proj"]

    # Build module name filter to restrict to encoder/decoder
    module_prefixes = []
    if apply_to_decoder:
        module_prefixes.append("model.decoder")
    if apply_to_encoder:
        module_prefixes.append("model.encoder")

    # PEFT LoraConfig with target_modules supports regex-like patterns
    # We use target_modules directly; PEFT will match them in both encoder/decoder
    # unless we filter via layers_to_transform or custom logic
    tt = TaskType.SEQ_2_SEQ_LM if task_type == "SEQ_2_SEQ_LM" else TaskType.CAUSAL_LM

    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias=bias,
        target_modules=target_modules,
        modules_to_save=modules_to_save or [],
        task_type=tt,
    )

    logger.info(
        "LoRA config: r=%d, alpha=%d, dropout=%.2f, targets=%s, encoder=%s, decoder=%s",
        r,
        alpha,
        dropout,
        target_modules,
        apply_to_encoder,
        apply_to_decoder,
    )
    return config


def apply_lora(
    model: Any,
    lora_cfg: dict[str, Any],
) -> Any:
    """Apply LoRA adapters to a model based on config dict.

    Args:
        model: A WhisperForConditionalGeneration model.
        lora_cfg: Dict with LoRA parameters from YAML config.

    Returns:
        PEFT-wrapped model.
    """
    r = lora_cfg.get("r", 16)
    if r <= 0:
        logger.info("LoRA rank is 0; skipping adapter application (full fine-tune).")
        return model

    config = build_lora_config(
        r=r,
        alpha=lora_cfg.get("alpha", 32),
        dropout=lora_cfg.get("dropout", 0.05),
        bias=lora_cfg.get("bias", "none"),
        target_modules=lora_cfg.get("target_modules"),
        modules_to_save=lora_cfg.get("modules_to_save"),
        task_type=lora_cfg.get("task_type", "SEQ_2_SEQ_LM"),
        apply_to_encoder=lora_cfg.get("apply_to_encoder", False),
        apply_to_decoder=lora_cfg.get("apply_to_decoder", True),
    )

    apply_encoder = lora_cfg.get("apply_to_encoder", False)
    apply_decoder = lora_cfg.get("apply_to_decoder", True)

    # Freeze parts that shouldn't get LoRA
    if not apply_encoder:
        for name, param in model.named_parameters():
            if "model.encoder" in name:
                param.requires_grad = False

    peft_model = get_peft_model(model, config)

    # Re-freeze encoder if only decoder LoRA
    if not apply_encoder:
        for name, param in peft_model.named_parameters():
            if "model.encoder" in name and "lora_" not in name:
                param.requires_grad = False

    peft_model.print_trainable_parameters()
    return peft_model


def load_lora_checkpoint(model: Any, checkpoint_path: str) -> Any:
    """Load LoRA adapter weights from a checkpoint directory."""
    logger.info("Loading LoRA checkpoint from: %s", checkpoint_path)
    model = PeftModel.from_pretrained(model, checkpoint_path)
    model = model.merge_and_unload()
    logger.info("LoRA adapters merged into base model.")
    return model
