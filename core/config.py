"""AutoLyrics — OmegaConf config loader with Pydantic validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

# ---------------------------------------------------------------------------
#  Pydantic-style dataclass schemas (used for runtime validation)
# ---------------------------------------------------------------------------


@dataclass
class AudioConfig:
    sampling_rate: int = 16_000
    max_duration_s: float = 30.0
    min_duration_s: float = 0.5
    channels: int = 1
    feature_type: str = "log_mel"
    n_mels: int = 80


@dataclass
class ModelConfig:
    type: str = "whisper"
    hf_name: str = "openai/whisper-small"
    task: str = "transcribe"
    language: str = "en"
    attn_implementation: str = "sdpa"
    use_cache: bool = False
    freeze_feature_encoder: bool = False
    generation: dict[str, Any] = field(
        default_factory=lambda: {
            "max_new_tokens": 225,
            "num_beams": 5,
            "no_repeat_ngram_size": 3,
            "length_penalty": 1.0,
            "temperature": 0.0,
            "suppress_blank": True,
        }
    )


@dataclass
class QuantizationConfig:
    enabled: bool = False
    bits: int = 4
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class LoRAConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: str = "none"
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "v_proj",
            "k_proj",
            "out_proj",
        ]
    )
    modules_to_save: list[str] = field(default_factory=list)
    task_type: str = "SEQ_2_SEQ_LM"
    apply_to_encoder: bool = False
    apply_to_decoder: bool = True


@dataclass
class EarlyStoppingConfig:
    enabled: bool = True
    patience: int = 4
    threshold: float = 0.001


@dataclass
class TrainingConfig:
    strategy: str = "lora"
    output_dir: str = "./runs/default"
    num_train_epochs: int = 10
    max_steps: int = -1
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    bf16: bool = True
    fp16: bool = False
    optim: str = "adamw_torch"
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    group_by_length: bool = True
    length_column_name: str = "input_length"
    predict_with_generate: bool = True
    generation_max_length: int = 225
    generation_num_beams: int = 1
    evaluation_strategy: str = "steps"
    eval_steps: int = 200
    save_strategy: str = "steps"
    save_steps: int = 200
    save_total_limit: int = 3
    logging_steps: int = 25
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "wer"
    greater_is_better: bool = False
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    report_to: list[str] = field(default_factory=lambda: ["mlflow", "tensorboard"])
    remove_unused_columns: bool = False
    label_names: list[str] = field(default_factory=lambda: ["labels"])
    resume_from_checkpoint: str | None = None
    gradient_checkpointing: bool = False


@dataclass
class DataConfig:
    name: str = "nus48e"
    loader: str = "nus48e"
    root: str = "./data/raw/nus48e"
    cache_dir: str = "./.cache/autolyrics/nus48e"
    language: str = "en"


@dataclass
class InferenceConfig:
    batch_size: int = 1
    chunk_length_s: float = 30.0
    stride_length_s: float = 5.0
    return_timestamps: bool = False
    return_segments: bool = False
    return_word_offsets: bool = False
    num_beams: int = 5
    temperature: float = 0.0
    no_repeat_ngram_size: int = 3
    length_penalty: float = 1.0
    task: str = "transcribe"
    language: str = "en"
    normalize_output: bool = True
    remove_punctuation: bool = False
    lowercase: bool = False
    torch_dtype: str = "float16"


@dataclass
class PostprocessingConfig:
    collapse_repetitions: bool = True
    strip_filler_tokens: bool = True
    filler_tokens: list[str] = field(default_factory=lambda: ["uh", "um", "ah"])


# ---------------------------------------------------------------------------
#  Config loading
# ---------------------------------------------------------------------------


def load_config(*yaml_paths: str | Path, overrides: list[str] | None = None) -> DictConfig:
    """Load and merge one or more YAML config files with optional CLI overrides.

    Args:
        yaml_paths: Paths to YAML files.  Later files override earlier ones.
        overrides: Dotlist-style overrides, e.g. ``["training.lr=3e-4"]``.

    Returns:
        A merged :class:`DictConfig`.
    """
    cfgs: list[DictConfig] = []
    base_path = Path("configuration/base.yaml")
    if base_path.exists() and base_path not in [Path(p) for p in yaml_paths]:
        cfgs.append(OmegaConf.load(str(base_path)))  # type: ignore[arg-type]
    for p in yaml_paths:
        p = Path(p)
        if p.exists():
            cfgs.append(OmegaConf.load(str(p)))  # type: ignore[arg-type]
    if not cfgs:
        cfg = OmegaConf.create({})
    else:
        cfg = OmegaConf.merge(*cfgs)
    if overrides:
        cli_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, cli_cfg)
    OmegaConf.resolve(cfg)
    return cfg  # type: ignore[return-value]


def load_configs_for_run(
    model_config: str | Path,
    data_config: str | Path,
    training_config: str | Path | None = None,
    overrides: list[str] | None = None,
) -> DictConfig:
    """Convenience wrapper that loads model, data, and training configs."""
    paths = [model_config, data_config]
    if training_config is not None:
        paths.append(training_config)
    return load_config(*paths, overrides=overrides)


def config_to_dict(cfg: DictConfig) -> dict[str, Any]:
    """Convert a DictConfig to a plain Python dict (recursively)."""
    return OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)  # type: ignore[return-value]


def save_config(cfg: DictConfig, path: str | Path) -> Path:
    """Save a DictConfig to a YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, str(path))
    return path
