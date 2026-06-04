"""AutoLyrics — Visualization helpers (matplotlib/seaborn)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")


def plot_waveform(
    waveform: np.ndarray,
    sr: int = 16_000,
    title: str = "Waveform",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot a 1-D waveform."""
    fig, ax = plt.subplots(figsize=(12, 3))
    t = np.arange(len(waveform)) / sr
    ax.plot(t, waveform, linewidth=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    return fig


def plot_spectrogram(
    spec: np.ndarray, title: str = "Log-Mel Spectrogram", save_path: str | Path | None = None
) -> plt.Figure:
    """Plot a 2-D spectrogram (n_mels x T)."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.imshow(spec, aspect="auto", origin="lower", interpolation="nearest")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Mel Bin")
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    return fig


def plot_training_curves(
    history: dict[str, list[float]],
    title: str = "Training Curves",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot loss/metric curves from training history."""
    fig, axes = plt.subplots(1, len(history), figsize=(6 * len(history), 4))
    if len(history) == 1:
        axes = [axes]
    for ax, (name, values) in zip(axes, history.items()):
        ax.plot(values, marker="o", markersize=3)
        ax.set_title(name)
        ax.set_xlabel("Step")
        ax.set_ylabel(name)
    fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    return fig


def plot_comparison_bar(
    metrics: dict[str, dict[str, float]],
    metric_name: str = "wer",
    title: str = "Model Comparison",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Bar chart comparing a single metric across multiple models."""
    models = list(metrics.keys())
    values = [metrics[m].get(metric_name, 0) for m in models]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = sns.color_palette("viridis", len(models))
    bars = ax.bar(models, values, color=colors)
    ax.set_ylabel(metric_name.upper())
    ax.set_title(title)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    fig.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    return fig


def plot_error_distribution(
    errors: list[dict[str, Any]], save_path: str | Path | None = None
) -> plt.Figure:
    """Plot distribution of substitution/insertion/deletion errors."""
    subs = sum(e.get("substitutions", 0) for e in errors)
    ins = sum(e.get("insertions", 0) for e in errors)
    dels = sum(e.get("deletions", 0) for e in errors)
    fig, ax = plt.subplots(figsize=(6, 6))
    labels = ["Substitutions", "Insertions", "Deletions"]
    sizes = [subs, ins, dels]
    colors = ["#ff6b6b", "#feca57", "#48dbfb"]
    ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=140)
    ax.set_title("Error Type Distribution")
    fig.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    return fig
