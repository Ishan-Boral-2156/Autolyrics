"""WER / CER metrics + lyric-friendly normalisation.

We use ``jiwer`` for the numerical computation and Whisper's
``EnglishTextNormalizer`` (or a built-in fallback) for case folding,
punctuation stripping, and number expansion before scoring. Lyrics often
contain markers such as ``[Chorus]`` or ``(x2)`` which we explicitly
remove so that they do not penalise the model.
"""

from __future__ import annotations

import re
import string
import unicodedata
from collections.abc import Sequence
from typing import Any

from core.logging import get_logger

logger = get_logger(__name__)


_SECTION_MARKERS = re.compile(
    r"\[(?:[^\]]{0,40})\]|\((?:verse|chorus|bridge|outro|intro|x\s*\d+|repeat[^\)]*)\)",
    flags=re.IGNORECASE,
)
_PARENTHETICAL_REPEAT = re.compile(r"\(\s*x\s*\d+\s*\)", flags=re.IGNORECASE)
_MULTISPACE = re.compile(r"\s+")
_punct_chars = string.punctuation.replace("'", "")
_PUNCT_TABLE = str.maketrans(_punct_chars, " " * len(_punct_chars))


def _build_normalizer():
    """Return a callable ``str -> str``.

    Tries Whisper's English normalizer first; otherwise uses a simple
    fallback that handles the most common normalisation steps.
    """
    try:
        from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

        # A minimal English spelling-normalisation table is bundled but
        # not required — we pass an empty mapping if the bundled one isn't
        # discoverable.
        try:
            from transformers.models.whisper.tokenization_whisper import (
                _build_english_spelling_normalizer,
            )

            mapping = _build_english_spelling_normalizer({})
        except Exception:  # pragma: no cover
            mapping = {}
        return EnglishTextNormalizer(mapping)
    except Exception:  # pragma: no cover
        return None


_WHISPER_NORMALIZER = _build_normalizer()


def normalize_for_lyrics(text: str) -> str:
    """Normalise a transcript for fair WER/CER scoring of sung lyrics."""
    if text is None:
        return ""
    s = unicodedata.normalize("NFKC", str(text))
    # Strip Whisper special tokens that occasionally leak through.
    s = re.sub(r"<\|.*?\|>", " ", s)
    # Remove lyric structure markers like [Chorus] or (x2).
    s = _SECTION_MARKERS.sub(" ", s)
    s = _PARENTHETICAL_REPEAT.sub(" ", s)
    # Apply Whisper normalizer when available (handles numbers, contractions,
    # case folding, common-spelling table, punctuation, etc.).
    if _WHISPER_NORMALIZER is not None:
        try:
            s = _WHISPER_NORMALIZER(s)
        except Exception:  # pragma: no cover
            pass
    else:
        s = s.lower()
        s = s.translate(_PUNCT_TABLE)
    s = _MULTISPACE.sub(" ", s).strip()
    return s


def _safe_jiwer_imports():
    try:
        from jiwer import cer as jiwer_cer
        from jiwer import wer as jiwer_wer
    except ImportError as exc:  # pragma: no cover
        raise ImportError("jiwer is required. Install with: pip install jiwer") from exc
    return jiwer_wer, jiwer_cer


def compute_wer(
    predictions: Sequence[str], references: Sequence[str], normalise: bool = True
) -> float:
    """Compute corpus-level Word Error Rate."""
    jiwer_wer, _ = _safe_jiwer_imports()
    if normalise:
        predictions = [normalize_for_lyrics(p) for p in predictions]
        references = [normalize_for_lyrics(r) for r in references]
    pairs = [(p, r) for p, r in zip(predictions, references) if r]
    if not pairs:
        logger.warning("compute_wer: no non-empty references after normalisation")
        return 1.0
    pred, ref = zip(*pairs)
    return float(jiwer_wer(list(ref), list(pred)))


def compute_cer(
    predictions: Sequence[str], references: Sequence[str], normalise: bool = True
) -> float:
    """Compute corpus-level Character Error Rate."""
    _, jiwer_cer = _safe_jiwer_imports()
    if normalise:
        predictions = [normalize_for_lyrics(p) for p in predictions]
        references = [normalize_for_lyrics(r) for r in references]
    pairs = [(p, r) for p, r in zip(predictions, references) if r]
    if not pairs:
        logger.warning("compute_cer: no non-empty references after normalisation")
        return 1.0
    pred, ref = zip(*pairs)
    return float(jiwer_cer(list(ref), list(pred)))


def compute_wer_cer(
    predictions: Sequence[str], references: Sequence[str], normalise: bool = True
) -> dict[str, float]:
    """Compute WER and CER together (slightly more efficient than separate calls)."""
    if normalise:
        predictions = [normalize_for_lyrics(p) for p in predictions]
        references = [normalize_for_lyrics(r) for r in references]
    return {
        "wer": compute_wer(predictions, references, normalise=False),
        "cer": compute_cer(predictions, references, normalise=False),
    }


def compute_per_clip_wer(
    predictions: Sequence[str], references: Sequence[str], normalise: bool = True
) -> list[float]:
    """WER for each pair independently. Returns 1.0 if reference is empty."""
    jiwer_wer, _ = _safe_jiwer_imports()
    out: list[float] = []
    for p, r in zip(predictions, references):
        if normalise:
            p = normalize_for_lyrics(p)
            r = normalize_for_lyrics(r)
        if not r:
            out.append(1.0)
            continue
        try:
            out.append(float(jiwer_wer(r, p)))
        except Exception:  # pragma: no cover
            out.append(1.0)
    return out


def compute_per_clip_cer(
    predictions: Sequence[str], references: Sequence[str], normalise: bool = True
) -> list[float]:
    """CER for each pair independently. Returns 1.0 if reference is empty."""
    _, jiwer_cer = _safe_jiwer_imports()
    out: list[float] = []
    for p, r in zip(predictions, references):
        if normalise:
            p = normalize_for_lyrics(p)
            r = normalize_for_lyrics(r)
        if not r:
            out.append(1.0)
            continue
        try:
            out.append(float(jiwer_cer(r, p)))
        except Exception:  # pragma: no cover
            out.append(1.0)
    return out


def compute_per_sample_metrics(
    predictions: Sequence[str], references: Sequence[str], normalise: bool = True
) -> list[dict[str, Any]]:
    wer_list = compute_per_clip_wer(predictions, references, normalise)
    cer_list = compute_per_clip_cer(predictions, references, normalise)
    return [
        {"wer": w, "cer": c, "pred": p, "ref": r}
        for w, c, p, r in zip(wer_list, cer_list, predictions, references)
    ]


__all__ = [
    "compute_cer",
    "compute_per_clip_cer",
    "compute_per_clip_wer",
    "compute_per_sample_metrics",
    "compute_wer",
    "compute_wer_cer",
    "normalize_for_lyrics",
]
