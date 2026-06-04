"""AutoLyrics — Error analysis: substitution/insertion/deletion breakdown."""

from __future__ import annotations

from collections import Counter
from typing import Any

from core.logging import get_logger
from jiwer import process_words

from evaluation.metrics import normalize_text

logger = get_logger(__name__)


def analyze_errors(
    predictions: list[str],
    references: list[str],
    normalize: bool = True,
    top_k: int = 20,
) -> dict[str, Any]:
    """Perform detailed error analysis on prediction–reference pairs.

    Returns:
        Dict with total error counts, common substitution pairs,
        common insertions/deletions, and per-sample breakdowns.
    """
    if normalize:
        predictions = [normalize_text(p) for p in predictions]
        references = [normalize_text(r) for r in references]

    total_subs = 0
    total_ins = 0
    total_dels = 0
    sub_pairs: Counter[tuple[str, str]] = Counter()
    insertion_words: Counter[str] = Counter()
    deletion_words: Counter[str] = Counter()
    per_sample: list[dict[str, Any]] = []

    for pred, ref in zip(predictions, references):
        if not ref.strip():
            continue
        result = process_words(ref, pred)

        subs = result.substitutions
        ins = result.insertions
        dels = result.deletions
        total_subs += subs
        total_ins += ins
        total_dels += dels

        # Extract alignment details
        ref_words = ref.split()
        pred_words = pred.split()

        for chunk in result.alignments:
            for align in chunk:
                if align.type == "substitute":
                    r_word = (
                        ref_words[align.ref_start_idx]
                        if align.ref_start_idx < len(ref_words)
                        else ""
                    )
                    p_word = (
                        pred_words[align.hyp_start_idx]
                        if align.hyp_start_idx < len(pred_words)
                        else ""
                    )
                    if r_word and p_word:
                        sub_pairs[(r_word, p_word)] += 1
                elif align.type == "insert":
                    p_word = (
                        pred_words[align.hyp_start_idx]
                        if align.hyp_start_idx < len(pred_words)
                        else ""
                    )
                    if p_word:
                        insertion_words[p_word] += 1
                elif align.type == "delete":
                    r_word = (
                        ref_words[align.ref_start_idx]
                        if align.ref_start_idx < len(ref_words)
                        else ""
                    )
                    if r_word:
                        deletion_words[r_word] += 1

        per_sample.append(
            {
                "reference": ref,
                "prediction": pred,
                "substitutions": subs,
                "insertions": ins,
                "deletions": dels,
            }
        )

    return {
        "total_substitutions": total_subs,
        "total_insertions": total_ins,
        "total_deletions": total_dels,
        "total_errors": total_subs + total_ins + total_dels,
        "top_substitutions": sub_pairs.most_common(top_k),
        "top_insertions": insertion_words.most_common(top_k),
        "top_deletions": deletion_words.most_common(top_k),
        "per_sample": per_sample,
    }
