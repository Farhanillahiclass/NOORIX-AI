"""Small, dependency-free metrics for reproducible OCR evaluation."""
from __future__ import annotations


def normalize_text(text: str) -> str:
    """Normalize whitespace only; preserve script, punctuation and case."""
    return " ".join(str(text).split())


def edit_distance(reference: str, hypothesis: str) -> int:
    """Return Levenshtein edit distance using O(min(n, m)) memory."""
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for row, ref_char in enumerate(reference, start=1):
        current = [row]
        for column, hyp_char in enumerate(hypothesis, start=1):
            current.append(min(current[-1] + 1, previous[column] + 1, previous[column - 1] + (ref_char != hyp_char)))
        previous = current
    return previous[-1]


def character_error_counts(reference: str, hypothesis: str) -> tuple[int, int]:
    reference = normalize_text(reference).replace(" ", "")
    hypothesis = normalize_text(hypothesis).replace(" ", "")
    return edit_distance(reference, hypothesis), len(reference)


def word_error_counts(reference: str, hypothesis: str) -> tuple[int, int]:
    reference_words = normalize_text(reference).split()
    hypothesis_words = normalize_text(hypothesis).split()
    return edit_distance(reference_words, hypothesis_words), len(reference_words)


def error_rate(errors: int, units: int) -> float | None:
    return errors / units if units else None
