"""Speech-recognition error-rate metrics.

Word error rate (WER) and character error rate (CER) both express the edit
distance between a hypothesis and a reference, normalised by the reference
length. They differ only in the token the edit distance is computed over.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from diapason.metrics._alignment import Alignment, align_sequences
from diapason.metrics._text import normalize_text
from diapason.registry import metrics

__all__ = ["cer", "normalize_text", "process_characters", "process_words", "wer"]


def _split_words(text: str) -> list[str]:
    """Split text into whitespace-delimited word tokens."""
    return text.split()


def _split_chars(text: str) -> list[str]:
    """Split text into character tokens, dropping whitespace."""
    return [char for char in text if not char.isspace()]


def process_words(reference: str, hypothesis: str) -> Alignment:
    """Align ``hypothesis`` against ``reference`` at the word level."""
    return align_sequences(_split_words(reference), _split_words(hypothesis))


def process_characters(reference: str, hypothesis: str) -> Alignment:
    """Align ``hypothesis`` against ``reference`` at the character level."""
    return align_sequences(_split_chars(reference), _split_chars(hypothesis))


def _best_across_references(
    scorer: Callable[[str, str], float],
    reference: str | Sequence[str],
    hypothesis: str,
) -> float:
    """Score against whichever reference yields the lowest (best) value."""
    references = [reference] if isinstance(reference, str) else list(reference)
    if not references:
        raise ValueError("at least one reference is required")
    return min(scorer(ref, hypothesis) for ref in references)


@metrics.register("wer")
def wer(
    reference: str | Sequence[str],
    hypothesis: str,
    *,
    normalize: bool = False,
    **options: object,
) -> float:
    """Word error rate.

    Args:
        reference: Ground-truth transcription, or several acceptable
            transcriptions. With several references the best (lowest) error
            rate is reported.
        hypothesis: Model transcription to score.
        normalize: When true, lowercase and strip punctuation before scoring.

    Returns:
        Edit distance over word tokens divided by the number of reference
        words. Lower is better; ``0.0`` is a perfect transcription.
    """

    def score_one(ref: str, hyp: str) -> float:
        return _wer_single(ref, hyp, normalize=normalize)

    return _best_across_references(score_one, reference, hypothesis)


def _wer_single(reference: str, hypothesis: str, *, normalize: bool) -> float:
    """Word error rate against a single reference transcription."""
    if normalize:
        reference = normalize_text(reference)
        hypothesis = normalize_text(hypothesis)
    alignment = process_words(reference, hypothesis)
    num_reference_words = len(_split_words(reference))
    if num_reference_words == 0:
        if hypothesis.strip():
            raise ValueError("cannot score a hypothesis against an empty reference")
        return 0.0
    return alignment.distance / num_reference_words


@metrics.register("cer")
def cer(
    reference: str | Sequence[str],
    hypothesis: str,
    *,
    normalize: bool = False,
    **options: object,
) -> float:
    """Character error rate.

    Args:
        reference: Ground-truth transcription, or several acceptable
            transcriptions. With several references the best (lowest) error
            rate is reported.
        hypothesis: Model transcription to score.
        normalize: When true, lowercase and strip punctuation before scoring.

    Returns:
        Edit distance over character tokens divided by the number of reference
        characters. Lower is better.
    """

    def score_one(ref: str, hyp: str) -> float:
        return _cer_single(ref, hyp, normalize=normalize)

    return _best_across_references(score_one, reference, hypothesis)


def _cer_single(reference: str, hypothesis: str, *, normalize: bool) -> float:
    """Character error rate against a single reference transcription."""
    if normalize:
        reference = normalize_text(reference)
        hypothesis = normalize_text(hypothesis)
    alignment = process_characters(reference, hypothesis)
    num_reference_chars = len(_split_chars(reference))
    if num_reference_chars == 0:
        if hypothesis.strip():
            raise ValueError("cannot score a hypothesis against an empty reference")
        return 0.0
    return alignment.distance / num_reference_chars
