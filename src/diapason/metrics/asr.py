"""Speech-recognition error-rate metrics.

Word error rate (WER) and character error rate (CER) both express the edit
distance between a hypothesis and a reference, normalised by the reference
length. They differ only in the token the edit distance is computed over.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from diapason.metrics._alignment import Alignment, align_sequences
from diapason.metrics._text import normalize_text
from diapason.registry import metrics

__all__ = [
    "WerBreakdown",
    "cer",
    "normalize_text",
    "process_characters",
    "process_words",
    "wer",
    "wer_breakdown",
]


def _split_words(text: str) -> list[str]:
    """Split text into whitespace-delimited word tokens."""
    return text.split()


def _split_chars(text: str) -> list[str]:
    """Split text into character tokens, dropping whitespace."""
    return [char for char in text if not char.isspace()]


@dataclass(frozen=True)
class WerBreakdown:
    """Word error rate split by edit operation.

    Each rate is the count of that operation divided by the number of
    reference words, so ``total`` equals the word error rate.

    Attributes:
        substitutions: Rate of reference words replaced by a different word.
        insertions: Rate of extra hypothesis words.
        deletions: Rate of reference words missing from the hypothesis.
    """

    substitutions: float
    insertions: float
    deletions: float

    @property
    def total(self) -> float:
        """The overall word error rate these rates decompose into."""
        return self.substitutions + self.insertions + self.deletions


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


def wer_breakdown(reference: str, hypothesis: str, *, normalize: bool = False) -> WerBreakdown:
    """Decompose the word error rate into substitution, insertion and deletion.

    Useful for diagnosing *how* a model errs rather than just how much: a
    deletion-heavy score suggests dropped words, an insertion-heavy one
    suggests hallucinated words.

    Args:
        reference: Ground-truth transcription; must contain at least one word.
        hypothesis: Model transcription to score.
        normalize: When true, lowercase and strip punctuation before scoring.

    Returns:
        A :class:`WerBreakdown` whose ``total`` equals :func:`wer`.

    Raises:
        ValueError: If the reference contains no words.
    """
    if normalize:
        reference = normalize_text(reference)
        hypothesis = normalize_text(hypothesis)
    alignment = process_words(reference, hypothesis)
    num_reference_words = len(_split_words(reference))
    if num_reference_words == 0:
        raise ValueError("cannot score a hypothesis against an empty reference")
    return WerBreakdown(
        substitutions=alignment.substitutions / num_reference_words,
        insertions=alignment.insertions / num_reference_words,
        deletions=alignment.deletions / num_reference_words,
    )


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
