"""Speech-recognition error-rate metrics.

Word error rate (WER) and character error rate (CER) both express the edit
distance between a hypothesis and a reference, normalised by the reference
length. They differ only in the token the edit distance is computed over.
"""

from __future__ import annotations

import string

from diapason.metrics._alignment import Alignment, align_sequences
from diapason.registry import metrics

_PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)


def normalize_text(text: str) -> str:
    """Lowercase ``text``, strip punctuation and collapse runs of whitespace."""
    lowered = text.lower().translate(_PUNCTUATION_TABLE)
    return " ".join(lowered.split())


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


@metrics.register("wer")
def wer(reference: str, hypothesis: str, *, normalize: bool = False, **options: object) -> float:
    """Word error rate.

    Args:
        reference: Ground-truth transcription.
        hypothesis: Model transcription to score.
        normalize: When true, lowercase and strip punctuation before scoring.

    Returns:
        Edit distance over word tokens divided by the number of reference
        words. Lower is better; ``0.0`` is a perfect transcription.
    """
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
def cer(reference: str, hypothesis: str, *, normalize: bool = False, **options: object) -> float:
    """Character error rate.

    Args:
        reference: Ground-truth transcription.
        hypothesis: Model transcription to score.
        normalize: When true, lowercase and strip punctuation before scoring.

    Returns:
        Edit distance over character tokens divided by the number of reference
        characters. Lower is better.
    """
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
