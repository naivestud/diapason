"""Speech-recognition error-rate metrics.

Word error rate (WER) and character error rate (CER) both express the edit
distance between a hypothesis and a reference, normalised by the reference
length. They differ only in the token the edit distance is computed over.
"""

from __future__ import annotations

from diapason.metrics._align import Alignment, align_sequences
from diapason.registry import metrics


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
def wer(reference: str, hypothesis: str, **options: object) -> float:
    """Word error rate.

    Args:
        reference: Ground-truth transcription.
        hypothesis: Model transcription to score.

    Returns:
        Edit distance over word tokens divided by the number of reference
        words. Lower is better; ``0.0`` is a perfect transcription.
    """
    alignment = process_words(reference, hypothesis)
    num_reference_words = len(_split_words(reference))
    if num_reference_words == 0:
        return 0.0
    return alignment.distance / num_reference_words


@metrics.register("cer")
def cer(reference: str, hypothesis: str, **options: object) -> float:
    """Character error rate.

    Args:
        reference: Ground-truth transcription.
        hypothesis: Model transcription to score.

    Returns:
        Edit distance over character tokens divided by the number of reference
        characters. Lower is better.
    """
    alignment = process_characters(reference, hypothesis)
    num_reference_chars = len(_split_chars(reference))
    if num_reference_chars == 0:
        return 0.0
    return alignment.distance / num_reference_chars
