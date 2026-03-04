"""Audio question answering metrics.

AQA answers are short spans of text, so they are scored with string-level
metrics: an exact match and a token-level F1 that tolerates small wording
differences.
"""

from __future__ import annotations

from collections import Counter

from diapason.registry import metrics


def _tokens(text: str) -> list[str]:
    """Split an answer into whitespace-delimited tokens."""
    return text.split()


@metrics.register("exact_match")
def exact_match(reference: str, prediction: str, **options: object) -> float:
    """Return ``1.0`` when the prediction equals the reference, else ``0.0``."""
    return 1.0 if prediction == reference else 0.0


@metrics.register("token_f1")
def token_f1(reference: str, prediction: str, **options: object) -> float:
    """Token-level F1 between prediction and reference.

    Args:
        reference: Ground-truth answer.
        prediction: Model answer to score.

    Returns:
        The harmonic mean of token precision and recall. Two empty answers
        score ``1.0``; an empty answer against a non-empty one scores ``0.0``.
    """
    ref_tokens = _tokens(reference)
    pred_tokens = _tokens(prediction)
    if not ref_tokens and not pred_tokens:
        return 1.0
    if not ref_tokens or not pred_tokens:
        return 0.0
    common = Counter(ref_tokens) & Counter(pred_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)
