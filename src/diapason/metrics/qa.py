"""Audio question answering metrics.

AQA answers are short spans of text, so they are scored with string-level
metrics: an exact match and a token-level F1 that tolerates small wording
differences.
"""

from __future__ import annotations

from collections import Counter

from diapason.metrics._text import normalize_text
from diapason.registry import metrics


def _tokens(text: str) -> list[str]:
    """Split an answer into whitespace-delimited tokens."""
    return text.split()


def _longest_common_subsequence_length(left: list[str], right: list[str]) -> int:
    """Length of the longest token subsequence shared by both answers.

    Uses a rolling two-row dynamic program, so memory stays proportional to
    the shorter answer rather than the product of both lengths.
    """
    if len(right) < len(left):
        left, right = right, left
    previous = [0] * (len(left) + 1)
    for right_token in right:
        current = [0] * (len(left) + 1)
        for index, left_token in enumerate(left, start=1):
            if left_token == right_token:
                current[index] = previous[index - 1] + 1
            else:
                current[index] = max(previous[index], current[index - 1])
        previous = current
    return previous[len(left)]


@metrics.register("exact_match")
def exact_match(
    reference: str, prediction: str, *, normalize: bool = False, **options: object
) -> float:
    """Return ``1.0`` when the prediction equals the reference, else ``0.0``.

    Args:
        normalize: When true, lowercase and strip punctuation before comparing.
    """
    if normalize:
        reference = normalize_text(reference)
        prediction = normalize_text(prediction)
    return 1.0 if prediction == reference else 0.0


@metrics.register("token_f1")
def token_f1(
    reference: str, prediction: str, *, normalize: bool = False, **options: object
) -> float:
    """Token-level F1 between prediction and reference.

    Args:
        reference: Ground-truth answer.
        prediction: Model answer to score.
        normalize: When true, lowercase and strip punctuation before scoring.

    Returns:
        The harmonic mean of token precision and recall. Two empty answers
        score ``1.0``; an empty answer against a non-empty one scores ``0.0``.
    """
    if normalize:
        reference = normalize_text(reference)
        prediction = normalize_text(prediction)
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


@metrics.register("rouge_l")
def rouge_l(
    reference: str, prediction: str, *, normalize: bool = False, **options: object
) -> float:
    """ROUGE-L F-measure between prediction and reference.

    Unlike :func:`token_f1`, the score rewards tokens that appear in the same
    order: it is built on the longest common subsequence of tokens.

    Args:
        reference: Ground-truth answer.
        prediction: Model answer to score.
        normalize: When true, lowercase and strip punctuation before scoring.

    Returns:
        The F-measure of the longest common subsequence. Two empty answers
        score ``1.0``; an empty answer against a non-empty one scores ``0.0``.
    """
    if normalize:
        reference = normalize_text(reference)
        prediction = normalize_text(prediction)
    ref_tokens = _tokens(reference)
    pred_tokens = _tokens(prediction)
    if not ref_tokens and not pred_tokens:
        return 1.0
    if not ref_tokens or not pred_tokens:
        return 0.0
    lcs = _longest_common_subsequence_length(ref_tokens, pred_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)
