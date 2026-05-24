"""Audio question answering metrics.

AQA answers are short spans of text, so they are scored with string-level
metrics: an exact match, a token-level F1 that tolerates small wording
differences, an order-sensitive ROUGE-L, and sentence-level BLEU.
"""

from __future__ import annotations

import math
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


def _ngrams(tokens: list[str], order: int) -> Counter[tuple[str, ...]]:
    """Count the ``order``-grams of a token list."""
    return Counter(tuple(tokens[index : index + order]) for index in range(len(tokens) - order + 1))


def _modified_precision(ref_tokens: list[str], pred_tokens: list[str], order: int) -> float | None:
    """Clipped n-gram precision of ``pred_tokens`` against ``ref_tokens``.

    Returns ``None`` when the prediction has no n-grams of this order, which
    happens when it is shorter than ``order`` tokens; callers skip such orders.
    """
    pred_ngrams = _ngrams(pred_tokens, order)
    if not pred_ngrams:
        return None
    ref_ngrams = _ngrams(ref_tokens, order)
    clipped = sum(min(count, ref_ngrams[ngram]) for ngram, count in pred_ngrams.items())
    return clipped / sum(pred_ngrams.values())


@metrics.register("bleu")
def bleu(
    reference: str,
    prediction: str,
    *,
    max_order: int = 4,
    normalize: bool = False,
    **options: object,
) -> float:
    """Sentence-level BLEU between prediction and reference.

    Combines modified n-gram precision up to ``max_order`` with a brevity
    penalty that discourages very short answers. N-gram orders longer than the
    prediction are skipped, so short answers are still scored.

    Args:
        reference: Ground-truth answer.
        prediction: Model answer to score.
        max_order: Largest n-gram order to use; must be positive.
        normalize: When true, lowercase and strip punctuation before scoring.

    Returns:
        BLEU in ``[0, 1]``. Two empty answers score ``1.0``; an empty answer
        against a non-empty reference scores ``0.0``.
    """
    if max_order <= 0:
        raise ValueError(f"max_order must be positive, got {max_order}")
    if normalize:
        reference = normalize_text(reference)
        prediction = normalize_text(prediction)
    ref_tokens = _tokens(reference)
    pred_tokens = _tokens(prediction)
    if not ref_tokens and not pred_tokens:
        return 1.0
    if not ref_tokens or not pred_tokens:
        return 0.0

    precisions = [
        precision
        for order in range(1, max_order + 1)
        if (precision := _modified_precision(ref_tokens, pred_tokens, order)) is not None
    ]
    if not precisions or any(precision == 0.0 for precision in precisions):
        return 0.0
    log_average = sum(math.log(precision) for precision in precisions) / len(precisions)

    brevity_penalty = 1.0
    if len(pred_tokens) < len(ref_tokens):
        brevity_penalty = math.exp(1 - len(ref_tokens) / len(pred_tokens))
    return brevity_penalty * math.exp(log_average)
