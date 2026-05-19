"""Sound event detection metrics.

Sound event detection (SED) asks a model to report *what* sound happened and
*when*. Predictions and references are therefore lists of labelled time
intervals rather than strings, and scoring compares the two sets of intervals.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from diapason.registry import metrics


@dataclass(frozen=True)
class Event:
    """A labelled sound event spanning ``[onset, offset]`` seconds.

    Predicted events may carry a confidence ``score``; reference events leave
    it as ``None``.
    """

    onset: float
    offset: float
    label: str
    score: float | None = None

    def __post_init__(self) -> None:
        if self.offset < self.onset:
            raise ValueError(
                f"event '{self.label}' has offset {self.offset} before onset {self.onset}"
            )


def parse_events(raw: Sequence[Mapping[str, Any]]) -> tuple[Event, ...]:
    """Build :class:`Event` objects from manifest dictionaries.

    Args:
        raw: Mappings with ``onset``, ``offset`` and ``label`` keys.

    Returns:
        A tuple of events in the order given.

    Raises:
        ValueError: If a mapping is missing a required key.
    """
    events = []
    for index, item in enumerate(raw):
        try:
            onset = float(item["onset"])
            offset = float(item["offset"])
            label = str(item["label"])
        except KeyError as exc:
            raise ValueError(f"event {index} is missing key {exc}") from exc
        raw_score = item.get("score")
        score = None if raw_score is None else float(raw_score)
        events.append(Event(onset=onset, offset=offset, label=label, score=score))
    return tuple(events)


@dataclass(frozen=True)
class EventScore:
    """Precision, recall and F1 aggregated over every event label."""

    precision: float
    recall: float
    f1: float


def _group_by_label(events: Sequence[Event]) -> dict[str, list[Event]]:
    """Group events by their label, preserving the order within each label."""
    by_label: dict[str, list[Event]] = {}
    for event in events:
        by_label.setdefault(event.label, []).append(event)
    return by_label


def _count_matches(refs: Sequence[Event], preds: Sequence[Event], collar: float) -> int:
    """Count how many predicted events match a reference event of the same label.

    Predictions are walked in onset order and each reference event is matched
    at most once, so a duplicated prediction cannot score twice.
    """
    matched: set[int] = set()
    true_positives = 0
    for pred in sorted(preds, key=lambda event: event.onset):
        for idx, ref in enumerate(refs):
            if idx in matched:
                continue
            onset_ok = abs(pred.onset - ref.onset) <= collar
            offset_ok = abs(pred.offset - ref.offset) <= collar
            if onset_ok and offset_ok:
                matched.add(idx)
                true_positives += 1
                break
    return true_positives


def score_events(
    reference: Sequence[Event],
    prediction: Sequence[Event],
    *,
    collar: float = 0.0,
) -> EventScore:
    """Match predicted events to reference events and compute P/R/F1.

    A predicted event counts as a true positive when a reference event with the
    same label has both its onset and offset within ``collar`` seconds. Each
    reference event can be matched at most once.

    Args:
        reference: Ground-truth events.
        prediction: Predicted events.
        collar: Tolerance in seconds applied to onset and offset.

    Returns:
        The aggregated :class:`EventScore`.
    """
    ref_by_label = _group_by_label(reference)
    pred_by_label = _group_by_label(prediction)

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    for label in set(ref_by_label) | set(pred_by_label):
        refs = ref_by_label.get(label, [])
        preds = pred_by_label.get(label, [])
        label_tp = _count_matches(refs, preds, collar)
        true_positives += label_tp
        false_positives += len(preds) - label_tp
        false_negatives += len(refs) - label_tp

    precision = _ratio(true_positives, true_positives + false_positives)
    recall = _ratio(true_positives, true_positives + false_negatives)
    f1 = _ratio(2 * precision * recall, precision + recall)
    return EventScore(precision=precision, recall=recall, f1=f1)


def _ratio(numerator: float, denominator: float) -> float:
    """Return ``numerator / denominator``, treating ``0/0`` as ``0.0``."""
    return numerator / denominator if denominator else 0.0


@metrics.register("event_f1")
def event_based_f1(
    reference: Sequence[Event],
    prediction: Sequence[Event],
    *,
    collar: float = 0.0,
    **options: object,
) -> float:
    """Event-based F1 score with a collar tolerance.

    Registered under the metric name ``event_f1``. Higher is better; ``1.0``
    means every reference event was matched with no spurious predictions.
    """
    return score_events(reference, prediction, collar=collar).f1


@metrics.register("event_precision")
def event_based_precision(
    reference: Sequence[Event],
    prediction: Sequence[Event],
    *,
    collar: float = 0.0,
    **options: object,
) -> float:
    """Event-based precision with a collar tolerance.

    Registered under the metric name ``event_precision``. The fraction of
    predicted events that match a reference event; higher is better.
    """
    return score_events(reference, prediction, collar=collar).precision


@metrics.register("event_recall")
def event_based_recall(
    reference: Sequence[Event],
    prediction: Sequence[Event],
    *,
    collar: float = 0.0,
    **options: object,
) -> float:
    """Event-based recall with a collar tolerance.

    Registered under the metric name ``event_recall``. The fraction of
    reference events that were matched by a prediction; higher is better.
    """
    return score_events(reference, prediction, collar=collar).recall


@metrics.register("psds_lite")
def psds_lite(
    reference: Sequence[Event],
    prediction: Sequence[Event],
    *,
    collar: float = 0.0,
    num_thresholds: int = 10,
    **options: object,
) -> float:
    """Approximate PSDS by averaging event F1 across detection thresholds.

    Predicted events with a higher confidence ``score`` survive higher
    thresholds, so sweeping the threshold traces out a precision/recall curve
    and averaging its F1 approximates the area-based PSDS. Predictions without
    a score are treated as maximally confident.

    Args:
        reference: Ground-truth events.
        prediction: Predicted events, ideally with confidence scores.
        collar: Tolerance in seconds applied to onset and offset.
        num_thresholds: Number of threshold steps between 0 and 1.

    Returns:
        The mean event F1 across the swept thresholds.
    """
    if num_thresholds <= 0:
        raise ValueError("num_thresholds must be positive")
    f1_scores = []
    for step in range(num_thresholds + 1):
        threshold = step / num_thresholds
        kept = [
            event
            for event in prediction
            if (event.score if event.score is not None else 1.0) >= threshold
        ]
        f1_scores.append(score_events(reference, kept, collar=collar).f1)
    return sum(f1_scores) / len(f1_scores)
