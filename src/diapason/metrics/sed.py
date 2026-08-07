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
    """A labelled sound event spanning ``[onset, offset]`` seconds."""

    onset: float
    offset: float
    label: str

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
        events.append(Event(onset=onset, offset=offset, label=label))
    return tuple(events)


@dataclass(frozen=True)
class EventScore:
    """Precision, recall and F1 aggregated over every event label."""

    precision: float
    recall: float
    f1: float


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
    ref_by_label: dict[str, list[Event]] = {}
    for event in reference:
        ref_by_label.setdefault(event.label, []).append(event)
    pred_by_label: dict[str, list[Event]] = {}
    for event in prediction:
        pred_by_label.setdefault(event.label, []).append(event)

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    for label in set(ref_by_label) | set(pred_by_label):
        refs = ref_by_label.get(label, [])
        preds = sorted(pred_by_label.get(label, []), key=lambda event: event.onset)
        matched: set[int] = set()
        label_tp = 0
        for pred in preds:
            for idx, ref in enumerate(refs):
                if idx in matched:
                    continue
                onset_ok = abs(pred.onset - ref.onset) <= collar
                offset_ok = abs(pred.offset - ref.offset) <= collar
                if onset_ok and offset_ok:
                    matched.add(idx)
                    label_tp += 1
                    break
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
    """Event-based F1 score with a collar tolerance. Higher is better."""
    return score_events(reference, prediction, collar=collar).f1
