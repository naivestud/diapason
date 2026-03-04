"""Sound event detection metrics.

Sound event detection (SED) asks a model to report *what* sound happened and
*when*. Predictions and references are therefore lists of labelled time
intervals rather than strings, and scoring compares the two sets of intervals.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


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


def parse_events(raw: Sequence[Mapping[str, object]]) -> tuple[Event, ...]:
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
