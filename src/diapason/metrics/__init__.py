"""Scoring metrics for the task families supported by diapason.

Importing this package imports every metric submodule, which runs their
``@metrics.register`` decorators and makes the built-in metrics available to
suite configs by name.
"""

from __future__ import annotations

from diapason.metrics.aggregate import (
    AGGREGATION_METHODS,
    aggregate_examples,
    aggregate_tasks,
    mean,
    median,
)
from diapason.metrics.asr import cer, normalize_text, wer
from diapason.metrics.qa import exact_match, token_f1
from diapason.metrics.sed import (
    Event,
    event_based_f1,
    parse_events,
    psds_lite,
    score_events,
)

__all__ = [
    "AGGREGATION_METHODS",
    "Event",
    "aggregate_examples",
    "aggregate_tasks",
    "cer",
    "event_based_f1",
    "exact_match",
    "mean",
    "median",
    "normalize_text",
    "parse_events",
    "psds_lite",
    "score_events",
    "token_f1",
    "wer",
]
