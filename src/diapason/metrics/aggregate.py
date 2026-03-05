"""Aggregation of metric values across examples and tasks.

Individual metrics score one example. The runner collects those per-example
values and reduces them here: first to a single number per task, then to a
single number for a whole suite.
"""

from __future__ import annotations

from collections.abc import Sequence


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean; an empty sequence aggregates to ``0.0``."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def aggregate_examples(values: Sequence[float]) -> float:
    """Collapse per-example metric values into a single task score.

    Args:
        values: One metric value per evaluated example.

    Returns:
        The mean of ``values``.
    """
    return mean(values)
