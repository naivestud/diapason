"""Aggregation of metric values across examples and tasks.

Individual metrics score one example. The runner collects those per-example
values and reduces them here: first to a single number per task, then to a
single number for a whole suite.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence


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


def aggregate_tasks(task_scores: Mapping[str, float]) -> float:
    """Collapse per-task scores into one suite score.

    This is a macro average: every task contributes equally regardless of how
    many examples it contains, which keeps large tasks from drowning out small
    ones.

    Args:
        task_scores: Mapping of task name to its aggregated score.

    Returns:
        The mean of the task scores.
    """
    return mean(list(task_scores.values()))
