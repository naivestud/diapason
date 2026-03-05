"""Tests for the aggregation helpers."""

from __future__ import annotations

import pytest

from diapason.metrics.aggregate import aggregate_examples, aggregate_tasks, mean


def test_mean_of_values() -> None:
    assert mean([1.0, 2.0, 3.0]) == 2.0


def test_mean_of_empty_is_zero() -> None:
    assert mean([]) == 0.0


def test_aggregate_examples_is_mean() -> None:
    assert aggregate_examples([0.2, 0.4]) == pytest.approx(0.3)


def test_aggregate_tasks_macro_average() -> None:
    assert aggregate_tasks({"small": 0.5, "large": 1.0}) == pytest.approx(0.75)


def test_aggregate_tasks_empty_is_zero() -> None:
    assert aggregate_tasks({}) == 0.0
