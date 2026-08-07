"""Tests for the sound event detection metrics."""

from __future__ import annotations

import pytest

from diapason.metrics.sed import (
    Event,
    event_based_f1,
    parse_events,
    psds_lite,
    score_events,
)


def test_event_rejects_offset_before_onset() -> None:
    with pytest.raises(ValueError, match="before onset"):
        Event(onset=1.0, offset=0.5, label="dog")


def test_parse_events_builds_tuple() -> None:
    raw = [
        {"onset": 0.0, "offset": 1.0, "label": "dog"},
        {"onset": 2.5, "offset": 3.0, "label": "cat"},
    ]
    events = parse_events(raw)
    assert len(events) == 2
    assert events[0].label == "dog"
    assert events[1].onset == 2.5


def test_parse_events_missing_key_raises() -> None:
    with pytest.raises(ValueError, match="missing key"):
        parse_events([{"onset": 0.0, "offset": 1.0}])


def test_perfect_detection_scores_one() -> None:
    ref = [Event(0.0, 1.0, "dog"), Event(2.0, 3.0, "cat")]
    pred = [Event(0.0, 1.0, "dog"), Event(2.0, 3.0, "cat")]
    assert event_based_f1(ref, pred) == 1.0


def test_disjoint_predictions_score_zero() -> None:
    ref = [Event(0.0, 1.0, "dog")]
    pred = [Event(5.0, 6.0, "cat")]
    assert event_based_f1(ref, pred) == 0.0


def test_collar_controls_tolerance() -> None:
    ref = [Event(0.0, 1.0, "dog")]
    pred = [Event(0.15, 1.1, "dog")]
    assert event_based_f1(ref, pred, collar=0.0) == 0.0
    assert event_based_f1(ref, pred, collar=0.2) == 1.0


def test_partial_match_precision_recall() -> None:
    ref = [Event(0.0, 1.0, "dog"), Event(2.0, 3.0, "cat")]
    pred = [Event(0.0, 1.0, "dog")]
    score = score_events(ref, pred)
    assert score.precision == 1.0
    assert score.recall == 0.5
    assert score.f1 == pytest.approx(2 / 3)


def test_false_positive_lowers_precision() -> None:
    ref = [Event(0.0, 1.0, "dog")]
    pred = [Event(0.0, 1.0, "dog"), Event(4.0, 5.0, "dog")]
    score = score_events(ref, pred)
    assert score.precision == 0.5
    assert score.recall == 1.0


def test_psds_lite_sweeps_thresholds() -> None:
    ref = [Event(0.0, 1.0, "dog")]
    pred = [Event(0.0, 1.0, "dog", score=0.9)]
    # With 10 steps the prediction survives thresholds 0.0..0.9 (F1 = 1) and is
    # dropped at 1.0 (F1 = 0), giving 10/11.
    assert psds_lite(ref, pred) == pytest.approx(10 / 11)


def test_psds_lite_without_predictions_is_zero() -> None:
    ref = [Event(0.0, 1.0, "dog")]
    assert psds_lite(ref, []) == 0.0


def test_psds_lite_rejects_nonpositive_thresholds() -> None:
    with pytest.raises(ValueError, match="num_thresholds"):
        psds_lite([], [], num_thresholds=0)


def test_parse_events_reads_optional_score() -> None:
    events = parse_events([{"onset": 0.0, "offset": 1.0, "label": "dog", "score": 0.7}])
    assert events[0].score == 0.7
