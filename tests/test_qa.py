"""Tests for the audio question answering metrics."""

from __future__ import annotations

import pytest

from diapason.metrics.qa import exact_match, rouge_l, token_f1


def test_exact_match_identical() -> None:
    assert exact_match("a piano", "a piano") == 1.0


def test_exact_match_different() -> None:
    assert exact_match("a piano", "a guitar") == 0.0


def test_exact_match_normalize_ignores_case_and_punctuation() -> None:
    assert exact_match("A Piano!", "a piano", normalize=True) == 1.0


def test_token_f1_identical() -> None:
    assert token_f1("the dog barks", "the dog barks") == 1.0


def test_token_f1_partial_overlap() -> None:
    # precision = 2/2, recall = 2/3 -> F1 = 0.8
    assert token_f1("the dog barks", "the dog") == pytest.approx(0.8)


def test_token_f1_no_overlap_is_zero() -> None:
    assert token_f1("cat", "dog") == 0.0


def test_token_f1_both_empty_is_one() -> None:
    assert token_f1("", "") == 1.0


def test_token_f1_empty_prediction_is_zero() -> None:
    assert token_f1("an answer", "") == 0.0


def test_token_f1_normalize() -> None:
    assert token_f1("The Dog.", "the dog", normalize=True) == 1.0


def test_rouge_l_identical() -> None:
    assert rouge_l("the dog barks", "the dog barks") == 1.0


def test_rouge_l_partial_overlap() -> None:
    # LCS = 2 (dog barks); precision = 2/2, recall = 2/3 -> F1 = 0.8
    assert rouge_l("the dog barks", "dog barks") == pytest.approx(0.8)


def test_rouge_l_penalizes_wrong_order() -> None:
    # token F1 is perfect here, but the LCS is a single token
    assert token_f1("a b c", "c b a") == 1.0
    assert rouge_l("a b c", "c b a") == pytest.approx(1 / 3)


def test_rouge_l_no_overlap_is_zero() -> None:
    assert rouge_l("cat", "dog") == 0.0


def test_rouge_l_both_empty_is_one() -> None:
    assert rouge_l("", "") == 1.0


def test_rouge_l_normalize() -> None:
    assert rouge_l("The Dog.", "the dog", normalize=True) == 1.0
