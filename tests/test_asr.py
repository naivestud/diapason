"""Tests for the word and character error rate metrics."""

from __future__ import annotations

import pytest

from diapason.metrics.asr import cer, normalize_text, wer


def test_wer_perfect_match_is_zero() -> None:
    assert wer("the cat sat", "the cat sat") == 0.0


def test_wer_single_substitution() -> None:
    assert wer("the cat sat", "the dog sat") == pytest.approx(1 / 3)


def test_wer_single_deletion() -> None:
    assert wer("the cat sat", "the cat") == pytest.approx(1 / 3)


def test_wer_single_insertion() -> None:
    # one insertion over a two-word reference
    assert wer("the cat", "the big cat") == pytest.approx(0.5)


def test_wer_can_exceed_one() -> None:
    assert wer("a b", "c d e") == pytest.approx(1.5)


def test_wer_normalize_ignores_case_and_punctuation() -> None:
    assert wer("Hello, World!", "hello world", normalize=True) == 0.0


def test_wer_without_normalize_is_case_sensitive() -> None:
    assert wer("Hello", "hello") == pytest.approx(1.0)


def test_cer_perfect_match_is_zero() -> None:
    assert cer("abc", "abc") == 0.0


def test_cer_single_substitution() -> None:
    assert cer("abc", "axc") == pytest.approx(1 / 3)


def test_cer_ignores_whitespace() -> None:
    assert cer("a b c", "abc") == 0.0


def test_normalize_text_lowercases_and_strips_punctuation() -> None:
    assert normalize_text("  Hello,   World! ") == "hello world"


def test_empty_reference_and_hypothesis_is_zero() -> None:
    assert wer("", "") == 0.0
    assert cer("", "") == 0.0


def test_empty_reference_with_hypothesis_raises() -> None:
    with pytest.raises(ValueError, match="empty reference"):
        wer("", "hello")
    with pytest.raises(ValueError, match="empty reference"):
        cer("", "hello")


def test_wer_multi_reference_picks_perfect_match() -> None:
    assert wer(["the cat sat", "a dog ran"], "a dog ran") == 0.0


def test_wer_multi_reference_reports_best_error() -> None:
    # Against "the cat sat" the hypothesis misses one word; against "the cat"
    # it is exact, so the best (zero) error is reported.
    assert wer(["the cat sat", "the cat"], "the cat") == 0.0


def test_wer_empty_reference_list_raises() -> None:
    with pytest.raises(ValueError, match="at least one reference"):
        wer([], "hello")


def test_cer_multi_reference() -> None:
    assert cer(["abc", "xyz"], "xyz") == 0.0
