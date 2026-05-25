"""Tests for the edit-distance alignment helper."""

from __future__ import annotations

from diapason.metrics._alignment import align_sequences


def test_identical_sequences_have_no_edits() -> None:
    alignment = align_sequences(["the", "cat", "sat"], ["the", "cat", "sat"])
    assert alignment.distance == 0
    assert alignment.hits == 3
    assert alignment.substitutions == 0
    assert alignment.insertions == 0
    assert alignment.deletions == 0


def test_single_substitution() -> None:
    alignment = align_sequences(["the", "cat", "sat"], ["the", "dog", "sat"])
    assert alignment.substitutions == 1
    assert alignment.hits == 2
    assert alignment.distance == 1


def test_single_insertion() -> None:
    alignment = align_sequences(["the", "cat"], ["the", "big", "cat"])
    assert alignment.insertions == 1
    assert alignment.distance == 1


def test_single_deletion() -> None:
    alignment = align_sequences(["the", "big", "cat"], ["the", "cat"])
    assert alignment.deletions == 1
    assert alignment.distance == 1


def test_empty_hypothesis_deletes_everything() -> None:
    alignment = align_sequences(["a", "b", "c"], [])
    assert alignment.deletions == 3
    assert alignment.hits == 0
    assert alignment.distance == 3


def test_empty_reference_inserts_everything() -> None:
    alignment = align_sequences([], ["a", "b", "c"])
    assert alignment.insertions == 3
    assert alignment.distance == 3


def test_mixed_edits_are_counted() -> None:
    # "a b c" -> "x b c y": substitute a->x, insert y
    alignment = align_sequences(["a", "b", "c"], ["x", "b", "c", "y"])
    assert alignment.substitutions == 1
    assert alignment.insertions == 1
    assert alignment.deletions == 0
    assert alignment.hits == 2
    assert alignment.distance == 2


def test_distance_is_symmetric() -> None:
    pairs = [
        (["the", "cat", "sat"], ["the", "dog", "sat"]),
        (["a", "b"], ["b", "a"]),
        (["a"], ["a", "b", "c"]),
        ([], ["x", "y"]),
    ]
    for reference, hypothesis in pairs:
        forward = align_sequences(reference, hypothesis)
        backward = align_sequences(hypothesis, reference)
        assert forward.distance == backward.distance


def test_operation_counts_are_consistent() -> None:
    alignment = align_sequences(["a", "b", "c", "d"], ["a", "x", "y", "d", "e"])
    assert alignment.distance == (
        alignment.substitutions + alignment.insertions + alignment.deletions
    )
    assert alignment.hits == 4 - alignment.substitutions - alignment.deletions
