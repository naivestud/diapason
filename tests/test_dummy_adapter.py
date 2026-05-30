"""Tests for the deterministic dummy adapter."""

from __future__ import annotations

import pytest

from diapason.adapters import DummyAdapter
from diapason.datasets.manifest import Example
from diapason.registry import adapters


def _example(text: str) -> Example:
    return Example(id="ex", reference_text=text)


def test_echoes_reference_by_default() -> None:
    adapter = DummyAdapter()
    assert adapter.predict(_example("the cat sat")) == "the cat sat"


def test_corruption_drops_words_deterministically() -> None:
    adapter = DummyAdapter(corruption=0.5)
    assert adapter.predict(_example("one two three four")) == "one three"


def test_empty_reference_returns_empty() -> None:
    adapter = DummyAdapter()
    assert adapter.predict(_example("")) == ""


def test_corruption_out_of_range_raises() -> None:
    with pytest.raises(ValueError, match="corruption"):
        DummyAdapter(corruption=1.5)


def test_dummy_is_registered() -> None:
    assert "dummy" in adapters.names()


def test_seeded_corruption_is_reproducible() -> None:
    first = DummyAdapter(corruption=0.5, seed=7)
    second = DummyAdapter(corruption=0.5, seed=7)
    example = _example("the quick brown fox jumps over the lazy dog")
    assert first.predict(example) == second.predict(example)


def test_seeded_corruption_varies_with_seed() -> None:
    example = _example("the quick brown fox jumps over the lazy dog")
    outputs = {DummyAdapter(corruption=0.25, seed=seed).predict(example) for seed in range(5)}
    assert len(outputs) > 1


def test_seeded_corruption_drops_expected_count() -> None:
    adapter = DummyAdapter(corruption=0.25, seed=1)
    words = ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
    kept = adapter.predict(_example(" ".join(words)))
    assert len(kept.split()) == len(words) - round(0.25 * len(words))


def test_seed_is_ignored_without_corruption() -> None:
    adapter = DummyAdapter(corruption=0.0, seed=3)
    assert adapter.predict(_example("the cat sat")) == "the cat sat"
