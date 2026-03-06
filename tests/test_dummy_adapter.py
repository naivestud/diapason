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
