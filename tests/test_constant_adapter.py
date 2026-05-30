"""Tests for the constant baseline adapter."""

from __future__ import annotations

from diapason.adapters import ConstantAdapter
from diapason.datasets.manifest import Example
from diapason.registry import adapters


def test_constant_returns_fixed_text() -> None:
    adapter = ConstantAdapter(text="a dog")
    assert adapter.predict(Example(id="1", prompt="What is this?")) == "a dog"
    assert adapter.predict(Example(id="2", prompt="Anything else?")) == "a dog"


def test_constant_defaults_to_empty_text() -> None:
    adapter = ConstantAdapter()
    assert adapter.predict(Example(id="1")) == ""


def test_constant_is_registered() -> None:
    assert "constant" in adapters.names()
