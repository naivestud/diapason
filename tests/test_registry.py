"""Tests for the name-to-factory registry."""

from __future__ import annotations

import pytest

from diapason.registry import Registry, RegistryError


def test_register_and_get() -> None:
    registry: Registry[str] = Registry("widget")
    registry.register("first")(lambda: "one")
    assert registry.get("first")() == "one"


def test_get_unknown_raises_with_known_names() -> None:
    registry: Registry[str] = Registry("widget")
    registry.register("first")(lambda: "one")
    with pytest.raises(RegistryError, match="unknown widget: 'second'"):
        registry.get("second")


def test_get_unknown_empty_registry_lists_none() -> None:
    registry: Registry[str] = Registry("widget")
    with pytest.raises(RegistryError, match="<none>"):
        registry.get("anything")


def test_duplicate_registration_raises() -> None:
    registry: Registry[str] = Registry("widget")
    registry.register("first")(lambda: "one")
    with pytest.raises(RegistryError, match="already registered"):
        registry.register("first")(lambda: "uno")


def test_names_are_sorted() -> None:
    registry: Registry[str] = Registry("widget")
    registry.register("zeta")(lambda: "z")
    registry.register("alpha")(lambda: "a")
    assert registry.names() == ["alpha", "zeta"]


def test_register_returns_factory_unchanged() -> None:
    registry: Registry[str] = Registry("widget")

    def factory() -> str:
        return "one"

    assert registry.register("first")(factory) is factory


def test_registry_error_is_key_error() -> None:
    # Callers may catch KeyError to handle missing components generically.
    assert issubclass(RegistryError, KeyError)
