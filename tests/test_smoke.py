"""Smoke tests to make sure the package imports and CI has something to run."""

import diapason


def test_package_imports() -> None:
    assert diapason.__name__ == "diapason"
