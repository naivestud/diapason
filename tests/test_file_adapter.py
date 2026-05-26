"""Tests for the file-backed prediction adapter."""

from __future__ import annotations

from pathlib import Path

import pytest

from diapason.adapters import FileAdapter
from diapason.datasets.manifest import Example
from diapason.registry import adapters


def _write_predictions(tmp_path: Path) -> Path:
    path = tmp_path / "predictions.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"id": "1", "prediction": "the cat sat"}',
                '{"id": "2", "prediction": "hello world"}',
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_file_adapter_serves_stored_predictions(tmp_path: Path) -> None:
    adapter = FileAdapter(path=str(_write_predictions(tmp_path)))
    assert adapter.predict(Example(id="1")) == "the cat sat"
    assert adapter.predict(Example(id="2")) == "hello world"


def test_file_adapter_missing_id_raises(tmp_path: Path) -> None:
    adapter = FileAdapter(path=str(_write_predictions(tmp_path)))
    with pytest.raises(KeyError, match="no prediction stored"):
        adapter.predict(Example(id="99"))


def test_file_adapter_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    path.write_text('{"id": "1", "prediction": "a"}\n\n', encoding="utf-8")
    adapter = FileAdapter(path=str(path))
    assert adapter.predict(Example(id="1")) == "a"


def test_file_is_registered() -> None:
    assert "file" in adapters.names()
