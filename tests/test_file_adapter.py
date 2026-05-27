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


def test_file_adapter_invalid_json_reports_line(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id": "1", "prediction": "a"}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad\.jsonl:2"):
        FileAdapter(path=str(path))


def test_file_adapter_missing_keys_report_line(tmp_path: Path) -> None:
    path = tmp_path / "incomplete.jsonl"
    path.write_text('{"id": "1"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"incomplete\.jsonl:1"):
        FileAdapter(path=str(path))


def test_file_adapter_custom_fields(tmp_path: Path) -> None:
    path = tmp_path / "custom.jsonl"
    path.write_text('{"example_id": "x1", "text": "a b"}\n', encoding="utf-8")
    adapter = FileAdapter(path=str(path), id_field="example_id", prediction_field="text")
    assert adapter.predict(Example(id="x1")) == "a b"


def test_file_adapter_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        FileAdapter(path=str(tmp_path / "does-not-exist.jsonl"))
