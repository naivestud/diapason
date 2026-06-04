"""Tests for JSONL manifest loading."""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from diapason.datasets.manifest import ManifestError, load_manifest


def _write(tmp_path: Path, lines: list[str]) -> Path:
    path = tmp_path / "manifest.jsonl"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def test_load_asr_manifest(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        [
            '{"id": "1", "reference": "the cat sat", "prediction": "the cat sat"}',
            '{"id": "2", "reference": "hello world"}',
        ],
    )
    examples = load_manifest(path, "asr")
    assert len(examples) == 2
    assert examples[0].reference_text == "the cat sat"
    assert examples[0].prediction_text == "the cat sat"
    assert examples[1].prediction_text is None


def test_aqa_requires_a_question(tmp_path: Path) -> None:
    path = _write(tmp_path, ['{"id": "1", "reference": "42"}'])
    with pytest.raises(ManifestError, match="question"):
        load_manifest(path, "aqa")


def test_aqa_reads_question_into_prompt(tmp_path: Path) -> None:
    path = _write(tmp_path, ['{"id": "1", "question": "how many?", "reference": "42"}'])
    example = load_manifest(path, "aqa")[0]
    assert example.prompt == "how many?"
    assert example.reference_text == "42"


def test_sed_parses_reference_events(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        ['{"id": "1", "reference": [{"onset": 0.0, "offset": 1.0, "label": "dog"}]}'],
    )
    example = load_manifest(path, "sed")[0]
    assert example.reference_events is not None
    assert example.reference_events[0].label == "dog"


def test_asr_missing_reference_raises(tmp_path: Path) -> None:
    path = _write(tmp_path, ['{"id": "1"}'])
    with pytest.raises(ManifestError, match="reference"):
        load_manifest(path, "asr")


def test_unknown_task_type_raises(tmp_path: Path) -> None:
    path = _write(tmp_path, ['{"id": "1", "reference": "x"}'])
    with pytest.raises(ManifestError, match="unknown task type"):
        load_manifest(path, "machine_translation")


def test_empty_manifest_raises(tmp_path: Path) -> None:
    path = _write(tmp_path, [])
    with pytest.raises(ManifestError, match="no examples"):
        load_manifest(path, "asr")


def test_invalid_json_raises(tmp_path: Path) -> None:
    path = _write(tmp_path, ['{"id": '])
    with pytest.raises(ManifestError, match="invalid JSON"):
        load_manifest(path, "asr")


def test_extra_fields_land_in_meta(tmp_path: Path) -> None:
    path = _write(tmp_path, ['{"id": "1", "reference": "x", "speaker": "spk1"}'])
    example = load_manifest(path, "asr")[0]
    assert example.meta["speaker"] == "spk1"


def test_gzip_manifest_is_read_transparently(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write('{"id": "1", "reference": "the cat sat"}\n')
    example = load_manifest(path, "asr")[0]
    assert example.reference_text == "the cat sat"
