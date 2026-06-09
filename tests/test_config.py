"""Tests for suite configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from diapason.config import ConfigError, SuiteConfig, TaskConfig, dump_suite, load_suite


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "suite.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_minimal_suite(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
suite: demo
description: a small demo
tasks:
  - name: asr_task
    type: asr
    manifest: data.jsonl
    metrics: [wer, cer]
""",
    )
    suite = load_suite(path)
    assert suite.name == "demo"
    assert suite.description == "a small demo"
    assert len(suite.tasks) == 1
    task = suite.tasks[0]
    assert task.metrics == ("wer", "cer")


def test_task_defaults(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
suite: d
tasks:
  - name: t
    type: asr
    manifest: m.jsonl
""",
    )
    task = load_suite(path).tasks[0]
    assert task.adapter == "dummy"
    assert task.metrics == ()
    assert task.adapter_options == {}
    assert task.aggregate == "mean"


def test_aggregate_method_from_yaml(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
suite: d
tasks:
  - name: t
    type: asr
    manifest: m.jsonl
    aggregate: median
""",
    )
    task = load_suite(path).tasks[0]
    assert task.aggregate == "median"


def test_unknown_aggregate_method_raises(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
suite: d
tasks:
  - name: t
    type: asr
    manifest: m.jsonl
    aggregate: mode
""",
    )
    with pytest.raises(ConfigError, match="aggregation method"):
        load_suite(path)


def test_unknown_task_type_raises(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
suite: d
tasks:
  - name: t
    type: translation
    manifest: m.jsonl
""",
    )
    with pytest.raises(ConfigError, match="unknown type"):
        load_suite(path)


def test_duplicate_task_names_raise(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
suite: d
tasks:
  - name: t
    type: asr
    manifest: a.jsonl
  - name: t
    type: asr
    manifest: b.jsonl
""",
    )
    with pytest.raises(ConfigError, match="duplicate task name"):
        load_suite(path)


def test_missing_manifest_raises(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """
suite: d
tasks:
  - name: t
    type: asr
""",
    )
    with pytest.raises(ConfigError, match="manifest"):
        load_suite(path)


def test_non_mapping_document_raises(tmp_path: Path) -> None:
    path = _write(tmp_path, "- just\n- a\n- list\n")
    with pytest.raises(ConfigError, match="mapping"):
        load_suite(path)


def test_suite_without_tasks_raises(tmp_path: Path) -> None:
    path = _write(tmp_path, "suite: empty\ntasks: []\n")
    with pytest.raises(ConfigError, match="no tasks"):
        load_suite(path)


def test_dump_suite_round_trips(tmp_path: Path) -> None:
    suite = SuiteConfig(
        name="generated",
        description="made in code",
        tasks=(
            TaskConfig(
                name="asr_task",
                type="asr",
                manifest="data.jsonl",
                adapter="dummy",
                adapter_options={"corruption": 0.2},
                metrics=("wer", "cer"),
                metric_options={"normalize": True},
                aggregate="median",
            ),
        ),
    )
    path = tmp_path / "suite.yaml"
    path.write_text(dump_suite(suite), encoding="utf-8")
    assert load_suite(path) == suite


def test_dump_suite_omits_defaults(tmp_path: Path) -> None:
    suite = SuiteConfig(
        name="minimal", tasks=(TaskConfig(name="t", type="asr", manifest="m.jsonl"),)
    )
    text = dump_suite(suite)
    assert "adapter" not in text
    assert "aggregate" not in text
    assert load_suite(_write(tmp_path, text)) == suite
