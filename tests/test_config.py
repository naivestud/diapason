"""Tests for suite configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from diapason.config import (
    ConfigError,
    SuiteConfig,
    TaskConfig,
    discover_suites,
    dump_suite,
    load_suite,
    validate_components,
)
from diapason.registry import Registry


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


def _registries() -> tuple[Registry[object], Registry[object]]:
    adapter_registry: Registry[object] = Registry("adapter")
    metric_registry: Registry[object] = Registry("metric")
    adapter_registry.register("dummy")(object)
    metric_registry.register("wer")(object)
    return adapter_registry, metric_registry


def test_validate_components_accepts_known_names() -> None:
    suite = SuiteConfig(
        name="s",
        tasks=(TaskConfig(name="t", type="asr", manifest="m", adapter="dummy", metrics=("wer",)),),
    )
    adapter_registry, metric_registry = _registries()
    validate_components(suite, adapter_registry, metric_registry)


def test_validate_components_rejects_unknown_adapter() -> None:
    suite = SuiteConfig(
        name="s",
        tasks=(TaskConfig(name="t", type="asr", manifest="m", adapter="missing"),),
    )
    adapter_registry, metric_registry = _registries()
    with pytest.raises(ConfigError, match="task 't': unknown adapter: 'missing'"):
        validate_components(suite, adapter_registry, metric_registry)


def test_validate_components_rejects_unknown_metric() -> None:
    suite = SuiteConfig(
        name="s",
        tasks=(TaskConfig(name="t", type="asr", manifest="m", metrics=("nope",)),),
    )
    adapter_registry, metric_registry = _registries()
    with pytest.raises(ConfigError, match="task 't': unknown metric: 'nope'"):
        validate_components(suite, adapter_registry, metric_registry)


def test_discover_suites_returns_yaml_files_sorted(tmp_path: Path) -> None:
    (tmp_path / "b.yml").write_text("suite: b\ntasks: []\n", encoding="utf-8")
    (tmp_path / "a.yaml").write_text("suite: a\ntasks: []\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("not a suite", encoding="utf-8")
    names = [path.name for path in discover_suites(tmp_path)]
    assert names == ["a.yaml", "b.yml"]


def test_discover_suites_missing_directory_is_empty(tmp_path: Path) -> None:
    assert discover_suites(tmp_path / "does-not-exist") == []


def test_discover_suites_ignores_subdirectories(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "inner.yaml").write_text("suite: inner\ntasks: []\n", encoding="utf-8")
    assert discover_suites(tmp_path) == []


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
