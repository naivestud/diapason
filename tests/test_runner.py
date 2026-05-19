"""End-to-end tests for the evaluation runner."""

from __future__ import annotations

from pathlib import Path

import pytest

from diapason.config import SuiteConfig, TaskConfig
from diapason.runner import run_suite, run_task


def _asr_manifest(tmp_path: Path) -> Path:
    path = tmp_path / "asr.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"id": "1", "reference": "the cat sat"}',
                '{"id": "2", "reference": "hello world"}',
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_dummy_adapter_scores_perfectly(tmp_path: Path) -> None:
    task = TaskConfig(
        name="asr", type="asr", manifest=str(_asr_manifest(tmp_path)), metrics=("wer", "cer")
    )
    result = run_task(task)
    assert result.scores["wer"] == 0.0
    assert result.scores["cer"] == 0.0
    assert result.num_examples == 2
    assert result.num_errors == 0


def test_corrupted_adapter_scores_worse(tmp_path: Path) -> None:
    task = TaskConfig(
        name="asr",
        type="asr",
        manifest=str(_asr_manifest(tmp_path)),
        adapter="dummy",
        adapter_options={"corruption": 0.5},
        metrics=("wer",),
    )
    result = run_task(task)
    assert result.scores["wer"] > 0.0


def test_sed_task_from_stored_predictions(tmp_path: Path) -> None:
    path = tmp_path / "sed.jsonl"
    path.write_text(
        '{"id": "1", "reference": [{"onset": 0.0, "offset": 1.0, "label": "dog"}], '
        '"prediction": [{"onset": 0.0, "offset": 1.0, "label": "dog"}]}',
        encoding="utf-8",
    )
    task = TaskConfig(name="sed", type="sed", manifest=str(path), metrics=("event_f1",))
    result = run_task(task)
    assert result.scores["event_f1"] == 1.0


def test_run_suite_collects_tasks(tmp_path: Path) -> None:
    suite = SuiteConfig(
        name="demo",
        tasks=(
            TaskConfig(
                name="t", type="asr", manifest=str(_asr_manifest(tmp_path)), metrics=("wer",)
            ),
        ),
    )
    run = run_suite(suite)
    assert run.suite == "demo"
    assert len(run.tasks) == 1
    assert run.tasks[0].scores["wer"] == 0.0


def test_relative_manifest_resolved_against_base_dir(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "asr.jsonl").write_text(
        '{"id": "1", "reference": "a b", "prediction": "a b"}', encoding="utf-8"
    )
    task = TaskConfig(name="t", type="asr", manifest="data/asr.jsonl", metrics=("wer",))
    result = run_task(task, base_dir=tmp_path)
    assert result.scores["wer"] == 0.0


def test_bad_example_is_isolated(tmp_path: Path) -> None:
    path = tmp_path / "mixed.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"id": "ok", "reference": "the cat", "prediction": "the cat"}',
                '{"id": "bad", "reference": "", "prediction": "hello"}',
            ]
        ),
        encoding="utf-8",
    )
    task = TaskConfig(name="t", type="asr", manifest=str(path), metrics=("wer",))
    result = run_task(task)
    assert result.num_errors == 1
    assert result.num_examples == 2
    assert result.scores["wer"] == 0.0


def test_median_aggregation_ignores_outlier(tmp_path: Path) -> None:
    path = tmp_path / "outlier.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"id": "1", "reference": "a b", "prediction": "a b"}',
                '{"id": "2", "reference": "c d", "prediction": "c d"}',
                '{"id": "3", "reference": "e f", "prediction": "x y"}',
            ]
        ),
        encoding="utf-8",
    )
    mean_task = TaskConfig(name="t", type="asr", manifest=str(path), metrics=("wer",))
    median_task = TaskConfig(
        name="t", type="asr", manifest=str(path), metrics=("wer",), aggregate="median"
    )
    assert run_task(mean_task).scores["wer"] == pytest.approx(1 / 3)
    assert run_task(median_task).scores["wer"] == 0.0
