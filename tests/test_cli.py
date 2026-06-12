"""Tests for the command line interface."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from diapason.cli import main
from diapason.report import ReportMetadata, write_json_report
from diapason.runner import RunResult, TaskResult


def _write_suite(tmp_path: Path) -> Path:
    manifest = tmp_path / "asr.jsonl"
    manifest.write_text('{"id": "1", "reference": "the cat sat"}', encoding="utf-8")
    config = tmp_path / "suite.yaml"
    config.write_text(
        "suite: demo\n"
        "tasks:\n"
        "  - name: t\n"
        "    type: asr\n"
        "    manifest: asr.jsonl\n"
        "    metrics: [wer]\n",
        encoding="utf-8",
    )
    return config


def test_version_option() -> None:
    result = CliRunner().invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "diapason" in result.output


def test_list_shows_registered_components() -> None:
    result = CliRunner().invoke(main, ["list"])
    assert result.exit_code == 0
    assert "dummy" in result.output
    assert "wer" in result.output


def test_list_json_format_is_machine_readable(tmp_path: Path) -> None:
    (tmp_path / "demo.yaml").write_text("suite: d\ntasks: []\n", encoding="utf-8")
    result = CliRunner().invoke(main, ["list", "--format", "json", "--config-dir", str(tmp_path)])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert "dummy" in payload["adapters"]
    assert "wer" in payload["metrics"]
    assert payload["suites"] == ["demo.yaml"]


def test_list_json_without_config_dir_omits_suites() -> None:
    result = CliRunner().invoke(main, ["list", "--format", "json"])
    payload = json.loads(result.output)
    assert "suites" not in payload


def test_run_writes_reports(tmp_path: Path) -> None:
    config = _write_suite(tmp_path)
    output_dir = tmp_path / "out"
    result = CliRunner().invoke(main, ["run", str(config), "--output-dir", str(output_dir)])
    assert result.exit_code == 0
    assert (output_dir / "demo.json").exists()
    assert (output_dir / "demo.md").exists()


def test_run_format_html_writes_only_html(tmp_path: Path) -> None:
    config = _write_suite(tmp_path)
    output_dir = tmp_path / "out"
    result = CliRunner().invoke(
        main, ["run", str(config), "--output-dir", str(output_dir), "--format", "html"]
    )
    assert result.exit_code == 0
    assert (output_dir / "demo.html").exists()
    assert not (output_dir / "demo.json").exists()
    assert not (output_dir / "demo.md").exists()


def test_run_format_all_writes_every_format(tmp_path: Path) -> None:
    config = _write_suite(tmp_path)
    output_dir = tmp_path / "out"
    result = CliRunner().invoke(
        main, ["run", str(config), "--output-dir", str(output_dir), "--format", "all"]
    )
    assert result.exit_code == 0
    assert (output_dir / "demo.json").exists()
    assert (output_dir / "demo.md").exists()
    assert (output_dir / "demo.html").exists()


def test_report_renders_from_saved_json(tmp_path: Path) -> None:
    config = _write_suite(tmp_path)
    output_dir = tmp_path / "out"
    CliRunner().invoke(main, ["run", str(config), "--output-dir", str(output_dir)])
    result = CliRunner().invoke(main, ["report", str(output_dir / "demo.json")])
    assert result.exit_code == 0
    assert "diapason report: demo" in result.output


def test_check_passes_for_valid_suite(tmp_path: Path) -> None:
    config = _write_suite(tmp_path)
    result = CliRunner().invoke(main, ["check", str(config)])
    assert result.exit_code == 0
    assert "t: 1 examples" in result.output
    assert "looks good" in result.output


def test_check_fails_for_unknown_metric(tmp_path: Path) -> None:
    manifest = tmp_path / "asr.jsonl"
    manifest.write_text('{"id": "1", "reference": "the cat sat"}', encoding="utf-8")
    config = tmp_path / "suite.yaml"
    config.write_text(
        "suite: demo\ntasks:\n  - name: t\n    type: asr\n"
        "    manifest: asr.jsonl\n    metrics: [not_a_metric]\n",
        encoding="utf-8",
    )
    result = CliRunner().invoke(main, ["check", str(config)])
    assert result.exit_code != 0
    assert "not_a_metric" in result.output


def test_check_fails_for_missing_manifest(tmp_path: Path) -> None:
    config = tmp_path / "suite.yaml"
    config.write_text(
        "suite: demo\ntasks:\n  - name: t\n    type: asr\n"
        "    manifest: missing.jsonl\n    metrics: [wer]\n",
        encoding="utf-8",
    )
    result = CliRunner().invoke(main, ["check", str(config)])
    assert result.exit_code != 0
    assert "manifest not found" in result.output


def test_check_fails_for_unparseable_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "bad.jsonl"
    manifest.write_text('{"id": "1"}', encoding="utf-8")  # missing reference
    config = tmp_path / "suite.yaml"
    config.write_text(
        "suite: demo\ntasks:\n  - name: t\n    type: asr\n"
        "    manifest: bad.jsonl\n    metrics: [wer]\n",
        encoding="utf-8",
    )
    result = CliRunner().invoke(main, ["check", str(config)])
    assert result.exit_code != 0
    assert "missing a reference" in result.output


def _write_report(tmp_path: Path, name: str, wer: float) -> Path:
    result = RunResult(
        suite="demo",
        tasks=(TaskResult(name="t", type="asr", scores={"wer": wer}, num_examples=1),),
    )
    path = tmp_path / f"{name}.json"
    write_json_report(result, path, ReportMetadata(suite="demo", generated_at="now"))
    return path


def test_compare_shows_delta(tmp_path: Path) -> None:
    base = _write_report(tmp_path, "base", wer=0.30)
    cand = _write_report(tmp_path, "cand", wer=0.25)
    result = CliRunner().invoke(main, ["compare", str(base), str(cand)])
    assert result.exit_code == 0
    assert "0.3000" in result.output
    assert "0.2500" in result.output
    assert "-0.0500" in result.output


def test_compare_handles_task_only_in_one_report(tmp_path: Path) -> None:
    base = _write_report(tmp_path, "base", wer=0.30)
    extra = RunResult(
        suite="demo",
        tasks=(
            TaskResult(name="t", type="asr", scores={"wer": 0.2}, num_examples=1),
            TaskResult(name="new", type="asr", scores={"wer": 0.1}, num_examples=1),
        ),
    )
    cand = tmp_path / "cand.json"
    write_json_report(extra, cand, ReportMetadata(suite="demo", generated_at="now"))
    result = CliRunner().invoke(main, ["compare", str(base), str(cand)])
    assert result.exit_code == 0
    assert "new" in result.output
    assert "—" in result.output
