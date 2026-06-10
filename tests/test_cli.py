"""Tests for the command line interface."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from diapason.cli import main


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


def test_run_writes_reports(tmp_path: Path) -> None:
    config = _write_suite(tmp_path)
    output_dir = tmp_path / "out"
    result = CliRunner().invoke(main, ["run", str(config), "--output-dir", str(output_dir)])
    assert result.exit_code == 0
    assert (output_dir / "demo.json").exists()
    assert (output_dir / "demo.md").exists()


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
