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
