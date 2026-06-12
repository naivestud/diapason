"""Tests for report generation."""

from __future__ import annotations

import json
from pathlib import Path

from diapason.config import SuiteConfig, TaskConfig
from diapason.report import (
    ReportMetadata,
    build_metadata,
    config_fingerprint,
    render_html,
    render_markdown,
    run_result_to_dict,
    write_html_report,
    write_json_report,
    write_markdown_report,
)
from diapason.runner import RunResult, TaskResult


def _result() -> RunResult:
    task = TaskResult(
        name="asr", type="asr", scores={"wer": 0.25, "cer": 0.1}, num_examples=4, num_errors=0
    )
    return RunResult(suite="demo", tasks=(task,))


def _metadata() -> ReportMetadata:
    return ReportMetadata(suite="demo", generated_at="2026-01-01T00:00:00+00:00")


def test_run_result_to_dict() -> None:
    data = run_result_to_dict(_result())
    assert data["suite"] == "demo"
    assert data["tasks"][0]["scores"]["wer"] == 0.25


def test_write_json_report_round_trips(tmp_path: Path) -> None:
    out = write_json_report(_result(), tmp_path / "report.json", _metadata())
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["results"]["suite"] == "demo"
    assert data["metadata"]["suite"] == "demo"


def test_render_markdown_contains_summary() -> None:
    markdown = render_markdown(_result(), _metadata())
    assert "# diapason report: demo" in markdown
    assert "wer" in markdown
    assert "0.2500" in markdown


def test_render_markdown_shows_fingerprint() -> None:
    metadata = ReportMetadata(suite="demo", generated_at="now", config_fingerprint="abc123")
    markdown = render_markdown(_result(), metadata)
    assert "abc123" in markdown


def test_config_fingerprint_is_stable() -> None:
    suite = SuiteConfig(
        name="s", tasks=(TaskConfig(name="t", type="asr", manifest="m.jsonl", metrics=("wer",)),)
    )
    assert config_fingerprint(suite) == config_fingerprint(suite)
    assert len(config_fingerprint(suite)) == 12


def test_config_fingerprint_changes_with_config() -> None:
    s1 = SuiteConfig(name="s", tasks=(TaskConfig(name="t", type="asr", manifest="a.jsonl"),))
    s2 = SuiteConfig(name="s", tasks=(TaskConfig(name="t", type="asr", manifest="b.jsonl"),))
    assert config_fingerprint(s1) != config_fingerprint(s2)


def test_build_metadata_populates_fields() -> None:
    suite = SuiteConfig(name="s", tasks=(TaskConfig(name="t", type="asr", manifest="m.jsonl"),))
    metadata = build_metadata(suite)
    assert metadata.suite == "s"
    assert metadata.generated_at != ""
    assert metadata.config_fingerprint != ""


def test_write_markdown_report(tmp_path: Path) -> None:
    out = write_markdown_report(_result(), tmp_path / "report.md", _metadata())
    assert out.read_text(encoding="utf-8").startswith("# diapason report: demo")


def test_render_html_contains_summary() -> None:
    html = render_html(_result(), _metadata())
    assert "<title>diapason report: demo</title>" in html
    assert "wer" in html
    assert "0.2500" in html


def test_render_html_escapes_task_names() -> None:
    task = TaskResult(
        name="<script>alert(1)</script>", type="asr", scores={"wer": 0.0}, num_examples=1
    )
    html = render_html(RunResult(suite="demo", tasks=(task,)), _metadata())
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_write_html_report(tmp_path: Path) -> None:
    out = write_html_report(_result(), tmp_path / "report.html", _metadata())
    assert out.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")
