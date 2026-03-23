"""Report generation for evaluation runs.

A run produces structured results; this module turns them into artifacts people
can read and compare: a JSON document for machines and a Markdown summary for
humans. Both carry a little metadata so a report can be traced back to the
exact suite definition and package version that produced it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from jinja2 import Template

from diapason.config import SuiteConfig
from diapason.runner import RunResult
from diapason.version import __version__

_MARKDOWN_TEMPLATE = Template(
    """\
# diapason report: {{ metadata.suite }}

_Generated {{ metadata.generated_at }} by diapason {{ metadata.diapason_version }}._
{% if metadata.config_fingerprint %}
Config fingerprint: `{{ metadata.config_fingerprint }}`
{% endif %}
## Summary

| Task | Type | Examples | Errors | Scores |
| --- | --- | ---: | ---: | --- |
{% for task in tasks -%}
| {{ task.name }} | {{ task.type }} | {{ task.num_examples }} | {{ task.num_errors }} | {{ task.summary }} |
{% endfor %}
## Task details
{% for task in tasks %}
### {{ task.name }}

| Metric | Score |
| --- | ---: |
{% for name, value in task.scores.items() -%}
| {{ name }} | {{ value }} |
{% endfor %}
{%- endfor %}
"""
)


@dataclass(frozen=True)
class ReportMetadata:
    """Provenance recorded alongside every report."""

    suite: str
    generated_at: str
    diapason_version: str = __version__
    config_fingerprint: str = ""


def run_result_to_dict(result: RunResult) -> dict[str, Any]:
    """Convert a run result into plain JSON-serializable data."""
    return {
        "suite": result.suite,
        "tasks": [
            {
                "name": task.name,
                "type": task.type,
                "scores": dict(task.scores),
                "num_examples": task.num_examples,
                "num_errors": task.num_errors,
            }
            for task in result.tasks
        ],
    }


def write_json_report(result: RunResult, path: str | Path, metadata: ReportMetadata) -> Path:
    """Write the run result and metadata as pretty-printed JSON.

    Args:
        result: The run result to serialize.
        path: Destination file; parent directories are created as needed.
        metadata: Provenance recorded next to the results.

    Returns:
        The path the report was written to.
    """
    payload = {"metadata": asdict(metadata), "results": run_result_to_dict(result)}
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def render_markdown(result: RunResult, metadata: ReportMetadata) -> str:
    """Render a run result as a Markdown report.

    Args:
        result: The run result to summarise.
        metadata: Provenance shown at the top of the report.

    Returns:
        The rendered Markdown string.
    """
    tasks = []
    for task in result.tasks:
        scores = {name: _format_score(value) for name, value in sorted(task.scores.items())}
        summary = ", ".join(f"{name}={value}" for name, value in scores.items()) or "—"
        tasks.append(
            {
                "name": task.name,
                "type": task.type,
                "num_examples": task.num_examples,
                "num_errors": task.num_errors,
                "scores": scores,
                "summary": summary,
            }
        )
    return cast(str, _MARKDOWN_TEMPLATE.render(metadata=metadata, tasks=tasks))


def write_markdown_report(result: RunResult, path: str | Path, metadata: ReportMetadata) -> Path:
    """Render and write a Markdown report, creating parent directories."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_markdown(result, metadata), encoding="utf-8")
    return output_path


def _format_score(value: float) -> str:
    """Format a metric value with four decimal places."""
    return f"{value:.4f}"


def config_fingerprint(suite: SuiteConfig) -> str:
    """Hash a suite definition so reports can be traced back to it.

    Args:
        suite: The suite whose definition should be fingerprinted.

    Returns:
        A short hex digest of the suite definition.
    """
    payload = json.dumps(
        {"name": suite.name, "tasks": [asdict(task) for task in suite.tasks]},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def build_metadata(suite: SuiteConfig) -> ReportMetadata:
    """Create provenance metadata for a run of ``suite`` at the current time."""
    return ReportMetadata(
        suite=suite.name,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        config_fingerprint=config_fingerprint(suite),
    )
