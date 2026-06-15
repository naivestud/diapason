"""Report generation for evaluation runs.

A run produces structured results; this module turns them into artifacts people
can read and compare: a JSON document for machines, and Markdown and HTML
summaries for humans. Both carry a little metadata so a report can be traced
back to the exact suite definition and package version that produced it.
"""

from __future__ import annotations

import csv
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


_HTML_TEMPLATE = Template(
    """\
<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<title>diapason report: {{ metadata.suite }}</title>
<style>
body { font-family: sans-serif; margin: 2rem auto; max-width: 60rem; color: #222; }
h1 { font-size: 1.4rem; } h2 { font-size: 1.1rem; margin-top: 2rem; }
table { border-collapse: collapse; margin: 0.5rem 0 1.5rem; }
th, td { border: 1px solid #ccc; padding: 0.3rem 0.7rem; text-align: left; }
th { background: #f4f4f4; } td.num { text-align: right; }
.meta { color: #666; font-size: 0.9rem; }
</style>
</head>
<body>
<h1>diapason report: {{ metadata.suite }}</h1>
<p class="meta">Generated {{ metadata.generated_at }} by diapason {{ metadata.diapason_version }}.
{% if metadata.config_fingerprint %}Config fingerprint: <code>{{ metadata.config_fingerprint }}</code>.{% endif %}
</p>
<h2>Summary</h2>
<table>
<tr><th>Task</th><th>Type</th><th>Examples</th><th>Errors</th><th>Scores</th></tr>
{% for task in tasks -%}
<tr><td>{{ task.name }}</td><td>{{ task.type }}</td><td class="num">{{ task.num_examples }}</td><td class="num">{{ task.num_errors }}</td><td>{{ task.summary }}</td></tr>
{% endfor -%}
</table>
<h2>Task details</h2>
{% for task in tasks -%}
<h3>{{ task.name }}</h3>
<table>
<tr><th>Metric</th><th>Score</th></tr>
{% for name, value in task.scores.items() -%}
<tr><td>{{ name }}</td><td class="num">{{ value }}</td></tr>
{% endfor -%}
</table>
{% endfor -%}
</body>
</html>
""",
    autoescape=True,
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


def _task_contexts(result: RunResult) -> list[dict[str, Any]]:
    """Build the per-task template context shared by every report renderer."""
    tasks = []
    for task in result.tasks:
        scores = {name: format_score(value) for name, value in sorted(task.scores.items())}
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
    return tasks


def render_markdown(result: RunResult, metadata: ReportMetadata) -> str:
    """Render a run result as a Markdown report.

    Args:
        result: The run result to summarise.
        metadata: Provenance shown at the top of the report.

    Returns:
        The rendered Markdown string.
    """
    return cast(str, _MARKDOWN_TEMPLATE.render(metadata=metadata, tasks=_task_contexts(result)))


def render_html(result: RunResult, metadata: ReportMetadata) -> str:
    """Render a run result as a self-contained HTML report.

    Args:
        result: The run result to summarise.
        metadata: Provenance shown at the top of the report.

    Returns:
        The rendered HTML document as a string.
    """
    return cast(str, _HTML_TEMPLATE.render(metadata=metadata, tasks=_task_contexts(result)))


def write_markdown_report(result: RunResult, path: str | Path, metadata: ReportMetadata) -> Path:
    """Render and write a Markdown report, creating parent directories."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_markdown(result, metadata), encoding="utf-8")
    return output_path


def write_html_report(result: RunResult, path: str | Path, metadata: ReportMetadata) -> Path:
    """Render and write an HTML report, creating parent directories."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_html(result, metadata), encoding="utf-8")
    return output_path


def write_csv_report(result: RunResult, path: str | Path, metadata: ReportMetadata) -> Path:
    """Write the run result as a tidy CSV table, one row per task and metric.

    CSV is meant for importing scores into spreadsheets and notebooks, so only
    the suite name is kept from the metadata.

    Args:
        result: The run result to export.
        path: Destination file; parent directories are created as needed.
        metadata: Provenance; only the suite name is written.

    Returns:
        The path the CSV was written to.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["suite", "task", "type", "metric", "score", "num_examples", "num_errors"])
        for task in result.tasks:
            for metric, value in sorted(task.scores.items()):
                writer.writerow(
                    [
                        metadata.suite,
                        task.name,
                        task.type,
                        metric,
                        format_score(value),
                        task.num_examples,
                        task.num_errors,
                    ]
                )
    return output_path


def format_score(value: float) -> str:
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
