"""Command line interface for diapason.

The CLI is a thin layer over the Python API: it parses arguments, loads the
suite config, and delegates to :mod:`diapason.runner` and
:mod:`diapason.report`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.table import Table

import diapason.adapters  # noqa: F401  (registers built-in adapters)
import diapason.metrics  # noqa: F401  (registers built-in metrics)
from diapason.config import load_suite
from diapason.registry import adapters as adapter_registry
from diapason.registry import metrics as metric_registry
from diapason.report import (
    ReportMetadata,
    build_metadata,
    render_markdown,
    write_json_report,
    write_markdown_report,
)
from diapason.runner import RunResult, TaskResult, run_suite
from diapason.version import __version__

_console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="diapason")
def main() -> None:
    """Reproducible evaluation harness for audio language models."""


@main.command()
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Directory for report files. Defaults to ./results.",
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["json", "md", "both"]),
    default="both",
    show_default=True,
    help="Which report formats to write.",
)
def run(config_path: Path, output_dir: Path | None, fmt: str) -> None:
    """Evaluate the suite defined in CONFIG_PATH and write reports."""
    suite = load_suite(config_path)
    result = run_suite(suite, base_dir=config_path.parent)
    metadata = build_metadata(suite)

    destination = output_dir if output_dir is not None else Path.cwd() / "results"
    if fmt in ("json", "both"):
        write_json_report(result, destination / f"{suite.name}.json", metadata)
    if fmt in ("md", "both"):
        write_markdown_report(result, destination / f"{suite.name}.md", metadata)

    _print_summary(result)
    _console.print(f"[dim]reports written to {destination}[/dim]")


@main.command(name="list")
def list_components() -> None:
    """List the registered adapters and metrics."""
    table = Table(title="registered components")
    table.add_column("kind")
    table.add_column("name")
    for name in adapter_registry.names():
        table.add_row("adapter", name)
    for name in metric_registry.names():
        table.add_row("metric", name)
    _console.print(table)


@main.command()
@click.argument("results_json", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Where to write the Markdown report. Defaults to stdout.",
)
def report(results_json: Path, output: Path | None) -> None:
    """Re-render a Markdown report from a saved RESULTS_JSON file."""
    data = json.loads(results_json.read_text(encoding="utf-8"))
    result = _result_from_dict(data["results"])
    metadata = _metadata_from_dict(data.get("metadata", {}))
    markdown = render_markdown(result, metadata)
    if output is None:
        _console.print(markdown, markup=False)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(markdown, encoding="utf-8")
        _console.print(f"[dim]report written to {output}[/dim]")


def _result_from_dict(payload: dict[str, Any]) -> RunResult:
    """Rebuild a run result from the dictionary written by the JSON report."""
    tasks = []
    for task in payload.get("tasks", []):
        tasks.append(
            TaskResult(
                name=str(task["name"]),
                type=str(task["type"]),
                scores={str(key): float(value) for key, value in task.get("scores", {}).items()},
                num_examples=int(task.get("num_examples", 0)),
                num_errors=int(task.get("num_errors", 0)),
            )
        )
    return RunResult(suite=str(payload.get("suite", "")), tasks=tuple(tasks))


def _metadata_from_dict(payload: dict[str, Any]) -> ReportMetadata:
    """Rebuild report metadata from the dictionary written by the JSON report."""
    return ReportMetadata(
        suite=str(payload.get("suite", "")),
        generated_at=str(payload.get("generated_at", "")),
        diapason_version=str(payload.get("diapason_version", __version__)),
        config_fingerprint=str(payload.get("config_fingerprint", "")),
    )


def _print_summary(result: RunResult) -> None:
    """Render a run result as a rich table on the console."""
    table = Table(title=f"diapason: {result.suite}")
    table.add_column("task")
    table.add_column("type")
    table.add_column("examples", justify="right")
    table.add_column("errors", justify="right")
    table.add_column("scores")
    for task in result.tasks:
        scores = ", ".join(f"{name}={value:.4f}" for name, value in sorted(task.scores.items()))
        table.add_row(task.name, task.type, str(task.num_examples), str(task.num_errors), scores)
    _console.print(table)


if __name__ == "__main__":
    main()
