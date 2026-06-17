"""Command line interface for diapason.

The CLI is a thin layer over the Python API: it parses arguments, loads the
suite config, and delegates to :mod:`diapason.runner` and
:mod:`diapason.report`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.table import Table

import diapason.adapters
import diapason.metrics  # noqa: F401  (registers built-in metrics)
from diapason.config import (
    ConfigError,
    SuiteConfig,
    discover_suites,
    load_suite,
    validate_components,
)
from diapason.datasets.manifest import ManifestError, load_manifest
from diapason.registry import adapters as adapter_registry
from diapason.registry import metrics as metric_registry
from diapason.report import (
    ReportMetadata,
    build_metadata,
    format_score,
    render_markdown,
    write_csv_report,
    write_html_report,
    write_json_report,
    write_markdown_report,
)
from diapason.runner import RunResult, TaskResult, resolve_manifest_path, run_suite
from diapason.version import __version__

_console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="diapason")
def main() -> None:
    """Reproducible evaluation harness for audio language models."""
    _configure_logging(0)


#: Log level for each verbosity count, indexed by ``verbosity + 1``.
_VERBOSITY_LEVELS = (logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG)


def _configure_logging(verbosity: int) -> None:
    """Set the root log level from a verbosity count.

    Negative values mean quiet (errors only), ``0`` warning, ``1`` info and
    ``2`` or more debug.
    """
    index = min(max(verbosity + 1, 0), len(_VERBOSITY_LEVELS) - 1)
    logging.basicConfig(
        level=_VERBOSITY_LEVELS[index], format="%(levelname)s %(name)s: %(message)s"
    )


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
    type=click.Choice(["json", "md", "html", "csv", "both", "all"]),
    default="both",
    show_default=True,
    help="Which report formats to write; both = json + md, all writes every format.",
)
@click.option("-v", "--verbose", count=True, help="Increase logging verbosity.")
@click.option("-q", "--quiet", is_flag=True, help="Only print errors.")
@click.option(
    "--include-examples",
    is_flag=True,
    help="Embed per-example scores and errors in the JSON report.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Validate the suite and print the plan without scoring anything.",
)
def run(
    config_path: Path,
    output_dir: Path | None,
    fmt: str,
    verbose: int,
    quiet: bool,
    include_examples: bool,
    dry_run: bool,
) -> None:
    """Evaluate the suite defined in CONFIG_PATH and write reports."""
    _configure_logging(-1 if quiet else verbose)
    suite = load_suite(config_path)
    if dry_run:
        _print_plan(suite, config_path.parent)
        return
    result = run_suite(suite, base_dir=config_path.parent)
    metadata = build_metadata(suite)

    destination = output_dir if output_dir is not None else Path.cwd() / "results"
    destination = _write_reports(result, suite.name, destination, fmt, metadata, include_examples)

    if not quiet:
        _print_summary(result)
        _console.print(f"[dim]reports written to {destination}[/dim]")


def _print_plan(suite: SuiteConfig, base_dir: Path) -> None:
    """Validate the suite and print what a full run would do, without scoring."""
    try:
        validate_components(suite, adapter_registry, metric_registry)
    except ConfigError as exc:
        raise click.ClickException(str(exc)) from exc
    for task in suite.tasks:
        manifest_path = resolve_manifest_path(task.manifest, base_dir)
        if not manifest_path.is_file():
            raise click.ClickException(f"task '{task.name}': manifest not found: {manifest_path}")
        try:
            examples = load_manifest(manifest_path, task.type)
        except ManifestError as exc:
            raise click.ClickException(f"task '{task.name}': {exc}") from exc
        adapter = "none (stored predictions)" if task.type == "sed" else task.adapter
        metrics = ", ".join(task.metrics) or "—"
        _console.print(
            f"  {task.name} [{task.type}]: {len(examples)} examples, "
            f"adapter={adapter}, metrics={metrics}",
            markup=False,
        )
    _console.print(f"[dim]dry run of suite '{suite.name}': nothing was scored[/dim]")


def _write_reports(
    result: RunResult,
    suite_name: str,
    destination: Path,
    fmt: str,
    metadata: ReportMetadata,
    include_examples: bool = False,
) -> Path:
    """Write the requested report formats into ``destination``.

    The directory is resolved to an absolute path and created when needed.
    Returns the directory the reports were written to.
    """
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if fmt in ("json", "both", "all"):
        write_json_report(
            result, destination / f"{suite_name}.json", metadata, include_examples=include_examples
        )
    if fmt in ("md", "both", "all"):
        write_markdown_report(result, destination / f"{suite_name}.md", metadata)
    if fmt in ("html", "all"):
        write_html_report(result, destination / f"{suite_name}.html", metadata)
    if fmt in ("csv", "all"):
        write_csv_report(result, destination / f"{suite_name}.csv", metadata)
    return destination


@main.command()
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def check(config_path: Path) -> None:
    """Validate the suite in CONFIG_PATH without running it.

    Checks the suite structure, that every adapter and metric name is
    registered, and that every task's manifest exists and parses.
    """
    try:
        suite = load_suite(config_path)
        validate_components(suite, adapter_registry, metric_registry)
    except ConfigError as exc:
        raise click.ClickException(str(exc)) from exc

    for task in suite.tasks:
        manifest_path = resolve_manifest_path(task.manifest, config_path.parent)
        if not manifest_path.is_file():
            raise click.ClickException(f"task '{task.name}': manifest not found: {manifest_path}")
        try:
            examples = load_manifest(manifest_path, task.type)
        except ManifestError as exc:
            raise click.ClickException(f"task '{task.name}': {exc}") from exc
        _console.print(f"[green]✓[/green] {task.name}: {len(examples)} examples")
    _console.print(f"[dim]suite '{suite.name}' looks good[/dim]")


@main.command(name="list")
@click.option(
    "--config-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Also list suite config files found in this directory.",
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["table", "json"]),
    default="table",
    show_default=True,
    help="Output format; json is meant for scripts.",
)
def list_components(config_dir: Path | None, fmt: str) -> None:
    """List registered adapters and metrics, and optionally discovered suites."""
    suites = discover_suites(config_dir) if config_dir is not None else []

    if fmt == "json":
        payload: dict[str, Any] = {
            "adapters": adapter_registry.names(),
            "metrics": metric_registry.names(),
        }
        if config_dir is not None:
            payload["suites"] = [path.name for path in suites]
        click.echo(json.dumps(payload, indent=2))
        return

    table = Table(title="registered components")
    table.add_column("kind")
    table.add_column("name")
    for name in adapter_registry.names():
        table.add_row("adapter", name)
    for name in metric_registry.names():
        table.add_row("metric", name)
    _console.print(table)

    if config_dir is not None and suites:
        _console.print(f"[bold]suites in {config_dir}[/bold]")
        for path in suites:
            _console.print(f"  {path.name}")


@main.command()
@click.argument("baseline", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("candidate", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def compare(baseline: Path, candidate: Path) -> None:
    """Show score differences between two saved report JSON files.

    BASELINE is the run to compare against; CANDIDATE is the new run. Tasks
    are matched by name and each metric's delta (candidate minus baseline) is
    shown with its sign.
    """
    base_result = _result_from_dict(json.loads(baseline.read_text(encoding="utf-8"))["results"])
    cand_result = _result_from_dict(json.loads(candidate.read_text(encoding="utf-8"))["results"])

    base_by_name = {task.name: task.scores for task in base_result.tasks}
    cand_by_name = {task.name: task.scores for task in cand_result.tasks}
    task_names = sorted(set(base_by_name) | set(cand_by_name))

    table = Table(title=f"compare: {base_result.suite}")
    table.add_column("task")
    table.add_column("metric")
    table.add_column("baseline", justify="right")
    table.add_column("candidate", justify="right")
    table.add_column("delta", justify="right")
    for name in task_names:
        base_scores = base_by_name.get(name, {})
        cand_scores = cand_by_name.get(name, {})
        for metric in sorted(set(base_scores) | set(cand_scores)):
            base_value = base_scores.get(metric)
            cand_value = cand_scores.get(metric)
            delta = (
                format_score(cand_value - base_value)
                if base_value is not None and cand_value is not None
                else "—"
            )
            table.add_row(
                name,
                metric,
                format_score(base_value) if base_value is not None else "—",
                format_score(cand_value) if cand_value is not None else "—",
                delta,
            )
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
        scores = ", ".join(
            f"{name}={format_score(value)}" for name, value in sorted(task.scores.items())
        )
        table.add_row(task.name, task.type, str(task.num_examples), str(task.num_errors), scores)
    _console.print(table)


if __name__ == "__main__":
    main()
