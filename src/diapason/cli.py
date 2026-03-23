"""Command line interface for diapason.

The CLI is a thin layer over the Python API: it parses arguments, loads the
suite config, and delegates to :mod:`diapason.runner` and
:mod:`diapason.report`.
"""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from diapason.config import load_suite
from diapason.report import build_metadata, write_json_report, write_markdown_report
from diapason.runner import RunResult, run_suite
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
