"""Report generation for evaluation runs.

A run produces structured results; this module turns them into artifacts people
can read and compare: a JSON document for machines and a Markdown summary for
humans. Both carry a little metadata so a report can be traced back to the
exact suite definition and package version that produced it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from diapason.runner import RunResult
from diapason.version import __version__


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


def write_json_report(
    result: RunResult, path: str | Path, metadata: ReportMetadata
) -> Path:
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
