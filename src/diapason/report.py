"""Report generation for evaluation runs.

A run produces structured results; this module turns them into artifacts people
can read and compare: a JSON document for machines and a Markdown summary for
humans. Both carry a little metadata so a report can be traced back to the
exact suite definition and package version that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass
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
