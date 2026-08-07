"""Configuration model for evaluation suites.

A suite is described by a small YAML document that lists the tasks to run,
which adapter produces predictions for each task, and which metrics score
those predictions. The dataclasses below are the in-memory representation of
that document; loading lives in :func:`load_suite`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Task families understood by the harness.
TASK_TYPES = ("asr", "sed", "aqa")


@dataclass(frozen=True)
class TaskConfig:
    """One evaluation task inside a suite.

    Attributes:
        name: Human readable identifier, unique within a suite.
        type: One of :data:`TASK_TYPES`.
        manifest: Path to the JSONL manifest holding examples.
        adapter: Registered adapter used to produce predictions.
        adapter_options: Keyword arguments forwarded to the adapter.
        metrics: Names of registered metrics to compute.
        metric_options: Keyword arguments forwarded to every metric.
    """

    name: str
    type: str
    manifest: str
    adapter: str = "dummy"
    adapter_options: dict[str, object] = field(default_factory=dict)
    metrics: tuple[str, ...] = ()
    metric_options: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class SuiteConfig:
    """A named collection of tasks evaluated together."""

    name: str
    description: str = ""
    tasks: tuple[TaskConfig, ...] = ()
