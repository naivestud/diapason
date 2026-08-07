"""Configuration model for evaluation suites.

A suite is described by a small YAML document that lists the tasks to run,
which adapter produces predictions for each task, and which metrics score
those predictions. The dataclasses below are the in-memory representation of
that document; loading lives in :func:`load_suite`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

#: Task families understood by the harness.
TASK_TYPES = ("asr", "sed", "aqa")


class ConfigError(ValueError):
    """Raised when a suite definition is malformed or invalid."""


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


def load_suite(path: str | Path) -> SuiteConfig:
    """Load a suite definition from a YAML document.

    Args:
        path: Path to a ``*.yaml`` / ``*.yml`` suite definition.

    Returns:
        The parsed :class:`SuiteConfig`.

    Raises:
        ConfigError: If the document is not a mapping of suite fields.
    """
    text = Path(path).read_text(encoding="utf-8")
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise ConfigError(f"suite definition must be a mapping, got {type(raw).__name__}")
    return _build_suite(raw)


def _build_suite(raw: dict[str, Any]) -> SuiteConfig:
    tasks = tuple(_build_task(entry) for entry in raw.get("tasks", []))
    return SuiteConfig(
        name=str(raw.get("suite", "unnamed")),
        description=str(raw.get("description", "")),
        tasks=tasks,
    )


def _build_task(raw: dict[str, Any]) -> TaskConfig:
    return TaskConfig(
        name=str(raw["name"]),
        type=str(raw["type"]),
        manifest=str(raw["manifest"]),
        adapter=str(raw.get("adapter", "dummy")),
        adapter_options=dict(raw.get("adapter_options", {})),
        metrics=tuple(raw.get("metrics", [])),
        metric_options=dict(raw.get("metric_options", {})),
    )
