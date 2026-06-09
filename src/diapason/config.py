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

from diapason.metrics.aggregate import AGGREGATION_METHODS
from diapason.registry import Registry, RegistryError

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
        aggregate: How per-example scores collapse into one task score;
            one of :data:`~diapason.metrics.aggregate.AGGREGATION_METHODS`.
    """

    name: str
    type: str
    manifest: str
    adapter: str = "dummy"
    adapter_options: dict[str, Any] = field(default_factory=dict)
    metrics: tuple[str, ...] = ()
    metric_options: dict[str, Any] = field(default_factory=dict)
    aggregate: str = "mean"


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
        raise ConfigError(f"{path}: suite definition must be a mapping, got {type(raw).__name__}")
    suite = _build_suite(raw)
    validate_suite(suite)
    return suite


def validate_suite(suite: SuiteConfig) -> None:
    """Check a suite definition for common mistakes.

    Args:
        suite: The suite to validate.

    Raises:
        ConfigError: If the suite has no tasks, or a task has an unknown type,
            an empty name or manifest, or a name that collides with another
            task in the same suite.
    """
    if not suite.tasks:
        raise ConfigError(f"suite '{suite.name}' has no tasks")
    seen: set[str] = set()
    for task in suite.tasks:
        if not task.name:
            raise ConfigError("task name must not be empty")
        if task.name in seen:
            raise ConfigError(f"duplicate task name '{task.name}' in suite '{suite.name}'")
        seen.add(task.name)
        if task.type not in TASK_TYPES:
            raise ConfigError(
                f"task '{task.name}' has unknown type '{task.type}' "
                f"(expected one of {', '.join(TASK_TYPES)})"
            )
        if not task.manifest:
            raise ConfigError(f"task '{task.name}' is missing a manifest path")
        if task.aggregate not in AGGREGATION_METHODS:
            raise ConfigError(
                f"task '{task.name}' has unknown aggregation method '{task.aggregate}' "
                f"(expected one of {', '.join(AGGREGATION_METHODS)})"
            )


def validate_components(
    suite: SuiteConfig,
    adapter_registry: Registry[Any],
    metric_registry: Registry[Any],
) -> None:
    """Check that every adapter and metric named by the suite is registered.

    This is a separate step from :func:`validate_suite` because it needs the
    registries to be populated first: import :mod:`diapason.adapters` and
    :mod:`diapason.metrics` (or your own component modules) before calling it.

    Args:
        suite: The suite to validate.
        adapter_registry: Registry holding the available adapters.
        metric_registry: Registry holding the available metrics.

    Raises:
        ConfigError: If a task names an adapter or metric that is not
            registered. The message names the task and lists known names.
    """
    for task in suite.tasks:
        try:
            adapter_registry.get(task.adapter)
        except RegistryError as exc:
            raise ConfigError(f"task '{task.name}': {exc.args[0]}") from exc
        for metric in task.metrics:
            try:
                metric_registry.get(metric)
            except RegistryError as exc:
                raise ConfigError(f"task '{task.name}': {exc.args[0]}") from exc


def dump_suite(suite: SuiteConfig) -> str:
    """Serialize a suite definition back to a YAML document.

    This is the inverse of :func:`load_suite`: ``load_suite`` of the returned
    text yields an equivalent suite. Useful for generating suite configs
    programmatically.

    Args:
        suite: The suite to serialize.

    Returns:
        A YAML document string with a trailing newline.
    """
    document: dict[str, Any] = {
        "suite": suite.name,
        "tasks": [_task_to_dict(t) for t in suite.tasks],
    }
    if suite.description:
        document["description"] = suite.description
    return yaml.safe_dump(document, sort_keys=False)


def _task_to_dict(task: TaskConfig) -> dict[str, Any]:
    """Convert a task back into the mapping shape load_suite understands."""
    entry: dict[str, Any] = {"name": task.name, "type": task.type, "manifest": task.manifest}
    if task.adapter != "dummy":
        entry["adapter"] = task.adapter
    if task.adapter_options:
        entry["adapter_options"] = dict(task.adapter_options)
    if task.metrics:
        entry["metrics"] = list(task.metrics)
    if task.metric_options:
        entry["metric_options"] = dict(task.metric_options)
    if task.aggregate != "mean":
        entry["aggregate"] = task.aggregate
    return entry


def _build_suite(raw: dict[str, Any]) -> SuiteConfig:
    tasks = tuple(_build_task(entry) for entry in raw.get("tasks", []))
    return SuiteConfig(
        name=str(raw.get("suite", "unnamed")),
        description=str(raw.get("description", "")),
        tasks=tasks,
    )


def _build_task(raw: dict[str, Any]) -> TaskConfig:
    return TaskConfig(
        name=_str_field(raw, "name"),
        type=_str_field(raw, "type"),
        manifest=_str_field(raw, "manifest"),
        adapter=_str_field(raw, "adapter", default="dummy"),
        adapter_options=_options_field(raw, "adapter_options"),
        metrics=_names_field(raw, "metrics"),
        metric_options=_options_field(raw, "metric_options"),
        aggregate=_str_field(raw, "aggregate", default="mean"),
    )


def _str_field(raw: dict[str, Any], key: str, default: str = "") -> str:
    """Read ``key`` from a raw task mapping and coerce it to a string."""
    return str(raw.get(key, default))


def _options_field(raw: dict[str, Any], key: str) -> dict[str, Any]:
    """Read an optional keyword-options mapping from a raw task mapping."""
    return dict(raw.get(key, {}))


def _names_field(raw: dict[str, Any], key: str) -> tuple[str, ...]:
    """Read an optional list of names from a raw task mapping."""
    return tuple(raw.get(key, []))


def discover_suites(directory: str | Path) -> list[Path]:
    """Return suite config files found directly inside a directory.

    Args:
        directory: Directory to scan. May not exist, in which case the result
            is empty.

    Returns:
        Paths ending in ``.yaml`` or ``.yml``, sorted for stable output.
    """
    root = Path(directory)
    if not root.is_dir():
        return []
    return sorted(path for path in root.iterdir() if path.suffix in (".yaml", ".yml"))
