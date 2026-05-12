"""Orchestrates evaluation and collects results.

The runner walks a suite's tasks, loads each manifest, produces predictions
(with an adapter when they are not stored in the manifest), scores them, and
returns structured results that the report writer can render.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import diapason.adapters
import diapason.metrics  # noqa: F401  (registers built-in metrics)
from diapason.adapters.base import ModelAdapter
from diapason.config import SuiteConfig, TaskConfig
from diapason.datasets.manifest import Example, load_manifest
from diapason.metrics.aggregate import aggregate_examples
from diapason.registry import adapters, metrics


@dataclass(frozen=True)
class ExampleResult:
    """Scores for a single example.

    Attributes:
        example_id: The example's manifest id.
        scores: Mapping of metric name to the value scored for this example.
        error: A message describing why this example could not be scored, or
            ``None`` when scoring succeeded.
    """

    example_id: str
    scores: dict[str, float] = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True)
class TaskResult:
    """Aggregated scores for one task."""

    name: str
    type: str
    scores: dict[str, float] = field(default_factory=dict)
    num_examples: int = 0
    num_errors: int = 0
    example_results: tuple[ExampleResult, ...] = ()


@dataclass(frozen=True)
class RunResult:
    """Results for every task in a suite."""

    suite: str
    tasks: tuple[TaskResult, ...] = ()


def run_suite(suite: SuiteConfig, base_dir: Path | None = None) -> RunResult:
    """Evaluate every task in a suite.

    Args:
        suite: The suite definition to run.
        base_dir: Directory relative manifests paths are resolved against.

    Returns:
        A :class:`RunResult` holding one :class:`TaskResult` per task.
    """
    task_results = tuple(run_task(task, base_dir) for task in suite.tasks)
    return RunResult(suite=suite.name, tasks=task_results)


def resolve_manifest_path(manifest: str, base_dir: Path | None = None) -> Path:
    """Resolve a task's manifest path against an optional base directory.

    Absolute paths are returned unchanged; relative paths are joined onto
    ``base_dir`` when one is given, which is how suite configs reference
    manifests that live next to the config file.
    """
    manifest_path = Path(manifest)
    if base_dir is not None and not manifest_path.is_absolute():
        manifest_path = base_dir / manifest_path
    return manifest_path


def run_task(task: TaskConfig, base_dir: Path | None = None) -> TaskResult:
    """Evaluate a single task and aggregate its scores.

    Args:
        task: The task definition from a suite config.
        base_dir: Directory the manifest path is resolved against when it is
            relative. Useful when the suite config lives next to its data.

    Returns:
        A :class:`TaskResult` with one aggregated score per configured metric.
    """
    examples = load_manifest(resolve_manifest_path(task.manifest, base_dir), task.type)

    adapter = None if task.type == "sed" else _build_adapter(task)
    metric_fns = {name: metrics.get(name) for name in task.metrics}

    example_results: list[ExampleResult] = []
    num_errors = 0
    for example in examples:
        result = _score_example(example, task, adapter, metric_fns)
        if result.error is not None:
            num_errors += 1
        example_results.append(result)

    aggregated = {
        metric_name: aggregate_examples(
            [result.scores[metric_name] for result in example_results if result.error is None]
        )
        for metric_name in metric_fns
    }
    return TaskResult(
        name=task.name,
        type=task.type,
        scores=aggregated,
        num_examples=len(example_results),
        num_errors=num_errors,
        example_results=tuple(example_results),
    )


def _score_example(
    example: Example,
    task: TaskConfig,
    adapter: ModelAdapter | None,
    metric_fns: dict[str, Callable[..., Any]],
) -> ExampleResult:
    """Score one example, converting any failure into an error result."""
    try:
        reference, prediction = _reference_and_prediction(example, task, adapter)
        scores = {
            metric_name: fn(reference, prediction, **task.metric_options)
            for metric_name, fn in metric_fns.items()
        }
        return ExampleResult(example_id=example.id, scores=scores)
    except Exception as exc:
        return ExampleResult(example_id=example.id, error=str(exc))


def _build_adapter(task: TaskConfig) -> ModelAdapter:
    """Instantiate the adapter named by the task with its options."""
    factory = adapters.get(task.adapter)
    return cast(ModelAdapter, factory(**task.adapter_options))


def _reference_and_prediction(
    example: Example, task: TaskConfig, adapter: ModelAdapter | None
) -> tuple[Any, Any]:
    """Pull the reference and prediction for an example given the task type."""
    if task.type == "sed":
        return example.reference_events, example.prediction_events
    prediction = example.prediction_text
    if prediction is None and adapter is not None:
        prediction = adapter.predict(example)
    reference: Any = example.reference_text
    if example.reference_alternatives:
        reference = [example.reference_text, *example.reference_alternatives]
    return reference, prediction
