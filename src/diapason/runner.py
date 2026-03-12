"""Orchestrates evaluation and collects results.

The runner walks a suite's tasks, loads each manifest, produces predictions
(with an adapter when they are not stored in the manifest), scores them, and
returns structured results that the report writer can render.
"""

from __future__ import annotations

from dataclasses import dataclass, field


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
