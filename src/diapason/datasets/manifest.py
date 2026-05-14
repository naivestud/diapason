"""Dataset manifest loading.

A manifest is a JSONL file: one evaluation example per line. Every line names
the example and carries the reference annotation the metrics score against.
Depending on the task type, the reference is a string (ASR, AQA) or a list of
labelled events (SED). Predictions may be stored ahead of time in the same
file; when they are absent an adapter generates them at run time.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from diapason.config import TASK_TYPES
from diapason.metrics.sed import Event, parse_events


class ManifestError(ValueError):
    """Raised when a manifest cannot be read or is malformed."""


@dataclass(frozen=True)
class Example:
    """A single evaluation example parsed from a manifest line."""

    id: str
    audio: str | None = None
    prompt: str | None = None
    reference_text: str | None = None
    reference_alternatives: tuple[str, ...] = ()
    prediction_text: str | None = None
    reference_events: tuple[Event, ...] | None = None
    prediction_events: tuple[Event, ...] | None = None
    meta: dict[str, object] = field(default_factory=dict)


def load_manifest(path: str | Path, task_type: str) -> list[Example]:
    """Read a JSONL manifest into a list of examples.

    Args:
        path: Path to a ``*.jsonl`` manifest.
        task_type: One of ``asr``, ``sed`` or ``aqa``; controls how the
            reference and prediction fields are interpreted.

    Returns:
        The parsed examples, in file order.

    Raises:
        ManifestError: If the file is empty, a line is not valid JSON, or an
            example is missing a field its task type requires.
    """
    if task_type not in TASK_TYPES:
        raise ManifestError(f"unknown task type '{task_type}'")
    examples: list[Example] = []
    text = Path(path).read_text(encoding="utf-8")
    for line_number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ManifestError(f"{path}:{line_number}: invalid JSON ({exc})") from exc
        if not isinstance(row, Mapping):
            raise ManifestError(f"{path}:{line_number}: expected a JSON object")
        try:
            example = _parse_row(dict(row), task_type)
        except ValueError as exc:
            raise ManifestError(f"{path}:{line_number}: {exc}") from exc
        _validate(example, task_type, line_number)
        examples.append(example)
    if not examples:
        raise ManifestError(f"manifest {path} contains no examples")
    return examples


def _validate(example: Example, task_type: str, line_number: int) -> None:
    """Check that an example carries the fields its task type needs."""
    where = f"line {line_number} (example '{example.id}')"
    if task_type == "sed":
        if example.reference_events is None:
            raise ManifestError(f"{where} is missing SED reference events")
    elif example.reference_text is None:
        raise ManifestError(f"{where} is missing a reference")
    if task_type == "aqa" and example.prompt is None:
        raise ManifestError(f"{where} is missing a question")


def _parse_row(row: dict[str, Any], task_type: str) -> Example:
    """Build an :class:`Example` from one manifest row."""
    if "id" not in row:
        raise ManifestError("manifest row is missing required key 'id'")

    reference = row.get("reference")
    prediction = row.get("prediction")

    reference_events: tuple[Event, ...] | None = None
    prediction_events: tuple[Event, ...] | None = None
    reference_text: str | None = None
    reference_alternatives: tuple[str, ...] = ()
    prediction_text: str | None = None

    if task_type == "sed":
        if reference is not None:
            reference_events = parse_events(reference)
        if prediction is not None:
            prediction_events = parse_events(prediction)
    else:
        if reference is not None:
            if isinstance(reference, list):
                texts = [str(item) for item in reference]
                reference_text = texts[0] if texts else None
                reference_alternatives = tuple(texts[1:])
            else:
                reference_text = str(reference)
        if prediction is not None:
            prediction_text = str(prediction)

    known = {"id", "audio", "prompt", "question", "reference", "prediction"}
    meta = {key: value for key, value in row.items() if key not in known}

    return Example(
        id=str(row["id"]),
        audio=_optional_str(row.get("audio")),
        prompt=_optional_str(row.get("question", row.get("prompt"))),
        reference_text=reference_text,
        reference_alternatives=reference_alternatives,
        prediction_text=prediction_text,
        reference_events=reference_events,
        prediction_events=prediction_events,
        meta=meta,
    )


def _optional_str(value: Any) -> str | None:
    """Return ``value`` as a string, or ``None`` when it is absent."""
    return None if value is None else str(value)
