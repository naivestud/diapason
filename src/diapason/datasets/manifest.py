"""Dataset manifest loading.

A manifest is a JSONL file: one evaluation example per line. Each line names
the example, optionally points at an audio file and a prompt, and carries the
reference annotation the metrics score against.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Example:
    """A single evaluation example parsed from a manifest line."""

    id: str
    audio: str | None = None
    prompt: str | None = None
    reference_text: str | None = None
    prediction_text: str | None = None
    meta: dict[str, object] = field(default_factory=dict)
