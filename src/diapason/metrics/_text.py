"""Shared text normalization used by several metrics.

ASR and QA metrics both want to compare text ignoring case and punctuation, so
the transform lives here rather than being duplicated in each metric module.
"""

from __future__ import annotations

import string

_PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)


def normalize_text(text: str) -> str:
    """Lowercase ``text``, strip punctuation and collapse runs of whitespace."""
    lowered = text.lower().translate(_PUNCTUATION_TABLE)
    return " ".join(lowered.split())
