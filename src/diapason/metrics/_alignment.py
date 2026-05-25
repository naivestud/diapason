"""Sequence alignment used by the error-rate metrics.

This is a standard Levenshtein dynamic program that, instead of returning only
the scalar distance, keeps track of how many substitutions, insertions and
deletions make up the optimal edit script. Those counts are what WER and CER
report. The program rolls two rows instead of materialising the full table, so
memory stays linear in the hypothesis length.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Alignment:
    """Edit-operation counts mapping a hypothesis onto a reference.

    Attributes:
        hits: Reference tokens matched by an identical hypothesis token.
        substitutions: Reference tokens replaced by a different token.
        insertions: Extra hypothesis tokens with no reference counterpart.
        deletions: Reference tokens missing from the hypothesis.
    """

    hits: int
    substitutions: int
    insertions: int
    deletions: int

    @property
    def distance(self) -> int:
        """Total edit distance (substitutions + insertions + deletions)."""
        return self.substitutions + self.insertions + self.deletions


def align_sequences(reference: Sequence[Hashable], hypothesis: Sequence[Hashable]) -> Alignment:
    """Align ``hypothesis`` against ``reference`` and count edit operations.

    Args:
        reference: The ground-truth token sequence.
        hypothesis: The sequence to score.

    Returns:
        An :class:`Alignment` with the operation counts of an optimal edit
        script.
    """
    n = len(reference)
    m = len(hypothesis)

    # Only the previous row of the dynamic program is needed to compute the
    # next one, so keep two rows instead of the full (n + 1) x (m + 1) table.
    # Each cell holds (substitutions, insertions, deletions).
    previous: list[tuple[int, int, int]] = [(0, j, 0) for j in range(m + 1)]
    for i in range(1, n + 1):
        ref_token = reference[i - 1]
        current: list[tuple[int, int, int]] = [(0, 0, i)]  # i deletions
        for j in range(1, m + 1):
            if ref_token == hypothesis[j - 1]:
                current.append(previous[j - 1])
            else:
                substitute = _bump(previous[j - 1], index=0)
                insert = _bump(current[j - 1], index=1)
                delete = _bump(previous[j], index=2)
                current.append(min(substitute, insert, delete, key=sum))
        previous = current

    substitutions, insertions, deletions = previous[m]
    hits = n - substitutions - deletions
    return Alignment(
        hits=hits,
        substitutions=substitutions,
        insertions=insertions,
        deletions=deletions,
    )


def _bump(counts: tuple[int, int, int], index: int) -> tuple[int, int, int]:
    """Return ``counts`` with the operation at ``index`` incremented."""
    bumped = list(counts)
    bumped[index] += 1
    return bumped[0], bumped[1], bumped[2]
