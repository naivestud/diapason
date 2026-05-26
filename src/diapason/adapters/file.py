"""Adapter that replays precomputed predictions stored in a JSONL file.

Sometimes predictions are produced ahead of time by another tool and dumped to
a file rather than generated during the run. The file adapter loads such a
dump once at construction and looks each example up by id, so scoring never
calls a model. Each line of the file is a JSON object carrying an example id
and its prediction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from diapason.adapters.base import ModelAdapter
from diapason.registry import adapters

if TYPE_CHECKING:
    from diapason.datasets.manifest import Example


@adapters.register("file")
class FileAdapter(ModelAdapter):
    """Serve predictions from a JSONL file keyed by example id.

    Args:
        path: Path to a ``*.jsonl`` file; one prediction per line.
        id_field: Key holding the example id on each line.
        prediction_field: Key holding the prediction text on each line.
    """

    name = "file"

    def __init__(
        self,
        path: str,
        id_field: str = "id",
        prediction_field: str = "prediction",
        **options: Any,
    ) -> None:
        super().__init__(path=path, id_field=id_field, prediction_field=prediction_field, **options)
        self.path = path
        self.id_field = id_field
        self.prediction_field = prediction_field
        self._predictions = self._load(path, id_field, prediction_field)

    def predict(self, example: Example) -> str:
        """Return the stored prediction for the example's id."""
        if example.id not in self._predictions:
            raise KeyError(f"no prediction stored for example '{example.id}' in {self.path}")
        return self._predictions[example.id]

    @staticmethod
    def _load(path: str, id_field: str, prediction_field: str) -> dict[str, str]:
        """Read the JSONL file into a mapping of example id to prediction."""
        predictions: dict[str, str] = {}
        text = Path(path).read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON ({exc})") from exc
            if id_field not in row or prediction_field not in row:
                raise ValueError(
                    f"{path}:{line_number}: row needs '{id_field}' and '{prediction_field}' keys"
                )
            predictions[str(row[id_field])] = str(row[prediction_field])
        return predictions
