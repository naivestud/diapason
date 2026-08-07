"""Compare two adapter configurations on the same ASR task.

Runs the bundled ASR manifest through dummy adapters with increasing corruption
and prints the resulting word error rate for each. It shows how the same task
can be scored under different adapter configurations without touching the data.

Usage:
    python examples/compare_adapters.py
"""

from __future__ import annotations

from pathlib import Path

from diapason.config import TaskConfig
from diapason.runner import run_task

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "examples" / "data" / "asr.jsonl"


def _wer(corruption: float) -> float:
    task = TaskConfig(
        name=f"asr-corrupt-{corruption}",
        type="asr",
        manifest=str(MANIFEST),
        adapter="dummy",
        adapter_options={"corruption": corruption},
        metrics=("wer",),
        metric_options={"normalize": True},
    )
    return run_task(task).scores["wer"]


def main() -> None:
    for corruption in (0.0, 0.2, 0.5):
        print(f"corruption={corruption:<4} wer={_wer(corruption):.4f}")


if __name__ == "__main__":
    main()
