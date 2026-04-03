"""Run a small ASR evaluation with the dummy adapter and print the scores.

This example needs no model and no audio files: the dummy adapter derives its
prediction from the reference annotation, so it runs anywhere.

Usage:
    python examples/run_asr.py
"""

from __future__ import annotations

from pathlib import Path

from diapason.config import load_suite
from diapason.runner import run_suite


def main() -> None:
    config_path = Path(__file__).resolve().parent.parent / "configs" / "asr_smoke.yaml"
    suite = load_suite(config_path)
    result = run_suite(suite, base_dir=config_path.parent)
    for task in result.tasks:
        print(f"{task.name} ({task.type}):")
        for metric, value in sorted(task.scores.items()):
            print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()
