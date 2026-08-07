"""Run every bundled smoke suite and write Markdown and JSON reports.

Reports land in the ``results/`` directory at the repository root. Like the
other examples, this needs no model and no audio files.

Usage:
    python examples/run_suite.py
"""

from __future__ import annotations

from pathlib import Path

from diapason.config import load_suite
from diapason.report import build_metadata, write_json_report, write_markdown_report
from diapason.runner import run_suite


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    configs_dir = repo_root / "configs"
    output_dir = repo_root / "results"
    for config_path in sorted(configs_dir.glob("*.yaml")):
        suite = load_suite(config_path)
        result = run_suite(suite, base_dir=config_path.parent)
        metadata = build_metadata(suite))
        write_json_report(result, output_dir / f"{suite.name}.json", metadata)
        write_markdown_report(result, output_dir / f"{suite.name}.md", metadata)
        print(f"evaluated {suite.name}: {len(suite.tasks)} task(s)")


if __name__ == "__main__":
    main()
