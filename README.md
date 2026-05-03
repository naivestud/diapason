# diapason

[![CI](https://github.com/naivestud/diapason/actions/workflows/ci.yml/badge.svg)](https://github.com/naivestud/diapason/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org)

A reproducible evaluation harness for audio language models.

`diapason` scores model outputs on three task families — speech recognition
(ASR), sound event detection (SED) and audio question answering (AQA) — and
turns the results into shareable Markdown and JSON reports.

The harness is deliberately model-agnostic. It evaluates *predictions*, so you
can score a manifest of stored predictions, or plug in an adapter that calls a
real model. A deterministic dummy adapter ships with the package so the whole
pipeline runs — and is tested — without any model or audio files.

## Task families

| Type | Task | Metrics |
| --- | --- | --- |
| `asr` | Speech recognition | `wer`, `cer` |
| `sed` | Sound event detection | `event_f1`, `psds_lite` |
| `aqa` | Audio question answering | `exact_match`, `token_f1` |

## Installation

```bash
pip install .
```

For development, install the extras:

```bash
pip install -e ".[dev]"
```

## Quickstart

Run one of the bundled smoke suites:

```bash
diapason run configs/asr_smoke.yaml
```

This evaluates the suite, prints a summary table, and writes Markdown and JSON
reports to `results/`.

Other useful commands:

```bash
diapason list                      # registered adapters and metrics
diapason list --config-dir configs # also show discovered suites
diapason report results/demo.json  # re-render a saved report
```

## Using the Python API

```python
from pathlib import Path

from diapason.config import load_suite
from diapason.report import build_metadata, write_markdown_report
from diapason.runner import run_suite

config_path = Path("configs/asr_smoke.yaml")
suite = load_suite(config_path)
result = run_suite(suite, base_dir=config_path.parent)

for task in result.tasks:
    print(task.name, task.scores)

write_markdown_report(result, "report.md", build_metadata(suite))
```

## Suite configs and manifests

A suite is a small YAML file listing tasks; each task points at a JSONL manifest
of examples. See [`docs/usage.md`](docs/usage.md) for the full schema of both.

## Reproducibility

Every report records the suite name, a fingerprint of the suite config, the
diapason version and a timestamp, so a result can be traced back to the exact
definition that produced it. See [`docs/design-notes.md`](docs/design-notes.md)
for the reasoning behind the design.

## Development

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for setting up a development
environment and running the checks. The short version:

```bash
pytest
ruff check . && ruff format --check .
mypy src/diapason tests
```

## License

MIT — see [`LICENSE`](LICENSE).
