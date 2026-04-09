# diapason

A reproducible evaluation harness for audio language models.

`diapason` scores model outputs on three task families — speech recognition
(ASR), sound event detection (SED) and audio question answering (AQA) — and
turns the results into shareable Markdown and JSON reports.

The harness is deliberately model-agnostic. It evaluates predictions, so you can
score a manifest of stored predictions, or plug in an adapter that calls a real
model. A deterministic dummy adapter ships with the package so the whole
pipeline runs — and is tested — without any model or audio files.

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

List the registered adapters and metrics:

```bash
diapason list
```

## Task families

| Type | Task | Metrics |
| --- | --- | --- |
| `asr` | Speech recognition | `wer`, `cer` |
| `sed` | Sound event detection | `event_f1` |
| `aqa` | Audio question answering | `exact_match`, `token_f1` |

## Reproducibility

Every report records the suite name, a fingerprint of the suite config, the
diapason version and a timestamp, so a result can be traced back to the exact
definition that produced it.

See `docs/usage.md` for the manifest schema and `docs/architecture.md` for how
the pieces fit together.
