# Usage

This guide covers the command line, the suite config format and the manifest
format. For the Python API see [api-reference.md](api-reference.md).

## Command line

```bash
diapason run CONFIG [--output-dir DIR] [--format json|md|both] [-v|-q]
diapason list [--config-dir DIR]
diapason report RESULTS_JSON [--output FILE]
```

- `run` evaluates a suite and writes reports. `--format` controls which report
  files are written (default `both`). `-v` increases logging verbosity and `-q`
  suppresses the summary table.
- `list` prints the registered adapters and metrics; with `--config-dir` it also
  lists the suite files found there.
- `report` re-renders a Markdown report from a saved JSON report.

## Suite config format

A suite is a YAML file with a name, an optional description, and a list of
tasks:

```yaml
suite: my_suite
description: What this suite measures
tasks:
  - name: transcription
    type: asr                    # asr | sed | aqa
    manifest: data/asr.jsonl     # relative to this file's directory
    adapter: dummy               # registered adapter name (text tasks only)
    adapter_options:
      corruption: 0.2
    metrics: [wer, cer]          # registered metric names
    metric_options:
      normalize: true
```

`manifest` paths are resolved relative to the config file's directory, so a
suite can live in one folder and its data in another. `adapter` and
`adapter_options` are ignored for `sed` tasks, whose predictions must be stored
in the manifest.

## Manifest format

A manifest is a JSONL file: one example per line. Every example needs an `id`.
The remaining fields depend on the task type.

### ASR

```json
{"id": "a1", "reference": "the cat sat"}
{"id": "a2", "reference": ["the cat sat", "a cat sat"], "prediction": "the cat sat"}
```

`reference` is a transcription, or a list of acceptable transcriptions (the
best match is scored). `prediction` is optional; when absent the task's adapter
generates it.

### SED

```json
{"id": "s1", "reference": [{"onset": 0.5, "offset": 2.0, "label": "dog"}],
 "prediction": [{"onset": 0.6, "offset": 2.1, "label": "dog", "score": 0.9}]}
```

Events are `{onset, offset, label}` objects in seconds. Predicted events may
carry a confidence `score`, used by `psds_lite`.

### AQA

```json
{"id": "q1", "question": "What animal is it?", "reference": "a dog",
 "prediction": "a dog"}
```

`question` is required; `reference` is the ground-truth answer and `prediction`
is optional.

## Bundled configs and data

The `configs/` directory holds three smoke suites (`asr_smoke`, `sed_smoke`,
`aqa_smoke`) that point at the manifests in `examples/data/`. Run one with:

```bash
diapason run configs/asr_smoke.yaml
```
