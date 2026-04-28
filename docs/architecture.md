# Architecture

diapason is organised as a small pipeline: a suite config names tasks, each
task loads a manifest of examples, an adapter produces predictions (when they
are not already stored), metrics score them, and a report writer renders the
aggregated results.

```
suite config (YAML)
    |
    v
config.py  -- load_suite / validate_suite
    |
    v
runner.py  -- run_suite / run_task
    |            |
    |            +-> datasets/manifest.py  (load examples)
    |            +-> adapters/*            (produce predictions)
    |            +-> metrics/*             (score predictions)
    |            +-> metrics/aggregate.py  (reduce scores)
    v
report.py  -- Markdown + JSON reports with provenance metadata
```

## Components

### `config.py`
Parses and validates suite definitions. A `SuiteConfig` is a name, a
description and a tuple of `TaskConfig` objects. Validation rejects unknown
task types, empty names and manifests, and duplicate task names.

### `datasets/manifest.py`
Reads JSONL manifests into `Example` objects. The meaning of the `reference`
and `prediction` fields depends on the task type: strings for ASR and AQA,
lists of labelled events for SED. Per-task validation ensures required fields
are present.

### `adapters/`
An adapter turns an `Example` into a text prediction. Adapters register
themselves by name (see `registry.py`) so configs can refer to them
symbolically. Two ship with the package:

- `dummy` — echoes the reference, optionally corrupting it deterministically.
  Used by tests and examples so the pipeline runs without a model.
- `http` — calls an OpenAI-compatible chat endpoint using the standard
  library, with retries and exponential backoff.

SED predictions are always read from the manifest; adapters produce text and
therefore only serve ASR and AQA.

### `metrics/`
Each metric is a callable `(reference, prediction, **options) -> float`
registered by name. `_alignment.py` holds the edit-distance dynamic program
shared by WER and CER; `_text.py` holds the normalization shared by ASR and
QA. `aggregate.py` reduces per-example values to a task score and task scores
to a suite score.

### `runner.py`
Orchestrates a task: load the manifest, build the adapter, score every example
and aggregate. A failing example is recorded and skipped rather than aborting
the whole task.

### `report.py`
Serialises a `RunResult` to JSON and renders a Markdown summary with jinja2.
Both carry `ReportMetadata` (suite, timestamp, version, config fingerprint).

### `registry.py`
A tiny generic name-to-factory registry used for both adapters and metrics.

## Design decisions

See [design-notes.md](design-notes.md) for the reasoning behind evaluating
predictions rather than running models, and for the reproducibility choices.
