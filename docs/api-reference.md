# API reference

The public surface of `diapason`. Everything here is importable from the
top-level package or its submodules as shown.

## Config — `diapason.config`

```python
load_suite(path) -> SuiteConfig
```
Parse and validate a YAML suite definition. Raises `ConfigError` on a malformed
or invalid document.

```python
validate_suite(suite) -> None
```
Check a `SuiteConfig` for unknown task types, empty/duplicate names and missing
manifests.

```python
discover_suites(directory) -> list[Path]
```
Return the `*.yaml` / `*.yml` files directly inside a directory, sorted.

`SuiteConfig` has `name`, `description` and `tasks`; `TaskConfig` has `name`,
`type`, `manifest`, `adapter`, `adapter_options`, `metrics` and
`metric_options`.

## Runner — `diapason.runner`

```python
run_suite(suite, base_dir=None) -> RunResult
run_task(task, base_dir=None) -> TaskResult
```
Evaluate a suite or a single task. `base_dir` resolves relative manifest paths.
`TaskResult` carries `name`, `type`, `scores` (metric name to aggregated
value), `num_examples`, `num_errors` and per-example results.

## Metrics — `diapason.metrics`

Speech recognition:

```python
wer(reference, hypothesis, *, normalize=False) -> float
cer(reference, hypothesis, *, normalize=False) -> float
```
`reference` may be a string or a list of acceptable strings (best match wins).

Sound event detection:

```python
event_based_f1(reference, prediction, *, collar=0.0) -> float
score_events(reference, prediction, *, collar=0.0) -> EventScore
psds_lite(reference, prediction, *, collar=0.0, num_thresholds=10) -> float
```
`reference` and `prediction` are sequences of `Event` objects.

Audio question answering:

```python
exact_match(reference, prediction, *, normalize=False) -> float
token_f1(reference, prediction, *, normalize=False) -> float
```

Aggregation:

```python
aggregate_examples(values) -> float
aggregate_tasks(task_scores) -> float
```

## Adapters — `diapason.adapters`

```python
class ModelAdapter(ABC):
    def predict(self, example) -> str: ...
```
Subclass and implement `predict` to add a model. Register it with
`@diapason.registry.adapters.register("name")` and import the module from
`diapason.adapters`.

Built-ins: `DummyAdapter` (`dummy`) and `HttpAdapter` (`http`).

## Reports — `diapason.report`

```python
write_json_report(result, path, metadata) -> Path
write_markdown_report(result, path, metadata) -> Path
render_markdown(result, metadata) -> str
build_metadata(suite) -> ReportMetadata
config_fingerprint(suite) -> str
```

## Manifests — `diapason.datasets.manifest`

```python
load_manifest(path, task_type) -> list[Example]
```
Read a JSONL manifest. Raises `ManifestError` on malformed input.
