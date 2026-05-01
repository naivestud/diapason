# Design notes

A few decisions shape how diapason works. They are collected here so the
reasoning is easier to find and to argue with.

## Evaluate predictions, not models

The core loop scores a reference against a prediction. It never loads audio and
never runs a model. That keeps the harness:

- **Fast and testable.** Metrics, manifests and the runner are pure functions of
  text and numbers, so the whole pipeline is unit tested without fixtures for
  audio or model weights.
- **Model-agnostic.** Anything that can fill in a prediction column — a local
  model, a hosted endpoint, a human — can be scored the same way.
- **Re-runnable.** Scoring a stored prediction file is deterministic, so a
  reported number can always be reproduced from its manifest.

Adapters are the extension point for the cases where you *do* want to run a
model. They are small, optional, and kept out of the scoring path.

## Metrics return a single float

Every metric reduces an example to one `float`, so aggregation is uniform: the
mean over examples gives a task score, and the macro mean over tasks gives a
suite score. Metrics that naturally produce more (precision/recall, a full
PSDS curve) expose the richer object as a separate function (`score_events`)
and register only the scalar under the metric name.

## Determinism and reproducibility

- The dummy adapter corrupts output deterministically rather than using a
  random number generator, so examples and tests are repeatable.
- Reports embed a fingerprint of the suite config (a hash of the parsed
  definition) alongside the version and a timestamp. Two reports with the same
  fingerprint were produced from the same suite definition.
- JSON reports are written with sorted keys so identical runs byte-compare
  equal apart from the timestamp.

## Errors are isolated, not fatal

A single example that fails to score (for instance an empty reference) is
recorded on the `TaskResult` and skipped; it does not abort the task. The
number of such errors is surfaced in the report so a suspiciously clean score
is visible.

## Small dependencies

The runtime depends on `pyyaml`, `jinja2`, `click` and `rich`. The HTTP adapter
uses the standard library `urllib` rather than adding an HTTP client, so scoring
a hosted model does not widen the dependency surface.
