# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-08-08

### Added

- Multiple reference transcriptions per example; WER and CER report the best
  match.
- `--quiet` and `--verbose` flags on the `run` command.

### Changed

- The `run` command resolves the output directory to an absolute path and
  creates it before writing reports.

## [0.2.0] - 2026-08-08

### Added

- HTTP adapter that calls an OpenAI-compatible chat endpoint, with retries and
  exponential backoff.
- `psds_lite` metric approximating the polyphonic sound detection score.
- `compare_adapters.py` example scoring one task under several adapter
  configurations.

## [0.1.0] - 2026-08-08

### Added

- Suite configs (YAML) describing ASR, SED and AQA tasks.
- JSONL manifest loading with per-task validation.
- Word error rate and character error rate metrics for speech recognition.
- Event-based F1 with a collar tolerance for sound event detection.
- Exact match and token F1 for audio question answering.
- Pluggable adapter and metric registries.
- A deterministic dummy adapter for tests and examples.
- A runner that scores tasks and aggregates results per task and per suite.
- Markdown and JSON report generation with reproducibility metadata.
- A `diapason` CLI with `run`, `report` and `list` subcommands.
