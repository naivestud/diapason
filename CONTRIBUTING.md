# Contributing to diapason

Thanks for your interest in improving diapason. This document explains how to
set up a development environment, run the checks, and submit changes.

## Development setup

diapason requires Python 3.10 or newer. Install the package in editable mode
with the development extras:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Optionally install the pre-commit hooks so formatting and linting run on every
commit:

```bash
pre-commit install
```

## Running the checks

The same commands run in CI, so it is worth running them before pushing:

```bash
pytest                    # tests
ruff check .              # lint
ruff format --check .     # formatting
mypy src/diapason tests   # type checking
```

## Making changes

- Keep changes focused: one logical change per pull request.
- Add or update tests alongside the code they cover.
- Update the docs under `docs/` when behaviour changes.
- Add a line to the `[Unreleased]` section of `CHANGELOG.md` for user-visible
  changes.

## Adding a metric or adapter

Metrics and adapters are registered by name so suite configs can refer to them.

- A **metric** is a callable `(reference, prediction, **options) -> float`
  registered with `@metrics.register("name")` in a module under
  `src/diapason/metrics/`. Import the module from
  `src/diapason/metrics/__init__.py` so it registers on package import.
- An **adapter** is a subclass of `ModelAdapter` registered with
  `@adapters.register("name")` in a module under `src/diapason/adapters/`.
  Import it from `src/diapason/adapters/__init__.py`.

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). Please be
respectful in issues and discussions.
