#!/usr/bin/env bash
# Prepare a release: verify the tree is clean and the checks pass, then build
# distributions. Tagging and pushing are left as explicit final steps so a
# release is never published by accident.
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -n "$(git status --porcelain)" ]]; then
  echo "error: working tree is not clean; commit or stash changes first" >&2
  exit 1
fi

echo "==> running checks"
ruff check .
ruff format --check .
mypy src/diapason tests
pytest -q

echo "==> building distributions"
python -m build

echo "==> current version"
python - <<'PY'
from diapason.version import __version__

print(__version__)
PY

cat <<'EOF'

When satisfied, tag and push:
  git tag -a vX.Y.Z -m "Release vX.Y.Z"
  git push origin main --tags
EOF
