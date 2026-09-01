#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 {build-benchmark|search|generate|score|metrics} [args...]" >&2
  exit 2
fi

exec python -m opbench.cli "$@"

