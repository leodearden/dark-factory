#!/usr/bin/env bash
# merge-deep-canary-predicate.sh -- entry point of the read-only deep
# merge-ahead canary health predicate (task eta1's before_done, kind=predicate).
# The contract (flags, window, legs, stdout, exit codes) lives in the module
# docstring of scripts/merge_deep_canary_predicate.py, which this execs.
set -euo pipefail
exec python3 -I "$(dirname "${BASH_SOURCE[0]}")/merge_deep_canary_predicate.py" "$@"
