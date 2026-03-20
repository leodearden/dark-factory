#!/usr/bin/env bash
# Point git at the tracked hooks directory so pre-commit (and any
# future hooks) are picked up automatically.
#
# Run once after cloning:
#   bash hooks/setup.sh

set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
git config core.hooksPath hooks
echo "Git hooks path set to hooks/"
