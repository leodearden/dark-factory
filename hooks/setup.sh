#!/usr/bin/env bash
# Generic git-hooks bootstrap.  Copy hooks/ into any repo then run:
#
#   bash hooks/setup.sh
#
# What it does:
#   1. Points core.hooksPath at hooks/ so pre-commit (and future hooks)
#      are picked up automatically.
#   2. Ensures .task/ is in .gitignore (agent scratch dir — never belongs
#      in a repo).
#   3. If .task/ is already tracked, removes it from the index.

set -euo pipefail
ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$ROOT"

# --- hooksPath ---------------------------------------------------------
git config core.hooksPath hooks
echo "hooks: core.hooksPath set to hooks/"

# --- .task/ gitignore ---------------------------------------------------
if ! grep -qxF '.task/' .gitignore 2>/dev/null; then
    echo '.task/' >> .gitignore
    echo "hooks: added .task/ to .gitignore"
fi

# --- .task/ index cleanup -----------------------------------------------
if git ls-files --cached --error-unmatch .task/ >/dev/null 2>&1; then
    git rm -r --cached .task/
    echo "hooks: removed tracked .task/ from index (was committed before .gitignore)"
    echo "       Run 'git commit' to finalise the removal."
fi

echo "hooks: setup complete."
