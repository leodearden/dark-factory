#!/bin/bash
# Verify the task 845 proc_group fix from a TTY (NOT inside the Plasma session).
#
# This runs the full proc_group-related test surface.  If the fix is
# incomplete and killpg still reaches the user systemd manager, a TTY
# session losing its shell is recoverable; the Plasma session is not.
#
# Run from a real TTY: Ctrl+Alt+F3, log in, then:
#     /home/leo/src/dark-factory/.worktrees/845/scripts/verify-task-845-tty.sh
#
# Each subproject has its own pyproject.toml with [tool.pytest.ini_options];
# pytest must be invoked from inside each subproject to avoid
# ImportPathMismatchError on the identically-named tests/conftest.py files.

set -euo pipefail

WORKTREE=/home/leo/src/dark-factory/.worktrees/845
PY="$WORKTREE/.venv/bin/python"

echo "=== branch: $(git -C "$WORKTREE" rev-parse --abbrev-ref HEAD) ==="
echo "=== head:   $(git -C "$WORKTREE" rev-parse --short HEAD) ==="
echo

echo
echo "========================================================"
echo "shared/tests — proc_group, cli_invoke, usage_gate"
echo "========================================================"
cd "$WORKTREE/shared"
"$PY" -m pytest \
    tests/test_proc_group.py \
    tests/test_cli_invoke.py::TestRunSubprocessProcessGroup \
    tests/test_usage_gate.py::TestUsageGateProbeProcessGroup \
    -x -v

echo
echo "========================================================"
echo "orchestrator/tests — shutdown, verify, mcp_retry, steward"
echo "========================================================"
cd "$WORKTREE/orchestrator"
"$PY" -m pytest \
    tests/test_shutdown.py \
    tests/test_verify.py::TestRunCmdProcessGroup \
    tests/test_mcp_retry.py::TestMcpLifecycleProcessGroup \
    tests/test_steward.py::TestStewardWatcherProcessGroup \
    -x -v

echo
echo "All proc_group test surfaces GREEN."
