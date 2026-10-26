"""``verify_classify.reports_disk_full`` — the one disk-full text predicate
(task 7032), driven through ``classify_failure`` for every ``ToolKind``.

On 2026-10-16 the merge lane held genuine test failures as "transient
infrastructure failure (disk pressure)": the bare ``enospc`` marker matched
test ids, assertion messages and log lines that merely NAME the condition.
Every negative below is text observed in those logs (task 7037); every
positive is an OS error rendering of a real ENOSPC.
"""

from __future__ import annotations

import pytest

from orchestrator.verify_categories import FailureCategory
from orchestrator.verify_classify import classify_failure, reports_disk_full
from orchestrator.verify_cmd import ToolKind

NAMES_THE_CONDITION = {
    'row6-assertion': (
        'E   AssertionError: expected the BOUNDED shared enospc/infra retry budget '
        '(one in-place retry) to exhaust rather than retry silently forever; '
        'got 0 call(s)'
    ),
    'failed-summary-line': (
        'FAILED tests/test_merge_queue.py::TestEnospcTransientInfraRetry::'
        'test_persistent_enospc_prunes_retries_once_then_blocks'
    ),
    'xdist-collection-diff': (
        '-tests/test_verify_classify.py::TestEnvironmentalGuardsApplyToEveryToolKind::'
        'test_enospc_is_disk_full[pytest]'
    ),
    'worker-log-line': (
        'post-merge verify hit ENOSPC; pruned 0 stale merge worktree(s), '
        'retrying verify once'
    ),
    'node-inotify': (
        "Error: ENOSPC: System limit for number of file watchers reached, watch '/repo/src'"
    ),
    'watchdog-inotify': 'OSError: [Errno 28] inotify watch limit reached',
}

OS_ERROR_REPORTS = {
    'python': 'E   OSError: [Errno 28] No space left on device',
    'rust': 'error: No space left on device (os error 28)',
    'node': 'ENOSPC: no space left on device, write',
    'git': 'fatal: write failure on standard output: No space left on device',
}


@pytest.mark.parametrize('tool', list(ToolKind))
class TestClassifyFailure:
    @pytest.mark.parametrize(
        'line', NAMES_THE_CONDITION.values(), ids=NAMES_THE_CONDITION.keys(),
    )
    def test_naming_the_condition_is_not_disk_full(self, tool: ToolKind, line: str) -> None:
        output = f'collected 120 items\n{line}\n1 failed, 119 passed in 3.2s\n'
        assert classify_failure(tool, 1, output, False) != FailureCategory.DISK_FULL

    @pytest.mark.parametrize('line', OS_ERROR_REPORTS.values(), ids=OS_ERROR_REPORTS.keys())
    def test_an_os_error_report_is_disk_full(self, tool: ToolKind, line: str) -> None:
        output = f'collected 120 items\n{line}\n1 failed, 119 passed in 3.2s\n'
        assert classify_failure(tool, 1, output, False) == FailureCategory.DISK_FULL


def test_match_ignores_case() -> None:
    assert reports_disk_full('NO SPACE LEFT ON DEVICE') is True
