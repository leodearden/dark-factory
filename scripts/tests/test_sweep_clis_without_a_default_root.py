"""A discovery-driven sweep CLI run from a copy outside any checkout, with no root
named, exits with its no-root code 2 and says why, rather than a traceback read as a verdict.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from _task_db_scan import (
    AUDIT_EXIT_NO_ROOT,
    NO_DB_RESOLVED_MESSAGE,
    NO_PROJECT_ROOT_RESOLVED_MESSAGE,
)
from cli_subprocess_timeout import cli_timeout_from_env
from linked_worktree_copy import checkout_copy, probe_env
from shared.project_layout import KNOWN_PROJECT_ROOTS_ENV
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

REPO_ROOT = Path(__file__).resolve().parents[2]

CLI_TIMEOUT = cli_timeout_from_env('SWEEP_CLIS_WITHOUT_A_DEFAULT_ROOT_TEST_TIMEOUT')

ROWS = [
    ('scan_task_toolcall_leaks', NO_DB_RESOLVED_MESSAGE),
    ('scan_provenance_note_log_leaks', NO_DB_RESOLVED_MESSAGE),
    ('audit_wiped_metadata_files', NO_PROJECT_ROOT_RESOLVED_MESSAGE),
    ('audit_combine_gate_marker_loss', NO_PROJECT_ROOT_RESOLVED_MESSAGE),
    ('census_tagger_debris', NO_PROJECT_ROOT_RESOLVED_MESSAGE),
    ('repair_wiped_metadata_files', NO_PROJECT_ROOT_RESOLVED_MESSAGE),
]


@pytest.fixture(scope='module')
def loose_copy(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return checkout_copy(tmp_path_factory.mktemp('no-default-root').resolve() / 'copy', REPO_ROOT)


def _env_naming_no_root() -> dict[str, str]:
    env = probe_env()
    env.pop(KNOWN_PROJECT_ROOTS_ENV, None)
    return env


@pytest.mark.parametrize(('script', 'no_root_message'), ROWS, ids=[script for script, _ in ROWS])
def test_an_underivable_default_root_is_the_no_root_exit(
    loose_copy: Path, script: str, no_root_message: str
) -> None:
    proc = subprocess.run(
        [sys.executable, '-s', str(loose_copy / 'scripts' / f'{script}.py')],
        cwd=loose_copy.parent,
        env=_env_naming_no_root(),
        capture_output=True,
        text=True,
        timeout=CLI_TIMEOUT,
        check=False,
    )

    assert proc.returncode == AUDIT_EXIT_NO_ROOT, f'exited {proc.returncode}\n{proc.stderr}'
    assert 'no git checkout encloses' in proc.stderr
    assert no_root_message in proc.stderr
    assert 'Traceback' not in proc.stderr
