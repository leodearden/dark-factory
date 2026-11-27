"""The root conftest's autouse roster pin points away from the operator's roster file.

tests/scripts and scripts/tests run with the repo root as rootdir, so the root
conftest.py applies to both. They exercise the real load_config through
scripts/hook_module_checks.py and scripts/restamp_degenerate_branch_bases.py,
and since task 11734 load_config reads the roster. The real constant and
resolver are imported here, so a rename that leaves the conftest literal stale
fails this test.
"""

import os

from orchestrator.verify_host_roster_file import ROSTER_FILE_ENV, roster_file_path


def test_the_autouse_pin_points_away_from_the_default_roster():
    default_path = roster_file_path({k: v for k, v in os.environ.items() if k != ROSTER_FILE_ENV})

    assert roster_file_path(os.environ) != default_path
    assert not roster_file_path(os.environ).exists()
