"""The escalation suite's autouse roster pin points away from the operator's roster file.

Escalation tests drive the real load_config (test_reload_config_integration.py),
and since task 11734 load_config reads the verify-host roster.
conftest.py::_isolate_verify_host_roster spells the env name out, because the
conftest must collect without orchestrator installed. The real constant and
resolver are imported here, so a rename that leaves the conftest literal stale
fails this test.
"""

import os

import pytest

roster_file = pytest.importorskip('orchestrator.verify_host_roster_file')


def test_the_autouse_pin_points_away_from_the_default_roster() -> None:
    default_path = roster_file.roster_file_path(
        {k: v for k, v in os.environ.items() if k != roster_file.ROSTER_FILE_ENV}
    )

    assert roster_file.roster_file_path(os.environ) != default_path
    assert not roster_file.roster_file_path(os.environ).exists()
