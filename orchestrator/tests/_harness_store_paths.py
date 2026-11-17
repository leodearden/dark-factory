"""Point every sqlite store ``Harness.__init__`` wires at a real directory.

On a ``MagicMock(spec_set=pydantic_spec(OrchestratorConfig))`` each store's
``*_db_path`` @property auto-mocks, and ``sqlite3.connect(str(...))`` of that
mock creates a ``<MagicMock ...>`` file in the CWD, which
``conftest.py::_no_mock_repr_stray_entries`` fails the test for.  A store newly
wired into ``Harness.__init__`` gets one entry in ``_HARNESS_STORE_FILENAMES``.
"""
from pathlib import Path
from unittest.mock import MagicMock

_HARNESS_STORE_FILENAMES = {
    'overrides_db_path': 'overrides.db',
    'park_eviction_requests_db_path': 'park_eviction_requests.db',
    'eligibility_clock_db_path': 'scheduler_eligibility.db',
}


def point_harness_stores_at(config: MagicMock, directory: Path) -> None:
    for attribute, filename in _HARNESS_STORE_FILENAMES.items():
        setattr(config, attribute, directory / filename)
