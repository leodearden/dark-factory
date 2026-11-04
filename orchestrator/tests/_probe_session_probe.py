"""The test that test_probe_session.py runs in a nested pytest session; not collected by the suite (no ``test_`` prefix)."""
import os
from pathlib import Path


def test_records_the_temp_root_it_inherited_in_its_tmp_path(tmp_path: Path) -> None:
    (tmp_path / 'inherited-temproot').write_text(os.environ.get('PYTEST_DEBUG_TEMPROOT', ''))
