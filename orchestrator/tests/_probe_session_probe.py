"""The test that test_probe_session.py runs in a nested pytest session; not collected by the suite (no ``test_`` prefix)."""
from pathlib import Path


def test_writes_into_its_tmp_path(tmp_path: Path) -> None:
    (tmp_path / 'written').write_text('')
