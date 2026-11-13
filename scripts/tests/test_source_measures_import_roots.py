"""The scripts pseudo-member's extra import roots are the directories below
scripts/ that this directory's conftest.py puts on ``sys.path`` (task 6458).

The census names a module by each root it imports from, so a root added to
``sys.path`` but not to ``source_measures`` would leave its modules uncounted.
"""
import sys
from pathlib import Path

from source_measures import PSEUDO_MEMBERS

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
_TESTS_DIR = _SCRIPTS_DIR / 'tests'


def test_the_extra_import_roots_are_the_sys_path_entries_below_scripts():
    below_scripts = {
        path
        for entry in sys.path
        if entry
        and (path := Path(entry).resolve()).is_relative_to(_SCRIPTS_DIR)
        and path != _SCRIPTS_DIR
        and not path.is_relative_to(_TESTS_DIR)
    }
    (scripts,) = [member for member in PSEUDO_MEMBERS if member.name == 'scripts']
    assert below_scripts == {_SCRIPTS_DIR.parent / root for root in scripts.extra_import_roots}
