"""A one-commit git repo whose head sha no sibling repo shares.

``_fm_helpers._init_git_repo`` makes byte-identical commits when it is called
twice within the same second, so two repos built with it can share a head sha.
A cross-repo commit lookup then cannot show which repo answered. Writing a
file named after the repo makes each sha distinct.

Lives beside ``_fm_helpers.py`` under the convention documented in
tests/conftest.py, which puts this directory on ``sys.path`` for ``tests/`` and
``tests/server/`` alike.
"""

from pathlib import Path

from _fm_helpers import _init_git_repo


def init_distinct_git_repo(root: Path) -> str:
    """Create *root* as a one-commit repo unique to its name; return the head sha."""
    root.mkdir()
    (root / 'repo-name.txt').write_text(f'{root.name}\n')
    return _init_git_repo(root)
