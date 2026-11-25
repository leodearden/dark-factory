"""A scratch project root that a real ``ReviewCheckpoint`` agrees to review.

``review_checkpoint.py::ReviewCheckpoint._run_review`` refuses any
``project_root`` containing ``/tmp/pytest``, so pytest's ``tmp_path`` cannot
stand in for one.  This module is the tests' one copy of that refusal.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

_REFUSED_FRAGMENT = '/tmp/pytest'


@contextmanager
def review_scratch_root(prefix: str, name: str = 'project') -> Iterator[Path]:
    """Yield an empty, created ``<scratch>/<name>`` outside pytest's basetemp.

    The scratch directory, reachable as the yielded path's ``parent``, is
    removed on exit.
    """
    with tempfile.TemporaryDirectory(prefix=prefix) as scratch:
        root = Path(scratch) / name
        root.mkdir()
        assert _REFUSED_FRAGMENT not in str(root), (
            f'{root} contains {_REFUSED_FRAGMENT!r}, which '
            'ReviewCheckpoint._run_review refuses as a project_root; a review '
            'scratch root must live outside pytest basetemp'
        )
        yield root
