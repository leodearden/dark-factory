"""The lease-dir isolation fixture, defined once for the modules that need it.

`import`-and-re-export rather than a `conftest.py` fixture, because an
autouse fixture in `conftest.py` would apply to EVERY test in the package —
including the real ``-m integration`` lane, whose whole job is to publish its
lease into the machine-global directory the live 6-hourly cron reads.
Redirecting that lane would silently disable the guard task 4775 exists to
build.  So the isolation is opt-in per module, and every module that opts in
takes the SAME definition:

    from _fm_lease_dir_fixture import lease_dir_fixture  # noqa: F401

pytest registers an autouse fixture bound into a module's namespace by
import exactly as if it had been defined there, so the behaviour is
identical to the five verbatim copies this replaces — with one definition to
keep correct instead of five that can drift apart.

The ATTRIBUTE is `lease_dir_fixture` while the fixture NAME stays
`lease_dir` (pytest reads the name off the marker when one is given).  That
is not cosmetic: a module-level binding called `lease_dir` is shadowed by
every `def test_x(self, lease_dir)` parameter in the importing module, which
ruff reports as F811 once per test — 27 of them across the five importers,
each needing a `noqa` that would then have to be kept in step by hand.

A sibling module rather than a new section of `_fm_helpers.py`, which is
where `conftest.py` directs shared test helpers: task 4775 holds no lock on
that file, and creating a file cannot collide with another task's concurrent
edits while editing a shared one can.  Folding it in is a mechanical move
whenever `_fm_helpers.py` is next open.
"""
from __future__ import annotations

import functools
from pathlib import Path

import pytest

_REAPER = Path(__file__).parent.parent / 'scripts' / 'cleanup_test_collections.py'


@functools.cache
def _lease_dir_env() -> str:
    """The env var name, read from the reaper that defines it.

    Read rather than spelled out here so a rename of the constant cannot
    leave this fixture setting a variable nothing consults any more — which
    would look exactly like isolation while every test fell through to the
    real machine-global directory.

    Loaded through `_fm_helpers.load_script_module` (`scripts/` is not a
    package), lazily and cached, and only the resulting STRING is kept: the
    modules that import this fixture load the reaper through their own
    loaders, and holding a second reference to a module object they may
    replace is a hazard with nothing to buy it.
    """
    from _fm_helpers import load_script_module  # noqa: PLC0415

    return load_script_module(_REAPER).LEASE_DIR_ENV


@pytest.fixture(autouse=True, name='lease_dir')
def lease_dir_fixture(tmp_path, monkeypatch):
    """Point ``DF_EPHEMERAL_COLLECTION_LEASE_DIR`` at a per-test directory.

    A hard isolation boundary, not a convenience.  The lease directory
    ``cleanup_test_collections.lease_dir()`` returns by default is a
    HARDCODED machine-global absolute path (that is the property the guard's
    correctness rests on — see the design note on that function), and it is
    the very directory the live 6-hourly cron reads.  A test that wrote a
    lease into it would hold a real sweep off this host; a test that reaped
    it would unlink the lease of a live bake-off running in another checkout.

    Autouse, and applied to EVERY test in an importing module rather than
    only its lease tests, for exactly that reason: a test that forgets to
    request the isolation must not be able to fall through silently to the
    real directory.  Tests that need the path can still request this fixture
    by name; the directory is not created here, because a lease-dir-absent
    case is one of the behaviours under test.
    """
    directory = tmp_path / 'ephemeral-collection-leases'
    monkeypatch.setenv(_lease_dir_env(), str(directory))
    return directory
