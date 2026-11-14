"""conftest.py for tests/repo_guards/ — the sys.path insertions its guards import through.

The root pyproject runs pytest with ``--import-mode=importlib``, which keeps a
test file's own directory off sys.path, so each directory below is inserted
here or its imports fail at collection:

1. this directory — the guards' sibling helper modules;
2. ``scripts/`` — ``from legibility import invariants`` in
   test_design_invariants_consistency.py, and ``import governed_list_agreement``
   in test_repo_guards_governed_lists.py;
3. ``shared/tests/cross_suite_helpers/`` — helpers that stay beside shared's own
   unit tests, such as ``capability_manifest_corpus``. The dependency runs one
   way only: nothing in shared/tests imports from here.

Inserted in reverse priority, so this directory wins a basename collision.

It also owns the session's whole-first-party-tree parse: the ``first_party_tree``
fixture every gate here walks, prebuilt in ``pytest_collection_finish``.
"""
from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

for _path in (
    REPO_ROOT / 'shared' / 'tests' / 'cross_suite_helpers',
    REPO_ROOT / 'scripts',
    Path(__file__).resolve().parent,
):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from session_prebuild import SessionPrebuild  # noqa: E402
from silent_fallthrough_scan import ParsedFile, parse_first_party_tree  # noqa: E402

_FIRST_PARTY_TREE = SessionPrebuild(
    'first_party_tree', partial(parse_first_party_tree, REPO_ROOT)
)


def pytest_collection_finish(session):
    """Build the first-party tree here: after collection, before any item's pytest-timeout timer."""
    _FIRST_PARTY_TREE.prebuild_if_requested(session)


@pytest.fixture(scope='session')
def first_party_tree(request: pytest.FixtureRequest) -> tuple[ParsedFile, ...]:
    """The whole first-party source tree, read and parsed ONCE per session.

    The single source every gate module in this directory walks (task 4520).
    The one read+parse happens in ``pytest_collection_finish``, and only when a
    collected item requests this fixture, so it sits outside every item's
    pytest-timeout timer (see ``session_prebuild``). Under xdist each worker
    collects every item, so each worker that collects a requester builds its
    own copy.

    The provider is memoized on the resolved root, so a direct
    ``parse_first_party_tree(REPO_ROOT)`` call returns the same object. Its
    other cost is memory: ~361 MB of ASTs retained until the process exits.
    ``parse_first_party_tree``'s docstring records both halves and why the trade
    is currently made in CPU's favour — read it before tuning either.

    Take it as a fixture argument: a dynamic
    ``request.getfixturevalue('first_party_tree')`` is invisible at collection
    and fails loudly. Consumers walk the ASTs READ-ONLY — they are shared with
    every other gate.
    """
    return _FIRST_PARTY_TREE.result(request.session)
