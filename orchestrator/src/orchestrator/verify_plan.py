"""Declarative decision layer for the merge/verify gate (verify-plan-prd.md task γ).

Unifies the twice-fixed scope decision between ``scope_module_config`` and
``_build_fallback_config`` (verify.py) behind a single pure
``derive_verify_plan``: file classification happens EXACTLY ONCE via
``FileKind``, so the class of bug independently fixed in both call sites
(task-1077 conftest, task-1852 data-module) closes by construction.
"""

from __future__ import annotations

import re
from enum import Enum


class FileKind(Enum):
    """The six mutually-exclusive classifications ``classify_file`` assigns a path.

    Precedence (highest to lowest): CONFTEST > COLLECTABLE_TEST > TEST_DATA >
    STRUCTURAL > SOURCE > INERT. TEST_DATA outranks STRUCTURAL so a
    Protocol-defining data module under ``tests/`` still triggers the full
    suite (D1) rather than merely widening pyright — the structural widening
    (D2) only matters for real source files outside the test tree.
    """

    CONFTEST = 'conftest'
    COLLECTABLE_TEST = 'collectable_test'
    TEST_DATA = 'test_data'
    STRUCTURAL = 'structural'
    SOURCE = 'source'
    INERT = 'inert'


# Matches a class that inherits from Protocol or TypedDict (as a base class).
# Deliberately a cheap content grep rather than an AST parse — mirrors
# verify.py's _PROTOCOL_RE/_TYPEDDICT_RE. Duplicated (not imported) so this
# module stays a standalone, dependency-free decision layer during the
# incremental rollout; unified when verify.py's predicates are rewired to
# delegate to classify_file (task γ step-16).
_PROTOCOL_RE = re.compile(r'\bclass\s+\w+\s*\([^)]*\bProtocol\b')
_TYPEDDICT_RE = re.compile(r'\bclass\s+\w+\s*\([^)]*\bTypedDict\b')


def classify_file(path: str, content: str | None) -> FileKind:
    """Classify *path* into exactly one ``FileKind``, given its (optional) *content*.

    Runs the precedence ladder exactly once per file: a non-``.py`` path is
    INERT (cargo/.rs scoping is handled downstream, in
    ``run_scoped_verification``'s execute step, not here); a ``conftest.py``
    basename is CONFTEST at any depth; a ``test_*.py``/``*_test.py`` basename
    is COLLECTABLE_TEST (the files pytest will actually collect); any other
    path under a ``tests/`` directory is TEST_DATA (a test-tree member that
    is not pytest-collectable); a ``.py`` file whose *content* defines a
    ``Protocol``/``TypedDict`` subclass is STRUCTURAL; everything else is
    SOURCE.

    *content* may be ``None`` — e.g. the caller has no worktree to read from,
    or chose not to fetch content for a file where STRUCTURAL detection is
    moot (a CONFTEST/COLLECTABLE_TEST/TEST_DATA classification never consults
    it). STRUCTURAL is then simply never detected; this never raises.
    """
    if not path.endswith('.py'):
        return FileKind.INERT

    name = path.rsplit('/', 1)[-1]

    if name == 'conftest.py':
        return FileKind.CONFTEST

    if name.startswith('test_') or name.endswith('_test.py'):
        return FileKind.COLLECTABLE_TEST

    if '/tests/' in path or path.startswith('tests/'):
        return FileKind.TEST_DATA

    if content is not None and (_PROTOCOL_RE.search(content) or _TYPEDDICT_RE.search(content)):
        return FileKind.STRUCTURAL

    return FileKind.SOURCE
