"""Planted-synthetic-layout harness for reify-dependent test suites.

The single home of the "load a COPY of a test module out of a SYNTHETIC
checkout tree" technique that every reify-dependent suite needs, promoted here
(task 4259) from the two copies that had grown in
``orchestrator/tests/test_verify_role_integration.py`` (task 3978) and
``shared/tests/test_locking.py`` (task 4080), which in turn reused the
technique first established in
``fused-memory/tests/test_lock_charter_guard.py`` (task 3843).  It is the layer
above `shared.reify_checkout`: that module owns HOW a reify checkout is
resolved, this one owns how a test proves the resolution answers for the tree
under test rather than for the developer's own machine.

Consumers import from this module directly::

    import shared.testing_reify_layout as reify_layout

and keep a THIN marker-bound adapter (``_load_module_copy``) that binds their
own ``__file__`` and marker — the same shape `shared.reify_checkout`'s
consumers already use for ``_resolve_reify_checkout``.  Call the shared
function through the MODULE attribute rather than a ``from ... import``
binding, so each suite's delegation pin can monkeypatch it.

Like `shared.reify_checkout` it is deliberately NOT re-exported from
``shared/__init__.py`` — see ``shared/tests/test_public_api.py``, which pins
``shared.__all__`` to the union of a hardcoded submodule list.  Direct
submodule import is the established convention for this family
(``shared.testing``, ``shared.testing_stdin``, ``shared.reify_checkout``).

WHY A REAL CHECKOUT CANNOT EXPRESS THE DEFECT.  In the ``.worktrees/<id>``
layout these suites normally run in, a fixed ``parents[5]`` index already
resolves correctly, so every BLACK-BOX assertion against THIS machine's real
checkout stays green both before and after the resolver is broken — or after a
faithful private copy of the walk quietly reappears.  Only a synthetic
ancestry — a module COPY imported from a planted tree — can express what a bare
(non-worktree) checkout sees.

WHY THE RED IS HOST-INDEPENDENT IN BOTH DIRECTIONS.  A fixed ``parents[N]``
answer can never equal a ``tmp_path``, and a ``tmp_path`` with no reify in its
ancestry must resolve to None even on a machine where a real reify checkout
exists.  Neither direction depends on what this developer happens to have
checked out.

WHY THE COPY BEHAVES LIKE THE ORIGINAL.  The copied module's own imports
(``shared.locking``, ``orchestrator.verify``, ``shared.reify_checkout``, ...)
resolve from the PARENT process's ``sys.path``, while its ``__file__`` is the
PLANTED path — so its import-time constant resolution walks the SYNTHETIC
ancestry, which is the whole point.  Import time is the moment that matters:
it is when a suite's module-level constants are computed and when a
``skipif`` decorator's argument is evaluated, i.e. exactly when a broken
resolver silently flips a cross-repo guard off.

This module is pure stdlib on purpose: it must NOT import pytest, which is in
shared's ``[dependency-groups] dev`` and not its ``[project]`` dependencies, so
a module-level pytest import here would break any consumer installing the
package without dev deps.  That constraint is enforced mechanically, not merely
asserted: this module is registered in ``PURE_STDLIB_LEAVES`` in
``shared/tests/test_pure_stdlib_leaves.py``, which imports it in a fresh
interpreter and flags any third-party top-level module that appears in
``sys.modules``.  It is why the contamination guard below raises a dedicated
exception rather than calling ``pytest.skip`` or ``pytest.fail``.

Semantics are owned by ``shared/tests/test_testing_reify_layout.py``; the two
consumer suites pin only their own wiring.
"""

from __future__ import annotations

import importlib.util
import re
import shutil
import sys
from pathlib import Path
from types import ModuleType

__all__ = [
    'AmbientReifyCheckoutError',
    'ambient_reify_roots',
    'bare_layout',
    'plant_reify_layout',
    'planted_reify_root',
    'worktree_layout',
]

#: Name every planted copy is written under, inside the planted tests dir.
_COPY_FILENAME = 'test_copy_probe.py'

#: Prefix for the derived, temporary ``sys.modules`` key of a planted copy.
_MODULE_NAME_PREFIX = '_reify_layout_probe_'

#: Content written for a planted marker.  Irrelevant to every consumer — the
#: resolver's predicate is ``is_file()``, never anything about the contents.
_MARKER_STUB = '#!/bin/sh\necho stub\n'


class AmbientReifyCheckoutError(RuntimeError):
    """A real reify checkout sits in the planted tree's ancestry.

    Raised when `plant_reify_layout` is asked for a discovery-MISS tree
    (``plant_marker=False``) but the environment cannot express one, because a
    genuine ``reify/<marker>`` exists above the planted tree — an unusual
    ``--basetemp`` (e.g. one placed inside the repo) or an ambient ``/tmp/reify``
    checkout.

    Loud on purpose.  Left unchecked, the caller's behavioural assertion fails
    with a message that blames the resolver for answering with a path, when the
    real cause is the environment.  An ``assert`` would be stripped under
    ``-O``, and a silent skip would retire the case without anyone noticing; a
    dedicated exception names the contaminating ancestors instead, and keeps
    this module pytest-free.
    """


def planted_reify_root(tmp_path: Path) -> Path:
    """The reify checkout `plant_reify_layout` plants under *tmp_path*.

    The ONE place that knows the synthetic layout, so call sites assert against
    this instead of respelling ``tmp_path / 'src' / 'reify'``.  It is the
    sibling of the planted ``src`` tree, mirroring how a real ``reify`` checkout
    sits beside ``dark-factory`` rather than inside it.
    """
    return tmp_path / 'src' / 'reify'


def bare_layout(package: str) -> str:
    """Tests-dir relpath for *package* in a plain (non-worktree) checkout."""
    return f'dark-factory/{package}/tests'


def worktree_layout(package: str, task_id: str) -> str:
    """Tests-dir relpath for *package* inside the ``.worktrees/<id>`` layout.

    The load-bearing fact, stated ONCE here rather than re-derived at each call
    site: ``.worktrees/<id>`` contributes exactly TWO path segments relative to
    `bare_layout`, so no fixed ``parents[N]`` index can be correct in both
    layouts.  That is why `shared.reify_checkout.resolve_reify_checkout` walks
    ancestors nearest-first instead of indexing, and why "change parents[5] to
    parents[3]" is a regression rather than a fix.
    """
    return f'dark-factory/.worktrees/{task_id}/{package}/tests'


def ambient_reify_roots(start: str | Path, marker: str | Path) -> list[Path]:
    """Ancestors of *start* that carry a real ``reify/<marker>``, nearest first.

    Uses the identical predicate `resolve_reify_checkout` walks, so a caller
    checking for contamination measures exactly what the resolver would see —
    a re-spelled approximation could pass while the resolver still found
    something.  Returns ``[]`` when the ancestry is clean.

    *start* is a path INSIDE the tree being checked (typically the planted
    copy's own file); only strict ancestors are considered, so a ``reify``
    checkout beside or below *start* is correctly not reported — it steers
    nothing.
    """
    return [
        ancestor
        for ancestor in Path(start).resolve().parents
        if (ancestor / 'reify' / marker).is_file()
    ]


def plant_reify_layout(
    source: str | Path,
    tmp_path: Path,
    tests_relpath: str,
    *,
    marker: str | Path,
    plant_marker: bool = True,
) -> ModuleType:
    """Import a COPY of *source* from a synthetic checkout layout under *tmp_path*.

    Plants ``planted_reify_root(tmp_path) / marker`` as a real file (content
    irrelevant) when *plant_marker*, creates
    ``tmp_path / 'src' / tests_relpath``, copies *source* in, and loads it via
    ``spec_from_file_location``.  Returns the loaded module.

    *source* is the CALLING module's own file — pass ``__file__`` from the call
    site.  It is required and first, for the same reason
    `resolve_reify_checkout`'s *start* is required and keyword-only: this
    module's own ``__file__`` is never the right answer, and a harness that
    defaulted to it would copy ITSELF into the planted tree and measure
    nothing.

    *plant_marker=False* is the discovery-MISS arm.  Only that arm is guarded
    against an ambient real checkout in the ancestry (see
    `AmbientReifyCheckoutError`): with a marker planted, the planted checkout is
    the NEAREST ancestor hit, so nearest-first resolution already shadows
    anything above it and contamination is harmless.

    The temporary ``sys.modules`` entry is popped in a ``finally`` so no copy
    outlives the call — including when the copy raises during exec, which is
    precisely the case a consumer suite is investigating when it reaches for
    this harness.  The exception propagates unchanged.

    The entry's name is DERIVED from *source*'s stem and *tests_relpath* rather
    than supplied by each caller.  That is what keeps two suites' copies of the
    same layout from clobbering each other in ``sys.modules`` without either
    suite having to remember to pick a distinct prefix — uniqueness is a
    property of this harness, not of every caller.
    """
    src = tmp_path / 'src'
    if plant_marker:
        planted = planted_reify_root(tmp_path) / marker
        planted.parent.mkdir(parents=True, exist_ok=True)
        planted.write_text(_MARKER_STUB)

    tests_dir = src / tests_relpath
    tests_dir.mkdir(parents=True, exist_ok=True)
    copied = tests_dir / _COPY_FILENAME
    shutil.copy2(source, copied)

    if not plant_marker:
        contaminated = ambient_reify_roots(copied, marker)
        if contaminated:
            raise AmbientReifyCheckoutError(
                f'a real reify checkout exists in the ancestry of {copied} '
                f'{contaminated!r} — likely an unusual --basetemp or an ambient '
                f'checkout carrying reify/{marker} — so this environment cannot '
                f'express a genuine discovery MISS here; this is an environment '
                f'problem, not a resolver regression'
            )

    module_name = _MODULE_NAME_PREFIX + re.sub(
        r'\W+', '_', f'{Path(source).stem}_{tests_relpath}'
    )
    spec = importlib.util.spec_from_file_location(module_name, copied)
    assert spec is not None and spec.loader is not None, f'could not load {copied}'
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(module_name, None)
    return mod
