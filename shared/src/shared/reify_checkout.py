"""Layout-independent discovery of a sibling ``reify`` checkout.

The single source of reify-checkout resolution for every reify-dependent test in
dark-factory.  Promoted here (task 3978) from
``fused-memory/tests/test_lock_charter_guard.py``, where task 3843 first
established the semantics, so the two cross-repo call sites — orchestrator's
``scripts/verify.sh`` gate and fused-memory's ``scripts/lock-charter-guard.sh``
drift guard — share one implementation instead of drifting apart.  Both the
resolution AND the skip wording live here: a call site that hand-rolled its own
"reify checkout not discoverable" string would re-open exactly the conflation
`reify_skip_reason` exists to prevent.

Consumers import from this module directly::

    from shared.reify_checkout import (
        REIFY_ROOT_ENV,
        checkout_skip_reason,
        reify_skip_reason,
        resolve_reify_checkout,
    )

It is deliberately NOT re-exported from ``shared/__init__.py`` — see
``shared/tests/test_public_api.py``, which pins ``shared.__all__`` to the union
of a hardcoded submodule list.  Direct submodule import is the established
convention for this kind of module here (``shared.testing``,
``shared.task_statuses``, ``shared.toolcall_markup``).

This module is pure stdlib on purpose: it must NOT import pytest, which is in
shared's ``[dependency-groups] dev`` and not its ``[project]`` dependencies.
Test-support callers get a skip *reason* string from `reify_skip_reason` or
`checkout_skip_reason` and turn it into a ``pytest.skip`` /
``pytest.mark.skipif`` themselves — which is
also the shape the two consumers need anyway, since they need DIFFERENT ones
(a runtime skip inside a test body vs a module-level skipif marker).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple

__all__ = [
    'REIFY_ROOT_ENV',
    'ReifyCheckout',
    'checkout_skip_reason',
    'reify_skip_reason',
    'resolve_reify_checkout',
]

#: The one env var steering every reify-dependent test in dark-factory.  Named
#: here rather than respelled at each call site, so one `export REIFY_ROOT=...`
#: cannot be half-honored by a consumer that typo'd the name.
REIFY_ROOT_ENV = 'REIFY_ROOT'


class ReifyCheckout(NamedTuple):
    """A resolved reify checkout, together with WHERE that answer came from.

    Provenance travels WITH the root on purpose.  `reify_skip_reason` needs it
    to say whether a missing marker is the operator's own REIFY_ROOT typo, and
    re-deriving it from ``os.environ`` at formatting time would let the two
    disagree: a caller that resolves once and formats a reason later — or a test
    that changes REIFY_ROOT in between — would get "(named by REIFY_ROOT)"
    attached to a root that was actually DISCOVERED, blaming the operator for a
    path they never named.  That is the same misattribution class this module
    exists to remove, so provenance is an explicit value, never an ambient
    re-read.
    """

    #: The checkout root, or None on a discovery miss.
    root: Path | None
    #: True iff *root* came from the REIFY_ROOT override rather than discovery.
    named_by_env: bool


def resolve_reify_checkout(marker: str | Path, *, start: Path) -> ReifyCheckout:
    """Locate the reify checkout carrying *marker*, independent of checkout layout.

    *marker* is a relpath INSIDE the checkout that identifies a usable one for
    the caller (e.g. ``scripts/verify.sh`` or ``scripts/lock-charter-guard.sh``).
    *start* is the calling module's own file — pass ``Path(__file__)`` from the
    CALL SITE.  It is keyword-only and required by design: defaulting it to this
    module's ``__file__`` would start every walk in ``shared/src/shared/``
    instead of at the caller, and both parameters being path-like makes a
    silently-swapped positional pair the obvious way to get a wrong answer.

    Returns a `ReifyCheckout` rather than a bare path so the ANSWER and its
    PROVENANCE are produced together and cannot disagree downstream — feed both
    to `reify_skip_reason`.

    Measured first-hand (task 3843), from a test file in ``<pkg>/tests/``:

        worktree:       parents[3]/reify -> /home/leo/src/dark-factory/.worktrees/reify  (absent)
                        parents[5]/reify -> /home/leo/src/reify                          (present)
        bare checkout:  parents[3]/reify -> /home/leo/src/reify                          (present)
                        parents[5]/reify -> /home/reify                                  (absent)

    ``.worktrees/<id>`` contributes exactly two path segments, so parents[5] is
    correct in a worktree and parents[3] is correct in a bare checkout — no
    single fixed index is correct in both layouts.  Task 3843's title-level
    prescription (change parents[5] to parents[3]) is therefore a REGRESSION,
    not a fix: applied literally, it would silently skip every reify-dependent
    guard in every worktree — which is where orchestrator verify runs.  Do not
    "fix" this back to a fixed parents[N] index.

    Resolution order:
      1. The `REIFY_ROOT` env var, if set (empty/whitespace-only counts as
         unset — the exported-but-empty shell accident, ``export REIFY_ROOT=``).
         Honored VERBATIM, even when the path does not exist on disk: a silent
         fallthrough to a discovered checkout would make a typo'd REIFY_ROOT
         look like it worked while actually answering for a DIFFERENT repo than
         the operator named — the same silently-wrong-answer class this module
         exists to remove.  A bad override must surface downstream as a skip
         naming the bad path (see `reify_skip_reason`), not a lucky-but-wrong
         discovery.  This arm is deliberately marker-INDEPENDENT — it returns
         before the walk — which is what makes one ``export REIFY_ROOT=...``
         steer every reify-dependent consumer to the same root regardless of
         which script each one gates on.  Only this arm sets ``named_by_env``.
      2. Otherwise, the nearest ancestor ``a`` of *start*, nearest first, for
         which ``a / 'reify' / marker`` is a file.  Nearest-first is
         load-bearing: it reaches /home/leo/src before /home/leo and /, so a
         stray higher-up ``reify`` directory (e.g. a bare one with no scripts/
         inside) cannot shadow the real sibling checkout.  Unlike arm 1 this is
         marker-specific: a checkout that lacks the caller's own script is not
         a usable checkout for that caller.
      3. Otherwise ``ReifyCheckout(None, False)`` — the legitimate
         standalone-checkout discovery MISS.

    Call sites typically resolve this into module-level constants, which are
    evaluated at IMPORT time.  REIFY_ROOT must therefore be exported BEFORE
    pytest starts to steer them; a mid-session ``monkeypatch.setenv`` only
    affects direct `resolve_reify_checkout` calls.
    """
    override = os.environ.get(REIFY_ROOT_ENV, '').strip()
    if override:
        return ReifyCheckout(Path(override).resolve(), True)

    for ancestor in Path(start).resolve().parents:
        if (ancestor / 'reify' / marker).is_file():
            return ReifyCheckout(ancestor / 'reify', False)
    return ReifyCheckout(None, False)


def reify_skip_reason(
    marker: str | Path, root: Path | None, *, named_by_env: bool
) -> str | None:
    """Why a reify-dependent test cannot run against *root*, or None if it can.

    Returns ``None`` when ``root / marker`` is a real file — the gate must RUN.
    Otherwise returns a non-empty reason string for the caller to feed to
    ``pytest.skip`` / ``pytest.mark.skipif``.  It never returns ``''``: a falsy
    reason would silently DISABLE a call site that gates on truthiness, turning
    a skip into a phantom pass.

    *named_by_env* is the provenance of *root* — pass
    ``ReifyCheckout.named_by_env`` from the very resolution that produced it.
    It is a required keyword rather than an ``os.environ`` re-read so the
    message can never disagree with the resolution: an ambient re-read would
    blame REIFY_ROOT for a DISCOVERED root whenever the env var happened to be
    set between the two calls.

    The two non-None arms are deliberately distinct (carried over from task
    3843's ``_skip_unless_checkout``), and conflating them is the failure this
    exists to prevent:

      * ``root is None`` is the legitimate standalone-checkout discovery MISS —
        nobody has a reify sibling checked out here.  That is expected and
        benign, so the reason says what was searched for and how to override it,
        and deliberately does NOT name a path: nobody named one.
      * a non-None *root* whose *marker* is missing is a checkout that cannot
        serve this caller.  `resolve_reify_checkout` only ever DISCOVERS a root
        whose marker IS a file, so in practice this is an operator's REIFY_ROOT
        naming a path that is not there — which is why the message names the
        path, and, when *named_by_env*, says so: it is the path *you* named.
        The override is honored verbatim rather than silently falling back to
        discovery, so a typo is self-evident in ``pytest -rs`` output instead of
        quietly answering for a different repo than the operator asked for.
    """
    if root is None:
        return (
            f'reify checkout not discoverable (no ancestor carries reify/{marker}); '
            f'set {REIFY_ROOT_ENV} to override'
        )
    if not (root / marker).is_file():
        named_by = f' (named by {REIFY_ROOT_ENV})' if named_by_env else ''
        return f'reify checkout at {root}{named_by} has no {marker}'
    return None


def checkout_skip_reason(repo: str, root: Path | None, *, marker: str | Path) -> str | None:
    """Why a checkout-dependent sweep cannot run against *root*, or None if it can.

    The CHECKOUT-reachability gate, one rung weaker than `reify_skip_reason`:
    the corpus sweeps that use it need only a git checkout to be on disk, not
    the guard script the Tier-2 gates need.  Returns ``None`` when *root* is a
    real directory — the sweep must RUN — and otherwise a non-empty reason for
    the caller to feed to ``pytest.skip``.  Same contract as its sibling: never
    ``''``, because a falsy reason silently disables a call site that gates on
    truthiness, turning a skip into a phantom pass.

    Consolidated here by task 4259 from two identical hand-rolled copies (in
    shared/tests/test_locking.py and fused-memory/tests/test_lock_charter_guard.py,
    both descended from task 3843's ``_skip_unless_checkout``).  It lives beside
    `reify_skip_reason` because this module already owns reify skip WORDING —
    see the module docstring — and a call site that hand-rolls its own string
    re-opens exactly the conflation that rule exists to prevent.

    The caller turns the reason into a ``pytest.skip`` itself; this module is
    pytest-free (module docstring, and mechanically ``PURE_STDLIB_LEAVES`` in
    shared/tests/test_pure_stdlib_leaves.py).

    The two non-None arms are deliberately distinct, and conflating them is the
    failure this exists to prevent:

      * ``root is None`` is the legitimate standalone-checkout discovery MISS —
        `resolve_reify_checkout` walked every ancestor and none carried
        ``reify/<marker>``, with REIFY_ROOT unset.  It is the SAME condition
        `reify_skip_reason` describes, so it is delegated there rather than
        restated: one wording for one condition, whatever gate is asking.
      * a *root* that is not a directory is an operator's REIFY_ROOT naming a
        path that is not there.  The override is honored verbatim rather than
        silently falling back to discovery, so the reason NAMES the path and a
        typo is self-evident in ``pytest -rs`` output instead of quietly
        answering for a different repo than the operator asked for.  This arm
        does NOT borrow the marker-based wording: these sweeps need only a
        checkout, so a reason built around the stronger marker would overstate
        what they actually require.

    The test is ``is_dir()``, not ``exists()``: a path that exists but is a
    regular file is not a checkout, and admitting it would fail deep inside git
    rather than skip with a message naming the bad path.

    *named_by_env* is hardcoded ``False`` on the delegated arm rather than being
    a parameter, and that is a fact about the resolver, not a simplification:
    `resolve_reify_checkout` returns a non-None ``Path(override)`` for the
    REIFY_ROOT arm — honored verbatim, even when absent on disk — so
    ``root is None`` implies the discovery-MISS arm BY CONSTRUCTION.  There is
    no reachable ``named_by_env=True`` None root to mis-attribute, and offering
    the parameter would invite a caller to claim one.
    """
    if root is None:
        reason = reify_skip_reason(marker, None, named_by_env=False)
        if not reason:
            raise RuntimeError(
                f'a None {repo} root is the discovery-miss arm, which always '
                f'yields a reason — a falsy one here would turn this skip into '
                f'a phantom pass (got {reason!r})'
            )
        return reason
    if not Path(root).is_dir():
        return f'{repo} checkout not present at {root}'
    return None
