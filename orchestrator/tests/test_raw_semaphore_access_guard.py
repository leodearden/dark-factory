"""Guard test: no raw ``_speculation_slot``/``_merge_ahead_cap``
``.acquire()``/``.release()`` calls outside :class:`PermitLedger`
(merge-queue-reliability PRD P-2, scope-3 leaf theta, task 2161).

Both semaphores are single-owned by a
:class:`~orchestrator.merge_speculation_controller.PermitLedger` instance
(``_speculation_ledger`` / ``_merge_ahead_ledger`` on
:class:`~orchestrator.merge_queue.SpeculativeMergeWorker`): every acquire
returns a permit token (``SpecPermit``/``CapPermit``) and every release goes
through ``ledger.release(token)`` -- or, for the shutdown-only over-release,
``ledger.release_for_shutdown(count)``. A raw
``self._speculation_slot.acquire()`` or ``self._merge_ahead_cap.release()``
anywhere else would bypass the ledger's conservation bookkeeping
(``slot_available + len(live) == depth``) silently.

Unlike the reach-back patch guard this idiom mirrors
(``test_merge_queue_reachback_patch_guard.py``), this guard is
allowlist-free: PRD P-2 requires ZERO raw access sites, not a frozen
residual, so any hit at all is a violation.

AST-based (not text/regex) so docstring/comment mentions of these calls
(e.g. this module's own docstring, ``merge_queue.py``'s
``_merge_ahead_ledger`` docstring, ``merge_speculation_controller.py``'s
``PermitLedger`` docstring) and non-call attribute reads (``._value``) are
never mistaken for a real acquire/release call site. The ledger's own
``self._slot.acquire()`` / ``self._slot.release()`` are naturally excluded
too -- ``_slot`` is a different attribute name from the two guarded ones.
"""
from __future__ import annotations

import ast
from pathlib import Path

_SRC_DIR = Path(__file__).parent.parent / 'src' / 'orchestrator'

_GUARDED_ATTRS = {'_speculation_slot', '_merge_ahead_cap'}
_GUARDED_METHODS = {'acquire', 'release'}


def _raw_semaphore_accesses(source: str) -> list[tuple[int, str]]:
    """Return ``(lineno, "<attr>.<method>")`` for each raw
    ``<expr>._speculation_slot.acquire()`` / ``.release()`` (or the
    ``_merge_ahead_cap`` equivalent) call in *source*: a :class:`ast.Call`
    whose ``func`` is an :class:`ast.Attribute` with ``attr`` in
    ``{'acquire', 'release'}``, over a ``value`` that is itself an
    :class:`ast.Attribute` with ``attr`` in ``{'_speculation_slot',
    '_merge_ahead_cap'}``. The receiver of that inner attribute (``self``,
    a worker instance, ...) is irrelevant -- only the two attribute names
    matter.

    AST-based (returns ``[]`` on a ``SyntaxError``) so docstrings, comments,
    and non-call attribute reads (e.g. ``._value``) are never mistaken for a
    real call site.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr in _GUARDED_METHODS):
            continue
        target = func.value
        if isinstance(target, ast.Attribute) and target.attr in _GUARDED_ATTRS:
            hits.append((node.lineno, f'{target.attr}.{func.attr}'))
    return hits


# ---------------------------------------------------------------------------
# _raw_semaphore_accesses(source) -- inline-fixture unit tests.
# ---------------------------------------------------------------------------


def test_raw_semaphore_accesses_flags_cap_acquire() -> None:
    source = (
        "class W:\n"
        "    async def go(self):\n"
        "        await self._merge_ahead_cap.acquire()\n"
    )
    hits = _raw_semaphore_accesses(source)
    assert [attr_method for _lineno, attr_method in hits] == ['_merge_ahead_cap.acquire']


def test_raw_semaphore_accesses_flags_speculation_slot_release_on_other_receiver() -> None:
    """The guarded receiver need not be ``self`` -- e.g. code accessing
    ``worker._speculation_slot.release()`` on a worker instance is flagged
    the same way."""
    source = (
        "def shutdown(worker):\n"
        "    worker._speculation_slot.release()\n"
    )
    hits = _raw_semaphore_accesses(source)
    assert [attr_method for _lineno, attr_method in hits] == ['_speculation_slot.release']


def test_raw_semaphore_accesses_ignores_ledger_own_slot_release() -> None:
    """PermitLedger's own ``self._slot.release()`` -- the wrapped semaphore
    under its ledger-internal name -- is NOT flagged: ``_slot`` is a
    different attribute name from the two guarded ones."""
    source = (
        "class PermitLedger:\n"
        "    def release(self, permit):\n"
        "        self._slot.release()\n"
    )
    assert _raw_semaphore_accesses(source) == []


def test_raw_semaphore_accesses_ignores_docstring_mention() -> None:
    """A comment/docstring merely mentioning the call is NOT flagged --
    proves the detector is AST-based, not text/regex."""
    source = (
        '"""Blocks at ``self._merge_ahead_cap.acquire()`` until drained."""\n'
        "# self._speculation_slot.release() -- example only, not a real call\n"
    )
    assert _raw_semaphore_accesses(source) == []


def test_raw_semaphore_accesses_ignores_value_read() -> None:
    """A ``._value`` read on the semaphore (census/introspection, not a
    mutating acquire/release) is NOT flagged."""
    source = (
        "class W:\n"
        "    def census(self):\n"
        "        return self._merge_ahead_cap._value\n"
    )
    assert _raw_semaphore_accesses(source) == []


# ---------------------------------------------------------------------------
# Required fixture: demonstrate fail-on-new.
# ---------------------------------------------------------------------------


def test_guard_flags_synthetic_new_raw_access() -> None:
    """A brand-new raw ``_merge_ahead_cap``/``_speculation_slot`` acquire or
    release call, anywhere in orchestrator/src, must be reported as a
    violation -- demonstrated here against a synthetic snippet rather than
    waiting for a real regression to prove the detector fires."""
    synthetic_source = (
        "class W:\n"
        "    async def go(self):\n"
        "        await self._merge_ahead_cap.acquire()\n"
    )
    hits = _raw_semaphore_accesses(synthetic_source)
    assert hits == [(3, '_merge_ahead_cap.acquire')], (
        f'expected the synthetic raw acquire() to be flagged, got {hits!r}'
    )


# ---------------------------------------------------------------------------
# Tree-scan: the actual ratchet. Allowlist-free -- PRD P-2 requires ZERO raw
# access sites outside PermitLedger, not a frozen residual.
# ---------------------------------------------------------------------------


def test_no_raw_semaphore_access_outside_permit_ledger() -> None:
    """No file under orchestrator/src/orchestrator may contain a raw
    ``_speculation_slot``/``_merge_ahead_cap`` ``.acquire()``/``.release()``
    call. Every acquire/release must route through the owning
    :class:`PermitLedger` instance (``_speculation_ledger`` /
    ``_merge_ahead_ledger``), which wraps the semaphore under the DIFFERENT
    attribute name ``_slot`` -- naturally excluded from this scan.
    """
    offenders: list[str] = []

    for py_file in sorted(_SRC_DIR.glob('*.py')):
        source = py_file.read_text(encoding='utf-8')
        for lineno, attr_method in _raw_semaphore_accesses(source):
            offenders.append(f'{py_file.name}:{lineno}: {attr_method}()')

    if offenders:
        offender_list = '\n  '.join(offenders)
        raise AssertionError(
            'Raw _speculation_slot/_merge_ahead_cap .acquire()/.release() '
            'call(s) found outside PermitLedger. Route the access through '
            'the owning ledger instead (self._speculation_ledger.acquire() / '
            '.release(permit), or the self._merge_ahead_ledger equivalents; '
            'for a shutdown over-release use '
            'PermitLedger.release_for_shutdown() instead).'
            f'\n\nOffending sites:\n  {offender_list}'
        )
