"""Guard test: exactly one ``lock_table.release(...)`` call under
orchestrator/src, and it must live inside ``Scheduler.release``.

This module docstring is the single canonical write-up of that rationale;
the scheduler.py comment and the behavioural tests in test_scheduler.py
point here rather than restating it.

``Scheduler.release`` (orchestrator/src/orchestrator/scheduler.py) is the
single writer for freeing a task's module locks — see contract C5 of
plans/scheduler-dispatch-scoring-and-lock-layer-prd.md. It is the only path
that emits a *full-release* ``lock_released`` (``lock_table.release_subset``
emits the narrower ``reason='plan_refinement'`` partial, which closes only
the modules it narrows away), and it also snapshots the held modules before
releasing, clears ``_dispatched`` / ``_dispatched_priority``, arms the
requeue cooldown, and runs the defensive ``clear_parks_for``.

A call that reaches ``lock_table.release`` directly is worse than a silent
release: it empties ``_held[task_id]`` first, so the workflow's later
teardown ``Scheduler.release`` snapshots ``modules == []`` and skips its
emit too. The stream is then left with a ``lock_acquired`` that has no
matching ``lock_released`` anywhere — a stuck-lock artifact that reads as an
infinite hold to any consumer measuring hold durations from acquire/release
pairs. Task 3818 removed the one such bypass (the blast-radius requeue);
this test makes any future one turn CI / pre-merge verify red instead of
silently reintroducing the hazard.

AST parsing (NOT text grep) because a call can span lines, and because the
enclosing scope of each match is what the invariant is actually about.

The scan is scoped to ``orchestrator/src`` deliberately, not incidentally:
``lock_table.release(`` also appears several times in
orchestrator/tests/test_scheduler.py as direct ``ModuleLockTable`` unit
tests, which are legitimate — scanning the tests tree would make the
assertion fail against itself.

Mirrors the AST-walk / rglob / offender file:line list / skip-self pattern
of orchestrator/tests/test_prune_chokepoint_guard.py, whose
``_chokepoint_ranges`` class-qualification recursion is reused here.
"""
from __future__ import annotations

import ast
import functools
from pathlib import Path
from typing import TypeGuard

_THIS_FILE = Path(__file__).name

_GUARDED_RECEIVER = 'lock_table'
_GUARDED_METHOD = 'release'
_CHOKEPOINT_CLASS = 'Scheduler'
_CHOKEPOINT_METHOD = 'release'

_SRC_DIR = Path(__file__).parent.parent / 'src'


def _lock_table_aliases(tree: ast.Module) -> frozenset[str]:
    """Local names bound to the lock table by an alias assignment.

    A bypass need not be spelled ``self.lock_table.release(...)``:
    ``lt = self.lock_table`` followed by ``lt.release(task_id)`` frees the
    same locks through an ``ast.Name`` receiver, which a receiver-anchored
    Attribute matcher alone would silently miss while the total stayed at 1
    and the guard stayed green. That alias idiom is live in this codebase
    (``lt = scheduler.lock_table`` in test_scheduler.py), so the scanner
    resolves it instead of trusting the spelling.

    Collected module-wide rather than per-function: strictly broader, and a
    false positive here fails the guard loudly (the safe direction) rather
    than leaving a real bypass unseen.
    """
    aliases: set[str] = set()

    def _record(target: ast.AST, value: ast.AST | None) -> None:
        # `<name> = <expr>.lock_table` — the alias shape. Tuple/starred
        # targets are deliberately not unpacked: `.lock_table` is not
        # iterable, so no real alias can hide there.
        if (
            isinstance(target, ast.Name)
            and isinstance(value, ast.Attribute)
            and value.attr == _GUARDED_RECEIVER
        ):
            aliases.add(target.id)

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                _record(target, node.value)
        elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)):
            # AnnAssign.value is Optional; _record's isinstance check drops it.
            _record(node.target, node.value)

    return frozenset(aliases)


def _is_lock_table_release(
    node: ast.AST, aliases: frozenset[str] = frozenset()
) -> TypeGuard[ast.Call]:
    """True if *node* is a ``release(...)`` call on the lock table.

    Matches both the direct ``<expr>.lock_table.release(...)`` spelling and a
    ``<name>.release(...)`` call whose receiver resolves to the lock table —
    either the bare name ``lock_table`` (a parameter or module-level binding)
    or one of *aliases*, the names :func:`_lock_table_aliases` found bound to
    ``<expr>.lock_table` in the same module.

    Four properties this matcher must have:

    - ``ast.Call``-only, so ``ModuleLockTable.release``'s own DEFINITION in
      scheduler.py (an ``ast.FunctionDef``) is not matched — the invariant
      is about call sites, not about the implementation they call.
    - ``attr == 'release'`` exactly, so ``lock_table.release_subset(...)``
      — which emits its own ``reason='plan_refinement'`` partial — is not
      matched.
    - Anchored on the lock table, so unrelated ``.release()`` calls
      elsewhere in the tree (semaphores, permit ledgers, lane pools) are not
      matched.
    - Alias-aware, so the anchor cannot be sidestepped by binding the table
      to a local first.
    """
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if not isinstance(func, ast.Attribute) or func.attr != _GUARDED_METHOD:
        return False
    receiver = func.value
    if isinstance(receiver, ast.Attribute):
        return receiver.attr == _GUARDED_RECEIVER
    if isinstance(receiver, ast.Name):
        return receiver.id == _GUARDED_RECEIVER or receiver.id in aliases
    return False


def _chokepoint_ranges(tree: ast.Module) -> list[tuple[int, int]]:
    """Line ranges of every ``release`` method defined directly on a class
    named ``Scheduler``.

    Class-qualification matters here: ``release`` is a common method name,
    so a plain name-only scan would also sanction a differently-classed or
    module-level function of the same name. This walks the tree tracking the
    nearest *directly enclosing* class, so only a method defined straight in
    ``class Scheduler:`` qualifies.
    """
    ranges: list[tuple[int, int]] = []

    def visit(node: ast.AST, enclosing_class: str | None) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                visit(child, child.name)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if (
                    child.name == _CHOKEPOINT_METHOD
                    and enclosing_class == _CHOKEPOINT_CLASS
                ):
                    end = getattr(child, 'end_lineno', child.lineno)
                    ranges.append((child.lineno, end))
                # A function body is no longer "directly inside" the
                # enclosing class, so a closure nested within the method
                # doesn't inherit the class qualification.
                visit(child, None)
            else:
                visit(child, enclosing_class)

    visit(tree, None)
    return ranges


def _find_lock_table_releases(source: str) -> list[tuple[int, bool]]:
    """Return (lineno, inside_chokepoint) for each ``lock_table.release(...)``
    call in *source*.

    ``inside_chokepoint`` is True when the call's line falls within the line
    range of a ``release`` method defined directly on the ``Scheduler`` class
    (see :func:`_chokepoint_ranges`). Returns an empty list if the file
    cannot be parsed.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    chokepoint_ranges = _chokepoint_ranges(tree)
    aliases = _lock_table_aliases(tree)

    results: list[tuple[int, bool]] = []
    for node in ast.walk(tree):
        if _is_lock_table_release(node, aliases):
            lineno = node.lineno
            inside = any(start <= lineno <= end for start, end in chokepoint_ranges)
            results.append((lineno, inside))
    # ast.walk is breadth-first, so hits come out in tree-depth order, not
    # source order. Sort so offender lists and the synthetic-source
    # assertions below read top-to-bottom.
    results.sort(key=lambda hit: hit[0])
    return results


@functools.lru_cache(maxsize=1)
def _scan_src() -> tuple[int, tuple[str, ...], tuple[str, ...]]:
    """Scan orchestrator/src, returning (total, offenders, outside_chokepoint).

    Entries are ``'<rel_path>:<lineno>: <line_text>'`` strings.

    Cached: parsing the whole src tree (scheduler.py alone is ~7k lines) is
    the module's dominant cost and every test here wants the same answer —
    nothing that could change it moves within a session. Results are tuples
    so a caller cannot mutate the shared cached value.
    """
    offenders: list[str] = []
    outside_chokepoint: list[str] = []
    total = 0

    for py_file in sorted(_SRC_DIR.rglob('*.py')):
        if py_file.name == _THIS_FILE:
            continue  # skip-self, in case this guard ever lives under src
        source = py_file.read_text(encoding='utf-8')
        source_lines = source.splitlines()
        rel = py_file.relative_to(_SRC_DIR)
        for lineno, inside in _find_lock_table_releases(source):
            total += 1
            line_text = (
                source_lines[lineno - 1].strip()
                if lineno <= len(source_lines)
                else ''
            )
            entry = f'{rel}:{lineno}: {line_text}'
            offenders.append(entry)
            if not inside:
                outside_chokepoint.append(entry)

    return total, tuple(offenders), tuple(outside_chokepoint)


def test_no_lock_table_release_outside_scheduler_release() -> None:
    """Every ``lock_table.release(...)`` must route through
    ``Scheduler.release`` — the single writer, and the only emitter of a
    full-release ``lock_released``.
    """
    total, offenders, outside_chokepoint = _scan_src()

    remediation = (
        'route it through `Scheduler.release(task_id, ...)`, the only site '
        'that emits a full-release `lock_released` (`release_subset` emits '
        'only the narrower `plan_refinement` partial) — see contract C5 of '
        'plans/scheduler-dispatch-scoring-and-lock-layer-prd.md. A direct '
        '`lock_table.release` empties _held[task_id] first, so the later '
        'teardown release() snapshots modules == [] and skips its emit too, '
        'leaving a lock_acquired with no matching lock_released.'
    )

    # Assert the TOTAL as well as the outside-list (belt and braces): a
    # future SECOND call added *inside* Scheduler.release would keep the
    # outside-list empty while still splitting the single-writer seam.
    if total != 1 or outside_chokepoint:
        offender_list = '\n  '.join(offenders) if offenders else '(none found)'
        message_lines = [
            'Expected exactly one `lock_table.release(...)` call under '
            f'orchestrator/src, found {total}.'
        ]
        if outside_chokepoint:
            outside_list = '\n  '.join(outside_chokepoint)
            message_lines.append(
                f'{len(outside_chokepoint)} site(s) are OUTSIDE '
                f'{_CHOKEPOINT_CLASS}.{_CHOKEPOINT_METHOD}:\n  {outside_list}'
            )
        message_lines.append(remediation)
        message_lines.append(f'\nAll sites found:\n  {offender_list}')
        raise AssertionError('\n'.join(message_lines))


def test_scanner_finds_the_sanctioned_site() -> None:
    """Anti-vacuity: the scanner must actually find the one real site.

    Without this, a matcher that silently broke (a renamed receiver, an AST
    shape change) would report zero hits and turn the guard above
    permanently, silently green.
    """
    total, offenders, outside_chokepoint = _scan_src()

    assert total == 1, (
        'the scanner must find the one sanctioned lock_table.release site '
        f'inside Scheduler.release; found {total}: {offenders!r}'
    )
    assert not outside_chokepoint, (
        f'the sanctioned site must be INSIDE the chokepoint; got '
        f'{outside_chokepoint!r}'
    )


def test_guard_flags_a_synthetic_bypass() -> None:
    """A brand-new ``lock_table.release`` call outside ``Scheduler.release``
    must be reported as outside-chokepoint.

    Demonstrated against a synthetic snippet rather than by reverting the
    real fix: the production fix is a single line, so this guard is
    necessarily authored after the bypass is already gone and would be born
    green against the real tree. This pins the detector's discriminating
    power independently of the tree's current state.
    """
    synthetic_source = (
        'class Scheduler:\n'
        '    def release(self, task_id):\n'
        '        self.lock_table.release(task_id)\n'   # line 3: sanctioned
        '\n'
        '    def handle_blast_radius_expansion(self, task_id):\n'
        '        self.lock_table.release(task_id)\n'   # line 6: the bypass
    )

    hits = _find_lock_table_releases(synthetic_source)

    assert hits == [(3, True), (6, False)], (
        'the detector must sanction the in-chokepoint call at line 3 and '
        f'flag the bypass at line 6; got {hits!r}'
    )
    outside = [lineno for lineno, inside in hits if not inside]
    assert outside == [6], (
        f'expected exactly one outside-chokepoint hit at line 6; got {outside!r}'
    )


def test_guard_flags_a_bypass_laundered_through_an_alias() -> None:
    """A bypass that binds the table to a local first must still be flagged.

    ``lt = self.lock_table; lt.release(task_id)`` frees exactly the same
    locks as the direct spelling, but presents an ``ast.Name`` receiver.
    Without alias resolution the scanner would miss it while the total stayed
    at 1 and the guard stayed silently green — and the alias idiom is live in
    this codebase, so this is not a hypothetical shape.
    """
    synthetic_source = (
        'class Scheduler:\n'
        '    def release(self, task_id):\n'
        '        self.lock_table.release(task_id)\n'   # line 3: sanctioned
        '\n'
        '    def sneaky(self, task_id):\n'
        '        lt = self.lock_table\n'
        '        lt.release(task_id)\n'                # line 7: aliased bypass
        '\n'
        '\n'
        'def helper(lock_table, task_id):\n'
        '    lock_table.release(task_id)\n'            # line 11: bare-name bypass
    )

    hits = _find_lock_table_releases(synthetic_source)

    assert hits == [(3, True), (7, False), (11, False)], (
        'the detector must resolve `lt = self.lock_table` and the bare '
        f'`lock_table` parameter back to the lock table; got {hits!r}'
    )


def test_guard_ignores_unrelated_names_that_merely_have_release() -> None:
    """Alias resolution must not turn every ``<name>.release()`` into a hit.

    Only names actually bound to ``<expr>.lock_table`` (or literally named
    ``lock_table``) count — otherwise the guard would fire on every semaphore
    and permit ledger in the tree and stop meaning anything.
    """
    synthetic_source = (
        'def unrelated(permits, task_id):\n'
        '    sem = permits.semaphore\n'
        '    sem.release()\n'
        '    ledger = self._speculation_ledger\n'
        '    ledger.release(task_id)\n'
    )

    assert _find_lock_table_releases(synthetic_source) == [], (
        'a `.release()` on a name not bound to the lock table must not match'
    )


def test_guard_ignores_release_subset_and_the_definition() -> None:
    """The matcher must not fire on ``release_subset`` (which already emits)
    nor on ``ModuleLockTable.release``'s own definition."""
    synthetic_source = (
        'class ModuleLockTable:\n'
        '    def release(self, task_id):\n'          # a def, not a Call
        '        self._held.pop(task_id, None)\n'
        '\n'
        'class Other:\n'
        '    def narrow(self, task_id, stale):\n'
        '        self.lock_table.release_subset(task_id, stale)\n'
        '        self.semaphore.release()\n'          # unrelated receiver
    )

    assert _find_lock_table_releases(synthetic_source) == [], (
        'release_subset, an unrelated .release(), and the method definition '
        'itself must not be matched'
    )
