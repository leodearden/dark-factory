"""Guard test: exactly one ``lock_table.release(...)`` call under
orchestrator/src, and it must live inside ``Scheduler.release``.

``Scheduler.release`` (orchestrator/src/orchestrator/scheduler.py) is the
single writer for freeing a task's module locks — see contract C5 of
plans/scheduler-dispatch-scoring-and-lock-layer-prd.md. It is the ONLY path
that emits ``lock_released``, and it also snapshots the held modules before
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
from pathlib import Path
from typing import TypeGuard

_THIS_FILE = Path(__file__).name

_GUARDED_RECEIVER = 'lock_table'
_GUARDED_METHOD = 'release'
_CHOKEPOINT_CLASS = 'Scheduler'
_CHOKEPOINT_METHOD = 'release'

_SRC_DIR = Path(__file__).parent.parent / 'src'


def _is_lock_table_release(node: ast.AST) -> TypeGuard[ast.Call]:
    """True if *node* is a ``<expr>.lock_table.release(...)`` call.

    Three properties this matcher must have:

    - ``ast.Call``-only, so ``ModuleLockTable.release``'s own DEFINITION in
      scheduler.py (an ``ast.FunctionDef``) is not matched — the invariant
      is about call sites, not about the implementation they call.
    - ``attr == 'release'`` exactly, so ``lock_table.release_subset(...)``
      — which already emits ``lock_released`` with
      ``reason='plan_refinement'`` — is not matched.
    - Receiver-anchored on ``lock_table``, so unrelated ``.release()``
      calls elsewhere in the tree (semaphores, locks, connections) are not
      matched.
    """
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if not isinstance(func, ast.Attribute) or func.attr != _GUARDED_METHOD:
        return False
    receiver = func.value
    return (
        isinstance(receiver, ast.Attribute)
        and receiver.attr == _GUARDED_RECEIVER
    )


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

    results: list[tuple[int, bool]] = []
    for node in ast.walk(tree):
        if _is_lock_table_release(node):
            lineno = node.lineno
            inside = any(start <= lineno <= end for start, end in chokepoint_ranges)
            results.append((lineno, inside))
    return results


def _scan_src() -> tuple[int, list[str], list[str]]:
    """Scan orchestrator/src, returning (total, offenders, outside_chokepoint).

    Entries are ``'<rel_path>:<lineno>: <line_text>'`` strings.
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

    return total, offenders, outside_chokepoint


def test_no_lock_table_release_outside_scheduler_release() -> None:
    """Every ``lock_table.release(...)`` must route through
    ``Scheduler.release`` — the single writer, and the only emitter of
    ``lock_released``.
    """
    total, offenders, outside_chokepoint = _scan_src()

    remediation = (
        'route it through `Scheduler.release(task_id, ...)`, the only site '
        'that emits `lock_released` — see contract C5 of '
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
