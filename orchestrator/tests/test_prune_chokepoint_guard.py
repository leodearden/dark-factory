"""Guard test: exactly one raw ``['git', 'worktree', 'prune']`` argv literal
under orchestrator/src, and it must live inside ``_prune_registrations``.

``GitOps._prune_registrations`` (orchestrator/src/orchestrator/git_ops.py) is
the single chokepoint for pruning worktree registrations — see
plans/gitops-chokepoints-prd.md. Every other call site must route through it
(directly or via its thin public delegate ``prune_worktrees``) rather than
shelling out to ``git worktree prune`` itself. This test makes any future
raw prune site turn CI / pre-merge verify red instead of silently
reintroducing the hazard the PRD guards against (pool-storage guard races,
missing escalation debounce, etc.).

This test uses AST parsing (NOT text grep) to scan every *.py under
orchestrator/src (recursively via rglob, excluding this file), and only
flags an actual List or Tuple literal whose elements are exactly the string
Constants ``'git'``, ``'worktree'``, ``'prune'`` in order. Comments,
docstrings, and log-format strings that merely mention the phrase (e.g.
verify.py's DD5 notes, git_ops.py's own docstrings) are never matched,
avoiding false positives. Mirrors the AST-walk / rglob / offender file:line
list / skip-self pattern used by
orchestrator/tests/test_event_loop_antipattern_guard.py.
"""
from __future__ import annotations

import ast
from pathlib import Path

_THIS_FILE = Path(__file__).name

_EXPECTED_ARGV = ('git', 'worktree', 'prune')
_CHOKEPOINT_FUNCTION = '_prune_registrations'


def _is_prune_argv_literal(node: ast.AST) -> bool:
    """True if *node* is a List/Tuple literal of exactly the prune argv."""
    if not isinstance(node, (ast.List, ast.Tuple)):
        return False
    if len(node.elts) != len(_EXPECTED_ARGV):
        return False
    for elt, expected in zip(node.elts, _EXPECTED_ARGV):
        if not isinstance(elt, ast.Constant) or elt.value != expected:
            return False
    return True


def _find_prune_argv_literals(source: str) -> list[tuple[int, bool]]:
    """Return (lineno, inside_chokepoint) for each prune argv literal in *source*.

    ``inside_chokepoint`` is True when the literal's line falls within the
    line range of an ``async def _prune_registrations`` (or plain
    ``def _prune_registrations``) function body. Returns an empty list if
    the file cannot be parsed.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    # Collect line ranges of every function named _prune_registrations,
    # walking the whole tree so nested/class-scoped defs are covered.
    chokepoint_ranges: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == _CHOKEPOINT_FUNCTION
        ):
            end = getattr(node, 'end_lineno', node.lineno)
            chokepoint_ranges.append((node.lineno, end))

    results: list[tuple[int, bool]] = []
    for node in ast.walk(tree):
        if _is_prune_argv_literal(node):
            lineno = node.lineno
            inside = any(start <= lineno <= end for start, end in chokepoint_ranges)
            results.append((lineno, inside))
    return results


def test_exactly_one_prune_argv_literal_inside_chokepoint() -> None:
    """Every raw ``['git', 'worktree', 'prune']`` argv must route through
    ``GitOps._prune_registrations`` — the single chokepoint.
    """
    src_dir = Path(__file__).parent.parent / 'src'
    offenders: list[str] = []
    outside_chokepoint: list[str] = []
    total = 0

    for py_file in sorted(src_dir.rglob('*.py')):
        if py_file.name == _THIS_FILE:
            continue  # skip-self, in case this guard ever lives under src
        source = py_file.read_text(encoding='utf-8')
        source_lines = source.splitlines()
        rel = py_file.relative_to(src_dir)
        for lineno, inside in _find_prune_argv_literals(source):
            total += 1
            line_text = source_lines[lineno - 1].strip() if lineno <= len(source_lines) else ''
            entry = f'{rel}:{lineno}: {line_text}'
            offenders.append(entry)
            if not inside:
                outside_chokepoint.append(entry)

    remediation = (
        "route through GitOps._prune_registrations(context=...) — "
        "see plans/gitops-chokepoints-prd.md"
    )

    if total != 1 or outside_chokepoint:
        offender_list = '\n  '.join(offenders) if offenders else '(none found)'
        raise AssertionError(
            "Expected exactly one ['git', 'worktree', 'prune'] argv literal "
            f"under orchestrator/src, found {total}, all inside "
            f"{_CHOKEPOINT_FUNCTION}.\n"
            f'{remediation}\n'
            f'\nOffending sites:\n  {offender_list}'
        )
