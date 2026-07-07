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
orchestrator/src (recursively via rglob, excluding this file), and flags
any List or Tuple literal whose first element is the string Constant
``'git'`` and whose remaining elements contain ``'worktree'`` followed
later by ``'prune'`` as an in-order subsequence — catching not just the
exact 3-element literal but also trailing-flag (e.g. ``--verbose``) and
``-C <path>``-prefixed variants that a strict equality check would miss.
Comments, docstrings, and log-format strings that merely mention the
phrase (e.g. verify.py's DD5 notes, git_ops.py's own docstrings) are never
matched, avoiding false positives. A literal only counts as "inside the
chokepoint" when it falls within a ``_prune_registrations`` method defined
directly on the ``GitOps`` class (not merely any same-named function).
Mirrors the AST-walk / rglob / offender file:line list / skip-self pattern
used by orchestrator/tests/test_event_loop_antipattern_guard.py.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import TypeGuard

_THIS_FILE = Path(__file__).name

_EXPECTED_ARGV = ('git', 'worktree', 'prune')
_CHOKEPOINT_FUNCTION = '_prune_registrations'
_CHOKEPOINT_CLASS = 'GitOps'


def _str_constant(elt: ast.expr) -> str | None:
    """The string value of *elt* if it's a string ``Constant``, else None."""
    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
        return elt.value
    return None


def _is_prune_argv_literal(node: ast.AST) -> TypeGuard[ast.List | ast.Tuple]:
    """True if *node* is a List/Tuple literal invoking ``git worktree prune``.

    Matches not just the exact ``['git', 'worktree', 'prune']`` literal but
    any literal whose first element is the string constant ``'git'`` and
    whose remaining elements contain ``'worktree'`` followed later by
    ``'prune'`` as an in-order (not necessarily contiguous) subsequence.
    This also catches variants a strict 3-element equality check would
    miss: extra trailing flags (``['git', 'worktree', 'prune',
    '--verbose']``) and ``-C <path>``-prefixed forms (``['git', '-C', path,
    'worktree', 'prune']``, where ``path`` may be a non-constant
    expression). Argv assembled dynamically (e.g. via ``shlex.split`` or
    list concatenation) is intentionally out of scope — this is a static,
    best-effort AST check, not a full data-flow analysis.
    """
    if not isinstance(node, (ast.List, ast.Tuple)):
        return False
    if not node.elts or _str_constant(node.elts[0]) != _EXPECTED_ARGV[0]:
        return False
    remaining = (_str_constant(elt) for elt in node.elts[1:])
    return all(token in remaining for token in _EXPECTED_ARGV[1:])


def _chokepoint_ranges(tree: ast.Module) -> list[tuple[int, int]]:
    """Line ranges of every ``_prune_registrations`` method defined directly
    on a class named ``GitOps``.

    A plain name-only scan (any function called ``_prune_registrations``,
    anywhere) would also treat a differently-classed or module-level
    function of the same name as sanctioned. This walks the tree tracking
    the nearest *directly enclosing* class, so only a method defined
    straight in ``class GitOps:`` (not a same-named function elsewhere, and
    not a function nested inside a GitOps method) qualifies — matching the
    contract that ``GitOps._prune_registrations`` specifically is the
    chokepoint.
    """
    ranges: list[tuple[int, int]] = []

    def visit(node: ast.AST, enclosing_class: str | None) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                visit(child, child.name)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if (
                    child.name == _CHOKEPOINT_FUNCTION
                    and enclosing_class == _CHOKEPOINT_CLASS
                ):
                    end = getattr(child, 'end_lineno', child.lineno)
                    ranges.append((child.lineno, end))
                # A function body is no longer "directly inside" the
                # enclosing class, so functions nested within it (e.g. a
                # closure defined inside a method) don't inherit the
                # class qualification.
                visit(child, None)
            else:
                visit(child, enclosing_class)

    visit(tree, None)
    return ranges


def _find_prune_argv_literals(source: str) -> list[tuple[int, bool]]:
    """Return (lineno, inside_chokepoint) for each prune argv literal in *source*.

    ``inside_chokepoint`` is True when the literal's line falls within the
    line range of a ``_prune_registrations`` method defined directly on the
    ``GitOps`` class (see :func:`_chokepoint_ranges`). Returns an empty
    list if the file cannot be parsed.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    chokepoint_ranges = _chokepoint_ranges(tree)

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
        message_lines = [
            "Expected exactly one ['git', 'worktree', 'prune']-style argv "
            f'literal under orchestrator/src, found {total}.'
        ]
        if outside_chokepoint:
            outside_list = '\n  '.join(outside_chokepoint)
            message_lines.append(
                f'{len(outside_chokepoint)} site(s) are OUTSIDE '
                f'{_CHOKEPOINT_CLASS}.{_CHOKEPOINT_FUNCTION}:\n  {outside_list}'
            )
        message_lines.append(remediation)
        message_lines.append(f'\nAll offending sites:\n  {offender_list}')
        raise AssertionError('\n'.join(message_lines))
