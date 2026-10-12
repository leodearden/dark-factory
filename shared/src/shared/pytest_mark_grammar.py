"""The static AST grammar of pytest marks: ``pytest.mark.NAME`` elements and ``pytestmark`` bindings."""

from __future__ import annotations

import ast

__all__ = ['mark_elements', 'marker_name', 'pytestmark_value']


def _is_pytestmark_target(node: ast.expr) -> bool:
    return isinstance(node, ast.Name) and node.id == 'pytestmark'


def pytestmark_value(statement: ast.stmt) -> ast.expr | None:
    """The value *statement* binds to ``pytestmark``, else None.

    Covers both the plain ``pytestmark = ...`` and the annotated
    ``pytestmark: list = ...`` spellings; an annotation with no value binds
    nothing.
    """
    if isinstance(statement, ast.Assign):
        if any(_is_pytestmark_target(target) for target in statement.targets):
            return statement.value
        return None
    if isinstance(statement, ast.AnnAssign) and _is_pytestmark_target(statement.target):
        return statement.value
    return None


def marker_name(element: ast.expr) -> str | None:
    """The marker name in a ``pytest.mark.NAME`` / ``pytest.mark.NAME(...)`` element.

    Anything else — a bare constant, a local name, an unrelated attribute chain —
    yields None, so a caller can skip it without suppressing its siblings.
    """
    if isinstance(element, ast.Call):
        element = element.func
    if not isinstance(element, ast.Attribute):
        return None
    owner = element.value
    if (
        isinstance(owner, ast.Attribute)
        and owner.attr == 'mark'
        and isinstance(owner.value, ast.Name)
        and owner.value.id == 'pytest'
    ):
        return element.attr
    return None


def mark_elements(value: ast.expr) -> list[ast.expr]:
    """The marks a ``pytestmark`` value names: a list/tuple's elements, else the value alone."""
    if isinstance(value, ast.List | ast.Tuple):
        return list(value.elts)
    return [value]
