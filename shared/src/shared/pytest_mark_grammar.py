"""The static AST grammar pytest reads: ``pytest.mark.NAME`` elements and scope bindings like ``pytestmark``."""

from __future__ import annotations

import ast
from collections.abc import Sequence

__all__ = ['mark_elements', 'marker_name', 'pytestmark_elements', 'pytestmark_value', 'scope_binding']


def _is_name(node: ast.expr, name: str) -> bool:
    return isinstance(node, ast.Name) and node.id == name


def _bound_value(statement: ast.stmt, name: str) -> ast.expr | None:
    if isinstance(statement, ast.Assign):
        if any(_is_name(target, name) for target in statement.targets):
            return statement.value
        return None
    if isinstance(statement, ast.AnnAssign) and _is_name(statement.target, name):
        return statement.value
    return None


def pytestmark_value(statement: ast.stmt) -> ast.expr | None:
    """The value *statement* binds to ``pytestmark``, else None.

    Covers both the plain ``pytestmark = ...`` and the annotated
    ``pytestmark: list = ...`` spellings; an annotation with no value binds
    nothing.
    """
    return _bound_value(statement, 'pytestmark')


def scope_binding(body: Sequence[ast.stmt], name: str) -> ast.expr | None:
    """The value the binding of *name* that wins in a scope *body* gives it, else None.

    Only DIRECT children of *body* bind the scope's *name* — one bound inside
    an ``if`` does not — and the LAST binding wins, as in Python itself. Both
    the plain and the annotated spelling bind; an annotation with no value
    binds nothing.
    """
    value: ast.expr | None = None
    for statement in body:
        bound = _bound_value(statement, name)
        if bound is not None:
            value = bound
    return value


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


def pytestmark_elements(body: Sequence[ast.stmt]) -> list[ast.expr]:
    """The mark elements of the ``pytestmark`` binding that wins in a scope *body*, by :func:`scope_binding`.

    No binding yields an empty list.
    """
    value = scope_binding(body, 'pytestmark')
    return mark_elements(value) if value is not None else []
