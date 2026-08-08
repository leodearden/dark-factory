"""The single CSV-tolerant "read a task's dependency ids" parser.

``get_tasks`` returns task ids as STRINGS but a task's own ``dependencies``
list as INTS (or, in some rows, a CSV-formatted string rather than a list).
This drift was independently reimplemented — with different tolerance —
in at least three places (task_curator._task_dependencies, and two inline
copies in reconciliation/targeted.py); a copy missing the ``str()``
coercion silently fails to match any dependency id, and a copy missing the
CSV-string fallback silently misses CSV-shaped rows. See task 3615.

This module is a dependency-free leaf (mirrors utils/task_naming.py) so it
can be imported from the task curator, the reconciler, and any future
cancellation/dependent-edge sweep without import cycles.
"""
from __future__ import annotations

__all__ = ['task_dependency_ids']


def task_dependency_ids(task: dict) -> list[str]:
    """Return *task*'s dependency ids as a list of non-empty strings.

    Tolerates ``task['dependencies']`` being any of:

    - a list (of ints, strs, or a mix) — falsy entries (``None``, ``0``,
      ``''``) are dropped and each surviving entry is coerced with
      ``str()``. Note whitespace-only strings survive the list branch
      (unlike the CSV branch, which strips). Tuples and sets take this
      same branch; or
    - a CSV-formatted string — split on commas, each piece stripped of
      surrounding whitespace, empty pieces dropped; or
    - any other shape (a bare ``int`` for a single-dep row, a dict, ...) —
      stringified WHOLE into a single-element list rather than raising or
      being iterated. Iterating would silently yield a dict's *keys*, and
      raising is worse still: every caller sits inside a broad
      ``except Exception`` sweep, where a ``TypeError`` here would
      silently skip the whole sweep rather than one malformed row. An
      id that matches nothing fails closed; a raise fails open.

    A missing or falsy ``dependencies`` key (absent, ``None``, ``[]``,
    ``''``) returns ``[]``.

    Args:
        task: A raw task dict as returned by ``get_tasks``/``get_task``.

    Returns:
        Dependency ids as strings, in the source's iteration order.
    """
    deps = task.get('dependencies') or []
    if isinstance(deps, str):
        return [d.strip() for d in deps.split(',') if d.strip()]
    if not isinstance(deps, (list, tuple, set)):
        return [str(deps)] if deps else []
    return [str(d) for d in deps if d]
