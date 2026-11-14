"""Pure readers for a Stage-1 flag's free-form, LLM-authored fields.

A LEAF: it imports nothing from ``fused_memory``, so a module that only needs
to match a token family against a flag's text depends on this rather than on
the module that does flag dedup.  A flag's task ids are read through the
sibling leaf ``flag_task_ids``.

Every function is total over malformed input — a wrong type or an empty value
yields an empty or ``False`` answer, never an exception.  Pure, sync, no I/O.
"""

from __future__ import annotations

from typing import Any

__all__ = ['collect_str_values', 'contains_any_casefolded']


def contains_any_casefolded(text: Any, family: tuple[str, ...]) -> bool:
    """Return True iff *text* is a non-empty ``str`` containing a *family* member.

    Both sides are casefolded and a member matches as a SUBSTRING.  A
    non-``str`` or empty *text*, or an empty *family*, is ``False``.
    """
    if not isinstance(text, str) or not text:
        return False
    folded = text.casefold()
    return any(member.casefold() in folded for member in family)


def collect_str_values(value: Any) -> list[str]:
    """Return every ``str`` VALUE reachable in *value*, in traversal order.

    Descends dicts (values only, never keys), lists and tuples.  Every other
    type (int, float, bool, None, bytes, ...) is ignored rather than
    stringified: only LLM-authored free text carries what a reader looks for,
    and a stringified number or structure could mint spurious tokens.
    """
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        children = value.values()
    elif isinstance(value, (list, tuple)):
        children = value
    else:
        return []
    return [text for child in children for text in collect_str_values(child)]
