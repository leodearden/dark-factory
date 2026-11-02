"""Renders pre-stringified rows as byte-deterministic fixed-width text tables,
with the shared '-'-for-unmeasured cell formatters.

Every deterministic eval report table renders through :func:`render_fixed_table`,
so the column-width and trailing-whitespace rules live in one place.
"""

from __future__ import annotations

from typing import Any


def render_fixed_table(
    columns: tuple[str, ...],
    rendered: list[dict[str, str]],
    *,
    header: str | None = None,
) -> list[str]:
    """Render PRE-STRINGIFIED rows as a fixed-width, byte-deterministic block.

    *rendered* is a list of ``{column: cell}`` dicts whose values are already
    strings — every value/None/precision decision stays with the caller, which
    is what keeps this generic. Column widths are the max of the header and all
    cells, so the block is a function of its input alone: no wall-clock, no
    dict-order dependence, and the same rows always render byte-identically.

    Returns the block as LINES (optional *header* line, column headers, a dashes
    rule, then the rows) rather than a joined string, so callers can append
    further sections — several emit two or three blocks separated by blanks.

    Lines are ``rstrip``ed: trailing padding on the last column is invisible
    whitespace that makes an otherwise-identical table differ byte-for-byte,
    and it lets a caller assert on a line's real ending (e.g. the per-config
    mean block's ``-`` for "nothing scored", which must never read as
    ``0.0000``).
    """
    widths = {
        col: (
            max(len(col), *(len(rr[col]) for rr in rendered))
            if rendered else len(col)
        )
        for col in columns
    }

    def _fmt(cells: dict[str, str]) -> str:
        return '  '.join(cells[col].ljust(widths[col]) for col in columns).rstrip()

    lines = [header] if header is not None else []
    lines.append(_fmt({col: col for col in columns}))
    lines.append('  '.join('-' * widths[col] for col in columns))
    lines.extend(_fmt(rr) for rr in rendered)
    return lines


def ci_cell(ci: dict[str, Any] | None) -> str:
    """Render a ``mean_ci95``-shaped dict's ``lo`` / ``hi`` as a ``[lo, hi]``
    interval (4 dp), or ``-`` when there is no interval (``None`` or ``{}``).
    """
    if not ci:
        return '-'
    return f'[{float(ci.get("lo", 0.0)):.4f}, {float(ci.get("hi", 0.0)):.4f}]'


def optional_float_cell(value: Any, *, precision: int = 4) -> str:
    """Render an OPTIONAL float cell: fixed precision, or ``-`` when unmeasured.

    A cell that measured NOTHING (``None``) must never render as one that
    scored zero, so ``None`` is ``-``, never ``0.0000``.
    """
    return '-' if value is None else f'{float(value):.{precision}f}'
