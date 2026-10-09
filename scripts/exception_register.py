"""INV-12's exception register: every governed exception, its disposition, and the gaps.

``plans/inv12-exceptions-owned-or-ratified-prd.md``, decisions **D3**, **D5**
and **D14**, Contract sections "Ratification table" and "CLIs".  One register
document is measured from the tree and rendered two ways:

* **Ratification table** (D3) — ``docs/legibility/exception-ratifications.yaml``,
  the closed world of operator rulings.  This module is its only loader.
* **Static discovery** (D5) — every ``governed_exceptions(...)`` call evaluated
  at import, read from the AST of the tracked tree.  Its literals are bound
  through the vocabulary's own signature and constructors; the key set and its
  count rules stay with the runtime call at collection.
* **Inline suppressions** — read from ``scripts/inline_suppressions.py --json``
  in a subprocess, and passed through verbatim.
* **Closed world** — every ``Policy`` in a declaration, an inline marker or a
  ratified class row must name a row of the table.
* **Unregistered candidates** (D14) — report-only: module-level collections
  whose names match the census pattern and that no declaration governs.
* **Sweep age** — the owner sweep's run stamp.

**Exit codes.**  ``--report``: 0 clean; 1 closed-world or malformed-disposition
violations, one per line on stderr; 2 an instrument failure or a stale or
unreadable sweep stamp.  ``--json``: 0 whenever a document could be built, with
violations and the sweep's status as data; 2 for an instrument failure.  An
``ImportError`` is 2, never 1.

**Consumers.**  The owner sweep reads ``--json`` as its only data source; the
declaration tasks, the day-one rulings, the rulings' application and the
integration gate read ``--report``.
"""

from __future__ import annotations

import dataclasses
import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'shared' / 'src'))

try:
    import yaml
    from shared.governed_exceptions import MalformedDisposition, Policy
except ImportError as exc:
    print(
        f'exception_register.py: {exc} -- run it as '
        '`uv run --project shared python scripts/exception_register.py`',
        file=sys.stderr,
    )
    raise SystemExit(2) from exc

#: The ratification table, relative to the checkout root.
TABLE_PATH = 'docs/legibility/exception-ratifications.yaml'

_ROW_KEYS = ('id', 'date', 'source', 'covers')


class InstrumentFailure(Exception):
    """The register could not take a measurement — exit 2, never a finding.

    Every message names the file, path or key at fault.
    """


@dataclasses.dataclass(frozen=True)
class Ratification:
    """One operator ruling: a row of the ratification table."""

    id: str
    date: datetime.date
    source: str
    covers: str


def _row_label(path: Path, number: int, row: object) -> str:
    """How a refusal names one row: the table, its 1-based position, and its id."""
    row_id = row.get('id') if isinstance(row, dict) else None
    named = f' (id {row_id!r})' if isinstance(row_id, str) else ''
    return f'{path}: row {number}{named}'


def _check_layout(path: Path, text: str) -> None:
    """Refuse a non-block layout, or a row whose first key is not ``id``.

    Read off the parser's own node model: the rulings task's delivered check
    greps ``^- id: inv12-``, which only a block-style row with ``id`` first
    satisfies.
    """
    root = yaml.compose(text, Loader=yaml.SafeLoader)
    if not isinstance(root, yaml.SequenceNode) or not root.value:
        return
    if root.flow_style:
        raise InstrumentFailure(
            f'{path}: rows must be written block-style (one `- id: ...` row per ruling), '
            'not as a flow-style `[...]` list'
        )
    for number, node in enumerate(root.value, start=1):
        if not isinstance(node, yaml.MappingNode):
            raise InstrumentFailure(f'{path}: row {number} is not a mapping')
        if node.flow_style:
            raise InstrumentFailure(
                f'{path}: row {number} is written flow-style `{{...}}`; write it block-style'
            )
        keys = [key.value for key, _ in node.value]
        if keys[:1] != ['id']:
            raise InstrumentFailure(
                f'{path}: row {number} must have `id` as its first key, not {keys[:1]}'
            )
        if len(keys) != len(set(keys)):
            raise InstrumentFailure(f'{path}: row {number} repeats a key: {keys}')


def _ratification(label: str, row: object) -> Ratification:
    """One row, checked field by field against the schema."""
    if not isinstance(row, dict):
        raise InstrumentFailure(f'{label} is not a mapping')
    keys = set(row)
    if keys != set(_ROW_KEYS):
        raise InstrumentFailure(
            f'{label} must have exactly the keys {list(_ROW_KEYS)}; '
            f'missing {sorted(set(_ROW_KEYS) - keys)}, extra {sorted(map(str, keys - set(_ROW_KEYS)))}'
        )
    try:
        Policy(row['id'])
    except MalformedDisposition as exc:
        raise InstrumentFailure(f'{label}: id {exc}') from exc
    if type(row['date']) is not datetime.date:
        raise InstrumentFailure(
            f'{label}: date={row["date"]!r} must be an unquoted ISO date (YYYY-MM-DD)'
        )
    for key in ('source', 'covers'):
        if not isinstance(row[key], str) or not row[key].strip():
            raise InstrumentFailure(f'{label}: {key}={row[key]!r} must be a non-blank string')
    return Ratification(
        id=row['id'], date=row['date'], source=row['source'], covers=row['covers']
    )


def load_ratifications(path: Path) -> tuple[Ratification, ...]:
    """The ratification table at *path*, in file order.

    Raises:
        InstrumentFailure: the file is absent or unreadable, is not YAML, or any
            row breaks the schema its own header comment states.
    """
    try:
        text = path.read_text(encoding='utf-8')
    except (OSError, UnicodeDecodeError) as exc:
        raise InstrumentFailure(f'{path}: the ratification table could not be read -- {exc}') from exc
    try:
        document = yaml.safe_load(text)
        _check_layout(path, text)
    except yaml.YAMLError as exc:
        raise InstrumentFailure(f'{path}: the ratification table is not valid YAML -- {exc}') from exc
    if not isinstance(document, list):
        raise InstrumentFailure(
            f'{path}: the ratification table must be a YAML list of rows (an empty table is '
            f'written `[]`), not {type(document).__name__}'
        )
    rows: list[Ratification] = []
    seen: dict[str, int] = {}
    for number, row in enumerate(document, start=1):
        label = _row_label(path, number, row)
        ratification = _ratification(label, row)
        if ratification.id in seen:
            raise InstrumentFailure(
                f'{label}: duplicates the id of row {seen[ratification.id]}; ids are unique'
            )
        seen[ratification.id] = number
        rows.append(ratification)
    return tuple(rows)
