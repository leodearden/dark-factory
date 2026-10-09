"""Tests for ``scripts/exception_register.py`` — INV-12's exception register.

Under test: the ratification-table loader (D3), static discovery of governed
lists (D5, scenario 13), the closed-world Policy check (scenario 14), the
report and its exit ladder (scenario 15), the D14 census and the sweep stamp.
Fixture trees are real git repositories under ``tmp_path`` built with
``inline_suppression_fixtures.track_fixture_tree``; fixture markers live in
string literals, never in ``#`` comments.  Xdist-safe: all mutable state stays
in ``tmp_path``, and every run passes an absent ``--sweep-state``.
"""

from __future__ import annotations

import datetime
from collections.abc import Sequence
from pathlib import Path

import exception_register
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

#: A well-formed row, as (key, raw YAML value) pairs in the order written.
_ROW = (
    ('id', 'inv12-a'),
    ('date', '2026-10-09'),
    ('source', 'esc-5602-1'),
    ('covers', 'Fixture rows only.'),
)


def _block(*rows: Sequence[tuple[str, str]]) -> str:
    """*rows* as a block-style YAML sequence, keys in the order given."""
    lines = []
    for row in rows:
        for index, (key, value) in enumerate(row):
            lines.append(f'{"- " if index == 0 else "  "}{key}: {value}')
    return '\n'.join(lines) + '\n'


def _with(row: Sequence[tuple[str, str]], key: str, value: str) -> tuple[tuple[str, str], ...]:
    """*row* with *key*'s raw value replaced by *value*."""
    return tuple((name, value if name == key else raw) for name, raw in row)


def _write_table(root: Path, text: str) -> Path:
    path = root / exception_register.TABLE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    return path


# ---------------------------------------------------------------------------
# D3 — the ratification table loader.


def test_table_path_is_the_one_the_prd_names():
    assert exception_register.TABLE_PATH == 'docs/legibility/exception-ratifications.yaml'


def test_a_block_style_table_loads_its_rows_in_file_order(tmp_path: Path):
    second = _with(_with(_ROW, 'id', 'inv12-a'), 'covers', 'The second row.')
    first = _with(_with(_ROW, 'id', 'inv12-b'), 'date', '2026-01-02')
    path = _write_table(tmp_path, _block(first, second))

    rows = exception_register.load_ratifications(path)

    assert rows == (
        exception_register.Ratification(
            id='inv12-b',
            date=datetime.date(2026, 1, 2),
            source='esc-5602-1',
            covers='Fixture rows only.',
        ),
        exception_register.Ratification(
            id='inv12-a',
            date=datetime.date(2026, 10, 9),
            source='esc-5602-1',
            covers='The second row.',
        ),
    )
    assert type(rows[0].date) is datetime.date


def test_an_empty_table_with_its_schema_comment_loads_as_no_rows(tmp_path: Path):
    path = _write_table(tmp_path, '# the schema comment\n# - id: inv12-example\n[]\n')

    assert exception_register.load_ratifications(path) == ()


def test_the_live_table_loads():
    """The delivered file, not a count pin: rows arrive as the operator rules."""
    rows = exception_register.load_ratifications(REPO_ROOT / exception_register.TABLE_PATH)

    assert isinstance(rows, tuple)


@pytest.mark.parametrize(
    ('text', 'fragments'),
    [
        pytest.param(None, (), id='absent'),
        pytest.param('- id: [unclosed\n', (), id='yaml-syntax-error'),
        pytest.param('id: inv12-a\n', (), id='a-mapping'),
        pytest.param('inv12-a\n', (), id='a-scalar'),
        pytest.param('# nothing but a comment\n', (), id='empty'),
        pytest.param(
            '[{id: inv12-a, date: 2026-10-09, source: s, covers: c}]\n', (), id='flow-sequence'
        ),
        pytest.param('- inv12-a\n', ('row 1',), id='row-not-a-mapping'),
        pytest.param(_block(_ROW[:3]), ('row 1', 'covers'), id='missing-key'),
        pytest.param(_block((*_ROW, ('owner', 'x'))), ('row 1', 'owner'), id='extra-key'),
        pytest.param(_block((_ROW[1], _ROW[0], *_ROW[2:])), ('row 1', 'id'), id='id-not-first'),
        pytest.param(
            '- {id: inv12-a, date: 2026-10-09, source: s, covers: c}\n',
            ('row 1',),
            id='flow-row',
        ),
        pytest.param(_block(_with(_ROW, 'id', 'Inv12_A')), ('row 1', 'Inv12_A'), id='non-kebab'),
        pytest.param(_block(_ROW, _ROW), ('row 2', 'inv12-a'), id='duplicate-id'),
        pytest.param(_block(_with(_ROW, 'date', "'2026-10-09'")), ('row 1', 'date'), id='quoted'),
        pytest.param(
            _block(_with(_ROW, 'date', '2026-10-09T10:00:00Z')), ('row 1', 'date'), id='datetime'
        ),
        pytest.param(_block(_with(_ROW, 'date', 'true')), ('row 1', 'date'), id='bool-date'),
        pytest.param(_block(_with(_ROW, 'source', "'  '")), ('row 1', 'source'), id='blank-src'),
        pytest.param(_block(_with(_ROW, 'source', '5602')), ('row 1', 'source'), id='int-src'),
        pytest.param(_block(_with(_ROW, 'covers', "''")), ('row 1', 'covers'), id='blank-covers'),
    ],
)
def test_a_table_that_breaks_the_schema_is_an_instrument_failure(
    tmp_path: Path, text: str | None, fragments: tuple[str, ...]
):
    """Every fault is exit 2: a table that cannot be read leaves the closed world
    uncheckable (INV-11), and a flow-style or reordered row would silently defeat
    the rulings task's ``^- id: inv12-`` delivered check."""
    path = tmp_path / exception_register.TABLE_PATH
    if text is not None:
        _write_table(tmp_path, text)

    with pytest.raises(exception_register.InstrumentFailure) as excinfo:
        exception_register.load_ratifications(path)

    message = str(excinfo.value)
    assert exception_register.TABLE_PATH in message
    for fragment in fragments:
        assert fragment in message
