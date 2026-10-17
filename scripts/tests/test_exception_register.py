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
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import exception_register
import inline_suppressions
import pytest
from inline_suppression_fixtures import (
    RUFF_CONFIG,
    SUBPROCESS_TIMEOUT_SECS,
    run_git,
    track_fixture_tree,
    write_files,
)
from shared.governed_exceptions import (
    DECLARATION_FORMS,
    INLINE_MARKER_FORMS,
    Debt,
    MalformedDisposition,
    Policy,
    TaskRef,
    TicketRef,
)

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


# ---------------------------------------------------------------------------
# D5 — static discovery of governed-list declarations (scenario 13).

#: The import every fixture declaration module opens with, so a call written
#: after it sits on line 2.
_HEADER = (
    'from shared.governed_exceptions import Debt, Policy, TaskRef, TicketRef, '
    'governed_exceptions\n'
)

_TICKET_ID = 'tkt_0RTCC80EM92A7WD08D6RF6ZZPY'


def _declaring(call_args: str) -> str:
    """A fixture module whose line 2 is ``governed_exceptions(<call_args>)``."""
    return f'{_HEADER}governed_exceptions({call_args})\n'


#: The reviewer's reproduction: the function bound as ``declare``, whose call
#: carries an unratified Policy the register must never report clean.
_ALIASED_DECLARATION = (
    'from shared.governed_exceptions import Policy, governed_exceptions as declare\n'
    "declare('pkg.x', 'r', ['a'], dispositions={'a': Policy('no-such-row')})\n"
)


def _discover(root: Path, files: dict[str, str]):
    track_fixture_tree(root, files)
    return exception_register.discover_declarations(root)


def _refusal(root: Path, files: dict[str, str]) -> str:
    track_fixture_tree(root, files)
    with pytest.raises(exception_register.InstrumentFailure) as excinfo:
        exception_register.discover_declarations(root)
    return str(excinfo.value)


def test_a_module_level_declaration_is_read_into_typed_values(tmp_path: Path):
    source = (
        _HEADER
        + 'ALLOWLIST = frozenset({"a", "b", "c"})\n'
        + 'governed_exceptions(\n'
        + "    'pkg.tests.allowlist',\n"
        + "    'every entry is owned or ratified',\n"
        + '    sorted(ALLOWLIST),\n'
        + '    default=Debt(TaskRef(5149)),\n'
        + '    default_covers=2,\n'
        + "    dispositions={'c': Policy('inv12-x')},\n"
        + ')\n'
    )

    discovery = _discover(tmp_path, {'tests/test_decl.py': source})

    (declaration,) = discovery.declarations
    assert (declaration.path, declaration.line) == ('tests/test_decl.py', 3)
    assert declaration.list_id == 'pkg.tests.allowlist'
    assert declaration.rule == 'every entry is owned or ratified'
    assert declaration.default == Debt(TaskRef(5149))
    assert declaration.default_covers == 2
    assert dict(declaration.overrides) == {'c': Policy('inv12-x')}
    assert discovery.faults == ()


def test_overrides_are_read_only(tmp_path: Path):
    discovery = _discover(
        tmp_path,
        {'tests/test_decl.py': _declaring("'pkg.x', 'r', ['a'], dispositions={'a': Policy('inv12-x')}")},
    )

    assert not hasattr(discovery.declarations[0].overrides, '__setitem__')


@pytest.mark.parametrize(
    ('source', 'list_id', 'default'),
    [
        pytest.param(
            _declaring("list_id='pkg.kw', rule='r', keys=[], dispositions={}"),
            'pkg.kw',
            None,
            id='keyword-binding',
        ),
        pytest.param(
            'import shared.governed_exceptions as ge\n'
            "ge.governed_exceptions('pkg.alias', 'r', [], default=ge.Debt(ge.TaskRef(1)), "
            'default_covers=0)\n',
            'pkg.alias',
            Debt(TaskRef(1)),
            id='module-attribute',
        ),
        pytest.param(
            _declaring(f"'pkg.ticket', 'r', [], default=Debt(TicketRef('{_TICKET_ID}')), default_covers=0"),
            'pkg.ticket',
            Debt(TicketRef(_TICKET_ID)),
            id='ticket-owner',
        ),
        pytest.param(
            _HEADER + 'class Holder:\n    governed_exceptions(\'pkg.cls\', \'r\', [])\n',
            'pkg.cls',
            None,
            id='class-body',
        ),
        pytest.param(
            _HEADER + "if True:\n    governed_exceptions('pkg.branch', 'r', [])\n",
            'pkg.branch',
            None,
            id='module-level-block',
        ),
        pytest.param(
            _HEADER + "governed_exceptions ('pkg.space', 'r', [])\n",
            'pkg.space',
            None,
            id='space-before-paren',
        ),
        pytest.param(
            'import shared.governed_exceptions as ge\n'
            "(ge.governed_exceptions\n)('pkg.paren', 'r', [])\n",
            'pkg.paren',
            None,
            id='parenthesised-callee',
        ),
        pytest.param(
            _HEADER
            + 'import pytest\n'
            + "@pytest.mark.parametrize('x', [governed_exceptions('pkg.deco', 'r', [])])\n"
            + 'def test_x(x):\n'
            + '    pass\n',
            'pkg.deco',
            None,
            id='decorator',
        ),
        pytest.param(
            _HEADER + "def f(x=governed_exceptions('pkg.arg', 'r', [])):\n    return x\n",
            'pkg.arg',
            None,
            id='argument-default',
        ),
        pytest.param(
            _HEADER + "async def f(*, x=governed_exceptions('pkg.kw', 'r', [])):\n    return x\n",
            'pkg.kw',
            None,
            id='keyword-default',
        ),
        pytest.param(
            _HEADER + "def f(x: governed_exceptions('pkg.ann', 'r', [])):\n    return x\n",
            'pkg.ann',
            None,
            id='annotation',
        ),
        pytest.param(
            _HEADER + "make = lambda x=governed_exceptions('pkg.lambda', 'r', []): x\n",
            'pkg.lambda',
            None,
            id='lambda-default',
        ),
    ],
)
def test_every_import_time_spelling_is_a_declaration(
    tmp_path: Path, source: str, list_id: str, default: object
):
    discovery = _discover(tmp_path, {'tests/test_decl.py': source})

    (declaration,) = discovery.declarations
    assert declaration.list_id == list_id
    assert declaration.default == default


def test_a_call_inside_a_function_or_lambda_is_not_a_declaration(tmp_path: Path):
    """A call in a def, a method or a lambda is a USE of the function, evaluated
    only when that code runs — the vocabulary's own tests are exactly this shape,
    with a non-literal list id and an unratified Policy — so it is ignored.
    """
    source = (
        _HEADER
        + 'LIST_ID = "pkg.x"\n'
        + 'def test_x():\n'
        + "    governed_exceptions(LIST_ID, 'r', ['a'], dispositions={'a': Policy('inv12-x')})\n"
        + 'async def test_async():\n'
        + "    governed_exceptions(LIST_ID, 'r', [])\n"
        + 'class TestY:\n'
        + '    def test_m(self):\n'
        + "        governed_exceptions(self.ID, 'r', [])\n"
        + "make = lambda: governed_exceptions(LIST_ID, 'r', [])\n"
    )

    discovery = _discover(tmp_path, {'tests/test_decl.py': source})

    assert discovery.declarations == ()
    assert discovery.faults == ()
    assert discovery.files_parsed == 1


def test_an_untracked_declaration_is_not_read(tmp_path: Path):
    track_fixture_tree(tmp_path, {'tests/test_tracked.py': 'x = 1\n'})
    (tmp_path / 'tests' / 'test_untracked.py').write_text(
        _declaring("'pkg.untracked', 'r', []"), encoding='utf-8'
    )

    discovery = exception_register.discover_declarations(tmp_path)

    assert discovery.declarations == ()


def test_only_files_naming_the_function_are_parsed(tmp_path: Path):
    """The byte prefilter is the function's NAME, so a tracked file that does not
    parse but never names it is not this instrument's business."""
    discovery = _discover(
        tmp_path,
        {
            'tests/test_decl.py': _declaring("'pkg.x', 'r', []"),
            'pkg/plain.py': 'x = 1\n',
            'pkg/mention.py': 'import shared.governed_exceptions as ge\n',
            'pkg/broken.py': 'def broken(:\n',
        },
    )

    assert discovery.files_enumerated == 4
    assert discovery.files_parsed == 2
    assert [declaration.list_id for declaration in discovery.declarations] == ['pkg.x']


def test_scenario_13a_a_computed_list_id_is_an_instrument_failure(tmp_path: Path):
    message = _refusal(
        tmp_path,
        {'tests/test_decl.py': _HEADER + "LIST_ID = 'pkg.x'\ngoverned_exceptions(LIST_ID, 'r', [])\n"},
    )

    assert 'tests/test_decl.py:3' in message
    assert '`list_id`' in message


def test_scenario_13b_two_declarations_sharing_a_list_id_is_an_instrument_failure(
    tmp_path: Path,
):
    message = _refusal(
        tmp_path,
        {
            'tests/test_a.py': _declaring("'pkg.same', 'r', []"),
            'tests/test_b.py': _declaring("'pkg.same', 'other rule', []"),
        },
    )

    assert 'pkg.same' in message
    assert 'tests/test_a.py:2' in message
    assert 'tests/test_b.py:2' in message


@pytest.mark.parametrize(
    ('call_args', 'parameter'),
    [
        pytest.param("5, 'r', []", 'list_id', id='non-str-list-id'),
        pytest.param("'pkg.x', 'r', [], default=DEFAULT, default_covers=1", 'default', id='name-default'),
        pytest.param(
            "'pkg.x', 'r', [], default=make_policy('x'), default_covers=1", 'default', id='call-default'
        ),
        pytest.param("'pkg.x', 'r', [], dispositions=OVERRIDES", 'dispositions', id='name-overrides'),
        pytest.param(
            "'pkg.x', 'r', [], dispositions={**OVERRIDES}", 'dispositions', id='spread-overrides'
        ),
        pytest.param('*ARGS', '*ARGS', id='starred-args'),
        pytest.param("'pkg.x', 'r', [], **KW", '**KW', id='double-starred-kwargs'),
        pytest.param("'pkg.x', 'r', [], bogus=1", 'bogus', id='unknown-keyword'),
        pytest.param("'pkg.x'", 'rule', id='missing-rule'),
        pytest.param("'pkg.x', 5, []", 'rule', id='non-str-rule'),
        pytest.param(
            "'pkg.x', 'r', [], default=Policy('inv12-x'), default_covers=True",
            'default_covers',
            id='bool-count',
        ),
        pytest.param(
            "'pkg.x', 'r', [], default=Policy('inv12-x'), default_covers='2'",
            'default_covers',
            id='str-count',
        ),
        pytest.param(
            "'pkg.x', 'r', [], default=TaskRef(1), default_covers=1", 'default', id='bare-ref-default'
        ),
        pytest.param(
            "'pkg.x', 'r', ['a'], dispositions={'a': TaskRef(1)}", 'dispositions', id='bare-ref-override'
        ),
        pytest.param(
            "'pkg.x', 'r', ['a'], dispositions={1: Policy('inv12-x')}",
            'dispositions',
            id='non-str-override-key',
        ),
    ],
)
def test_a_declaration_the_static_reader_cannot_bind_is_an_instrument_failure(
    tmp_path: Path, call_args: str, parameter: str
):
    """D5: a non-literal argument other than ``keys`` is exit 2, never a skip."""
    message = _refusal(tmp_path, {'tests/test_decl.py': _declaring(call_args)})

    assert 'tests/test_decl.py:2' in message
    assert f'`{parameter}`' in message


@pytest.mark.parametrize(
    ('source', 'line'),
    [
        pytest.param(_ALIASED_DECLARATION, 1, id='import-alias'),
        pytest.param(
            'from pkg.helpers import governed_exceptions as declare\n',
            1,
            id='import-alias-from-any-module',
        ),
        pytest.param(_HEADER + 'declare = governed_exceptions\n', 2, id='assignment'),
        pytest.param(
            'import shared.governed_exceptions as ge\ndeclare = ge.governed_exceptions\n',
            2,
            id='attribute-assignment',
        ),
        pytest.param(
            _HEADER + 'declare: object = governed_exceptions\n', 2, id='annotated-assignment'
        ),
        pytest.param(_HEADER + '(declare := governed_exceptions)\n', 2, id='walrus'),
        pytest.param(
            _HEADER + 'class Holder:\n    declare = governed_exceptions\n', 3, id='class-body'
        ),
    ],
)
def test_binding_the_function_under_another_name_is_an_instrument_failure(
    tmp_path: Path, source: str, line: int
):
    """D5: the register follows calls by the function's own name, so an
    import-time alias would hide them — here, and in any module importing it."""
    message = _refusal(tmp_path, {'tests/test_decl.py': source})

    assert f'tests/test_decl.py:{line}' in message
    assert '`declare`' in message
    assert 'governed_exceptions' in message


def test_naming_the_function_without_renaming_it_is_not_refused(tmp_path: Path):
    source = (
        _HEADER
        + 'NAME = governed_exceptions.__name__\n'
        + 'governed_exceptions = governed_exceptions\n'
        + 'import shared.governed_exceptions as ge\n'
        + 'def helper():\n'
        + '    from shared.governed_exceptions import governed_exceptions as inner\n'
        + '    rebound = governed_exceptions\n'
        + '    return inner, rebound\n'
    )

    discovery = _discover(tmp_path, {'tests/test_decl.py': source})

    assert discovery.declarations == ()
    assert discovery.faults == ()


@pytest.mark.parametrize(
    ('path', 'import_line'),
    [
        pytest.param(
            'tests/test_decl.py', 'from shared import governed_exceptions as ge', id='absolute'
        ),
        pytest.param(
            'shared/src/shared/holder.py', 'from . import governed_exceptions as ge', id='relative'
        ),
    ],
)
def test_aliasing_the_vocabulary_module_from_its_package_is_followed(
    tmp_path: Path, path: str, import_line: str
):
    """``from shared import governed_exceptions as ge`` binds the MODULE, as
    ``import shared.governed_exceptions as ge`` does, so it renames nothing."""
    source = (
        f'{import_line}\n'
        "ge.governed_exceptions('pkg.x', 'r', [], default=ge.Policy('inv12-x'), default_covers=0)\n"
    )

    discovery = _discover(tmp_path, {path: source})

    (declaration,) = discovery.declarations
    assert (declaration.path, declaration.line, declaration.list_id) == (path, 2, 'pkg.x')
    assert declaration.default == Policy('inv12-x')


@pytest.mark.parametrize(
    'keys',['sorted(ALLOWLIST)', '(key for key in ALLOWLIST)', 'ALLOWLIST', 'mod.KEYS']
)
def test_keys_are_never_evaluated(tmp_path: Path, keys: str):
    """Only the runtime call sees the key set (D5), so any expression is accepted."""
    discovery = _discover(tmp_path, {'tests/test_decl.py': _declaring(f"'pkg.x', 'r', {keys}")})

    assert [declaration.list_id for declaration in discovery.declarations] == ['pkg.x']


def test_a_tracked_file_holding_the_call_that_does_not_parse_is_an_instrument_failure(
    tmp_path: Path,
):
    message = _refusal(tmp_path, {'tests/test_decl.py': "governed_exceptions('pkg.x',\n"})

    assert 'tests/test_decl.py' in message


@pytest.mark.parametrize(
    ('call_args', 'construct'),
    [
        pytest.param(
            "'pkg.x', 'r', [], default=Debt(TaskRef(0)), default_covers=1",
            lambda: TaskRef(0),
            id='task-zero',
        ),
        pytest.param(
            "'pkg.x', 'r', ['a'], dispositions={'a': Policy('Not_Kebab')}",
            lambda: Policy('Not_Kebab'),
            id='non-kebab-policy',
        ),
        pytest.param(
            "'pkg.x', 'r', [], default=Debt(TicketRef('nope')), default_covers=1",
            lambda: TicketRef('nope'),
            id='bad-ticket',
        ),
    ],
)
def test_a_malformed_disposition_literal_is_a_fault_at_the_site(
    tmp_path: Path, call_args: str, construct
):
    """Exit 1, not 2: the vocabulary fixes a malformed disposition at the site.
    The declaration stays, less the disposition its fault already names."""
    with pytest.raises(MalformedDisposition) as expected:
        construct()

    discovery = _discover(tmp_path, {'tests/test_decl.py': _declaring(call_args)})

    (fault,) = discovery.faults
    assert (fault.path, fault.line) == ('tests/test_decl.py', 2)
    assert str(expected.value) in fault.reason
    assert fault.forms == DECLARATION_FORMS
    (declaration,) = discovery.declarations
    assert declaration.list_id == 'pkg.x'
    assert (declaration.default, dict(declaration.overrides)) == (None, {})


def test_a_declaration_with_a_malformed_disposition_still_holds_its_list_id(tmp_path: Path):
    message = _refusal(
        tmp_path,
        {
            'tests/test_a.py': _declaring("'pkg.same', 'r', ['a'], dispositions={'a': Policy('Bad_Id')}"),
            'tests/test_b.py': _declaring("'pkg.same', 'r', []"),
        },
    )

    assert 'tests/test_a.py:2' in message
    assert 'tests/test_b.py:2' in message


# ---------------------------------------------------------------------------
# D3 — the closed world, and the exit ladder, through main().

_UNRATIFIED_TREES = {
    'declaration-default': {
        'tests/test_decl.py': _declaring(
            "'pkg.x', 'r', [], default=Policy('no-such-row'), default_covers=0"
        )
    },
    'declaration-override': {
        'tests/test_decl.py': _declaring(
            "'pkg.x', 'r', ['a'], dispositions={'a': Policy('no-such-row')}"
        )
    },
    'inline-marker': {'m.py': 'x = 1  # type: ignore[arg-type]  # ratified: no-such-row\n'},
}


def _register_tree(root: Path, files: dict[str, str], *, table: str = '[]\n') -> None:
    """A fixture checkout the register can measure: a table, a ruff config, *files*."""
    track_fixture_tree(
        root,
        {'pyproject.toml': RUFF_CONFIG, exception_register.TABLE_PATH: table, **files},
    )


def _main(root: Path, *args: str) -> int:
    """The register over *root*, never reading host sweep state."""
    return exception_register.main(
        [*args, '--root', str(root), '--sweep-state', str(root / 'absent.json')]
    )


@pytest.mark.parametrize(
    ('tree', 'site', 'subject', 'forms'),
    [
        ('declaration-default', 'tests/test_decl.py:2', 'pkg.x', DECLARATION_FORMS),
        ('declaration-override', 'tests/test_decl.py:2', 'pkg.x', DECLARATION_FORMS),
        ('inline-marker', 'm.py:1', 'type: ignore', INLINE_MARKER_FORMS),
    ],
)
def test_scenario_14_a_policy_naming_no_table_row_is_one_violation_line(
    tmp_path: Path, capsys, tree: str, site: str, subject: str, forms: tuple[str, ...]
):
    _register_tree(tmp_path, _UNRATIFIED_TREES[tree])

    assert _main(tmp_path, '--report') == 1

    (line,) = capsys.readouterr().err.strip().splitlines()
    assert 'no-such-row' in line
    assert exception_register.TABLE_PATH in line
    assert site in line
    assert subject in line
    for form in forms:
        assert form in line


@pytest.mark.parametrize('tree', sorted(_UNRATIFIED_TREES))
def test_scenario_14_the_same_policy_is_green_once_the_row_exists(
    tmp_path: Path, capsys, tree: str
):
    _register_tree(
        tmp_path, _UNRATIFIED_TREES[tree], table=_block(_with(_ROW, 'id', 'no-such-row'))
    )

    assert _main(tmp_path, '--report') == 0, capsys.readouterr().err


def _inline_document(**overrides: object) -> dict[str, object]:
    """A hand-written inline report of the schema the register pins."""
    document: dict[str, object] = {
        'schema_version': exception_register.INLINE_REPORT_SCHEMA_VERSION,
        'status': 'advisory',
        'files_enumerated': 0,
        'files_tokenized': 0,
        'sites': 0,
        'kind_totals': {},
        'ownership': {'class': 0, 'debt': 0, 'policy': 0, 'unowned': 0},
        'by_kind_code': [],
        'by_file': [],
        'debt': [],
        'ratified': [],
        'classes': [],
        'class_table': [],
        'baseline': {'path': 'b.json', 'present': False, 'excess': None, 'slack': None},
        'violations': [],
    }
    document.update(overrides)
    return document


def test_a_class_row_citing_no_table_row_is_a_closed_world_violation():
    """The class table is reached only through the inline report, so it is
    tested at the parsing seam: the scanner's class-table injection runs in
    another process and cannot be reached from here."""
    inline = exception_register.parse_inline_report(
        _inline_document(class_table=[{'class': 'noqa[E731]@tests', 'ratified': 'no-such-row'}])
    )

    (violation,) = exception_register.closed_world_violations((), (), inline)

    rendered = violation.render()
    assert 'no-such-row' in rendered
    assert 'noqa[E731]@tests' in rendered
    assert exception_register.TABLE_PATH in rendered
    row = exception_register.Ratification(
        id='no-such-row', date=datetime.date(2026, 10, 9), source='s', covers='c'
    )
    assert exception_register.closed_world_violations((row,), (), inline) == ()


def test_a_malformed_disposition_literal_fails_the_report(tmp_path: Path, capsys):
    _register_tree(
        tmp_path,
        {'tests/test_decl.py': _declaring("'pkg.x', 'r', [], default=Debt(TaskRef(0)), default_covers=1")},
    )

    assert _main(tmp_path, '--report') == 1
    assert 'tests/test_decl.py:2' in capsys.readouterr().err


def test_scenario_13_through_main_is_exit_two_with_no_traceback(tmp_path: Path, capsys):
    _register_tree(
        tmp_path,
        {'tests/test_decl.py': _HEADER + "LIST_ID = 'pkg.x'\ngoverned_exceptions(LIST_ID, 'r', [])\n"},
    )

    assert _main(tmp_path, '--report') == 2

    err = capsys.readouterr().err
    assert err.startswith('exception_register.py:')
    assert 'Traceback' not in err


def test_an_aliased_declaration_is_exit_two_and_never_a_clean_report(tmp_path: Path, capsys):
    _register_tree(tmp_path, {'tests/test_decl.py': _ALIASED_DECLARATION})

    assert _main(tmp_path, '--report') == 2

    err = capsys.readouterr().err
    assert err.startswith('exception_register.py:')
    assert 'tests/test_decl.py:1' in err
    assert 'Traceback' not in err
    assert _main(tmp_path, '--json') == 2


def test_a_failure_of_the_inline_scanner_is_the_registers_exit_two(tmp_path: Path, capsys):
    _register_tree(tmp_path, {'broken.py': 'value = (  # noqa: E402\n'})

    assert _main(tmp_path, '--report') == 2

    err = capsys.readouterr().err
    assert err.startswith('exception_register.py:')
    assert 'inline_suppressions.py' in err


def test_an_inline_report_of_another_schema_version_is_refused():
    with pytest.raises(exception_register.InstrumentFailure) as excinfo:
        exception_register.parse_inline_report(_inline_document(schema_version=7))

    message = str(excinfo.value)
    assert 'schema_version 7' in message
    assert f'schema_version {exception_register.INLINE_REPORT_SCHEMA_VERSION}' in message


@pytest.mark.parametrize(
    'block', ['baseline', 'by_file', 'by_kind_code', 'class_table', 'ownership', 'ratified', 'sites']
)
def test_an_inline_report_missing_a_block_is_refused(block: str):
    document = _inline_document()
    del document[block]

    with pytest.raises(exception_register.InstrumentFailure) as excinfo:
        exception_register.parse_inline_report(document)

    assert block in str(excinfo.value)


def test_an_inline_report_with_a_mistyped_field_is_refused():
    with pytest.raises(exception_register.InstrumentFailure) as excinfo:
        exception_register.parse_inline_report(_inline_document(sites='many'))

    assert 'sites' in str(excinfo.value)


def test_a_missing_shared_import_is_exit_two_and_never_exit_one(tmp_path: Path):
    """Only a real process can prove it: the fault happens before main exists.

    The script is copied with ``source_measures.py``, its one sibling import, so
    the import that fails is yaml's or shared's; ``-S`` hides the editable
    install, which an emptied ``PYTHONPATH`` alone cannot.
    """
    scripts = REPO_ROOT / 'scripts'
    for name in ('exception_register.py', 'source_measures.py'):
        shutil.copyfile(scripts / name, tmp_path / name)
    env = {key: value for key, value in os.environ.items() if key != 'PYTHONPATH'}

    completed = subprocess.run(
        [sys.executable, '-S', str(tmp_path / 'exception_register.py'), '--report'],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT_SECS,
        env=env,
    )

    assert completed.returncode == 2, completed.stderr
    assert 'uv run --project shared' in completed.stderr
    assert 'Traceback' not in completed.stderr


# ---------------------------------------------------------------------------
# The report (scenario 15) and the --json document.


def _report(root: Path, capsys) -> tuple[int, str, str]:
    capsys.readouterr()
    code = _main(root, '--report')
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _json_text(root: Path, capsys) -> str:
    capsys.readouterr()
    code = _main(root, '--json')
    captured = capsys.readouterr()
    assert code == 0, captured.err
    return captured.out


def _document(root: Path, capsys) -> dict:
    return json.loads(_json_text(root, capsys))


def _normalised_lines(text: str) -> list[str]:
    """Each line of *text* with its whitespace runs collapsed, for row lookups."""
    return [' '.join(line.split()) for line in text.splitlines()]


_SCENARIO_15_TREE = {
    'pkg/__init__.py': '',
    'pkg/allow.py': (
        "ALLOWLIST = frozenset({'secret-key-alpha', 'secret-key-beta', 'secret-key-gamma'})\n"
    ),
    'tests/test_decl.py': (
        _HEADER
        + 'from pkg.allow import ALLOWLIST\n'
        + 'governed_exceptions(\n'
        + "    'pkg.allowlist',\n"
        + "    'every allowed key is owned or ratified',\n"
        + '    sorted(ALLOWLIST),\n'
        + '    default=Debt(TaskRef(5149)),\n'
        + '    default_covers=2,\n'
        + "    dispositions={'secret-key-gamma': Policy('inv12-x')},\n"
        + ')\n'
    ),
}

_TABLE_WITH_INV12_X = _block(_with(_ROW, 'id', 'inv12-x'))


def test_scenario_15_a_defaulted_list_is_reported_without_its_key_set(tmp_path: Path, capsys):
    _register_tree(tmp_path, _SCENARIO_15_TREE, table=_TABLE_WITH_INV12_X)

    code, out, err = _report(tmp_path, capsys)

    assert code == 0, err
    lines = _normalised_lines(out)
    assert 'governed lists: 1 declaration' in lines
    assert 'pkg.allowlist tests/test_decl.py:3' in lines
    assert 'rule: every allowed key is owned or ratified' in lines
    assert 'default: Debt(TaskRef(5149)) covers 2' in lines
    assert 'overrides: 1' in lines
    assert "'secret-key-gamma': Policy('inv12-x')" in lines
    assert 'secret-key-alpha' not in out
    assert 'secret-key-beta' not in out


def test_json_declarations_carry_the_literals_and_never_the_key_set(tmp_path: Path, capsys):
    _register_tree(tmp_path, _SCENARIO_15_TREE, table=_TABLE_WITH_INV12_X)

    (declaration,) = _document(tmp_path, capsys)['declarations']

    assert declaration == {
        'list_id': 'pkg.allowlist',
        'rule': 'every allowed key is owned or ratified',
        'path': 'tests/test_decl.py',
        'line': 3,
        'default': {'type': 'debt', 'owner': {'type': 'task', 'id': 5149}},
        'default_covers': 2,
        'overrides': [
            {'key': 'secret-key-gamma', 'disposition': {'type': 'policy', 'ratified': 'inv12-x'}}
        ],
    }


def test_an_unseeded_tree_heads_the_report_with_not_seeded(tmp_path: Path, capsys):
    _register_tree(tmp_path, {'m.py': 'a = 1  # type: ignore[arg-type]\n'})

    code, out, err = _report(tmp_path, capsys)

    assert code == 0, err
    assert out.splitlines()[0] == 'grandfathered (unowned): not seeded'
    assert 'baseline slack: n/a' in _normalised_lines(out)
    assert _document(tmp_path, capsys)['grandfathered'] == {'seeded': False, 'unowned': None}


def _seed(root: Path, files: dict[str, str]) -> None:
    """Track *files* and seed the inline baseline from them, where the register reads it."""
    _register_tree(root, files)
    (root / 'scripts').mkdir(exist_ok=True)
    assert inline_suppressions.main(['--seed', '--root', str(root)]) == 0


def _retrack(root: Path, files: dict[str, str], *, removing: tuple[str, ...] = ()) -> None:
    write_files(root, files)
    for relative in removing:
        (root / relative).unlink()
    run_git(['add', '-A', '-f'], cwd=root)


_TWO_UNOWNED = {'m.py': 'a = 1  # type: ignore[arg-type]\nb = 2  # type: ignore[attr-defined]\n'}


def test_the_headline_counts_only_what_the_baseline_holds(tmp_path: Path, capsys):
    """A new unowned marker is excess — a violation at the inline gate — and is
    never folded into the grandfathered population, which can only fall."""
    _seed(tmp_path, _TWO_UNOWNED)
    _retrack(tmp_path, {'n.py': 'c = 3  # type: ignore[no-any-return]\n'})

    code, out, err = _report(tmp_path, capsys)

    assert code == 0, err
    assert out.splitlines()[0] == 'grandfathered (unowned): 2'
    assert 'baseline excess: 1' in _normalised_lines(out)
    assert _document(tmp_path, capsys)['grandfathered'] == {'seeded': True, 'unowned': 2}


def test_a_removed_grandfathered_marker_is_baseline_slack(tmp_path: Path, capsys):
    _seed(tmp_path, _TWO_UNOWNED)
    _retrack(tmp_path, {'m.py': 'a = 1  # type: ignore[arg-type]\nb = 2\n'})

    _, out, _ = _report(tmp_path, capsys)

    assert out.splitlines()[0] == 'grandfathered (unowned): 1'
    assert 'baseline slack: 1' in _normalised_lines(out)
    document = _document(tmp_path, capsys)
    assert document['baseline'] == {'present': True, 'slack_total': 1}


def test_the_report_tabulates_inline_sites_by_kind_and_code_and_by_file(
    tmp_path: Path, capsys
):
    files = {
        f'pkg/f{index:02d}.py': 'x = 1  # pragma: no cover\n'
        for index in range(exception_register.TOP_FILES + 2)
    }
    files['pkg/f00.py'] = (
        'x = 1  # type: ignore[arg-type]\n'
        'y = 2  # type: ignore[arg-type]\n'
        'z = 3  # noqa: E402, F401\n'
    )
    _register_tree(tmp_path, files)

    _, out, _ = _report(tmp_path, capsys)

    lines = _normalised_lines(out)
    assert '2 type: ignore[arg-type]' in lines
    assert '1 noqa[E402]' in lines
    assert '1 noqa[F401]' in lines
    assert f'{exception_register.TOP_FILES + 1} pragma: no cover' in lines
    assert '3 pkg/f00.py' in lines
    assert '1 pkg/f01.py' in lines
    assert '… 2 more files' in lines
    assert f'1 pkg/f{exception_register.TOP_FILES + 1:02d}.py' not in lines


def test_a_tree_with_no_declarations_says_so(tmp_path: Path, capsys):
    _register_tree(tmp_path, {'m.py': 'x = 1\n'})

    _, out, _ = _report(tmp_path, capsys)

    assert 'governed lists: 0 declarations' in _normalised_lines(out)


def test_json_exits_zero_and_carries_a_closed_world_violation_as_data(tmp_path: Path, capsys):
    _register_tree(tmp_path, _UNRATIFIED_TREES['declaration-default'])

    (violation,) = _document(tmp_path, capsys)['violations']

    assert (violation['path'], violation['line']) == ('tests/test_decl.py', 2)
    assert 'no-such-row' in violation['reason']


def test_two_json_runs_over_one_tree_emit_identical_bytes(tmp_path: Path, capsys):
    _register_tree(tmp_path, _SCENARIO_15_TREE, table=_TABLE_WITH_INV12_X)

    first = _json_text(tmp_path, capsys)
    second = _json_text(tmp_path, capsys)

    assert first == second
    document = json.loads(first)
    assert list(document) == sorted(document)
    assert document['schema_version'] == exception_register.REGISTER_SCHEMA_VERSION


def test_json_carries_the_table_and_the_inline_report_verbatim(tmp_path: Path, capsys):
    _register_tree(
        tmp_path, {'m.py': 'a = 1  # type: ignore[arg-type]\n'}, table=_TABLE_WITH_INV12_X
    )

    document = _document(tmp_path, capsys)

    assert document['ratification_table'] == {
        'path': exception_register.TABLE_PATH,
        'rows': [
            {'id': 'inv12-x', 'date': '2026-10-09', 'source': 'esc-5602-1', 'covers': 'Fixture rows only.'}
        ],
    }
    assert inline_suppressions.main(['--json', '--root', str(tmp_path)]) == 0
    assert document['inline'] == json.loads(capsys.readouterr().out)


# ---------------------------------------------------------------------------
# D14 — the report-only census of unregistered candidate lists.

_CENSUS_TREE = {
    'pkg/__init__.py': '',
    'pkg/a.py': (
        'import re\n'
        "ALLOWLIST = frozenset({'x'})\n"
        "_GRANDFATHERED = ('y',)\n"
        "KNOWN_FLAKES: tuple[str, ...] = ('z',)\n"
        'PINNED_SITES = {k: v for k, v in []}\n'
        "_EXEMPT_RE = re.compile('q')\n"
        "NAMES = ['n']\n"
        'def build():\n'
        '    ALLOWLIST = []\n'
        '    return ALLOWLIST\n'
    ),
    'tests/test_decl.py': (
        _HEADER
        + 'from pkg.a import ALLOWLIST\n'
        + "governed_exceptions('pkg.allowlist', 'rule', sorted(ALLOWLIST), "
        + "default=Policy('r'), default_covers=1)\n"
    ),
    'tests/test_decl2.py': (
        _HEADER
        + 'import pkg.a as a\n'
        + "governed_exceptions('pkg.flakes', 'rule', a.KNOWN_FLAKES, "
        + "default=Policy('r'), default_covers=1)\n"
    ),
}


def _ref(path: str, symbol: str):
    return exception_register.ListRef(path=path, symbol=symbol)


def _unregistered(root: Path, not_exception_lists=None) -> tuple:
    return exception_register.unregistered_candidates(
        exception_register.collection_candidates(root),
        exception_register.discover_declarations(root).declarations,
        {} if not_exception_lists is None else not_exception_lists,
    )


def test_a_list_ref_renders_as_path_and_symbol():
    assert _ref('pkg/a.py', 'ALLOWLIST').render() == 'pkg/a.py::ALLOWLIST'


def test_the_census_finds_module_level_collections_whose_names_match(tmp_path: Path):
    track_fixture_tree(tmp_path, _CENSUS_TREE)

    assert exception_register.collection_candidates(tmp_path) == (
        _ref('pkg/a.py', 'ALLOWLIST'),
        _ref('pkg/a.py', 'KNOWN_FLAKES'),
        _ref('pkg/a.py', 'PINNED_SITES'),
        _ref('pkg/a.py', '_GRANDFATHERED'),
    )


def test_a_container_a_declaration_imports_is_not_unregistered(tmp_path: Path):
    """Both spellings resolve through the declaring module's own imports:
    ``from pkg.a import ALLOWLIST`` and ``import pkg.a as a`` then ``a.KNOWN_FLAKES``."""
    track_fixture_tree(tmp_path, _CENSUS_TREE)

    assert _unregistered(tmp_path) == (
        _ref('pkg/a.py', 'PINNED_SITES'),
        _ref('pkg/a.py', '_GRANDFATHERED'),
    )


@pytest.mark.parametrize(
    'keys', ['LOCAL_ALLOWLIST', 'LOCAL_ALLOWLIST.keys()', 'sorted(LOCAL_ALLOWLIST.keys())']
)
def test_a_same_file_container_is_declared_by_that_files_declaration(tmp_path: Path, keys: str):
    track_fixture_tree(
        tmp_path,
        {
            'tests/test_local.py': (
                _HEADER
                + "LOCAL_ALLOWLIST = {'a': 'why'}\n"
                + f"governed_exceptions('pkg.local', 'r', {keys}, "
                + "default=Policy('r'), default_covers=1)\n"
            )
        },
    )

    assert exception_register.collection_candidates(tmp_path) == (
        _ref('tests/test_local.py', 'LOCAL_ALLOWLIST'),
    )
    assert _unregistered(tmp_path) == ()


@pytest.mark.parametrize(
    ('import_line', 'keys'),
    [
        pytest.param('from pkg.a import ALLOWLIST', 'ALLOWLIST.keys()', id='imported-name'),
        pytest.param('import pkg.a as a', 'sorted(a.ALLOWLIST.keys())', id='module-attribute'),
    ],
)
def test_an_imported_container_read_through_an_attribute_is_declared(
    tmp_path: Path, import_line: str, keys: str
):
    track_fixture_tree(
        tmp_path,
        {
            'pkg/__init__.py': '',
            'pkg/a.py': "ALLOWLIST = {'x': 'why'}\n",
            'tests/test_decl.py': (
                f'{_HEADER}{import_line}\n'
                f"governed_exceptions('pkg.allowlist', 'r', {keys}, default=Policy('r'), "
                'default_covers=1)\n'
            ),
        },
    )

    assert _unregistered(tmp_path) == ()


def test_a_declaration_with_a_malformed_disposition_still_governs_its_container(
    tmp_path: Path,
):
    """One mistake is one fault: the container is not also reported unregistered."""
    track_fixture_tree(
        tmp_path,
        {
            'tests/test_local.py': (
                _HEADER
                + "LOCAL_ALLOWLIST = ('a',)\n"
                + "governed_exceptions('pkg.local', 'r', LOCAL_ALLOWLIST, "
                + "dispositions={'a': Policy('Bad_Id')})\n"
            )
        },
    )

    assert len(exception_register.discover_declarations(tmp_path).faults) == 1
    assert _unregistered(tmp_path) == ()


@pytest.mark.parametrize(
    ('module', 'unregistered'),
    [
        pytest.param('foo', 'pkg/src/pkg/foo.py', id='top-level-module'),
        pytest.param('pkg.foo', 'scripts/foo.py', id='src-layout-package'),
    ],
)
def test_an_import_reaches_only_the_module_its_dotted_name_resolves_to(
    tmp_path: Path, module: str, unregistered: str
):
    """A ``<pkg>/src/<pkg>/`` file imports as ``pkg.…``, never by a bare suffix, so
    two same-named modules' same-named containers are told apart."""
    track_fixture_tree(
        tmp_path,
        {
            'scripts/foo.py': "ALLOWLIST = ('a',)\n",
            'pkg/src/pkg/__init__.py': '',
            'pkg/src/pkg/foo.py': "ALLOWLIST = ('b',)\n",
            'scripts/tests/test_decl.py': (
                f'{_HEADER}from {module} import ALLOWLIST\n'
                "governed_exceptions('x.allowlist', 'r', ALLOWLIST, default=Policy('r'), "
                'default_covers=1)\n'
            ),
        },
    )

    assert _unregistered(tmp_path) == (_ref(unregistered, 'ALLOWLIST'),)


def test_a_same_symbol_in_another_module_is_not_declared_by_it(tmp_path: Path):
    """Matching by symbol alone would mark both guard tests' ``ALLOWLIST`` declared."""
    files = dict(_CENSUS_TREE)
    files['pkg/b.py'] = "ALLOWLIST = ['other']\n"
    track_fixture_tree(tmp_path, files)

    assert _ref('pkg/b.py', 'ALLOWLIST') in _unregistered(tmp_path)
    assert _ref('pkg/a.py', 'ALLOWLIST') not in _unregistered(tmp_path)


def test_a_reasoned_not_exception_row_removes_its_candidate_and_a_stale_one_is_inert(
    tmp_path: Path,
):
    track_fixture_tree(tmp_path, _CENSUS_TREE)
    rows = {
        _ref('pkg/a.py', '_GRANDFATHERED'): 'grants authority; design, not an exception',
        _ref('pkg/gone.py', 'ALLOWLIST'): 'the module this row named was deleted',
    }

    assert _unregistered(tmp_path, rows) == (_ref('pkg/a.py', 'PINNED_SITES'),)


def test_the_census_reaches_the_report_and_never_the_exit_code(tmp_path: Path, capsys):
    _register_tree(tmp_path, _CENSUS_TREE, table=_block(_with(_ROW, 'id', 'r')))

    code, out, err = _report(tmp_path, capsys)

    assert code == 0, err
    lines = _normalised_lines(out)
    assert 'unregistered candidates (report-only): 2' in lines
    assert 'pkg/a.py::PINNED_SITES' in lines
    assert 'pkg/a.py::_GRANDFATHERED' in lines
    assert _document(tmp_path, capsys)['unregistered'] == [
        {'path': 'pkg/a.py', 'symbol': 'PINNED_SITES'},
        {'path': 'pkg/a.py', 'symbol': '_GRANDFATHERED'},
    ]


@pytest.mark.parametrize('reason', ['', '   '])
def test_a_not_exception_row_with_a_blank_reason_is_refused(reason: str):
    """A classification says the rule does not apply, so it needs a reason (D14)."""
    with pytest.raises(exception_register.InstrumentFailure) as excinfo:
        exception_register.validate_not_exception_lists({_ref('pkg/a.py', 'ALLOWLIST'): reason})

    assert 'pkg/a.py::ALLOWLIST' in str(excinfo.value)


# ---------------------------------------------------------------------------
# The age of the owner sweep's last run.

_NOW = datetime.datetime(2026, 10, 9, 12, 0, tzinfo=datetime.UTC)


def _stamp(last_run: datetime.datetime | str, *, ok: object = True, **extra: object) -> dict:
    """A sweep result object of the Contract's shape."""
    return {
        'checked': 3,
        'findings': 0,
        'filed': 0,
        'escalated': 0,
        'skipped_existing': 0,
        'overflow': 0,
        'slack': 0,
        'last_run': last_run if isinstance(last_run, str) else last_run.isoformat(),
        'ok': ok,
        **extra,
    }


def _write_state(path: Path, content: object) -> Path:
    path.write_text(content if isinstance(content, str) else json.dumps(content), encoding='utf-8')
    return path


def test_the_default_sweep_stamp_is_gitignored():
    """The sweep writes its stamp into the checkout it measures, which must stay clean."""
    checked = run_git(['check-ignore', exception_register.SWEEP_STATE_PATH], cwd=REPO_ROOT)

    assert checked.stdout.strip() == exception_register.SWEEP_STATE_PATH


def test_an_absent_stamp_means_the_sweep_has_never_run(tmp_path: Path):
    reading = exception_register.read_sweep(tmp_path / 'state.json', now=_NOW)

    assert reading.status is exception_register.SweepStatus.NEVER_RUN
    assert reading.state_path == tmp_path / 'state.json'


def test_a_recent_stamp_is_fresh_and_carries_its_age_and_result(tmp_path: Path):
    stamp = _stamp(_NOW - datetime.timedelta(hours=1))
    path = _write_state(tmp_path / 'state.json', stamp)

    reading = exception_register.read_sweep(path, now=_NOW)

    assert reading.status is exception_register.SweepStatus.FRESH
    assert reading.age == datetime.timedelta(hours=1)
    assert reading.ok is True
    assert reading.last_run == _NOW - datetime.timedelta(hours=1)
    assert reading.result == stamp


@pytest.mark.parametrize(
    ('age', 'status'),
    [
        (2 * datetime.timedelta(days=1), 'FRESH'),
        (2 * datetime.timedelta(days=1) + datetime.timedelta(seconds=1), 'STALE'),
    ],
)
def test_a_stamp_goes_stale_after_two_timer_periods(
    tmp_path: Path, age: datetime.timedelta, status: str
):
    path = _write_state(tmp_path / 'state.json', _stamp(_NOW - age))

    reading = exception_register.read_sweep(path, now=_NOW)

    assert reading.status is exception_register.SweepStatus[status]


def test_a_failed_recent_run_is_fresh_and_says_it_failed(tmp_path: Path):
    """The sweep leaves last_run unadvanced on failure, so age is the liveness signal."""
    path = _write_state(tmp_path / 'state.json', _stamp(_NOW - datetime.timedelta(hours=2), ok=False))

    reading = exception_register.read_sweep(path, now=_NOW)

    assert reading.status is exception_register.SweepStatus.FRESH
    assert reading.ok is False


@pytest.mark.parametrize(
    ('content', 'field'),
    [
        pytest.param('{not json', 'JSON', id='not-json'),
        pytest.param([1, 2], 'object', id='not-an-object'),
        pytest.param({'ok': True}, 'last_run', id='missing-last-run'),
        pytest.param(_stamp('2026-10-09T11:00:00'), 'last_run', id='naive-timestamp'),
        pytest.param(_stamp('yesterday'), 'last_run', id='unparseable-last-run'),
        pytest.param(_stamp(_NOW, ok='yes'), 'ok', id='non-bool-ok'),
    ],
)
def test_a_stamp_that_cannot_be_read_is_unreadable_naming_the_field(
    tmp_path: Path, content: object, field: str
):
    path = _write_state(tmp_path / 'state.json', content)

    reading = exception_register.read_sweep(path, now=_NOW)

    assert reading.status is exception_register.SweepStatus.UNREADABLE
    assert reading.detail is not None
    assert field in reading.detail


def _main_reading(root: Path, state: Path, *args: str) -> int:
    return exception_register.main([*args, '--root', str(root), '--sweep-state', str(state)])


def _sweep_tree(root: Path) -> None:
    _register_tree(root, {'m.py': 'x = 1\n'})


def test_a_fresh_stamp_reports_its_age_and_keeps_the_report_green(tmp_path: Path, capsys):
    _sweep_tree(tmp_path)
    recent = datetime.datetime.now(datetime.UTC) - datetime.timedelta(hours=1)
    state = _write_state(tmp_path / 'state.json', _stamp(recent))

    code = _main_reading(tmp_path, state, '--report')

    captured = capsys.readouterr()
    assert code == 0, captured.err
    (line,) = [line for line in captured.out.splitlines() if line.startswith('last sweep:')]
    assert 'h ago' in line


def test_a_stale_stamp_is_exit_two_after_the_whole_report(tmp_path: Path, capsys):
    _sweep_tree(tmp_path)
    state = _write_state(tmp_path / 'state.json', _stamp('2020-01-01T00:00:00+00:00'))

    code = _main_reading(tmp_path, state, '--report')

    captured = capsys.readouterr()
    assert code == 2
    assert captured.out.splitlines()[0] == 'grandfathered (unowned): not seeded'
    assert str(state) in captured.err
    assert 'two timer periods (48 h)' in captured.err


def test_an_unreadable_stamp_is_exit_two(tmp_path: Path, capsys):
    _sweep_tree(tmp_path)
    state = _write_state(tmp_path / 'state.json', '{not json')

    assert _main_reading(tmp_path, state, '--report') == 2
    assert str(state) in capsys.readouterr().err


def test_an_absent_stamp_reads_never_run_and_stays_green(tmp_path: Path, capsys):
    _sweep_tree(tmp_path)
    state = tmp_path / 'state.json'

    code = _main_reading(tmp_path, state, '--report')

    captured = capsys.readouterr()
    assert code == 0, captured.err
    (line,) = [line for line in captured.out.splitlines() if line.startswith('last sweep:')]
    assert 'never run' in line
    assert str(state) in line


def test_json_publishes_a_stale_sweep_as_data_and_exits_zero(tmp_path: Path, capsys):
    _sweep_tree(tmp_path)
    stamp = _stamp('2020-01-01T00:00:00+00:00')
    state = _write_state(tmp_path / 'state.json', stamp)

    code = _main_reading(tmp_path, state, '--json')

    captured = capsys.readouterr()
    assert code == 0, captured.err
    sweep = json.loads(captured.out)['sweep']
    assert sweep['status'] == 'stale'
    assert sweep['state_path'] == str(state)
    assert sweep['last_run'] == '2020-01-01T00:00:00+00:00'
    assert sweep['ok'] is True
    assert sweep['result'] == stamp


def test_json_names_the_default_state_path_under_the_root(tmp_path: Path, capsys):
    """The sweep takes its write path from here, so the path has one home."""
    _sweep_tree(tmp_path)
    capsys.readouterr()

    assert exception_register.main(['--json', '--root', str(tmp_path)]) == 0

    sweep = json.loads(capsys.readouterr().out)['sweep']
    assert sweep['status'] == 'never_run'
    assert sweep['state_path'] == str(tmp_path / exception_register.SWEEP_STATE_PATH)


# ---------------------------------------------------------------------------
# The live tree — the merge-time guard that the real register is clean.
#
# Invariants only, never populations: declarations, candidates, the seed and
# NOT_EXCEPTION_LISTS all change legitimately as the batch's later tasks land.


def test_the_live_register_is_clean(tmp_path: Path, capsys):
    """Every Policy in every declaration, inline marker and class-table row
    names a row of the real ratification table, and nothing is malformed."""
    code = exception_register.main(
        ['--report', '--root', str(REPO_ROOT), '--sweep-state', str(tmp_path / 'absent.json')]
    )

    assert code == 0, capsys.readouterr().err


def test_the_live_register_document_reads_the_real_tree(tmp_path: Path, capsys):
    capsys.readouterr()
    code = exception_register.main(
        ['--json', '--root', str(REPO_ROOT), '--sweep-state', str(tmp_path / 'absent.json')]
    )
    captured = capsys.readouterr()
    assert code == 0, captured.err
    document = json.loads(captured.out)

    assert document['inline']['sites'] >= 1000
    assert document['discovery']['files_parsed'] >= 1
    seeded = (REPO_ROOT / 'scripts' / 'inline_suppression_baseline.json').exists()
    assert document['grandfathered']['seeded'] is seeded
    assert (document['grandfathered']['unowned'] is None) is (not seeded)
    rows = exception_register.load_ratifications(REPO_ROOT / exception_register.TABLE_PATH)
    assert document['ratification_table']['rows'] == [
        {'id': row.id, 'date': row.date.isoformat(), 'source': row.source, 'covers': row.covers}
        for row in sorted(rows, key=lambda row: row.id)
    ]
