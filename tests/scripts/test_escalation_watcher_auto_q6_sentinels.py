"""Mirror guard: the L1 skill's Q6 detector-rule table names exactly the code's sentinels.

Task 8428 (``plans/decision-stream-prd.md`` ζ). ``skills/escalation-watcher-auto/SKILL.md``
holds one rule per harness detector whose L1 records are monitor signals,
keyed by the sentinel ``task_id`` the detector files under. A rule keyed by a
sentinel the code no longer emits never fires, and a detector with no row falls
through to the generic promote — both silently. This is an INV-10 tier-2 mirror
(``docs/legibility/design-invariants.md``): the marker-delimited table's first
column is checked against the two constants it restates.

STRUCTURE ONLY, NEVER PROSE. Only the first cell of each data row is read; the
rule wording in the other columns stays free to change. Whether the rules work
is measured in production by task θ2, not here.

MUST NOT SKIP. No ``pytest.importorskip``: a missing SKILL.md or an
unimportable constant fails.
"""
from __future__ import annotations

import pathlib

import pytest
import verify_command_invariants as vci
from orchestrator.recovery_emission import RECOVERY_VETO_STREAK_SENTINEL_PREFIX
from orchestrator.stale_l0_strand_identity import STALE_L0_STRAND_SENTINEL
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

REPO_ROOT = pathlib.Path(__file__).parents[2]
SKILL = REPO_ROOT / 'skills' / 'escalation-watcher-auto' / 'SKILL.md'

_SLUG = 'q6-detector-sentinels'
_CODE_SENTINELS = (
    RECOVERY_VETO_STREAK_SENTINEL_PREFIX,
    STALE_L0_STRAND_SENTINEL,
)


def _is_separator_cell(cell: str) -> bool:
    stripped = cell.strip()
    return bool(stripped) and set(stripped) <= {'-', ':'}


def _span_sentinels(markdown: str, *, source: str) -> list[str]:
    body = vci.marked_body(
        markdown,
        slug=_SLUG,
        source=source,
        label='the Q6 detector-rule table',
        task='8428',
    )
    table = [line.strip() for line in body.splitlines() if line.strip().startswith('|')]
    assert len(table) >= 3, (
        f'the {_SLUG} span in {source} must hold a header row, a separator row '
        f'and at least one sentinel row; found {len(table)} table line(s) (task 8428)'
    )
    separator_cells = table[1].strip('|').split('|')
    assert all(_is_separator_cell(cell) for cell in separator_cells), (
        f'the second table line of the {_SLUG} span in {source} must be the '
        f'markdown separator row, got {table[1]!r} (task 8428)'
    )
    sentinels = []
    for row in table[2:]:
        first = row.strip('|').split('|')[0].strip()
        assert (
            len(first) > 2
            and first.startswith('`')
            and first.endswith('`')
            and '`' not in first[1:-1]
        ), (
            f'the first column of the {_SLUG} table in {source} must be a single '
            f'backticked sentinel task_id, got {first!r} in row {row!r} (task 8428)'
        )
        sentinels.append(first[1:-1])
    return sentinels


def test_q6_rule_span_names_exactly_the_detector_sentinel_constants() -> None:
    assert SKILL.is_file(), f'{SKILL} is missing; this guard must fail, not skip'
    documented = _span_sentinels(
        SKILL.read_text(encoding='utf-8'),
        source='skills/escalation-watcher-auto/SKILL.md',
    )
    assert len(documented) == len(set(documented)), (
        f'the {_SLUG} table lists a sentinel twice: {documented!r}'
    )
    assert sorted(documented) == sorted(_CODE_SENTINELS), (
        f'the {_SLUG} table names {sorted(documented)!r} but the detectors file '
        f'under {sorted(_CODE_SENTINELS)!r} '
        '(recovery_emission.py::RECOVERY_VETO_STREAK_SENTINEL_PREFIX, '
        'stale_l0_strand_identity.py::STALE_L0_STRAND_SENTINEL). A renamed constant must '
        'be mirrored in the table, and a table row needs a constant behind it.'
    )


_SYNTHETIC_SPAN = (
    '<!-- q6-detector-sentinels:begin note -->\n'
    '| sentinel | promote only when | otherwise |\n'
    '|---|---|---|\n'
    '| `__a__` | rule text one | rest one |\n'
    '| `__b__` | rule text two | rest two |\n'
    '<!-- q6-detector-sentinels:end -->\n'
)


def test_renaming_a_sentinel_cell_changes_the_extraction_but_rewording_the_rule_cells_does_not() -> None:
    assert _span_sentinels(_SYNTHETIC_SPAN, source='synthetic') == ['__a__', '__b__']

    reworded = (
        _SYNTHETIC_SPAN.replace('rule text one', 'a different rule')
        .replace('rest one', 'another fallback')
        .replace('rule text two', 'reworded')
        .replace('rest two', 'also reworded')
    )
    assert _span_sentinels(reworded, source='synthetic') == ['__a__', '__b__']

    renamed = _SYNTHETIC_SPAN.replace('`__a__`', '`__a2__`')
    assert _span_sentinels(renamed, source='synthetic') == ['__a2__', '__b__']


def test_a_span_with_no_sentinel_rows_fails_loudly() -> None:
    header_only = (
        '<!-- q6-detector-sentinels:begin note -->\n'
        '| sentinel | promote only when | otherwise |\n'
        '|---|---|---|\n'
        '<!-- q6-detector-sentinels:end -->\n'
    )
    with pytest.raises(AssertionError, match='at least one sentinel row'):
        _span_sentinels(header_only, source='synthetic')
