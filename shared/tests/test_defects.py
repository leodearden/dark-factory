"""Tests for shared.defects, the defect-tests vocabulary (plans/defect-tests-prd.md).

Every test drives a public seam. The oracles (expected hashes, I2 destinations)
are written here from the PRD, never derived from the module's own tables.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta, timezone

import pytest

from shared.defects import (
    DefectOrigin,
    MalformedDefectRef,
    mint_explore_ref,
    mint_finding_ref,
    mint_followup_ref,
    mint_human_ref,
    mint_portal_ref,
    mint_redmain_ref,
    mint_review_ref,
    parse_ref,
)
from shared.finding_key import finding_key
from shared.task_metadata import RetryLedger


def _norm(text: str) -> str:
    return ' '.join(text.split()).lower()


def _issue_hash(location: str, description: str) -> str:
    basis = f'{_norm(location)}\x1f{_norm(description)}'
    return hashlib.sha256(basis.encode()).hexdigest()[:12]


def _hash8(summary: str) -> str:
    return hashlib.sha256(summary.encode()).hexdigest()[:8]


_SIGNATURE = RetryLedger.compute_merge_outcome_signature('cat', 'hint')
_FINDING_KEY = finding_key('area', 'x.py', 'tag')
_FILED_AT = datetime(2026, 10, 18, 12, 34, 56, 123456, tzinfo=UTC)
_HEX12 = 'abcdef012345'


class TestMint:
    def test_review_ref_is_task_and_issue_hash(self) -> None:
        expected = f'review:7043:{_issue_hash("foo.py::bar", "bad thing")}'
        assert mint_review_ref(7043, 'foo.py::bar', 'bad thing') == expected

    def test_review_ref_normalises_whitespace_and_case(self) -> None:
        assert mint_review_ref(7043, '  Foo.py::bar ', 'Bad   THING\n') == mint_review_ref(
            7043, 'foo.py::bar', 'bad thing'
        )

    def test_review_ref_keeps_location_and_description_apart(self) -> None:
        assert mint_review_ref(7043, 'a', 'bc') != mint_review_ref(7043, 'ab', 'c')

    def test_finding_ref_wraps_the_real_finding_key(self) -> None:
        assert mint_finding_ref(_FINDING_KEY) == f'finding:{_FINDING_KEY}'

    def test_redmain_ref_takes_the_real_main_health_signature(self) -> None:
        ref = mint_redmain_ref(_SIGNATURE)
        assert ref == f'redmain:{_SIGNATURE}'
        assert parse_ref(ref).component('signature') == _SIGNATURE

    def test_followup_ref(self) -> None:
        assert mint_followup_ref(7043, 'abc-1.2_x') == 'followup:7043:abc-1.2_x'

    def test_human_ref_is_basic_utc_seconds_and_hash8(self) -> None:
        expected = f'human:20261018T123456Z:{_hash8("Some summary")}'
        assert mint_human_ref(_FILED_AT, 'Some summary') == expected

    def test_human_ref_converts_an_aware_offset_to_utc(self) -> None:
        plus_two = datetime(2026, 10, 18, 14, 34, 56, tzinfo=timezone(timedelta(hours=2)))
        expected = f'human:20261018T123456Z:{_hash8("Some summary")}'
        assert mint_human_ref(plus_two, 'Some summary') == expected

    def test_human_ref_rejects_a_naive_datetime(self) -> None:
        with pytest.raises(MalformedDefectRef) as excinfo:
            mint_human_ref(datetime(2026, 10, 18, 12, 34, 56), 'Some summary')
        assert excinfo.value.origin is DefectOrigin.HUMAN

    def test_portal_ref(self) -> None:
        assert mint_portal_ref('acme', 'block-3', 2) == 'portal:acme:block-3:2'

    def test_explore_ref(self) -> None:
        assert mint_explore_ref('run-1', 4) == 'explore:run-1:4'


_MINT_REJECTIONS = [
    pytest.param(lambda: mint_review_ref(0, 'l', 'd'), DefectOrigin.REVIEW, id='review-task-0'),
    pytest.param(lambda: mint_review_ref(-1, 'l', 'd'), DefectOrigin.REVIEW, id='review-task--1'),
    pytest.param(
        lambda: mint_review_ref(True, 'l', 'd'), DefectOrigin.REVIEW, id='review-task-bool'
    ),
    pytest.param(
        lambda: mint_followup_ref(0, 'abc'), DefectOrigin.AGENT_FOLLOWUP, id='followup-task-0'
    ),
    pytest.param(
        lambda: mint_followup_ref(-1, 'abc'), DefectOrigin.AGENT_FOLLOWUP, id='followup-task--1'
    ),
    pytest.param(
        lambda: mint_followup_ref(True, 'abc'),
        DefectOrigin.AGENT_FOLLOWUP,
        id='followup-task-bool',
    ),
    pytest.param(
        lambda: mint_followup_ref(7043, 'a:b'),
        DefectOrigin.AGENT_FOLLOWUP,
        id='followup-token-colon',
    ),
    pytest.param(
        lambda: mint_followup_ref(7043, 'a b'),
        DefectOrigin.AGENT_FOLLOWUP,
        id='followup-token-space',
    ),
    pytest.param(lambda: mint_portal_ref('acme', 'b', 0), DefectOrigin.CLIENT, id='portal-n-0'),
    pytest.param(lambda: mint_portal_ref('acme', 'b', -1), DefectOrigin.CLIENT, id='portal-n--1'),
    pytest.param(
        lambda: mint_portal_ref('acme', 'b', True), DefectOrigin.CLIENT, id='portal-n-bool'
    ),
    pytest.param(
        lambda: mint_portal_ref('ac:me', 'b', 1), DefectOrigin.CLIENT, id='portal-token-colon'
    ),
    pytest.param(
        lambda: mint_portal_ref('acme', 'b 1', 1), DefectOrigin.CLIENT, id='portal-token-space'
    ),
    pytest.param(lambda: mint_explore_ref('r', 0), DefectOrigin.EXPLORATORY, id='explore-n-0'),
    pytest.param(lambda: mint_explore_ref('r', -1), DefectOrigin.EXPLORATORY, id='explore-n--1'),
    pytest.param(
        lambda: mint_explore_ref('r', True), DefectOrigin.EXPLORATORY, id='explore-n-bool'
    ),
    pytest.param(
        lambda: mint_explore_ref('r:1', 1), DefectOrigin.EXPLORATORY, id='explore-token-colon'
    ),
    pytest.param(
        lambda: mint_explore_ref('r 1', 1), DefectOrigin.EXPLORATORY, id='explore-token-space'
    ),
    pytest.param(
        lambda: mint_finding_ref('not-a-key'),
        DefectOrigin.REVIEW_INSTRUMENT,
        id='finding-not-fk',
    ),
    pytest.param(
        lambda: mint_redmain_ref(_SIGNATURE.upper()),
        DefectOrigin.RED_MAIN,
        id='redmain-uppercase',
    ),
    pytest.param(
        lambda: mint_redmain_ref(_SIGNATURE[:-1]), DefectOrigin.RED_MAIN, id='redmain-short'
    ),
    pytest.param(
        lambda: mint_redmain_ref(_SIGNATURE + '0'), DefectOrigin.RED_MAIN, id='redmain-long'
    ),
]


@pytest.mark.parametrize(('mint', 'origin'), _MINT_REJECTIONS)
def test_mint_rejection_names_the_origin(mint, origin: DefectOrigin) -> None:
    with pytest.raises(MalformedDefectRef) as excinfo:
        mint()
    assert excinfo.value.origin is origin
    assert origin.value in str(excinfo.value)


_ROUND_TRIP = {
    DefectOrigin.REVIEW: (
        lambda: mint_review_ref(7043, 'foo.py::bar', 'bad thing'),
        {'task': '7043', 'issue_hash': _issue_hash('foo.py::bar', 'bad thing')},
    ),
    DefectOrigin.REVIEW_INSTRUMENT: (lambda: mint_finding_ref(_FINDING_KEY), {'key': _FINDING_KEY}),
    DefectOrigin.RED_MAIN: (lambda: mint_redmain_ref(_SIGNATURE), {'signature': _SIGNATURE}),
    DefectOrigin.AGENT_FOLLOWUP: (
        lambda: mint_followup_ref(7043, 'abc-1.2_x'),
        {'spawned_from': '7043', 'suggestion_hash': 'abc-1.2_x'},
    ),
    DefectOrigin.HUMAN: (
        lambda: mint_human_ref(_FILED_AT, 'Some summary'),
        {'filed_at': '20261018T123456Z', 'hash8': _hash8('Some summary')},
    ),
    DefectOrigin.CLIENT: (
        lambda: mint_portal_ref('acme', 'block-3', 2),
        {'project': 'acme', 'block': 'block-3', 'n': '2'},
    ),
    DefectOrigin.EXPLORATORY: (
        lambda: mint_explore_ref('run-1', 4),
        {'run': 'run-1', 'n': '4'},
    ),
}


def test_round_trip_table_covers_every_origin() -> None:
    assert set(_ROUND_TRIP) == set(DefectOrigin)


@pytest.mark.parametrize('origin', list(_ROUND_TRIP), ids=str)
def test_minted_ref_parses_back_to_its_components(origin: DefectOrigin) -> None:
    mint, components = _ROUND_TRIP[origin]
    minted = mint()
    parsed = parse_ref(minted)
    assert parsed.origin is origin
    assert parsed.ref == minted
    assert dict(parsed.components) == components
    for name, value in components.items():
        assert parsed.component(name) == value


def test_component_accessor_names_the_valid_components() -> None:
    parsed = parse_ref(mint_explore_ref('run-1', 4))
    with pytest.raises(KeyError) as excinfo:
        parsed.component('task')
    assert 'run' in str(excinfo.value)


_PARSE_REJECTIONS = [
    pytest.param('', None, id='empty'),
    pytest.param('bogus:1', None, id='unknown-prefix'),
    pytest.param('review:7043', DefectOrigin.REVIEW, id='too-few-components'),
    pytest.param(f'review:7043:{_HEX12}:x', DefectOrigin.REVIEW, id='too-many-components'),
    pytest.param(f'review:07043:{_HEX12}', DefectOrigin.REVIEW, id='leading-zero'),
    pytest.param('review:7043:ABCDEF012345', DefectOrigin.REVIEW, id='uppercase-hex'),
    pytest.param('human:2026-10-18T12:34:56Z:abcd1234', DefectOrigin.HUMAN, id='colon-timestamp'),
    pytest.param('human:20261399T999999Z:abcd1234', DefectOrigin.HUMAN, id='impossible-time'),
    pytest.param(f'redmain:{_SIGNATURE}\n', DefectOrigin.RED_MAIN, id='trailing-newline'),
    pytest.param('portal:acme:b:0', DefectOrigin.CLIENT, id='portal-n-zero'),
    pytest.param(123, None, id='non-str'),
]


@pytest.mark.parametrize(('value', 'origin'), _PARSE_REJECTIONS)
def test_parse_rejection_names_value_and_origin(
    value: object, origin: DefectOrigin | None
) -> None:
    with pytest.raises(MalformedDefectRef) as excinfo:
        parse_ref(value)
    assert excinfo.value.value == value
    assert excinfo.value.origin is origin
    assert repr(value) in str(excinfo.value)
    if origin is not None:
        assert origin.value in str(excinfo.value)


def test_unknown_prefix_lists_the_known_prefixes() -> None:
    with pytest.raises(MalformedDefectRef) as excinfo:
        parse_ref('bogus:1')
    for prefix in ('review', 'finding', 'redmain', 'followup', 'human', 'portal', 'explore'):
        assert prefix in str(excinfo.value)
