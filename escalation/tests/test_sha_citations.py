"""cited_shas and resolve_cited_shas: the record scanner and entry builder
behind resolve_escalation_shas, against a stub resolver (no git, no server).
"""

from __future__ import annotations

import logging
from typing import Any, cast

import pytest
from _citation_resolver_stub import StubCitationResolver
from _sha_noise import NOISE, UUIDS

from escalation.models import Escalation, EvidenceEntry
from escalation.sha_citations import MAX_CITED_SHAS_RESOLVED, cited_shas, resolve_cited_shas

SHA_A = 'a7ceed6597'
SHA_B = '570dc88ad6'
FULL = '0123456789abcdef0123456789abcdef01234567'
TWIN_B = {'landed': 'b' * 40, 'subject': 'feat: b', 'method': 'patch-id'}


def _esc(
    summary: str = '', detail: str = '', evidence: list[dict[str, Any]] | None = None,
) -> Escalation:
    return Escalation(
        id='esc-5751-1',
        task_id='5751',
        agent_role='implementer',
        severity='info',
        category='design_concern',
        summary=summary,
        detail=detail,
        evidence=cast(list[EvidenceEntry], evidence or []),
    )


class TestCitedShas:
    def test_scans_summary_then_detail_then_evidence_observations(self):
        esc = _esc(
            summary=f'landed {SHA_A}',
            detail=f'see {FULL}',
            evidence=[{'observation': 'main at 1234567', 'measured_at': 'x', 'ref': 'y'}],
        )

        assert cited_shas(esc) == [SHA_A, FULL, '1234567']

    def test_an_arrow_between_two_shas_yields_both(self):
        assert cited_shas(_esc(detail=f'{SHA_A}→{SHA_B}')) == [SHA_A, SHA_B]

    def test_ignores_too_short_too_long_and_non_hex_runs(self):
        esc = _esc(detail=f'abcdef {FULL}0 deadbeefz cafebabe-ish zzzzzzz')

        assert cited_shas(esc) == ['cafebabe']

    def test_tolerates_evidence_without_a_str_observation(self):
        esc = _esc(evidence=[
            {'measured_at': SHA_A},
            {'observation': 1234567},
            {'observation': f'rerun at {SHA_B}'},
        ])

        assert cited_shas(esc) == [SHA_B]

    def test_a_token_repeated_across_fields_appears_once(self):
        esc = _esc(
            summary=SHA_A,
            detail=f'{SHA_B} then {SHA_A}',
            evidence=[{'observation': SHA_B}],
        )

        assert cited_shas(esc) == [SHA_A, SHA_B]


class TestResolveCitedShas:
    @pytest.mark.asyncio
    async def test_builds_entries_in_cited_order_and_drops_misses(self):
        resolver = StubCitationResolver({
            SHA_B: {'landed': 'b' * 40, 'subject': 'feat: b', 'method': 'subject'},
            SHA_A: {'landed': 'a' * 40, 'subject': 'feat: a', 'method': 'patch-id'},
        })
        esc = _esc(detail=f'{SHA_A} {FULL} {SHA_B}')

        result = await resolve_cited_shas(esc, resolver)

        assert result.resolved == (
            {'cited': SHA_A, 'landed': 'a' * 40, 'subject': 'feat: a', 'method': 'patch-id'},
            {'cited': SHA_B, 'landed': 'b' * 40, 'subject': 'feat: b', 'method': 'subject'},
        )
        assert result.skipped == ()

    @pytest.mark.asyncio
    async def test_resolves_each_distinct_token_once(self):
        resolver = StubCitationResolver()
        esc = _esc(summary=SHA_A, detail=f'{SHA_A} {SHA_B}', evidence=[{'observation': SHA_B}])

        assert (await resolve_cited_shas(esc, resolver)).resolved == ()
        assert resolver.lookups == [SHA_A, SHA_B]

    @pytest.mark.asyncio
    async def test_looks_up_only_the_first_max_tokens_and_returns_the_rest_skipped(self):
        tokens = [f'{n:07x}' for n in range(MAX_CITED_SHAS_RESOLVED + 2)]
        resolver = StubCitationResolver()

        result = await resolve_cited_shas(_esc(detail=' '.join(tokens)), resolver)

        assert resolver.lookups == tokens[:MAX_CITED_SHAS_RESOLVED]
        assert result.skipped == tuple(tokens[MAX_CITED_SHAS_RESOLVED:])

    @pytest.mark.asyncio
    async def test_exactly_max_tokens_skips_none(self):
        tokens = [f'{n:07x}' for n in range(MAX_CITED_SHAS_RESOLVED)]
        resolver = StubCitationResolver()

        result = await resolve_cited_shas(_esc(detail=' '.join(tokens)), resolver)

        assert resolver.lookups == tokens
        assert result.skipped == ()


@pytest.mark.asyncio
class TestCapCountsOnlyCommits:
    async def test_tokens_naming_no_commit_do_not_count_against_the_cap(self):
        resolver = StubCitationResolver(commits={SHA_A, SHA_B})
        esc = _esc(detail=f'{NOISE} {SHA_A} {SHA_B}')

        result = await resolve_cited_shas(esc, resolver)

        assert len(cited_shas(esc)) > MAX_CITED_SHAS_RESOLVED
        assert resolver.classifications == [cited_shas(esc)]
        assert resolver.lookups == [SHA_A, SHA_B]
        assert result.skipped == ()

    async def test_only_commit_tokens_past_the_cap_are_skipped(self):
        commits = [f'{n:07x}' for n in range(MAX_CITED_SHAS_RESOLVED + 2)]
        detail = ' '.join(f'{UUIDS[i % len(UUIDS)]} {c}' for i, c in enumerate(commits))
        resolver = StubCitationResolver(commits=set(commits))

        result = await resolve_cited_shas(_esc(detail=detail), resolver)

        assert resolver.lookups == commits[:MAX_CITED_SHAS_RESOLVED]
        assert result.skipped == tuple(commits[MAX_CITED_SHAS_RESOLVED:])


@pytest.mark.asyncio
class TestFaultsAreFailedNotDropped:
    async def test_a_lookup_git_could_not_answer_is_failed_not_dropped(self):
        resolver = StubCitationResolver(
            {SHA_B: TWIN_B}, faults={SHA_A: RuntimeError('fatal: bad object a7ceed6597')},
        )

        result = await resolve_cited_shas(_esc(detail=f'{SHA_A} {SHA_B}'), resolver)

        assert result.resolved == ({'cited': SHA_B, **TWIN_B},)
        assert result.failed == (
            {'cited': SHA_A, 'error': 'RuntimeError: fatal: bad object a7ceed6597'},
        )

    async def test_when_git_cannot_classify_tokens_every_token_is_failed(self):
        resolver = StubCitationResolver(classify_fault=RuntimeError('fatal: not a git repository'))

        result = await resolve_cited_shas(_esc(detail=f'{SHA_A} {SHA_B}'), resolver)

        assert [f['cited'] for f in result.failed] == [SHA_A, SHA_B]
        assert result.resolved == ()
        assert result.skipped == ()
        assert resolver.lookups == []

    async def test_when_git_cannot_classify_tokens_every_token_counts_against_the_cap(self):
        resolver = StubCitationResolver(classify_fault=RuntimeError('fatal: not a git repository'))
        esc = _esc(detail=f'{NOISE} {SHA_A}')
        tokens = cited_shas(esc)

        result = await resolve_cited_shas(esc, resolver)

        assert [f['cited'] for f in result.failed] == tokens[:MAX_CITED_SHAS_RESOLVED]
        assert result.skipped == tuple(tokens[MAX_CITED_SHAS_RESOLVED:])
        assert resolver.lookups == []


@pytest.mark.asyncio
class TestFaultsAreLogged:
    async def test_a_failed_lookup_is_logged_with_its_traceback(self, caplog):
        resolver = StubCitationResolver(faults={SHA_A: RuntimeError('fatal: bad object')})

        with caplog.at_level(logging.WARNING, logger='escalation.sha_citations'):
            await resolve_cited_shas(_esc(detail=SHA_A), resolver)

        (record,) = caplog.records
        assert record.levelno == logging.WARNING
        assert SHA_A in record.getMessage()
        assert 'esc-5751-1' in record.getMessage()
        assert record.exc_info is not None

    async def test_a_failed_classification_is_logged_once_with_its_traceback(self, caplog):
        resolver = StubCitationResolver(classify_fault=RuntimeError('fatal: not a git repository'))

        with caplog.at_level(logging.WARNING, logger='escalation.sha_citations'):
            await resolve_cited_shas(_esc(detail=f'{SHA_A} {SHA_B}'), resolver)

        (record,) = caplog.records
        assert record.levelno == logging.WARNING
        assert 'esc-5751-1' in record.getMessage()
        assert record.exc_info is not None
