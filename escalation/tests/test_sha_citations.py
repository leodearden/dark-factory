"""cited_shas and resolve_cited_shas: the record scanner and entry builder
behind resolve_escalation_shas, against a stub resolver (no git, no server).
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import Any, cast

import pytest

from escalation.models import Escalation, EvidenceEntry
from escalation.sha_citations import MAX_CITED_SHAS_RESOLVED, cited_shas, resolve_cited_shas

SHA_A = 'a7ceed6597'
SHA_B = '570dc88ad6'
FULL = '0123456789abcdef0123456789abcdef01234567'
TWIN_B = {'landed': 'b' * 40, 'subject': 'feat: b', 'method': 'patch-id'}

# Each contributes two SHA-shaped tokens (its 8- and 12-hex segments).
UUIDS = (
    '3f2a9c1e-4b7d-4e21-9a0c-5c8d0b7f1a33',
    '7b1d2e3f-0a9c-4d8e-b7f6-1e2d3c4b5a69',
    'c0ffee12-3456-4789-abcd-ef0123456789',
    'deadbeef-cafe-4bab-8e5e-0123456789ab',
    '1a2b3c4d-5e6f-4a7b-8c9d-0e1f2a3b4c5d',
    '9f8e7d6c-5b4a-4392-8170-6f5e4d3c2b1a',
)
NOISE = f"{' '.join(UUIDS)} pid 4194304 on 20261023 hash 9c1e7b4d5a609e21"


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


class _StubResolver:
    """*commits* None means every token names a commit."""

    def __init__(
        self,
        answers: Mapping[str, Mapping[str, object]],
        *,
        commits: Collection[str] | None = None,
        faults: Mapping[str, Exception] | None = None,
        classify_fault: Exception | None = None,
    ) -> None:
        self._answers = answers
        self._commits = commits
        self._faults = faults or {}
        self._classify_fault = classify_fault
        self.calls: list[str] = []
        self.classified: list[str] = []

    async def commits_named(self, tokens: Sequence[str]) -> Mapping[str, str]:
        self.classified = list(tokens)
        if self._classify_fault is not None:
            raise self._classify_fault
        return {t: t for t in tokens if self._commits is None or t in self._commits}

    async def resolve_rebased_citation(self, sha: str) -> Mapping[str, object] | None:
        self.calls.append(sha)
        if sha in self._faults:
            raise self._faults[sha]
        return self._answers.get(sha)


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
        resolver = _StubResolver({
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
        resolver = _StubResolver({})
        esc = _esc(summary=SHA_A, detail=f'{SHA_A} {SHA_B}', evidence=[{'observation': SHA_B}])

        assert (await resolve_cited_shas(esc, resolver)).resolved == ()
        assert resolver.calls == [SHA_A, SHA_B]

    @pytest.mark.asyncio
    async def test_looks_up_only_the_first_max_tokens_and_returns_the_rest_skipped(self):
        tokens = [f'{n:07x}' for n in range(MAX_CITED_SHAS_RESOLVED + 2)]
        resolver = _StubResolver({})

        result = await resolve_cited_shas(_esc(detail=' '.join(tokens)), resolver)

        assert resolver.calls == tokens[:MAX_CITED_SHAS_RESOLVED]
        assert result.skipped == tuple(tokens[MAX_CITED_SHAS_RESOLVED:])

    @pytest.mark.asyncio
    async def test_exactly_max_tokens_skips_none(self):
        tokens = [f'{n:07x}' for n in range(MAX_CITED_SHAS_RESOLVED)]
        resolver = _StubResolver({})

        result = await resolve_cited_shas(_esc(detail=' '.join(tokens)), resolver)

        assert resolver.calls == tokens
        assert result.skipped == ()


@pytest.mark.asyncio
class TestCapCountsOnlyCommits:
    async def test_tokens_naming_no_commit_do_not_count_against_the_cap(self):
        resolver = _StubResolver({}, commits={SHA_A, SHA_B})

        result = await resolve_cited_shas(_esc(detail=f'{NOISE} {SHA_A} {SHA_B}'), resolver)

        assert resolver.calls == [SHA_A, SHA_B]
        assert result.skipped == ()

    async def test_only_commit_tokens_past_the_cap_are_skipped(self):
        commits = [f'{n:07x}' for n in range(MAX_CITED_SHAS_RESOLVED + 2)]
        detail = ' '.join(f'{UUIDS[i % len(UUIDS)]} {c}' for i, c in enumerate(commits))
        resolver = _StubResolver({}, commits=set(commits))

        result = await resolve_cited_shas(_esc(detail=detail), resolver)

        assert resolver.calls == commits[:MAX_CITED_SHAS_RESOLVED]
        assert result.skipped == tuple(commits[MAX_CITED_SHAS_RESOLVED:])


@pytest.mark.asyncio
class TestFaultsAreFailedNotDropped:
    async def test_a_lookup_git_could_not_answer_is_failed_not_dropped(self):
        resolver = _StubResolver(
            {SHA_B: TWIN_B},
            faults={SHA_A: RuntimeError('fatal: Unable to create index.lock')},
        )

        result = await resolve_cited_shas(_esc(detail=f'{SHA_A} {SHA_B}'), resolver)

        assert result.resolved == ({'cited': SHA_B, **TWIN_B},)
        assert [f['cited'] for f in result.failed] == [SHA_A]
        assert 'index.lock' in result.failed[0]['error']
        assert 'RuntimeError' in result.failed[0]['error']

    async def test_when_git_cannot_classify_tokens_every_token_is_failed(self):
        resolver = _StubResolver(
            {}, classify_fault=RuntimeError('fatal: not a git repository'),
        )

        result = await resolve_cited_shas(_esc(detail=f'{SHA_A} {SHA_B}'), resolver)

        assert [f['cited'] for f in result.failed] == [SHA_A, SHA_B]
        assert result.resolved == ()
        assert result.skipped == ()
        assert resolver.calls == []
