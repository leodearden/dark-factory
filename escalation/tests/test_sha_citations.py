"""cited_shas and resolve_cited_shas: the record scanner and entry builder
behind resolve_escalation_shas, against a stub resolver (no git, no server).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import pytest

from escalation.models import Escalation, EvidenceEntry
from escalation.sha_citations import cited_shas, resolve_cited_shas

SHA_A = 'a7ceed6597'
SHA_B = '570dc88ad6'
FULL = '0123456789abcdef0123456789abcdef01234567'


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
    def __init__(self, answers: Mapping[str, Mapping[str, object]]) -> None:
        self._answers = answers
        self.calls: list[str] = []

    async def resolve_rebased_citation(self, sha: str) -> Mapping[str, object] | None:
        self.calls.append(sha)
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

        assert await resolve_cited_shas(esc, resolver) == [
            {'cited': SHA_A, 'landed': 'a' * 40, 'subject': 'feat: a', 'method': 'patch-id'},
            {'cited': SHA_B, 'landed': 'b' * 40, 'subject': 'feat: b', 'method': 'subject'},
        ]

    @pytest.mark.asyncio
    async def test_resolves_each_distinct_token_once(self):
        resolver = _StubResolver({})
        esc = _esc(summary=SHA_A, detail=f'{SHA_A} {SHA_B}', evidence=[{'observation': SHA_B}])

        assert await resolve_cited_shas(esc, resolver) == []
        assert resolver.calls == [SHA_A, SHA_B]
