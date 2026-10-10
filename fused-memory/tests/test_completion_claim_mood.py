"""Unit tests for the completion-claim gate's non-assertive-mood filter (task 6677).

A completion marker only makes a claim when the writer ASSERTS it. Quoted
text, modal forms, conditional/interrogative/imperative scopes, present-tense
temporal clauses and attributive participles mention a task's landing without
asserting it, and the reify escalations behind this task
(esc-unverified-claim-5471-*, -7407-3, -8246-5, -6020-2, -5023-2) were each
tagged for exactly that confusion.

Only the public :func:`extract_completion_claims` is driven here.
"""

from __future__ import annotations

import pytest

from fused_memory.services.completion_claim_gate import (
    CompletionClaim,
    extract_completion_claims,
)

_KNOWN = frozenset({'reify', 'dark_factory'})

# Verbatim from reify memory 4e6fb418. The quotation spans the '...' clause
# boundary, so the clause holding 'shipped' has no closing quote of its own.
_ESC_5471_4_MEMORY = (
    '- esc-unverified-claim-5471-3: "#5471 owns the shipped Bool-balance ... '
    'examples" while 5471 was pending.'
)


def _extract(text: str, default_project_id: str = 'reify') -> list[CompletionClaim]:
    return extract_completion_claims(
        text,
        default_project_id=default_project_id,
        known_project_ids=_KNOWN,
    )


def _triples(text: str) -> list[tuple[str, str, str]]:
    return [(claim.kind, claim.subject, claim.ref) for claim in _extract(text)]


class TestQuotationsAreMentions:

    @pytest.mark.parametrize(
        'text',
        [
            pytest.param('the memory said "task 5422 has landed" earlier', id='straight'),
            pytest.param('the memory said “task 5422 has landed” earlier', id='curly'),
            pytest.param(_ESC_5471_4_MEMORY, id='esc-unverified-claim-5471-4'),
            pytest.param(
                'which had asserted reify task 5638 "was re-filed into '
                "dark_factory's task tree as ticket tkt_0RRRC5AASJ9Z630VP4PCN9H376\"",
                id='quoted-filing',
            ),
        ],
    )
    def test_a_quoted_marker_is_not_a_claim(self, text):
        assert _triples(text) == []

    @pytest.mark.parametrize(
        'text',
        [
            'evidence="task 3016 merged 1a925afbf4"',
            'status:"task 3016 merged"',
        ],
    )
    def test_a_key_value_literal_is_still_read(self, text):
        assert _triples(text) == [('applied_work', 'task', '3016')]

    def test_an_unpaired_quote_does_not_cross_a_newline(self):
        assert _triples('note "x\ntask 5422 has landed') == [
            ('applied_work', 'task', '5422'),
        ]

    def test_backticks_do_not_delimit_a_quotation(self):
        assert _triples('task 5422 was left `cancelled`') == [
            ('disposition', 'task', '5422'),
        ]

    def test_an_apostrophe_does_not_open_a_quotation(self):
        assert _triples("task 5422's fix has been applied") == [
            ('applied_work', 'task', '5422'),
        ]
