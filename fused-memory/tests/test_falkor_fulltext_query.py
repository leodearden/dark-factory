"""Unit tests for the hardened FalkorDB/RediSearch fulltext query assembly (task 3334).

Background — dead-letter 9950 (and an identical 2026-04-02 occurrence) failed the
Graphiti durable-queue write path with::

    RediSearch: Syntax error at offset 122 near note

That reads like a reserved-word collision, but it is not.  ``note`` parses fine
standalone and in every position.  The RediSearch parser names the token *before*
the fault, and the actual fault is a token the tokenizer reduces to **nothing**
(a lone ``_``, a bare/double backtick) surviving into the ``|``-joined operand
list, which leaves the union operator with no right-hand operand.

Every "searchable / not searchable" claim below was measured against a live
FalkorDB (module v41800 / 4.18.0) behind an index-readiness barrier; see
``tests/test_falkor_fulltext_integration.py`` for the live counterpart.
"""

from __future__ import annotations

import pytest

from fused_memory.backends.falkor_fulltext import is_searchable_term

# --- Genuine FAULT tokens -------------------------------------------------
# RediSearch's tokenizer reduces each of these to nothing, so emitting one into
# the ``|``-joined operand list leaves the union operator dangling and the query
# fails to parse.  Each was verified to produce a syntax error on a live index
# (5/5 runs, index-readiness barrier applied).
FAULT_TOKENS: list[str] = ['_', '__', '`', '``', '']

# --- SAFE OVER-APPROXIMATION ---------------------------------------------
# These are dropped because they are never meaningful search terms, NOT because
# they break the parser.  MEASURED: a bare em-dash / bullet / arrow as a
# standalone term PARSES FINE against live FalkorDB (OK, 5/5).  Do not "correct"
# this comment into a claim that they are fault tokens — that would be false.
OVER_APPROXIMATED_TOKENS: list[str] = ['—', '•', '→']

# --- Must survive verbatim ------------------------------------------------
# Every one of these was executed against a live FalkorDB fulltext index behind
# the index-readiness barrier and parsed cleanly (OK, 5/5).
SEARCHABLE_TOKENS: list[str] = [
    '_x',
    'x_',
    '_1',
    '1_',
    'A_B_C',
    'a`',
    '`a',
    '```a```',
    'note',
    '3334',
    'café',
    '日本語1',
    'x—y',
    'x•y',
    '__a__',
    'x±1',
    'a​b',  # zero-width space between two alnum characters
]


class TestIsSearchableTerm:
    """Pin the single safety predicate: a term must carry ≥1 alphanumeric char."""

    @pytest.mark.parametrize('token', FAULT_TOKENS)
    def test_fault_tokens_are_not_searchable(self, token: str) -> None:
        """Tokens RediSearch tokenizes to nothing must never reach the query.

        These are the genuine parser faults — each one, emitted inside a
        ``|``-joined operand list, produces ``Syntax error at offset N near
        <preceding word>`` on a live FalkorDB index.
        """
        assert is_searchable_term(token) is False

    @pytest.mark.parametrize('token', OVER_APPROXIMATED_TOKENS)
    def test_over_approximated_tokens_are_not_searchable(self, token: str) -> None:
        """Dropped as a deliberate safe over-approximation, not as fault tokens.

        MEASURED against live FalkorDB: a bare ``—`` / ``•`` / ``→`` standalone
        term parses fine (OK, 5/5).  They are dropped only because they are never
        meaningful search terms; the alnum invariant is a closed-direction
        over-approximation that happens to include them.
        """
        assert is_searchable_term(token) is False

    @pytest.mark.parametrize('token', SEARCHABLE_TOKENS)
    def test_alnum_bearing_tokens_are_searchable(self, token: str) -> None:
        """Any token carrying at least one alphanumeric character survives verbatim."""
        assert is_searchable_term(token) is True

    def test_predicate_is_unicode_aware_not_ascii_only(self) -> None:
        """``café`` / ``日本語1`` pin that the predicate uses ``str.isalnum()``.

        An ASCII regex such as ``[A-Za-z0-9]`` would silently drop non-Latin
        content, degrading recall for exactly the multilingual episodes we store.
        """
        assert is_searchable_term('日本語') is True
        assert is_searchable_term('naïve') is True
        assert is_searchable_term('Ω') is True
