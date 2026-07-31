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
from graphiti_core.driver.falkordb_driver import STOPWORDS, FalkorDriver

from fused_memory.backends.falkor_fulltext import (
    build_query,
    escape_group_id,
    is_searchable_term,
)

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


class TestEscapeGroupId:
    """Pin group-id escaping for the ``(@group_id:"...")`` filter.

    Upstream emits ``f'(@group_id:"{gid}")'`` with a comment claiming the quoting
    handles "special characters like hyphens".  MEASURED against live FalkorDB
    (5/5 both directions): that claim is false — ``(@group_id:"a-b") (alpha)``
    fails to parse while ``(@group_id:"a\\-b") (alpha)`` succeeds.  An unescaped
    ``"`` additionally lets a group_id break out of the quoted filter, which is a
    latent query-injection seam.
    """

    def test_ordinary_group_id_is_unchanged(self) -> None:
        """The overwhelmingly common case must stay byte-identical to upstream.

        Every real project_id reaching this code today is already
        ``[A-Za-z0-9_]``-only (``canonicalize_project_id`` folds ``-`` to ``_`` at
        every backend entry, CGL seam S4 / task 2269).  If escaping perturbed
        these, search behaviour would shift for every existing query — so the
        no-op case is asserted first and explicitly.
        """
        assert escape_group_id('dark_factory') == 'dark_factory'
        assert escape_group_id('main') == 'main'
        assert escape_group_id('proj123') == 'proj123'

    def test_hyphen_is_escaped(self) -> None:
        """``a-b`` → ``a\\-b``; the bare form is rejected by the live parser."""
        assert escape_group_id('a-b') == r'a\-b'

    def test_double_quote_is_escaped(self) -> None:
        """``q"uote`` → ``q\\"uote``; the bare form escapes the quoted filter."""
        assert escape_group_id('q"uote') == r'q\"uote'

    def test_backslash_is_escaped(self) -> None:
        """A literal backslash doubles."""
        assert escape_group_id('x\\y') == 'x\\\\y'

    def test_backslash_is_escaped_first(self) -> None:
        """Ordering is load-bearing, so it is asserted rather than assumed.

        Escaping ``\\`` *after* the ``-``/``"`` rules would double-escape the
        backslashes those rules just introduced (``a-b`` → ``a\\-b`` → ``a\\\\-b``),
        producing a literal-backslash match instead of an escaped hyphen.
        """
        assert escape_group_id('a-b') == 'a' + '\\' + '-b'
        # A backslash already adjacent to a hyphen is the discriminating case:
        # escape-last would emit four backslashes here instead of three.
        assert escape_group_id('a\\-b') == 'a' + '\\\\' + '\\-' + 'b'

    def test_all_three_escaped_exactly_once(self) -> None:
        """Combined input: each special character is escaped once, not twice."""
        assert escape_group_id('a-b"c\\d') == 'a\\-b\\"c\\\\d'


class TestBuildQueryUpstreamParity:
    """Pin byte-parity with upstream for inputs that ALREADY worked.

    The hardening in this module only ever *removes* operands that RediSearch
    cannot parse.  For every input that upstream already assembled successfully,
    ``build_query`` must emit the byte-identical string — otherwise the fix would
    silently change recall for the entire existing corpus, which is a far worse
    regression than the dead-letter it repairs.

    ``build_query`` takes text that has ALREADY been through
    ``FalkorDriver.sanitize``; it does not sanitize itself.  The driver override
    (steps 12/14) is what composes the two, so the character-class rules stay
    upstream's rather than being forked here.
    """

    def test_ordinary_query_is_byte_identical_to_upstream_shape(self) -> None:
        """The canonical shape: quoted group filter, space, pipe-joined operands.

        ``about`` is asserted to survive because it is NOT in upstream's
        ``STOPWORDS`` (verified against the real list below, not guessed).
        """
        assert (
            build_query('task 3334 note about RediSearch', ['dark_factory'], 128)
            == '(@group_id:"dark_factory") (task | 3334 | note | about | RediSearch)'
        )

    def test_stopword_removal_uses_upstream_list(self) -> None:
        """Stopwords come from upstream's ``STOPWORDS``, not a local re-declaration.

        The expectation is built FROM the imported list, so a graphiti-core
        upgrade that changes the stopword set cannot leave this test asserting a
        stale literal.
        """
        assert 'the' in STOPWORDS
        assert 'about' not in STOPWORDS

        text = 'the quick brown fox'
        expected_terms = [w for w in text.split() if w.lower() not in STOPWORDS]
        assert expected_terms == ['quick', 'brown', 'fox']
        assert build_query(text, ['g'], 128) == '(@group_id:"g") (quick | brown | fox)'

    def test_stopword_match_is_case_insensitive(self) -> None:
        """Upstream lowercases the token before the ``STOPWORDS`` lookup."""
        assert build_query('The quick', ['g'], 128) == '(@group_id:"g") (quick)'

    @pytest.mark.parametrize('group_ids', [None, []])
    def test_no_group_ids_preserves_upstream_leading_space(
        self, group_ids: list[str] | None
    ) -> None:
        """No filter → upstream still emits the leading space; keep it byte-exact.

        Upstream computes ``group_filter + ' (' + joined + ')'`` with an empty
        filter, so the result genuinely starts with a space.  Trimming it would
        be a cosmetic "improvement" that breaks parity for every ungrouped query.
        """
        assert build_query('alpha beta', group_ids, 128) == ' (alpha | beta)'

    def test_multiple_group_ids_are_pipe_joined_inside_one_paren(self) -> None:
        """Upstream shape for a multi-tenant filter: ``(@group_id:"a"|"b")``."""
        assert build_query('alpha', ['a', 'b'], 128) == '(@group_id:"a"|"b") (alpha)'

    def test_over_length_query_returns_empty_string(self) -> None:
        """200 distinct words at ``max_query_length=128`` → the no-query sentinel."""
        text = ' '.join(f'w{i}' for i in range(200))
        assert build_query(text, ['dark_factory'], 128) == ''

    def test_over_length_arithmetic_matches_upstream_exactly(self) -> None:
        """Pin upstream's ODD length arithmetic at its exact boundary.

        Upstream measures ``len(joined.split(' ')) + len(group_ids or '')`` where
        ``joined`` is ``' | '``-separated — so the pipes are COUNTED as fields
        (N terms → 2N-1), and ``len(group_ids or '')`` is the group_id COUNT.

        That looks like a bug and is tempting to "fix" to a plain word count.  It
        is deliberately preserved: changing it would alter which queries upstream
        drops, i.e. silently change recall.  The boundary below discriminates the
        two readings — with one group_id, 64 terms trips the guard (2*64-1+1 =
        128) while a naive word count (64+1 = 65) would not.
        """
        assert build_query(' '.join(f'w{i}' for i in range(64)), ['g'], 128) == ''
        assert build_query(' '.join(f'w{i}' for i in range(63)), ['g'], 128) != ''

    def test_group_id_count_participates_in_the_length_guard(self) -> None:
        """``len(group_ids or '')`` is the COUNT of group_ids, not a string length.

        Boundary measured against the stock upstream method, not derived: 63
        terms is 125 fields, so the guard (``>= 128``) trips at exactly three
        group_ids and not at one or two.  That the addend has to reach 3 to
        matter is precisely what proves it scales with the group_id COUNT — a
        string-length reading of ``len(group_ids or '')`` would have tripped far
        earlier, and a dropped addend would never trip here at all.
        """
        text = ' '.join(f'w{i}' for i in range(63))
        assert build_query(text, ['g'], 128) != ''
        assert build_query(text, ['g', 'h'], 128) != ''
        assert build_query(text, ['g', 'h', 'i'], 128) == ''

    def test_build_query_does_not_sanitize_its_input(self) -> None:
        """Input is pre-sanitized by the caller; ``build_query`` never re-does it.

        A comma would have been mapped to a space by ``FalkorDriver.sanitize``.
        Seeing ``alpha,beta`` survive as ONE token proves the character rules are
        not forked into this module — the driver override reuses upstream's
        ``sanitize()`` and only term assembly is replaced.
        """
        assert build_query('alpha,beta', ['g'], 128) == '(@group_id:"g") (alpha,beta)'

    @pytest.mark.parametrize(
        'text',
        [
            'task 3334 note about RediSearch',
            'the quick brown fox',
            'alpha beta gamma',
            'café naïve 日本語 note',
        ],
    )
    @pytest.mark.parametrize('group_ids', [None, [], ['dark_factory'], ['a', 'b']])
    def test_matches_stock_upstream_output_for_benign_inputs(
        self, text: str, group_ids: list[str] | None
    ) -> None:
        """Cross-check against the REAL upstream method, not a transcription of it.

        For inputs upstream already handled (no fault tokens, non-empty term
        list, no group_id needing escaping) the two must agree byte-for-byte.
        Calling the unbound upstream method on an uninitialised instance keeps
        this a pure-assembly comparison with no live connection.
        """
        stock = object.__new__(FalkorDriver)
        assert build_query(text, group_ids, 128) == stock.build_fulltext_query(
            text, group_ids, 128
        )
