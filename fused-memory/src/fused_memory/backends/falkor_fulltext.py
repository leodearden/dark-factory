"""RediSearch fulltext query assembly for the FalkorDB driver (task 3334).

Why this module exists
----------------------
Graphiti's ``FalkorDriver.build_fulltext_query`` (graphiti-core 0.28.2) builds a
RediSearch query by ``' | '``-joining the sanitized, stopword-filtered tokens of
the caller's text::

    (@group_id:"dark_factory") (task | 3334 | note | about | RediSearch)

``FalkorDriver.sanitize()`` maps ``,.<>{}[]"':;!@#$%^&*()-+=~?|/\\`` to spaces,
but it does **not** strip ``_`` or a backtick.  A token that RediSearch's own
tokenizer reduces to **nothing** therefore survives into the operand list and
leaves the ``|`` union operator with no right-hand operand.  RediSearch reports
that as::

    RediSearch: Syntax error at offset 122 near note

which is what dead-letter 9950 (and an identical 2026-04-02 occurrence) recorded.
The message is misleading in a specific and costly way: **this is not a
reserved-word collision.**  ``note`` parses fine standalone and in every
position.  The parser names the token *preceding* the fault, and ``note`` merely
happened to sit before the unparseable ``_``.  Two prior investigations chased
the reserved-word reading and no fix landed; the comment is here so the third
does not.

Real-world triggers for a bare ``_`` token are ordinary in stored content: the
Python throwaway variable (``for _ in ...``), markdown-escaped ``\\_`` (the
backslash is sanitized to a space, leaving a lone ``_``), ``**_**``, and ``_``
used as a placeholder or separator.  A bare or double backtick from an empty
code span does the same.

The invariant
-------------
**Every emitted term must contain at least one alphanumeric character.**  This is
a closed-direction over-approximation rather than a blocklist of RediSearch
reserved words or special characters: we cannot enumerate RediSearch's
tokenizer, so a blocklist would need re-patching for every new variant — exactly
the "recurs without a fix landing" pattern this module exists to break.

Validated against a live FalkorDB (module v41800 / 4.18.0) behind an
index-readiness barrier; see ``tests/test_falkor_fulltext_integration.py``.

This module deliberately imports nothing from ``graphiti_client`` so the query
rules stay unit-testable with a plain import and no client/LLM/embedder stack.
"""

from __future__ import annotations

import logging

from graphiti_core.driver.falkordb_driver import STOPWORDS

logger = logging.getLogger(__name__)


def is_searchable_term(token: str) -> bool:
    """Return True when ``token`` is safe to emit as a RediSearch operand.

    The predicate is "carries at least one alphanumeric character".  It is
    unicode-aware by construction (``str.isalnum()``), so ``café``, ``naïve`` and
    ``日本語1`` keep working while the genuine fault tokens — ``_``, ``__``,
    a bare backtick, a double backtick, and the empty string — are dropped.

    A handful of non-fault tokens (``—``, ``•``, ``→``) are also dropped.  That is
    a deliberate **safe over-approximation**, not a claim that they break the
    parser: measured against live FalkorDB, a bare em-dash / bullet / arrow as a
    standalone term parses fine.  They are simply never meaningful search terms,
    and accepting the over-approximation is what lets the predicate stay closed
    against token classes we failed to enumerate.
    """
    return any(ch.isalnum() for ch in token)


def escape_group_id(group_id: str) -> str:
    r"""Escape a group_id for use inside RediSearch's ``(@group_id:"...")`` filter.

    Upstream emits ``f'(@group_id:"{gid}")'`` with a comment asserting the
    quoting "prevents RediSearch syntax errors with reserved words like 'main'
    or special characters like hyphens".  **The hyphen half of that claim is
    wrong.**  Measured against a live FalkorDB (module v41800, 5/5 both
    directions):

    * ``(@group_id:"a-b") (alpha)``   → ``Syntax error``
    * ``(@group_id:"a\-b") (alpha)``  → parses
    * ``(@group_id:"q"uote") (alpha)``  → ``Syntax error`` (the bare ``"``
      terminates the quoted filter early — a latent query-injection seam, since
      the remainder of the group_id is then parsed as query syntax)
    * ``(@group_id:"q\"uote") (alpha)`` → parses

    Backslash is escaped **first** so a literal backslash in the input is not
    confused with an escape this function itself introduced by the ``-``/``"``
    rules.

    Ordinary group_ids are unaffected: ``canonicalize_project_id`` folds ``-`` to
    ``_`` at every backend entry (CGL seam S4 / task 2269), so no hyphenated or
    quote-bearing group_id can reach the driver today.  This is therefore
    defense-in-depth on the same code path, not a live bug fix — and
    ``'dark_factory'`` round-trips byte-identically, so search behaviour for
    every existing query is unchanged.

    KNOWN NOT HANDLED (deliberate): a degenerate group_id containing no
    alphanumeric character at all (e.g. literally ``_``) stays unparseable even
    once escaped.  No such project_id exists.  Silently dropping the group
    filter to rescue it would trade one loud, diagnosable failure for silently
    unscoped cross-project search results — the exact trade the project's
    loud-over-silent-degradation norm forbids.
    """
    return group_id.replace('\\', '\\\\').replace('"', '\\"').replace('-', '\\-')


def build_query(
    sanitized_text: str,
    group_ids: list[str] | None,
    max_query_length: int,
) -> str:
    """Assemble a RediSearch fulltext query from ALREADY-sanitized text.

    ``sanitized_text`` must have been through ``FalkorDriver.sanitize()``
    already: this function deliberately does not sanitize, so the character-class
    rules stay upstream's single source of truth and only term *assembly* is
    replaced here.

    Everything upstream does that already works is preserved byte-for-byte —
    the quoted group filter, the ``' | '`` join, the ``STOPWORDS`` list (imported,
    never re-declared), the leading space when there is no group filter, and the
    over-length arithmetic.

    That arithmetic — ``len(joined.split(' ')) + len(group_ids or '')`` — reads
    like a bug: the ``' | '`` separators are counted as fields (N terms → 2N-1)
    and ``len(group_ids or '')`` is the group_id COUNT.  It is reproduced exactly
    on purpose.  "Fixing" it would change which queries get dropped, i.e.
    silently change recall for every existing caller.
    """
    if not group_ids:
        group_filter = ''
    else:
        group_values = '|'.join(f'"{escape_group_id(gid)}"' for gid in group_ids)
        group_filter = f'(@group_id:{group_values})'

    # `is_searchable_term` is the repair, and it is NOT redundant with sanitize().
    # sanitize() leaves `_` and backticks alone, but RediSearch's own tokenizer
    # reduces a token like `_`, `__` or ``` `` ``` to NOTHING.  Emitted into the
    # ' | '-joined operand list, such a token leaves the union operator with no
    # right-hand operand, and the whole query fails to parse:
    #     RediSearch: Syntax error at offset N near <the PRECEDING word>
    # The message names the word BEFORE the fault, so it reads as a reserved-word
    # collision and has twice sent an investigation down that path (dead-letter
    # 9950, and an identical 2026-04-02 occurrence). Do not "simplify" this
    # predicate away because the tokens look harmless in the source text.
    terms = [
        w
        for w in sanitized_text.split()
        if w.lower() not in STOPWORDS and is_searchable_term(w)
    ]
    joined = ' | '.join(terms)

    # Upstream's over-length guard, arithmetic included. See docstring.
    if len(joined.split(' ')) + len(group_ids or '') >= max_query_length:
        return ''

    return group_filter + ' (' + joined + ')'
