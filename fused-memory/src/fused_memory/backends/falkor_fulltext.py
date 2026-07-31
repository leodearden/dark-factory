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
