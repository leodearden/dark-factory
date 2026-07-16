"""Attributed-claim extraction/verification for Stage 2 recon task-filing (task 2438).

Provides a pure, deterministic pre-check that extracts specific code-level
claims a Stage 2 recon candidate attributes to a completed task/commit/
ACTION (a metadata key, a stamp like ``foo=true``, or a named identifier)
so they can be verified against the live source tree + git history before
they are embedded in a filed task's description as fact.

Motivating incident: Stage 2 reconciliation filed task 2433 asserting, as
verified fact, that "task 2372 added an ACTION #5 stamp
(metadata.done_provenance_invalidated=true) on task reopen." No such
stamp/token has ever existed — a FABRICATION: a confidently-specific,
plausible-sounding but false claim attributed to another completed task's
implementation. This module extracts exactly the kind of claim that
incident embedded (``extract_attributed_claims``), verifies it against a
caller-supplied probe (``verify_attributed_claims``), and composes both
(``unverified_claims_in_text``). The one impure git-backed probe
(``make_source_and_history_probe``) lives here too, mirroring
recon_code_fix_premise_guard.py's split between pure detection and a
live-state re-check performed fresh on every call.

Usage::

    from fused_memory.middleware.recon_claim_verification_guard import (
        make_source_and_history_probe,
        unverified_claims_in_text,
    )

    probe = make_source_and_history_probe(repo_root)
    unverified = unverified_claims_in_text(candidate_text, probe)
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

__all__ = [
    "AttributedClaim",
    "extract_attributed_claims",
    "unverified_claims_in_text",
    "verify_attributed_claims",
]


@dataclass(frozen=True)
class AttributedClaim:
    """One specific code-level claim attributed to a completed task/commit/ACTION.

    ``token`` is the code literal to verify (e.g. a metadata key or stamp
    identifier, with any ``metadata.`` prefix stripped so it matches how the
    identifier would actually appear in source). ``attribution`` is the
    cited anchor text tying the claim to specific prior work (e.g.
    "per task 2372, ACTION #5"). ``span`` is a whitespace-collapsed snippet
    of the surrounding text, for use in advisory/log output.
    """

    token: str
    attribution: str
    span: str


# A specific code-level token: a `metadata.<snake_case>` reference, a
# backtick-quoted dotted/snake_case identifier, or an `<ident>=<value>`
# stamp form. Exactly one of the three named groups captures per match.
_CODE_TOKEN_PATTERN = re.compile(
    r"""
    metadata\.(?P<meta>[A-Za-z_][A-Za-z0-9_]*)
    |
    `(?P<tick>[A-Za-z_][A-Za-z0-9_.]*)`
    |
    \b(?P<stamp>[A-Za-z_][A-Za-z0-9_]*)=(?:true|false|[A-Za-z0-9_]+)\b
    """,
    re.VERBOSE,
)

# An attribution anchor: a citation of specific prior work. Requires a
# NUMBER (task/ACTION) or a plausible commit sha — a bare "task" or "ACTION"
# mention with no identifying number is never treated as an anchor, which is
# what keeps prospective new-work specs ("add a new task for X") from being
# mistaken for retrospective claims about completed work.
_ATTRIBUTION_PATTERN = re.compile(
    r"""
    \bper\s+task\s+\d+\b
    |
    \btask\s+\d+\b
    |
    \bACTION\s*\#\d+\b
    |
    \bcommit\s+[0-9a-f]{7,40}\b
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Character radius scanned on each side of a code-token match for a
# co-occurring attribution anchor. Deliberately small (roughly sentence-
# scale) — the anchor requirement is what separates a retrospective
# false-fact claim from a prospective new-work spec, so the window must not
# be so wide that it picks up an unrelated task/commit mention elsewhere in
# a long candidate description.
_CO_OCCURRENCE_WINDOW_CHARS = 100


def extract_attributed_claims(text: str) -> list[AttributedClaim]:
    """Extract code-level claims *text* attributes to a completed task/commit/ACTION.

    Emits a claim only when a specific CODE TOKEN co-occurs, within
    :data:`_CO_OCCURRENCE_WINDOW_CHARS` characters, with an ATTRIBUTION
    ANCHOR. A code token with no nearby anchor is a prospective new-work
    spec (e.g. "add a `metadata.foo` field") and is never flagged — only
    the combination signals a retrospective fact claim about prior work.

    Deduplicates by token, keeping the FIRST occurrence that has a
    co-occurring anchor (an earlier anchor-less mention of the same token
    does not suppress a later attributed one). Order is first-seen.

    Pure — no I/O, no verification (see ``verify_attributed_claims`` for
    that). Never raises; empty/whitespace/plain prose text returns ``[]``.
    """
    if not text:
        return []

    claims: list[AttributedClaim] = []
    seen_tokens: set[str] = set()

    for match in _CODE_TOKEN_PATTERN.finditer(text):
        token = match.group('meta') or match.group('tick') or match.group('stamp')
        if not token or token in seen_tokens:
            continue

        window_start = max(0, match.start() - _CO_OCCURRENCE_WINDOW_CHARS)
        window_end = min(len(text), match.end() + _CO_OCCURRENCE_WINDOW_CHARS)
        window = text[window_start:window_end]

        anchors: list[str] = []
        for anchor_match in _ATTRIBUTION_PATTERN.finditer(window):
            anchor_text = anchor_match.group(0)
            if anchor_text not in anchors:
                anchors.append(anchor_text)
        if not anchors:
            continue

        seen_tokens.add(token)
        claims.append(
            AttributedClaim(
                token=token,
                attribution=', '.join(anchors),
                span=' '.join(window.split()),
            )
        )

    return claims


def verify_attributed_claims(
    claims: list[AttributedClaim], probe: Callable[[str], bool],
) -> list[AttributedClaim]:
    """Return the subset of *claims* whose token *probe* reports ABSENT.

    ``probe(token) -> bool`` is the injectable verification seam: it should
    return ``True`` when the token is present somewhere relevant (e.g. the
    live source tree or git history) and ``False`` when it is absent. Only
    tokens the probe reports absent are returned — a claim whose token the
    probe confirms present is filtered out (self-correcting: a legitimate
    reference like ``get_statuses`` verifies clean regardless of how loosely
    it was extracted).

    Pure orchestration over the injected *probe* — no I/O of its own. Never
    raises on its own; a *probe* that raises propagates to the caller.
    """
    return [claim for claim in claims if not probe(claim.token)]


def unverified_claims_in_text(
    text: str, probe: Callable[[str], bool],
) -> list[AttributedClaim]:
    """Extract attributed claims from *text* and return the unverified subset.

    Composes :func:`extract_attributed_claims` and
    :func:`verify_attributed_claims` — mirrors
    ``recon_code_fix_premise_guard.premise_refuted_entry``'s
    detect-then-verify composition shape. Short-circuits to ``[]`` (without
    calling *probe*) when *text* carries no attributed claims at all.
    """
    claims = extract_attributed_claims(text)
    if not claims:
        return []
    return verify_attributed_claims(claims, probe)
