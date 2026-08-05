"""Shared test kit: the capacity-failure skip guard.

Canonical home for the capacity-failure detection helper that was previously
copied byte-for-byte into ``test_cli_invoke_integration.py`` (as
``_CAPACITY_FAILURE_MARKERS`` / ``_looks_like_capacity_failure(AgentResult)``)
and ``test_usage_gate.py`` (same list, ``_looks_like_capacity_failure(str)``).
Both copies feed live ``pytest.skip`` call sites in real-CLI integration tests,
so a divergence between them is a test that silently skips — or silently goes
red — on one path and not the other.  Two entry points remain because the call
sites genuinely differ in what they hold: an ``AgentResult`` in the cli_invoke
suite, a pre-combined ``f'{stdout}\\n{stderr}'`` string in the usage-gate probe
guard.  One policy, two calling conventions.

Consumers import by bare module name (``from _capacity_skip import ...``) --
``conftest.py`` prepends ``shared/tests`` to ``sys.path`` (same convention as
``_usage_gate_test_helpers.py``).  The leading underscore keeps pytest from
collecting this module as a test file; its contract tests live in
``test_capacity_skip.py``.
"""

from __future__ import annotations

from shared.cli_invoke import AgentResult

CAPACITY_FAILURE_MARKERS: tuple[str, ...] = (
    ' capped',          # leading space prevents matching 'uncapped' as a false positive
    'rate limit',
    'account unavailable',  # narrowed from bare 'unavailable' to avoid generic network errors
    'out of extra usage',
    'usage limit',
    # Production's own prefix, verbatim: invocation_outcome.CAP_HIT_PREFIXES[0].
    # It was once narrowed to "you've hit your usage" to avoid matching
    # "You've hit a snag" — but that narrowing was NOT load-bearing ("hit a" is
    # not "hit your", so the snag phrasing never matched either way) and it
    # cost every non-'usage' limit variant: weekly, session, 5-hour, Opus.  Do
    # not re-narrow it; the boundary negative it was meant to guard is pinned
    # in test_capacity_skip.py and stays green with the broad prefix.
    "you've hit your",
    # Lead-in-independent backstops, so a CLI rewording that drops the
    # "You've hit your" preamble still trips the guard.  The 5-hour variant
    # needs no marker of its own — it is covered twice over, by the prefix
    # above and by 'usage limit'.
    'weekly limit',
    'session limit',
    # THIS narrowing IS load-bearing, unlike the "hit" one above: a bare
    # "you've used" matches the pinned negative "You've used the wrong
    # format."  Leave it narrow.
    "you've used all",
)


# Verbatim real-CLI capacity messages, each with recorded provenance.  This is
# the SINGLE source for these strings — test modules derive their corpora from
# it rather than restating the text (a second hand-maintained copy of a
# verbatim transcript is the same drift trap this module exists to close).
# Used by test_capacity_skip.py's drift guard, which asserts that the skip
# helper and the PRODUCTION detector (invocation_outcome.classify_invocation)
# agree these are cap messages.  Add an entry only with a transcript to cite.
#
# Split by the outcome production assigns each message, because CapHit and
# NearCap are NOT interchangeable there: usage_gate routes CapHit to
# ``_handle_cap_detected`` (CAPPED phase, cap-retry / account failover) and
# NearCap to ``_handle_near_cap_warning``, which is annotation-only.  The split
# lets the drift guard assert the exact variant, so a regression that downgrades
# a genuine cap-hit to a near-cap warning — silently disabling cap-retry — goes
# red instead of passing a loose isinstance check.  The skip helper itself does
# not distinguish them (it skips on either), so it consumes the union below.

# Messages production classifies as a confirmed cap hit.
REAL_CLI_CAP_HIT_MESSAGES: tuple[str, ...] = (
    # Observed 2026-08-01, claude CLI 2.1.220, task 3454, during the
    # cross-account resume probe: account B answered the resumed turn with
    # this and no model turn ever ran.  Recorded verbatim in the measurement
    # comment at ``shared/src/shared/cli_invoke.py`` L1435.  This is the ONLY
    # entry observed as a transcript first-hand.
    "You've hit your weekly limit · resets Aug 5, 11am",
    # The cap-hit messages that motivated the marker list.  Their original
    # provenance pointer read "shared.usage_gate inline comments, lines 64-75";
    # task 2129 moved those string tables to ``shared.invocation_outcome``
    # (CAP_HIT_PREFIXES / CAP_CONFIRM_KEYWORDS / NEAR_CAP_PREFIXES), so do not
    # go chasing them in usage_gate.py.
    "You've hit your usage limit for Claude Pro. Your plan resets in 3 hours.",
    "You've used all available credits. Upgrade your plan for more capacity.",
    "You're out of extra usage for this billing period. Your plan resets in 2h.",
    # DELIBERATELY ABSENT: the session-limit and 5-hour variants.  The
    # 2026-08-01 observation described a session limit only in prose (see
    # cli_invoke.py L1445), never verbatim.  Inventing a plausible
    # "You've hit your session limit · resets ..." and pinning it HERE — the
    # one place a future engineer will read as ground truth — would put a
    # fabricated transcript in the record.  Those variants are covered
    # structurally by the markers instead.
)

# Messages production classifies as approaching, but not yet at, the cap.
REAL_CLI_NEAR_CAP_MESSAGES: tuple[str, ...] = (
    "You're close to reaching your usage limit. Your plan resets in 1h.",
)

# The union — what the skip helper sees, since it skips on either variant.
REAL_CLI_CAP_MESSAGES: tuple[str, ...] = (
    REAL_CLI_CAP_HIT_MESSAGES + REAL_CLI_NEAR_CAP_MESSAGES
)


def looks_like_capacity_failure(text: str) -> bool:
    """Return True when *text* looks like a Claude CLI capacity / quota failure.

    Case-insensitive substring match of *text* against a small focused list of
    markers drawn from real Claude CLI cap messages.  (The verbatim strings
    those markers were derived from lived in ``shared.usage_gate`` inline
    comments until task 2129 moved the tables to
    ``shared.invocation_outcome`` — ``CAP_HIT_PREFIXES`` /
    ``CAP_CONFIRM_KEYWORDS`` / ``NEAR_CAP_PREFIXES``.)

    **Conservative bias (fail loudly when uncertain).** This helper is used
    at ``pytest.skip`` call sites, so a false positive — skipping on a real
    regression — is the exact failure mode we are trying to prevent. The list
    is therefore intentionally small and obvious; anything not matching a
    well-known capacity signal falls through to an ``assert`` that fails the
    test loudly.

    **Purpose-built list, not an import of the production detector.** The
    production cap detector (``usage_gate.detect_cap_hit``, which delegates to
    ``invocation_outcome.classify_invocation(..., strict_confirm=True)``)
    requires BOTH a prefix AND a confirm-keyword match — a strict combined
    policy designed to avoid marking healthy accounts as capped. Re-using those
    lists here would either collapse the combined check to a loose OR (pulling
    in confirm keywords like ``"resets"`` as standalone signals) or miss real
    cap messages that arrive without the expected prefix. A purpose-built
    substring list is the correct shape for this use-case.
    """
    haystack = text.lower()
    return any(marker in haystack for marker in CAPACITY_FAILURE_MARKERS)


def result_looks_like_capacity_failure(result: AgentResult) -> bool:
    """``looks_like_capacity_failure`` over an ``AgentResult``.

    Inspects both ``result.output`` and ``result.stderr`` — a cap message can
    arrive on either stream depending on how the CLI failed.

    **Must remain a pure delegation.** This is a calling convention, not a
    second policy: everything it decides is decided by
    ``looks_like_capacity_failure``. Growing independent matching logic here
    re-creates the two-implementations-that-can-disagree bug this module was
    extracted to kill, one level down.
    """
    return looks_like_capacity_failure(f'{result.output}\n{result.stderr}')
