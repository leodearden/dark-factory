"""Canonical cap / auth banner markers — the loose-substring detection contract.

The single home for "does this CLI reply mean no useful model output is
coming?".  Two independent consumers share it:

* ``shared/tests/_capacity_skip.py`` (task 3483) — the ``pytest.skip`` guard
  for real-CLI integration tests.  Matches CAPACITY only.
* ``scripts/legibility/census.py`` — the usage-headroom probe that gates the
  legibility census's mining and verify stages.  Matches capacity OR auth,
  since either means the probe reply is a banner rather than a model turn.

**Why this lives in ``src/`` rather than ``tests/``.**  The list started in
``shared/tests/_capacity_skip.py``, which is importable solely via that
suite's ``conftest.py`` sys.path insert — so the census RUNTIME could not
reach it, and grew its own four-marker copy instead.  That copy missed
``"You've hit your weekly limit · resets Aug 5, 11am"`` entirely, so a weekly
cap passed the census preflight and every subsequent verify call was
fail-closed rejected as an ordinary verdict (task 3645).  Two sites, two
lists, the same blind spot found twice: the fix is one production module both
can import.

**Purpose-built list, NOT an import of the production cap detector.**  The
production detector (``usage_gate.detect_cap_hit``, delegating to
``invocation_outcome.classify_invocation(..., strict_confirm=True)``) requires
BOTH a prefix AND a confirm-keyword match — a deliberately strict combined
policy tuned to avoid marking a healthy account as capped, because there the
consequence is account failover.  Reusing those tables here would either
collapse the combined check to a loose OR (pulling in standalone confirm
keywords like ``"resets"``) or miss real cap messages arriving without the
expected prefix.  The loose OR-substring shape is the correct one for THIS
job — a skip guard and a defer gate, where the cost of a miss is a silently
wrong result and the cost of a false positive is a deferred run.  The two
policies are different contracts for different jobs and are kept apart
deliberately; ``test_capacity_skip.py`` carries a bidirectional drift guard
asserting they still agree on the real-CLI corpus below.

**The marker list is a CONTRACT, not a literal.**  Cap text has changed
before and will change again.  ``REAL_CLI_CAP_MESSAGES`` is parametrized over
by both ``shared/tests/test_cap_markers.py`` and
``scripts/tests/test_legibility_census.py``, so adding a newly-observed
phrasing to the corpus turns BOTH suites red until the markers cover it.  Add
a corpus entry only with a transcript to cite.

Not re-exported from ``shared/__init__.py``: ``shared.invocation_outcome``
isn't either, submodule imports are this package's norm, and
``test_public_api.py`` pins the curated top-level surface.
"""

from __future__ import annotations

CAPACITY_BANNER_MARKERS: tuple[str, ...] = (
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
"""Capacity / quota banners.  Matched ALONE by the ``pytest.skip`` guard,
which must stay capacity-only: an auth failure there has to remain a loud red
test, never a quiet skip."""


AUTH_BANNER_MARKERS: tuple[str, ...] = (
    'please run /login',
    'invalid api key',
)
"""Auth banners.  Lifted from the census's original four-entry marker literal
(deleted with task 3645's amendment pass, which left this module the only
home).  Kept as a SEPARATE tuple from the capacity
markers because the two consumers want different sets — merging them would
silently convert every auth failure in ``test_cli_invoke_integration.py``
from a red test into a skip."""


BLOCKING_BANNER_MARKERS: tuple[str, ...] = CAPACITY_BANNER_MARKERS + AUTH_BANNER_MARKERS
"""The union — every banner meaning "no useful model output is coming".  What
the census headroom probe scans, since it must defer on either cause.
Derived, never hand-maintained."""


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


def _first_marker(text: str, markers: tuple[str, ...]) -> str | None:
    """Return the first marker in *markers* present in *text*, else None.

    The single matching implementation both public matchers delegate to, so
    the two can never disagree about what "matching" means.  Lowercases the
    HAYSTACK once and never the needle — which is why every marker tuple must
    already be lowercase (an uppercase entry would be permanently dead, a
    silent coverage hole; ``test_cap_markers.py`` pins this structurally).
    """
    haystack = text.lower()
    for marker in markers:
        if marker in haystack:
            return marker
    return None


def looks_like_capacity_banner(text: str) -> str | None:
    """Return the matched capacity marker when *text* is a capacity/quota
    banner, else None.

    Case-insensitive loose substring match against ``CAPACITY_BANNER_MARKERS``.
    Auth banners deliberately do NOT match — use ``looks_like_blocking_banner``
    when auth counts too.

    **Conservative bias (fail loudly when uncertain).**  This feeds
    ``pytest.skip`` call sites, so a false positive — skipping on a real
    regression — is the exact failure mode being prevented.  The list stays
    small and obvious; anything not matching a well-known capacity signal
    falls through to an assertion that fails the test loudly.

    Returns the MARKER rather than a bool so a caller can name which signal
    fired: the census puts it verbatim in its deferral reason, which is the
    difference between an operator seeing "capped on the weekly limit" and
    seeing "deferred".
    """
    return _first_marker(text, CAPACITY_BANNER_MARKERS)


def looks_like_blocking_banner(text: str) -> str | None:
    """Return the matched marker when *text* is a capacity OR auth banner.

    The census headroom probe's matcher: it must defer on either cause, since
    both mean the reply is a banner rather than a model turn and no useful
    output is coming.  Capacity markers are checked first (see
    ``BLOCKING_BANNER_MARKERS``' ordering), so a message that somehow carried
    both reports the capacity cause — the one with a recovery path an operator
    can act on.
    """
    return _first_marker(text, BLOCKING_BANNER_MARKERS)
