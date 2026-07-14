"""InvocationOutcome sum type + classify_invocation — the W4 cap/error classification seam.

Consolidates the cap/near-cap/auth-failure/CLI-error/wedge classification
logic previously scattered across ``shared.usage_gate`` (Claude/Codex/Gemini
cap string tables, ``detect_cap_hit``) and ``shared.cli_invoke``
(``NON_CAP_CLI_ERROR_MARKERS``, ``is_zero_output_timeout``) into one total,
pure classifier: :func:`classify_invocation`.

This module is additive only — it does not modify ``usage_gate.py`` or
``cli_invoke.py``. Rewiring those consumers to call ``classify_invocation``
is a follow-up task; until then the string tables are intentionally
duplicated between the old and new homes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

# Real (non-TYPE_CHECKING) import: is_zero_output_timeout is CALLED by the
# ZeroOutputWedge tier below, not just referenced as a type annotation. This
# keeps the transcript-authoritative + legacy-fallback wedge definition
# single-sourced (cli_invoke.py:357) so it cannot drift from this module's
# copy. Safe: shared.cli_invoke does not import shared.invocation_outcome, so
# this introduces no import cycle.
from shared.cli_invoke import is_zero_output_timeout

if TYPE_CHECKING:
    from shared.cli_invoke import AgentResult

__all__ = [
    'InvocationOutcome',
    'OK',
    'CapHit',
    'NearCap',
    'AuthFailed',
    'CliLocalError',
    'ModelNotFound',
    'ZeroOutputWedge',
    'Failure',
    'classify_invocation',
]


class InvocationOutcome:
    """Base class for the invocation-outcome tagged union.

    Every concrete variant is a frozen dataclass subclass — see ``__all__``
    for the full set. Use ``isinstance(outcome, <Variant>)`` to discriminate.
    """


@dataclass(frozen=True)
class OK(InvocationOutcome):
    """The invocation completed successfully."""


@dataclass(frozen=True)
class CapHit(InvocationOutcome):
    """The backend reported (or the CLI heuristically detected) a usage cap."""

    resets_at: datetime | None
    reason: str


@dataclass(frozen=True)
class NearCap(InvocationOutcome):
    """The backend warned that a usage cap is imminent (not yet blocking)."""

    reason: str


@dataclass(frozen=True)
class AuthFailed(InvocationOutcome):
    """The backend rejected the request as unauthorized (HTTP 401/403)."""

    status: int


@dataclass(frozen=True)
class CliLocalError(InvocationOutcome):
    """A local CLI/usage error occurred that must never be treated as a cap hit."""

    marker: str


@dataclass(frozen=True)
class ModelNotFound(InvocationOutcome):
    """The requested model string does not exist / is not available to this
    account (task beta, plans/adaptive-model-routing-prd.md).

    TERMINAL: a model-not-found result is zero-cost, near-instant, and
    <=1 turn — it must never be treated as a usage-cap signal (that
    misclassification is what previously drove whole-pool failover churn in
    ``invoke_with_cap_retry``; see ``shared.cli_invoke``).
    """

    reason: str


@dataclass(frozen=True)
class ZeroOutputWedge(InvocationOutcome):
    """The invocation timed out having produced no transcript turns (a wedge)."""


@dataclass(frozen=True)
class Failure(InvocationOutcome):
    """The invocation failed for a reason not covered by the other variants."""

    kind: str


_MONTH_ABBR = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
}

# NOTE — fork, not drift-guarded: _parse_resets_at and _extract_cap_message
# below are ported from usage_gate.py (their regex/parsing logic originated
# there), but unlike CAP_HIT_PREFIXES/CAP_CONFIRM_KEYWORDS/NEAR_CAP_PREFIXES/
# CODEX_CAP_PATTERNS/GEMINI_CAP_PATTERNS/NON_CAP_CLI_ERROR_MARKERS (pinned by
# TestStringTableDriftGuard in test_invocation_outcome.py), these two
# functions have NO automated guard keeping them in sync with their
# usage_gate.py originals — and have already intentionally diverged: this
# copy of _parse_resets_at returns None on parse failure and accepts an
# injectable ``now`` (PRD 7.1.a — an unknown reset time must be reported as
# explicitly unknown, never fabricated, and tests need determinism), whereas
# usage_gate.py's ``_parse_resets_at(text) -> datetime`` fabricates
# ``now + 1h`` on parse failure and always reads the real wall clock. Both
# copies are pure and forked deliberately (this task is additive-only; see
# the module docstring), so a future edit to one is not expected to be
# mirrored in the other — but it also won't be caught if it should have
# been. Resolved by the beta consumer-rewire, which is expected to delete
# usage_gate.py's copies in favour of these once callers are repointed here.


def _parse_resets_at(text: str, *, now: datetime | None = None) -> datetime | None:
    """Parse reset time from cap-hit message text.

    Handles:
    - "resets in 3h" / "resets in 45m" / "resets in 2d"
    - "resets [on] Mar 30, 6pm (Europe/London)" (date + time + tz)
    - "resets 9pm (Europe/London)" / "resets 3:00 AM (US/Pacific)"

    Returns ``None`` when no pattern matches — PRD 7.1.a: an unknown reset
    time must be reported as explicitly unknown, never fabricated as
    "now + 1 hour".

    ``now`` is the reference "current time" used both for relative-delta
    arithmetic and for inferring the year / next-day rollover of absolute
    times. It defaults to the real wall clock (``datetime.now(UTC)``) —
    tests inject a fixed value for determinism.
    """
    if now is None:
        now = datetime.now(UTC)

    # Relative: "resets in Xh", "resets in Xm", "resets in Xd" (word forms
    # like "3 hours" / "45 minutes" / "2 days" match too — only the first
    # letter of the unit word is read). The (?!onth) guard stops "3 months"
    # from being misread as "3 minutes": "months" also starts with 'm', but
    # is not a supported unit, so it must fall through to None (7.1.a)
    # rather than silently parsing as minutes.
    m = re.search(r'resets\s+in\s+(\d+)\s*([hmd])(?!onth)', text, re.IGNORECASE)
    if m:
        amount = int(m.group(1))
        unit = m.group(2).lower()
        delta = {
            'h': timedelta(hours=amount),
            'm': timedelta(minutes=amount),
            'd': timedelta(days=amount),
        }.get(unit, timedelta(hours=1))
        return now + delta

    # Absolute with date: "resets Mar 30, 6pm (Europe/London)" or
    # "resets on June 5, 7pm (Europe/London)" (the "on" is optional). Month
    # accepts 3-9 chars (any abbreviation through full name) and is matched
    # against _MONTH_ABBR by its lowercased first three characters, since
    # every English month is uniquely identified by them.
    m = re.search(
        r'resets\s+(?:on\s+)?([A-Za-z]{3,9})\s+(\d{1,2}),?\s+'
        r'(\d{1,2}(?::\d{2})?\s*[ap]m)\s*\(([^)]+)\)',
        text, re.IGNORECASE,
    )
    if m:
        try:
            import zoneinfo
            month_str = m.group(1).lower()[:3]
            day = int(m.group(2))
            time_str = m.group(3).strip()
            tz_str = m.group(4).strip()
            tz = zoneinfo.ZoneInfo(tz_str)
            month = _MONTH_ABBR.get(month_str)
            if month is None:
                raise ValueError(f'Unknown month: {month_str}')
            for fmt in ('%I:%M %p', '%I%p', '%I:%M%p', '%I %p'):
                try:
                    parsed_time = datetime.strptime(time_str, fmt).time()
                    break
                except ValueError:
                    continue
            else:
                raise ValueError(f'Cannot parse time: {time_str}')
            now_in_tz = now.astimezone(tz)
            year = now_in_tz.year
            target = now_in_tz.replace(
                year=year, month=month, day=day,
                hour=parsed_time.hour, minute=parsed_time.minute,
                second=0, microsecond=0,
            )
            # If target is in the past, assume next year
            if target <= now_in_tz:
                target = target.replace(year=year + 1)
            return target.astimezone(UTC)
        except Exception:
            pass

    # Absolute: "resets Xpm (TZ)" or "resets X:XX AM (TZ)"
    m = re.search(
        r'resets\s+(\d{1,2}(?::\d{2})?\s*[ap]m)\s*\(([^)]+)\)',
        text, re.IGNORECASE,
    )
    if m:
        try:
            import zoneinfo
            time_str = m.group(1).strip()
            tz_str = m.group(2).strip()
            tz = zoneinfo.ZoneInfo(tz_str)
            for fmt in ('%I:%M %p', '%I%p', '%I:%M%p', '%I %p'):
                try:
                    parsed_time = datetime.strptime(time_str, fmt).time()
                    break
                except ValueError:
                    continue
            else:
                return None

            now_in_tz = now.astimezone(tz)
            target = now_in_tz.replace(
                hour=parsed_time.hour,
                minute=parsed_time.minute,
                second=0, microsecond=0,
            )
            if target <= now_in_tz:
                target += timedelta(days=1)
            return target.astimezone(UTC)
        except Exception:
            pass

    # No pattern matched: unknown reset time (PRD 7.1.a) — explicit None,
    # never a fabricated now+1h.
    return None


# Concrete CLI/usage errors that exit instantly with no output but are NOT
# usage caps. Matched case-insensitively against stderr/output so the
# zero-cost cap heuristic doesn't misfire (and loop forever) on a local CLI
# failure. Copied from cli_invoke.py:120 (NON_CAP_CLI_ERROR_MARKERS) — this
# module owns the classification decision now, but that module's copy stays
# in place until the beta (consumer-rewire) task removes it.
NON_CAP_CLI_ERROR_MARKERS = [
    'is already in use',        # --session-id collision (reify-3604)
    'unrecognized arguments',
    'unknown option',
    'invalid value',
    'no such file or directory',
    'permission denied',
]

# Substrings (case-insensitive) indicating the requested model does not
# exist / is not available (task beta, plans/adaptive-model-routing-prd.md).
# Checked as a fallback when no structured api_error_status==404 is present
# — e.g. a body-only error where the CLI didn't surface a status code.
MODEL_NOT_FOUND_MARKERS = [
    'not_found_error',
    'model not found',
]

# Patterns that indicate a usage cap has been hit (from Claude Code CLI
# output). Copied from usage_gate.py:48 (CAP_HIT_PREFIXES).
CAP_HIT_PREFIXES = [
    "You've hit your",
    "You've used",
    "You're out of extra",
    "You're now using extra",
]
# Secondary confirmation — at least one of these keywords must also appear in
# the same text for a CAP_HIT or NEAR_CAP prefix match to be accepted
# (defense-in-depth against ambiguous prefix false positives).
# NOTE: 'upgrade' was narrowed to multi-word phrases because the bare verb is
# too common in unrelated CLI messaging (e.g. 'Upgrade to v2 for more features')
# and would effectively reduce the guard to a near-prefix-only match in those
# cases.  'upgrade your plan' and 'upgrade your subscription' are natural SaaS
# cap-message phrases unlikely to appear in non-cap contexts.  The primary
# defense remains the CAP_HIT_PREFIXES / NEAR_CAP_PREFIXES prefix match.
# Copied from usage_gate.py:78 (CAP_CONFIRM_KEYWORDS).
CAP_CONFIRM_KEYWORDS = ["resets", "usage limit", "upgrade your plan", "upgrade your subscription"]

# Patterns for near-cap warnings (pause proactively). Copied from
# usage_gate.py:81 (NEAR_CAP_PREFIXES).
NEAR_CAP_PREFIXES = [
    "You're close to",
]

# Codex (OpenAI) cap-hit patterns. Copied from usage_gate.py:86 (CODEX_CAP_PATTERNS).
CODEX_CAP_PATTERNS = ['usage limit reached', 'rate limit', 'quota exceeded',
                      'insufficient_quota', 'rate_limit_exceeded']

# Gemini (Google) cap-hit patterns. Copied from usage_gate.py:90 (GEMINI_CAP_PATTERNS).
GEMINI_CAP_PATTERNS = ['quota exceeded', 'rate limit', 'resource exhausted',
                       'RESOURCE_EXHAUSTED', 'quota_exceeded']


def _extract_cap_message(text: str, prefix: str) -> str:
    """Extract the full sentence containing the cap-hit prefix.

    Forked from usage_gate.py, same as ``_parse_resets_at`` above — see the
    fork/no-drift-guard note by that function's definition. Not covered by
    TestStringTableDriftGuard, which pins only the plain string-table
    constants.
    """
    lower = text.lower()
    idx = lower.find(prefix.lower())
    if idx == -1:
        return ''
    # Find the end of the sentence
    end = text.find('\n', idx)
    if end == -1:
        end = min(idx + 200, len(text))
    return text[idx:end].strip()


def classify_invocation(
    result: AgentResult,
    *,
    strict_confirm: bool,
    backend: str = 'claude',
) -> InvocationOutcome:
    """Classify a CLI agent invocation result into an ``InvocationOutcome``.

    Total and pure: every ``AgentResult`` maps to exactly one variant, in
    precedence order AuthFailed > ModelNotFound (404) > OK > ModelNotFound
    (marker) > CliLocalError > CapHit/NearCap > ZeroOutputWedge > Failure.
    Reads only ``result`` (and ``backend`` / ``strict_confirm``) — no I/O,
    no gate mutation.
    """
    # AuthFailed — narrow to {401, 403}. 429 is deliberately EXCLUDED: it
    # carries a real cap-message body ("You're out of extra usage · resets
    # ...") that the cap-hit tier recognises, so routing 429 here would skip
    # cap detection entirely and never trip AllAccountsCappedException
    # (mirrors cli_invoke.py:799-805).
    if result.api_error_status in (401, 403):
        return AuthFailed(status=result.api_error_status)

    combined = f'{result.stderr or ""}\n{result.output or ""}'
    combined_lower = combined.lower()

    # ModelNotFound (structured, status-based) — a 404 is never a successful
    # invocation, so keying on the structured status code here carries no
    # false-positive risk, exactly like AuthFailed's placement above. Must
    # sit ABOVE the `if result.success` short-circuit for the same reason.
    # TERMINAL: a model-not-found result is zero-cost / near-instant /
    # <=1 turn — classifying it here (rather than falling through to
    # Failure(unclassified)) is what lets invoke_with_cap_retry stop it
    # before the heuristic cap safety-net mislabels it a CapHit and cycles
    # every account in the pool.
    if result.api_error_status == 404:
        detail = combined.strip().splitlines()[0] if combined.strip() else ''
        reason = 'HTTP 404 model-not-found' + (f': {detail}' if detail else '')
        return ModelNotFound(reason=reason)

    # OK — a successful invocation short-circuits every remaining
    # string-based heuristic below: CLI-local-error markers, cap/near-cap
    # prefixes, and wedge detection alike. Without this, a successful run
    # whose output happens to *quote* incidental marker text — a CLI-error
    # marker (e.g. an agent that ran `ls` on a missing path and echoed "no
    # such file or directory", or discussed a "permission denied" error) or
    # cap-like text (e.g. echoing a usage-limit string back) — would be
    # misclassified as CliLocalError/CapHit/NearCap even though nothing
    # failed. The legacy code only ever ran these checks on the failure path
    # (``_is_non_cap_cli_error`` / ``detect_cap_hit``), so this restores that
    # scoping. AuthFailed stays ABOVE this check because it keys off a
    # structured, exact status code rather than fuzzy substring matching
    # against free-form text, so it doesn't carry the same false-positive
    # risk against incidental output.
    if result.success:
        return OK()

    # ModelNotFound (fuzzy, marker-based) — a fallback for a body-only
    # model-not-found error with no structured api_error_status==404.
    # Placed BELOW the OK short-circuit (so a successful run that merely
    # *quotes* a marker substring, e.g. discussing an error format, stays
    # OK) and ABOVE CliLocalError/cap/wedge (so a real failed
    # model-not-found is caught before the heuristic cap safety-net can
    # mislabel the zero-cost/instant result a CapHit).
    for marker in MODEL_NOT_FOUND_MARKERS:
        if marker in combined_lower:
            return ModelNotFound(reason=f'model-not-found marker {marker!r} in output')

    # CliLocalError — placed ABOVE cap detection. This precedence position IS
    # the reify-3604 structural fix: a local CLI/usage error (e.g. a
    # --session-id collision) must never be treated as a usage cap, even when
    # the same output also happens to contain cap-like text.
    for marker in NON_CAP_CLI_ERROR_MARKERS:
        if marker in combined_lower:
            return CliLocalError(marker=marker)

    # Backend-specific cap patterns checked first, ahead of the claude prefix
    # logic below (mirrors usage_gate.py:386-397). Codex/Gemini error bodies
    # don't carry a parseable reset time, so resets_at is always None here.
    if backend == 'codex':
        for pattern in CODEX_CAP_PATTERNS:
            if pattern.lower() in combined_lower:
                return CapHit(resets_at=None, reason=f'Codex cap hit: {pattern}')
    elif backend == 'gemini':
        for pattern in GEMINI_CAP_PATTERNS:
            if pattern.lower() in combined_lower:
                return CapHit(resets_at=None, reason=f'Gemini cap hit: {pattern}')

    # Claude cap/near-cap detection: require both a prefix match AND a
    # secondary confirmation keyword (defence against false positives on
    # generic prefixes like "You've used" or "You're close to") — UNLESS
    # strict_confirm=False, which skips the confirm-keyword guard entirely.
    #
    # NOTE — intentional asymmetry, DO NOT 'fix' by unifying the two regimes:
    # strict_confirm=True is the detect_cap_hit regime (retry-loop / gate
    # mutation path): a false-positive cap detection here would wrongly pause
    # a healthy account, so it demands the extra confirm-keyword evidence.
    # strict_confirm=False is the DD-2 _run_probe regime: the probe only runs
    # while an account is ALREADY blocked (capped or auth_failed), so any
    # whiff of a cap-like prefix means "still capped, do not resume" — a
    # false positive there just means one extra probe cycle, whereas a false
    # negative (wrongly resuming a still-capped account) burns quota on a
    # limited account. The two regimes optimise for different failure costs,
    # so the guard is intentionally asymmetric. See
    # test_probe_prefix_only_without_confirm_keyword_still_returns_false in
    # test_usage_gate_exhaustive.py for the historical rationale.
    has_confirm_keyword = any(kw in combined_lower for kw in CAP_CONFIRM_KEYWORDS)
    if not strict_confirm or has_confirm_keyword:
        for prefix in CAP_HIT_PREFIXES:
            if prefix.lower() in combined_lower:
                resets_at = _parse_resets_at(combined)
                reason = _extract_cap_message(combined, prefix) or f'Cap detected: {prefix}'
                return CapHit(resets_at=resets_at, reason=reason)

        for prefix in NEAR_CAP_PREFIXES:
            if prefix.lower() in combined_lower:
                reason = _extract_cap_message(combined, prefix) or f'Near-cap warning: {prefix}'
                return NearCap(reason=reason)

    # ZeroOutputWedge — a full-timeout invocation that produced no transcript
    # progress. Delegates to is_zero_output_timeout (cli_invoke.py:357) so the
    # transcript-authoritative (transcript_turns == 0) + legacy fallback
    # (transcript_turns is None and turns == 0 and cost_usd == 0.0) wedge
    # definition stays single-sourced. Placed below cap detection so a cap
    # message on a zero-turn timeout still classifies as CapHit, not a wedge.
    if is_zero_output_timeout(result):
        return ZeroOutputWedge()

    return Failure(kind='unclassified')
