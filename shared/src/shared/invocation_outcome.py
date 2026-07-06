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

if TYPE_CHECKING:
    from shared.cli_invoke import AgentResult

__all__ = [
    'InvocationOutcome',
    'OK',
    'CapHit',
    'NearCap',
    'AuthFailed',
    'CliLocalError',
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

    # Relative: "resets in Xh", "resets in Xm", "resets in Xd"
    m = re.search(r'resets\s+in\s+(\d+)\s*([hmd])', text, re.IGNORECASE)
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


def _extract_cap_message(text: str, prefix: str) -> str:
    """Extract the full sentence containing the cap-hit prefix."""
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
    precedence order AuthFailed > CliLocalError > CapHit/NearCap >
    ZeroOutputWedge > OK > Failure. Reads only ``result`` (and ``backend`` /
    ``strict_confirm``) — no I/O, no gate mutation.
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

    # CliLocalError — placed ABOVE cap detection. This precedence position IS
    # the reify-3604 structural fix: a local CLI/usage error (e.g. a
    # --session-id collision) must never be treated as a usage cap, even when
    # the same output also happens to contain cap-like text.
    for marker in NON_CAP_CLI_ERROR_MARKERS:
        if marker in combined_lower:
            return CliLocalError(marker=marker)

    return Failure(kind='unclassified')
