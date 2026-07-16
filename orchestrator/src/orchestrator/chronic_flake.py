"""Chronic pool-infra flake auto-file detector (task 2358).

Policy (Leo, 2026-07-08, verify-flakiness survey follow-up): after a verify
completes, detect a chronic pool-infra flake and auto-file a medium-priority
De-flake fix task into the project's task tree — non-blocking (the gate
stays green; retry-once already absorbed the failure) — so the flake debt
becomes visible, owned work instead of a warning nobody reads.

Substrate (reify task 5142, lands separately, cross-project runtime — NOT a
DF-code dependency): reify's ``tests/infra/run_all.sh`` persists every
serial-retry pass to ``<project_root>/data/verify-logs/flaky-ledger.jsonl``
(``{ts, test, role, flaky_count_window}`` per line) and emits a
line-anchored ``=== CHRONIC-FLAKY test=<name> count=<n> window=<m> ===``
marker when a test is flaky ``>= threshold`` times in the last ``window``
ledger-recorded runs.

Detection strategy (hedges reify's windowing ambiguity): the MARKER (parsed
from verify output) is reify's own authoritative "this is chronic" trigger;
the LEDGER read (from the stable project-root path) supplies the rich
evidence (dates/counts/roles) for the filed task's description AND an
independent/fallback trigger via this module's own configured
threshold/window. :func:`maybe_file_chronic_flake_tasks` unions both.

This module never talks to the fused-memory MCP directly (mirrors
``offline_lane.OfflineLaneTaskClient``'s cross-project scope boundary) —
:class:`ChronicFlakeTaskClient` is the pluggable seam, and
:class:`SchedulerChronicFlakeTaskClient` is the concrete adapter over a
duck-typed scheduler exposing ``dispatch_tool`` (self-contained here to
avoid a workflow<->harness import cycle).

Non-blocking guarantee: :func:`maybe_file_chronic_flake_tasks` is internally
catch-all-defensive (mirrors ``compute_failing_test_set_fingerprint``'s
fail-safe philosophy) so a ledger/MCP/search error can never fail the
verify/merge path; callers (``TaskWorkflow._maybe_file_chronic_flakes``)
wrap it in a second try/except as belt-and-suspenders.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CHRONIC-FLAKY marker parsing (step-3/step-4)
# ---------------------------------------------------------------------------


@dataclass
class ChronicFlakeMarker:
    """A single parsed ``=== CHRONIC-FLAKY test=<name> count=<n> window=<m> ===``
    marker line emitted by reify's ``tests/infra/run_all.sh`` (reify task 5142)."""

    test: str
    count: int
    window: int


# Line-anchored (lstrip + fullmatch semantics): ``^\s*`` tolerates leading
# whitespace only (never an arbitrary log/harness prefix), and ``\s*$``
# tolerates trailing whitespace only — so a marker embedded MID-LINE
# (assertion prose quoting the token) or prefixed by a harness/log tag
# cannot match. Mirrors the anchored discipline of
# ``verify._match_clock_marker`` (reify esc-4791-52 / task 4998 regression:
# a substring match let infra tests' own assertion prose masquerade as a
# real marker).
_CHRONIC_FLAKY_MARKER_RE = re.compile(
    r'^\s*=== CHRONIC-FLAKY test=(?P<test>\S+) count=(?P<count>\d+) window=(?P<window>\d+) ===\s*$'
)


def _match_chronic_flaky_marker(line: str) -> ChronicFlakeMarker | None:
    """Return a :class:`ChronicFlakeMarker` if *line* is a genuine, anchored
    CHRONIC-FLAKY marker; else ``None``.

    See module-level ``_CHRONIC_FLAKY_MARKER_RE`` docstring comment for the
    anchoring discipline. Malformed count/window (non-digit) fails the
    regex itself; the ``int()`` calls are defense-in-depth and should never
    raise given the ``\\d+`` capture groups.
    """
    m = _CHRONIC_FLAKY_MARKER_RE.match(line)
    if m is None:
        return None
    try:
        count = int(m.group('count'))
        window = int(m.group('window'))
    except (TypeError, ValueError):
        return None
    return ChronicFlakeMarker(test=m.group('test'), count=count, window=window)


def parse_chronic_flaky_markers(output: str) -> list[ChronicFlakeMarker]:
    """Scan *output* line-by-line and return every valid CHRONIC-FLAKY marker,
    in order. Non-matching lines (plain output, embedded/prefixed
    occurrences) are silently skipped. Returns ``[]`` for falsy *output*."""
    if not output:
        return []
    markers = []
    for line in output.splitlines():
        marker = _match_chronic_flaky_marker(line)
        if marker is not None:
            markers.append(marker)
    return markers
