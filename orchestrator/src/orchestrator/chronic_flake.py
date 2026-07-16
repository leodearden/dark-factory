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

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

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


# ---------------------------------------------------------------------------
# Flaky ledger read + chronic-test computation (step-5/step-6)
# ---------------------------------------------------------------------------


def read_flaky_ledger(path: str | Path) -> list[dict]:
    """Best-effort read of reify's ``flaky-ledger.jsonl`` (task 5142):
    one ``{ts, test, role, flaky_count_window}`` JSON object per line.

    Tolerant by design — this is evidence-gathering for a non-blocking
    filing decision, never a correctness-critical parse: blank lines,
    malformed JSON, and well-formed-but-non-dict rows are logged and
    skipped rather than raising. A missing ledger file (reify:5142 not yet
    landed, or no flakes recorded yet) returns ``[]``.
    """
    ledger_path = Path(path)
    if not ledger_path.exists():
        return []
    entries: list[dict] = []
    for line in ledger_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            logger.warning('Skipping malformed flaky-ledger line: %s', line[:120])
            continue
        if not isinstance(row, dict):
            logger.warning('Skipping non-dict flaky-ledger line: %s', line[:120])
            continue
        entries.append(row)
    return entries


@dataclass
class ChronicFlakeEvidence:
    """Evidence bundle for a test computed as chronic from the last
    ``window`` flaky-ledger entries — feeds
    :func:`build_chronic_flake_fix_task_arguments`'s description."""

    test: str
    count: int
    window: int
    dates: list[str]
    roles: list[str]
    entries: list[dict]


def compute_chronic_flakes(
    entries: list[dict], threshold: int, window: int
) -> list[ChronicFlakeEvidence]:
    """Group the last *window* ledger *entries* by ``test`` and flag every
    test occurring ``>= threshold`` times as chronic.

    Entries are assumed chronologically ordered (oldest first, as appended
    by reify's ``run_all.sh``), so "the last `window` entries" is a plain
    tail slice. Sub-threshold tests are excluded. Returns ``[]`` for empty
    *entries* or when nothing meets *threshold*.
    """
    if not entries:
        return []
    considered = entries[-window:] if window > 0 else []
    by_test: dict[str, list[dict]] = {}
    for entry in considered:
        if not isinstance(entry, dict):
            continue
        test = entry.get('test')
        if not test:
            continue
        by_test.setdefault(test, []).append(entry)
    evidence = []
    for test, matches in by_test.items():
        count = len(matches)
        if count < threshold:
            continue
        dates = [str(match.get('ts', '')) for match in matches]
        roles = sorted({str(match['role']) for match in matches if match.get('role')})
        evidence.append(
            ChronicFlakeEvidence(
                test=test, count=count, window=window, dates=dates, roles=roles, entries=matches
            )
        )
    return evidence


# ---------------------------------------------------------------------------
# De-flake fix-task argument builder (step-7/step-8)
# ---------------------------------------------------------------------------

# Fixed root-cause guidance, always appended to the filed task's
# description. Deliberately steers AWAY from the laziest "fix": widening a
# sleep/timeout, which just makes the flake rarer and harder to reproduce
# later. Condition-polling (wait for the actual state the test needs) and
# structural asserts (assert on the state itself, not on prose/log output)
# are the durable fix per the infra-test-wallclock-deflake toolkit.
_ROOT_CAUSE_INSTRUCTION = (
    'ROOT-CAUSE this, do not paper over it: replace wall-clock sleeps with '
    'condition-polling (wait for the actual state the test depends on) and '
    'prefer structural asserts (assert on state, not on log/prose output) '
    'per the infra-test-wallclock-deflake toolkit. NEVER "fix" this with a '
    'blind timeout bump — that only makes the flake rarer and harder to '
    'reproduce, it does not remove it.'
)


def build_chronic_flake_fix_task_arguments(
    evidence: ChronicFlakeEvidence, project_root: str | Path
) -> dict:
    """Build the ``submit_task`` argument block for an auto-filed De-flake
    fix task (task 2358).

    Modeled on :func:`~orchestrator.workflow.build_offline_lane_fix_task_arguments`
    (title/description/priority/project_root/metadata shape). Unlike the
    offline-lane red-fix task, this fires on a test that is CURRENTLY
    GREEN overall (retry-once absorbed the flake) — so it is filed at
    ``priority='medium'``, non-blocking, purely to make the flake debt
    visible and owned.
    """
    dates_str = ', '.join(evidence.dates) if evidence.dates else '(none recorded)'
    roles_str = ', '.join(evidence.roles) if evidence.roles else '(none recorded)'
    title = f'De-flake {evidence.test}: chronic pool flake (auto-filed)'
    description = (
        f'The chronic-flake auto-file detector observed {evidence.test} flaky '
        f'{evidence.count} time(s) within the last {evidence.window} '
        f'flaky-ledger-recorded run(s) — at or above the configured chronic '
        f'threshold.\n\n'
        f'Evidence (from data/verify-logs/flaky-ledger.jsonl):\n'
        f'  test:    {evidence.test}\n'
        f'  count:   {evidence.count}\n'
        f'  window:  {evidence.window}\n'
        f'  dates:   {dates_str}\n'
        f'  role(s): {roles_str}\n\n'
        f'{_ROOT_CAUSE_INSTRUCTION}\n\n'
        f'This task was auto-filed by the chronic-flake detector (task 2358) '
        f'after a verify completed. The gate stayed green — retry-once '
        f'already absorbed the failure — so this is non-blocking, visible '
        f'flake debt, not an incident.'
    )
    return {
        'title': title,
        'description': description,
        'priority': 'medium',
        'project_root': str(project_root),
        'metadata': {
            'spawn_context': 'chronic_flake_auto_file',
            'chronic_flake_test': evidence.test,
            'chronic_flake_count': evidence.count,
            'chronic_flake_window': evidence.window,
            'roles': evidence.roles,
        },
    }
