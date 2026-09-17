"""Tests for scripts/fm_wedge_forensics.py — the STDLIB-ONLY forensic analyzer
that turns a `journalctl --user -u fused-memory.service` capture into measured
stall episodes (task 5544).

Every input here is built from REAL lines of the user journal for the episodes
the task was filed for, embedded as a module-level string constant.
Deliberately NOT read from the live journal: the user journal rotates (it
currently reaches back only to 2026-08-24), so both episodes age out and a
journal-driven test would then fail for a reason having nothing to do with this
code. It would also be unrunnable in CI or on any host that never had these
incidents. Same stdin-driven pure-function shape as
scripts/tests/test_recon_busy_check.py, for the same stated reason: the pure
analyze() gets pytest coverage and the CLI gets subprocess coverage, with no
live service involved.

The excerpts are REDUCED (tens of lines, not the ~6700/~8900 the two windows
actually hold), keeping the pre-stall content lines, the silence boundary and
every systemd[...] line. The reduction preserves the real gap structure: every
inter-line gap above 20s in each full window is still present here at its real
magnitude, so the excerpts cannot manufacture — or hide — an episode.

Two inputs go further than reduction and are ASSEMBLED, each saying so at its
own definition and naming what the retained journal does not contain:
UNIT_LEFT_DOWN_JOURNAL (a down-window long enough to register) and
LOGGING_CHILD_JOURNAL (a spawned child that wrote a line of its own). Both
cover branches that a real capture in this retention window cannot reach, and
both use real line texts with only the separation or the pid attribution
changed.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from fm_wedge_forensics import (
    DEFAULT_STALL_THRESHOLD_SECONDS,
    analyze,
    parse_consumed,
)

# ---------------------------------------------------------------------------
# Episode 2 — 2026-09-16 12:04. The decisive one: fm did not service SIGTERM
# for the full TimeoutStopSec and had to be SIGKILLed, and a `git` child was
# still alive in the unit's cgroup when it was.
#
# Real gaps preserved: 171s (12:04:01 -> 12:06:52, the stall), 85s
# (12:06:57 -> 12:08:22) and 30s (12:08:31 -> 12:09:01), the latter two both
# below the default threshold.
# ---------------------------------------------------------------------------
EPISODE_2_JOURNAL = """\
2026-09-16T12:03:51+01:00 leo-MS-7C35 uv[3655756]: 2026-09-16 12:03:51 - fused_memory.reconciliation.harness - INFO - Project reconciliation loop started for dark_factory
2026-09-16T12:03:56+01:00 leo-MS-7C35 uv[3655756]: 2026-09-16 12:03:56 - fused_memory.reconciliation.harness - INFO - reconciliation.run_started
2026-09-16T12:03:59+01:00 leo-MS-7C35 uv[3655756]: 2026-09-16 12:03:59 - fused_memory.reconciliation.index_drift_detector - INFO - reconciliation.index_drift_escalation_suppressed: graph=dark_factory already has an open level-1 index-drift escalation
2026-09-16T12:04:00+01:00 leo-MS-7C35 uv[3655756]: 2026-09-16 12:04:00 - fused_memory.reconciliation.event_buffer - INFO - reconciliation.event_buffered
2026-09-16T12:04:01+01:00 leo-MS-7C35 uv[3655756]: INFO:     127.0.0.1:41736 - "POST /mcp HTTP/1.1" 200 OK
2026-09-16T12:04:01+01:00 leo-MS-7C35 uv[3655756]: 2026-09-16 12:04:01 - mcp.server.streamable_http - INFO - Terminating session: None
2026-09-16T12:06:52+01:00 leo-MS-7C35 systemd[2626]: Stopping fused-memory.service - Fused Memory MCP Server (dark-factory)...
2026-09-16T12:06:56+01:00 leo-MS-7C35 uv[3655756]: 2026-09-16 12:06:56 - __main__ - INFO - Received SIGTERM — initiating operator shutdown
2026-09-16T12:06:57+01:00 leo-MS-7C35 uv[3655756]: 2026-09-16 12:06:57 - fused_memory.reconciliation.harness - INFO - Remediation: 1 actionable findings from run 624be135-f878-4707-a705-35dcec3a986d, triggering second pass
2026-09-16T12:08:22+01:00 leo-MS-7C35 systemd[2626]: fused-memory.service: State 'stop-sigterm' timed out. Killing.
2026-09-16T12:08:22+01:00 leo-MS-7C35 systemd[2626]: fused-memory.service: Killing process 3655692 (uv) with signal SIGKILL.
2026-09-16T12:08:22+01:00 leo-MS-7C35 systemd[2626]: fused-memory.service: Killing process 3655756 (python3) with signal SIGKILL.
2026-09-16T12:08:22+01:00 leo-MS-7C35 systemd[2626]: fused-memory.service: Killing process 1289425 (git) with signal SIGKILL.
2026-09-16T12:08:22+01:00 leo-MS-7C35 systemd[2626]: fused-memory.service: Main process exited, code=killed, status=9/KILL
2026-09-16T12:08:22+01:00 leo-MS-7C35 systemd[2626]: fused-memory.service: Failed with result 'timeout'.
2026-09-16T12:08:22+01:00 leo-MS-7C35 systemd[2626]: fused-memory.service: Consumed 2h 29min 21.259s CPU time, 3.2G memory peak, 934.4M memory swap peak.
2026-09-16T12:08:22+01:00 leo-MS-7C35 systemd[2626]: Starting fused-memory.service - Fused Memory MCP Server (dark-factory)...
2026-09-16T12:08:23+01:00 leo-MS-7C35 docker[1289624]:  Container docker-falkordb-1 Running
2026-09-16T12:08:27+01:00 leo-MS-7C35 uv[1289738]: 2026-09-16 12:08:27 - __main__ - INFO - Fused Memory MCP Server starting
2026-09-16T12:08:31+01:00 leo-MS-7C35 uv[1289738]: 2026-09-16 12:08:31 - __main__ - INFO - idempotent_ops retention prune at startup: 228 rows
2026-09-16T12:09:01+01:00 leo-MS-7C35 uv[1289738]: 2026-09-16 12:09:01 - fused_memory.services.write_journal - INFO - Pruned 130000 write_ops rows (read>30.0d: 130000, search>365.0d: 0, write>730.0d: 0); frees pages for reuse, does not shrink the file
2026-09-16T12:09:03+01:00 leo-MS-7C35 systemd[2626]: Started fused-memory.service - Fused Memory MCP Server (dark-factory).
"""

# ---------------------------------------------------------------------------
# Episode 1 — 2026-09-15 23:55. Same signature, milder outcome: the loop
# recovered enough to run its SIGTERM handler, so systemd stopped it cleanly
# and never had to SIGKILL anything.
#
# Real gaps preserved: 120s (23:55:10 -> 23:57:10, the stall), 44s
# (23:57:10 -> 23:57:54, SIGTERM latency) and 30s (23:58:11 -> 23:58:41).
# ---------------------------------------------------------------------------
EPISODE_1_JOURNAL = """\
2026-09-15T23:55:07+01:00 leo-MS-7C35 uv[2905995]: 2026-09-15 23:55:07 - mcp.server.lowlevel.server - INFO - Processing request of type CallToolRequest
2026-09-15T23:55:08+01:00 leo-MS-7C35 uv[2905995]: 2026-09-15 23:55:08 - fused_memory.reconciliation.harness - INFO - reconciliation.run_started
2026-09-15T23:55:08+01:00 leo-MS-7C35 uv[2905995]: 2026-09-15 23:55:08 - fused_memory.reconciliation.index_drift_detector - INFO - reconciliation.index_drift_escalation_suppressed: graph=dark_factory already has an open level-1 index-drift escalation
2026-09-15T23:55:10+01:00 leo-MS-7C35 uv[2905995]: 2026-09-15 23:55:10 - httpx - INFO - HTTP Request: POST http://localhost:6333/collections/fused_solar_challenge/points/count "HTTP/1.1 200 OK"
2026-09-15T23:57:10+01:00 leo-MS-7C35 systemd[2626]: Stopping fused-memory.service - Fused Memory MCP Server (dark-factory)...
2026-09-15T23:57:54+01:00 leo-MS-7C35 uv[2905995]: 2026-09-15 23:57:54 - __main__ - INFO - Received SIGTERM — initiating operator shutdown
2026-09-15T23:58:00+01:00 leo-MS-7C35 uv[2905995]: asyncio.exceptions.CancelledError
2026-09-15T23:58:00+01:00 leo-MS-7C35 systemd[2626]: Stopped fused-memory.service - Fused Memory MCP Server (dark-factory).
2026-09-15T23:58:00+01:00 leo-MS-7C35 systemd[2626]: fused-memory.service: Consumed 1h 22min 57.690s CPU time, 3.3G memory peak, 529.9M memory swap peak.
2026-09-15T23:58:03+01:00 leo-MS-7C35 systemd[2626]: Starting fused-memory.service - Fused Memory MCP Server (dark-factory)...
2026-09-15T23:58:03+01:00 leo-MS-7C35 docker[1761328]:  Container docker-qdrant-1 Running
2026-09-15T23:58:11+01:00 leo-MS-7C35 uv[1761597]: 2026-09-15 23:58:11 - __main__ - INFO - idempotent_ops retention prune at startup: 129 rows
2026-09-15T23:58:41+01:00 leo-MS-7C35 uv[1761597]: 2026-09-15 23:58:41 - fused_memory.services.write_journal - INFO - Pruned 135000 write_ops rows (read>30.0d: 135000, search>365.0d: 0, write>730.0d: 0); frees pages for reuse, does not shrink the file
2026-09-15T23:58:42+01:00 leo-MS-7C35 systemd[2626]: Started fused-memory.service - Fused Memory MCP Server (dark-factory).
"""

# The measured NORMAL quiet ceiling: fm's reconciliation harness logs one
# `Project reconciliation loop started` line per project on a 61s cadence, so
# an otherwise-idle fm legitimately goes 61s between lines. These are the six
# real consecutive dark_factory lines from 2026-09-16 11:58-12:03, every gap
# exactly 61s. This is the upper bound the default threshold has to clear.
NORMAL_HARNESS_CADENCE_JOURNAL = """\
2026-09-16T11:58:46+01:00 leo-MS-7C35 uv[3655756]: 2026-09-16 11:58:46 - fused_memory.reconciliation.harness - INFO - Project reconciliation loop started for dark_factory
2026-09-16T11:59:47+01:00 leo-MS-7C35 uv[3655756]: 2026-09-16 11:59:47 - fused_memory.reconciliation.harness - INFO - Project reconciliation loop started for dark_factory
2026-09-16T12:00:48+01:00 leo-MS-7C35 uv[3655756]: 2026-09-16 12:00:48 - fused_memory.reconciliation.harness - INFO - Project reconciliation loop started for dark_factory
2026-09-16T12:01:49+01:00 leo-MS-7C35 uv[3655756]: 2026-09-16 12:01:49 - fused_memory.reconciliation.harness - INFO - Project reconciliation loop started for dark_factory
2026-09-16T12:02:50+01:00 leo-MS-7C35 uv[3655756]: 2026-09-16 12:02:50 - fused_memory.reconciliation.harness - INFO - Project reconciliation loop started for dark_factory
2026-09-16T12:03:51+01:00 leo-MS-7C35 uv[3655756]: 2026-09-16 12:03:51 - fused_memory.reconciliation.harness - INFO - Project reconciliation loop started for dark_factory
"""

# A THIRD real stall, 2026-09-16 12:16:07 -> 12:18:47 = 160s, seven minutes
# after episode 2's restart. Nothing stopped the unit: the loop simply resumed
# on its own and systemd never acted. Six such stalls were measured in the
# 12:10-18:00 window alone (160s, 100s, 113s, 106s, 116s, 136s), so this is
# the COMMON outcome, not a curiosity — the two episodes the task was filed
# for are the minority that the watchdog happened to catch.
SELF_RECOVERED_STALL_JOURNAL = """\
2026-09-16T12:16:02+01:00 leo-MS-7C35 uv[1289738]: 2026-09-16 12:16:02 - fused_memory.reconciliation.harness - INFO - Project reconciliation loop started for solar_challenge_platform
2026-09-16T12:16:06+01:00 leo-MS-7C35 uv[1289738]: 2026-09-16 12:16:06 - fused_memory.reconciliation.event_buffer - INFO - reconciliation.event_buffered
2026-09-16T12:16:07+01:00 leo-MS-7C35 uv[1289738]: 2026-09-16 12:16:07 - fused_memory.reconciliation.stages.task_knowledge_sync - WARNING - reconciliation.done_provenance_section_truncated
2026-09-16T12:18:47+01:00 leo-MS-7C35 uv[1289738]: 2026-09-16 12:18:47 - __main__ - INFO - thread_monitor: threads=33 delta=-1
2026-09-16T12:18:47+01:00 leo-MS-7C35 uv[1289738]: 2026-09-16 12:18:47 - shared.usage_gate - INFO - Account max-b: firing probe #1
"""

# A FOURTH real stall, 2026-09-16 16:15:05 -> 16:17:21 = 136s, the one the
# watchdog did catch in that window. Carried here for its `Consumed` line,
# which uses systemd's plain-`Ymin` CPU-time spelling rather than the two
# episodes' `Xh Ymin` one — a real instance of the other form, not a
# hand-written variant of it.
WATCHDOG_CAUGHT_STALL_JOURNAL = """\
2026-09-16T16:15:05+01:00 leo-MS-7C35 uv[3161215]: 2026-09-16 16:15:05 - mcp.server.streamable_http - INFO - Terminating session: None
2026-09-16T16:15:05+01:00 leo-MS-7C35 uv[3161215]: 2026-09-16 16:15:05 - httpx - INFO - HTTP Request: POST http://localhost:6333/collections/fused_solar_challenge/points/count "HTTP/1.1 200 OK"
2026-09-16T16:17:21+01:00 leo-MS-7C35 systemd[2626]: Stopping fused-memory.service - Fused Memory MCP Server (dark-factory)...
2026-09-16T16:17:43+01:00 leo-MS-7C35 uv[3161215]: 2026-09-16 16:17:43 - __main__ - INFO - Received SIGTERM — initiating operator shutdown
2026-09-16T16:18:03+01:00 leo-MS-7C35 systemd[2626]: Stopped fused-memory.service - Fused Memory MCP Server (dark-factory).
2026-09-16T16:18:03+01:00 leo-MS-7C35 systemd[2626]: fused-memory.service: Consumed 20min 30.697s CPU time, 2.3G memory peak, 403.3M memory swap peak.
2026-09-16T16:18:03+01:00 leo-MS-7C35 systemd[2626]: Starting fused-memory.service - Fused Memory MCP Server (dark-factory)...
2026-09-16T16:18:44+01:00 leo-MS-7C35 systemd[2626]: Started fused-memory.service - Fused Memory MCP Server (dark-factory).
"""

# The same 160s self-recovered stall, then an UNRELATED restart. Two real,
# non-adjacent stretches of one capture: the ~2h49m of dense logging between
# them is elided, which is why this input shows two silences where the capture
# shows one.
#
# That elision does NOT cost the fixture its teeth — it IS the executable guard
# on restart_sequence_after's `_TEARDOWN_START` bound. Measured by deleting
# that bound and re-running analyze() on this exact input: stall 1 flips to
# `stopped-on-signal` with teardown_seconds == 3.0 and total_seconds ==
# 10300.0, the three wrong numbers the test below names. Do not treat this
# fixture as decorative.
SELF_RECOVERED_THEN_UNRELATED_RESTART_JOURNAL = """\
2026-09-16T12:16:07+01:00 leo-MS-7C35 uv[1289738]: 2026-09-16 12:16:07 - fused_memory.reconciliation.stages.task_knowledge_sync - WARNING - reconciliation.done_provenance_section_truncated
2026-09-16T12:18:47+01:00 leo-MS-7C35 uv[1289738]: 2026-09-16 12:18:47 - __main__ - INFO - thread_monitor: threads=33 delta=-1
2026-09-16T15:07:03+01:00 leo-MS-7C35 systemd[2626]: Stopping fused-memory.service - Fused Memory MCP Server (dark-factory)...
2026-09-16T15:07:06+01:00 leo-MS-7C35 systemd[2626]: Stopped fused-memory.service - Fused Memory MCP Server (dark-factory).
2026-09-16T15:07:06+01:00 leo-MS-7C35 systemd[2626]: fused-memory.service: Consumed 29min 6.600s CPU time, 2.7G memory peak, 628.5M memory swap peak.
2026-09-16T15:07:06+01:00 leo-MS-7C35 systemd[2626]: Starting fused-memory.service - Fused Memory MCP Server (dark-factory)...
2026-09-16T15:07:47+01:00 leo-MS-7C35 systemd[2626]: Started fused-memory.service - Fused Memory MCP Server (dark-factory).
"""

# ASSEMBLED, not captured: the real 15:07 stop sequence above with the restart
# pushed out, so the silence begins with the unit already DOWN rather than
# wedged. Only the separation is synthetic; every line text is real.
#
# Why no capture can supply this. fused-memory is restarted rather than left
# down, and measured across 2026-09-14 → 2026-09-17 every `Stopped
# fused-memory.service` in the retained journal is followed by `Starting`
# within 0–3s — far below any workable stall threshold — so the whole
# down-window branch is unreachable from this host's real captures. The branch
# is still load-bearing: the orchestrator-watchdog's own journal recorded
# `port-down` for episode 1's third probe at 23:58:03, taken while fm sat
# between `Stopped` (23:58:00) and `Started` (23:58:42).
UNIT_LEFT_DOWN_JOURNAL = """\
2026-09-16T15:07:03+01:00 leo-MS-7C35 systemd[2626]: Stopping fused-memory.service - Fused Memory MCP Server (dark-factory)...
2026-09-16T15:07:06+01:00 leo-MS-7C35 systemd[2626]: Stopped fused-memory.service - Fused Memory MCP Server (dark-factory).
2026-09-16T15:07:06+01:00 leo-MS-7C35 systemd[2626]: fused-memory.service: Consumed 29min 6.600s CPU time, 2.7G memory peak, 628.5M memory swap peak.
2026-09-16T15:11:06+01:00 leo-MS-7C35 systemd[2626]: Starting fused-memory.service - Fused Memory MCP Server (dark-factory)...
2026-09-16T15:11:47+01:00 leo-MS-7C35 systemd[2626]: Started fused-memory.service - Fused Memory MCP Server (dark-factory).
"""

# ASSEMBLED, not captured: episode 2 reduced to its teardown, plus ONE added
# line — a git error attributed to the pid systemd later SIGKILLs as `git`.
# A spawned child inherits the unit's stderr, so its output is journaled under
# this same unit; that is the shape any pid- or name-based exclusion silently
# drops, taking the single most decisive signal in this diagnosis with it.
#
# The added line is the only synthetic element, and it is synthetic because the
# hazard is LATENT here rather than realized: measured over the real 12:03–12:10
# capture (6686 lines), pid 1289425 appears exactly once, in the `Killing
# process 1289425 (git)` line itself — the child wrote nothing. Covering a
# branch a capture cannot reach is the whole reason this constant exists.
LOGGING_CHILD_JOURNAL = """\
2026-09-16T12:04:01+01:00 leo-MS-7C35 uv[3655756]: 2026-09-16 12:04:01 - mcp.server.streamable_http - INFO - Terminating session: None
2026-09-16T12:06:52+01:00 leo-MS-7C35 systemd[2626]: Stopping fused-memory.service - Fused Memory MCP Server (dark-factory)...
2026-09-16T12:06:57+01:00 leo-MS-7C35 uv[1289425]: error: cannot lock ref 'refs/heads/main': Unable to create '/home/leo/src/dark-factory/.git/refs/heads/main.lock': File exists.
2026-09-16T12:08:22+01:00 leo-MS-7C35 systemd[2626]: fused-memory.service: State 'stop-sigterm' timed out. Killing.
2026-09-16T12:08:22+01:00 leo-MS-7C35 systemd[2626]: fused-memory.service: Killing process 3655756 (python3) with signal SIGKILL.
2026-09-16T12:08:22+01:00 leo-MS-7C35 systemd[2626]: fused-memory.service: Killing process 1289425 (git) with signal SIGKILL.
2026-09-16T12:08:22+01:00 leo-MS-7C35 systemd[2626]: Starting fused-memory.service - Fused Memory MCP Server (dark-factory)...
2026-09-16T12:09:03+01:00 leo-MS-7C35 systemd[2626]: Started fused-memory.service - Fused Memory MCP Server (dark-factory).
"""

# A healthy busy window: real consecutive lines from 2026-09-16 12:03, where fm
# emits hundreds of lines per second. Nothing here is near any threshold.
DENSE_HEALTHY_JOURNAL = """\
2026-09-16T12:03:55+01:00 leo-MS-7C35 uv[3655756]: 2026-09-16 12:03:55 - mcp.server.lowlevel.server - INFO - Processing request of type CallToolRequest
2026-09-16T12:03:55+01:00 leo-MS-7C35 uv[3655756]: 2026-09-16 12:03:55 - mcp.server.streamable_http - INFO - Terminating session: None
2026-09-16T12:03:56+01:00 leo-MS-7C35 uv[3655756]: 2026-09-16 12:03:56 - fused_memory.reconciliation.harness - INFO - reconciliation.run_started
2026-09-16T12:03:57+01:00 leo-MS-7C35 uv[3655756]: 2026-09-16 12:03:57 - mcp.server.lowlevel.server - INFO - Processing request of type CallToolRequest
2026-09-16T12:03:58+01:00 leo-MS-7C35 uv[3655756]: 2026-09-16 12:03:58 - mcp.server.streamable_http - INFO - Terminating session: None
"""


def test_analyze_finds_the_episode_2_stall_between_its_bounding_log_lines():
    """The stall is the gap between the LAST line before the silence and the
    FIRST line after it — both measured off the journal, not assumed."""
    episodes = analyze(EPISODE_2_JOURNAL)

    assert len(episodes) == 1
    episode = episodes[0]
    assert episode.stall_started_at.isoformat() == "2026-09-16T12:04:01+01:00"
    assert episode.stall_ended_at.isoformat() == "2026-09-16T12:06:52+01:00"
    assert episode.stall_seconds == 171.0


def test_analyze_reports_no_episode_for_a_dense_healthy_window():
    assert analyze(DENSE_HEALTHY_JOURNAL) == []


# ---------------------------------------------------------------------------
# The default threshold, pinned against the three measurements that derive it.
#
# The bound is checkable here rather than merely asserted: it must sit ABOVE
# the measured 61s normal harness cadence and BELOW the shorter (120s) of the
# two observed stalls. These tests fail if either side of that margin is ever
# eroded, whichever direction the constant is moved.
# ---------------------------------------------------------------------------

def test_default_threshold_clears_the_normal_61s_harness_cadence():
    """An idle-but-healthy fm must not read as a stall."""
    assert analyze(NORMAL_HARNESS_CADENCE_JOURNAL) == []


def test_default_threshold_catches_the_shorter_episode_1_stall():
    episodes = analyze(EPISODE_1_JOURNAL)

    assert len(episodes) == 1
    assert episodes[0].stall_seconds == 120.0


def test_default_threshold_catches_the_longer_episode_2_stall():
    episodes = analyze(EPISODE_2_JOURNAL)

    assert len(episodes) == 1
    assert episodes[0].stall_seconds == 171.0


def test_threshold_is_a_parameter_not_baked_into_the_detection():
    """Lowering the threshold below the harness cadence makes the same healthy
    window register — proving the default is a tunable bound and the detection
    logic carries no hidden magnitude of its own."""
    episodes = analyze(NORMAL_HARNESS_CADENCE_JOURNAL, 30.0)

    assert len(episodes) == 5
    assert {episode.stall_seconds for episode in episodes} == {61.0}


def test_default_threshold_sits_between_the_cadence_and_the_shorter_stall():
    """The derivation itself, asserted directly on the constant.

    The two behavioural tests above check the default's CONSEQUENCES on the
    real captures; this checks the MARGIN, which is what a future tuner
    actually needs to preserve. 61.0 is the measured normal per-project
    harness cadence and 120.0 the shorter of the two observed stalls, so any
    value strictly between them separates signal from healthy idle. Naming
    both endpoints here means moving the constant toward either one fails
    loudly at the boundary rather than silently eroding the margin.
    """
    assert 61.0 < DEFAULT_STALL_THRESHOLD_SECONDS < 120.0


# ---------------------------------------------------------------------------
# What the loop was doing when systemd tried to stop it.
#
# asyncio signal handlers run ON the event loop, so a process that cannot
# service SIGTERM for the whole stop timeout has a BLOCKED loop — not a merely
# descheduled one, which would be scheduled and exit well inside it. That
# distinction is the single most load-bearing observation in this diagnosis,
# so it is a classification the analyzer reports rather than something a
# reader has to re-derive from the systemd lines by eye.
# ---------------------------------------------------------------------------

def test_episode_2_never_serviced_sigterm_so_the_loop_was_still_blocked():
    episode = analyze(EPISODE_2_JOURNAL)[0]

    assert episode.outcome == "sigterm-unserviced"


def test_episode_1_stopped_cleanly_so_the_loop_ran_its_signal_handler():
    """Same stall signature, milder outcome: `Stopping` -> `Stopped` with no
    stop-sigterm timeout, so the loop recovered enough to service the signal."""
    episode = analyze(EPISODE_1_JOURNAL)[0]

    assert episode.outcome == "stopped-on-signal"


def test_a_stall_with_no_systemd_stop_at_all_is_self_recovered():
    episode = analyze(SELF_RECOVERED_STALL_JOURNAL)[0]

    assert episode.stall_seconds == 160.0
    assert episode.outcome == "self-recovered"


def test_every_stall_reports_the_watchdog_verdict_its_probe_would_have_returned():
    """The process is up and its socket stays bound throughout a stall, so
    scripts/orchestrator-watchdog.py's probe_port passes while the /alive fetch
    served by the same blocked loop does not — which is its 'wedged' verdict.
    Reported in the watchdog's own word so the output collates with its
    journal lines instead of needing a translation step."""
    for journal in (EPISODE_1_JOURNAL, EPISODE_2_JOURNAL, SELF_RECOVERED_STALL_JOURNAL):
        assert analyze(journal)[0].watchdog_verdict == "wedged"


def test_a_silence_that_begins_with_the_unit_already_down_reports_port_down():
    """The other half of the verdict, and the half the diagnosis leans on: a
    silence whose last preceding line is the unit's own stop accounting is a
    DOWN window, so the probe never reached a bound socket at all. The outcome
    vocabulary describes how a blocked loop answered a signal and says nothing
    useful here; this verdict is what tells a reader the silence was never a
    stall.

    Unpinned, both this reading and _UNIT_DOWN_MARKERS are free: returning
    'wedged' unconditionally leaves the rest of the suite green.
    """
    episodes = analyze(UNIT_LEFT_DOWN_JOURNAL)

    assert len(episodes) == 1
    assert episodes[0].stall_seconds == 240.0
    assert episodes[0].watchdog_verdict == "port-down"


# ---------------------------------------------------------------------------
# The three pieces of evidence that made this diagnosis possible by hand.
#
# Each took a separate ad-hoc `journalctl | grep` pass to recover the first
# time. Extracting them is most of why this script exists: episode 3 should
# cost one command, not an afternoon.
# ---------------------------------------------------------------------------

def test_pre_stall_context_shows_what_the_loop_was_doing_when_it_went_quiet():
    context = analyze(EPISODE_2_JOURNAL)[0].pre_stall_context

    assert any("reconciliation.run_started" in line for line in context)
    assert any("index_drift_escalation_suppressed" in line for line in context)
    assert not any("Stopping fused-memory" in line for line in context)


def test_context_depth_is_a_parameter_not_baked_into_the_extraction():
    """A full window buries the application's own loggers under hundreds of
    access lines per second, so the depth has to be dialable. Same shape as the
    threshold test: pass a non-default and the output follows it."""
    context = analyze(EPISODE_2_JOURNAL, context_lines=2)[0].pre_stall_context

    assert len(context) == 2
    assert context[-1].endswith("Terminating session: None")


def test_consumed_line_is_parsed_into_separate_resource_fields():
    """`Xh Ymin Z.Zs` CPU time plus G/M sizes, as structured numbers rather
    than a string the reader has to re-parse."""
    resources = analyze(EPISODE_2_JOURNAL)[0].resources

    assert resources is not None
    assert resources.cpu_seconds == 2 * 3600 + 29 * 60 + 21.259
    assert resources.memory_peak_bytes == 3.2 * 1024**3
    assert resources.swap_peak_bytes == 934.4 * 1024**2


def test_consumed_line_parses_both_systemd_cpu_time_spellings():
    """Episode 1 carries `1h 22min 57.690s`; the 16:15 stall carries systemd's
    hourless `20min 30.697s`. Both are real lines, not constructed variants."""
    with_hours = analyze(EPISODE_1_JOURNAL)[0].resources
    without_hours = analyze(WATCHDOG_CAUGHT_STALL_JOURNAL)[0].resources

    assert with_hours is not None and without_hours is not None
    assert with_hours.cpu_seconds == 3600 + 22 * 60 + 57.690
    assert with_hours.swap_peak_bytes == 529.9 * 1024**2
    assert without_hours.cpu_seconds == 20 * 60 + 30.697
    assert without_hours.memory_peak_bytes == 2.3 * 1024**3


def test_a_consumed_line_without_memory_accounting_keeps_its_cpu_time():
    """systemd prints CPU time alone for a unit with memory accounting off —
    real on this host, e.g. `df-verify-reify-4ae45bbd-6a64e60bb1f4.scope:
    Consumed 8min 25.125s CPU time.` at 2026-09-16 12:11:40. Requiring the
    memory clause would throw away a measurement systemd took and report the
    whole line as unmeasured."""
    totals = parse_consumed(
        "2026-09-16T12:11:40+01:00 leo-MS-7C35 systemd[2626]: "
        "df-verify-reify-4ae45bbd-6a64e60bb1f4.scope: Consumed 8min 25.125s CPU time."
    )

    assert totals is not None
    assert totals.cpu_seconds == 8 * 60 + 25.125
    assert totals.memory_peak_bytes is None
    assert totals.swap_peak_bytes is None


def test_a_measured_zero_is_not_reported_as_unmeasured():
    """systemd renders a zero with a bare `B` suffix (`0B memory swap peak`,
    real at 2026-09-16 10:13:35). Dropping `B` from the size units would turn
    "measured, and it was zero" into "not measured" — the same absence-of-
    evidence confusion this module refuses everywhere else."""
    totals = parse_consumed(
        "2026-09-16T10:13:35+01:00 leo-MS-7C35 systemd[2626]: reify-warm-lane-gc.service: "
        "Consumed 16min 24.123s CPU time, 1.1G memory peak, 0B memory swap peak."
    )

    assert totals is not None
    assert totals.memory_peak_bytes == 1.1 * 1024**3
    assert totals.swap_peak_bytes == 0.0


def test_a_stall_with_no_consumed_line_reports_absent_not_zero():
    """A self-recovered stall never produced a `Consumed` line, and reporting
    it as 0 would read as "consumed nothing" — the opposite of the truth."""
    assert analyze(SELF_RECOVERED_STALL_JOURNAL)[0].resources is None


def test_every_sigkilled_process_is_reported_with_the_unit_s_own_annotated():
    """The `git` child still alive at teardown is the evidence that a
    synchronous git subprocess was in flight on the blocked loop. Every kill is
    listed, with the unit's own `uv`/`python3` entries annotated rather than
    removed, so the child stands out without any of them being hidden."""
    killed = analyze(EPISODE_2_JOURNAL)[0].killed_processes

    assert [(process.pid, process.name) for process in killed] == [
        (3655692, "uv"),
        (3655756, "python3"),
        (1289425, "git"),
    ]
    assert [process.name for process in killed if not process.wrote_journal_lines] == ["git"]


def test_a_child_that_wrote_a_journal_line_is_still_reported():
    """The suppression this guards against. A child inherits the unit's stderr,
    so writing even one line credits it with journal output — and a filter keyed
    on that would drop the `git` entry entirely, leaving an empty list that
    reads as "nothing outlived the stop". The annotation records the credit; it
    must not gate the report."""
    killed = analyze(LOGGING_CHILD_JOURNAL)[0].killed_processes
    child = [process for process in killed if process.name == "git"]

    assert [(process.pid, process.name) for process in child] == [(1289425, "git")]
    assert child[0].wrote_journal_lines is True


def test_an_episode_that_killed_nothing_reports_no_processes():
    assert analyze(EPISODE_1_JOURNAL)[0].killed_processes == ()


# ---------------------------------------------------------------------------
# What an episode actually cost, decomposed from its own observed timestamps.
#
# Measured, never predicted from configuration: episode 2's 90s teardown is
# an IGNORED-SIGTERM timeout, which only the observed
# Stopping -> stop-sigterm-timed-out -> Killing sequence reveals. An estimate
# derived from the unit file's TimeoutStopSec would have assumed a clean stop
# and understated the dominant term by design.
# ---------------------------------------------------------------------------

def test_episode_2_cost_decomposes_into_teardown_startup_and_total():
    costs = analyze(EPISODE_2_JOURNAL)[0].costs

    assert costs.teardown_seconds == 90.0
    assert costs.startup_seconds == 41.0
    assert costs.total_seconds == 302.0


def test_episode_2_teardown_dominates_startup():
    """Of the two recovery terms this capture can witness, teardown is the
    larger — 90s of ignored SIGTERM against a 41s start.

    Deliberately NOT an answer to "should the port-down streak be shortened?".
    The streak is logged in the watchdog's journal, so detection_seconds is
    structurally unavailable here and dominant_recovery_term cannot see it; the
    measured streak (121s) in fact exceeds this term. That comparison belongs
    to plans/fused-memory-wedge-diagnosis.md, which has both journals."""
    assert analyze(EPISODE_2_JOURNAL)[0].costs.dominant_recovery_term == "teardown"


def test_episode_1_cost_decomposes_from_its_own_clean_stop():
    costs = analyze(EPISODE_1_JOURNAL)[0].costs

    assert costs.teardown_seconds == 50.0
    assert costs.startup_seconds == 39.0


def test_detection_interval_is_reported_unavailable_not_fabricated():
    """The watchdog's consecutive-failure streak is logged in the WATCHDOG's
    journal, not fused-memory's, so no fm-only capture can witness it. Absent
    rather than guessed."""
    for journal in (EPISODE_1_JOURNAL, EPISODE_2_JOURNAL):
        assert analyze(journal)[0].costs.detection_seconds is None


def test_a_self_recovered_stall_has_no_teardown_or_startup_cost():
    """Nothing was torn down or started, so those terms are absent — not zero,
    which would read as an instantaneous restart that never happened."""
    costs = analyze(SELF_RECOVERED_STALL_JOURNAL)[0].costs

    assert costs.teardown_seconds is None
    assert costs.startup_seconds is None
    assert costs.dominant_recovery_term is None


def test_a_later_unrelated_restart_is_not_attributed_to_a_self_recovered_stall():
    """Found by running the analyzer over the real 12:10-18:00 capture rather
    than the excerpts: the 160s stall at 12:16 self-recovered, and the next
    restart was 2h51m later and unconnected. Attributing it produced a
    'stopped-on-signal' verdict, a 3s teardown and a 10300s total — three
    wrong numbers a reader would have had no way to doubt."""
    episodes = analyze(SELF_RECOVERED_THEN_UNRELATED_RESTART_JOURNAL)

    self_recovered, ended_by_restart = episodes
    assert self_recovered.stall_seconds == 160.0
    assert self_recovered.outcome == "self-recovered"
    assert self_recovered.costs.teardown_seconds is None
    assert self_recovered.costs.total_seconds is None
    assert self_recovered.resources is None
    # The restart belongs to the silence it ENDED, and to that one only.
    assert ended_by_restart.outcome == "stopped-on-signal"
    assert ended_by_restart.costs.teardown_seconds == 3.0
    assert ended_by_restart.resources is not None
    assert ended_by_restart.resources.cpu_seconds == 29 * 60 + 6.600


# ---------------------------------------------------------------------------
# CLI (reads the journal from stdin) — driven via subprocess.run.
#
# Same shape as scripts/tests/test_recon_busy_check.py: SCRIPT resolved from
# __file__, and a bounded per-subprocess budget with an env override, because
# concurrent orchestrator agents can push interpreter startup past a tight one
# even when the CLI is behaving. Written compactly rather than copied from
# that file verbatim — a third copy of its 25-line resolver would be the SPOT
# problem its own comment already records as an open shared-helper question.
# ---------------------------------------------------------------------------

SCRIPT = Path(__file__).parent.parent / "fm_wedge_forensics.py"
_CLI_TIMEOUT = float(os.environ.get("FM_WEDGE_FORENSICS_TEST_TIMEOUT", "").strip() or 60.0)


def _run_cli(stdin_text: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["python3", str(SCRIPT), *args],
        input=stdin_text,
        capture_output=True,
        text=True,
        timeout=_CLI_TIMEOUT,
    )


def test_cli_reports_the_stall_duration_and_its_classification():
    result = _run_cli(EPISODE_2_JOURNAL)

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "171" in result.stdout
    assert "sigterm-unserviced" in result.stdout


def test_cli_json_carries_the_same_episode_records():
    result = _run_cli(EPISODE_2_JOURNAL, "--json")

    assert result.returncode == 0, f"stderr={result.stderr!r}"
    episodes = json.loads(result.stdout)["episodes"]
    assert len(episodes) == 1
    assert episodes[0]["stall_seconds"] == 171.0
    assert episodes[0]["outcome"] == "sigterm-unserviced"
    assert episodes[0]["costs"]["teardown_seconds"] == 90.0
    assert episodes[0]["killed_processes"][-1] == {
        "pid": 1289425,
        "name": "git",
        "wrote_journal_lines": False,
    }


def test_cli_threshold_flag_overrides_the_default():
    result = _run_cli(NORMAL_HARNESS_CADENCE_JOURNAL, "--json", "--threshold", "30")

    assert result.returncode == 0, f"stderr={result.stderr!r}"
    assert len(json.loads(result.stdout)["episodes"]) == 5


def test_cli_context_flag_overrides_the_default():
    result = _run_cli(EPISODE_2_JOURNAL, "--json", "--context", "2")

    assert result.returncode == 0, f"stderr={result.stderr!r}"
    assert len(json.loads(result.stdout)["episodes"][0]["pre_stall_context"]) == 2


def test_cli_reports_an_uneventful_window_as_zero_episodes_and_exits_0():
    """A window with nothing in it is a legitimate answer, not an error."""
    result = _run_cli(DENSE_HEALTHY_JOURNAL, "--json")

    assert result.returncode == 0, f"stderr={result.stderr!r}"
    assert json.loads(result.stdout)["episodes"] == []


def test_cli_says_so_in_words_when_a_window_holds_no_stall():
    """The --json shape above is for a pipeline; this is what an operator
    mid-incident actually reads, and "nothing here" has to be legible as an
    answer rather than as empty output."""
    result = _run_cli(DENSE_HEALTHY_JOURNAL)

    assert result.returncode == 0, f"stderr={result.stderr!r}"
    assert "no stalls above the threshold" in result.stdout


def test_cli_human_report_marks_absent_terms_as_not_measured():
    """A self-recovered stall has no teardown and no startup, and the report
    has to say NOT MEASURED rather than print a zero an operator would read as
    an instantaneous restart."""
    result = _run_cli(SELF_RECOVERED_STALL_JOURNAL)

    assert result.returncode == 0, f"stderr={result.stderr!r}"
    assert "teardown         not measured" in result.stdout
    assert "startup          not measured" in result.stdout
    assert "self-recovered" in result.stdout


def test_cli_fails_loudly_when_nothing_in_the_input_parses():
    """The trap this exists to catch: system-scope `journalctl -u
    fused-memory.service` prints "-- No entries --" because the unit is a
    systemd --user unit. Reporting that as "0 episodes" would hand back a
    clean bill of health for a capture that contains no evidence at all."""
    result = _run_cli("-- No entries --\n")

    assert result.returncode != 0
    assert "--user" in result.stdout + result.stderr


def test_cli_empty_stdin_also_fails_loudly():
    result = _run_cli("")

    assert result.returncode != 0
