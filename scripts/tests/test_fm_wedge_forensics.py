"""Tests for scripts/fm_wedge_forensics.py — the STDLIB-ONLY forensic analyzer
that turns a `journalctl --user -u fused-memory.service` capture into measured
stall episodes (task 5544).

Every input here is an EXCERPT of the real user journal for the two episodes
the task was filed for, embedded as a module-level string constant. Deliberately
NOT read from the live journal: the user journal rotates (it currently reaches
back only to 2026-08-24), so both episodes age out and a journal-driven test
would then fail for a reason having nothing to do with this code. It would also
be unrunnable in CI or on any host that never had these incidents. Same
stdin-driven pure-function shape as scripts/tests/test_recon_busy_check.py,
for the same stated reason: the pure analyze() gets pytest coverage and the CLI
gets subprocess coverage, with no live service involved.

The excerpts are REDUCED (tens of lines, not the ~6700/~8900 the two windows
actually hold), keeping the pre-stall content lines, the silence boundary and
every systemd[...] line. The reduction preserves the real gap structure: every
inter-line gap above 20s in each full window is still present here at its real
magnitude, so the excerpts cannot manufacture — or hide — an episode.
"""
from __future__ import annotations

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
    from fm_wedge_forensics import analyze

    episodes = analyze(EPISODE_2_JOURNAL)

    assert len(episodes) == 1
    episode = episodes[0]
    assert episode.stall_started_at.isoformat() == "2026-09-16T12:04:01+01:00"
    assert episode.stall_ended_at.isoformat() == "2026-09-16T12:06:52+01:00"
    assert episode.stall_seconds == 171.0


def test_analyze_reports_no_episode_for_a_dense_healthy_window():
    from fm_wedge_forensics import analyze

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
    from fm_wedge_forensics import analyze

    assert analyze(NORMAL_HARNESS_CADENCE_JOURNAL) == []


def test_default_threshold_catches_the_shorter_episode_1_stall():
    from fm_wedge_forensics import analyze

    episodes = analyze(EPISODE_1_JOURNAL)

    assert len(episodes) == 1
    assert episodes[0].stall_seconds == 120.0


def test_default_threshold_catches_the_longer_episode_2_stall():
    from fm_wedge_forensics import analyze

    episodes = analyze(EPISODE_2_JOURNAL)

    assert len(episodes) == 1
    assert episodes[0].stall_seconds == 171.0


def test_threshold_is_a_parameter_not_baked_into_the_detection():
    """Lowering the threshold below the harness cadence makes the same healthy
    window register — proving the default is a tunable bound and the detection
    logic carries no hidden magnitude of its own."""
    from fm_wedge_forensics import analyze

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
    from fm_wedge_forensics import DEFAULT_STALL_THRESHOLD_SECONDS

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
    from fm_wedge_forensics import analyze

    episode = analyze(EPISODE_2_JOURNAL)[0]

    assert episode.outcome == "sigterm-unserviced"


def test_episode_1_stopped_cleanly_so_the_loop_ran_its_signal_handler():
    """Same stall signature, milder outcome: `Stopping` -> `Stopped` with no
    stop-sigterm timeout, so the loop recovered enough to service the signal."""
    from fm_wedge_forensics import analyze

    episode = analyze(EPISODE_1_JOURNAL)[0]

    assert episode.outcome == "stopped-on-signal"


def test_a_stall_with_no_systemd_stop_at_all_is_self_recovered():
    from fm_wedge_forensics import analyze

    episode = analyze(SELF_RECOVERED_STALL_JOURNAL)[0]

    assert episode.stall_seconds == 160.0
    assert episode.outcome == "self-recovered"


def test_every_stall_reports_the_watchdog_verdict_its_probe_would_have_returned():
    """The process is up and its socket stays bound throughout a stall, so
    scripts/orchestrator-watchdog.py's probe_port passes while the /alive fetch
    served by the same blocked loop does not — which is its 'wedged' verdict.
    Reported in the watchdog's own word so the output collates with its
    journal lines instead of needing a translation step."""
    from fm_wedge_forensics import analyze

    for journal in (EPISODE_1_JOURNAL, EPISODE_2_JOURNAL, SELF_RECOVERED_STALL_JOURNAL):
        assert analyze(journal)[0].watchdog_verdict == "wedged"


# ---------------------------------------------------------------------------
# The three pieces of evidence that made this diagnosis possible by hand.
#
# Each took a separate ad-hoc `journalctl | grep` pass to recover the first
# time. Extracting them is most of why this script exists: episode 3 should
# cost one command, not an afternoon.
# ---------------------------------------------------------------------------

def test_pre_stall_context_shows_what_the_loop_was_doing_when_it_went_quiet():
    from fm_wedge_forensics import analyze

    context = analyze(EPISODE_2_JOURNAL)[0].pre_stall_context

    assert any("reconciliation.run_started" in line for line in context)
    assert any("index_drift_escalation_suppressed" in line for line in context)
    assert not any("Stopping fused-memory" in line for line in context)


def test_consumed_line_is_parsed_into_separate_resource_fields():
    """`Xh Ymin Z.Zs` CPU time plus G/M sizes, as structured numbers rather
    than a string the reader has to re-parse."""
    from fm_wedge_forensics import analyze

    resources = analyze(EPISODE_2_JOURNAL)[0].resources

    assert resources.cpu_seconds == 2 * 3600 + 29 * 60 + 21.259
    assert resources.memory_peak_bytes == 3.2 * 1024**3
    assert resources.swap_peak_bytes == 934.4 * 1024**2


def test_consumed_line_parses_both_systemd_cpu_time_spellings():
    """Episode 1 carries `1h 22min 57.690s`; the 16:15 stall carries systemd's
    hourless `20min 30.697s`. Both are real lines, not constructed variants."""
    from fm_wedge_forensics import analyze

    with_hours = analyze(EPISODE_1_JOURNAL)[0].resources
    without_hours = analyze(WATCHDOG_CAUGHT_STALL_JOURNAL)[0].resources

    assert with_hours.cpu_seconds == 3600 + 22 * 60 + 57.690
    assert with_hours.swap_peak_bytes == 529.9 * 1024**2
    assert without_hours.cpu_seconds == 20 * 60 + 30.697
    assert without_hours.memory_peak_bytes == 2.3 * 1024**3


def test_a_stall_with_no_consumed_line_reports_absent_not_zero():
    """A self-recovered stall never produced a `Consumed` line, and reporting
    it as 0 would read as "consumed nothing" — the opposite of the truth."""
    from fm_wedge_forensics import analyze

    assert analyze(SELF_RECOVERED_STALL_JOURNAL)[0].resources is None


def test_a_foreign_process_sigkilled_from_the_cgroup_is_reported_by_name():
    """The `git` child still alive at teardown is the evidence that a
    synchronous git subprocess was in flight on the blocked loop. It must not
    be diluted by the unit's own `uv`/`python3` entries, which are expected."""
    from fm_wedge_forensics import analyze

    foreign = analyze(EPISODE_2_JOURNAL)[0].foreign_killed_processes

    assert [(process.pid, process.name) for process in foreign] == [(1289425, "git")]


def test_an_episode_that_killed_nothing_reports_no_foreign_processes():
    from fm_wedge_forensics import analyze

    assert analyze(EPISODE_1_JOURNAL)[0].foreign_killed_processes == ()


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
    from fm_wedge_forensics import analyze

    costs = analyze(EPISODE_2_JOURNAL)[0].costs

    assert costs.teardown_seconds == 90.0
    assert costs.startup_seconds == 41.0
    assert costs.total_seconds == 302.0


def test_episode_2_teardown_dominates_startup():
    """The answerable form of "should the port-down streak be shortened?" —
    with numbers rather than intuition."""
    from fm_wedge_forensics import analyze

    assert analyze(EPISODE_2_JOURNAL)[0].costs.dominant_recovery_term == "teardown"


def test_episode_1_cost_decomposes_from_its_own_clean_stop():
    from fm_wedge_forensics import analyze

    costs = analyze(EPISODE_1_JOURNAL)[0].costs

    assert costs.teardown_seconds == 50.0
    assert costs.startup_seconds == 39.0


def test_detection_interval_is_reported_unavailable_not_fabricated():
    """The watchdog's consecutive-failure streak is logged in the WATCHDOG's
    journal, not fused-memory's, so no fm-only capture can witness it. Absent
    rather than guessed."""
    from fm_wedge_forensics import analyze

    for journal in (EPISODE_1_JOURNAL, EPISODE_2_JOURNAL):
        assert analyze(journal)[0].costs.detection_seconds is None


def test_a_self_recovered_stall_has_no_teardown_or_startup_cost():
    """Nothing was torn down or started, so those terms are absent — not zero,
    which would read as an instantaneous restart that never happened."""
    from fm_wedge_forensics import analyze

    costs = analyze(SELF_RECOVERED_STALL_JOURNAL)[0].costs

    assert costs.teardown_seconds is None
    assert costs.startup_seconds is None
    assert costs.dominant_recovery_term is None
