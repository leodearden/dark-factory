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
