"""know_live-specific reconciliation guardrail policy.

All know_live-specific constants live here so that shared reconciliation
infrastructure (stage2.py, task_knowledge_sync.py) does not embed
project-specific magic values.
"""

from __future__ import annotations

# Canonical project identifier — single source of truth for stage2 prompt injection.
KNOW_LIVE_PROJECT_ID: str = 'know_live'

# Snapshot write paths blocked-by-design for know_live.
#
# Two paths exist for task-count snapshot edges:
#   1. Direct add_memory(category='temporal_facts') from recon-stage-* agents:
#      rejected by the project-agnostic ReconSnapshotWriteRejected guard in
#      server/tools.py (the guard has no project_id condition — it rejects
#      ANY recon-stage-* agent's category='temporal_facts' +
#      is_count_snapshot(content) write). Confirmed rejected every cycle for
#      know_live (evidence: memories ca450d09-31ff-4bfa-82e7-267bdb70de68,
#      0c37377c-aa6a-48f5-b59a-981beb4ea556, fd21efe6-4030-46c2-b3e6-4b37a741e4e4,
#      22a492ec-d849-4d52-8eec-7f0a268f51f5; mem0 writeup
#      52e676ef-6d9f-4c9a-8ead-7037ee445bcd) — recurring "non-actionable;
#      human decision required" across 10+ cycles since ~2026-05-08.
#   2. Graphiti async queue path: unlike autopilot_video, this path is NOT a
#      guaranteed no-op for know_live — it succeeded exactly once (edge
#      3aa711d9, cycle c54bf1d0, 2026-06-28) across those 10+ cycles. That is
#      intermittent, not reliable.
#
# Because the only RELIABLE write path (direct) is unconditionally blocked,
# neither a missing edge nor a stale one is actionable for Stage 2
# remediation — remediation would retry the direct path and be rejected
# again. The flag is therefore a blanket True (mirroring autopilot_video),
# not a category-scoped variant: both missing_knowledge (edge absent) and
# memory_stale (edge stale) findings are false positives to suppress.
KNOW_LIVE_SNAPSHOT_WRITES_BLOCKED: bool = True
