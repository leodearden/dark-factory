"""dark_factory-specific reconciliation guardrail policy.

All dark_factory-specific constants live here so that shared reconciliation
infrastructure does not embed project-specific magic values.
"""

from __future__ import annotations

# Canonical project identifier — single source of truth for the
# snapshot-write-blocked registry (see policies/__init__.py).
DARK_FACTORY_PROJECT_ID: str = 'dark_factory'

# Snapshot write paths blocked-by-design for dark_factory.
#
# Unlike autopilot_video/know_live — where the write PATH itself is rejected
# by a server-side guard — dark_factory's write path is not blocked at all.
# Live Mem0 evidence shows dark_factory has NEVER written a
# kind='task_count_snapshot' memory: the per-project task-count census is
# simply not in use for this project, so the ABSENCE of a snapshot is
# correct-by-design, not a gap to remediate. (task 2325, follow-up to 2278)
#
# As a result, Stage 3 findings asserting the task-count snapshot is missing
# or stale for dark_factory are false positives and must be suppressed.
#
# Reversible: if dark_factory later wants the per-project census, remove this
# exemption AND wire up the deterministic write (see
# task_count_snapshot_cadence.build_task_count_snapshot_content and
# stages/task_knowledge_sync._write_task_count_snapshot) rather than relying
# on the LLM-driven write this flag currently exempts dark_factory from.
DARK_FACTORY_SNAPSHOT_WRITES_BLOCKED: bool = True
