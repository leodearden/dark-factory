"""solar_challenge_platform-specific reconciliation guardrail policy.

All solar_challenge_platform-specific constants live here so that shared
reconciliation infrastructure does not embed project-specific magic values.
"""

from __future__ import annotations

# Canonical project identifier — single source of truth for the
# snapshot-write-blocked registry (see policies/__init__.py).
SOLAR_CHALLENGE_PLATFORM_PROJECT_ID: str = 'solar_challenge_platform'

# Snapshot write paths blocked-by-design for solar_challenge_platform.
#
# Unlike autopilot_video/know_live — where the write PATH itself is rejected
# by a server-side guard — solar_challenge_platform's write path is not
# blocked at all. Live Mem0 evidence shows solar_challenge_platform has
# NEVER written a kind='task_count_snapshot' memory: the per-project
# task-count census is simply not in use for this project, so the ABSENCE of
# a snapshot is correct-by-design, not a gap to remediate. (task 2325,
# follow-up to 2278)
#
# As a result, Stage 3 findings asserting the task-count snapshot is missing
# or stale for solar_challenge_platform are false positives and must be
# suppressed.
#
# Reversible: if solar_challenge_platform later wants the per-project
# census, remove this exemption AND wire up the deterministic write (see
# task_count_snapshot_cadence.build_task_count_snapshot_content and
# stages/task_knowledge_sync._write_task_count_snapshot) rather than relying
# on the LLM-driven write this flag currently exempts solar_challenge_platform from.
SOLAR_CHALLENGE_PLATFORM_SNAPSHOT_WRITES_BLOCKED: bool = True
