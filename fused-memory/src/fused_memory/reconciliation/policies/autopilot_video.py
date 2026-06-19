"""Autopilot-video-specific reconciliation guardrail policy.

All autopilot_video-specific constants and helpers live here so that
shared reconciliation infrastructure (stage2.py, task_knowledge_sync.py)
does not embed project-specific magic values.
"""

from __future__ import annotations

# Canonical project identifier — single source of truth for stage2 prompt injection.
AUTOPILOT_VIDEO_PROJECT_ID: str = 'autopilot_video'

# Snapshot write paths blocked-by-design for autopilot_video.
#
# Two paths are blocked:
#   1. Direct add_memory(category='temporal_facts') from recon-stage-* agents:
#      rejected by the Layer-2 ReconSnapshotWriteRejected guard in server/tools.py.
#   2. Graphiti async queue path: silently no-ops (memory_ids=[]) so no edge lands.
#
# As a result the ABSENCE of a task-count snapshot temporal_fact edge is the
# CORRECT state for this project.  Stage 3 findings asserting that the edge is
# missing or stale are therefore false positives and must be suppressed.
# (evidence: memory fccfa232-16fe-404a-880a-8eca32eb974e)
AUTOPILOT_VIDEO_SNAPSHOT_WRITES_BLOCKED: bool = True

# Prompt fragment injected by build_stage2_system_prompt() only when
# project_id == AUTOPILOT_VIDEO_PROJECT_ID.  Content-based: cross-project
# contamination is detected by cited file paths/modules belonging to another
# repo, NOT by numeric task-ID magnitude (high IDs are normal project growth).
AUTOPILOT_VIDEO_CONTAMINATION_GUARDRAIL: str = """\
## Cross-Project Contamination Guardrail (Pre-flight)
There is NO task-ID ceiling for autopilot_video. High task IDs (e.g. 607, 700, \
1000+) are normal project growth and are never evidence of cross-project contamination \
on their own. Do NOT abort or suppress task actions based on task-ID magnitude alone.

Cross-project contamination is identified by **content**: cited file paths or modules \
that belong to a different repository. The path-scope guard (``DarkFactoryPathScopeViolation``) \
already rejects mis-routed task writes at the API level, and the \
**## Cross-Project Routing** section in this prompt explains how to handle findings \
whose scope belongs to another project.

When a finding appears to belong to another project:
- If the target project is listed in the payload's "Known Projects" section, \
route the task there via that project's ``project_root``; the path-scope guard will \
validate the routing.
- If no matching project is in "Known Projects", do NOT file the task in the current \
project as a workaround. Instead, emit a finding via recon_report: call \
``mcp__recon-report__add_finding(severity='moderate', \
category='cross_project_routing', flag_type='cross_project', actionable=False, \
description=<one-line summary + target_project_hint>, \
suggested_action=<short evidence notes>, task_id=None)`` so the operator can route it \
manually. No dedicated cross-project tool is needed — the category/flag_type encoding \
carries the routing signal. Do NOT suppress normal reconciliation work for this project.

"""
