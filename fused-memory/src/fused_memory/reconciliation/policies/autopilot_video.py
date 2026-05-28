"""Autopilot-video-specific reconciliation guardrail policy.

All autopilot_video-specific constants and helpers live here so that
shared reconciliation infrastructure (stage2.py, task_knowledge_sync.py)
does not embed project-specific magic values.
"""

from __future__ import annotations

# Canonical project identifier — single source of truth for stage2 prompt injection.
AUTOPILOT_VIDEO_PROJECT_ID: str = 'autopilot_video'

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
- If no matching project is in "Known Projects", add a ``cross_project_findings`` \
entry to your structured report (with ``summary``, ``target_project_hint``, and \
``evidence``) so the operator can route it manually. Do NOT suppress normal \
reconciliation work for this project.

"""
