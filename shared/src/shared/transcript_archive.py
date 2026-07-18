"""Best-effort archival of per-task Claude Code agent transcripts.

The per-task ``CLAUDE_CONFIG_DIR`` (see :class:`shared.config_dir.TaskConfigDir`)
stores each agent session's raw JSONL transcript under
``<config_dir>/projects/<enc>/<session_id>.jsonl`` (plus subagent transcripts
at ``<config_dir>/projects/<enc>/<session_id>/subagents/agent-*.jsonl``). That
config dir lives *inside* the task worktree and is destroyed at teardown, so
this module gzips the transcripts to a durable archive root outside the
worktree while each session is still born-complete.

Design properties (see plans/agent-transcript-archival-prd.md, task α):

* **Credential-safe by construction** — only ``*.jsonl`` files *under*
  ``projects/`` are ever globbed. The per-task OAuth ``.credentials.json``
  lives at the config-dir root and is structurally unreachable, as are any
  non-``.jsonl`` files.
* **Idempotent** — the source ``.jsonl`` mtime is mirrored onto the gzipped
  copy via :func:`os.utime`; an already-current archive is skipped. A resumed
  session only ever grows its transcript, so mtime strictly advances and the
  grown transcript is re-archived (last-write-wins).
* **Best-effort / never-raises** — any per-file I/O or gzip error is caught,
  logged once as a structured WARNING, and counted in the module-level
  :data:`_ARCHIVAL_FAILURES` counter. The function is total so the producer
  can call it inside ``_invoke``'s ``finally`` during exception propagation.
"""

from __future__ import annotations

import gzip
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Module-level count of per-file archival failures this process.
# Loud-over-silent (design-invariants INV-2/INV-4): a systemic breakage
# (e.g. disk full → every archive fails) shows this counter climbing rather
# than failing silently. A small read/reset accessor pair below keeps tests
# isolated without reaching into module internals.
_ARCHIVAL_FAILURES: int = 0


def _archival_failures() -> int:
    """Return the current module-level archival failure count (test aid)."""
    return _ARCHIVAL_FAILURES


def _reset_archival_failures() -> None:
    """Reset the module-level archival failure count (test isolation)."""
    global _ARCHIVAL_FAILURES
    _ARCHIVAL_FAILURES = 0


def archive_task_transcripts(
    config_dir: Path,
    task_id: str,
    session_id: str | None = None,
    *,
    archive_root: Path,
) -> int:
    """Archive a task's agent transcripts to a durable gzipped mirror.

    STUB — behaviour is filled in over the plan's TDD steps.
    """
    return 0
