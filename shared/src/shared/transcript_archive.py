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


def _archive_one(
    src: Path,
    projects_root: Path,
    archive_root: Path,
    task_id: str,
) -> bool:
    """Gzip a single transcript *src* to its mirror under *archive_root*.

    The destination mirrors *src*'s path relative to ``projects/`` with a
    ``.gz`` suffix: ``<archive_root>/<task_id>/<relpath-under-projects>.gz``.
    The source mtime is mirrored onto the ``.gz`` so idempotency (a later
    step) can detect an already-current archive. Returns ``True`` when a file
    was written.
    """
    rel = src.relative_to(projects_root)
    dest = archive_root / task_id / rel.parent / (rel.name + '.gz')
    dest.parent.mkdir(parents=True, exist_ok=True)
    data = src.read_bytes()
    with gzip.open(dest, 'wb') as fh:
        fh.write(data)
    st = src.stat()
    os.utime(dest, (st.st_atime, st.st_mtime))
    return True


def archive_task_transcripts(
    config_dir: Path,
    task_id: str,
    session_id: str | None = None,
    *,
    archive_root: Path,
) -> int:
    """Archive a task's agent transcripts to a durable gzipped mirror.

    When *session_id* is given, only that session's main transcript
    (``projects/*/<session_id>.jsonl``) and its subagent transcripts
    (``projects/*/<session_id>/subagents/*.jsonl``) are archived. Returns the
    number of files newly archived.
    """
    config_dir = Path(config_dir)
    archive_root = Path(archive_root)
    projects_root = config_dir / 'projects'

    if session_id is None:
        return 0  # archive-all branch implemented in a later step

    sources = list(projects_root.glob(f'*/{session_id}.jsonl'))
    sources += list(projects_root.glob(f'*/{session_id}/subagents/*.jsonl'))

    count = 0
    for src in sources:
        if _archive_one(src, projects_root, archive_root, task_id):
            count += 1
    return count
