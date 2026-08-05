"""Best-effort archival of per-task Claude Code agent transcripts.

The per-task ``CLAUDE_CONFIG_DIR`` (see :class:`shared.config_dir.TaskConfigDir`)
stores each agent session's raw JSONL transcript under
``<config_dir>/projects/<enc>/<session_id>.jsonl`` (plus subagent transcripts
at ``<config_dir>/projects/<enc>/<session_id>/subagents/agent-*.jsonl``). That
config dir lives *inside* the task worktree and is destroyed at teardown, so
this module copies the transcripts to a durable archive root outside the
worktree while each session is still born-complete.

Design properties (see plans/agent-transcript-archival-prd.md, task α):

* **Credential-safe by construction** — only ``*.jsonl`` files *under*
  ``projects/`` are ever globbed. The per-task OAuth ``.credentials.json``
  lives at the config-dir root and is structurally unreachable, as are any
  non-``.jsonl`` files.
* **Idempotent** — the source ``.jsonl`` mtime is mirrored onto the archived
  copy via :func:`os.utime`; an already-current archive is skipped. A resumed
  session only ever grows its transcript, so mtime strictly advances and the
  grown transcript is re-archived (last-write-wins).
* **Best-effort / never-raises** — any per-file I/O error is caught and
  logged once as a structured WARNING (``path``/``task_id``/``errno``) — the
  production signal an operator greps for a systemic breakage — while the
  module-level :data:`_ARCHIVAL_FAILURES` counter accrues the same failures as
  machine-readable substrate for a future health/digest consumer. The function
  is total so the producer can call it inside ``_invoke``'s ``finally`` during
  exception propagation.

The archive is a VERBATIM copy: one plain, greppable ``.jsonl`` corpus, so
``rg`` works across it and no reader needs a decompression branch (task 3618,
leaf α of plans/transcript-preservation-seam-prd.md).
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

# Module-level count of per-file archival failures this process.
#
# Loud-over-silent (design-invariants INV-2/INV-4) is carried TODAY by the
# per-file structured WARNING emitted in _record_failure (path/task_id/errno):
# that log line is the production signal a systemic breakage (e.g. disk full →
# every archive fails) surfaces on. This counter is the machine-readable
# companion to that log — process-cumulative substrate a later health/digest
# consumer can sample to render a climbing failure rate. It is deliberately
# owned-but-not-yet-consumed by this α task, exactly as α lays down the
# retention.* config knobs here for the δ GC consumer (task 2731) to enforce;
# the small read/reset accessor pair below keeps tests isolated in the meantime.
_ARCHIVAL_FAILURES: int = 0


def _archival_failures() -> int:
    """Return the current module-level archival failure count (test aid)."""
    return _ARCHIVAL_FAILURES


def _reset_archival_failures() -> None:
    """Reset the module-level archival failure count (test isolation)."""
    global _ARCHIVAL_FAILURES
    _ARCHIVAL_FAILURES = 0


def _record_failure(src: Path, task_id: str, exc: OSError) -> None:
    """Count and loudly (but non-fatally) log a single per-file archive failure.

    Increments :data:`_ARCHIVAL_FAILURES` and emits one structured WARNING so a
    systemic breakage (e.g. disk full → every file fails) is visible as a
    climbing counter rather than failing silently (design-invariants
    INV-2/INV-4).
    """
    global _ARCHIVAL_FAILURES
    _ARCHIVAL_FAILURES += 1
    logger.warning(
        'transcript_archive: failed to archive %s: %s',
        src,
        exc,
        extra={
            'path': str(src),
            'task_id': task_id,
            'errno': getattr(exc, 'errno', None),
        },
    )


def _archive_one(
    src: Path,
    projects_root: Path,
    archive_root: Path,
    task_id: str,
) -> bool:
    """Copy a single transcript *src* to its mirror under *archive_root*.

    The destination mirrors *src*'s path relative to ``projects/`` with NO
    added suffix: ``<archive_root>/<task_id>/<relpath-under-projects>``.
    The source mtime is mirrored onto the copy so an already-current
    archive is skipped: if the dest exists and its (int-truncated) mtime
    equals the source's, the file is left untouched and ``False`` is
    returned. A resumed session only ever grows its transcript, so its mtime
    strictly advances and the grown file is re-archived (last-write-wins).
    Returns ``True`` when a file was newly written.
    """
    rel = src.relative_to(projects_root)
    dest = archive_root / task_id / rel.parent / rel.name
    st = src.stat()
    # Idempotency: skip when the archive is already current. int-truncate to
    # dodge FS mtime-granularity mismatch between source and the utime copy.
    if dest.exists() and int(dest.stat().st_mtime) == int(st.st_mtime):
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    # shutil.copyfile uses the platform fast-copy path (copy_file_range /
    # sendfile) and is exactly as memory-bounded as an explicit chunked loop:
    # agent-session JSONL can be many MB and only grows on resume, so peak RSS
    # stays flat regardless of transcript size. This is a COPY, not a rename —
    # the session may still be live and reading its own source file. Any I/O
    # error here is an OSError, caught + counted by the caller (best-effort).
    shutil.copyfile(src, dest)
    os.utime(dest, (st.st_atime, st.st_mtime))
    return True


def archive_task_transcripts(
    config_dir: Path,
    task_id: str,
    session_id: str | None = None,
    *,
    archive_root: Path,
) -> int:
    """Archive a task's agent transcripts to a durable plain-.jsonl mirror.

    When *session_id* is given, only that session's main transcript
    (``projects/*/<session_id>.jsonl``) and its subagent transcripts
    (``projects/*/<session_id>/subagents/*.jsonl``) are archived. When
    *session_id* is ``None``, every ``projects/**/*.jsonl`` is archived.
    Returns the number of files newly archived.
    """
    config_dir = Path(config_dir)
    archive_root = Path(archive_root)
    projects_root = config_dir / 'projects'

    if session_id is None:
        sources = list(projects_root.glob('**/*.jsonl'))
    else:
        sources = list(projects_root.glob(f'*/{session_id}.jsonl'))
        sources += list(projects_root.glob(f'*/{session_id}/subagents/*.jsonl'))

    count = 0
    for src in sources:
        # Best-effort: a per-file OSError is logged + counted, never re-raised,
        # so one bad file cannot lose its siblings and the whole function stays
        # total — the property the producer relies on to call it in a finally.
        try:
            if _archive_one(src, projects_root, archive_root, task_id):
                count += 1
        except OSError as exc:
            _record_failure(src, task_id, exc)
    return count
