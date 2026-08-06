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
* **Best-effort / never-raises** — any per-file I/O or gzip error is caught and
  logged once as a structured WARNING (``path``/``task_id``/``errno``) — the
  production signal an operator greps for a systemic breakage — while the
  module-level :data:`_ARCHIVAL_FAILURES` counter accrues the same failures as
  machine-readable substrate for a future health/digest consumer. The function
  is total so the producer can call it inside ``_invoke``'s ``finally`` during
  exception propagation.

Each transcript is streamed source → gzip in fixed-size chunks rather than
slurped whole, so peak memory stays bounded even for the multi-MB transcripts a
long or resumed session produces; the producer additionally offloads the whole
call to a worker thread so the CPU-bound compression never stalls the event loop.
"""

from __future__ import annotations

import gzip
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
    """Gzip a single transcript *src* to its mirror under *archive_root*.

    The destination mirrors *src*'s path relative to ``projects/`` with a
    ``.gz`` suffix: ``<archive_root>/<task_id>/<relpath-under-projects>.gz``.
    The source mtime is mirrored onto the ``.gz`` so an already-current
    archive is skipped: if the dest exists and its (int-truncated) mtime
    equals the source's, the file is left untouched and ``False`` is
    returned. A resumed session only ever grows its transcript, so its mtime
    strictly advances and the grown file is re-archived (last-write-wins).
    Returns ``True`` when a file was newly written.
    """
    rel = src.relative_to(projects_root)
    dest = archive_root / task_id / rel.parent / (rel.name + '.gz')
    st = src.stat()
    # Idempotency: skip when the archive is already current. int-truncate to
    # dodge FS mtime-granularity mismatch between source and the utime copy.
    if dest.exists() and int(dest.stat().st_mtime) == int(st.st_mtime):
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Stream source -> gzip in fixed-size chunks (shutil's default 64 KiB
    # buffer) rather than slurping the whole transcript via read_bytes():
    # agent-session JSONL can be many MB and only grows on resume, so streaming
    # bounds peak RSS to one buffer instead of the full file. Any I/O/gzip error
    # here is still an OSError, caught + counted by the caller (best-effort).
    with src.open('rb') as sfh, gzip.open(dest, 'wb') as dfh:
        shutil.copyfileobj(sfh, dfh)
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
        # Best-effort: a per-file OSError (incl. gzip.BadGzipFile, which
        # subclasses OSError) is logged + counted, never re-raised, so one bad
        # file cannot lose its siblings and the whole function stays total —
        # the property the producer relies on to call it inside a finally.
        try:
            if _archive_one(src, projects_root, archive_root, task_id):
                count += 1
        except OSError as exc:
            _record_failure(src, task_id, exc)
    return count


def durable_archive_path(
    archive_root: Path,
    task_id: str,
    session_id: str,
) -> Path | None:
    """Locate *session_id*'s archived main transcript under *archive_root*.

    The read side of this module. Returns the archived transcript's path, or
    ``None`` when this session has no archive. The located layout is the one
    :func:`_archive_one` writes::

        <archive_root>/<task_id>/<encoded-cwd>/<session_id>.jsonl[.gz]

    Contract (plans/session-resume-eligibility-seam-prd.md §8):

    * **I-A total** — never raises, for any input. Modelled on
      :func:`shared.cli_invoke._resolve_transcript_path`: one blanket
      ``except Exception`` → ``None``, not an enumeration of ``OSError``
      subclasses. Callers on the dispatch path (the orchestrator's
      session-resume guard and its fallback emit) lean on that totality to
      stay total themselves.
    * **I-B cwd-agnostic** — the encoded-cwd component is *globbed*, never
      assumed, so a session archived from one worktree lane is still found
      after re-dispatch into a different lane.
    * **I-C format-agnostic** — matches both the gzipped ``.jsonl.gz`` shape
      and the plain ``.jsonl`` shape task 3618 moves to, so the cutover needs
      no flag day and this locator carries no dependency on it.
    * **I-D read-only** — glob and ``stat`` only. Nothing is created, moved,
      deleted or decompressed here; *restoration* is task 3578's, under task
      3619's archive-before-delete guard.
    * **I-E sole locator** — this is the only session-id-keyed lookup into the
      archive. Consumers call it rather than re-deriving the layout, so the
      producer and every reader cannot drift (INV-5).
    * **I-F deterministic** — when several lanes archived the same session,
      the newest-mtime match wins, with a stable tiebreak.

    Returns ``Path | None``, never ``bool`` (PRD decision D6): existence is
    spelled ``is not None``, and task 3578's restore reuses the very path
    returned here instead of growing a second glob that would have to agree
    with this one byte-for-byte forever.

    .. note::
       Built up across task 3727's steps: this initial body satisfies only
       I-B/I-D/I-E. Steps 4, 6 and 8 widen it to satisfy I-C (format span and
       the directory-decoy filter), I-F (newest-mtime + tiebreak) and I-A
       (the blanket guard and input coercion) respectively.
    """
    archive_root = Path(archive_root)
    matches = list(archive_root.glob(f'{task_id}/*/{session_id}.jsonl.gz'))
    return matches[0] if matches else None
