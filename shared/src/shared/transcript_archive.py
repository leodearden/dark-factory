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

The module also owns the **read** side of that archive:
:func:`durable_archive_path` is the single session-id-keyed locator into it
(task 3727 / plans/session-resume-eligibility-seam-prd.md §8), so every reader
resolves the layout from the same place the writer defines it and the two
cannot drift. It is strictly read-only — glob and ``stat``, nothing created,
moved, deleted or decompressed. *Restoring* an archived transcript back into a
live config dir is task 3578's, under task 3619's archive-before-delete guard;
that restore reuses this locator rather than growing a second glob.
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
      stay total themselves. A plain *miss* is silent (it is the common case);
      only a genuine fault is swallowed, and that is logged loudly.
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
    """
    try:
        # Coerce inside the try: a caller can easily hold a numeric task_id or
        # a str archive_root, and neither may raise from path composition.
        archive_root = Path(archive_root)
        task_id = str(task_id)
        session_id = str(session_id)
        # Two load-bearing details in this one line:
        #   * the trailing `*` on `.jsonl*` spans task 3618's gzip drop — it
        #     matches today's `.jsonl.gz` AND tomorrow's plain `.jsonl`, so the
        #     cutover needs no flag day and this locator no dependency on it;
        #   * `is_file()` excludes the SUBAGENT directory, which _archive_one
        #     names for the session itself (`<enc>/<sid>/subagents/agent-*.gz`).
        #     That dir carries no `.jsonl` suffix so the pattern misses it by
        #     luck today; the filter makes returning a directory as "the
        #     transcript" structurally impossible, not incidentally avoided.
        matches = [
            p for p in archive_root.glob(f'{task_id}/*/{session_id}.jsonl*') if p.is_file()
        ]
        if not matches:
            return None
        # Newest mtime wins: a resumed session's transcript only ever grows, so
        # the newest archive is the most complete one. The `str(p)` tiebreak is
        # not decoration — mtime alone is NOT a total order here, because
        # _archive_one mirrors the SOURCE mtime onto the archived copy via
        # os.utime (:124), so one session archived from two lanes can tie
        # exactly. Under a tie max() would fall back to filesystem-dependent
        # glob order; appending the path string makes the answer reproducible,
        # which is what makes a later resume reproducible.
        #
        # p.stat() can raise (a file unlinked between glob and stat — e.g. task
        # 2731's GC sweep landing mid-lookup). That is deliberately NOT caught
        # per-file: the blanket guard below is the single place totality is
        # enforced, so a miss and a race degrade to the same answer.
        return max(matches, key=lambda p: (p.stat().st_mtime, str(p)))
    except Exception as exc:
        # I-A: one blanket swallow, mirroring cli_invoke._resolve_transcript_path
        # rather than enumerating OSError subclasses — the whole point of the
        # contract is that NOTHING escapes, which is what lets the orchestrator's
        # session-resume guard and its fallback emit stay total on the dispatch
        # path.
        #
        # Level: WARNING, and note what does NOT reach here. The COMMON case —
        # this session simply has no archive (the PRD §2 reference measurement
        # puts ~36% of sessions there) — returns None from the empty-`matches`
        # branch above and logs nothing at all, so the ~36% can never become log
        # noise. Only a genuine fault lands in this handler: a glob that raised
        # (archive root unreadable/unmounted), a stat that raised (task 2731's
        # GC unlinking mid-lookup), or a non-coercible id from a caller bug.
        # Each of those is a real breakage an operator should see, so silence or
        # DEBUG here would be exactly the silent degradation design-invariants
        # INV-2/INV-4 forbid — and shared/tests/test_silent_fallthrough_gate.py
        # enforces that for new code (its baseline allowlist is a burn-down
        # ratchet for pre-existing sites like _resolve_transcript_path, not a
        # hatch new code may add to). Structured extra=, matching the shape
        # _record_failure already emits, so both archive-side failure signals
        # are greppable the same way.
        logger.warning(
            'durable_archive_path: lookup failed for session %s (task %s) under %s: %s',
            session_id,
            task_id,
            archive_root,
            exc,
            extra={
                'path': str(archive_root),
                'task_id': str(task_id),
                'errno': getattr(exc, 'errno', None),
            },
        )
        return None
