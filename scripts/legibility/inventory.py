#!/usr/bin/env python3
"""Session inventory — enumerate yesterday's sessions for a project (PRD §5.2 point 2).

Walks ``~/.claude/projects/<enc>`` (or an injected ``projects_root``, so
tests never touch the real ``~/.claude`` tree — mirrors
``session_registry.fleet_root(root=)``) for a project whose agents span
many encoded cwd directories: a project's config lists **cwd prefixes**
(``docs/legibility/legibility.yaml``'s ``cwd_prefixes``), and membership is
resolved from each session's REAL ``cwd`` (read from a transcript line) via
path-component semantics, never a raw string-prefix match on the encoded
dir name — the ``~/.claude/projects`` encoding is lossy (``/``, ``.`` and
``_`` all map to ``-``), so a sibling project sharing the same literal
prefix (e.g. ``dark-factory-cockpit``) would otherwise be over-included.

Task β of the confusion-reduction PRD (plans/confusion-reduction-prd.md
§5.2, contract §7.4). Self-contained — does not import task α's
``digest.py``.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any


def encode_cwd(cwd: str) -> str:
    """Mirror ``session_registry.encode_cwd``'s encoding of Claude Code's dir naming.

    ``/``, ``.`` AND ``_`` all map to ``-``; ``-`` passes through and CASE
    IS PRESERVED (the encoder does not lowercase). Validated against 738
    real (encoded-dir, decoded-cwd) pairs sampled from a live
    ``~/.claude/projects`` tree, over which the only non-alphanumeric
    characters appearing in any cwd were ``- . / _`` — so the rule is
    complete over the observed domain, and unverified outside it.

    This is a hand-written mirror of the canonical implementation,
    ``orchestrator.session_registry.encode_cwd`` (named, not line-cited, so
    this docstring doesn't drift as that file evolves). It is duplicated
    rather than imported because this module is deliberately self-contained
    — its ``__main__`` bootstrap puts only ``scripts/`` on ``sys.path``, so
    a module-scope orchestrator import would break the standalone nightly
    invocation. The "kept in lockstep" claim is therefore not left to
    goodwill: ``scripts/tests/test_legibility_inventory.py``'s
    ``TestEncoderLockstep`` mechanically asserts this function, the
    canonical, and every other in-repo copy agree row-for-row with real
    on-disk dir names. Task 3272 added it after the ``_`` rule was found
    missing from all four copies at once.
    """
    return cwd.replace('/', '-').replace('.', '-').replace('_', '-')


def iter_project_dirs(projects_root: Path | str, cwd_prefixes: Sequence[str]) -> Iterator[Path]:
    """Yield candidate project dirs under *projects_root* by a cheap encoded-prefix pre-filter.

    Intentionally imprecise: a directory name that merely *starts with* one
    of the encoded prefixes is yielded, even when it actually belongs to an
    unrelated sibling project (e.g. ``-home-leo-src-dark-factory-cockpit``
    starts with the encoded ``-home-leo-src-dark-factory`` prefix). This is
    only a cheap candidate filter over a directory listing that can hold
    hundreds of entries — callers MUST additionally confirm membership via
    :func:`is_member` against each session's real ``cwd``
    (:func:`session_cwd`) before treating a session as belonging to the
    project.
    """
    root = Path(projects_root)
    if not root.is_dir():
        return
    encoded_prefixes = [encode_cwd(prefix) for prefix in cwd_prefixes]
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        if any(entry.name.startswith(enc) for enc in encoded_prefixes):
            yield entry


def is_member(cwd: str, cwd_prefixes: Sequence[str]) -> bool:
    """True iff *cwd* is one of *cwd_prefixes* or a descendant of one.

    Uses ``Path.is_relative_to`` path-COMPONENT semantics (never a raw
    string prefix match), so a sibling directory that merely shares a
    literal string prefix (``dark-factory-cockpit`` vs ``dark-factory``) is
    correctly excluded, while a ``.worktrees``/``.claude-worktrees`` child
    of a real prefix is correctly included.
    """
    cwd_path = Path(cwd)
    return any(cwd_path.is_relative_to(Path(prefix)) for prefix in cwd_prefixes)


UNREADABLE_FILE_ERRORS = (UnicodeDecodeError,)
"""Exception types an unreadable transcript can raise that are NOT ``OSError``.

With the archive stored plain (task 3618) exactly ONE such shape remains:
``UnicodeDecodeError``. It is reachable on any transcript, since the reader
opens under strict ``encoding='utf-8'``, and being a ``ValueError`` subclass
it escapes an ``except OSError``-only handler entirely — so it still needs
normalizing even though the gzip container shapes are gone.

Exported deliberately: ``digest.load_transcript`` catches on THIS name rather
than a copy, so the two readers cannot drift apart at that boundary.
"""


def as_unreadable_file_error(exc: BaseException) -> OSError:
    """Normalize an unreadable-transcript exception to one ``OSError``.

    THE single answer to "which exceptions mean this transcript cannot be
    read, and what does that read as to a caller". Both transcript readers —
    :func:`iter_json_lines` and its slurping sibling ``digest.load_transcript``
    — funnel through here, so the normalization is structurally shared rather
    than copy-pasted and test-pinned in sync (reviewer_comprehensive
    /duplication, task 3214 amendment pass).

    Why normalization is needed at all: an unreadable transcript surfaces as
    FOUR different exception types, only the first of which is already an
    ``OSError``::

        undecodable byte  -> UnicodeDecodeError   (a ValueError, not an OSError)

    The last three are exactly what a fire-and-forget writer produces when a
    unit is killed mid-write, a file is read while still being compressed, or
    a stored byte flips; the archived fleet transcripts these readers walk are
    live runtime state, so all three are expected shapes, not theoretical
    ones. The decode shape is the odd one out in two ways worth stating: it is
    the only shape reachable on a PLAIN ``.jsonl`` path as well as a ``.gz``
    one (both are opened under strict ``encoding='utf-8'``), and it arrives as
    a ``ValueError`` subclass, so an ``except OSError``-only handler misses it
    however the gzip shapes are handled. Left un-normalized, any of them
    escapes every consumer's degrade path and aborts a whole-archive walk over
    one bad file.

    ``OSError`` is the normalized type because it is what ``sampling``,
    ``check_transcript_persistence`` and the cross-package
    ``memory_eval_transcript_corpus`` extractor already code against. The
    original message is preserved in the text, so a coverage report can still
    tell a half-written transcript from a corrupted one from an undecodable
    one — the decode shape gets its own wording rather than being labelled a
    "gzip stream" failure, which would misdirect an operator reading a
    disclosed reason for a plain ``.jsonl`` file.

    Total and idempotent: an exception that is ALREADY an ``OSError`` (the bad
    magic shape) is returned unchanged rather than double-wrapped, so a caller
    can hand it anything it caught without first classifying it.
    """
    if isinstance(exc, OSError):
        return exc
    if isinstance(exc, UnicodeDecodeError):
        return OSError(f'undecodable transcript bytes: {exc}')
    return OSError(f'unreadable transcript: {exc}')


def iter_json_lines(path: Path) -> Iterator[dict[str, Any]]:
    """Yield parsed dict records from a JSONL file, skipping blank/malformed lines.

    **The** low-level streaming transcript reader — public, and deliberately
    so: cross-package consumers exist (``memory_eval_transcript_corpus.py``
    in ``fused-memory/scripts/`` mines the archived fleet transcripts through
    it), and a consumer reaching for an underscore name across a package
    boundary is how a second copy of this function gets written instead.

    Streaming rather than slurping is the point of this one: callers walk
    thousands of multi-MB archived transcripts, so memory stays bounded by
    the largest single record. ``digest.load_transcript`` is the slurping
    sibling for the single-file case, with a byte-identical parse contract.

    Mirrors ``digest.load_transcript``'s graceful-degrade contract: a
    transcript is written fire-and-forget and can have a truncated or
    corrupt trailing line, which must not abort the whole read. Raises
    ``OSError`` if *path* cannot be read at all.

    That split is a contract, not an implementation detail: a corrupt LINE
    degrades silently, an unreadable FILE raises. Callers that report
    coverage rely on it to count unreadable files without inflating the
    count with truncated trailing lines.

    Making that ``OSError`` promise true takes explicit normalization: the
    decode shape is a ``ValueError`` subclass, not an ``OSError``.
    :func:`as_unreadable_file_error` is the
    single place that says which ones and why, and ``digest.load_transcript``
    funnels through the same helper, so the two readers are interchangeable
    at this boundary by construction rather than by matching copies.

    Note the decode shape is normalized at the FILE level, not degraded at
    the line level: a bad byte aborts the underlying text-IO read, so there
    is no well-formed "rest of the file" left to keep yielding. Counting the
    file once as unreadable is the honest report; silently substituting
    replacement characters would let corrupt bytes enter the corpus as
    plausible-looking data.
    """
    with open(path, encoding='utf-8') as f:
        # The wrap covers only the read iteration — decoding
        # happens lazily HERE, per chunk, not at open() — while the
        # JSONDecodeError skip stays inside the loop, so the file-level vs
        # line-level split above is preserved. The catch tuple is the shared
        # UNREADABLE_FILE_ERRORS rather than a local literal, deliberately:
        # `except Exception` would swallow errors from the caller's
        # consumption of this generator, and a local tuple would be a second
        # answer that could drift from digest.load_transcript's.
        try:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    yield record
        except UNREADABLE_FILE_ERRORS as exc:
            raise as_unreadable_file_error(exc) from exc


def _session_cwd_and_date(path: Path) -> tuple[str | None, date | None]:
    """Single-pass read of *path*, returning ``(cwd, date)`` together.

    :func:`session_cwd` and :func:`_first_timestamp_date` each scan *path*
    independently for their own first-occurrence value, so
    :func:`enumerate_sessions` calling both on every candidate session read
    the same transcript twice (reviewer_comprehensive/performance, task
    2573 amendment pass #2). This helper collects both facts in ONE pass —
    each still only takes its own first occurrence, so the values returned
    are identical to calling ``session_cwd(path)`` and
    ``_first_timestamp_date(path)`` separately — stopping as soon as both
    have been found. Degrades to ``(None, None)`` on an unreadable path,
    matching both functions' individual degrade-to-None contracts.
    """
    cwd: str | None = None
    session_date: date | None = None
    try:
        for record in iter_json_lines(path):
            if cwd is None:
                candidate_cwd = record.get('cwd')
                if isinstance(candidate_cwd, str) and candidate_cwd:
                    cwd = candidate_cwd

            if session_date is None:
                ts = record.get('timestamp')
                if isinstance(ts, str) and ts:
                    try:
                        parsed = datetime.fromisoformat(ts)
                    except ValueError:
                        parsed = None
                    if parsed is not None:
                        if parsed.tzinfo is None:
                            parsed = parsed.replace(tzinfo=UTC)
                        session_date = parsed.astimezone(UTC).date()

            if cwd is not None and session_date is not None:
                break
    except OSError:
        return None, None
    return cwd, session_date


def session_cwd(path: Path) -> str | None:
    """Return the first non-empty ``cwd`` string found in *path*, else None.

    Real transcripts are heterogeneous: some early lines (``ai-title``,
    ``agent-name``, ``queue-operation``) carry no ``cwd`` at all, and a few
    metadata-only stub sessions carry no ``cwd`` anywhere in the file. Both
    an unreadable path and a cwd-less file degrade to ``None`` rather than
    raising. Delegates to :func:`_session_cwd_and_date` — a standalone
    caller wanting only the cwd also pays for that helper's timestamp scan,
    but this function is no longer on :func:`enumerate_sessions`'s hot path
    (it calls :func:`_session_cwd_and_date` directly), so that trade only
    affects callers wanting the cwd in isolation.
    """
    cwd, _ = _session_cwd_and_date(path)
    return cwd


@dataclass(frozen=True)
class SessionRecord:
    """One enumerated session transcript.

    ``encoded_dir`` is the containing directory's basename (a
    ``~/.claude/projects/<enc>`` entry); ``cwd`` is the REAL decoded cwd
    read from the transcript (the value :func:`is_member` was checked
    against), not a re-derivation from ``encoded_dir``.
    """

    path: Path
    encoded_dir: str
    cwd: str
    date: date
    size_bytes: int


def _first_timestamp_date(path: Path) -> date | None:
    """Return the UTC date of the first valid ISO-8601 ``timestamp`` line in *path*.

    Returns None when the file is unreadable, has no timestamp line, or
    every timestamp line is unparseable — callers fall back to file mtime.
    Delegates to :func:`_session_cwd_and_date`; see its docstring for why.
    """
    _, session_date = _session_cwd_and_date(path)
    return session_date


def resolve_agent_transcript_roots(
    project_root: Path | str, roots: Sequence[str]
) -> list[Path]:
    """Resolve ``legibility.yaml``'s ``agent_transcript_roots`` against *project_root*.

    Each relative root (e.g. ``data/orchestrator/agent-transcripts``) is
    joined onto *project_root* so archive mining is independent of whatever
    CWD the trickle/census process happens to inherit; an already-absolute
    root is returned unchanged (``Path`` division returns the right operand
    verbatim when it is absolute). An empty *roots* yields ``[]`` — the
    byte-parity baseline in which :func:`enumerate_sessions` reads only the
    ``~/.claude/projects`` tree.
    """
    return [Path(project_root) / r for r in roots]


def _build_session_record(
    session_path: Path,
    encoded_dir: str,
    cwd_prefixes: Sequence[str],
    date_ok: Callable[[date], bool],
) -> SessionRecord | None:
    """Build a :class:`SessionRecord` for one transcript file, or ``None`` to skip it.

    Skips (returns ``None``) a non-existent/unreadable/empty file, a session
    whose real ``cwd`` is not a project member (:func:`is_member`), and a
    session whose date — first-timestamp (:func:`_session_cwd_and_date`),
    falling back to file mtime — is rejected by *date_ok* (``date_ok(d)`` is
    ``False``). The *date_ok* predicate is what lets one shared record-builder
    back both the single-date enumerator (``date_ok = d == target_date``) and
    the range enumerator (``date_ok = start <= d <= end``) with no duplicated
    walk. Shared by the ``~/.claude/projects`` loop and the archive-roots loop
    in :func:`_enumerate`: the projects path is behaviorally unchanged (guarded
    by ``TestEnumerateSessions``), while the archive path passes the encoded
    worktree dir (``session_path.parent.name``) as *encoded_dir*.
    """
    try:
        st = session_path.stat()
    except OSError:
        return None
    if st.st_size == 0:
        return None

    cwd, session_date = _session_cwd_and_date(session_path)
    if cwd is None or not is_member(cwd, cwd_prefixes):
        return None

    if session_date is None:
        # Reuse the single ``stat()`` above for the mtime fallback rather than
        # re-stat'ing the same path (reviewer_comprehensive/efficiency, task
        # 2730 amendment pass): ``datetime.fromtimestamp`` can still raise
        # ``OSError`` for an out-of-range mtime, so the degrade-to-None guard
        # is preserved verbatim.
        try:
            session_date = datetime.fromtimestamp(st.st_mtime, tz=UTC).date()
        except OSError:
            return None
    if not date_ok(session_date):
        return None

    return SessionRecord(
        path=session_path,
        encoded_dir=encoded_dir,
        cwd=cwd,
        date=session_date,
        size_bytes=st.st_size,
    )


def _iter_archive_transcripts(root: Path) -> Iterator[Path]:
    """Yield every ``*.jsonl`` transcript under *root*, recursively.

    The archived fleet-transcript tree (``shared.transcript_archive``) nests
    transcripts under ``<task_id>/<enc>/<sid>.jsonl`` with an even-deeper
    ``<sid>/subagents/agent-*.jsonl`` variant, so — unlike the
    ``~/.claude/projects`` pre-filter (:func:`iter_project_dirs`), whose
    top-level dir names ARE the encoded cwds — the archive is walked
    RECURSIVELY and membership is decided per-file downstream via
    :func:`is_member` (in :func:`_build_session_record`), the sole
    membership authority. Yields nothing when *root* is not an existing
    directory: an absent, not-yet-created archive root is normal (the tree
    is git-ignored and may not exist yet).

    The archive holds ONE class of transcript (task 3618 dropped the gzipped
    form), so this is a single ``rglob('*.jsonl')`` with no suffix filter.
    """
    if not root.is_dir():
        return
    yield from sorted(root.rglob('*.jsonl'))


def _archive_enc(session_path: Path, archive_root: Path) -> str | None:
    """Return the encoded ``<enc>`` cwd dir of an archive transcript, or ``None``.

    The archived fleet-transcript tree nests transcripts under
    ``<task_id>/<enc>/<sid>.jsonl`` (main) and
    ``<task_id>/<enc>/<sid>/subagents/agent-*.jsonl`` (subagent), so the
    encoded cwd directory is ``parts[1]`` of the archive-root-relative path in
    BOTH layouts — NOT ``session_path.parent.name`` (which is ``'subagents'``
    for the subagent variant, and would make the pre-filter drop EVERY subagent
    transcript). Returns ``None`` when *session_path* is not under
    *archive_root* (``relative_to`` raises ``ValueError``) or the relative path
    has fewer than two components — a short/degenerate layout whose ``<enc>``
    is underivable, in which case the caller does NOT pre-filter and falls back
    to ``session_path.parent.name``, keeping :func:`is_member` the sole
    membership authority. Never raises.
    """
    try:
        parts = session_path.relative_to(archive_root).parts
    except ValueError:
        return None
    return parts[1] if len(parts) >= 2 else None


def _enumerate(
    projects_root: Path | str,
    cwd_prefixes: Sequence[str],
    date_ok: Callable[[date], bool],
    agent_transcript_roots: Sequence[Path | str] = (),
) -> list[SessionRecord]:
    """Shared single-walk core behind :func:`enumerate_sessions` and :func:`enumerate_sessions_in_range`.

    Aggregates across every directory :func:`iter_project_dirs` yields
    (never assumes one dir per project — a project's agents span many
    encodings). For each ``*.jsonl`` file: skip non-existent/unreadable/
    empty files, read the session's real ``cwd`` and first-timestamp date
    together in one pass (:func:`_session_cwd_and_date`) and confirm
    membership via :func:`is_member`, falling back to file mtime for the
    date when no timestamp is present, and keep only sessions whose date
    satisfies *date_ok* (all via :func:`_build_session_record`).

    *agent_transcript_roots* is an ADDITIONAL, opt-in list of archive roots
    (the ``shared.transcript_archive`` fleet-transcript tree, resolved via
    :func:`resolve_agent_transcript_roots`) walked recursively ALONGSIDE the
    ``~/.claude/projects`` tree — each ``*.jsonl`` under a
    root is admitted by the same per-file membership + date filter. A cheap
    encoded-``<enc>`` pre-filter — mirroring :func:`iter_project_dirs`'
    superset pre-filter for the projects tree — skips a proven-foreign
    archive file, one whose ``<enc>`` (:func:`_archive_enc`, the
    archive-root-relative ``parts[1]``) does not encoded-prefix-match any
    ``cwd_prefixes`` entry, WITHOUT opening it; :func:`is_member` on the real
    cwd stays the SOLE membership authority (a lossy false-positive like a
    ``-cockpit`` sibling that string-startswith the prefix falls through to
    it), so the RESULT SET is unchanged. That same ``<enc>`` becomes the
    record's ``encoded_dir`` (falling back to ``session_path.parent.name``
    only when it is underivable). It defaults to an empty tuple: when empty,
    the archive loop does not execute and this walk is byte-identical to the
    projects-only path.

    The *date_ok* predicate is the ONLY thing that varies between the two
    public enumerators, so the walk itself — and its byte-parity for the
    single-date path — is defined exactly once here.
    """
    root = Path(projects_root)
    records: list[SessionRecord] = []
    for project_dir in iter_project_dirs(root, cwd_prefixes):
        for session_path in sorted(project_dir.glob('*.jsonl')):
            record = _build_session_record(
                session_path, project_dir.name, cwd_prefixes, date_ok
            )
            if record is not None:
                records.append(record)

    encoded_prefixes = [encode_cwd(prefix) for prefix in cwd_prefixes]
    for archive_root in agent_transcript_roots:
        archive_root_path = Path(archive_root)
        for session_path in _iter_archive_transcripts(archive_root_path):
            enc = _archive_enc(session_path, archive_root_path)
            # Cheap encoded-<enc> pre-filter (mirrors iter_project_dirs): skip
            # a proven-foreign archive file WITHOUT opening it. A None enc
            # (underivable layout) is NOT pre-filtered — is_member stays the
            # sole authority — and lossy false-positives (a -cockpit sibling
            # that startswith the prefix) fall through to is_member too.
            if enc is not None and not any(
                enc.startswith(ep) for ep in encoded_prefixes
            ):
                continue
            record = _build_session_record(
                session_path,
                enc if enc is not None else session_path.parent.name,
                cwd_prefixes,
                date_ok,
            )
            if record is not None:
                records.append(record)
    return records


def enumerate_sessions(
    projects_root: Path | str,
    cwd_prefixes: Sequence[str],
    target_date: date,
    agent_transcript_roots: Sequence[Path | str] = (),
) -> list[SessionRecord]:
    """Enumerate every session transcript for *target_date* across all matching encoded dirs.

    A thin single-date wrapper over :func:`_enumerate` with the equality
    predicate ``date_ok = (d == target_date)`` — behaviorally unchanged from
    the pre-refactor implementation (same walk, same equality filter),
    guarded by ``TestEnumerateSessions`` and ``TestEnumerateArchiveRoots``.
    See :func:`_enumerate` for the walk semantics and the
    *agent_transcript_roots* archive-mining contract, and
    :func:`enumerate_sessions_in_range` for the single-walk range variant the
    census uses to avoid re-opening each file once per window date.
    """
    return _enumerate(
        projects_root,
        cwd_prefixes,
        lambda d: d == target_date,
        agent_transcript_roots,
    )


def enumerate_sessions_in_range(
    projects_root: Path | str,
    cwd_prefixes: Sequence[str],
    start_date: date,
    end_date: date,
    agent_transcript_roots: Sequence[Path | str] = (),
) -> list[SessionRecord]:
    """Enumerate every session in the inclusive ``[start_date, end_date]`` window — in ONE walk.

    The single-walk O(total_files) replacement for calling
    :func:`enumerate_sessions` once per calendar date across a census window:
    a thin wrapper over :func:`_enumerate` with the range predicate
    ``date_ok = (start_date <= d <= end_date)``, so each session file is
    opened exactly once regardless of the window's length (the per-date loop
    opened it ``window_days`` times). Because a census window is a contiguous
    inclusive calendar range and each file matches at most one date, the
    result set is identical to the union of per-date :func:`enumerate_sessions`
    calls. See :func:`_enumerate` for the walk semantics and the
    *agent_transcript_roots* archive-mining contract.
    """
    return _enumerate(
        projects_root,
        cwd_prefixes,
        lambda d: start_date <= d <= end_date,
        agent_transcript_roots,
    )


def main(argv: Sequence[str]) -> int:
    """Thin standalone CLI: list sessions matching cwd_prefixes for a UTC date.

    A debug/inspection entry point for inventory.py alone — the full
    inventory->score->sample pipeline's CLI acceptance surface is
    ``sampling.main`` (task 2573 step-18).
    """
    parser = argparse.ArgumentParser(
        description="List legibility sessions matching cwd_prefixes for a given UTC date."
    )
    parser.add_argument(
        '--projects-root',
        default=str(Path.home() / '.claude' / 'projects'),
        help='Root of the ~/.claude/projects tree (default: %(default)s)',
    )
    parser.add_argument(
        '--cwd-prefix',
        dest='cwd_prefixes',
        action='append',
        required=True,
        help='A cwd prefix to match (repeatable).',
    )
    parser.add_argument(
        '--date',
        default=None,
        help='UTC date to filter sessions to, YYYY-MM-DD (default: yesterday UTC).',
    )
    args = parser.parse_args(argv)

    target_date = (
        date.fromisoformat(args.date)
        if args.date
        else (datetime.now(UTC) - timedelta(days=1)).date()
    )
    records = enumerate_sessions(args.projects_root, args.cwd_prefixes, target_date)
    for record in records:
        print(
            f'{record.path}\t{record.encoded_dir}\t{record.cwd}\t'
            f'{record.date}\t{record.size_bytes}'
        )
    return 0


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.exit(main(sys.argv[1:]))
