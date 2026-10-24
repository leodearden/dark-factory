#!/usr/bin/env python3
"""One pass over a run's spawned sessions: resolve each to its registry record, archive its hand-back.

What a session's standing means, and when it is final, is
``plans/review-all-program-tooling-prd.md`` §4 H5 and its B10 row.
"""
from __future__ import annotations

import argparse
import enum
import math
import sys
import types
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sessions
import state
from sessions import SpawnedSession, Standing

from shared import safe_io

sys.path.insert(0, str(state.CHECKOUT_ROOT / 'orchestrator' / 'src'))

from orchestrator.session_registry import (
    RESULT_FILENAME,
    TERMINAL_RESULT_OUTCOMES,
    TERMINAL_STATUSES,
    CorruptSessionRecord,
    ResultOutcome,
    SessionRecord,
    _pid_alive,
    parse_result_outcome,
    parse_spawn_identity,
    read_record,
    record_path_for_slug,
    sessions_dir,
)


class Exit(enum.IntEnum):
    WATCHED = 0
    APPLY_REFUSED = 1
    COULD_NOT_RUN = 2


_EXIT_MEANINGS: Mapping[Exit, str] = types.MappingProxyType({
    Exit.WATCHED: 'the pass ran',
    Exit.APPLY_REFUSED: 'a final return could not be applied; the lead acts',
    Exit.COULD_NOT_RUN: 'could not run',
})

_LABELS: Mapping[Standing, str] = types.MappingProxyType({
    Standing.AWAITING_RECORD: 'no registry record yet',
    Standing.AMBIGUOUS: 'several registry records match',
    Standing.UNREADABLE: 'unreadable',
    Standing.RUNNING: 'running, no hand-back yet',
    Standing.CHECKPOINT: 'checkpoint',
    Standing.FINISHED_NOT_EXITED: 'finished, not exited',
    Standing.ENDED_WITHOUT_HANDBACK: 'ended without a hand-back — relaunch or resume',
    Standing.ENDED_AT_CHECKPOINT: 'ended at checkpoint — relaunch or resume',
    Standing.FINAL: 'final',
})
if _LABELS.keys() != set(Standing):
    raise RuntimeError(f'unlabelled standings: {", ".join(set(Standing) - _LABELS.keys())}')


class Unreadable(Exception):
    """A record or hand-back that exists but cannot be read; the message names its path and cause."""


@dataclass(frozen=True)
class RegistryEntry:
    """One registry dir as this pass read it: its record, or why it could not be read."""

    record_dir: Path
    mtime: float
    record: SessionRecord | None
    unreadable: str | None


@dataclass(frozen=True)
class HandBack:
    text: str
    outcome: ResultOutcome | None


@dataclass(frozen=True)
class Seen:
    """Where one session stands this pass, the hand-back archived for it, and what its line adds."""

    standing: Standing
    archived: Path | None = None
    detail: str = ''


def _cause(bad: Exception) -> str:
    """What went wrong reading a file, without restating its path."""
    if isinstance(bad, FileNotFoundError):
        return 'absent'
    if isinstance(bad, OSError) and bad.strerror:
        return bad.strerror
    return str(bad.__cause__ or bad)


def _registry_entry(record_dir: Path) -> RegistryEntry | None:
    path = record_path_for_slug(record_dir.name)
    try:
        mtime = path.stat().st_mtime
    except (FileNotFoundError, NotADirectoryError):
        return None
    except OSError as bad:
        return RegistryEntry(record_dir, math.inf, None, f'{path}: {_cause(bad)}')
    try:
        return RegistryEntry(record_dir, mtime, read_record(record_dir.name), None)
    except FileNotFoundError:
        return None
    except (CorruptSessionRecord, OSError) as bad:
        return RegistryEntry(record_dir, mtime, None, f'{path}: {_cause(bad)}')


def scan_registry() -> tuple[RegistryEntry, ...]:
    """Every record in the registry, read once; a dir without a record.json is not a record."""
    try:
        children = sorted(sessions_dir().iterdir())
    except FileNotFoundError:
        return ()
    entries = (_registry_entry(child) for child in children)
    return tuple(entry for entry in entries if entry is not None)


def resolve(session: SpawnedSession, entries: Iterable[RegistryEntry]) -> RegistryEntry | Seen:
    """The one record written after the spawn with the session's title and role, else why none is."""
    spawned_at = datetime.fromisoformat(session.spawned_after).timestamp()
    role = parse_spawn_identity({}, session.title, '', '').role
    candidates = [entry for entry in entries if entry.mtime >= spawned_at]
    matches = [
        entry for entry in candidates
        if entry.record is not None
        and entry.record.title == session.title and entry.record.role == role
    ]
    if len(matches) == 1:
        return matches[0]
    if matches:
        return Seen(Standing.AMBIGUOUS, detail=': ' + ', '.join(m.record_dir.name for m in matches))
    unreadable = [entry.unreadable for entry in candidates if entry.unreadable is not None]
    if unreadable:
        return Seen(Standing.UNREADABLE, detail=': ' + '; '.join(unreadable))
    return Seen(Standing.AWAITING_RECORD)


def read_resolved(record_dir: Path) -> SessionRecord:
    """The record in *record_dir*; Unreadable when it is gone or does not parse."""
    root = record_dir.parent.parent
    try:
        return read_record(record_dir.name, root=root)
    except (CorruptSessionRecord, OSError) as bad:
        raise Unreadable(f'{record_path_for_slug(record_dir.name, root=root)}: {_cause(bad)}') from None


def read_hand_back(record_dir: Path) -> HandBack | None:
    """The session's result.md, or None when it has written none; Unreadable when it cannot be read."""
    path = record_dir / RESULT_FILENAME
    try:
        text = path.read_bytes().decode('utf-8')
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError) as bad:
        raise Unreadable(f'{path}: {_cause(bad)}') from None
    return HandBack(text, parse_result_outcome(text))


def is_over(record: SessionRecord, owner_alive: Callable[[int], bool] = _pid_alive) -> bool:
    """Whether the session has ended: a terminal status, or a self-exit attempt whose owner is dead."""
    if record.status in TERMINAL_STATUSES:
        return True
    owner = record.claude_owner_pid
    return record.exit_requested_at is not None and owner is not None and not owner_alive(owner)


def is_final(outcome: ResultOutcome | None, *, over: bool) -> bool:
    """H5's rule: a hand-back is final only when its outcome is terminal and its session is over."""
    return over and outcome in TERMINAL_RESULT_OUTCOMES


def standing(hand_back: HandBack | None, over: bool) -> Standing:
    if hand_back is None:
        return Standing.ENDED_WITHOUT_HANDBACK if over else Standing.RUNNING
    if is_final(hand_back.outcome, over=over):
        return Standing.FINAL
    if over:
        return Standing.ENDED_AT_CHECKPOINT
    if hand_back.outcome in TERMINAL_RESULT_OUTCOMES:
        return Standing.FINISHED_NOT_EXITED
    return Standing.CHECKPOINT


def _archive_target(
    home: state.Home, session: SpawnedSession, hand_back: HandBack, *, final: bool,
) -> Path | None:
    def handback_file(checkpoint: int | None = None) -> Path:
        return home.handback_file(session.stream, session.pass_, session.item, checkpoint=checkpoint)

    if final:
        return None if handback_file().exists() else handback_file()
    taken = 0
    while handback_file(taken + 1).exists():
        taken += 1
    if taken and handback_file(taken).read_bytes() == hand_back.text.encode('utf-8'):
        return None
    return handback_file(taken + 1)


def archive(
    home: state.Home, session: SpawnedSession, hand_back: HandBack, *, final: bool,
) -> Path | None:
    """Copy a hand-back not yet archived into handbacks/: the final file, or the next checkpoint."""
    target = _archive_target(home, session, hand_back, final=final)
    if target is not None:
        safe_io.atomic_write_text(target, hand_back.text, mkdir=True)
    return target


def _observed(home: state.Home, session: SpawnedSession, record_dir: Path) -> Seen:
    record = read_resolved(record_dir)
    hand_back = read_hand_back(record_dir)
    now = standing(hand_back, is_over(record))
    archived = None if hand_back is None else archive(
        home, session, hand_back, final=now is Standing.FINAL,
    )
    outcome = 'none' if hand_back is None or hand_back.outcome is None else hand_back.outcome
    return Seen(now, archived, f' (outcome {outcome}, status {record.status})')


def observe(
    home: state.Home, session: SpawnedSession, entries: Iterable[RegistryEntry],
) -> tuple[SpawnedSession, Seen]:
    """*session*, its record_dir filled once resolved, and where it stands this pass."""
    record_dir = session.record_dir
    if record_dir is None:
        found = resolve(session, entries)
        if isinstance(found, Seen):
            return session, found
        record_dir = str(found.record_dir)
        session = replace(session, record_dir=record_dir)
    try:
        return session, _observed(home, session, Path(record_dir))
    except Unreadable as bad:
        return session, Seen(Standing.UNREADABLE, detail=f': {bad}')


def line(home: state.Home, title: str, seen: Seen) -> str:
    archived = '' if seen.archived is None else f', archived {seen.archived.relative_to(home.root)}'
    return f'{title}: {_LABELS[seen.standing]}{archived}{seen.detail}'


def watch(home: state.Home) -> Exit:
    recorded = sessions.read(home)
    if not recorded:
        print(f'no spawned session is recorded in {home.sessions_json}')
        return Exit.WATCHED
    unresolved = any(session.record_dir is None for session in recorded.values())
    entries = scan_registry() if unresolved else ()
    changed: list[SpawnedSession] = []
    for was in recorded.values():
        now, seen = observe(home, was, entries)
        if now.record_dir != was.record_dir:
            print(f'{now.title}: record {now.record_dir}')
        print(line(home, now.title, seen))
        now = replace(now, reported=seen.standing)
        if now != was:
            changed.append(now)
    if changed:
        sessions.update_watched(home, changed)
    return Exit.WATCHED


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='watch_sessions.py', description=(__doc__ or '').partition('\n')[0],
        epilog='exit codes: ' + '; '.join(f'{int(c)} {says}' for c, says in _EXIT_MEANINGS.items()),
    )
    parser.add_argument('--run', required=True, help='the run id whose spawned sessions to watch')
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return watch(state.run_home(args.run))
    except (state.StateError, OSError) as failed:
        print(f'watch_sessions.py: could not run: {failed}', file=sys.stderr)
        return Exit.COULD_NOT_RUN


if __name__ == '__main__':
    raise SystemExit(main())
