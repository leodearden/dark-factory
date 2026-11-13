"""The on-disk temp files a Claude CLI spawn is handed by argv path: their names, and the
reclamation of the ones a killed spawner orphaned."""

from __future__ import annotations

import logging
import os
import re
import stat
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)

ORPHAN_MIN_AGE_SECS = 24 * 3600.0


@dataclass(frozen=True)
class SpawnTempName:
    """One kind of spawn temp file, named the way ``tempfile.mkstemp`` names it.

    ``matches`` mirrors CPython's mkstemp naming (8 chars of ``[a-z0-9_]``), so if
    CPython ever changes it the sweep under-deletes rather than over-deletes, and
    shared/tests/test_spawn_temp_files.py fails loudly against the real mkstemp.
    """

    prefix: str
    suffix: str
    _shape: re.Pattern[str] = field(init=False, repr=False, compare=False)
    _occurrence: re.Pattern[str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        shape = re.escape(self.prefix) + '[a-z0-9_]{8}' + re.escape(self.suffix)
        object.__setattr__(self, '_shape', re.compile(shape))
        object.__setattr__(self, '_occurrence', re.compile(f'(?=({shape}))'))

    def mkstemp(self) -> tuple[int, str]:
        return tempfile.mkstemp(prefix=self.prefix, suffix=self.suffix)

    def matches(self, name: str) -> bool:
        return self._shape.fullmatch(name) is not None

    def names_in(self, text: str) -> frozenset[str]:
        """Every name of this kind occurring anywhere in *text*, overlapping ones included."""
        return frozenset(occurrence.group(1) for occurrence in self._occurrence.finditer(text))


SYSPROMPT_TEMP_NAME = SpawnTempName('sysprompt_', '.txt')
MCP_CONFIG_TEMP_NAME = SpawnTempName('mcp_', '.json')
SPAWN_TEMP_NAMES = (SYSPROMPT_TEMP_NAME, MCP_CONFIG_TEMP_NAME)


def sweep_orphaned_spawn_temp_files(
    *,
    base_dir: Path | None = None,
    proc_root: Path = Path('/proc'),
    min_age_secs: float = ORPHAN_MIN_AGE_SECS,
    deadline_secs: float = 5.0,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    """Delete the spawn temp files in *base_dir* that no live process names on its cmdline.

    The per-call ``finally`` unlink in every caller of
    shared/src/shared/cli_invoke.py::build_claude_argv is the eager path; this is the
    SIGKILL-recovery half, run once at process start. Returns the number of files this
    call genuinely removed. Contains every ``OSError`` and never raises one.

    A file is deleted only when ALL of these hold: its name has a ``SPAWN_TEMP_NAMES``
    kind's exact mkstemp shape; it is a regular file (not a symlink, not a dir); it is
    owned by this euid; it is at least *min_age_secs* old; and its basename appears in no
    live process's ``/proc/<pid>/cmdline`` (anywhere in the NUL-joined blob, so a path
    embedded in an ``sh -c`` token or a ``--flag=value`` is caught). Why cmdline and not
    fds: shared/tests/test_spawn_temp_files.py::test_keeps_a_stale_file_a_live_process_names_on_its_cmdline.

    Fails safe: when liveness cannot be established (*proc_root* unlistable, this
    process absent from it, a cmdline unreadable for any reason but the pid exiting, or
    the read unfinished at the deadline), nothing is deleted and one WARNING says so.

    No TOCTOU: every invocation, resume included since task 3983, mkstemps FRESH files,
    so no new process can come to reference a file already past *min_age_secs*.

    Returns within about *deadline_secs* on *clock*, however big *base_dir* or slow
    *proc_root*. The walk may spend half of it, which leaves the rest to check and
    reclaim whatever a walk too big to finish did find. The cmdline read runs on a
    daemon thread that is abandoned if it is still blocked at the deadline (a read can
    block behind a process in uninterruptible sleep). Any stop on the deadline is
    logged as INCOMPLETE.
    """
    base = base_dir or Path(tempfile.gettempdir())
    start = clock()
    deadline = start + deadline_secs
    try:
        scan = _scan_stale_candidates(base, min_age_secs, start + deadline_secs / 2, clock)
    except OSError:
        logger.warning('Could not scan %s for orphaned agent-spawn temp files — skipping the sweep',
                       base, exc_info=True)
        return 0
    removed, unlinked_every_orphan = 0, True
    if scan.candidates:
        referenced = _names_on_live_cmdlines(proc_root, deadline, clock)
        if referenced is None:
            logger.warning(
                'Deleted nothing from %s: could not establish process liveness from %s by the '
                'deadline (it must be listable, list this process, pid %d, and have every '
                'cmdline readable)',
                base, proc_root, os.getpid(),
            )
            return 0
        orphans = [path for path in scan.candidates if path.name not in referenced]
        removed, unlinked_every_orphan = _unlink_orphans(orphans, deadline, clock)
    if not (scan.complete and unlinked_every_orphan):
        _warn_incomplete(deadline_secs, scan.examined, removed)
    if removed:
        logger.info('Reclaimed %d orphaned agent-spawn temp file(s) from %s', removed, base)
    return removed


class _Scan(NamedTuple):
    candidates: list[Path]
    examined: int
    complete: bool


def _scan_stale_candidates(
    base: Path, min_age_secs: float, deadline: float, clock: Callable[[], float],
) -> _Scan:
    euid = os.geteuid()
    now = time.time()
    candidates: list[Path] = []
    examined = 0
    with os.scandir(base) as entries:
        for entry in entries:
            if clock() >= deadline:
                return _Scan(candidates, examined, complete=False)
            examined += 1
            if not any(kind.matches(entry.name) for kind in SPAWN_TEMP_NAMES):
                continue
            try:
                st = entry.stat(follow_symlinks=False)
            except FileNotFoundError:
                continue
            if stat.S_ISREG(st.st_mode) and st.st_uid == euid and now - st.st_mtime >= min_age_secs:
                candidates.append(Path(entry.path))
    return _Scan(candidates, examined, complete=True)


def _names_on_live_cmdlines(
    proc_root: Path, deadline: float, clock: Callable[[], float],
) -> frozenset[str] | None:
    """Every spawn temp-file name a live process's cmdline mentions, or None when that is
    not known by *deadline*."""
    read: list[list[bytes] | None] = []
    reader = threading.Thread(
        target=lambda: read.append(_read_live_cmdlines(proc_root, deadline, clock)),
        name='spawn-temp-liveness', daemon=True,
    )
    reader.start()
    reader.join(timeout=max(0.0, deadline - clock()))
    cmdlines = read[0] if read else None
    if cmdlines is None:
        return None
    text = os.fsdecode(b'\0'.join(cmdlines))
    return frozenset(name for kind in SPAWN_TEMP_NAMES for name in kind.names_in(text))


def _read_live_cmdlines(
    proc_root: Path, deadline: float, clock: Callable[[], float],
) -> list[bytes] | None:
    try:
        pids = [name for name in os.listdir(proc_root) if name.isdigit()]
    except OSError:
        return None
    if str(os.getpid()) not in pids:
        return None
    cmdlines: list[bytes] = []
    for pid in pids:
        if clock() >= deadline:
            return None
        try:
            cmdlines.append((proc_root / pid / 'cmdline').read_bytes())
        except (FileNotFoundError, ProcessLookupError):
            continue
        except OSError:
            return None
    return cmdlines


def _unlink_orphans(
    orphans: list[Path], deadline: float, clock: Callable[[], float],
) -> tuple[int, bool]:
    """Unlink *orphans* until the deadline; return (removed, whether every one was tried)."""
    removed = 0
    for path in orphans:
        if clock() >= deadline:
            return removed, False
        try:
            path.unlink()
        except FileNotFoundError:
            continue
        except OSError:
            logger.warning('Could not reclaim orphaned agent-spawn temp file %s', path, exc_info=True)
            continue
        removed += 1
    return removed, True


def _warn_incomplete(deadline_secs: float, examined: int, removed: int) -> None:
    logger.warning(
        'Orphaned agent-spawn temp-file sweep stopped on the %.1fs deadline — this sweep is '
        'INCOMPLETE (examined %d, removed %d); whatever it did not reach is left in place.',
        deadline_secs, examined, removed,
    )
