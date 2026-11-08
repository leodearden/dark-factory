"""The on-disk temp files a Claude CLI spawn is handed by argv path: their names, and the
reclamation of the ones a killed spawner orphaned."""

from __future__ import annotations

import logging
import os
import re
import stat
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

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

    def __post_init__(self) -> None:
        shape = re.compile(re.escape(self.prefix) + '[a-z0-9_]{8}' + re.escape(self.suffix))
        object.__setattr__(self, '_shape', shape)

    def mkstemp(self) -> tuple[int, str]:
        return tempfile.mkstemp(prefix=self.prefix, suffix=self.suffix)

    def matches(self, name: str) -> bool:
        return self._shape.fullmatch(name) is not None


SYSPROMPT_TEMP_NAME = SpawnTempName('sysprompt_', '.txt')
MCP_CONFIG_TEMP_NAME = SpawnTempName('mcp_', '.json')
SPAWN_TEMP_NAMES = (SYSPROMPT_TEMP_NAME, MCP_CONFIG_TEMP_NAME)


def sweep_orphaned_spawn_temp_files(
    *,
    base_dir: Path | None = None,
    proc_root: Path = Path('/proc'),
    min_age_secs: float = ORPHAN_MIN_AGE_SECS,
    deadline_secs: float = 5.0,
) -> int:
    """Delete the spawn temp files in *base_dir* that no live process names on its cmdline.

    The per-call ``finally`` unlink in every caller of
    shared/src/shared/cli_invoke.py::build_claude_argv is the eager path; this is the
    SIGKILL-recovery half, run once at process start. Returns the number of files this
    call genuinely removed.

    A file is deleted only when ALL of these hold: its name has a ``SPAWN_TEMP_NAMES``
    kind's exact mkstemp shape; it is a regular file (not a symlink, not a dir); it is
    owned by this euid; it is at least *min_age_secs* old; and its basename appears in no
    live process's ``/proc/<pid>/cmdline`` (substring of the NUL-joined blob, so a path
    embedded in an ``sh -c`` token or a ``--flag=value`` is caught). Why cmdline and not
    fds: shared/tests/test_spawn_temp_files.py::test_keeps_a_stale_file_a_live_process_names_on_its_cmdline.

    No TOCTOU: every invocation, resume included since task 3983, mkstemps FRESH files,
    so no new process can come to reference a file already past *min_age_secs*.
    """
    base = base_dir or Path(tempfile.gettempdir())
    candidates = _stale_candidates(base, min_age_secs)
    if not candidates:
        return 0
    referenced = _referenced_by_live_processes(proc_root, [path.name for path in candidates])
    removed = _unlink_orphans([path for path in candidates if path.name not in referenced])
    if removed:
        logger.info('Reclaimed %d orphaned agent-spawn temp file(s) from %s', removed, base)
    return removed


def _stale_candidates(base: Path, min_age_secs: float) -> list[Path]:
    euid = os.geteuid()
    now = time.time()
    candidates: list[Path] = []
    with os.scandir(base) as entries:
        for entry in entries:
            if not any(kind.matches(entry.name) for kind in SPAWN_TEMP_NAMES):
                continue
            try:
                st = entry.stat(follow_symlinks=False)
            except FileNotFoundError:
                continue
            if stat.S_ISREG(st.st_mode) and st.st_uid == euid and now - st.st_mtime >= min_age_secs:
                candidates.append(Path(entry.path))
    return candidates


def _referenced_by_live_processes(proc_root: Path, names: list[str]) -> frozenset[str]:
    cmdlines: list[bytes] = []
    for pid in (name for name in os.listdir(proc_root) if name.isdigit()):
        try:
            cmdlines.append((proc_root / pid / 'cmdline').read_bytes())
        except (FileNotFoundError, ProcessLookupError):
            continue
    blob = b'\0'.join(cmdlines)
    return frozenset(name for name in names if os.fsencode(name) in blob)


def _unlink_orphans(orphans: list[Path]) -> int:
    removed = 0
    for path in orphans:
        try:
            path.unlink()
        except FileNotFoundError:
            continue
        except OSError:
            logger.warning('Could not reclaim orphaned agent-spawn temp file %s', path, exc_info=True)
            continue
        removed += 1
    return removed
