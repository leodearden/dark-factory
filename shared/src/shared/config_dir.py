"""Per-task isolated Claude Code config directory.

Each task gets its own ``CLAUDE_CONFIG_DIR`` so that:
- ``.credentials.json`` can be rewritten per-account without races
- ``--resume`` reads the correct credential (not the global one)
- Sessions are stored per-task, avoiding cross-task contamination
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

_HOME_CLAUDE = Path.home() / '.claude'

# Files to symlink from ~/.claude/ into the per-task config dir.
# Provides settings/hooks without duplicating config.
# Do NOT include projects/, sessions/, telemetry/ — those must be per-task.
_SYMLINK_FILES = ['settings.json', 'settings.local.json']

# Single source of truth for the config-dir naming template. Both
# TaskConfigDir.__init__ (which constructs the names) and
# sweep_stale_pid_dirs (which reclaims them) key off this, so the
# construction template and the sweep prefix cannot drift apart.
CONFIG_DIR_PREFIX = 'claude-config-'

# Trailing `-<digits>` — the owning process's PID, as appended by callers
# that embed one (e.g. UsageGate's `usage-gate-probe-<account>-<pid>`).
_PID_SUFFIX_RE = re.compile(r'-(\d+)$')


def _pid_alive(pid: int) -> bool:
    """Return True if the process identified by *pid* is alive.

    Copied (not imported) from ``orchestrator/session_registry.py`` — and
    ultimately ``orchestrator/harness.py`` — to keep ``shared`` at the bottom
    of the dependency stack with no import edge on ``orchestrator``. This is
    the fourth instance of that deliberate copy-don't-import pattern.

    - Returns False for pid <= 0 (invalid).
    - Uses ``os.kill(pid, 0)``: success -> alive; ProcessLookupError -> dead;
      PermissionError -> alive (visible but unsignalable — load-bearing here,
      since a dir owned by another user's live process must never be
      reclaimed); other OSError -> treated as dead.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def sweep_stale_pid_dirs(prefix: str, *, base_dir: Path | None = None) -> int:
    """Remove ``{prefix}*`` directories whose embedded trailing PID is dead.

    Bounds the population of per-PID scratch config dirs at
    (live processes x dirs-per-process). Teardown hooks — ``atexit``,
    context managers, an explicit ``shutdown()`` — cover only clean exits;
    nothing survives SIGKILL, and the fleet SIGKILLs and restarts its units
    routinely. Reclaiming *other* processes' dead-PID leftovers at startup is
    the only self-healing half. Returns the number of directories removed.

    Best-effort by contract: a glob matching nothing is a silent no-op and the
    function never raises.

    Safety — three independent conservative guards, all of which must pass
    before anything is deleted:

    1. **No TOCTOU.** A scratch dir only comes into existence *after* its
       owning process exists, so "dir present + PID dead" implies the owner is
       genuinely gone. There is no window in which a live owner's dir looks
       dead.
    2. **PID reuse only ever under-deletes.** A recycled PID makes a dead
       owner look alive, so we skip; a later sweep reclaims it. It can never
       cause a wrong deletion.
    3. **Unattributable dirs are never touched.** No parseable trailing
       ``-<digits>`` means we cannot attribute the dir to a process, so we
       leave it alone.

    Blast radius is *prefix-scoped* — the caller passes the full prefix (e.g.
    ``CONFIG_DIR_PREFIX + 'usage-gate-probe-'``), never the bare
    ``CONFIG_DIR_PREFIX``: per-task config dirs and test fixtures share the
    ``claude-config-`` stem and must not be swept. Symlinks and plain files
    matching the prefix are skipped.

    ASSUMPTION — a **shared PID namespace**. ``os.kill(pid, 0)`` is only
    meaningful if the sweeper and the dir's owner see the same PID space.
    That holds today: the sandbox backend is ``landlock`` (a filesystem LSM,
    no namespaces) and ``--unshare-pid`` appears nowhere in the tree. A future
    container/PID-namespace backend BREAKS this and must revisit the function
    — there is no cheap runtime way to detect the mismatch, so the invariant
    is documented here rather than defended in code.
    """
    base = base_dir or Path(tempfile.gettempdir())
    removed = 0
    for path in base.glob(f'{prefix}*'):
        if path.is_symlink() or not path.is_dir():
            continue
        match = _PID_SUFFIX_RE.search(path.name)
        if match is None:
            continue
        if _pid_alive(int(match.group(1))):
            continue
        shutil.rmtree(path, ignore_errors=True)
        removed += 1
    return removed


class TaskConfigDir:
    """Manages an isolated ``CLAUDE_CONFIG_DIR`` for a single task.

    On creation, symlinks shared config from ``~/.claude/`` and provides
    a ``write_credentials()`` method to set per-invocation OAuth tokens.
    """

    def __init__(self, task_id: str, base_dir: Path | None = None):
        base = base_dir or Path(tempfile.gettempdir())
        self._dir = base / f'{CONFIG_DIR_PREFIX}{task_id}'
        self._dir.mkdir(parents=True, exist_ok=True)
        self._setup_symlinks()

    def _setup_symlinks(self) -> None:
        """Symlink shared config files from ~/.claude/."""
        for name in _SYMLINK_FILES:
            src = _HOME_CLAUDE / name
            dst = self._dir / name
            if src.exists() and not dst.exists():
                try:
                    dst.symlink_to(src)
                except OSError as e:
                    logger.warning('Failed to symlink %s → %s: %s', src, dst, e)

    def write_credentials(self, oauth_token: str) -> None:
        """Write ``.credentials.json`` with the given OAuth token."""
        # Recreate dir if a previous `claude` CLI process removed it.
        # Diagnosed 2026-04-13: reviewer_comprehensive hit FileNotFoundError
        # on the second review cycle — root cause unclear, but defensive
        # mkdir prevents the failure.
        if not self._dir.exists():
            logger.warning('Config dir %s was deleted — recreating', self._dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        creds = {
            'claudeAiOauth': {
                'accessToken': oauth_token,
            },
        }
        creds_path = self._dir / '.credentials.json'
        creds_path.write_text(json.dumps(creds))
        # Restrict permissions (token is sensitive)
        creds_path.chmod(0o600)

    @property
    def path(self) -> Path:
        """Absolute path to the config directory."""
        return self._dir

    def cleanup(self) -> None:
        """Remove the config directory and all contents."""
        shutil.rmtree(self._dir, ignore_errors=True)
