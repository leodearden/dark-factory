"""Per-task isolated Claude Code config directory.

Each task gets its own ``CLAUDE_CONFIG_DIR`` so that:
- ``.credentials.json`` can be rewritten per-account without races
- ``--resume`` reads the correct credential (not the global one)
- Sessions are stored per-task, avoiding cross-task contamination
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import re
import shutil
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

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

# Paths already registered for atexit teardown in this process. The same path
# is constructed repeatedly within one process — a probe dir is named
# `usage-gate-probe-<account>-<pid>`, and orchestrator/evals/runner.py builds a
# fresh UsageGate per eval run — so without this ledger the atexit table would
# accumulate one duplicate entry (each pinning a Path) per gate x account, and
# every duplicate would re-run rmtree on the same path at shutdown. One hook
# per path is sufficient because the hook targets the resolved Path, not the
# instance: a later TaskConfigDir at the same path is already covered by it.
_atexit_registered_dirs: set[Path] = set()


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


def sweep_stale_pid_dirs(
    prefix: str,
    *,
    base_dir: Path | None = None,
    min_age_secs: float = 300.0,
    deadline_secs: float = 5.0,
) -> int:
    """Remove ``{prefix}*`` directories whose embedded trailing PID is dead.

    Bounds the population of per-PID scratch config dirs at
    (live processes x dirs-per-process). Teardown hooks — ``atexit``,
    context managers, an explicit ``shutdown()`` — cover only clean exits;
    nothing survives SIGKILL, and the fleet SIGKILLs and restarts its units
    routinely. Reclaiming *other* processes' dead-PID leftovers at startup is
    the only self-healing half. Returns the number of directories this call
    genuinely removed — an entry that could not be reclaimed is logged at
    WARNING and excluded from the count, so the operator-visible tally can
    never overstate the drain.

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
    4. **An mtime floor** (``min_age_secs``) covers the residual window where
       a dir was created microseconds before its owner died, plus any clock
       skew.

    ``deadline_secs`` bounds *blocking time*, not removals. Callers run this
    synchronously on the event-loop thread at startup, and the pathological
    /tmp this guards against has a 40 MB directory inode — a full readdir plus
    hundreds of thousands of ``rmtree`` calls could stall startup for minutes.
    A wall-clock deadline is the right bound because the population still
    drains fully across successive process starts, whereas a hard removal cap
    would leave a permanent residue. A bounded stop is logged at WARNING with
    examined/removed counts and says so explicitly — it must never read as
    "swept everything".

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
    deadline = time.monotonic() + deadline_secs
    now = time.time()
    removed = 0
    examined = 0
    try:
        for path in base.glob(f'{prefix}*'):
            if time.monotonic() >= deadline:
                logger.warning(
                    'sweep_stale_pid_dirs(%s): stopped on the %.1fs deadline — this sweep '
                    'is INCOMPLETE (examined %d, removed %d). Remaining stale dirs are '
                    'reclaimed by subsequent sweeps.',
                    prefix, deadline_secs, examined, removed,
                )
                break
            examined += 1
            if path.is_symlink() or not path.is_dir():
                continue
            match = _PID_SUFFIX_RE.search(path.name)
            if match is None:
                continue
            if _pid_alive(int(match.group(1))):
                continue
            try:
                if now - path.stat().st_mtime < min_age_secs:
                    continue
                # Deliberately NOT ignore_errors=True. A genuinely unremovable
                # dir (EACCES, EBUSY, an immutable inode, a partially-removed
                # tree) must raise so the handler below is live: with errors
                # ignored, `removed` would count a removal that never happened
                # and the operator-visible "reclaimed N" INFO would report a
                # drain that did not occur.
                shutil.rmtree(path)
            except FileNotFoundError:
                # Raced by a concurrent sweeper — the fleet restarts its units
                # together, so several gates can sweep at once — or by the OS
                # tmp reaper. Already reclaimed: not ours to count, and not a
                # failure worth a WARNING either.
                continue
            except OSError:
                logger.warning(
                    'sweep_stale_pid_dirs(%s): failed to reclaim %s — skipping it and '
                    'continuing the sweep', prefix, path, exc_info=True,
                )
                continue
            removed += 1
    except OSError:
        logger.warning(
            'sweep_stale_pid_dirs(%s): could not scan %s — skipping the sweep',
            prefix, base, exc_info=True,
        )
    return removed


#: Prefixes already swept in this process, by :func:`sweep_stale_pid_dirs_once`.
#: PER-PREFIX rather than one boolean — see that function's docstring.
_swept_prefixes: set[str] = set()


def sweep_stale_pid_dirs_once(
    prefix: str,
    *,
    sweep: Callable[..., int],
    on_reclaimed: Callable[[int], None] | None = None,
    on_failure: Callable[[BaseException], None] | None = None,
    **sweep_kwargs: Any,
) -> int:
    """Run *sweep* over *prefix* at most once per process. Never raises.

    Returns the number of dirs *sweep* reported removing — 0 when this prefix was
    already swept in this process, and 0 on failure.

    WHY ONCE. The sweep reclaims OTHER (dead) processes' leftovers, not this
    one's, so its result cannot change during a process's life in any way this
    process caused. Re-running it per gate construction or per probe re-scans a
    potentially 40 MB /tmp directory inode for no benefit, on the event-loop
    thread at startup.

    WHY PER-PREFIX. The one-shot state is keyed by prefix rather than being a
    single module-level flag because the callers sweep DIFFERENT prefixes
    (``usage-gate-probe-``, ``startup-probe-``). Under one flag, whichever caller
    initialised first would mark the process as swept and suppress the other's
    sweep entirely — silently converting the probe's SIGKILL-recovery half into a
    no-op inside any process that also builds a UsageGate.

    WHY THE MARK IS SET BEFORE THE CALL. So a sweep that raises every time cannot
    re-run on every subsequent construction. The cost of the ordering is that one
    failure forfeits the sweep for the life of the process; the next process start
    retries, and the population still drains.

    WHY THE ``except`` IS BROAD. ``sweep_stale_pid_dirs`` already contains
    ``OSError`` internally, so anything reaching here is UNFORESEEN — a future
    bug, a pathological tree, a mocked side effect in a sibling suite. Tmp hygiene
    must never be able to fail orchestrator startup or a probe capture that costs
    real money to retake, both of which are strictly worse outcomes than leaving a
    stale /tmp dir behind.

    WHY REPORTING IS INJECTED. The two callers report through genuinely different
    sinks and neither converts without loss: ``usage_gate`` logs under its own
    logger name with ``exc_info=True`` (its tests assert on those caplog records),
    while the probe deliberately has no logger and prints to stderr (its tests
    read capsys). ``on_reclaimed`` fires only on a NON-ZERO count — silent in the
    steady state so an operator sees the population draining rather than
    rebuilding — and ``on_failure`` receives the exception INSTANCE, so a caller
    can interpolate it or let ``exc_info`` pick it up. Both are optional.

    *sweep* is REQUIRED and keyword-only, and is deliberately NOT defaulted to
    :func:`sweep_stale_pid_dirs`. A default binds THIS module's global at ``def``
    time, which would make ``shared.config_dir`` the single interception point;
    three existing fixtures instead patch each CALLER's module-level
    ``sweep_stale_pid_dirs`` name, and one of them
    (``test_startup_completion_probe.py::_confine_stale_dir_sweep``, autouse and
    module-wide) is the only thing stopping that suite from rmtree-ing real
    ``/tmp/claude-config-startup-probe-*`` dirs. Under a def-time default all
    three would silently stop intercepting: green tests, real deletions. Callers
    therefore pass their own module-level name explicitly, which is a call-time
    global lookup and keeps every existing patch target working.

    Extra keyword arguments are forwarded verbatim to *sweep*.
    """
    if prefix in _swept_prefixes:
        return 0
    _swept_prefixes.add(prefix)
    try:
        reclaimed = sweep(prefix, **sweep_kwargs)
        if reclaimed and on_reclaimed is not None:
            on_reclaimed(reclaimed)
        return reclaimed
    except Exception as exc:  # noqa: BLE001  (deliberately broad — see docstring)
        if on_failure is not None:
            on_failure(exc)
        return 0


def reset_sweep_once_state(prefix: str | None = None) -> None:
    """Forget that *prefix* (or every prefix) was swept in this process.

    A TEST hook for simulating a fresh process, not production API: nothing in
    production should ever want the sweep to run twice, which is the whole point
    of :func:`sweep_stale_pid_dirs_once`.

    ``discard`` semantics — resetting a prefix that was never swept is a silent
    no-op, so a fixture calling this in both setup and teardown cannot become
    order-dependent.
    """
    if prefix is None:
        _swept_prefixes.clear()
    else:
        _swept_prefixes.discard(prefix)


class TaskConfigDir:
    """Manages an isolated ``CLAUDE_CONFIG_DIR`` for a single task.

    On creation, symlinks shared config from ``~/.claude/`` and provides
    a ``write_credentials()`` method to set per-invocation OAuth tokens.
    """

    def __init__(
        self,
        task_id: str,
        base_dir: Path | None = None,
        *,
        cleanup_at_exit: bool = False,
    ):
        """Create (or adopt) the config dir for *task_id*.

        ``cleanup_at_exit`` registers a best-effort ``atexit`` teardown,
        mirroring ``neutral_cwd.py``: it binds the resolved ``Path`` only —
        never ``self`` — so the atexit table cannot keep this object (and
        transitively an owning ``UsageGate`` and its account tokens) alive
        for the life of the process. ``ignore_errors=True`` makes it a
        harmless no-op after an explicit ``cleanup()``. Registration is
        deduped by resolved path (``_atexit_registered_dirs``), so repeatedly
        constructing the same path in one process adds exactly one hook.

        This is the CLEAN-EXIT half only. No atexit hook survives SIGKILL,
        and the fleet SIGKILLs and restarts its units routinely — what
        actually bounds the on-disk population is ``sweep_stale_pid_dirs``,
        run by the next process to start.

        Defaults to False, and that default is load-bearing: per-task and
        per-investigation config dirs (created under a worktree's ``.task/``)
        are deliberately preserved so the session JSONL inside them survives
        for transcript archival and ``--resume``. Only ephemeral scratch dirs
        under /tmp — the ``UsageGate`` probe dirs — opt in.
        """
        base = base_dir or Path(tempfile.gettempdir())
        self._dir = base / f'{CONFIG_DIR_PREFIX}{task_id}'
        self._dir.mkdir(parents=True, exist_ok=True)
        if cleanup_at_exit and self._dir not in _atexit_registered_dirs:
            _atexit_registered_dirs.add(self._dir)
            atexit.register(shutil.rmtree, self._dir, ignore_errors=True)
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
