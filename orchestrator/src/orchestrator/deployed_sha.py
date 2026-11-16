"""Which commit's code this orchestrator process is running (task 6565, design D1).

The remote gate must run the dispatcher's own code, so the dispatcher must know
which commit that is.  The working tree it was launched from keeps moving
(every landing advances ``HEAD``), but the process's verify wire path is pinned
at the moment ``orchestrator.harness`` is first imported.  This module brackets
that import with two reads of ``HEAD`` and records the answer as a
:class:`DeployedSha`, which ``cli.py::run`` injects downward; there is no
module-level copy of it.

It imports nothing from the wire path, so the wire path may import it.
"""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Tracked files under these paths make up the verify wire path.  The main
# checkout always carries untracked files, so only tracked edits count.
WIRE_PATH_PATHSPECS: tuple[str, ...] = ('orchestrator/src', 'shared/src')


def resolve_code_checkout(start: Path | None = None) -> Path | None:
    """Walk up from *start* to the checkout root (the ``.git`` marker).

    *start* defaults to this module's own file, so the running orchestrator
    resolves the checkout that holds its OWN code.

    Returns the first ancestor directory containing a ``.git`` entry (a
    directory in a main checkout or a clone, a FILE in a linked worktree), or
    ``None`` when there is none, as for an installed, non-editable package.
    """
    here = Path(__file__).resolve() if start is None else start
    # A file start begins the walk at its parent; a directory start is itself a
    # candidate.  Either way we ascend to the filesystem root.
    candidates = [here, *here.parents] if here.is_dir() else list(here.parents)
    for d in candidates:
        if (d / '.git').exists():
            return d
    return None


@dataclass(frozen=True)
class DeployedSha:
    """The commit the process's verify wire path was imported from.

    It measures the import window of ``orchestrator.harness`` (which loads every
    wire module), not every module the process will ever load: a module first
    imported later comes from whatever the tree holds then.

    ``sha`` is ``HEAD`` read just AFTER that import.  ``indeterminate`` is True
    when ``HEAD`` moved during the import, so some wire modules may come from
    the earlier commit.  ``dirty_wire_path`` is True when a tracked file under
    :data:`WIRE_PATH_PATHSPECS` had uncommitted edits just before the import,
    so the loaded code is not exactly ``sha``.  Either flag is logged, emitted
    as ``deployed_sha_indeterminate`` and shown in the fleet heartbeat; the
    remote gate still runs ``sha`` (Leo, 2026-10-20).
    """

    sha: str
    indeterminate: bool
    dirty_wire_path: bool

    @property
    def flagged(self) -> bool:
        """True when the loaded code may differ from ``sha``."""
        return self.indeterminate or self.dirty_wire_path

    def to_wire(self) -> dict[str, str | bool]:
        """The JSON-native form carried by heartbeats and events."""
        return {
            'sha': self.sha,
            'indeterminate': self.indeterminate,
            'dirty_wire_path': self.dirty_wire_path,
        }


def _git(checkout: Path, *args: str) -> str:
    return subprocess.run(
        ['git', '-C', str(checkout), *args],
        capture_output=True, text=True, check=True,
    ).stdout


def checkout_head(checkout: Path) -> str:
    """``git rev-parse HEAD`` of *checkout*; raises on any git failure."""
    return _git(checkout, 'rev-parse', 'HEAD').strip()


def wire_path_is_dirty(checkout: Path) -> bool:
    """True iff a TRACKED file under the wire path has uncommitted changes."""
    status = _git(checkout, 'status', '--porcelain', '-uno', '--', *WIRE_PATH_PATHSPECS)
    return bool(status.strip())


def capture_deployed_sha(
    checkout: Path | None,
    import_wire_path: Callable[[], object],
) -> DeployedSha | None:
    """Run *import_wire_path* bracketed by two ``HEAD`` reads of *checkout*.

    Always calls *import_wire_path* exactly once.  Returns ``None`` when there
    is no checkout or git cannot read it; the caller decides whether a process
    without a deployed sha may run (it may not when a remote runner is
    enabled: the remote gate needs a sha to build).
    """
    if checkout is None:
        import_wire_path()
        return None
    try:
        before = checkout_head(checkout)
        dirty = wire_path_is_dirty(checkout)
    except (OSError, subprocess.CalledProcessError) as exc:
        logger.warning('deployed sha: cannot read %s before the import: %s', checkout, exc)
        import_wire_path()
        return None
    import_wire_path()
    try:
        after = checkout_head(checkout)
    except (OSError, subprocess.CalledProcessError) as exc:
        logger.warning('deployed sha: cannot read %s after the import: %s', checkout, exc)
        return None
    return DeployedSha(sha=after, indeterminate=before != after, dirty_wire_path=dirty)
