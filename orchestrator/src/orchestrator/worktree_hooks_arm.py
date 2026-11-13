"""Arm a worktree's stash-guard hooks with the worktree's OWN hooks-armed-guard.sh.

Runs ``<worktree>/scripts/hooks-armed-guard.sh arm <worktree>`` (reify task
6059), whose exit contract is normative in that script's header and is
branched here as ``0 | 2 | *``. Best-effort but loud: it never raises and never
blocks dispatch. GitOps calls it on every worktree hand-off that skips
``seed-warm-lane.sh --fresh-checkout``, which arms on its own.
"""

from __future__ import annotations

import logging
import os
from enum import Enum
from pathlib import Path

from shared.git_async import GitResult, run_git

logger = logging.getLogger(__name__)

HOOKS_ARM_TIMEOUT_SECS: float = 60.0

_GUARD_RELPATH = Path('scripts') / 'hooks-armed-guard.sh'
_PIN_CANNOT_FIX_RC = 2


class HooksArmOutcome(Enum):
    ARMED = 'armed'
    PIN_CANNOT_FIX = 'pin_cannot_fix'
    TIMED_OUT = 'timed_out'
    FAILED = 'failed'
    NO_SCRIPT = 'no_script'
    NOT_EXECUTABLE = 'not_executable'


async def arm_worktree_hooks(
    worktree: Path, *, timeout_secs: float = HOOKS_ARM_TIMEOUT_SECS,
) -> HooksArmOutcome:
    """Arm *worktree*'s hooks; the outcome is logged and returned, never raised."""
    try:
        return await _arm(worktree, timeout_secs)
    except Exception:
        logger.warning(
            'arm_worktree_hooks: could not run the hooks arm for %s', worktree, exc_info=True,
        )
        return HooksArmOutcome.FAILED


async def _arm(worktree: Path, timeout_secs: float) -> HooksArmOutcome:
    script = worktree / _GUARD_RELPATH
    if not script.exists():
        logger.debug('arm_worktree_hooks: no %s, nothing to arm', script)
        return HooksArmOutcome.NO_SCRIPT
    if not (script.is_file() and os.access(script, os.X_OK)):
        logger.warning(
            'arm_worktree_hooks: %s is not an executable file; %s left unarmed',
            script, worktree,
        )
        return HooksArmOutcome.NOT_EXECUTABLE
    result = await run_git(
        [str(script), 'arm', str(worktree)],
        cwd=worktree, timeout=timeout_secs, bounded=False,
    )
    outcome = _outcome_for(result)
    _log_outcome(worktree, outcome, result)
    return outcome


def _outcome_for(result: GitResult) -> HooksArmOutcome:
    if result.timed_out:
        return HooksArmOutcome.TIMED_OUT
    if result.returncode == 0:
        return HooksArmOutcome.ARMED
    if result.returncode == _PIN_CANNOT_FIX_RC:
        return HooksArmOutcome.PIN_CANNOT_FIX
    return HooksArmOutcome.FAILED


def _log_outcome(worktree: Path, outcome: HooksArmOutcome, result: GitResult) -> None:
    stderr = result.stderr.strip()
    if outcome is HooksArmOutcome.ARMED:
        logger.info('arm_worktree_hooks: stash-guard hooks armed in %s', worktree)
    elif outcome is HooksArmOutcome.PIN_CANNOT_FIX:
        logger.warning(
            'arm_worktree_hooks: the hooks pin cannot arm %s (rc=%d); its stash-guard '
            'hooks stay unarmed: %s',
            worktree, result.returncode, stderr,
        )
    elif outcome is HooksArmOutcome.TIMED_OUT:
        logger.warning(
            'arm_worktree_hooks: arm did not finish for %s (%s); its stash-guard '
            'hooks may be unarmed',
            worktree, stderr,
        )
    else:
        logger.warning(
            'arm_worktree_hooks: arm failed for %s (rc=%d); its stash-guard hooks '
            'may be unarmed: %s',
            worktree, result.returncode, stderr,
        )
