"""Recognise eval-lane provenance on an escalation filing.

An eval-lane escalation is never a production signal: an adversarial fixture's
refuse-and-escalate is the measured behaviour, and the measurement is scored
from the eval cell's result artifacts, never from the escalation queue. This
module is the single recognition site; its consumers are the escalation filing
gate (``escalation/src/escalation/server.py::_chokepoint_or_submit``),
``escalation/src/escalation/server.py::promote_to_l2``, the orphan L0 reaper
(``orchestrator/src/orchestrator/harness.py::Harness._reap_orphan_l0_escalations``),
and the fixture loader that enforces the grammar at run time
(``orchestrator/src/orchestrator/evals/runner.py::load_task``).

The task-id signal is a NARROW grammar, not "any non-numeric task id": the
production queue's non-numeric ids are harness sentinels (``task-path-guard``,
``__scheduler__``, ``main-sweep-<hex>`` …) whose L2s must still reach a human.

Pure and stdlib-only, so any lightweight importer can use it.
"""

from __future__ import annotations

import os
import re
from pathlib import PurePath

#: ``<repo>_task_<n>[_<suffix>]`` — minted by
#: ``orchestrator/src/orchestrator/evals/task_sampler.py::build_fixture_record``,
#: plus the hand-authored ``_adv_<kind>`` adversarial variants.
CORPUS_FIXTURE_ID_RE = re.compile(r'^[a-z0-9]+_task_\d+(?:_[a-z0-9_]+)?$')

#: ``shadow_<task_id>_<cell_id>`` — minted by
#: ``orchestrator/src/orchestrator/evals/live_fixture.py::build_live_fixture``.
SHADOW_FIXTURE_ID_RE = re.compile(r'^shadow_\S+_[0-9A-Za-z]+$')

#: Eval worktree root directory names, matched against whole path components:
#: the sibling ``<repo>-eval-worktrees/`` minted by
#: ``orchestrator/src/orchestrator/evals/snapshots.py::eval_worktree_root`` and the
#: legacy in-repo ``.eval-worktrees/``. A component that merely contains
#: ``eval-worktree`` (a project named ``eval-worktree-tools``) is production.
EVAL_WORKTREE_ROOT_SUFFIX = '-eval-worktrees'
LEGACY_EVAL_WORKTREE_ROOT = '.eval-worktrees'


def is_eval_fixture_task_id(task_id: str | None) -> bool:
    """True when *task_id* is an eval fixture id (corpus or live shadow)."""
    candidate = (task_id or '').strip()
    if not candidate:
        return False
    return bool(
        CORPUS_FIXTURE_ID_RE.fullmatch(candidate)
        or SHADOW_FIXTURE_ID_RE.fullmatch(candidate)
    )


def is_eval_worktree_path(path: str | os.PathLike[str] | None) -> bool:
    """True when *path* lies under an eval worktree root."""
    if not path:
        return False
    return any(
        part == LEGACY_EVAL_WORKTREE_ROOT or part.endswith(EVAL_WORKTREE_ROOT_SUFFIX)
        for part in PurePath(path).parts
    )


def eval_lane_provenance(
    task_id: str | None, worktree: str | os.PathLike[str] | None = None
) -> str | None:
    """Name the eval-lane signal a filing carries, or None for a production filing.

    The id is checked first: it is the only signal that survives the orphan
    reaper's ``worktree=None`` L1 re-escalation.
    """
    if is_eval_fixture_task_id(task_id):
        return f'fixture-task-id:{task_id}'
    if is_eval_worktree_path(worktree):
        return f'eval-worktree:{worktree}'
    return None
