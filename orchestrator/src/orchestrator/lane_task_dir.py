"""Evict a warm lane's previous occupant's state from the lane's in-worktree ``.task/``.

A pooled lane serves many tasks in turn, and ``.task/`` is untracked, so
``git checkout -f`` leaves the previous occupant's ``plan.json``,
``iterations.jsonl``, ``reviews/`` and per-task ``claude-config-<id>`` dirs in
place for the next one to inherit (reify esc-4920-163: ``_lane-26`` 4949 to
4920 contamination).

``git clean -xfd`` is not a substitute. A single ``-f`` skips nested git
repositories, and a Claude config dir always holds one
(``plugins/marketplaces/<name>/.git``), so the clean leaves a
``claude-config-<prev>`` husk that boot recovery reports as
``session_config_dir_ambiguous`` (task 7329). The clean also deletes the
occupant's transcripts unarchived. So each foreign config dir is first made
durable through ``shared/src/shared/transcript_archive.py::archive_before_delete``,
and everything else goes through
``orchestrator/src/orchestrator/tree_removal.py::remove_tree``, off the event
loop (INV-8, task 7366).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from shared.config_dir import CONFIG_DIR_PREFIX
from shared.transcript_archive import archive_before_delete, resolve_archive_root

from orchestrator.config import TranscriptArchiveConfig
from orchestrator.tree_removal import remove_tree

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreviousOccupantEviction:
    """Entry names of ``.task/`` that were removed and that survived, both sorted."""

    evicted: tuple[str, ...] = ()
    retained: tuple[str, ...] = ()


def archive_root_for(
    transcript_archive: TranscriptArchiveConfig | None, project_root: Path,
) -> Path | None:
    """The durable transcript archive root, or ``None`` when archiving is off."""
    if transcript_archive is None or not transcript_archive.enabled:
        return None
    return resolve_archive_root(project_root, transcript_archive.root)


async def evict_previous_occupant(
    task_dir: Path, *, keep_task_id: str | None, archive_root: Path | None,
) -> PreviousOccupantEviction:
    """Remove every entry of *task_dir* except ``claude-config-<keep_task_id>``.

    Total: a failure is logged and reported in ``retained``, never raised, so
    lane hygiene cannot fault an acquire. Cancellation still propagates.
    """
    kept_name = None if keep_task_id is None else f'{CONFIG_DIR_PREFIX}{keep_task_id}'
    visited = [entry for entry in _list_entries(task_dir) if entry.name != kept_name]
    for entry in visited:
        await _evict_entry(entry, archive_root)
    result = PreviousOccupantEviction(
        evicted=tuple(e.name for e in visited if not os.path.lexists(e)),
        retained=tuple(e.name for e in visited if os.path.lexists(e)),
    )
    _log_outcome(task_dir, keep_task_id, result)
    return result


def _list_entries(task_dir: Path) -> list[Path]:
    if not task_dir.is_dir():
        return []
    try:
        return sorted(task_dir.iterdir())
    except OSError:
        logger.warning('lane_task_dir: cannot list %s', task_dir, exc_info=True)
        return []


async def _evict_entry(entry: Path, archive_root: Path | None) -> None:
    try:
        if archive_root is not None and _is_real_config_dir(entry):
            # Inline and synchronous by archive_before_delete's own contract:
            # an await here would be a cancellation point between archive and delete.
            archive_before_delete(
                entry, entry.name[len(CONFIG_DIR_PREFIX):], archive_root=archive_root,
            )
        else:
            await remove_tree(entry, ignore_errors=True)
    except Exception:
        logger.warning('lane_task_dir: failed to evict %s', entry, exc_info=True)


def _is_real_config_dir(entry: Path) -> bool:
    return (
        not entry.is_symlink()
        and entry.is_dir()
        and entry.name.startswith(CONFIG_DIR_PREFIX)
    )


def _log_outcome(
    task_dir: Path, keep_task_id: str | None, result: PreviousOccupantEviction,
) -> None:
    if result.evicted:
        logger.info(
            'lane_task_dir: evicted previous occupant entries from %s: %s',
            task_dir, ', '.join(result.evicted),
        )
    if result.retained:
        logger.warning(
            'lane_task_dir: %s (keeping task %s) still holds previous occupant '
            'entries it could not evict: %s',
            task_dir, keep_task_id, ', '.join(result.retained),
        )
