"""Soak wrapper that drives two task backends in lock-step.

Routes every call to both *primary* and *secondary*, compares the
responses, and logs structured divergences. The caller's return value is
always the *primary*'s — secondary failures are swallowed (logged) so the
wrapper is safe to flip on/off without changing externally-observable
behaviour.

This is the engine for the cutover-soak between Taskmaster (the legacy
default primary) and SqliteTaskBackend (the eventual successor). Once the
divergence log goes silent for a soak window, the operator flips the
``dual_compare_primary`` config field and runs the inverse comparison
before retiring the legacy backend entirely.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from typing import Any

from fused_memory.backends.task_backend_protocol import TaskBackendProtocol

logger = logging.getLogger(__name__)


def _summarize(value: Any, *, limit: int = 240) -> str:
    """Render *value* as a stable JSON-ish string, clipped at *limit* chars.

    Used in divergence log lines so operators can spot wire-shape drift at
    a glance without flooding the log with multi-KB DTOs.
    """
    try:
        rendered = json.dumps(value, sort_keys=True, default=str)
    except (TypeError, ValueError):
        rendered = repr(value)
    if len(rendered) > limit:
        rendered = rendered[:limit] + '…'
    return rendered


class DualCompareBackend:
    """Wraps two backends; serves the primary, mirrors writes to the secondary.

    Reads compare both responses; writes apply both then compare. The
    secondary's exceptions never propagate — they are logged so a buggy
    secondary can never break the production hot path.
    """

    def __init__(
        self,
        primary: TaskBackendProtocol,
        secondary: TaskBackendProtocol,
        *,
        primary_label: str = 'primary',
        secondary_label: str = 'secondary',
    ) -> None:
        self.primary = primary
        self.secondary = secondary
        self.primary_label = primary_label
        self.secondary_label = secondary_label
        self.divergence_count = 0

    # ── Lifecycle ──────────────────────────────────────────────────────

    @property
    def connected(self) -> bool:
        return self.primary.connected

    @property
    def restart_count(self) -> int:
        return self.primary.restart_count

    async def start(self) -> None:
        await self.primary.start()
        try:
            await self.secondary.start()
        except Exception as exc:
            logger.warning(
                'dual_compare: secondary.start() failed: %s; comparator disabled',
                exc,
            )

    async def initialize(self) -> None:
        await self.start()

    async def ensure_connected(self) -> None:
        await self.primary.ensure_connected()

    async def close(self) -> None:
        await self.primary.close()
        try:
            await self.secondary.close()
        except Exception as exc:
            logger.warning('dual_compare: secondary.close() failed: %s', exc)

    async def is_alive(self) -> tuple[bool, str | None]:
        return await self.primary.is_alive()

    # ── Comparator core ────────────────────────────────────────────────

    async def _dispatch(
        self, method_name: str, args: tuple, kwargs: dict,
        *, normalize: Callable[[Any], Any] | None = None,
    ) -> Any:
        """Run *method_name* on both backends; log divergences; surface primary.

        Both calls run concurrently via ``asyncio.gather(return_exceptions=True)``
        so an outer cancellation propagates to both subtasks (no detached
        secondary survives) and exceptions from the secondary are captured
        without disturbing the primary's outcome.
        """
        primary_method = getattr(self.primary, method_name)
        secondary_method = getattr(self.secondary, method_name)

        primary_outcome, secondary_outcome = await asyncio.gather(
            primary_method(*args, **kwargs),
            secondary_method(*args, **kwargs),
            return_exceptions=True,
        )

        primary_exc = primary_outcome if isinstance(primary_outcome, BaseException) else None
        primary_value = None if primary_exc is not None else primary_outcome
        secondary_exc = (
            secondary_outcome if isinstance(secondary_outcome, BaseException) else None
        )
        secondary_value = None if secondary_exc is not None else secondary_outcome

        self._compare(
            method_name, args, kwargs,
            primary_value, primary_exc,
            secondary_value, secondary_exc,
            normalize=normalize,
        )

        if primary_exc is not None:
            raise primary_exc
        return primary_value

    def _compare(
        self,
        method_name: str,
        args: tuple,
        kwargs: dict,
        primary_value: Any,
        primary_exc: BaseException | None,
        secondary_value: Any,
        secondary_exc: BaseException | None,
        *,
        normalize: Callable[[Any], Any] | None,
    ) -> None:
        """Log a structured divergence line if the two responses disagree."""
        primary_kind = type(primary_exc).__name__ if primary_exc else 'value'
        secondary_kind = type(secondary_exc).__name__ if secondary_exc else 'value'

        if primary_exc is not None or secondary_exc is not None:
            if primary_kind != secondary_kind:
                self._log_divergence(
                    method_name, args, kwargs,
                    f'{self.primary_label}={primary_kind}({primary_exc})',
                    f'{self.secondary_label}={secondary_kind}({secondary_exc})',
                )
            return

        p = normalize(primary_value) if normalize else primary_value
        s = normalize(secondary_value) if normalize else secondary_value
        if p == s:
            return
        self._log_divergence(
            method_name, args, kwargs,
            f'{self.primary_label}={_summarize(p)}',
            f'{self.secondary_label}={_summarize(s)}',
        )

    def _log_divergence(
        self,
        method_name: str,
        args: tuple,
        kwargs: dict,
        primary_repr: str,
        secondary_repr: str,
    ) -> None:
        self.divergence_count += 1
        logger.warning(
            'dual_compare.divergence method=%s args=%s kwargs=%s %s %s',
            method_name,
            _summarize(list(args)),
            _summarize(kwargs),
            primary_repr,
            secondary_repr,
        )

    # ── Read methods ───────────────────────────────────────────────────

    async def get_tasks(self, project_root: str, tag: str | None = None):
        return await self._dispatch(
            'get_tasks', (project_root,), {'tag': tag},
            normalize=_normalize_tasks_tree,
        )

    async def get_task(
        self, task_id: str, project_root: str, tag: str | None = None,
    ):
        return await self._dispatch(
            'get_task', (task_id, project_root), {'tag': tag},
            normalize=_normalize_task,
        )

    # ── Mutation methods ───────────────────────────────────────────────

    async def set_task_status(
        self,
        task_id: str,
        status: str,
        project_root: str,
        tag: str | None = None,
    ):
        return await self._dispatch(
            'set_task_status',
            (task_id, status, project_root),
            {'tag': tag},
            normalize=_normalize_set_status,
        )

    async def add_task(self, project_root: str, **kwargs):
        # add_task mints an id — the two backends won't agree on that, so
        # we strip ids out of the comparator and only check that both either
        # succeeded or both raised.
        return await self._dispatch(
            'add_task', (project_root,), kwargs,
            normalize=_normalize_id_only,
        )

    async def update_task(
        self,
        task_id: str,
        project_root: str,
        prompt: str | None = None,
        metadata: str | None = None,
        append: bool = False,
        tag: str | None = None,
    ):
        return await self._dispatch(
            'update_task',
            (task_id, project_root),
            {'prompt': prompt, 'metadata': metadata, 'append': append, 'tag': tag},
            normalize=_normalize_id_only,
        )

    async def add_subtask(self, parent_id: str, project_root: str, **kwargs):
        return await self._dispatch(
            'add_subtask',
            (parent_id, project_root),
            kwargs,
            normalize=_normalize_id_only,
        )

    async def remove_task(
        self, task_id: str, project_root: str, tag: str | None = None,
    ):
        return await self._dispatch(
            'remove_task',
            (task_id, project_root),
            {'tag': tag},
            normalize=_normalize_remove,
        )

    async def add_dependency(
        self,
        task_id: str,
        depends_on: str,
        project_root: str,
        tag: str | None = None,
    ):
        return await self._dispatch(
            'add_dependency',
            (task_id, depends_on, project_root),
            {'tag': tag},
            normalize=_normalize_id_only,
        )

    async def remove_dependency(
        self,
        task_id: str,
        depends_on: str,
        project_root: str,
        tag: str | None = None,
    ):
        return await self._dispatch(
            'remove_dependency',
            (task_id, depends_on, project_root),
            {'tag': tag},
            normalize=_normalize_id_only,
        )

    async def validate_dependencies(
        self, project_root: str, tag: str | None = None,
    ):
        return await self._dispatch(
            'validate_dependencies', (project_root,), {'tag': tag},
            normalize=lambda v: 'ok' if 'success' in v.get('message', '').lower() else 'fail',
        )


# ── Normalisers ───────────────────────────────────────────────────────
#
# The comparator deliberately throws away noisy fields that we know will
# differ between Taskmaster (which records granular timestamps + telemetry)
# and SqliteTaskBackend (which doesn't). The bar is "do the two backends
# agree on the task semantics?" — not "do they emit byte-identical wire."
# Each normaliser returns a stable, comparable shape; mismatches in the
# stripped fields will not surface as divergences.


_VOLATILE_TASK_FIELDS = {'updatedAt', 'metadata'}


def _normalize_task(task: Any) -> Any:
    if not isinstance(task, dict):
        return task
    out = {k: v for k, v in task.items() if k not in _VOLATILE_TASK_FIELDS}
    if 'subtasks' in out and isinstance(out['subtasks'], list):
        out['subtasks'] = [_normalize_task(s) for s in out['subtasks']]
    if 'dependencies' in out and isinstance(out['dependencies'], list):
        out['dependencies'] = sorted(
            int(d) if isinstance(d, (int, str)) and str(d).isdigit() else d
            for d in out['dependencies']
        )
    # id may surface as int (get_task) or str (get_tasks); coerce so the
    # comparator doesn't fire on the asymmetry alone.
    if 'id' in out:
        try:
            out['id'] = int(out['id'])
        except (TypeError, ValueError):
            pass
    return out


def _normalize_tasks_tree(result: Any) -> Any:
    if not isinstance(result, dict):
        return result
    tasks = result.get('tasks', [])
    if not isinstance(tasks, list):
        return result
    return {'tasks': [_normalize_task(t) for t in tasks]}


def _normalize_set_status(result: Any) -> Any:
    if not isinstance(result, dict):
        return result
    tasks = result.get('tasks') or []
    flags = sorted(
        (str(t.get('taskId', '')), t.get('newStatus', '')) for t in tasks
    )
    return {'count': len(tasks), 'transitions': flags}


def _normalize_id_only(result: Any) -> Any:
    """Strip volatile fields; only meaningful field is "did it succeed"."""
    if not isinstance(result, dict):
        return result
    return {k: True for k in ('id', 'message') if k in result}


def _normalize_remove(result: Any) -> Any:
    if not isinstance(result, dict):
        return result
    return {
        'successful': result.get('successful', 0),
        'failed': result.get('failed', 0),
    }


__all__ = ['DualCompareBackend']
