"""Fakes for the merge lane's ports (PRD ``plans/merge-lane-quality-prd.md`` task β).

Injected in place of the production adapters through
``MergeLane(..., verifier=FakeVerifier(...), clock=FakeClock(...))``.
``FakeVerifier`` scripts the verify outcome per task id; ``FakeClock`` is a
hand-advanced clock whose ``sleep`` advances it instead of waiting.

Imported by bare module name (``from _merge_lane_fakes import ...``), like
``_orch_helpers`` -- ``orchestrator/tests/`` has no ``__init__.py``.
"""
from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Collection, Coroutine, Mapping
from pathlib import Path
from typing import Any

from orchestrator.merge_gates import PostMergePyrightResult
from orchestrator.merge_lane.types import DiskGuardOutcome
from orchestrator.verify import VerifyResult


@dataclasses.dataclass(frozen=True)
class VerifyScript:
    """What ``FakeVerifier.run_scoped`` does for one task.

    Exactly one of the shapes below applies: return ``result``, raise
    ``error``, or wait for ``release`` first and then return ``result``.
    """

    result: VerifyResult
    error: BaseException | None = None
    release: asyncio.Event | None = None


def passes(summary: str = 'fake verify passed') -> VerifyScript:
    return VerifyScript(result=VerifyResult(
        passed=True, test_output='', lint_output='', type_output='', summary=summary,
    ))


def fails(*, category: str, summary: str) -> VerifyScript:
    return VerifyScript(result=VerifyResult(
        passed=False, test_output='', lint_output='', type_output='',
        summary=summary, category=category,
    ))


def raises(error: BaseException) -> VerifyScript:
    return VerifyScript(result=passes().result, error=error)


def hangs_until(release: asyncio.Event) -> VerifyScript:
    return VerifyScript(result=passes().result, release=release)


class FakeVerifier:
    """``VerifyPort`` scripted per task id.

    ``run_scoped`` follows ``scripts[task_id]``, or ``default`` for a task
    without a script, and records every task id it was asked about in
    ``verified``. The gates a merge passes through after a green scoped
    verify all report clean, the disk guard always proceeds, and dry-run
    investigations are recorded in ``investigations`` rather than run.
    """

    def __init__(
        self,
        default: VerifyScript | None = None,
        scripts: Mapping[str | None, VerifyScript] | None = None,
    ) -> None:
        self.default = passes() if default is None else default
        self.scripts: dict[str | None, VerifyScript] = dict(scripts or {})
        self.verified: list[str | None] = []
        self.investigations: list[dict[str, Any]] = []

    async def run_scoped(
        self,
        worktree: Path,
        config: Any,
        module_configs: list[Any],
        task_files: list[str] | None = None,
        **options: Any,
    ) -> VerifyResult:
        task_id = options.get('task_id')
        self.verified.append(task_id)
        script = self.scripts.get(task_id, self.default)
        if script.release is not None:
            await script.release.wait()
        if script.error is not None:
            raise script.error
        return script.result

    async def run_unscoped_typechecks(
        self, worktree: Path, config: Any, module_configs: list[Any], **options: Any,
    ) -> PostMergePyrightResult:
        return PostMergePyrightResult()

    async def check_post_merge_pyright(
        self, advanced_sha: str, git_ops: Any, config: Any, module_configs: list[Any],
        **options: Any,
    ) -> PostMergePyrightResult:
        return PostMergePyrightResult()

    async def check_post_merge_equivalence(
        self, task_worktree: Path, advanced_sha: str, git_ops: Any, main_sha: str,
        **options: Any,
    ) -> list[str]:
        return []

    async def ensure_disk_space(
        self,
        git_ops: Any,
        merge_wt: Path,
        min_free_bytes: int,
        task_id: str,
        keep_worktrees: Collection[Path] | None = None,
    ) -> DiskGuardOutcome:
        return DiskGuardOutcome(reason=None)

    async def cold_shadow(
        self, git_ops: Any, req: Any, merge_commit: str, event_store: Any,
    ) -> dict[str, str]:
        return {}

    def dry_run_unblock(self, **investigation: Any) -> Coroutine[Any, Any, None]:
        self.investigations.append(investigation)
        return _nothing()


async def _nothing() -> None:
    return None


class FakeClock:
    """``ClockPort`` that moves only when told to.

    ``now`` and ``monotonic`` read ``time``; ``sleep`` advances it by the
    requested seconds and yields once so other tasks run; every requested
    sleep is kept in ``sleeps``. ``newest_content_mtime`` reports
    ``content_mtime``.
    """

    def __init__(self, *, time: float = 1_000_000.0, content_mtime: float | None = None) -> None:
        self.time = time
        self.content_mtime = content_mtime
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.time

    def monotonic(self) -> float:
        return self.time

    def newest_content_mtime(self, root: Path) -> float | None:
        return self.content_mtime

    async def sleep(self, secs: float) -> None:
        self.sleeps.append(secs)
        self.time += secs
        await asyncio.sleep(0)
