"""Doubles for probing bare main for its failing test ids (task 5627).

``MainProbeHarness`` stands in for everything ``verify.main_baseline_failing_ids``
reaches when it probes main: a recording MAIN_PROBE worktree, the per-module
main-side ``run_verification`` of a narrowed probe, and the whole-tree
``run_scoped_verification``. The ``verify`` tests and the merge-queue
main-health tests both drive the real baseline cache through it, so the doubles
track the probe's call signature in one place.
"""
from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from orchestrator import verify
from orchestrator.config import ModuleConfig, OrchestratorConfig
from orchestrator.git_ops import WorktreeKind
from orchestrator.verify import VerifyResult

PROBE_MODULES = {
    prefix: ModuleConfig(
        prefix=prefix, test_command=f'pytest {prefix.lower()}/tests',
        lint_command=None, type_check_command=None,
    )
    for prefix in ('A', 'B', 'C')
}


class MainProbeHarness:
    """Main-side doubles for ``main_baseline_failing_ids`` at one SHA.

    ``main_ids[prefix]`` is what that module fails on main: a list, ``None``
    (no junit collected) or an exception to raise. ``whole_tree_ids`` left at
    ``None`` means the whole-tree probe must not run. Install the doubles with
    :meth:`patched`.
    """

    def __init__(self, config: OrchestratorConfig, main_sha: str, probe_dir: Path) -> None:
        self.config = config
        self.main_sha = main_sha
        self.main_ids: dict[str, list[str] | None | Exception] = {}
        self.whole_tree_ids: list[str] | None = None
        self.worktrees: list[tuple[WorktreeKind, str, bool]] = []
        self.module_runs: list[tuple[ModuleConfig, dict]] = []
        self.whole_tree_runs: list[dict] = []
        self.git_ops = MagicMock()
        self.git_ops.ephemeral_worktree = self._ephemeral_worktree
        self.git_ops.get_main_sha = AsyncMock(return_value=main_sha)
        self._probe_dir = probe_dir
        self._probe_dir.mkdir(parents=True, exist_ok=True)
        self._held: dict[str, tuple[asyncio.Event, asyncio.Event]] = {}

    @contextlib.contextmanager
    def patched(self) -> Iterator[MainProbeHarness]:
        with (
            patch.object(verify, 'run_verification', side_effect=self.run_verification),
            patch.object(
                verify, 'run_scoped_verification', side_effect=self.run_scoped_verification,
            ),
        ):
            yield self

    def hold(self, prefix: str) -> tuple[asyncio.Event, asyncio.Event]:
        """Park *prefix*'s main-side runs; returns (entered, release).

        *entered* is set once a run has parked, and setting *release* lets it answer.
        """
        entered, release = asyncio.Event(), asyncio.Event()
        self._held[prefix] = (entered, release)
        return entered, release

    @contextlib.asynccontextmanager
    async def _ephemeral_worktree(self, kind, sha, *, warm_seed=False):
        self.worktrees.append((kind, sha, warm_seed))
        yield self._probe_dir

    async def run_verification(self, worktree, config, module_config=None, **kwargs):
        assert module_config is not None
        self.module_runs.append((module_config, kwargs))
        if module_config.prefix in self._held:
            entered, release = self._held[module_config.prefix]
            entered.set()
            await release.wait()
        outcome = self.main_ids[module_config.prefix]
        if isinstance(outcome, Exception):
            raise outcome
        return VerifyResult(
            passed=not outcome, test_output='', lint_output='', type_output='',
            summary='main side', failing_test_ids=outcome,
            failing_test_ids_by_module=(
                None if outcome is None else {module_config.prefix: outcome}
            ),
        )

    async def run_scoped_verification(self, worktree, config, module_configs, task_files=None, **kwargs):
        self.whole_tree_runs.append({'task_files': task_files, **kwargs})
        if self.whole_tree_ids is None:
            raise AssertionError('the whole-tree probe must not run on this path')
        return VerifyResult(
            passed=False, test_output='', lint_output='', type_output='',
            summary='main side', failing_test_ids=self.whole_tree_ids,
        )

    async def probe(self, *red_module_prefixes: str) -> frozenset[str] | None:
        return await verify.main_baseline_failing_ids(
            self.config, list(PROBE_MODULES.values()), self.git_ops, self.main_sha,
            red_module_prefixes=frozenset(red_module_prefixes),
        )

    def baseline(self, *red_module_prefixes: str) -> frozenset[str] | None:
        return asyncio.run(self.probe(*red_module_prefixes))

    def probed_prefixes(self) -> list[str]:
        return [mc.prefix for mc, _ in self.module_runs]
