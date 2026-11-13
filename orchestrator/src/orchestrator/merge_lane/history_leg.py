"""The verify runner for a project's declared merge-gate history leg (task 7248).

On a same-tree rebase a project that declares
``OrchestratorConfig.merge_gate_history_test_command`` re-runs only that
command, through ``worker.py::_run_post_merge_verify``, which builds a
:class:`HistoryLegRunner` in place of the full gate's ``LocalRunner``. So a
red or unrunnable leg gets every failure path a red re-verify gets; this
module only runs the command and reports a result that cannot be mistaken for
the whole gate's.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import ClassVar

from orchestrator.config import ModuleConfig, OrchestratorConfig
from orchestrator.verify import VerifyResult, run_verification
from orchestrator.verify_runner import MergeVerifySpec

HISTORY_LEG_PREFIX = 'merge-gate-history'


class HistoryLegRunner:
    """A local ``VerifyRunner`` that runs only the declared history leg command.

    Its result carries no per-test ids: they would describe only the leg's
    subset of the gate, yet the pass path seeds main's failing-id baseline
    from them and the main-health probe diffs them as the whole gate's.
    """

    name: str = 'history-leg'
    is_local: ClassVar[bool] = True

    def __init__(
        self,
        merge_wt: Path,
        config: OrchestratorConfig,
        command: str,
        *,
        task_id: str | None,
        archive_root: Path | None,
    ) -> None:
        self._merge_wt = merge_wt
        self._config = config
        self._command = command
        self._task_id = task_id
        self._archive_root = archive_root

    async def health(self) -> bool:
        return True

    async def run_merge_verify(self, merge_sha: str, spec: MergeVerifySpec) -> VerifyResult:
        result = await run_verification(
            self._merge_wt, self._config,
            ModuleConfig(prefix=HISTORY_LEG_PREFIX, test_command=self._command),
            max_retries=0, is_merge_verify=True, role='merge',
            task_id=self._task_id, archive_root=self._archive_root,
        )
        return dataclasses.replace(
            result,
            summary=f'history leg: {result.summary}',
            failing_test_ids=None,
            skipped_test_ids=None,
            failing_test_ids_by_module=None,
        )
