"""The scoped-verify double the merge-lane tests inject, and what it stubs.

``ScriptedVerifier`` is ``_merge_lane_fakes.FakeVerifier`` plus the two
readings per-task-id scripting cannot express: WHICH WORKTREE each scoped
verify ran in (``worktrees``), and a per-CALL impl (``impl``) for a test that
scripts the first call differently from the rest -- a gate that blocks one
verify and lets the next through, a span recorder, a failure followed by a
pass.  ``FakeVerifier``'s docstring sanctions a ``run_scoped`` override for
exactly this and makes calling ``_note_entry`` the override's obligation;
discharging it here once is the point, so no test re-derives it.

WHY THIS IS NOT IN ``_merge_lane_fakes.py``.  That is its home -- recording
``worktrees`` belongs beside ``FakeVerifier.verified``, and then this module
disappears.  It is separate only because task 5026 (PRD
``plans/merge-lane-quality-prd.md`` task gamma3) does not hold the lock on
``_merge_lane_fakes.py``, which nine sibling groups rebase through the merge
lane concurrently.  Whoever next holds that lock should fold this class into
``FakeVerifier`` and delete this file.

WHAT INJECTING A ``VerifyPort`` DOES AND DOES NOT STUB.  Measured on this
tree by counting ``verifier.<method>`` call sites under
``orchestrator/src/orchestrator/``, because the answer is narrower than the
port's method list suggests and a reader should not have to re-derive it:

* ``check_post_merge_pyright`` and ``check_post_merge_equivalence`` have ZERO
  lane call sites.  ``merge_gates.py::_run_pyright_gate`` and
  ``merge_gates.py::_run_equivalence_gate`` reach
  ``merge_queue.py::_check_post_merge_pyright`` and
  ``merge_queue.py::_check_post_merge_equivalence`` through a deferred
  ``import``, which bypasses the injected port -- so both gates still run
  PRODUCTION code under an injected verifier, exactly as they did under the
  ``patch('orchestrator.merge_queue.run_scoped_verification', ...)`` these
  tests replaced.
* ``run_unscoped_typechecks`` IS injected, but
  ``merge_queue.py::_run_unscoped_typechecks`` returns a clean
  ``PostMergePyrightResult()`` before doing any work when no ``module_configs``
  entry carries a ``type_check_command``.  Every request these tests build
  passes ``module_configs=[]``, so production returned precisely what the
  fake returns.
* ``ensure_disk_space`` IS injected, and is the one real narrowing.
  ``merge_queue.py::_ensure_verify_disk_space`` returns "proceed" whenever
  free bytes are at or above the threshold -- the branch any machine able to
  run this suite takes.  Its other branch prunes stale ``_merge-*``
  worktrees, which no test here reached deterministically and which is
  covered as a unit by ``test_merge_queue.py::TestEnsureVerifyDiskSpace``.
  Injecting the port also makes these tests independent of the host's free
  disk, which that branch made them.

Imported by bare module name (``from _merge_lane_verifier_doubles import
...``), like ``_merge_lane_fakes`` -- ``orchestrator/tests/`` has no
``__init__.py``.

This module imports no cluster module, so ``scripts/merge_lane_metrics.py``
does not measure it and it holds no ratchet key -- which is exactly why lane
internals must not be reached from here.  Moving a private read into an
unmeasured shared helper does not reduce it, it hides it; keep every such read
in the test file whose baseline key accounts for it.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from _merge_lane_fakes import FakeVerifier

from orchestrator.verify import VerifyResult

#: A per-CALL scoped verify, called with ``run_scoped``'s own arguments.
PerCallScopedVerify = Callable[..., Awaitable[Any]]


class ScriptedVerifier(FakeVerifier):
    """``FakeVerifier`` that records the verify worktree and can script per CALL.

    ``worktrees[i]`` is the worktree the i-th scoped verify ran in, in step
    with ``verified[i]``'s task id: post-merge verify reaches the port through
    ``verify_runner.py::LocalRunner``, which passes the merge worktree it was
    built on as ``run_scoped``'s first positional argument, so ``worktrees``
    is the injected reading of the lane's worktree-routing decision.

    With no *impl* this is ``FakeVerifier``'s per-task-id scripting unchanged.
    With one, every scoped verify is delegated to *impl* instead, which is how
    a test scripts a sequence the ``scripts=``/``default=`` mapping cannot
    express -- one that varies by call rather than by task id.
    """

    def __init__(self, impl: PerCallScopedVerify | None = None) -> None:
        super().__init__()
        self.worktrees: list[Path] = []
        self._impl = impl

    async def run_scoped(
        self,
        worktree: Path,
        config: Any,
        module_configs: list[Any],
        task_files: list[str] | None = None,
        **options: Any,
    ) -> VerifyResult:
        self.worktrees.append(worktree)
        if self._impl is None:
            return await super().run_scoped(
                worktree, config, module_configs, task_files, **options,
            )
        self._note_entry(options.get('task_id'))
        return await self._impl(
            worktree, config, module_configs, task_files, **options,
        )
