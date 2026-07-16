"""B+H boundary suite for the verify-scope-inversion PRD (task ξ).

plans/verify-scope-inversion-prd.md's task ξ is the PRD's **integration
gate**: ONE module driving the REAL merge-queue/workflow verify seams BOTH
WAYS (producer plan/gate state AND consumer observable) for every row of the
PRD's 10-row Boundary-test sketch. Every scenario below asserts BOTH sides of
its seam. This module passing green in CI is the leaf signal that the whole
inversion composes end-to-end — the merge-gate correctness guarantee.

Every capability asserted here is delivered by upstream siblings, all DONE on
main before this task started:

- κ (2588): plan-authoritative execution — ``run_scoped_verification``
  derives→executes→aggregates the ``VerifyPlan``; ``VerifyResult.plan`` is
  the EXECUTED plan (verify.py:4503, :2516).
- λ (2589): role-differentiated policy + the ``merge_verify_breadth`` knob —
  ``derive_verify_plan(role=...)`` forks; merge+full expands every
  registered ``ModuleConfig`` to its FULL_SUITE per-module; task-role adds
  an owning-module pytest floor; breadth='scoped' stays byte-identical
  legacy (config.py/defaults.yaml/verify_plan.py).
- μ (2590): merge-gate baseline attribution — ``seed_main_baseline`` /
  ``cached_main_baseline_failing_ids`` / ``diff_new_failures`` (verify.py);
  junit ``failing_test_ids`` on ``VerifyResult``; the merge worker cites
  NEW-only ids vs routing MAIN_HEALTH_RED (merge_queue.py
  ``_classify_main_health_red`` / ``_run_post_merge_verify``).
- ν (2591): infra-transient outcomes consume zero attempts at both the
  workflow ``_verify_debugfix_loop`` and merge accounting
  (``INFRA_TRANSIENT_CATEGORIES``, verify_categories.py).

Row → seam map (PRD's 10-row Boundary-test sketch):

=====  =========================================  ===============================================
Row    Name                                        Seam(s) driven
=====  =========================================  ===============================================
1      source-only sibling break rejected          producer: ``run_scoped_verification`` (merge+full
                                                     widens to the untouched sibling module); consumer:
                                                     ``_run_post_merge_verify`` (real merge gate, BLOCKED
                                                     citing the sibling's failing id)
2      task-role signal                            producer+observable: ``run_scoped_verification``
                                                     (role='task' — the floor never widens beyond the
                                                     owning module)
3      docs-only trivial, both roles                producer+consumer: ``run_scoped_verification`` at
                                                     both role='task'/'merge' (TRIVIAL short-circuit,
                                                     zero commands)
4      new-vs-preexisting baseline attribution      consumer: ``_run_post_merge_verify`` (mixed
                                                     baseline — blocked citing only the NEW id)
5      wholly pre-existing + cache hit              consumer: ``_run_post_merge_verify`` x2 (MAIN_HEALTH_RED
                                                     routing, no branch charge, one probe across two merges)
6      infra non-consumption, both consumers        task-verify: ``TaskWorkflow._verify_debugfix_loop``;
                                                     merge-verify: ``_run_post_merge_verify`` accounting
7      train amortization                           ``MergeWorker._do_merge`` (exactly one broad,
                                                     per-module merge-role verify of the train tip)
8      rollback path (breadth='scoped')              producer: ``run_scoped_verification`` (byte-identical
                                                     legacy merge-role plan)
9      fallback narrowing (never the global chain)   producer: ``run_scoped_verification`` (scoped fallback
                                                     only, never the whole-repo opaque chain)
10     plan authority, both roles                    producer: ``run_scoped_verification`` (executed ==
                                                     ``plan.runs``; ``result.plan`` is the EXECUTED plan)
=====  =========================================  ===============================================

Harness approach — drive REAL seams both ways, fake runners, no ssh (the
1737/2260/2309 shape): the PRODUCER/plan+execution side calls the REAL
``run_scoped_verification`` under an instrumented fake
``orchestrator.verify.run_verification`` (never a real subprocess) and
inspects the executed ``ModuleConfig``(s) / ``VerifyResult.plan`` /
``VerifyResult.summary``. The CONSUMER/merge-gate side drives the REAL
``_run_post_merge_verify`` chokepoint (merge_queue.py), opting OUT of the
autouse ``_mock_merge_queue_verification`` stub via
``@pytest.mark.exercise_merge_verify`` so ``orchestrator.merge_queue.
run_scoped_verification`` stays bound to the REAL function — which, in turn,
calls the SAME patched ``orchestrator.verify.run_verification`` fake. One
patched seam, two chokepoints, both real end-to-end.

conftest.py's autouse fixtures (``_clear_probe_cache`` — resets BOTH
``verify._PROBE_CACHE`` and ``verify._BASELINE_FAILING_IDS_CACHE`` before
and after every test — ``_mock_merge_queue_verification``,
``_neutralize_verify_admission``) already apply to this module; nothing to
import or opt into beyond the ``exercise_merge_verify`` marker on the rows
that need it. Neither this module nor its dependencies touch conftest.py or
any production file — scope is this single new test module.

Golden diff (row 1) — CONSTRUCTED, not mined: the confusion-codebook's
``verify-scope-asymmetry`` entry (docs/legibility/confusion-codebook.yaml:75,
16 ``sightings_2026_06``) records the cluster's *count* but its ``sightings``
list is empty — no per-incident diff is extractable from it. A quick
git-log mining pass (``git log --all -i --grep=MAIN_HEALTH_RED`` /
``--grep='sibling.*test'`` over verify.py/merge_queue.py, and repo-wide
``--grep='fix-forward'``) surfaced only the (also-DONE) task-2564/task-1690
main-health machinery commits themselves — no standalone minimal
source-breaks-sibling-test incident diff to mine, confirming the codebook's
empty ``sightings`` list. Per the task's recipe and capability manifest, row
1 therefore uses the CONSTRUCTED minimal two-module shape (a source edit
under module A that breaks a test in sibling module B) — explicitly labeled
constructed here, NOT dressed up with a fabricated "historical" attribution.
The cluster's 16 incidents validate the premise either way: a source-only
diff breaking a sibling module's suite escapes a scoped merge gate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal
from unittest.mock import AsyncMock, patch

import pytest
from test_merge_queue_main_health import _make_config, _make_git_ops, _make_req
from test_verify_scope_kappa import _executed_module_configs, _run_verification_spy
from test_verify_scope_lambda import _two_module_registry

from orchestrator import verify
from orchestrator.config import ModuleConfig, OrchestratorConfig
from orchestrator.merge_queue import (
    MAIN_HEALTH_RED_REASON_PREFIX,
    MergeOutcome,
    _run_post_merge_verify,
)
from orchestrator.verify import VerifyResult, run_scoped_verification, seed_main_baseline
from orchestrator.verify_categories import INFRA_TRANSIENT_CATEGORIES  # noqa: F401 — row 6 (later iteration)

# ---------------------------------------------------------------------------
# Golden diff (row 1) — CONSTRUCTED. See module docstring "Golden diff
# (row 1)" above for why this is constructed rather than mined.
# ---------------------------------------------------------------------------

# The touched file (the "diff" proper) — a plain source edit under module A.
MODA_SOURCE_PATH: str = 'moda/helpers.py'
MODA_SOURCE_CONTENT: str = 'def helper():\n    return 1\n'

# The untouched sibling test coupled to modA's contract — present on disk for
# the narrative fidelity of the two-module shape, but never itself part of
# any row's task_files: modB is discovered purely via the registry + a
# merge+full breadth expansion (see _two_module_registry, reused unmodified
# from test_verify_scope_lambda.py), never because it was directly touched.
MODB_SIBLING_TEST_PATH: str = 'modb/tests/test_sibling.py'
MODB_SIBLING_TEST_CONTENT: str = (
    'from moda.helpers import helper\n\n\n'
    'def test_sibling_contract():\n'
    '    assert helper() == 1\n'
)

# The pytest node id the row-1/4/5 instrumented fakes report as failing for
# modB — shaped like a real junit-derived id (VerifyResult.failing_test_ids).
MODB_FAILING_TEST_ID: str = 'modb/tests/test_sibling.py::test_sibling_contract'


def _row1_golden_diff(
    tmp_path: Path, *, breadth: Literal['scoped', 'full'] = 'full',
) -> tuple[ModuleConfig, ModuleConfig, OrchestratorConfig]:
    """Write the CONSTRUCTED row-1 golden diff to *tmp_path* and return the
    2-module registry (modA/modB) it belongs to.

    Modeled on (and delegates to) ``test_verify_scope_lambda._two_module_registry``
    for the ModuleConfig/OrchestratorConfig registry — this function's own
    addition is writing the golden-diff FILES: modA's touched source file
    (``MODA_SOURCE_PATH``) and modB's untouched sibling test
    (``MODB_SIBLING_TEST_PATH``), matching row 1's "a source edit under
    module A that breaks a test in sibling module B" shape. Reused by every
    row built on this same golden diff (1, 2, 8).
    """
    mod_a, mod_b, config = _two_module_registry(tmp_path, breadth=breadth)

    moda_full = tmp_path / MODA_SOURCE_PATH
    moda_full.parent.mkdir(parents=True, exist_ok=True)
    moda_full.write_text(MODA_SOURCE_CONTENT)

    modb_full = tmp_path / MODB_SIBLING_TEST_PATH
    modb_full.parent.mkdir(parents=True, exist_ok=True)
    modb_full.write_text(MODB_SIBLING_TEST_CONTENT)

    return mod_a, mod_b, config


def _fake_run_verification_by_module(
    results: dict[str, tuple[bool, list[str]]],
    *, category: str = 'test_failure',
) -> AsyncMock:
    """Instrumented fake for ``orchestrator.verify.run_verification``.

    Returns a DIFFERENT canned per-module ``VerifyResult`` keyed by
    ``ModuleConfig.prefix`` — never spawns a real subprocess. *results* maps
    ``prefix -> (passed, failing_test_ids)``; a prefix absent from *results*
    (or the ``module_config=None`` global-fallback call shape) defaults to a
    clean pass with ``failing_test_ids=[]`` — never ``None``, so merge+full's
    junit-collection signal (VerifyResult.failing_test_ids's B3 None-vs-``[]``
    contract) stays present for every executed module, exactly as a real
    merge+full pytest run would collect an empty junit report for a clean
    module.

    Patch via ``patch.object(verify, 'run_verification', new=...)`` — the
    SAME seam :func:`test_verify_scope_kappa._run_verification_spy` patches
    — and recover the ordered executed ``ModuleConfig``(s) afterward via
    :func:`_executed_module_configs`. Shared by rows 1, 4, and 5 (only the
    *results* mapping and baseline seeding differ per row).
    """
    async def _fake(worktree, config, module_config=None, **kwargs):
        prefix = module_config.prefix if module_config is not None else None
        passed, failing_ids = results.get(prefix, (True, []))
        if passed:
            return VerifyResult(
                passed=True, test_output='', lint_output='', type_output='',
                summary='All checks passed', failing_test_ids=list(failing_ids),
            )
        return VerifyResult(
            passed=False, test_output='FAILED some test', lint_output='', type_output='',
            summary='Failures: tests failed', category=category,
            cause_hint='AssertionError: sibling contract broken',
            failing_test_ids=list(failing_ids),
        )
    return AsyncMock(side_effect=_fake)


async def _drive_merge_gate(
    tmp_path: Path,
    *,
    task_id: str,
    module_configs_registry: dict[str, ModuleConfig],
    touched_module_configs: list[ModuleConfig],
    task_files: list[str],
    task_files_content: dict[str, str],
    main_sha: str,
    run_verification_fake: AsyncMock,
) -> MergeOutcome | None:
    """Drive the REAL merge-gate consumer chokepoint, ``_run_post_merge_verify``.

    Builds a merge+full ``OrchestratorConfig`` (with *module_configs_registry*
    installed as its module registry — mirrors ``_two_module_registry``'s
    ``config._module_configs`` assignment), a ``MergeRequest`` touching only
    *touched_module_configs* / *task_files* (mirrors a real branch diff —
    untouched-but-registered modules are discovered via the registry, not
    listed here), and a ``GitOps`` double whose ``get_main_sha`` resolves to
    *main_sha*. Patches ``orchestrator.verify.run_verification`` with
    *run_verification_fake* for the duration of the drive so the REAL
    ``run_scoped_verification`` (unmocked — the caller must mark its test
    ``@pytest.mark.exercise_merge_verify`` to opt out of the autouse
    ``_mock_merge_queue_verification`` stub) executes against canned
    per-module results, never a real subprocess.

    *task_files_content* (``{relative path: file content}``) is written into
    the MERGE worktree (``merge_wt`` — the actual worktree
    ``_run_post_merge_verify`` scopes against), not the outer *tmp_path*
    scratch dir. Omitting this (an empty dict, or a path *task_files* names
    but *task_files_content* doesn't cover) leaves ``merge_wt`` without the
    touched file on disk, which degrades plan derivation to the TRIVIAL
    "no source files" short-circuit — zero ``run_verification`` calls and a
    silent fall-through to the real (unmocked) unscoped type-check gate.

    *task_id* must be distinct per call within a test that drives the gate
    more than once (e.g. row 5's two-merge cache-hit scenario), since it
    seeds the task/merge worktree directory names. Shared by rows 1, 4, and 5.
    """
    config = _make_config(tmp_path, merge_verify_breadth='full')
    config._module_configs = dict(module_configs_registry)
    git_ops = _make_git_ops(tmp_path)
    git_ops.get_main_sha = AsyncMock(return_value=main_sha)

    task_wt = tmp_path / f'task-wt-{task_id}'
    task_wt.mkdir(parents=True, exist_ok=True)
    merge_wt = tmp_path / f'merge-wt-{task_id}'
    merge_wt.mkdir(parents=True, exist_ok=True)
    for rel_path, content in task_files_content.items():
        full = merge_wt / rel_path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content)

    req = _make_req(task_id, task_wt, config)
    req.task_files = list(task_files)
    req.module_configs = list(touched_module_configs)

    with patch.object(verify, 'run_verification', new=run_verification_fake):
        return await _run_post_merge_verify(
            git_ops, req, merge_wt,
            timeouts={}, enospc_retries={}, max_timeouts=3, max_enospc=1,
        )


async def _drive_producer(
    tmp_path: Path,
    config: OrchestratorConfig,
    module_configs: list[ModuleConfig],
    *,
    task_files: list[str],
    role: Literal['task', 'merge'],
    is_merge_verify: bool = False,
    run_verification_fake: AsyncMock | None = None,
) -> tuple[VerifyResult, dict[str, ModuleConfig], AsyncMock]:
    """Drive the REAL producer/plan+execution seam, ``run_scoped_verification``,
    under an instrumented fake ``orchestrator.verify.run_verification`` —
    never a real subprocess.

    *run_verification_fake* defaults to the always-passing
    :func:`_run_verification_spy` (test_verify_scope_kappa.py) when omitted;
    pass an instrumented per-module fake (e.g.
    :func:`_fake_run_verification_by_module`) for rows that need a specific
    module to fail.

    Returns ``(result, executed_by_prefix, fake)``: *executed_by_prefix* is
    the ordered executed ``ModuleConfig``(s) keyed by
    ``ModuleConfig.prefix`` (via :func:`_executed_module_configs`); *fake* is
    the patched mock itself, for callers that need e.g. ``await_count``.
    Shared by every producer-only row (2, 3, 8, 9, 10) — the row-1 producer
    side predates this helper and stays inline (it also drives the row-1
    consumer side against the SAME fake below, unlike these later rows).
    """
    fake = run_verification_fake if run_verification_fake is not None else _run_verification_spy()
    with patch.object(verify, 'run_verification', new=fake):
        result = await run_scoped_verification(
            tmp_path, config, module_configs, task_files=task_files, role=role,
            is_merge_verify=is_merge_verify,
        )
    executed = {mc.prefix: mc for mc in _executed_module_configs(fake)}
    return result, executed, fake


# ---------------------------------------------------------------------------
# Row 1: "the hole, closed" — source-only sibling break rejected at the
# merge gate. Builds the shared row-1 harness (instrumented fake
# run_verification + the _run_post_merge_verify consumer driver), reused
# unmodified by rows 4 and 5.
# ---------------------------------------------------------------------------

# A dedicated, row-scoped main SHA constant (never reused verbatim from
# test_merge_queue_main_health.MAIN_SHA) so this module's _BASELINE_FAILING_
# IDS_CACHE / _PROBE_CACHE keys can never collide with that module's under
# parallel (-n auto --dist loadgroup) execution across separate files.
ROW1_MAIN_SHA: str = 'r1main0000000000000000000000000000000000'


class TestRow1SourceOnlySiblingBreakRejectedAtMergeGate:
    """Row 1 (PRD boundary-test sketch): a source-only diff under modA that
    breaks modB's sibling suite must be REJECTED at the merge gate under
    merge+full breadth — the exact hole verify-scope-asymmetry (confusion-
    codebook docs/legibility/confusion-codebook.yaml:75, 16 sightings)
    documents: a scoped merge gate would never even look at modB, landing a
    red main. See the module docstring's "Golden diff (row 1)" section for
    why this diff is CONSTRUCTED rather than mined.
    """

    @pytest.mark.asyncio
    @pytest.mark.exercise_merge_verify
    async def test_row1_source_only_sibling_break_rejected_at_merge_gate(
        self, tmp_path: Path,
    ) -> None:
        mod_a, mod_b, config = _row1_golden_diff(tmp_path, breadth='full')

        # -- PRODUCER side: merge+full widens execution to the untouched,
        # registry-only sibling modB, which is where the golden diff's break
        # actually lives (mirrors test_verify_scope_lambda's merge+full
        # expansion golden, but this row goes on to also drive the consumer
        # side below against the SAME fake failure).
        producer_fake = _fake_run_verification_by_module(
            {mod_b.prefix: (False, [MODB_FAILING_TEST_ID])},
        )
        with patch.object(verify, 'run_verification', new=producer_fake):
            producer_result = await run_scoped_verification(
                tmp_path, config, [mod_a], task_files=[MODA_SOURCE_PATH],
                role='merge', is_merge_verify=True,
            )
        executed = {mc.prefix: mc for mc in _executed_module_configs(producer_fake)}
        assert set(executed) == {'moda', 'modb'}, (
            f'expected merge+full to widen execution to the untouched sibling '
            f'modB; got {set(executed)!r}'
        )
        assert executed['modb'].test_command == mod_b.test_command, (
            f"expected modB to run its verbatim FULL_SUITE test command; "
            f'got {executed["modb"].test_command!r}'
        )
        assert not producer_result.passed, (
            'the aggregate producer-side result must reflect modB failing'
        )

        # -- CONSUMER side: drive the REAL merge gate. Main is clean (empty
        # seeded baseline), so modB's failure is a NEW id, never
        # MAIN_HEALTH_RED — the branch must be blocked citing it by name.
        seed_main_baseline(ROW1_MAIN_SHA, frozenset())
        outcome = await _drive_merge_gate(
            tmp_path,
            task_id='row1-9101',
            module_configs_registry={'moda': mod_a, 'modb': mod_b},
            touched_module_configs=[mod_a],
            task_files=[MODA_SOURCE_PATH],
            task_files_content={MODA_SOURCE_PATH: MODA_SOURCE_CONTENT},
            main_sha=ROW1_MAIN_SHA,
            run_verification_fake=_fake_run_verification_by_module(
                {mod_b.prefix: (False, [MODB_FAILING_TEST_ID])},
            ),
        )

        assert outcome is not None, 'expected a blocked MergeOutcome, got the verify-passed sentinel'
        assert outcome.status == 'blocked', f'expected blocked; got {outcome.status!r}'
        assert not outcome.reason.startswith(MAIN_HEALTH_RED_REASON_PREFIX), (
            f'a NEW sibling-test break (against an empty main baseline) must '
            f'never route MAIN_HEALTH_RED; got {outcome.reason!r}'
        )
        assert MODB_FAILING_TEST_ID in outcome.reason, (
            f'expected the failing modB test id to be cited in the block '
            f'reason; got {outcome.reason!r}'
        )


# ---------------------------------------------------------------------------
# Row 2: "task-role signal" — the task-role pytest floor (λ, R3) never widens
# beyond the owning module. Producer-only (no consumer/merge-gate side).
# ---------------------------------------------------------------------------


class TestRow2TaskRoleSignal:
    """Row 2 (PRD boundary-test sketch): the SAME row-1 golden diff, but at
    role='task' — ONLY the owning module modA executes (the task-role pytest
    floor, R3); the registered sibling modB must NEVER run (R1-task: the
    floor never widens beyond owning modules — only the knob-gated
    merge+full gate row 1 pins does that widening).
    """

    @pytest.mark.asyncio
    async def test_row2_same_diff_task_role_owning_module_only(self, tmp_path: Path) -> None:
        mod_a, _mod_b, config = _row1_golden_diff(tmp_path, breadth='full')

        result, executed, _fake = await _drive_producer(
            tmp_path, config, [mod_a], task_files=[MODA_SOURCE_PATH], role='task',
        )

        assert result.passed
        assert set(executed) == {'moda'}, (
            f'expected ONLY the owning module to execute at role=task (the '
            f'floor never widens beyond owning modules); got {set(executed)!r}'
        )
        assert executed['moda'].test_command == mod_a.test_command, (
            f"expected the task-role floor to full-suite the owning module's "
            f'pytest: {executed["moda"].test_command!r}'
        )
        assert 'modb' not in executed, (
            'the registered sibling modB must never execute at role=task '
            '(no cross-module widening from the task-role floor)'
        )
