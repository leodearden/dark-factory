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
  the EXECUTED plan (verify.py).
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

import asyncio
from pathlib import Path
from typing import Literal, cast
from unittest.mock import AsyncMock, patch

import pytest
from _serial_merge_worker import MergeWorker
from test_merge_queue_main_health import _make_config, _make_git_ops, _make_req
from test_train_integration import _SpyEventStore, build_group_merge_request, make_stacked_member
from test_verify import _real_worktree_reader
from test_verify_scope_kappa import (
    _FLEET_LINT_COMMAND,
    _FLEET_TEST_COMMAND,
    _FLEET_TYPE_COMMAND,
    UNREGISTERED_PATH_DIFF,
    _executed_module_configs,
    _run_verification_spy,
)
from test_verify_scope_lambda import _two_module_registry
from test_workflow_verify_infra_resume import _infra_category_result
from test_workflow_verify_infra_resume import _make as _make_workflow

from orchestrator import verify, verify_plan
from orchestrator.config import GitConfig, ModuleConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.merge_queue import (
    MAIN_HEALTH_RED_REASON_PREFIX,
    TRANSIENT_INFRA_REASON_PREFIX,
    MergeOutcome,
    _run_post_merge_verify,
)
from orchestrator.verify import (
    VerifyResult,
    _mc_attr_for_run_or_none,
    run_scoped_verification,
    seed_main_baseline,
    verify_failure_is_preexisting_on_main,
)
from orchestrator.verify_categories import INFRA_TRANSIENT_CATEGORIES
from orchestrator.verify_cmd import render
from orchestrator.workflow import WorkflowOutcome

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

# Row 3's docs-only diff — no `.py`/`.rs` extension, so it can never satisfy
# verify_plan._has_source_files; the ScopeKind.TRIVIAL short-circuit fires
# ahead of both the module-config and fallback branches, independent of role
# or merge_verify_breadth (R2).
DOCS_ONLY_PATH: str = 'skills/foo.md'
DOCS_ONLY_CONTENT: str = '# foo\n'


def _write_docs_only_diff(root: Path) -> list[str]:
    """Write row 3's docs-only diff (a single ``.md`` file) under *root* and
    return its ``task_files`` list.

    Used for the PRODUCER side (*root* is *tmp_path*, the worktree
    ``run_scoped_verification`` reads directly). The CONSUMER-side parity
    check does NOT call this — ``_drive_merge_gate`` writes
    ``task_files_content`` into the MERGE worktree itself (see its
    docstring's footgun paragraph), so the consumer side passes
    ``DOCS_ONLY_PATH``/``DOCS_ONLY_CONTENT`` directly instead.
    """
    full = root / DOCS_ONLY_PATH
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(DOCS_ONLY_CONTENT)
    return [DOCS_ONLY_PATH]


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
        if module_config is not None:
            passed, failing_ids = results.get(module_config.prefix, (True, []))
        else:
            passed, failing_ids = True, []
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
    git_ops: GitOps | None = None,
    timeouts: dict[str, int] | None = None,
    enospc_retries: dict[str, int] | None = None,
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

    *git_ops* lets a caller share ONE ``GitOps`` double across multiple
    drives against the same main SHA — default ``None`` builds a fresh one
    via ``_make_git_ops`` (byte-identical to every pre-existing call site).
    Row 5 shares one across its two merges so it can assert
    ``git_ops.ephemeral_worktree`` was never invoked: the direct, accurate
    "no real main-probe worktree was created" signal for a μ B2 baseline
    cache hit (see :func:`orchestrator.verify.main_baseline_failing_ids`) —
    the cache lives one layer inside ``verify_failure_is_preexisting_on_main``,
    so that entry point itself is still called once per merge; only the
    expensive worktree-probe fallback is what the cache actually elides.

    *timeouts* / *enospc_retries* let a caller inspect the merge-side
    attempt-accounting dicts ``_run_post_merge_verify`` mutates in place
    (the timeout loop-breaker and the shared ENOSPC/classified-infra retry
    budget, respectively) — default ``None`` builds fresh empty dicts
    per-call (byte-identical to every pre-existing call site, which never
    needed to inspect them afterward). Row 6 passes its own dicts to assert
    a persistent infra-transient outcome bumps NEITHER (no merge verify
    attempt consumed).
    """
    config = _make_config(tmp_path, merge_verify_breadth='full')
    config._module_configs = dict(module_configs_registry)
    if git_ops is None:
        git_ops = _make_git_ops(tmp_path)
    git_ops.get_main_sha = AsyncMock(return_value=main_sha)
    if timeouts is None:
        timeouts = {}
    if enospc_retries is None:
        enospc_retries = {}

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
            timeouts=timeouts, enospc_retries=enospc_retries,
            max_timeouts=3, max_enospc=1,
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


# ---------------------------------------------------------------------------
# Row 3: "docs-only trivial, both roles" — R2 parity survives the inversion.
# A `.md`-only diff stays TRIVIAL (ScopeKind.TRIVIAL, verify_plan.py) at the
# PRODUCER seam (``run_scoped_verification`` with is_merge_verify=False) for
# BOTH role='task'/'merge', breadth-independent — zero commands.
#
# INV-1 (task 2883) UPDATE: the CONSUMER seam is the *adoptable* merge verdict
# (``_run_post_merge_verify`` → is_merge_verify=True). INV-1 abolishes the
# no-evidence trivial pass there: a docs-only ('no_source_files') diff at the
# real merge gate no longer trivially passes — it escalates to the project's
# full gate when a full-gate command exists (see
# test_merge_verdict_integrity_inv1.py's
# test_zero_module_global_command_runs_full_gate_and_emits_event). So the
# consumer side of this row now runs exactly the escalated global full gate,
# not zero commands; the producer/task-role trivial short-circuit is unchanged.
# ---------------------------------------------------------------------------

# A dedicated row-3 main SHA (see ROW1_MAIN_SHA's docstring note on why every
# row uses its own constant rather than sharing test_merge_queue_main_health.MAIN_SHA).
ROW3_MAIN_SHA: str = 'r3main0000000000000000000000000000000000'


class TestRow3DocsOnlyTrivialBothRoles:
    """Row 3 (PRD boundary-test sketch): a docs-only diff (no ``.py``/``.rs``
    file) short-circuits to ``ScopeKind.TRIVIAL`` ahead of BOTH the
    module-config and fallback branches (verify_plan.derive_verify_plan) —
    zero ``run_verification`` calls, independent of role or
    ``merge_verify_breadth`` (R2). This row pins that the inversion (κ/λ)
    never regresses this pre-existing invariant at the PRODUCER
    (``run_scoped_verification``, is_merge_verify=False) seam.

    INV-1 (task 2883) amends the CONSUMER seam: the adoptable merge verdict
    (``_run_post_merge_verify`` → is_merge_verify=True) no longer trivially
    passes a no-evidence resolution. A docs-only diff at the real merge gate
    now escalates to the project's full gate (a ``trivial_pass_escalated``
    event with reason='no_source_files', resolution='full_gate') and runs the
    escalated global command — so the consumer assertion below pins exactly
    ONE escalated full-gate run, not zero. The producer/task-role trivial
    short-circuit is untouched (INV-1 gates on is_merge_verify).
    """

    @pytest.mark.asyncio
    @pytest.mark.exercise_merge_verify
    @pytest.mark.parametrize('role', ['task', 'merge'])
    async def test_row3_docs_only_trivial_both_roles(
        self, tmp_path: Path, role: Literal['task', 'merge'],
    ) -> None:
        mod_a, _mod_b, config = _two_module_registry(tmp_path, breadth='full')
        docs_files = _write_docs_only_diff(tmp_path)

        # -- PRODUCER side: TRIVIAL short-circuit at THIS parametrized role.
        result, executed, fake = await _drive_producer(
            tmp_path, config, [mod_a], task_files=docs_files, role=role,
        )
        assert result.passed
        assert 'No source files' in result.summary, (
            f'expected the TRIVIAL short-circuit summary; got {result.summary!r}'
        )
        assert fake.await_count == 0, (
            f'docs-only diff must execute zero commands (role={role!r}); '
            f'got {fake.await_count} call(s)'
        )
        assert executed == {}

        # -- CONSUMER side: INV-1 (task 2883). The SAME docs-only diff at the
        # real *adoptable* merge gate (always is_merge_verify=True under the
        # hood, regardless of this test's parametrized role) no longer
        # trivially passes: with a zero-module-config project whose global
        # command is non-empty (_make_config), the no-evidence resolution
        # escalates to the project's full gate and runs it exactly once. The
        # verdict still PASSES (the escalated global gate's fake returns pass),
        # so the outcome is still the verify-passed sentinel (None) — but via
        # real evidence, not a vacuous short-circuit.
        consumer_fake = _run_verification_spy()
        outcome = await _drive_merge_gate(
            tmp_path,
            task_id=f'row3-4041-{role}',
            module_configs_registry={},
            touched_module_configs=[],
            task_files=docs_files,
            task_files_content={DOCS_ONLY_PATH: DOCS_ONLY_CONTENT},
            main_sha=ROW3_MAIN_SHA,
            run_verification_fake=consumer_fake,
        )
        assert outcome is None, f'expected the verify-passed sentinel (None); got {outcome!r}'
        assert consumer_fake.await_count == 1, (
            f'INV-1: a docs-only diff at the adoptable merge gate must escalate '
            f'to the global full gate and run it exactly once (never a '
            f'no-evidence trivial pass); got {consumer_fake.await_count} call(s)'
        )


# ---------------------------------------------------------------------------
# Row 4: "new-vs-preexisting" baseline attribution (μ, B1). Consumer-only.
# ---------------------------------------------------------------------------

ROW4_MAIN_SHA: str = 'r4main0000000000000000000000000000000000'
ROW4_PREEXISTING_TEST_ID: str = 'moda/tests/test_x.py::test_x'
ROW4_NEW_TEST_ID: str = 'modb/tests/test_y.py::test_y'


def _row4_mixed_baseline_diff(
    tmp_path: Path,
) -> tuple[ModuleConfig, ModuleConfig, dict[str, str]]:
    """Build row 4's mixed-baseline diff: a merge directly touching TWO
    registered modules (moda, modb) — moda's failure will be seeded into
    the main baseline as pre-existing (:data:`ROW4_PREEXISTING_TEST_ID`);
    modb's is new (:data:`ROW4_NEW_TEST_ID`).

    Returns ``(mod_a, mod_b, task_files_content)`` — *task_files_content*
    is written into the MERGE worktree by :func:`_drive_merge_gate` (see its
    docstring's footgun paragraph), keyed by relative path.
    """
    mod_a, mod_b, _config = _two_module_registry(tmp_path, breadth='full')
    task_files_content = {
        'moda/thing.py': 'x = 1\n',
        'modb/thing.py': 'y = 1\n',
    }
    return mod_a, mod_b, task_files_content


class TestRow4NewVsPreexistingBaselineAttribution:
    """Row 4 (PRD boundary-test sketch): baseline seeded with
    ``ROW4_PREEXISTING_TEST_ID`` (moda); the branch fails BOTH moda (the
    seeded pre-existing id) AND modb (a genuinely new id). The merge gate
    must block citing ONLY the new modb id — never the pre-existing moda id
    — and must NOT route MAIN_HEALTH_RED (that route is reserved for a
    WHOLLY pre-existing failure — row 5).
    """

    @pytest.mark.asyncio
    @pytest.mark.exercise_merge_verify
    async def test_row4_new_failure_blocked_preexisting_routes_main_health(
        self, tmp_path: Path,
    ) -> None:
        mod_a, mod_b, task_files_content = _row4_mixed_baseline_diff(tmp_path)
        seed_main_baseline(ROW4_MAIN_SHA, frozenset({ROW4_PREEXISTING_TEST_ID}))

        outcome = await _drive_merge_gate(
            tmp_path,
            task_id='row4-6061',
            module_configs_registry={'moda': mod_a, 'modb': mod_b},
            touched_module_configs=[mod_a, mod_b],
            task_files=list(task_files_content),
            task_files_content=task_files_content,
            main_sha=ROW4_MAIN_SHA,
            run_verification_fake=_fake_run_verification_by_module({
                mod_a.prefix: (False, [ROW4_PREEXISTING_TEST_ID]),
                mod_b.prefix: (False, [ROW4_NEW_TEST_ID]),
            }),
        )

        assert outcome is not None, 'expected a blocked MergeOutcome, got the verify-passed sentinel'
        assert outcome.status == 'blocked', f'expected blocked; got {outcome.status!r}'
        assert not outcome.reason.startswith(MAIN_HEALTH_RED_REASON_PREFIX), (
            f'a MIXED failure (one new id alongside a pre-existing one) must '
            f'never route MAIN_HEALTH_RED; got {outcome.reason!r}'
        )
        assert ROW4_NEW_TEST_ID in outcome.reason, (
            f'expected the NEW modb test id to be cited; got {outcome.reason!r}'
        )
        assert ROW4_PREEXISTING_TEST_ID not in outcome.reason, (
            f'the PRE-EXISTING moda test id must NOT be charged to the '
            f'branch; got {outcome.reason!r}'
        )


# ---------------------------------------------------------------------------
# Row 5: "wholly pre-existing" + cache hit (μ, B1/B2). Consumer-only.
# ---------------------------------------------------------------------------

ROW5_MAIN_SHA: str = 'r5main0000000000000000000000000000000000'
ROW5_PREEXISTING_TEST_ID: str = 'moda/tests/test_z.py::test_z'


def _row5_wholly_preexisting_diff(
    tmp_path: Path,
) -> tuple[ModuleConfig, dict[str, str]]:
    """Build row 5's wholly-pre-existing diff: a merge touching ONE registered
    module (moda) whose only failing id is the SAME id seeded into the main
    baseline (:data:`ROW5_PREEXISTING_TEST_ID`) — the branch introduces no
    failure of its own, mirroring :func:`_row4_mixed_baseline_diff` but
    scoped to a single module (row 5 has no NEW-failing sibling).

    Returns ``(mod_a, task_files_content)`` — *task_files_content* is written
    into the MERGE worktree by :func:`_drive_merge_gate` (see its docstring's
    footgun paragraph), keyed by relative path.
    """
    mod_a, _mod_b, _config = _two_module_registry(tmp_path, breadth='full')
    task_files_content = {'moda/thing.py': 'x = 1\n'}
    return mod_a, task_files_content


class TestRow5WhollyPreexistingMainHealthRedAndCacheHit:
    """Row 5 (PRD boundary-test sketch): main already carries a pre-existing
    red (``ROW5_PREEXISTING_TEST_ID``, seeded into the per-main-SHA
    baseline); the branch introduces NO new failure — every failing id it
    reports is already in the baseline. TWO separate merges against the
    SAME main SHA must BOTH route MAIN_HEALTH_RED (the branch is never
    charged). The REAL classification entry point
    (``verify_failure_is_preexisting_on_main``) — wrapped, not replaced, so
    its real logic genuinely runs — is expected to be invoked ONCE PER
    MERGE (twice total): that entry point is not itself the cache. The
    actual cache-hit guarantee (μ B2) lives one layer inside, in
    ``main_baseline_failing_ids``'s ``_BASELINE_FAILING_IDS_CACHE``, and is
    asserted by the real, expensive main-probe worktree
    (``git_ops.ephemeral_worktree``) never being reached — zero calls
    across both merges against the pre-seeded baseline.
    """

    @pytest.mark.asyncio
    @pytest.mark.exercise_merge_verify
    async def test_row5_wholly_preexisting_main_health_red_and_cache_hit(
        self, tmp_path: Path,
    ) -> None:
        mod_a, task_files_content = _row5_wholly_preexisting_diff(tmp_path)
        seed_main_baseline(ROW5_MAIN_SHA, frozenset({ROW5_PREEXISTING_TEST_ID}))

        # Shared across both drives (not the default fresh-per-call double)
        # so ``ephemeral_worktree`` — the real, expensive main-probe
        # worktree the μ B2 baseline cache is supposed to make unnecessary —
        # can be inspected once, after both merges, below.
        shared_git_ops = _make_git_ops(tmp_path)
        probe_spy = AsyncMock(wraps=verify_failure_is_preexisting_on_main)
        with patch(
            'orchestrator.merge_queue.verify_failure_is_preexisting_on_main',
            new=probe_spy,
        ):
            outcome1 = await _drive_merge_gate(
                tmp_path,
                task_id='row5-8081-a',
                module_configs_registry={'moda': mod_a},
                touched_module_configs=[mod_a],
                task_files=list(task_files_content),
                task_files_content=task_files_content,
                main_sha=ROW5_MAIN_SHA,
                run_verification_fake=_fake_run_verification_by_module(
                    {mod_a.prefix: (False, [ROW5_PREEXISTING_TEST_ID])},
                ),
                git_ops=shared_git_ops,
            )
            outcome2 = await _drive_merge_gate(
                tmp_path,
                task_id='row5-8081-b',
                module_configs_registry={'moda': mod_a},
                touched_module_configs=[mod_a],
                task_files=list(task_files_content),
                task_files_content=task_files_content,
                main_sha=ROW5_MAIN_SHA,
                run_verification_fake=_fake_run_verification_by_module(
                    {mod_a.prefix: (False, [ROW5_PREEXISTING_TEST_ID])},
                ),
                git_ops=shared_git_ops,
            )

        for label, outcome in (('first', outcome1), ('second', outcome2)):
            assert outcome is not None, f'expected a blocked MergeOutcome ({label} merge)'
            assert outcome.status == 'blocked', (
                f'expected blocked ({label} merge); got {outcome.status!r}'
            )
            assert outcome.reason.startswith(MAIN_HEALTH_RED_REASON_PREFIX), (
                f'a wholly pre-existing failure ({label} merge) must route '
                f'MAIN_HEALTH_RED — the branch must never be charged; got '
                f'{outcome.reason!r}'
            )

        # The classification entry point (_classify_main_health_red ->
        # verify_failure_is_preexisting_on_main) is dispatched fresh on
        # EVERY failing merge — that's expected, not a cache miss: the μ B2
        # baseline cache lives ONE LAYER INSIDE this entry point (see
        # main_baseline_failing_ids's _BASELINE_FAILING_IDS_CACHE lookup),
        # so the entry point itself is genuinely called once per merge. This
        # assertion is a sanity check that classification was truly engaged
        # both times, not skipped or short-circuited some other way.
        assert probe_spy.await_count == 2, (
            f'expected the classification entry point to be consulted once '
            f'per merge (both merges must independently engage main-health '
            f'classification); got {probe_spy.await_count} call(s)'
        )
        # The actual "cache hit, never re-probed" guarantee (B2): a real
        # main-probe worktree is the EXPENSIVE operation the baseline cache
        # exists to avoid. The baseline was pre-seeded above, so
        # main_baseline_failing_ids must cache-hit on BOTH calls and never
        # fall through to git_ops.ephemeral_worktree at all.
        _ephemeral_worktree_mock = cast(AsyncMock, shared_git_ops.ephemeral_worktree)
        assert _ephemeral_worktree_mock.call_count == 0, (
            f'expected ZERO real main-probe worktrees across both merges '
            f'against the same pre-seeded main SHA; got '
            f'{_ephemeral_worktree_mock.call_count} call(s) — the '
            f'μ B2 baseline cache should have short-circuited before ever '
            f'reaching a cold probe'
        )


# ---------------------------------------------------------------------------
# Row 6: "infra non-consumption" (ν, I1), BOTH consumers. A PERSISTENT
# infra-transient classified VerifyResult (never clearing — a member of
# INFRA_TRANSIENT_CATEGORIES, e.g. a semaphore-acquisition timeout, 2549
# pattern) must consume ZERO attempts and dispatch NO debugger/steward at
# EITHER consumer seam, and must eventually exhaust into a LOUD hold — never
# a silent infinite retry — at both.
# ---------------------------------------------------------------------------

ROW6_MAIN_SHA: str = 'r6main0000000000000000000000000000000000'
# Matches the default INFRA_TRANSIENT_CATEGORIES member both
# test_workflow_verify_infra_resume._infra_category_result and
# test_merge_queue._infra_category_verify_result already default to.
ROW6_INFRA_CATEGORY: str = 'semaphore_timeout'


class TestRow6InfraTransientConsumesNoAttemptBothConsumers:
    """Row 6 (PRD boundary-test sketch): a PERSISTENT infra-transient
    classified ``VerifyResult`` (never clears) must consume NO attempt and
    dispatch NO debugger/steward at either consumer seam — ν's I1 guarantee —
    and the bounded retry window must eventually EXHAUST into a LOUD hold
    (never a silent infinite retry) at both.
    """

    @pytest.mark.asyncio
    @pytest.mark.exercise_merge_verify
    async def test_row6_infra_transient_consumes_no_attempt_both_consumers(
        self, tmp_path: Path,
    ) -> None:
        assert ROW6_INFRA_CATEGORY in INFRA_TRANSIENT_CATEGORIES, (
            f'sanity: {ROW6_INFRA_CATEGORY!r} must be an INFRA_TRANSIENT_CATEGORIES '
            f'member for this row to exercise the intended ν policy path'
        )

        # -- TASK-verify side: drive the REAL _verify_debugfix_loop (via the
        # test_workflow_verify_infra_resume.py TaskWorkflow factory) with a
        # PERSISTENT infra-transient VerifyResult (never clears).
        wf = _make_workflow(verify_infra_retry_max_attempts=3)

        task_call_count = 0

        async def always_infra_category(*args, **kwargs):
            nonlocal task_call_count
            task_call_count += 1
            return _infra_category_result(category=ROW6_INFRA_CATEGORY)

        with (
            patch(
                'orchestrator.workflow.run_scoped_verification',
                side_effect=always_infra_category,
            ),
            patch('asyncio.sleep', new_callable=AsyncMock),
            patch.object(wf, '_invoke', new=AsyncMock()) as mock_invoke,
        ):
            task_outcome = await wf._verify_debugfix_loop()

        assert task_outcome == WorkflowOutcome.BLOCKED, (
            f'expected the persistent infra-transient hold pathway (BLOCKED); '
            f'got {task_outcome!r}'
        )
        assert task_call_count == 3, (
            f'expected the BOUNDED retry window (verify_infra_retry_max_attempts='
            f'3) to exhaust rather than retry silently forever; got '
            f'{task_call_count} call(s)'
        )
        assert wf.metrics.verify_attempts == 0, (
            f'a persistent infra-transient failure must consume ZERO verify '
            f'attempts; got {wf.metrics.verify_attempts}'
        )
        mock_invoke.assert_not_awaited()
        assert not wf._mark_blocked.called, (  # type: ignore[attr-defined]
            '_verify_debugfix_loop must not call _mark_blocked directly on '
            'infra exhaustion (the hold pathway) — that is run()\'s job via '
            'the _infra_hold_info stash'
        )
        # The LOUD tail: exhaustion must stamp a loud, human-escalating hold —
        # never a silent drop.
        assert wf._infra_hold_info is not None, (
            '_infra_hold_info must be set on exhaustion (the loud hold signal)'
        )
        assert wf._infra_hold_info.get('category') == 'infra_issue'
        assert wf._infra_hold_info.get('escalate_to_human') is True

        # -- MERGE-verify side: drive the REAL merge gate (_run_post_merge_verify)
        # with the SAME persistent infra-transient category. A single touched,
        # unregistered-sibling module keeps the aggregate category exactly
        # ROW6_INFRA_CATEGORY (no widening to muddy the classification).
        mod_a, _mod_b, _config = _two_module_registry(tmp_path, breadth='full')
        my_timeouts: dict[str, int] = {}
        my_enospc_retries: dict[str, int] = {}
        merge_fake = _fake_run_verification_by_module(
            {mod_a.prefix: (False, [])}, category=ROW6_INFRA_CATEGORY,
        )
        outcome = await _drive_merge_gate(
            tmp_path,
            task_id='row6-1213',
            module_configs_registry={'moda': mod_a},
            touched_module_configs=[mod_a],
            task_files=['moda/thing.py'],
            task_files_content={'moda/thing.py': 'x = 1\n'},
            main_sha=ROW6_MAIN_SHA,
            run_verification_fake=merge_fake,
            timeouts=my_timeouts,
            enospc_retries=my_enospc_retries,
        )

        assert outcome is not None, (
            'expected a blocked MergeOutcome (infra hold), got the '
            'verify-passed sentinel'
        )
        assert outcome.status == 'blocked', f'expected blocked; got {outcome.status!r}'
        assert outcome.reason.startswith(TRANSIENT_INFRA_REASON_PREFIX), (
            f'a persistent infra-transient merge-verify failure must route the '
            f'loud transient-infra hold; got {outcome.reason!r}'
        )
        assert my_timeouts == {}, (
            f'a persistent infra-transient outcome must not bump the merge '
            f'timeout loop-breaker (no merge verify attempt consumed); got '
            f'{my_timeouts}'
        )
        assert merge_fake.await_count == 2, (
            f'expected the BOUNDED shared enospc/infra retry budget (one '
            f'in-place retry) to exhaust rather than retry silently forever; '
            f'got {merge_fake.await_count} call(s)'
        )


# ---------------------------------------------------------------------------
# Row 7: "train amortization" (λ/σ, T1). A 3-member line-stacked Python
# train lands via exactly ONE broad, per-module merge-role verify of the
# TIP — the amortization win — driven through the REAL
# ``MergeWorker._do_merge`` -> ``_do_train_merge`` -> ``_run_post_merge_verify``
# chokepoint chain. Adapts test_train_integration.py's real-git train harness
# (``make_stacked_member`` / ``build_group_merge_request`` / ``_SpyEventStore``
# / the test-local ``MergeWorker`` reference) to non-cargo Python modules
# with an instrumented fake ``run_verification`` — no cargo, no ssh.
# ---------------------------------------------------------------------------


async def _seed_train_workspace_repo(tmp_path: Path) -> Path:
    """Git-init a fresh repo seeded with a trivial 3-module Python layout
    (``moda``/``modb``/``modc``, one placeholder source file each) and an
    initial commit on ``main``.

    Row 7's non-cargo analogue of
    ``test_train_integration.seed_workspace_repo`` (which copies a REAL
    cargo fixture) — no cargo and no real pytest: row 7's verify is driven
    entirely through an instrumented fake ``run_verification``, so the
    on-disk module contents only need to exist for real ``git diff``/rebase
    plumbing to have something to operate on.
    """
    repo = tmp_path
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    for prefix in ('moda', 'modb', 'modc'):
        mod_dir = repo / prefix
        mod_dir.mkdir(parents=True, exist_ok=True)
        (mod_dir / 'thing.py').write_text(f'{prefix}_value = 0\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'initial workspace'], cwd=repo)
    return repo


def _three_module_train_registry(
    repo: Path,
) -> tuple[ModuleConfig, ModuleConfig, ModuleConfig, OrchestratorConfig]:
    """A 3-module registry (moda/modb/modc), ``merge_verify_breadth='full'``,
    for the row-7 train — the ``_two_module_registry`` shape extended to a
    third module (one per stacked member).

    Unlike :func:`test_verify_scope_lambda._two_module_registry`,
    ``type_check_command``/``lint_command`` are left at their ``None``
    default. Row 7 is the ONLY row in this module whose scoped verify phase
    actually PASSES (the merge must land, unlike every other row here, which
    asserts a blocked/held outcome) — so it uniquely reaches
    ``LocalRunner.run_merge_verify``'s post-scoped unscoped-pyright gate
    (``_run_unscoped_typechecks``, called once pre-advance and again inside
    ``_finalize_advanced_merge`` post-advance). Every other row's scoped
    phase fails first, short-circuiting before that gate ever runs (see
    ``LocalRunner.run_merge_verify``: ``if not scoped.passed: return
    scoped``). A ``None`` ``type_check_command`` keeps
    ``_run_unscoped_typechecks``'s ``active`` list empty, so neither call
    ever spawns a real, unpatched ``pyright`` subprocess — that function is
    reached via ``orchestrator.merge_queue``'s OWN imported
    ``run_verification`` reference, a DIFFERENT binding from the
    ``orchestrator.verify.run_verification`` seam this module patches
    everywhere else, so it would not be faked away like the scoped phase is.
    """
    mod_a = ModuleConfig(prefix='moda', test_command='uv run --directory moda pytest tests/')
    mod_b = ModuleConfig(prefix='modb', test_command='uv run --directory modb pytest tests/')
    mod_c = ModuleConfig(prefix='modc', test_command='uv run --directory modc pytest tests/')
    config = OrchestratorConfig(
        project_root=repo,
        merge_verify_breadth='full',
        git=GitConfig(
            main_branch='main', branch_prefix='task/', remote='origin',
            worktree_dir='.worktrees', push_after_advance=False,
        ),
    )
    config._module_configs = {'moda': mod_a, 'modb': mod_b, 'modc': mod_c}
    return mod_a, mod_b, mod_c, config


class TestRow7TrainAmortizationOneFullBreadthVerify:
    """Row 7 (PRD boundary-test sketch): a 3-member line-stacked Python train
    lands via exactly ONE post-merge verify (``is_merge_verify=True,
    role='merge'``) of the tip — the amortization win (λ/σ, T1) — and that
    ONE verify's executed configs are PER-MODULE (each registered module's
    own ``test_command``), never a single opaque global &&-chain call.
    """

    @pytest.mark.asyncio
    async def test_row7_three_member_train_one_full_breadth_verify(
        self, tmp_path: Path,
    ) -> None:
        repo = await _seed_train_workspace_repo(tmp_path)
        mod_a, mod_b, mod_c, config = _three_module_train_registry(repo)
        git_ops = GitOps(config.git, repo)

        def edit_a(wt: Path) -> None:
            (wt / 'moda' / 'thing.py').write_text('moda_value = 1\n')

        def edit_b(wt: Path) -> None:
            (wt / 'modb' / 'thing.py').write_text('modb_value = 1\n')

        def edit_c(wt: Path) -> None:
            (wt / 'modc' / 'thing.py').write_text('modc_value = 1\n')

        _, main_sha, _ = await _run(['git', 'rev-parse', 'main'], cwd=repo)
        main_sha = main_sha.strip()

        wt_1, sha_1 = await make_stacked_member(git_ops, 'row7_m1', main_sha, edit_a)
        wt_2, sha_2 = await make_stacked_member(git_ops, 'row7_m2', sha_1, edit_b)
        wt_3, _sha_3 = await make_stacked_member(git_ops, 'row7_m3', sha_2, edit_c)
        del wt_1, wt_2  # only the tip worktree is threaded into the merge request

        verify_calls: list[dict] = []

        async def _spy_verify(*args, **kwargs):
            verify_calls.append({'args': args, 'kwargs': kwargs})
            return await run_scoped_verification(*args, **kwargs)

        spy_events = _SpyEventStore()
        req = build_group_merge_request(
            git_ops=git_ops, config=config, train_id='train-row7',
            member_names=['row7_m1', 'row7_m2', 'row7_m3'],
            tip_name='row7_m3', tip_worktree=wt_3,
        )
        # Non-empty so run_scoped_verification's module_configs branch is even
        # entered; merge+full then widens it wholesale to the FULL registry
        # (config.module_configs_or_empty — all 3 modules) regardless of
        # which subset is listed here (lambda's train-tip-shaped-call
        # precedent, test_train_tip_shaped_call_widens_to_every_registered_module,
        # explicitly labeled "boundary row 7 plan shape"). Left empty (the
        # build_group_merge_request default), the call instead falls through
        # to the no-module_configs FALLBACK path — a single synthetic
        # '__fallback__' ModuleConfig, not a per-module fan-out (step-13's
        # RED state).
        req.module_configs = [mod_a]

        run_verification_fake = _fake_run_verification_by_module({
            mod_a.prefix: (True, []), mod_b.prefix: (True, []), mod_c.prefix: (True, []),
        })

        queue: asyncio.Queue = asyncio.Queue()
        worker = MergeWorker(git_ops, queue, event_store=spy_events)

        with (
            patch('orchestrator.merge_queue.run_scoped_verification', side_effect=_spy_verify),
            patch.object(verify, 'run_verification', new=run_verification_fake),
        ):
            outcome = await worker._do_merge(req)

        assert outcome is not None
        assert outcome.status == 'done', f'expected the train to land; got {outcome!r}'

        merge_verify_calls = [
            c for c in verify_calls if c['kwargs'].get('is_merge_verify') is True
        ]
        assert len(merge_verify_calls) == 1, (
            f'expected exactly ONE post-merge verify — the amortization win '
            f'across 3 members, not one verify per member; got '
            f'{len(merge_verify_calls)}: {verify_calls}'
        )
        assert merge_verify_calls[0]['kwargs'].get('role') == 'merge', (
            f"expected role='merge' on the post-merge call; got "
            f"{merge_verify_calls[0]['kwargs']!r}"
        )

        executed = {mc.prefix: mc for mc in _executed_module_configs(run_verification_fake)}
        assert set(executed) == {'moda', 'modb', 'modc'}, (
            f'expected the one broad verify to fan out per-module across '
            f'every REGISTERED module (never a single opaque global-chain '
            f'call); got {set(executed)!r}'
        )
        for mc in (mod_a, mod_b, mod_c):
            assert executed[mc.prefix].test_command == mc.test_command, (
                f"expected {mc.prefix} to run its OWN verbatim full-suite "
                f'test command, not a combined/opaque command; got '
                f'{executed[mc.prefix].test_command!r}'
            )


# ---------------------------------------------------------------------------
# Row 8: "rollback path" (λ, R4). ``merge_verify_breadth='scoped'`` (the
# shipped default) keeps the merge-role plan for a source-only diff
# BYTE-IDENTICAL to the pre-λ legacy shape — the opposite pole from row 1
# (breadth='full' widens to the whole registry) and row 2 (role='task'
# floors pytest, R3): NEITHER applies here, so the touched module's OWN
# pytest stays SKIPPED, exactly as it always has. Producer-only (see the
# module docstring's row->seam map — no consumer/merge-gate side for this
# row).
# ---------------------------------------------------------------------------


def _row8_assert_legacy_scoped_shape(executed_mc: ModuleConfig) -> None:
    """Assert *executed_mc* (the executed rendering of the touched module)
    matches the BYTE-IDENTICAL-TO-LEGACY shape ``role='merge'`` +
    ``merge_verify_breadth='scoped'`` must produce for a source-only diff
    (R4, the rollback golden) — the legacy-golden reference row 8 pins.

    pytest must be SKIPPED (``None``): a source-only diff has no
    collectable test file to file-scope, and the R3 task-role pytest floor
    (``TestRow2TaskRoleSignal`` above) is gated to ``role == 'task'`` only
    (``verify_plan._derive_module_runs``'s ``elif role == 'task' and
    production_trigger is not None:`` branch — moved ABOVE the
    collectable-tests branch by task 3294, still role-gated, so this row's
    outcome is unchanged) — ``role == 'merge'`` always falls through to the
    legacy ``else:`` SKIPPED arm, unchanged by λ (R4). lint/pyright, in contrast, stay
    FILE_SCOPED and non-``None``: D1/D2 file-scoping never forked on role
    or ``merge_verify_breadth`` to begin with, so R4 has nothing to roll
    back there.
    """
    assert executed_mc.test_command is None, (
        f"expected pytest SKIPPED for a source-only diff at role='merge' + "
        f"breadth='scoped' (the pre-λ legacy shape — no task-role floor "
        f'leaks into merge role, R4); got {executed_mc.test_command!r}'
    )
    assert executed_mc.lint_command is not None, (
        'expected lint to still run FILE_SCOPED (R4 only concerns the '
        'pytest floor; lint/pyright file-scoping is role-independent)'
    )
    assert executed_mc.type_check_command is not None, (
        'expected pyright to still run FILE_SCOPED (R4 only concerns the '
        'pytest floor; lint/pyright file-scoping is role-independent)'
    )


class TestRow8ScopedBreadthMergePlanByteIdenticalLegacy:
    """Row 8 (PRD boundary-test sketch): ``merge_verify_breadth='scoped'``
    (the shipped default) keeps the merge-role plan byte-identical to the
    pre-λ legacy shape (R4, the rollback path). Reuses the SAME row-1
    golden diff (source-only under modA, sibling test under modB), this
    time with ``breadth='scoped'`` — modB must never execute (that
    widening is breadth='full'-gated, row-1 territory) and modA's own
    pytest must stay SKIPPED (that floor is role='task'-gated, row-2
    territory).
    """

    @pytest.mark.asyncio
    async def test_row8_scoped_breadth_merge_plan_byte_identical_legacy(
        self, tmp_path: Path,
    ) -> None:
        mod_a, _mod_b, config = _row1_golden_diff(tmp_path, breadth='scoped')

        result, executed, fake = await _drive_producer(
            tmp_path, config, [mod_a], task_files=[MODA_SOURCE_PATH], role='merge',
            is_merge_verify=True,
        )

        assert result.passed, (
            'a scoped-breadth merge verify of a source-only diff must pass '
            '(pytest SKIPPED is a no-op leg, never a failure)'
        )
        assert set(executed) == {'moda'}, (
            f'expected NO widening to the registered-but-untouched sibling '
            f'modB (merge+full-only widening, row-1 territory, is gated '
            f'off by breadth=scoped); got {set(executed)!r}'
        )
        assert fake.await_count == 1, (
            f'expected exactly ONE run_verification call (modA only — no '
            f'widening, no per-tool fan-out); got {fake.await_count} call(s)'
        )
        _row8_assert_legacy_scoped_shape(executed['moda'])


# ---------------------------------------------------------------------------
# Row 9: "fallback narrowing" (the wall-clock win, asserted structurally). A
# diff touching an UNREGISTERED path (kappa's UNREGISTERED_PATH_DIFF shape)
# against the REAL dark_factory root-level fleet lint/type/test chains
# (_FLEET_*_COMMAND, test_verify_scope_kappa.py) must route through the
# plan-driven FALLBACK branch scoped to ONE subproject — never a
# per-fleet-subproject fan-out and never the whole-repo opaque chain
# leaking other subprojects' names into the executed command. Producer-only
# (see the module docstring's row->seam map).
# ---------------------------------------------------------------------------


def _row9_unregistered_scripts_diff(tmp_path: Path) -> tuple[OrchestratorConfig, list[str]]:
    """Write row 9's fallback-narrowing diff to *tmp_path* and return
    ``(config, task_files)``.

    Reuses kappa's ``UNREGISTERED_PATH_DIFF`` shape verbatim (a single
    test file under an unregistered ``tests/scripts/`` path — no
    ``ModuleConfig`` claims it, so ``module_configs=[]`` at every call
    site) against the REAL dark_factory root-level fleet lint/type/test
    chains (``_FLEET_LINT_COMMAND``/``_FLEET_TYPE_COMMAND``/
    ``_FLEET_TEST_COMMAND``, test_verify_scope_kappa.py) — the same
    OPAQUE multi-clause ``&&``-chain commands the real orchestrator/
    config.yaml carries — so the fallback branch's OPAQUE first-clause
    subproject rescoping (``_build_fallback_config``) is exercised for
    real, not a synthetic single-clause stand-in.
    """
    test_path = UNREGISTERED_PATH_DIFF[0]
    full = tmp_path / test_path
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text('def test_x():\n    pass\n')

    config = OrchestratorConfig(
        project_root=tmp_path,
        test_command=_FLEET_TEST_COMMAND,
        lint_command=_FLEET_LINT_COMMAND,
        type_check_command=_FLEET_TYPE_COMMAND,
    )
    return config, list(UNREGISTERED_PATH_DIFF)


class TestRow9FallbackNarrowingNeverWholeRepoChain:
    """Row 9 (PRD boundary-test sketch): a diff touching only an
    unregistered path (no ``ModuleConfig`` claims it) against the REAL
    dark_factory root-level fleet lint/type/test chains must scope to
    exactly ONE fallback ``ModuleConfig`` — never a per-fleet-subproject
    fan-out (5 separate ``run_verification`` calls) and never the
    unscoped opaque chain leaking every OTHER subproject's name into the
    executed lint/type command (the wall-clock win: this is what makes a
    task-verify of a single-file diff fast instead of running the entire
    fleet's suite).
    """

    @pytest.mark.asyncio
    async def test_row9_scripts_module_scoped_no_whole_repo_chain(
        self, tmp_path: Path,
    ) -> None:
        config, task_files = _row9_unregistered_scripts_diff(tmp_path)

        result, executed, fake = await _drive_producer(
            tmp_path, config, [], task_files=task_files, role='task',
        )

        assert result.passed
        assert fake.await_count == 1, (
            f'expected exactly ONE scoped call, not a per-subproject '
            f'fan-out or an extra global fallthrough; got '
            f'{fake.await_count} call(s)'
        )
        assert set(executed) == {'__fallback__'}, (
            f'expected exactly one scoped ModuleConfig routed through the '
            f'plan-driven fallback branch, not a per-subproject fan-out or '
            f'the unscoped global branch; got {set(executed)!r}'
        )
        fallback = executed['__fallback__']
        test_path = UNREGISTERED_PATH_DIFF[0]
        # Positive equality (mirrors kappa's identical golden,
        # test_opaque_fleet_chain_lint_type_scoped_to_first_clause) rather
        # than a substring exclusion list: this pins the exact scoped shape,
        # so ANY fleet subproject name surviving scoping — not just the ones
        # named in a hardcoded exclusion set — fails the assertion.
        assert fallback.lint_command == f'uv run --project shared ruff check {test_path}', (
            f'expected the fleet lint command scoped to exactly the touched '
            f'file, no other subproject name surviving; got '
            f'{fallback.lint_command!r}'
        )
        assert fallback.type_check_command == f'npx pyright {test_path}', (
            f'expected the fleet type-check command scoped to exactly the '
            f'touched file, no other subproject name surviving; got '
            f'{fallback.type_check_command!r}'
        )


# ---------------------------------------------------------------------------
# Row 10: "plan authority, both roles" (κ, A1). The runs executed are
# EXACTLY plan.runs, at EITHER role — run_scoped_verification never
# re-derives or drifts from derive_verify_plan's decision — and
# VerifyResult.plan reflects the EXECUTED plan, never a stale
# independently-rederived diagnostic mirror. Producer-only (see the module
# docstring's row->seam map). This is the FINAL row: once GREEN, the whole
# 10-row module passes — the leaf signal that the inversion composes
# end-to-end.
# ---------------------------------------------------------------------------


# modA's touched file is a collectable TEST file and NOTHING ELSE under that
# prefix — no SOURCE/STRUCTURAL file — so pytest is FILE_SCOPED to it at BOTH
# roles. (Narrow claim, deliberately: since task 3294 the collectable-tests
# branch is reached only when no production file was touched under the same
# prefix, because the role='task' floor now sits ABOVE it. A modA diff that
# ALSO touched a production file would fork by role, which is not the shape
# this row exercises.) modB's touched file is a plain SOURCE file with no
# collectable test alongside it, which DOES fork by role: FULL_SUITE at
# role='task' (the R3 owning-module pytest floor) vs SKIPPED at role='merge'
# (R4 — the legacy shape).
ROW10_MODA_TEST_PATH: str = 'moda/tests/test_thing.py'
ROW10_MODA_TEST_CONTENT: str = 'def test_thing():\n    pass\n'
ROW10_MODB_SOURCE_PATH: str = 'modb/helpers.py'
ROW10_MODB_SOURCE_CONTENT: str = 'def helper():\n    return 1\n'


def _row10_mixed_diff(
    tmp_path: Path,
) -> tuple[ModuleConfig, ModuleConfig, OrchestratorConfig, list[str]]:
    """Write row 10's mixed two-module diff to *tmp_path* and return
    ``(mod_a, mod_b, config, task_files)``.

    Reuses ``_two_module_registry(breadth='scoped')`` — the shipped-default
    breadth — so role='merge' never widens to merge+full's
    FULL_SUITE-everything gate (row-1/8 territory): that widening would put
    BOTH modules on FULL_SUITE regardless of which files were touched,
    masking the very role-fork this row exists to pin (see module-level
    constants above for which file forks and which doesn't).
    """
    mod_a, mod_b, config = _two_module_registry(tmp_path, breadth='scoped')

    test_full = tmp_path / ROW10_MODA_TEST_PATH
    test_full.parent.mkdir(parents=True, exist_ok=True)
    test_full.write_text(ROW10_MODA_TEST_CONTENT)

    source_full = tmp_path / ROW10_MODB_SOURCE_PATH
    source_full.parent.mkdir(parents=True, exist_ok=True)
    source_full.write_text(ROW10_MODB_SOURCE_CONTENT)

    task_files = [ROW10_MODA_TEST_PATH, ROW10_MODB_SOURCE_PATH]
    return mod_a, mod_b, config, task_files


def _row10_expected_command(
    mc: ModuleConfig, run: verify_plan.PlannedRun,
) -> str | None:
    """The command *mc* is expected to execute for *run* — independently
    derived from the plan, never by delegating to
    ``verify._executed_module_configs_from_plan`` itself (that bridge is
    what this row pins, not a helper to lean on).

    Mirrors the two rendering rules that bridge's own docstring documents:
    a SKIPPED run (``run.cmd is None``) means that check never runs at all
    (``None``); a FULL_SUITE run reuses *mc*'s OWN verbatim command
    (preserving e.g. a ``--directory`` flag — ``render(run.cmd)`` would
    instead normalise it to a leading ``cd``, which is argv-equivalent but
    NOT byte-identical, per ``verify_cmd.render``'s own docstring — so
    FULL_SUITE must compare against *mc*'s field directly, never a
    re-rendering of ``run.cmd``); every other non-SKIPPED scope_kind
    reachable here (FILE_SCOPED — TRIVIAL never carries a real *mc*) renders
    ``run.cmd`` via ``orchestrator.verify_cmd.render``.
    """
    if run.cmd is None:
        return None
    attr = _mc_attr_for_run_or_none(run)
    if attr is None:
        return None
    if run.scope_kind is verify_plan.ScopeKind.FULL_SUITE:
        return getattr(mc, attr)
    return render(run.cmd)


class TestRow10PlanAuthorityBothRoles:
    """Row 10 (PRD boundary-test sketch): the runs executed are exactly
    ``plan.runs`` — A1 — at BOTH role='task' and role='merge'. A mixed diff
    (a touched TEST file under modA, role-independent; a touched SOURCE-only
    file under modB, which forks by role — FULL_SUITE pytest at role='task'
    (R3), SKIPPED at role='merge' (R4, breadth='scoped')) exercises both a
    role-independent and a role-forking module in one plan.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize('role', ['task', 'merge'])
    async def test_row10_executed_commands_equal_plan_runs_both_roles(
        self, tmp_path: Path, role: Literal['task', 'merge'],
    ) -> None:
        mod_a, mod_b, config, task_files = _row10_mixed_diff(tmp_path)
        module_configs = [mod_a, mod_b]
        existing_files = [f for f in task_files if (tmp_path / f).exists()]

        result, executed, _fake = await _drive_producer(
            tmp_path, config, module_configs, task_files=task_files, role=role,
        )

        assert result.passed

        expected_plan = verify_plan.derive_verify_plan(
            existing_files, module_configs, config,
            _real_worktree_reader(tmp_path), role=role,
        )

        # A1: no module executes that the plan didn't include, and no
        # module the plan included (with a real command) is left out.
        plan_prefixes_with_cmd = {
            run.module_prefix for run in expected_plan.runs if run.cmd is not None
        }
        assert set(executed) == plan_prefixes_with_cmd, (
            f'expected the executed module set to equal the plan '
            f'(role={role!r}); executed={set(executed)!r} '
            f'plan={plan_prefixes_with_cmd!r}'
        )

        # A1: every executed command is EXACTLY what the plan's run for
        # that (module, tool) prescribes — never a re-derived drift.
        runs_by_module: dict[str, list[verify_plan.PlannedRun]] = {}
        for run in expected_plan.runs:
            runs_by_module.setdefault(run.module_prefix, []).append(run)

        for mc in module_configs:
            executed_mc = executed.get(mc.prefix)
            for run in runs_by_module.get(mc.prefix, []):
                attr = _mc_attr_for_run_or_none(run)
                if attr is None:
                    continue
                expected_cmd = _row10_expected_command(mc, run)
                actual_cmd = getattr(executed_mc, attr) if executed_mc is not None else None
                assert actual_cmd == expected_cmd, (
                    f'{mc.prefix}.{attr} (role={role!r}): executed '
                    f'{actual_cmd!r} != plan-prescribed {expected_cmd!r} '
                    f'(scope_kind={run.scope_kind!r})'
                )

        # A1: VerifyResult.plan is the plan that DROVE execution above, not
        # an independently re-derived diagnostic mirror.
        assert result.plan == expected_plan.to_dict(), (
            f'VerifyResult.plan (role={role!r}) must reflect the EXECUTED plan'
        )


# ---------------------------------------------------------------------------
# Row 6b (task 3173): the SAME ν/I1 non-consumption contract as row 6, driven
# by an EXTERNALLY KILLED leg — the category whose whole point is that the
# gate produced no verdict at all.
#
# The measured incident surfaced this to the human as
#   "Post-merge verification failed: Failures: lint issues"
# — a branch-blaming, human-routed BRANCH verdict for a lint process that was
# SIGKILLed at 0.31s having emitted zero diagnostics.  It must instead route
# the loud transient-infra hold (an infra_issue that bypasses the steward and
# consumes no merge attempt), and the reason must SAY what happened.
#
# The category AND summary here are produced by the real `_summarize_checks`
# rather than hand-authored literals, so this row cannot drift from the
# producer it is asserting on.
# ---------------------------------------------------------------------------

ROW6B_MAIN_SHA: str = 'r6bmain000000000000000000000000000000000'
ROW6B_LINT_CMD: str = './scripts/verify.sh lint --scope branch --include-infra'
ROW6B_LINT_OUT: str = 'DF_VERIFY_ROLE=merge — forcing --scope all\n'
ROW6B_LINT_DURATION: float = 0.31010722508654


def _fake_run_verification_killed_lint(passed_prefixes: dict[str, bool]) -> AsyncMock:
    """Row 6b's analogue of :func:`_fake_run_verification_by_module`, whose
    failing result's ``category``/``summary`` come from the REAL
    ``verify._summarize_checks`` for a lint leg killed by signal 9 at 0.31s.
    """
    from orchestrator.verify import _summarize_checks

    # Five-tuple since the task-3173 review amendment: the fifth element is
    # one category per FAILING leg, which merge_queue's veto gate reads instead
    # of inferring verdict-lessness from the severity-ranked aggregate.
    passed_flag, category, cause_hint, summary, failing_legs = _summarize_checks(
        0, '', False, None,
        -9, ROW6B_LINT_OUT, False, ROW6B_LINT_CMD,
        0, '', False, None,
        lint_duration=ROW6B_LINT_DURATION,
    )
    assert not passed_flag and category == 'infra_kill', (
        f'sanity: the producer must classify this as infra_kill; got {category!r}'
    )
    assert failing_legs == ['infra_kill'], (
        f'sanity: the only failing leg here is the killed lint leg; got {failing_legs!r}'
    )

    async def _fake(worktree, config, module_config=None, **kwargs):
        if module_config is None or passed_prefixes.get(module_config.prefix, True):
            return VerifyResult(
                passed=True, test_output='', lint_output='', type_output='',
                summary='All checks passed', failing_test_ids=[],
            )
        return VerifyResult(
            passed=False, test_output='', lint_output=ROW6B_LINT_OUT, type_output='',
            summary=summary, category=category, cause_hint=cause_hint,
            failing_test_ids=[],
        )
    return AsyncMock(side_effect=_fake)


class TestRow6bExternalKillIsAnInfraHoldNotABranchVerdict:
    """Row 6b (task 3173): a PERSISTENT externally-killed merge verify routes
    the loud transient-infra hold, consumes no attempt, and names the kill —
    it never surfaces as a branch verdict.
    """

    @pytest.mark.asyncio
    @pytest.mark.exercise_merge_verify
    async def test_row6b_persistent_kill_routes_the_infra_hold_and_says_so(
        self, tmp_path: Path,
    ) -> None:
        assert 'infra_kill' in INFRA_TRANSIENT_CATEGORIES, (
            "sanity: 'infra_kill' must be an INFRA_TRANSIENT_CATEGORIES member "
            'for this row to exercise the intended ν policy path'
        )

        mod_a, _mod_b, _config = _two_module_registry(tmp_path, breadth='full')
        my_timeouts: dict[str, int] = {}
        my_enospc_retries: dict[str, int] = {}
        merge_fake = _fake_run_verification_killed_lint({mod_a.prefix: False})
        outcome = await _drive_merge_gate(
            tmp_path,
            task_id='row6b-3173',
            module_configs_registry={'moda': mod_a},
            touched_module_configs=[mod_a],
            task_files=['moda/thing.py'],
            task_files_content={'moda/thing.py': 'x = 1\n'},
            main_sha=ROW6B_MAIN_SHA,
            run_verification_fake=merge_fake,
            timeouts=my_timeouts,
            enospc_retries=my_enospc_retries,
        )

        # (1) the loud transient-infra hold, NOT a branch verdict.
        assert outcome is not None, (
            'expected a blocked MergeOutcome (infra hold), got the '
            'verify-passed sentinel'
        )
        assert outcome.status == 'blocked', f'expected blocked; got {outcome.status!r}'
        assert outcome.reason.startswith(TRANSIENT_INFRA_REASON_PREFIX), (
            f'a persistent externally-killed merge verify must route the loud '
            f'transient-infra hold (infra_issue, bypasses the steward, consumes '
            f'no attempt); got {outcome.reason!r}'
        )

        # (2) the reason SAYS what actually happened — every clause measured.
        assert 'infra_kill' in outcome.reason
        assert 'signal 9' in outcome.reason
        assert 'no diagnostics produced' in outcome.reason
        assert 'indeterminate' in outcome.reason

        # (3) and never the two branch-blaming strings the incident surfaced.
        assert 'Post-merge verification failed' not in outcome.reason, (
            f'the incident surfaced the kill as a branch verdict; got '
            f'{outcome.reason!r}'
        )
        assert 'Failures: lint issues' not in outcome.reason, (
            f'the exact string the measured incident surfaced for a process '
            f'that emitted ZERO diagnostics; got {outcome.reason!r}'
        )

        # (4) ν/I1 non-consumption, matching row 6's both-consumers assertions.
        assert my_timeouts == {}, (
            f'a persistent externally-killed outcome must not bump the merge '
            f'timeout loop-breaker (no merge verify attempt consumed); got '
            f'{my_timeouts}'
        )
        assert merge_fake.await_count == 2, (
            f'expected the BOUNDED shared enospc/infra retry budget (one '
            f'in-place retry) to exhaust rather than retry silently forever; '
            f'got {merge_fake.await_count} call(s)'
        )
