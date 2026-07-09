"""Tests for orchestrator.merge_gates: extracted post-merge gates, finalize,
and reason-prefix constants (MQ-refactor task β).

These tests encode the behavior-preserving contracts of the module split,
mirroring task α's test_merge_types.py:

1. Module-existence — ``orchestrator.merge_gates`` exists and exports the
   full closure of moved symbols (reason prefixes, types, sentinel,
   functions).
2. Logger-name — the module logs under the ``orchestrator.merge_queue``
   logger name (not ``orchestrator.merge_gates``) so existing ``caplog``
   assertions filtered to the merge_queue logger keep capturing the moved
   gates' fail-open/fail-closed warnings.
3. Reach-back / string-path monkeypatch routing — the existing test suite
   monkeypatches merge-gate dependencies by STRING PATH
   ``orchestrator.merge_queue.<name>``.  A moved function must resolve a
   monkeypatched-or-staying sibling via a function-local deferred import so
   those patches stay effective even though the function body now lives in
   this module.  Each test below patches BOTH namespaces with CONTRASTING
   return values — the merge_gates-local (naive) patch would steer the
   outcome one way, the merge_queue (reach-back target) patch the other —
   so the assertion is unambiguous about which one governed.
4. Shim re-export identity (added in a later step, once merge_queue.py's
   shim swap lands).
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def test_merge_gates_exports_moved_public_symbols() -> None:
    from orchestrator.merge_gates import (
        _OVERLAP_GIT_ERROR_SENTINEL,
        DROPPED_PLAN_TARGETS_REASON_PREFIX,
        PLAN_FILES_NOT_TOUCHED_REASON_PREFIX,
        POST_MERGE_EQUIVALENCE_FAILED_REASON_PREFIX,
        POST_MERGE_PYRIGHT_BROKEN_REASON_PREFIX,
        DropGuardResult,
        PlanFilesTouchedResult,
        PostMergePyrightResult,
        _check_plan_files_touched_in_branch,
        _check_plan_targets_in_tree,
        _check_post_merge_equivalence,
        _check_post_merge_pyright,
        _commit_is_linear,
        _finalize_advanced_merge,
        _GenerationChainContext,
        _map_advance_failure,
        _normalize_plan_path,
        _rebase_delta_touched_overlap,
        _resolve_second_parent,
        _reverify_rebased_tree,
    )

    for name, obj in {
        'DROPPED_PLAN_TARGETS_REASON_PREFIX': DROPPED_PLAN_TARGETS_REASON_PREFIX,
        'PLAN_FILES_NOT_TOUCHED_REASON_PREFIX': PLAN_FILES_NOT_TOUCHED_REASON_PREFIX,
        'POST_MERGE_EQUIVALENCE_FAILED_REASON_PREFIX': POST_MERGE_EQUIVALENCE_FAILED_REASON_PREFIX,
        'POST_MERGE_PYRIGHT_BROKEN_REASON_PREFIX': POST_MERGE_PYRIGHT_BROKEN_REASON_PREFIX,
        'DropGuardResult': DropGuardResult,
        'PlanFilesTouchedResult': PlanFilesTouchedResult,
        'PostMergePyrightResult': PostMergePyrightResult,
        '_GenerationChainContext': _GenerationChainContext,
        '_OVERLAP_GIT_ERROR_SENTINEL': _OVERLAP_GIT_ERROR_SENTINEL,
        '_check_plan_targets_in_tree': _check_plan_targets_in_tree,
        '_normalize_plan_path': _normalize_plan_path,
        '_check_plan_files_touched_in_branch': _check_plan_files_touched_in_branch,
        '_check_post_merge_equivalence': _check_post_merge_equivalence,
        '_rebase_delta_touched_overlap': _rebase_delta_touched_overlap,
        '_reverify_rebased_tree': _reverify_rebased_tree,
        '_check_post_merge_pyright': _check_post_merge_pyright,
        '_resolve_second_parent': _resolve_second_parent,
        '_commit_is_linear': _commit_is_linear,
        '_finalize_advanced_merge': _finalize_advanced_merge,
        '_map_advance_failure': _map_advance_failure,
    }.items():
        assert obj is not None, f'{name} must not be None'


def test_merge_gates_logger_name_is_merge_queue() -> None:
    """merge_gates emits under the 'orchestrator.merge_queue' logger name.

    RED (pre-module): ``orchestrator.merge_gates`` does not exist yet.

    Required so existing ``caplog.at_level(..., logger='orchestrator.merge_queue')``
    assertions in test_merge_queue_equivalence.py / test_merge_queue.py keep
    capturing the moved gates' WARNING-level fail-open/fail-closed messages
    after the functions relocate to this module.
    """
    import orchestrator.merge_gates as merge_gates

    assert merge_gates.logger.name == 'orchestrator.merge_queue'


@pytest.mark.asyncio
class TestReachBackRouting:
    """Reach-back / string-path monkeypatch routing contract.

    Each test patches the SAME logical dependency in both namespaces with
    CONTRASTING values: the merge_gates-local (naive bare-global) patch
    steers the outcome one way, the merge_queue (reach-back target) patch
    the other.  Asserting on the merge_queue-steered outcome proves the
    call went through the deferred import rather than the co-located
    merge_gates sibling.
    """

    async def test_reverify_rebased_tree_reachback_to_rebase_delta_overlap(self) -> None:
        """(a) _reverify_rebased_tree must resolve _rebase_delta_touched_overlap
        via orchestrator.merge_queue, not the co-located merge_gates copy."""
        from orchestrator.merge_gates import _reverify_rebased_tree

        git_ops = MagicMock()
        req = MagicMock()
        req.task_id = 'task-rvrt-reachback'
        req.worktree = MagicMock()
        merge_wt = MagicMock()
        sentinel_outcome = MagicMock(name='sentinel-verify-outcome')

        with (
            # Naive-resolution target: disjoint (empty) → would return None
            # WITHOUT ever calling _run_post_merge_verify.
            patch(
                'orchestrator.merge_gates._rebase_delta_touched_overlap',
                AsyncMock(return_value=[]),
            ),
            # Reach-back target: overlapping → must delegate to
            # _run_post_merge_verify (itself already reach-back, per step-2).
            patch(
                'orchestrator.merge_queue._rebase_delta_touched_overlap',
                AsyncMock(return_value=['overlap.py']),
            ),
            patch(
                'orchestrator.merge_queue._run_post_merge_verify',
                AsyncMock(return_value=sentinel_outcome),
            ),
        ):
            result = await _reverify_rebased_tree(
                git_ops, req, merge_wt,
                rebased_from='from-sha',
                rebased_onto='onto-sha',
                timeouts={},
                enospc_retries={},
                max_timeouts=3,
                max_enospc=1,
            )

        assert result is sentinel_outcome, (
            f'expected the orchestrator.merge_queue-patched overlap to govern '
            f'the re-verify decision and return its sentinel outcome, got {result!r}'
        )

    async def test_finalize_advanced_merge_reachback_to_equivalence_and_pyright(self) -> None:
        """(b) _finalize_advanced_merge must resolve _check_post_merge_equivalence
        and _check_post_merge_pyright via orchestrator.merge_queue, not the
        co-located merge_gates copies."""
        from orchestrator.merge_gates import _finalize_advanced_merge

        git_ops = MagicMock()
        git_ops.push_main = AsyncMock(return_value='pushed')
        git_ops.cleanup_merge_worktree = AsyncMock()
        # NOTE (task 1997): the post-rebase SHA is threaded via the explicit
        # advanced_sha= kwarg below, NOT the git_ops._last_advanced_sha side
        # channel — deliberately left unset here.
        req = MagicMock()
        req.task_id = 'task-finalize-reachback'
        req.branch = 'br-finalize-reachback'
        req.worktree = MagicMock()
        req.config = MagicMock()
        req.module_configs = []
        cas_retries = {req.task_id: 1}
        timeouts = {req.task_id: 1}
        enospc_retries = {req.task_id: 1}

        naive_broken_pyright = MagicMock(broken=True, failing_subprojects=['naive-pkg'], detail='naive-detail')
        reachback_clean_pyright = MagicMock(broken=False, failing_subprojects=[], detail='')

        with (
            # Naive-resolution targets: equivalence diverged + pyright broken →
            # would return 'blocked' before ever reaching push_main.
            patch(
                'orchestrator.merge_gates._check_post_merge_equivalence',
                AsyncMock(return_value=['naive-diverged.py']),
            ),
            patch(
                'orchestrator.merge_gates._check_post_merge_pyright',
                AsyncMock(return_value=naive_broken_pyright),
            ),
            # Reach-back targets: equivalence clean + pyright clean → must
            # reach the 'done' path and call push_main.
            patch(
                'orchestrator.merge_queue._check_post_merge_equivalence',
                AsyncMock(return_value=[]),
            ),
            patch(
                'orchestrator.merge_queue._check_post_merge_pyright',
                AsyncMock(return_value=reachback_clean_pyright),
            ),
        ):
            outcome = await _finalize_advanced_merge(
                git_ops, req, None,
                merge_commit_fallback='fallback-sha',
                base_sha='base-sha',
                started_monotonic=0.0,
                cas_retries=cas_retries,
                timeouts=timeouts,
                enospc_retries=enospc_retries,
                merged_branch_tip='trusted-tip',
                advanced_sha='abc123def',
            )

        assert outcome.status == 'done', (
            f'expected the orchestrator.merge_queue-patched equivalence/pyright '
            f'results to govern the outcome (done), got {outcome.status}: {outcome.reason!r}'
        )
        git_ops.push_main.assert_awaited_once()

    async def test_check_post_merge_pyright_reachback_to_run_unscoped_typechecks(self) -> None:
        """(c) _check_post_merge_pyright must resolve _run_unscoped_typechecks
        via orchestrator.merge_queue (it has no merge_gates-local copy at all —
        this reach-back was added directly in step-2, not deferred)."""
        from orchestrator.config import ModuleConfig, OrchestratorConfig
        from orchestrator.merge_gates import PostMergePyrightResult, _check_post_merge_pyright

        git_ops = MagicMock()
        git_ops._create_merge_worktree = AsyncMock(return_value=('fake-merge-wt', None))
        git_ops.cleanup_merge_worktree = AsyncMock()
        module_configs = [ModuleConfig(prefix='pkg', type_check_command='pyright src/')]
        patched_result = PostMergePyrightResult(
            failing_subprojects=['pkg'], detail='patched-detail',
        )

        with patch(
            'orchestrator.merge_queue._run_unscoped_typechecks',
            AsyncMock(return_value=patched_result),
        ):
            result = await _check_post_merge_pyright(
                'deadbeef', git_ops, OrchestratorConfig(), module_configs,
                task_id='task-pyright-reachback',
            )

        assert result is patched_result, (
            f'expected the orchestrator.merge_queue-patched _run_unscoped_typechecks '
            f'result to be returned unchanged, got {result!r}'
        )
        git_ops.cleanup_merge_worktree.assert_awaited_once()


def test_merge_queue_reexports_identical_objects() -> None:
    """merge_queue re-exports the SAME objects from merge_gates (shim identity).

    Covers every one of the 20 moved names.

    RED (pre-shim): merge_queue.py still defines its own independent copies
    of these names (the duplicate definitions left in place by the EXPAND
    step), so ``getattr(merge_queue, name) is getattr(merge_gates, name)``
    fails for every name — two distinct objects that merely share a name.
    """
    import orchestrator.merge_gates as merge_gates
    import orchestrator.merge_queue as merge_queue

    moved_names = [
        'DROPPED_PLAN_TARGETS_REASON_PREFIX',
        'PLAN_FILES_NOT_TOUCHED_REASON_PREFIX',
        'POST_MERGE_EQUIVALENCE_FAILED_REASON_PREFIX',
        'POST_MERGE_PYRIGHT_BROKEN_REASON_PREFIX',
        'DropGuardResult',
        'PlanFilesTouchedResult',
        'PostMergePyrightResult',
        '_GenerationChainContext',
        '_OVERLAP_GIT_ERROR_SENTINEL',
        '_check_plan_targets_in_tree',
        '_normalize_plan_path',
        '_check_plan_files_touched_in_branch',
        '_check_post_merge_equivalence',
        '_rebase_delta_touched_overlap',
        '_reverify_rebased_tree',
        '_check_post_merge_pyright',
        '_resolve_second_parent',
        '_commit_is_linear',
        '_finalize_advanced_merge',
        '_map_advance_failure',
    ]

    for name in moved_names:
        mq_obj = getattr(merge_queue, name)
        mg_obj = getattr(merge_gates, name)
        assert mq_obj is mg_obj, (
            f'{name}: orchestrator.merge_queue.{name} and '
            f'orchestrator.merge_gates.{name} must be the identical object'
        )


# --- Declarative post-advance gate chain (MQ-refactor task λ) --------------
#
# GateVerdict / Gate / _PostAdvanceContext / POST_ADVANCE_GATES /
# _run_equivalence_gate / _run_pyright_gate do not exist yet in
# orchestrator.merge_gates — every test below is RED via ImportError until
# step-2 lands the building blocks.


def test_gate_verdict_value_type() -> None:
    """GateVerdict is a 2-state frozen value object: ok() | block(...)."""
    import dataclasses

    from orchestrator.merge_gates import GateVerdict
    from orchestrator.merge_types import OutcomeKind

    ok = GateVerdict.ok()
    assert ok.passed is True
    assert ok.reason is None
    assert ok.merge_sha is None
    assert ok.emit_subtype is None

    blocked = GateVerdict.block(
        reason='r', merge_sha='s', emit_subtype=OutcomeKind.post_merge_pyright_broken,
    )
    assert blocked.passed is False
    assert blocked.reason == 'r'
    assert blocked.merge_sha == 's'
    assert blocked.emit_subtype == OutcomeKind.post_merge_pyright_broken

    # Frozen: dataclasses.replace works (produces a new instance)...
    replaced = dataclasses.replace(ok, reason='changed')
    assert replaced.reason == 'changed'
    assert ok.reason is None
    # ...but direct attribute assignment raises.
    with pytest.raises(dataclasses.FrozenInstanceError):
        ok.reason = 'mutated'  # type: ignore[misc]


def test_gate_and_context_construct() -> None:
    """Gate defaults on_blocked=None; _PostAdvanceContext bundles the
    documented fields verbatim (single typed argument for gate callables)."""
    from orchestrator.merge_gates import Gate, GateVerdict, _PostAdvanceContext

    async def _spy_run(ctx: object) -> GateVerdict:
        return GateVerdict.ok()

    gate = Gate(name='x', run=_spy_run)
    assert gate.name == 'x'
    assert gate.run is _spy_run
    assert gate.on_blocked is None

    git_ops = MagicMock()
    req = MagicMock()
    ctx = _PostAdvanceContext(
        git_ops=git_ops,
        req=req,
        event_store=None,
        advanced_sha='adv-sha',
        base_sha='base-sha',
        resolved_merged_tip='tip-sha',
        allow_worktree_head_fallback=True,
        started_monotonic=0.0,
        train_id=None,
        member_task_ids=None,
        chain_ctx=None,
        merged_branch_tip=None,
        log_label='',
    )
    assert ctx.git_ops is git_ops
    assert ctx.req is req
    assert ctx.event_store is None
    assert ctx.advanced_sha == 'adv-sha'
    assert ctx.base_sha == 'base-sha'
    assert ctx.resolved_merged_tip == 'tip-sha'
    assert ctx.allow_worktree_head_fallback is True
    assert ctx.started_monotonic == 0.0
    assert ctx.train_id is None
    assert ctx.member_task_ids is None
    assert ctx.chain_ctx is None
    assert ctx.merged_branch_tip is None
    assert ctx.log_label == ''


def test_post_advance_gates_registry_shape() -> None:
    """POST_ADVANCE_GATES is [equivalence, pyright], in order; only the
    equivalence gate carries the γ2 auto-chain on_blocked hook; the shim
    re-exports the identical list object (not a copy)."""
    import orchestrator.merge_gates as merge_gates
    import orchestrator.merge_queue as merge_queue
    from orchestrator.merge_gates import POST_ADVANCE_GATES, Gate

    assert isinstance(POST_ADVANCE_GATES, list)
    assert all(isinstance(g, Gate) for g in POST_ADVANCE_GATES)
    assert [g.name for g in POST_ADVANCE_GATES] == ['equivalence', 'pyright']

    equivalence_gate, pyright_gate = POST_ADVANCE_GATES
    assert callable(equivalence_gate.run)
    assert callable(pyright_gate.run)
    assert equivalence_gate.on_blocked is not None
    assert callable(equivalence_gate.on_blocked)
    assert pyright_gate.on_blocked is None

    assert merge_queue.POST_ADVANCE_GATES is merge_gates.POST_ADVANCE_GATES


@pytest.mark.asyncio
class TestGateFunctionsReachBack:
    """_run_equivalence_gate / _run_pyright_gate unit + reach-back contract.

    Mirrors ``TestReachBackRouting`` above: each block-path test patches the
    SAME dependency in both namespaces with CONTRASTING values so the
    assertion is unambiguous about which one governed the verdict.
    """

    def _make_ctx(self, **overrides: object):
        from orchestrator.merge_gates import _PostAdvanceContext

        req = MagicMock()
        req.task_id = 'task-gate-unit'
        req.worktree = MagicMock()
        req.config = MagicMock()
        req.module_configs = []
        defaults: dict = dict(
            git_ops=MagicMock(),
            req=req,
            event_store=None,
            advanced_sha='advanced-sha-123',
            base_sha='base-sha-456',
            resolved_merged_tip='resolved-tip',
            allow_worktree_head_fallback=True,
            started_monotonic=0.0,
            train_id=None,
            member_task_ids=None,
            chain_ctx=None,
            merged_branch_tip=None,
            log_label='',
        )
        defaults.update(overrides)
        return _PostAdvanceContext(**defaults)

    async def test_run_equivalence_gate_ok_when_clean(self) -> None:
        from orchestrator.merge_gates import _run_equivalence_gate

        ctx = self._make_ctx()
        with patch(
            'orchestrator.merge_queue._check_post_merge_equivalence',
            AsyncMock(return_value=[]),
        ):
            verdict = await _run_equivalence_gate(ctx)

        assert verdict.passed is True

    async def test_run_equivalence_gate_reachback_governs_block(self) -> None:
        from orchestrator.merge_gates import (
            POST_MERGE_EQUIVALENCE_FAILED_REASON_PREFIX,
            _run_equivalence_gate,
        )

        ctx = self._make_ctx()
        with (
            # Naive-resolution target: clean → would pass if this governed.
            patch(
                'orchestrator.merge_gates._check_post_merge_equivalence',
                AsyncMock(return_value=[]),
            ),
            # Reach-back target: diverged → must govern the verdict.
            patch(
                'orchestrator.merge_queue._check_post_merge_equivalence',
                AsyncMock(return_value=['x.py']),
            ),
        ):
            verdict = await _run_equivalence_gate(ctx)

        assert verdict.passed is False
        assert verdict.merge_sha == ctx.advanced_sha
        assert verdict.emit_subtype == 'post_merge_equivalence_failed'
        assert verdict.reason is not None
        assert verdict.reason.startswith(POST_MERGE_EQUIVALENCE_FAILED_REASON_PREFIX)

    async def test_run_pyright_gate_ok_when_clean(self) -> None:
        from orchestrator.merge_gates import _run_pyright_gate

        ctx = self._make_ctx()
        clean = MagicMock(broken=False, failing_subprojects=[], detail='')
        with patch(
            'orchestrator.merge_queue._check_post_merge_pyright',
            AsyncMock(return_value=clean),
        ):
            verdict = await _run_pyright_gate(ctx)

        assert verdict.passed is True

    async def test_run_pyright_gate_reachback_governs_block(self) -> None:
        from orchestrator.merge_gates import (
            POST_MERGE_PYRIGHT_BROKEN_REASON_PREFIX,
            _run_pyright_gate,
        )

        ctx = self._make_ctx()
        naive_clean = MagicMock(broken=False, failing_subprojects=[], detail='')
        reachback_broken = MagicMock(broken=True, failing_subprojects=['pkg'], detail='boom')
        with (
            # Naive-resolution target: clean → would pass if this governed.
            patch(
                'orchestrator.merge_gates._check_post_merge_pyright',
                AsyncMock(return_value=naive_clean),
            ),
            # Reach-back target: broken → must govern the verdict.
            patch(
                'orchestrator.merge_queue._check_post_merge_pyright',
                AsyncMock(return_value=reachback_broken),
            ),
        ):
            verdict = await _run_pyright_gate(ctx)

        assert verdict.passed is False
        assert verdict.merge_sha == ctx.advanced_sha
        assert verdict.emit_subtype == 'post_merge_pyright_broken'
        assert verdict.reason is not None
        assert verdict.reason.startswith(POST_MERGE_PYRIGHT_BROKEN_REASON_PREFIX)


@pytest.mark.asyncio
class TestFinalizeDrivesRegistry:
    """_finalize_advanced_merge iterates POST_ADVANCE_GATES (task λ).

    RED against the still-inline body: ``test_finalize_runs_registered_gate``
    and ``test_finalize_logs_gate_names_per_landing`` fail today because the
    inline _finalize_advanced_merge never reads POST_ADVANCE_GATES at all.
    The remaining two tests pin the existing equivalence-block / γ2-chain
    behavior so it survives the coming registry-driven rewrite unchanged.
    """

    def _make_finalize_args(self, **overrides: object) -> dict:
        git_ops = MagicMock()
        git_ops.push_main = AsyncMock(return_value='pushed')
        git_ops.cleanup_merge_worktree = AsyncMock()
        # NOTE (task 1997): the post-rebase SHA is threaded via the explicit
        # advanced_sha= kwarg below, NOT the git_ops._last_advanced_sha side
        # channel — deliberately left unset here so these tests pin the
        # post-migration contract.
        req = MagicMock()
        req.task_id = 'task-finalize-registry'
        req.branch = 'br-finalize-registry'
        req.worktree = MagicMock()
        req.config = MagicMock()
        req.module_configs = []
        defaults: dict = dict(
            git_ops=git_ops,
            req=req,
            event_store=None,
            merge_commit_fallback='fallback-sha',
            base_sha='base-sha',
            started_monotonic=0.0,
            cas_retries={},
            timeouts={},
            enospc_retries={},
            advanced_sha='finalize-registry-sha',
        )
        defaults.update(overrides)
        return defaults

    async def test_finalize_runs_registered_gate(self, monkeypatch) -> None:
        """Headline signal: a gate appended to POST_ADVANCE_GATES runs during
        _finalize_advanced_merge WITHOUT any edit to _finalize's body."""
        import orchestrator.merge_gates as merge_gates
        from orchestrator.merge_gates import Gate, GateVerdict, _finalize_advanced_merge

        spy_run = AsyncMock(return_value=GateVerdict.ok())
        monkeypatch.setattr(
            merge_gates, 'POST_ADVANCE_GATES',
            [*merge_gates.POST_ADVANCE_GATES, Gate('noop-probe', spy_run)],
        )

        clean_pyright = MagicMock(broken=False, failing_subprojects=[], detail='')
        args = self._make_finalize_args()
        with (
            patch(
                'orchestrator.merge_queue._check_post_merge_equivalence',
                AsyncMock(return_value=[]),
            ),
            patch(
                'orchestrator.merge_queue._check_post_merge_pyright',
                AsyncMock(return_value=clean_pyright),
            ),
        ):
            outcome = await _finalize_advanced_merge(**args)

        spy_run.assert_awaited_once()
        assert outcome.status == 'done'
        args['git_ops'].push_main.assert_awaited_once()

    async def test_finalize_logs_gate_names_per_landing(self, caplog) -> None:
        from orchestrator.merge_gates import _finalize_advanced_merge

        clean_pyright = MagicMock(broken=False, failing_subprojects=[], detail='')
        args = self._make_finalize_args()
        with (
            patch(
                'orchestrator.merge_queue._check_post_merge_equivalence',
                AsyncMock(return_value=[]),
            ),
            patch(
                'orchestrator.merge_queue._check_post_merge_pyright',
                AsyncMock(return_value=clean_pyright),
            ),
            caplog.at_level(logging.INFO, logger='orchestrator.merge_queue'),
        ):
            await _finalize_advanced_merge(**args)

        matching = [
            r for r in caplog.records
            if 'post-advance gates run:' in r.getMessage()
        ]
        assert len(matching) == 1, (
            f'expected exactly one gate-names INFO line, got {len(matching)}: '
            f'{[r.getMessage() for r in caplog.records]}'
        )
        assert 'equivalence' in matching[0].getMessage()
        assert 'pyright' in matching[0].getMessage()

    async def test_finalize_equivalence_block_via_registry(self, caplog) -> None:
        from orchestrator.merge_gates import (
            POST_MERGE_EQUIVALENCE_FAILED_REASON_PREFIX,
            _finalize_advanced_merge,
        )

        args = self._make_finalize_args(chain_ctx=None)
        with (
            patch(
                'orchestrator.merge_queue._check_post_merge_equivalence',
                AsyncMock(return_value=['f.py']),
            ),
            caplog.at_level(logging.INFO, logger='orchestrator.merge_queue'),
        ):
            outcome = await _finalize_advanced_merge(**args)

        assert outcome.status == 'blocked'
        assert outcome.reason.startswith(POST_MERGE_EQUIVALENCE_FAILED_REASON_PREFIX)
        assert outcome.merge_sha == args['advanced_sha']
        args['git_ops'].push_main.assert_not_awaited()

        gate_lines = [
            r.getMessage() for r in caplog.records
            if 'post-advance gates run:' in r.getMessage()
        ]
        assert len(gate_lines) == 1, (
            f'expected exactly one gate-names INFO line, got: {gate_lines}'
        )
        assert 'equivalence' in gate_lines[0]
        assert 'pyright' not in gate_lines[0]

    async def test_finalize_pyright_block_via_registry(self, caplog) -> None:
        """Second-gate-in-chain path: equivalence passes, pyright blocks.

        This is the case the registry refactor changes the most — a
        terminal built from a gate WITHOUT an on_blocked hook, reached only
        after a prior gate in the chain has already run and passed.
        """
        from orchestrator.merge_gates import (
            POST_MERGE_PYRIGHT_BROKEN_REASON_PREFIX,
            _finalize_advanced_merge,
        )

        broken_pyright = MagicMock(
            broken=True, failing_subprojects=['pkg'], detail='pyright-detail',
        )
        args = self._make_finalize_args(chain_ctx=None)
        with (
            patch(
                'orchestrator.merge_queue._check_post_merge_equivalence',
                AsyncMock(return_value=[]),
            ),
            patch(
                'orchestrator.merge_queue._check_post_merge_pyright',
                AsyncMock(return_value=broken_pyright),
            ),
            caplog.at_level(logging.INFO, logger='orchestrator.merge_queue'),
        ):
            outcome = await _finalize_advanced_merge(**args)

        assert outcome.status == 'blocked'
        assert outcome.reason.startswith(POST_MERGE_PYRIGHT_BROKEN_REASON_PREFIX)
        assert outcome.merge_sha == args['advanced_sha']
        args['git_ops'].push_main.assert_not_awaited()

        gate_lines = [
            r.getMessage() for r in caplog.records
            if 'post-advance gates run:' in r.getMessage()
        ]
        assert len(gate_lines) == 1, (
            f'expected exactly one gate-names INFO line, got: {gate_lines}'
        )
        assert 'equivalence' in gate_lines[0]
        assert 'pyright' in gate_lines[0]

    async def test_finalize_gamma2_chain_via_registry(self) -> None:
        from orchestrator.merge_gates import _finalize_advanced_merge, _GenerationChainContext
        from orchestrator.merge_types import MergeOutcome

        chain_ctx = _GenerationChainContext(
            queue=MagicMock(), counts={}, max_auto_generations=3,
        )
        chained_outcome = MergeOutcome('superseded', merge_sha='chained-sha')
        event_store = MagicMock()
        args = self._make_finalize_args(
            chain_ctx=chain_ctx, merged_branch_tip='trusted-tip', event_store=event_store,
        )

        with (
            patch(
                'orchestrator.merge_queue._check_post_merge_equivalence',
                AsyncMock(return_value=['f.py']),
            ),
            patch('orchestrator.merge_queue.AUTO_CHAIN_GENERATIONS_ENABLED', True),
            patch(
                'orchestrator.merge_queue._maybe_auto_chain_generation',
                AsyncMock(return_value=chained_outcome),
            ) as maybe_chain_mock,
        ):
            outcome = await _finalize_advanced_merge(**args)

        assert outcome is chained_outcome
        maybe_chain_mock.assert_awaited_once()

        emitted = [call.kwargs['data']['outcome'] for call in event_store.emit.call_args_list]
        assert emitted == ['post_merge_equivalence_failed', 'post_merge_generation_chained'], (
            f'expected the equivalence-failed emit before the chained emit, got: {emitted}'
        )

    async def test_finalize_merge_sha_falls_back_to_merge_commit_fallback_when_advanced_sha_none(
        self,
    ) -> None:
        """advanced_sha=None (e.g. no rebase occurred) falls back to
        merge_commit_fallback for outcome.merge_sha — the other half of the
        advanced_sha contract pinned above (populated case)."""
        from orchestrator.merge_gates import _finalize_advanced_merge

        clean_pyright = MagicMock(broken=False, failing_subprojects=[], detail='')
        args = self._make_finalize_args(advanced_sha=None)
        with (
            patch(
                'orchestrator.merge_queue._check_post_merge_equivalence',
                AsyncMock(return_value=[]),
            ),
            patch(
                'orchestrator.merge_queue._check_post_merge_pyright',
                AsyncMock(return_value=clean_pyright),
            ),
        ):
            outcome = await _finalize_advanced_merge(**args)

        assert outcome.status == 'done'
        assert outcome.merge_sha == args['merge_commit_fallback']


@pytest.mark.asyncio
class TestMapAdvanceFailurePopConflictAdvancedSha:
    """_map_advance_failure's pop_conflict branch threads advanced_sha
    (task 1997 / MQ-refactor μ) instead of reading the
    git_ops._last_advanced_sha getattr side channel.

    RED: _map_advance_failure does not yet accept an advanced_sha kwarg —
    every test below fails with a TypeError until step-4 lands it.
    """

    def _make_git_ops(self) -> MagicMock:
        git_ops = MagicMock()
        git_ops.push_main = AsyncMock(return_value='pushed')
        # Deliberately do NOT set _last_advanced_sha — the mapper must
        # source the SHA from the advanced_sha kwarg, not the side channel.
        return git_ops

    async def test_pop_conflict_merge_sha_uses_advanced_sha_kwarg(self) -> None:
        from orchestrator.merge_gates import _map_advance_failure

        git_ops = self._make_git_ops()

        outcome = await _map_advance_failure(
            git_ops, 'pop_conflict',
            task_id='task-map-adv-sha',
            merge_commit_fallback='fallback-sha',
            halt=MagicMock(),
            unhalt=MagicMock(),
            cas_retries={},
            advanced_sha='post-rebase-sha',
        )

        assert outcome.status == 'done_wip_recovery'
        assert outcome.merge_sha == 'post-rebase-sha'

    async def test_pop_conflict_merge_sha_falls_back_to_merge_commit_fallback(self) -> None:
        from orchestrator.merge_gates import _map_advance_failure

        git_ops = self._make_git_ops()

        outcome = await _map_advance_failure(
            git_ops, 'pop_conflict',
            task_id='task-map-adv-fallback',
            merge_commit_fallback='fallback-sha',
            halt=MagicMock(),
            unhalt=MagicMock(),
            cas_retries={},
            advanced_sha=None,
        )

        assert outcome.status == 'done_wip_recovery'
        assert outcome.merge_sha == 'fallback-sha'


@pytest.mark.asyncio
class TestMapAdvanceFailureConflictMarkers:
    """_map_advance_failure must map the 'conflict_markers' advance_main
    result code (esc-2128-8 Layer-2 pre-merge gate, task 2282) to a
    blocked, human-actionable MergeOutcome with a specific reason — not
    the generic 'advance_main failed (conflict_markers)' catch-all text.
    """

    def _make_git_ops(self) -> MagicMock:
        git_ops = MagicMock()
        git_ops.push_main = AsyncMock(return_value='pushed')
        return git_ops

    async def test_conflict_markers_maps_to_blocked_with_specific_reason(self) -> None:
        from orchestrator.merge_gates import _map_advance_failure

        git_ops = self._make_git_ops()

        outcome = await _map_advance_failure(
            git_ops, 'conflict_markers',
            task_id='task-map-adv-markers',
            merge_commit_fallback='fallback-sha',
            halt=MagicMock(),
            unhalt=MagicMock(),
            cas_retries={},
        )

        assert outcome.status == 'blocked'
        assert outcome.reason is not None and 'conflict marker' in outcome.reason.lower(), (
            f'Expected a specific conflict-marker reason, got {outcome.reason!r}'
        )
        assert outcome.reason != f"advance_main failed (conflict_markers) for task task-map-adv-markers", (
            'Must not fall through to the generic catch-all reason text'
        )

    async def test_conflict_markers_pops_cas_retries(self) -> None:
        from orchestrator.merge_gates import _map_advance_failure

        git_ops = self._make_git_ops()
        cas_retries = {'task-map-adv-markers-2': 2}

        await _map_advance_failure(
            git_ops, 'conflict_markers',
            task_id='task-map-adv-markers-2',
            merge_commit_fallback='fallback-sha',
            halt=MagicMock(),
            unhalt=MagicMock(),
            cas_retries=cas_retries,
        )

        assert 'task-map-adv-markers-2' not in cas_retries

    async def test_conflict_markers_does_not_halt_queue(self) -> None:
        """Unlike unmerged_state/wip_overlap, conflict_markers is a
        per-branch content problem, not a queue-wide condition — no halt."""
        from orchestrator.merge_gates import _map_advance_failure

        git_ops = self._make_git_ops()
        halt = MagicMock()

        await _map_advance_failure(
            git_ops, 'conflict_markers',
            task_id='task-map-adv-markers-3',
            merge_commit_fallback='fallback-sha',
            halt=halt,
            unhalt=MagicMock(),
            cas_retries={},
        )

        halt.assert_not_called()
