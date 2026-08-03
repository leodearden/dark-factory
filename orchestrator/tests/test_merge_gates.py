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
        assert outcome.reason != 'advance_main failed (conflict_markers) for task task-map-adv-markers', (
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


@pytest.mark.asyncio
class TestMapAdvanceFailureStashFailed:
    """_map_advance_failure must route the 'stash_failed' advance_main result
    code to the halt-plus-single-escalation path (parallel to unmerged_state),
    NOT the per-task 'blocked' catch-all.

    ``stash_failed`` is a SHARED main-checkout-hygiene fault: parking
    project_root's dirty tracked tree fails identically for every subsequent
    task, so the old catch-all produced a serial fleet-wide pileup of N silent
    per-task blocks (the 2026-07-12 reify incident, verified merge 57fe8667
    never landed). Halting the queue collapses that to ONE loud signal owned
    by a single escalation.

    RED: today ``stash_failed`` falls through to the
    ``not_descendant``/``contaminated`` catch-all → status 'blocked', halt
    NOT called, and MergeOutcome has no ``dirty_files`` field.
    """

    def _make_git_ops(self) -> MagicMock:
        git_ops = MagicMock()
        git_ops.push_main = AsyncMock(return_value='pushed')
        # Default: the _last_stash_dirty_files side channel is UNSET, so
        # getattr(..., None) in the mapper falls back to []. Deleting the
        # auto-vivified MagicMock attribute makes getattr return its default
        # (a bare MagicMock would auto-vivify a truthy child, defeating `or []`).
        del git_ops._last_stash_dirty_files
        return git_ops

    async def test_stash_failed_halts_queue(self) -> None:
        """(a) stash_failed halts the whole queue (shared fault), with a reason
        naming the failed park/stash of project_root WIP."""
        from orchestrator.merge_gates import _map_advance_failure

        git_ops = self._make_git_ops()
        halt = MagicMock()

        await _map_advance_failure(
            git_ops, 'stash_failed',
            task_id='task-stashf-a',
            merge_commit_fallback='fallback-sha',
            halt=halt,
            unhalt=MagicMock(),
            cas_retries={},
        )

        halt.assert_called_once()
        halt_reason = halt.call_args.args[0]
        assert 'stash_failed' in halt_reason
        assert 'park' in halt_reason.lower(), (
            f'halt reason should name the failed park, got {halt_reason!r}'
        )

    async def test_stash_failed_status_is_stash_failed_not_blocked(self) -> None:
        """(b) outcome.status is the distinct 'stash_failed', NOT the generic
        catch-all 'blocked'."""
        from orchestrator.merge_gates import _map_advance_failure

        git_ops = self._make_git_ops()

        outcome = await _map_advance_failure(
            git_ops, 'stash_failed',
            task_id='task-stashf-b',
            merge_commit_fallback='fallback-sha',
            halt=MagicMock(),
            unhalt=MagicMock(),
            cas_retries={},
        )

        assert outcome.status == 'stash_failed'

    async def test_stash_failed_surfaces_dirty_files_from_side_channel(self) -> None:
        """(c) dirty_files is populated from the git_ops._last_stash_dirty_files
        side channel and the reason names the dirty tracked paths + task_id."""
        from orchestrator.merge_gates import _map_advance_failure

        git_ops = self._make_git_ops()
        git_ops._last_stash_dirty_files = ['a.py', 'b.py']

        outcome = await _map_advance_failure(
            git_ops, 'stash_failed',
            task_id='task-stashf-c',
            merge_commit_fallback='fallback-sha',
            halt=MagicMock(),
            unhalt=MagicMock(),
            cas_retries={},
        )

        assert outcome.dirty_files == ['a.py', 'b.py']
        assert 'a.py' in outcome.reason and 'b.py' in outcome.reason, (
            f'reason should name the dirty tracked paths, got {outcome.reason!r}'
        )
        assert 'task-stashf-c' in outcome.reason

    async def test_stash_failed_without_side_channel_defaults_empty_and_halts(self) -> None:
        """(d) with NO side-channel attr set, dirty_files defaults to [] and the
        branch still halts (no crash on the absent getattr)."""
        from orchestrator.merge_gates import _map_advance_failure

        git_ops = self._make_git_ops()  # _last_stash_dirty_files deleted
        halt = MagicMock()

        outcome = await _map_advance_failure(
            git_ops, 'stash_failed',
            task_id='task-stashf-d',
            merge_commit_fallback='fallback-sha',
            halt=halt,
            unhalt=MagicMock(),
            cas_retries={},
        )

        assert outcome.status == 'stash_failed'
        assert outcome.dirty_files == []
        halt.assert_called_once()

    async def test_stash_failed_pops_cas_retries(self) -> None:
        """(e) terminal-for-this-task: the cas_retries entry is popped."""
        from orchestrator.merge_gates import _map_advance_failure

        git_ops = self._make_git_ops()
        cas_retries = {'task-stashf-e': 2}

        await _map_advance_failure(
            git_ops, 'stash_failed',
            task_id='task-stashf-e',
            merge_commit_fallback='fallback-sha',
            halt=MagicMock(),
            unhalt=MagicMock(),
            cas_retries=cas_retries,
        )

        assert 'task-stashf-e' not in cas_retries


@pytest.mark.asyncio
class TestMapAdvanceFailurePerBranchStillBlocks:
    """Regression lock: ``not_descendant`` / ``contaminated`` remain per-task
    'blocked' with NO queue halt. They are per-branch content problems (this
    branch isn't a descendant of main / carries contamination) that do NOT
    recur for other tasks, so halting the whole queue for them would wrongly
    block unrelated healthy work. Only ``stash_failed`` (a shared main-checkout
    fault) was promoted to the halt path — this pins that scoping decision."""

    def _make_git_ops(self) -> MagicMock:
        git_ops = MagicMock()
        git_ops.push_main = AsyncMock(return_value='pushed')
        return git_ops

    @pytest.mark.parametrize('result', ['not_descendant', 'contaminated'])
    async def test_per_branch_failure_blocks_without_halt(self, result: str) -> None:
        from orchestrator.merge_gates import _map_advance_failure

        git_ops = self._make_git_ops()
        halt = MagicMock()
        task_id = f'task-{result}'
        cas_retries = {task_id: 1}

        outcome = await _map_advance_failure(
            git_ops, result,
            task_id=task_id,
            merge_commit_fallback='fallback-sha',
            halt=halt,
            unhalt=MagicMock(),
            cas_retries=cas_retries,
        )

        assert outcome.status == 'blocked'
        assert outcome.reason == f'advance_main failed ({result}) for task {task_id}'
        halt.assert_not_called()
        assert task_id not in cas_retries


class TestIsCrossRepoTask:
    """Unit tests for the pure ``is_cross_repo_task`` cross-repo classifier.

    The helper lets the pre-merge Decision-1 gate recognize a task whose
    declared plan files ALL belong to a *different* project (the reify-task
    5308 shape) so the workflow can route it to the honest
    ``plan_files_cross_repo`` terminal outcome instead of false-flagging the
    (legitimately empty) branch as "plan files not touched".
    """

    def test_marker_true_with_any_files(self, tmp_path):
        # (a) explicit metadata.cross_repo marker → True regardless of paths.
        from orchestrator.merge_gates import is_cross_repo_task

        assert is_cross_repo_task(
            ['orchestrator/src/orchestrator/offline_lane.py'],
            tmp_path / 'reify',
            {'cross_repo': True},
        ) is True

    def test_all_absolute_foreign_no_marker(self, tmp_path):
        # (b) every entry is an absolute path resolving OUTSIDE project_root → True.
        from orchestrator.merge_gates import is_cross_repo_task

        project_root = tmp_path / 'reify'
        files = [
            '/home/leo/src/dark-factory/orchestrator/src/orchestrator/offline_lane.py',
            '/home/leo/src/dark-factory/orchestrator/tests/test_offline_lane.py',
        ]
        assert is_cross_repo_task(files, project_root, None) is True

    def test_mixed_absolute_foreign_and_relative_local(self, tmp_path):
        # (c) one absolute-foreign + one relative entry, no marker → False.
        from orchestrator.merge_gates import is_cross_repo_task

        files = [
            '/home/leo/src/dark-factory/orchestrator/src/orchestrator/offline_lane.py',
            'reify/src/reify/local.py',
        ]
        assert is_cross_repo_task(files, tmp_path / 'reify', None) is False

    def test_all_relative_no_marker(self, tmp_path):
        # (d) all relative entries, no marker → False (orchestrator can't
        # classify a relative foreign path without the fused-memory registry).
        from orchestrator.merge_gates import is_cross_repo_task

        files = [
            'orchestrator/src/orchestrator/offline_lane.py',
            'orchestrator/tests/test_offline_lane.py',
        ]
        assert is_cross_repo_task(files, tmp_path / 'reify', None) is False

    def test_empty_plan_files(self, tmp_path):
        # (e) empty plan_files → False (empty check precedes the marker).
        from orchestrator.merge_gates import is_cross_repo_task

        assert is_cross_repo_task([], tmp_path / 'reify', None) is False
        assert is_cross_repo_task([], tmp_path / 'reify', {'cross_repo': True}) is False

    def test_absolute_path_inside_project_root(self, tmp_path):
        # (f) no marker and an absolute file INSIDE project_root → False.
        from orchestrator.merge_gates import is_cross_repo_task

        project_root = tmp_path / 'proj'
        project_root.mkdir()
        inside = project_root / 'src' / 'mod.py'
        assert is_cross_repo_task([str(inside)], project_root, None) is False
        # A falsy marker is treated as absent.
        assert is_cross_repo_task([str(inside)], project_root, {'cross_repo': False}) is False


class TestParkLockContendedIsNotAHaltResult:
    """The structural contract that makes task 3060's fix work.

    Because `park_lock_contended` is absent from `_HALT_ADVANCE_RESULTS`,
    merge_queue's existing plumbing already routes it past the halt path
    untouched — the explicit mapper branch only upgrades the reason text to
    structured facts. The fix is therefore safe-by-default: an unhandled new
    code already means "no halt".

    NOTE: test_merge_queue.py::TestHaltAdvanceResults already asserts EXACT
    frozenset equality against a literal 5-element set, so that pin passes
    unchanged and that file needs no edit. This assertion is the explicit,
    NAMED contract for this code, not a duplicate of it — it records WHY
    park_lock_contended must stay out.
    """

    def test_park_lock_contended_is_not_a_halt_result(self) -> None:
        from orchestrator.merge_queue import _HALT_ADVANCE_RESULTS

        assert 'park_lock_contended' not in _HALT_ADVANCE_RESULTS, (
            'park_lock_contended must NEVER halt the merge queue — adding it '
            'here reinstates the 2+/day queue halt task 3060 exists to remove'
        )


@pytest.mark.asyncio
class TestMapAdvanceFailureParkLockContended:
    """`park_lock_contended` must be disposed of PER TASK, never as a queue
    halt — the structural contract that makes task 3060's fix work.

    Contrast `stash_failed` (above): that is a SHARED main-checkout-hygiene
    fault that recurs identically for every subsequent task, so halting
    collapses N silent per-task blocks into one loud signal.
    `park_lock_contended` is the opposite — a FOREIGN git process (dominantly
    a `git commit --only` holding the index lock across its pre-commit hook)
    owns project_root's index for a bounded, SELF-CLEARING window. Halting
    the queue for it is the 2+/day halt this task exists to remove.
    """

    def _make_git_ops(self) -> MagicMock:
        git_ops = MagicMock()
        git_ops.push_main = AsyncMock(return_value='pushed')
        # Same gotcha as TestMapAdvanceFailureStashFailed._make_git_ops: a
        # bare MagicMock auto-vivifies a TRUTHY child attribute, which would
        # defeat the mapper's `getattr(..., None) or <default>` fallback. Any
        # side channel not deliberately set must be deleted.
        del git_ops._last_stash_dirty_files
        del git_ops._last_park_lock_info
        return git_ops

    async def test_park_lock_contended_blocks_without_halting(self) -> None:
        """Per-task 'blocked' with structured facts; halt/unhalt untouched."""
        from orchestrator.merge_gates import _map_advance_failure

        git_ops = self._make_git_ops()
        git_ops._last_park_lock_info = {
            'lock_path': '/p/.git/index.lock',
            'age_seconds': 301.0,
            'waited_seconds': 300.0,
            # A coherent real shape: the lock was ~1s old when first observed
            # and the full 300s grace was waited out. (This test asserts
            # nothing about recovery advice — see the dedicated tests below.)
            'initial_age_seconds': 1.0,
            'grace_seconds': 300.0,
            'dirty_files': [],
        }
        halt = MagicMock()
        unhalt = MagicMock()
        cas_retries = {'t1': 2}

        outcome = await _map_advance_failure(
            git_ops, 'park_lock_contended',
            task_id='t1',
            merge_commit_fallback='deadbeef',
            halt=halt,
            unhalt=unhalt,
            cas_retries=cas_retries,
        )

        # (1)/(2) The queue is left strictly alone.
        halt.assert_not_called()
        unhalt.assert_not_called()

        # (3) Per-task disposition, reusing the existing status.
        assert outcome.status == 'blocked'

        # (4) The reason carries the SUBSTANTIVE facts an operator needs —
        # asserted on the path and the numbers, never on prose wording.
        assert '/p/.git/index.lock' in outcome.reason
        assert '301' in outcome.reason, (
            f'reason must report the observed lock age; got {outcome.reason!r}'
        )
        assert '300' in outcome.reason, (
            f'reason must report how long we waited; got {outcome.reason!r}'
        )
        assert 'transient' in outcome.reason.lower(), (
            f'reason must state this is transient/retried; got {outcome.reason!r}'
        )

        # (5) Terminal-for-this-attempt bookkeeping.
        assert 't1' not in cas_retries

    async def test_park_lock_contended_survives_an_unset_side_channel(self) -> None:
        """Defensive: with _last_park_lock_info unset the mapper must still
        return a per-task 'blocked' without halting, never raise.

        Mirrors the `getattr(git_ops, '_last_stash_dirty_files', None) or []`
        idiom — a missing side channel degrades the reason's detail, never
        the disposition.
        """
        from orchestrator.merge_gates import _map_advance_failure

        git_ops = self._make_git_ops()  # side channel deliberately deleted
        halt = MagicMock()

        outcome = await _map_advance_failure(
            git_ops, 'park_lock_contended',
            task_id='t2',
            merge_commit_fallback='deadbeef',
            halt=halt,
            unhalt=MagicMock(),
            cas_retries={},
        )

        halt.assert_not_called()
        assert outcome.status == 'blocked'

    async def _map(self, info: dict | None) -> tuple:
        """Map a `park_lock_contended` with *info* as the side channel.

        Returns ``(outcome, halt)`` so each caller can assert both the
        recovery text and the (invariant) non-halting disposition.
        """
        from orchestrator.merge_gates import _map_advance_failure

        git_ops = self._make_git_ops()
        if info is not None:
            git_ops._last_park_lock_info = info
        halt = MagicMock()
        outcome = await _map_advance_failure(
            git_ops, 'park_lock_contended',
            task_id='t9',
            merge_commit_fallback='deadbeef',
            halt=halt,
            unhalt=MagicMock(),
            cas_retries={},
        )
        return (outcome, halt)

    async def test_live_commit_shape_gets_no_destructive_advice(self) -> None:
        """A live `git commit --only` must NEVER be told to `rm -f` its lock.

        This is the headline regression.  Telling an operator to delete a
        live commit's index.lock corrupts that in-flight commit — the exact
        destruction the implementation forbids ITSELF from doing elsewhere
        (see test_foreign_lock_file_is_never_deleted, and git_ops' "the
        foreign lock left strictly alone").

        The shape below is an ordinary docs-direct-commit-on-main: the lock
        was 2s old when we first saw it and its pre-commit hook (this repo's
        runs pyright; CLAUDE.md instructs `timeout: 300000`) merely outlived
        the 300s grace by a couple of seconds.  Note `age_seconds` (302) is
        greater than `waited_seconds` (300) — which is why a staleness test
        keyed on the POST-wait age fires here, wrongly.
        """
        outcome, halt = await self._map({
            'lock_path': '/p/.git/index.lock',
            'age_seconds': 302.0,
            'waited_seconds': 300.0,
            'initial_age_seconds': 2.0,
            'grace_seconds': 300.0,
            'dirty_files': [],
        })

        halt.assert_not_called()
        assert outcome.status == 'blocked'

        assert 'rm -f' not in outcome.reason, (
            'a live commit must never be offered destructive lock removal; '
            f'got {outcome.reason!r}'
        )
        assert 'crashed' not in outcome.reason.lower(), (
            'a 2-second-old lock must not be described as a crashed leftover; '
            f'got {outcome.reason!r}'
        )

        # Suppressing the ADVICE must not suppress the DIAGNOSIS.
        assert '/p/.git/index.lock' in outcome.reason
        assert '302' in outcome.reason, (
            f'reason must still report the observed age; got {outcome.reason!r}'
        )
        assert '300' in outcome.reason, (
            f'reason must still report how long we waited; got {outcome.reason!r}'
        )

    async def test_crashed_leftover_shape_still_gets_the_advice(self) -> None:
        """A lock already older than a full grace when FIRST observed is the
        one shape for which `rm -f` is defensible."""
        outcome, halt = await self._map({
            'lock_path': '/p/.git/index.lock',
            'age_seconds': 3900.0,
            'waited_seconds': 300.0,
            'initial_age_seconds': 3600.0,
            'grace_seconds': 300.0,
            'dirty_files': [],
        })

        halt.assert_not_called()
        assert outcome.status == 'blocked'

        assert 'rm -f /p/.git/index.lock' in outcome.reason, (
            'an hour-old leftover must carry actionable recovery; got '
            f'{outcome.reason!r}'
        )
        # Asserted on the substantive token, never on prose wording: the
        # advice is only safe when paired with the liveness check.
        assert 'no git process' in outcome.reason.lower(), (
            'destructive advice must tell the operator to confirm no git '
            f'process is running in project_root first; got {outcome.reason!r}'
        )

    async def test_zero_grace_young_lock_gets_no_destructive_advice(self) -> None:
        """grace=0 must not turn EVERY lock into a "crashed leftover".

        `git.merge_park_lock_grace_seconds` is tunable to 0 — a blessed,
        documented probe-only fail-fast off-switch (GitConfig's docstring
        and test_zero_is_accepted_as_probe_only_off_switch).  A staleness
        test keyed on the grace ALONE (`initial_age > grace`) makes every
        non-zero age exceed it, so an ordinary live `git commit --only`
        whose lock is half a second old gets told to `rm -f` it — deleting
        a live commit's index.lock and corrupting that in-flight commit.

        Staleness must therefore clear max(grace, _STALE_LOCK_FLOOR_S):
        how the operator tuned the WAIT carries no information about
        whether the lock's owner is alive.
        """
        for initial_age in (0.5, 2.0, 299.0):
            outcome, halt = await self._map({
                'lock_path': '/p/.git/index.lock',
                'age_seconds': initial_age,
                'waited_seconds': 0.0,
                'initial_age_seconds': initial_age,
                'grace_seconds': 0.0,
                'dirty_files': [],
            })

            halt.assert_not_called()
            assert outcome.status == 'blocked'
            assert 'rm -f' not in outcome.reason, (
                f'a {initial_age}s-old lock under a grace=0 off-switch is a '
                'live commit, not a crashed leftover, and must never be '
                f'offered destructive lock removal; got {outcome.reason!r}'
            )
            assert 'crashed' not in outcome.reason.lower(), (
                f'a {initial_age}s-old lock must not be described as a '
                f'crashed leftover; got {outcome.reason!r}'
            )
            # Suppressing the ADVICE must not suppress the DIAGNOSIS.
            assert '/p/.git/index.lock' in outcome.reason

    async def test_zero_grace_still_reaches_the_floor_for_a_real_leftover(
        self,
    ) -> None:
        """The floor SUPPRESSES false positives; it must not suppress the
        one true positive.  An hour-old lock is a crashed-git leftover
        whatever the grace is tuned to — including the grace=0 off-switch.
        """
        outcome, halt = await self._map({
            'lock_path': '/p/.git/index.lock',
            'age_seconds': 3600.0,
            'waited_seconds': 0.0,
            'initial_age_seconds': 3600.0,
            'grace_seconds': 0.0,
            'dirty_files': [],
        })

        halt.assert_not_called()
        assert outcome.status == 'blocked'
        assert 'rm -f /p/.git/index.lock' in outcome.reason, (
            'an hour-old leftover must still carry actionable recovery even '
            f'when the wait is switched off; got {outcome.reason!r}'
        )
        assert 'no git process' in outcome.reason.lower(), (
            'destructive advice must remain paired with the liveness check; '
            f'got {outcome.reason!r}'
        )

    async def test_a_grace_above_the_floor_still_governs(self) -> None:
        """The floor is a FLOOR, not a replacement.  With a grace tuned
        ABOVE the floor, a lock older than the floor but younger than the
        grace is still an ordinary slow pre-commit hook, not a leftover.
        """
        outcome, _halt = await self._map({
            'lock_path': '/p/.git/index.lock',
            'age_seconds': 400.0,
            'waited_seconds': 0.0,
            'initial_age_seconds': 400.0,   # > 300s floor, < 900s grace
            'grace_seconds': 900.0,
            'dirty_files': [],
        })

        assert 'rm -f' not in outcome.reason, (
            'an operator who RAISED the grace has declared hooks that long '
            f'to be normal; got {outcome.reason!r}'
        )

    async def test_floor_matches_the_documented_pre_commit_budget(self) -> None:
        """The floor is a literal (orchestrator.config is TYPE_CHECKING-only
        in merge_gates), so pin it against the config default it mirrors —
        otherwise the two drift silently.

        (`async` only to satisfy this module's global asyncio pytestmark.)
        """
        from orchestrator.config import GitConfig
        from orchestrator.merge_gates import _STALE_LOCK_FLOOR_S

        assert GitConfig().merge_park_lock_grace_seconds == pytest.approx(
            _STALE_LOCK_FLOOR_S
        ), (
            '_STALE_LOCK_FLOOR_S must track '
            'GitConfig.merge_park_lock_grace_seconds\'s default (this repo\'s '
            'documented pre-commit budget)'
        )

    async def test_missing_staleness_keys_default_to_no_advice(self) -> None:
        """Absent evidence of staleness is not evidence of staleness.

        A pre-step-15 (legacy) side-channel dict carries neither
        `initial_age_seconds` nor `grace_seconds`.  The failure mode of a
        false positive here is data loss, so the conservative default is no
        advice — and never a raise.
        """
        outcome, halt = await self._map({
            'lock_path': '/p/.git/index.lock',
            'age_seconds': 3900.0,
            'waited_seconds': 300.0,
            'dirty_files': [],
        })

        halt.assert_not_called()
        assert outcome.status == 'blocked'
        assert 'rm -f' not in outcome.reason, (
            'a side channel with no staleness evidence must not produce '
            f'destructive advice; got {outcome.reason!r}'
        )
