"""The γ user-observable signal: a red in an UNTOUCHED module is suppressible
at the real merge boundary (``merge_queue._run_post_merge_verify``).

Flake-ledger PRD §8.2 / task 3787 (γ). This module drives the production
chokepoint every merge verify flows through, not a helper in isolation.

THE BUG γ CLOSES (§3.1). Under ``merge_verify_breadth='full'`` a merge-role
verify EXECUTES every registered module — ``run_scoped_verification`` expanded
the set from the registry internally. But that expansion rebound a LOCAL that
never propagated out, so ``LocalRunner._module_configs`` — the set the
merge-flake suppression gate maps failing node-ids against — was still the
TASK's own modules. A red naming ``beta/tests/test_b.py::test_b`` on an
alpha-only task therefore mapped to no given subproject,
``_group_node_ids_by_subproject`` returned ``None``, and the gate answered
"unconfirmable" -> merge stays red. The gate was inverted exactly where it
mattered: the more clearly unrelated the failure, the less able it was to say
so. ``EventType.merge_flake_suppressed`` has a lifetime count of 0 because of
it.

γ resolves the effective set ONCE at the merge-request boundary and hands the
identical set to the local runner, the wire spec (and hence the remote's
reconstruction of it) and the suppression gate — by construction, not by
asserting that two sites agree (INV-5).

INV-4 (storm-escape-required) is pinned END-TO-END here rather than assumed:
``apply_merge_flake_suppression`` is the sole production suppression path and
unconditionally bumps the streak, so the newly-firing suppressions ride the
EXISTING escape. γ changes the gate's INPUT, not its plumbing — this module
makes that non-regressable.

Harness provenance: ``_make_config``/``_make_git_ops``/``_make_req`` are
cross-imported from ``test_merge_queue_main_health`` (the pattern
``test_verify_scope_inversion_boundary`` already uses for this same
chokepoint), and ``_materialize``/``_module_config``/``_FakeEventStore``/
``_FakeEscalationQueue`` from ``test_verify_merge_flake_suppression`` — the
on-disk node-id -> subproject probe is ``Path.exists``-based, so the merge
worktree holds REAL files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal
from unittest.mock import AsyncMock, patch

import pytest
from test_merge_queue_main_health import _make_config, _make_git_ops, _make_req
from test_verify_merge_flake_suppression import (
    _FakeEscalationQueue,
    _FakeEventStore,
    _materialize,
    _module_config,
)

from orchestrator import verify
from orchestrator.config import ModuleConfig
from orchestrator.event_store import EventType
from orchestrator.merge_gates import PostMergePyrightResult
from orchestrator.merge_queue import MergeOutcome, _run_post_merge_verify
from orchestrator.verify import VerifyResult

_MERGE_SHA = 'd' * 40

# Real files in the merge worktree — `_group_node_ids_by_subproject` probes
# `(worktree / prefix / relpath).exists()` and `(worktree / relpath).exists()`,
# so a node-id can only map if its file is genuinely on disk.
_ALPHA_TEST = 'alpha/tests/test_a.py'
_BETA_TEST = 'beta/tests/test_b.py'
# In NO registered module, but PRESENT on disk — so the control below fails on
# "not a registered module", not on "file missing".
_DELTA_TEST = 'delta/tests/test_d.py'

_BETA_FAILING_ID = f'{_BETA_TEST}::test_b'
_DELTA_FAILING_ID = f'{_DELTA_TEST}::test_d'


def _failing_scoped_result(node_id: str) -> VerifyResult:
    """A merge-verify red naming exactly one failing test, in the
    ``FAILED <nodeid>`` shape ``_extract_failing_test_ids`` recovers.
    """
    return VerifyResult(
        passed=False,
        test_output=f'FAILED {node_id}\n',
        lint_output='',
        type_output='',
        summary='Failures: tests failed',
        category='test_failure',
        cause_hint='AssertionError: boom',
    )


def _passing_result() -> VerifyResult:
    return VerifyResult(
        passed=True, test_output='', lint_output='', type_output='',
        summary='All checks passed',
    )


def _failing_rerun_result() -> VerifyResult:
    """The isolated re-run itself failing — a GENUINE red, not a load flake."""
    return VerifyResult(
        passed=False, test_output='FAILED again\n', lint_output='', type_output='',
        summary='Failures: tests failed', category='test_failure',
    )


@pytest.fixture(autouse=True)
def _reset_suppression_streak():
    """The INV-4 streak is a module global; reset it around every test here so
    a threshold assertion measures THIS test's suppressions only.
    """
    verify._merge_flake_suppression_streak = 0
    yield
    verify._merge_flake_suppression_streak = 0


async def _drive_merge_boundary(
    tmp_path: Path,
    *,
    task_id: str,
    breadth: Literal['scoped', 'full'],
    registry: dict[str, ModuleConfig],
    touched: list[ModuleConfig],
    failing_node_id: str = _BETA_FAILING_ID,
    isolated_rerun_passes: bool = True,
    event_store=None,
    escalation_queue=None,
) -> MergeOutcome | None:
    """Drive the REAL ``_run_post_merge_verify`` — the single funnel every
    production merge verify flows through.

    Patches, at the ``merge_queue`` lookup sites the boundary resolves at
    ``LocalRunner`` construction time:
      * ``run_scoped_verification`` -> a failing VerifyResult naming
        *failing_node_id* (this also re-patches over the autouse
        ``_mock_merge_queue_verification`` conftest stub, the standard idiom);
      * ``_run_unscoped_typechecks`` -> a clean ``PostMergePyrightResult``, so
        the post-scoped pyright gate never decides the outcome here;
      * ``orchestrator.verify.run_verification`` -> the isolated re-run engine
        ``_merge_gate_isolated_rerun`` calls, passing or failing per
        *isolated_rerun_passes*.

    Returns the boundary's own return value: ``None`` means the merge LANDS.
    """
    config = _make_config(tmp_path, merge_verify_breadth=breadth)
    config._module_configs = dict(registry)

    git_ops = _make_git_ops(tmp_path)

    task_wt = tmp_path / f'task-wt-{task_id}'
    task_wt.mkdir(parents=True, exist_ok=True)
    merge_wt = tmp_path / f'merge-wt-{task_id}'
    merge_wt.mkdir(parents=True, exist_ok=True)
    _materialize(merge_wt, _ALPHA_TEST, _BETA_TEST, _DELTA_TEST)

    req = _make_req(task_id, task_wt, config)
    req.module_configs = list(touched)

    rerun = _passing_result() if isolated_rerun_passes else _failing_rerun_result()

    with (
        patch(
            'orchestrator.merge_queue.run_scoped_verification',
            new=AsyncMock(return_value=_failing_scoped_result(failing_node_id)),
        ),
        patch(
            'orchestrator.merge_queue._run_unscoped_typechecks',
            new=AsyncMock(return_value=PostMergePyrightResult()),
        ),
        patch.object(verify, 'run_verification', new=AsyncMock(return_value=rerun)),
    ):
        return await _run_post_merge_verify(
            git_ops, req, merge_wt,
            timeouts={},
            enospc_retries={},
            max_timeouts=3,
            max_enospc=1,
            event_store=event_store,
            escalation_queue=escalation_queue,
            merge_sha=_MERGE_SHA,
        )


def _suppression_events(store: _FakeEventStore) -> list[tuple]:
    return [e for e in store.emits if e[0] is EventType.merge_flake_suppressed]


class TestUntouchedModuleRedIsSuppressibleAtTheMergeBoundary:
    """A red in a REGISTERED-but-untouched module, re-run green in isolation,
    is suppressed — and the merge lands.

    The task touched ONLY alpha (``req.module_configs == [mc_alpha]``); the
    registry holds alpha AND beta; the red names a beta test. Under
    ``breadth='full'`` the merge verify was already EXECUTING beta's suite, so
    the gate must be able to reason about beta's failures too.
    """

    @staticmethod
    def _two_module_registry() -> tuple[ModuleConfig, ModuleConfig, dict[str, ModuleConfig]]:
        mc_alpha, mc_beta = _module_config('alpha'), _module_config('beta')
        return mc_alpha, mc_beta, {'alpha': mc_alpha, 'beta': mc_beta}

    # -- (a) the merge LANDS ---------------------------------------------------

    @pytest.mark.asyncio
    async def test_merge_lands_when_untouched_module_red_passes_isolated_rerun(
        self, tmp_path: Path,
    ):
        """(a) Outcome is None — the merge lands.

        Today this is a blocked MergeOutcome: the gate receives only
        [mc_alpha], so ``beta/tests/test_b.py::test_b`` maps to no given
        subproject and the verdict collapses to 'unconfirmable' -> None ->
        merge stays red.
        """
        mc_alpha, _mc_beta, registry = self._two_module_registry()

        outcome = await _drive_merge_boundary(
            tmp_path, task_id='a1', breadth='full',
            registry=registry, touched=[mc_alpha],
        )

        assert outcome is None, (
            f'expected the confirmed flake to be suppressed and the merge to '
            f'LAND; got {outcome!r}'
        )

    # -- (b) the measured-floor signal ----------------------------------------

    @pytest.mark.asyncio
    async def test_emits_exactly_one_merge_flake_suppressed_fact(self, tmp_path: Path):
        """(b) The INV-2 structured fact whose lifetime count is 0 today —
        emitted exactly once, naming the node-ids examined and the merge SHA.
        """
        mc_alpha, _mc_beta, registry = self._two_module_registry()
        store = _FakeEventStore()

        outcome = await _drive_merge_boundary(
            tmp_path, task_id='b1', breadth='full',
            registry=registry, touched=[mc_alpha], event_store=store,
        )

        assert outcome is None
        events = _suppression_events(store)
        assert len(events) == 1, (
            f'expected exactly one merge_flake_suppressed fact; got {len(events)}'
        )
        _event_type, _task_id, data = events[0]
        assert data['node_ids'] == [_BETA_FAILING_ID]
        assert data['merge_sha'] == _MERGE_SHA

    # -- (c) INV-4: the storm escape is ridden, not bypassed -------------------

    @pytest.mark.asyncio
    async def test_one_suppression_advances_the_storm_streak_by_exactly_one(
        self, tmp_path: Path,
    ):
        """(c1) The suppression bumps the INV-4 streak — the fail-soft path is
        not escape-less."""
        mc_alpha, _mc_beta, registry = self._two_module_registry()
        assert verify._merge_flake_suppression_streak == 0

        outcome = await _drive_merge_boundary(
            tmp_path, task_id='c1', breadth='full',
            registry=registry, touched=[mc_alpha],
            escalation_queue=_FakeEscalationQueue(),
        )

        assert outcome is None
        assert verify._merge_flake_suppression_streak == 1, (
            'a confirmed suppression at the merge boundary must advance the '
            'storm streak by exactly 1'
        )

    @pytest.mark.asyncio
    async def test_threshold_suppressions_file_one_l2_storm_escalation_and_reset(
        self, tmp_path: Path,
    ):
        """(c2) Driving the SAME scenario ``_MERGE_FLAKE_SUPPRESSION_STREAK_THRESHOLD``
        times files exactly ONE born-at-L2 storm escalation and resets the
        counter — end-to-end from the merge boundary, not from a direct call to
        the bump helper.
        """
        mc_alpha, _mc_beta, registry = self._two_module_registry()
        threshold = verify._MERGE_FLAKE_SUPPRESSION_STREAK_THRESHOLD
        queue = _FakeEscalationQueue(open_l2=None)

        for i in range(threshold):
            outcome = await _drive_merge_boundary(
                tmp_path, task_id=f'c2-{i}', breadth='full',
                registry=registry, touched=[mc_alpha], escalation_queue=queue,
            )
            assert outcome is None, f'suppression {i} did not land the merge'

        assert len(queue.submitted) == 1, (
            f'expected exactly ONE storm escalation at the threshold; '
            f'got {len(queue.submitted)}'
        )
        esc = queue.submitted[0]
        assert esc.task_id == verify._MERGE_FLAKE_SUPPRESSION_STORM_SENTINEL
        assert esc.task_id == 'merge-flake-suppression-storm'
        assert esc.severity == 'critical'
        assert esc.level == 2
        assert verify._merge_flake_suppression_streak == 0, (
            'the window must reset so the counter cannot grow unbounded'
        )


class TestMergeBoundarySuppressionControls:
    """Each of these must stay EXACTLY as it is today — γ widens the MAPPABLE
    set, it does not weaken the fail-closed gate.
    """

    @staticmethod
    def _two_module_registry() -> tuple[ModuleConfig, ModuleConfig, dict[str, ModuleConfig]]:
        mc_alpha, mc_beta = _module_config('alpha'), _module_config('beta')
        return mc_alpha, mc_beta, {'alpha': mc_alpha, 'beta': mc_beta}

    # -- (i) breadth='scoped': byte-identical legacy ---------------------------

    @pytest.mark.asyncio
    async def test_scoped_breadth_still_blocks_and_suppresses_nothing(
        self, tmp_path: Path,
    ):
        """(i) At the shipped default the merge verify never widened past the
        task's modules, so the gate has no business reasoning about beta — the
        merge stays blocked, no fact, streak untouched.
        """
        mc_alpha, _mc_beta, registry = self._two_module_registry()
        store = _FakeEventStore()

        outcome = await _drive_merge_boundary(
            tmp_path, task_id='i1', breadth='scoped',
            registry=registry, touched=[mc_alpha], event_store=store,
            escalation_queue=_FakeEscalationQueue(),
        )

        assert isinstance(outcome, MergeOutcome)
        assert outcome.status == 'blocked'
        assert _suppression_events(store) == []
        assert verify._merge_flake_suppression_streak == 0

    # -- (ii) empty registry: safe degrade ------------------------------------

    @pytest.mark.asyncio
    async def test_full_breadth_empty_registry_suppresses_nothing(self, tmp_path: Path):
        """(ii) breadth='full' with an EMPTY registry resolves to the passed set
        — never suppress against a set that was never resolved."""
        mc_alpha, _mc_beta, _registry = self._two_module_registry()
        store = _FakeEventStore()

        outcome = await _drive_merge_boundary(
            tmp_path, task_id='ii1', breadth='full',
            registry={}, touched=[mc_alpha], event_store=store,
            escalation_queue=_FakeEscalationQueue(),
        )

        assert isinstance(outcome, MergeOutcome)
        assert outcome.status == 'blocked'
        assert _suppression_events(store) == []
        assert verify._merge_flake_suppression_streak == 0

    # -- (iii) a red in NO registered module: still fail-closed ---------------

    @pytest.mark.asyncio
    async def test_red_in_unregistered_module_still_blocks(self, tmp_path: Path):
        """(iii) The red names a file that is on disk but under NO registered
        module. γ widens the mappable set; it does not make the gate guess.
        """
        mc_alpha, _mc_beta, registry = self._two_module_registry()
        store = _FakeEventStore()

        outcome = await _drive_merge_boundary(
            tmp_path, task_id='iii1', breadth='full',
            registry=registry, touched=[mc_alpha],
            failing_node_id=_DELTA_FAILING_ID,
            event_store=store, escalation_queue=_FakeEscalationQueue(),
        )

        assert isinstance(outcome, MergeOutcome)
        assert outcome.status == 'blocked'
        assert _suppression_events(store) == []
        assert verify._merge_flake_suppression_streak == 0

    # -- (iv) a GENUINE red in an untouched module: never suppressed ----------

    @pytest.mark.asyncio
    async def test_genuine_red_in_untouched_module_is_not_suppressed(
        self, tmp_path: Path,
    ):
        """(iv) Same widened mapping, but the isolated re-run STILL FAILS — the
        failure is real, so it must block. This is the control that separates
        "the gate can now SEE untouched modules" from "the gate now waves them
        through".
        """
        mc_alpha, _mc_beta, registry = self._two_module_registry()
        store = _FakeEventStore()

        outcome = await _drive_merge_boundary(
            tmp_path, task_id='iv1', breadth='full',
            registry=registry, touched=[mc_alpha],
            isolated_rerun_passes=False,
            event_store=store, escalation_queue=_FakeEscalationQueue(),
        )

        assert isinstance(outcome, MergeOutcome)
        assert outcome.status == 'blocked'
        assert _suppression_events(store) == []
        assert verify._merge_flake_suppression_streak == 0


# ---------------------------------------------------------------------------
# step-7: wire parity — the SPEC leg gets the same effective set, so
# local ≡ remote BY CONSTRUCTION (PRD §8.2 ordering invariant, INV-5)
# ---------------------------------------------------------------------------

_GLOBAL_TEST_COMMAND = 'uv run pytest tests/'
_GLOBAL_TYPECHECK_COMMAND = 'uv run pyright src/'


async def _capture_boundary_consumers(
    tmp_path: Path,
    *,
    task_id: str,
    breadth: Literal['scoped', 'full'],
    registry: dict[str, ModuleConfig],
    touched: list[ModuleConfig],
    global_commands: bool = False,
) -> dict:
    """Drive ``_run_post_merge_verify`` and capture what each consumer of the
    module set actually RECEIVED.

    Spies wrap (and delegate to) the real ``build_merge_verify_spec`` and
    ``LocalRunner`` at their ``merge_queue`` lookup sites — the boundary
    resolves both names there when it constructs the pool — so the captured
    values are the genuine arguments, not a re-derivation. ``VerifyRunnerPool.
    dispatch`` is stubbed to a pass so the run completes without executing
    anything.

    ``global_commands=True`` additionally gives the config LIVE global
    commands, which is what makes INV-1's zero-module ``global_verify_command``
    fallback observable at all (see the empty-set class below).

    Returns ``{'spec_arg', 'spec', 'local_args', 'config', 'merge_wt'}``.
    """
    from orchestrator import merge_queue as merge_queue_module

    config = _make_config(tmp_path, merge_verify_breadth=breadth)
    config._module_configs = dict(registry)
    if global_commands:
        # INV-1 (task 2883): build_merge_verify_spec only sources
        # `global_verify_command` when the LIVE config actually has global
        # commands to source, so the zero-module fallback is unobservable
        # without these — the assertion would pass vacuously on None.
        config.test_command = _GLOBAL_TEST_COMMAND
        config.type_check_command = _GLOBAL_TYPECHECK_COMMAND

    git_ops = _make_git_ops(tmp_path)

    task_wt = tmp_path / f'task-wt-{task_id}'
    task_wt.mkdir(parents=True, exist_ok=True)
    merge_wt = tmp_path / f'merge-wt-{task_id}'
    merge_wt.mkdir(parents=True, exist_ok=True)
    _materialize(merge_wt, _ALPHA_TEST, _BETA_TEST, _DELTA_TEST)

    req = _make_req(task_id, task_wt, config)
    req.module_configs = list(touched)

    captured: dict = {'local_args': [], 'config': config, 'merge_wt': merge_wt}

    real_build = merge_queue_module.build_merge_verify_spec
    real_local_runner = merge_queue_module.LocalRunner

    def _spy_build(cfg, module_configs, task_files):
        captured['spec_arg'] = list(module_configs)
        spec = real_build(cfg, module_configs, task_files)
        captured['spec'] = spec
        return spec

    def _spy_local_runner(*args, **kwargs):
        captured['local_args'].append(list(args[2]))
        return real_local_runner(*args, **kwargs)

    with (
        patch('orchestrator.merge_queue.build_merge_verify_spec', new=_spy_build),
        patch('orchestrator.merge_queue.LocalRunner', new=_spy_local_runner),
        patch.object(
            merge_queue_module.VerifyRunnerPool, 'dispatch',
            new=AsyncMock(return_value=_passing_result()),
        ),
    ):
        await _run_post_merge_verify(
            git_ops, req, merge_wt,
            timeouts={}, enospc_retries={},
            max_timeouts=3, max_enospc=1,
            merge_sha=_MERGE_SHA,
        )

    return captured


def _prefixes(module_configs) -> list[str]:
    return [mc.prefix for mc in module_configs]


class TestMergeBoundaryShipsOneEffectiveSetToBothConsumers:
    """The local runner and the wire spec receive the IDENTICAL module set.

    This is the §8.2 ordering invariant asserted AT THE BOUNDARY rather than
    as two independent expectations: the property under test is not "both
    happen to equal the registry", it is "both came from one resolution", so
    the decisive assertion is that the two captured sets are EQUAL TO EACH
    OTHER. The remote reconstructs its module set from ``spec.verify_commands``
    (verify_runner ``_module_config_from_command``), so projecting the spec
    from the same effective set is what makes local ≡ remote by construction.
    """

    @staticmethod
    def _three_module_registry():
        mc_a = _module_config('alpha')
        mc_b = _module_config('beta')
        mc_g = _module_config('gamma')
        return mc_a, mc_b, mc_g, {'alpha': mc_a, 'beta': mc_b, 'gamma': mc_g}

    # -- (a) both consumers, one set ------------------------------------------

    @pytest.mark.asyncio
    async def test_spec_and_local_runner_receive_the_same_effective_set(
        self, tmp_path: Path,
    ):
        """(a) Both captured sets are the whole registry in registry order, and
        EQUAL to each other. Today the LocalRunner arg is already correct
        (step-6) but the spec arg is still [mc_alpha] -> RED."""
        mc_a, _mc_b, _mc_g, registry = self._three_module_registry()

        captured = await _capture_boundary_consumers(
            tmp_path, task_id='w1', breadth='full',
            registry=registry, touched=[mc_a],
        )

        assert captured['local_args'], 'no LocalRunner was constructed'
        local_set = captured['local_args'][0]
        spec_set = captured['spec_arg']

        assert _prefixes(local_set) == ['alpha', 'beta', 'gamma']
        assert _prefixes(spec_set) == ['alpha', 'beta', 'gamma']
        # THE property: one resolution reached both consumers.
        assert _prefixes(spec_set) == _prefixes(local_set), (
            'the wire spec and the local runner must receive the identical set '
            'BY CONSTRUCTION, not by two sites independently agreeing'
        )

    # -- (b) the spec projection ----------------------------------------------

    @pytest.mark.asyncio
    async def test_spec_projection_covers_every_effective_module(self, tmp_path: Path):
        """(b) The spec's own projected commands follow the effective set —
        both the per-module verify_commands and the unscoped typecheck gate."""
        mc_a, _mc_b, _mc_g, registry = self._three_module_registry()

        captured = await _capture_boundary_consumers(
            tmp_path, task_id='w2', breadth='full',
            registry=registry, touched=[mc_a],
        )
        spec = captured['spec']
        effective = captured['spec_arg']

        assert [vc.prefix for vc in spec.verify_commands] == _prefixes(effective)
        expected_unscoped = [
            mc.prefix for mc in effective if mc.type_check_command is not None
        ]
        assert [vc.prefix for vc in spec.unscoped_typecheck.commands] == expected_unscoped
        assert expected_unscoped == ['alpha', 'beta', 'gamma'], (
            'the fixture must give every module a type_check_command, else this '
            'assertion is vacuous'
        )

    # -- (c) the remote reconstruction leg ------------------------------------

    @pytest.mark.asyncio
    async def test_remote_reconstruction_yields_the_same_module_set(
        self, tmp_path: Path,
    ):
        """(c) Feed the spec the boundary ACTUALLY built through the remote
        entry point and assert the module set it reconstructs matches the
        local runner's — i.e. the remote merge gate now maps node-ids against
        the same modules the local gate does.

        Compared by PREFIX and command strings, never by object identity: the
        remote set is reconstructed through the wire codec, so identity is
        meaningless across it.
        """
        from orchestrator import verify_runner

        mc_a, _mc_b, _mc_g, registry = self._three_module_registry()

        captured = await _capture_boundary_consumers(
            tmp_path, task_id='w3', breadth='full',
            registry=registry, touched=[mc_a],
        )
        local_set = captured['local_args'][0]

        remote_scoped = AsyncMock(return_value=_passing_result())
        remote_unscoped = AsyncMock(return_value=PostMergePyrightResult())
        await verify_runner.run_merge_verify_on_worktree(
            captured['merge_wt'], captured['config'], captured['spec'],
            merge_sha=_MERGE_SHA,
            run_scoped=remote_scoped,
            run_unscoped=remote_unscoped,
        )

        assert remote_scoped.await_args is not None
        reconstructed = remote_scoped.await_args.args[2]
        assert _prefixes(reconstructed) == _prefixes(local_set), (
            f'remote reconstructed {_prefixes(reconstructed)!r} but the local '
            f'runner got {_prefixes(local_set)!r} — the two gates would map '
            f'node-ids against different module sets'
        )
        assert [mc.test_command for mc in reconstructed] == [
            mc.test_command for mc in local_set
        ]

    # -- CONTROLS --------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_scoped_breadth_spec_carries_the_passed_set_unchanged(
        self, tmp_path: Path,
    ):
        """Legacy byte-identical: at the shipped default the spec still carries
        only the task's own modules."""
        mc_a, _mc_b, _mc_g, registry = self._three_module_registry()

        captured = await _capture_boundary_consumers(
            tmp_path, task_id='w4', breadth='scoped',
            registry=registry, touched=[mc_a],
        )

        assert _prefixes(captured['spec_arg']) == ['alpha']
        assert _prefixes(captured['local_args'][0]) == ['alpha']

    @pytest.mark.asyncio
    async def test_full_breadth_empty_registry_spec_carries_the_passed_set(
        self, tmp_path: Path,
    ):
        """Safe degrade: an empty registry resolves to the passed set, so the
        spec is unchanged from today — never an empty projection."""
        mc_a, _mc_b, _mc_g, _registry = self._three_module_registry()

        captured = await _capture_boundary_consumers(
            tmp_path, task_id='w5', breadth='full',
            registry={}, touched=[mc_a],
        )

        assert _prefixes(captured['spec_arg']) == ['alpha']
        assert _prefixes(captured['local_args'][0]) == ['alpha']


# ---------------------------------------------------------------------------
# amendment (reviewer suggestion 1): the ZERO-MODULE task keeps today's gate
# ---------------------------------------------------------------------------


class TestMergeBoundaryLeavesTheZeroModuleTaskAlone:
    """An EMPTY ``req.module_configs`` is NOT widened, even at breadth='full'
    with a populated registry.

    ``WorkflowRunner._resolve_module_configs`` returns ``[]`` whenever the task
    has no assigned modules — docs-only and root-file tasks — and this repo's
    live config sets ``merge_verify_breadth: "full"`` over 9 registered
    modules, so this is a LIVE path, not a corner case. Widening ``[]`` to the
    registry would silently replace such a task's whole gate: the wire spec
    would flip from INV-1's global-command fallback (task 2883) to per-module
    commands with ``global_verify_command=None``, the scoped leg would leave
    ``_derive_fallback_runs`` for a per-module full-suite fan-out, and
    ``_run_unscoped_typechecks`` would cover every registered module.

    γ's mandate is to make the suppression gate map node-ids against the set
    that ACTUALLY RAN — not to change what runs. So the boundary mirrors
    ``run_scoped_verification``'s own ``if module_configs:`` guard and this
    class pins that: the widening applies exactly where it applied before,
    and the zero-module task is left where it already was.
    """

    @staticmethod
    def _three_module_registry():
        mc_a = _module_config('alpha')
        mc_b = _module_config('beta')
        mc_g = _module_config('gamma')
        return mc_a, mc_b, mc_g, {'alpha': mc_a, 'beta': mc_b, 'gamma': mc_g}

    @pytest.mark.asyncio
    async def test_empty_passed_set_reaches_both_consumers_unwidened(
        self, tmp_path: Path,
    ):
        """breadth='full' + POPULATED registry + no assigned modules: both the
        spec leg and the local runner still receive the empty set."""
        _mc_a, _mc_b, _mc_g, registry = self._three_module_registry()

        captured = await _capture_boundary_consumers(
            tmp_path, task_id='z1', breadth='full',
            registry=registry, touched=[],
        )

        assert captured['spec_arg'] == [], (
            f"the spec leg must not be widened for a zero-module task; got "
            f"{_prefixes(captured['spec_arg'])!r}"
        )
        assert captured['local_args'], 'no LocalRunner was constructed'
        assert captured['local_args'][0] == [], (
            f"the local runner must not be widened either; got "
            f"{_prefixes(captured['local_args'][0])!r}"
        )

    @pytest.mark.asyncio
    async def test_zero_module_spec_still_sources_the_global_verify_command(
        self, tmp_path: Path,
    ):
        """INV-1 preserved with a POPULATED registry — the case the original
        comment block reasoned about only for an EMPTY registry.

        The spec ships zero per-module commands and the live globals instead,
        exactly as before γ, so the remote reconstructs the same global gate.
        """
        _mc_a, _mc_b, _mc_g, registry = self._three_module_registry()

        captured = await _capture_boundary_consumers(
            tmp_path, task_id='z2', breadth='full',
            registry=registry, touched=[], global_commands=True,
        )
        spec = captured['spec']

        assert spec.verify_commands == ()
        assert spec.unscoped_typecheck.commands == ()
        assert spec.global_verify_command is not None, (
            'a zero-module spec must carry the live global gate (INV-1, task '
            '2883), not an empty projection'
        )
        assert spec.global_verify_command.test_command == _GLOBAL_TEST_COMMAND

    @pytest.mark.asyncio
    async def test_non_empty_passed_set_still_widens_at_the_same_breadth(
        self, tmp_path: Path,
    ):
        """The CONTROL that keeps the guard narrow: same config, same registry,
        one assigned module -> γ's widening still fires on both legs (and the
        global fallback is correctly NOT sourced). Without this, a guard that
        disabled the widening outright would pass the two tests above."""
        mc_a, _mc_b, _mc_g, registry = self._three_module_registry()

        captured = await _capture_boundary_consumers(
            tmp_path, task_id='z3', breadth='full',
            registry=registry, touched=[mc_a], global_commands=True,
        )

        assert _prefixes(captured['spec_arg']) == ['alpha', 'beta', 'gamma']
        assert _prefixes(captured['local_args'][0]) == ['alpha', 'beta', 'gamma']
        assert captured['spec'].global_verify_command is None


class TestMergeBoundaryModuleConfigsWrapper:
    """``merge_queue._merge_boundary_module_configs`` — the guard, and nothing
    but the guard.

    INV-5: the wrapper must DELEGATE the expansion to the one shared
    ``verify_plan.effective_merge_module_configs`` rather than reimplement it,
    since a second copy of that expression is exactly the drift PRD §8.2
    exists to remove. Pinned with a spy that returns a value neither the
    registry nor the passed set could produce.
    """

    @staticmethod
    def _config(tmp_path: Path):
        config = _make_config(tmp_path, merge_verify_breadth='full')
        config._module_configs = {'alpha': _module_config('alpha')}
        return config

    def test_non_empty_passed_set_delegates_to_the_shared_helper(
        self, tmp_path: Path, monkeypatch,
    ):
        from orchestrator import merge_queue as merge_queue_module

        config = self._config(tmp_path)
        passed = [_module_config('beta')]
        sentinel = [_module_config('sentinel')]
        calls: list[tuple] = []

        def _spy(cfg, module_configs):
            calls.append((cfg, list(module_configs)))
            return list(sentinel)

        monkeypatch.setattr(
            merge_queue_module, 'effective_merge_module_configs', _spy,
        )

        result = merge_queue_module._merge_boundary_module_configs(config, passed)

        assert _prefixes(result) == ['sentinel']
        assert len(calls) == 1
        assert calls[0][0] is config
        assert _prefixes(calls[0][1]) == ['beta']

    def test_empty_passed_set_never_consults_the_helper_at_all(
        self, tmp_path: Path, monkeypatch,
    ):
        """Identity, and not even a call: the guard is the boundary's own
        policy, so the breadth helper's contract stays untouched (its other
        caller, verify.py's force_workspace branch, still widens an empty
        passed set exactly as it did before γ)."""
        from orchestrator import merge_queue as merge_queue_module

        config = self._config(tmp_path)
        calls: list[tuple] = []

        def _spy(cfg, module_configs):
            calls.append((cfg, list(module_configs)))
            return [_module_config('sentinel')]

        monkeypatch.setattr(
            merge_queue_module, 'effective_merge_module_configs', _spy,
        )

        passed: list[ModuleConfig] = []
        result = merge_queue_module._merge_boundary_module_configs(config, passed)

        assert result is passed
        assert calls == []
