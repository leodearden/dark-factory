"""Tests that drive orchestrator.evals.cli._run_single_eval with both executors patched.

They pin role-based routing, the per-cell echo, and the one campaign gate shared
across the CLI's config loop.
"""

from __future__ import annotations

import contextlib
import functools
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from _campaign_gate_helpers import NO_INJECTED_GATE
from _eval_refusal import REFUSAL_TEXT

from orchestrator.evals.metrics import RoleUnderTest

# ---------------------------------------------------------------------------
# CLI dispatch — _run_single_eval routes architect candidates (step-13/14)
#
# The user-observable signal: `orchestrator eval` with an architect-role
# candidate (role=='architect') routes to run_architect_eval (NOT run_eval) and
# surfaces the per-fixture plan_quality score. An ordinary implementer config
# still routes to run_eval, unchanged. Hermetic: both runners and config
# resolution are patched — no live worktree, no LLM.
# ---------------------------------------------------------------------------

def _fake_eval_result(**over):
    from orchestrator.evals.results import EvalResult

    defaults: dict = dict(
        task_id='df_task_2605',
        config_name='x',
        outcome='done',
        metrics={'composite_score': 1.0},
        worktree_path='/tmp/wt',
        wall_clock_ms=1000,
    )
    defaults.update(over)
    return EvalResult(**defaults)


def _dispatch_single_eval(cfg, capsys, arch_metrics=None):
    """Drive ``orchestrator.evals.cli._run_single_eval`` with a resolved config, both runners patched.

    ``get_config_by_name`` is patched to return ``cfg`` so the dispatch is
    exercised purely on ``cfg.role``. ``arch_metrics`` overrides the fake
    architect result's metrics dict. Returns ``(out, run_eval, run_architect)``.

    The campaign gate's builder is bound to degrade to ``None`` (task 4427).
    ``base_config`` here is a bare ``MagicMock``, so ``.usage_cap.enabled`` is a
    TRUTHY Mock attribute — once ``_run_single_eval`` owns a campaign gate, the
    production builder would construct a REAL ``UsageGate`` out of a mock config
    on every one of these routing cases (touching the filesystem for probe dirs
    and installing a SIGHUP handler). ``None`` is the ungated campaign, which is
    what every routing/echo case here already meant.
    """
    from orchestrator.evals import cli, eval_gate

    arch_result = _fake_eval_result(
        config_name=cfg.name,
        metrics=arch_metrics or {
            'role_under_test': 'architect', 'plan_quality': 0.75,
        },
    )
    impl_result = _fake_eval_result(config_name=cfg.name)

    mock_run_eval = AsyncMock(return_value=impl_result)
    mock_run_arch = AsyncMock(return_value=arch_result)

    with contextlib.ExitStack() as es:
        p = es.enter_context
        p(patch('orchestrator.evals.runner.run_eval', mock_run_eval))
        p(patch('orchestrator.evals.runner.run_architect_eval', mock_run_arch))
        p(patch('orchestrator.evals.configs.get_config_by_name',
                MagicMock(return_value=cfg)))
        p(patch('orchestrator.evals.eval_gate.campaign_usage_gate',
                functools.partial(eval_gate.campaign_usage_gate,
                                  factory=AsyncMock(return_value=None))))
        cli._run_single_eval(
            Path('/fake/task.json'), cfg.name, base_config=MagicMock(),
        )
    out = capsys.readouterr().out
    return out, mock_run_eval, mock_run_arch


def _dispatch_mixed_eval(cfgs, capsys, *, gate=None, raise_on: str | None = None):
    """Drive ``orchestrator.evals.cli._run_single_eval`` over a MIXED config list, gate seam patched.

    The sibling of :func:`_dispatch_single_eval` for the CAMPAIGN question
    (task 4427): ``_run_single_eval``'s ``for cfg in configs`` loop is a second,
    independent campaign loop the stage functions never pass through, and the
    property worth pinning — one gate shared by BOTH the architect and the
    implementer dispatch branch — only exists when several configs run in ONE
    call. ``config_name='all'`` plus a patched ``EVAL_CONFIGS`` is what makes
    the loop iterate the supplied list.

    Each fake executor asserts AT CALL TIME that the gate is not yet shut down:
    an after-the-fact ``assert_awaited_once`` cannot tell "torn down once after
    the loop" from "torn down between configs". ``raise_on=<config name>`` makes
    that config's executor raise, which propagates out of ``asyncio.run``.
    """
    from orchestrator.evals import cli, eval_gate

    def _make(metrics):
        async def fake(_task_path, cfg, *_a, usage_gate=NO_INJECTED_GATE, **_k):
            calls.append((cfg.name, usage_gate))
            if gate is not None:
                assert gate.shutdown.await_count == 0, (
                    'the campaign gate was torn down mid-loop — the configs '
                    'after it would run without failover'
                )
            if raise_on is not None and cfg.name == raise_on:
                raise RuntimeError('executor exploded')
            return _fake_eval_result(config_name=cfg.name, metrics=dict(metrics))
        return fake

    calls: list[tuple[str, object]] = []
    mock_run_eval = AsyncMock(side_effect=_make({'composite_score': 1.0}))
    mock_run_arch = AsyncMock(
        side_effect=_make({'role_under_test': 'architect', 'plan_quality': 0.75}),
    )
    mock_build_gate = AsyncMock(return_value=gate)

    with contextlib.ExitStack() as es:
        p = es.enter_context
        p(patch('orchestrator.evals.runner.run_eval', mock_run_eval))
        p(patch('orchestrator.evals.runner.run_architect_eval', mock_run_arch))
        p(patch('orchestrator.evals.configs.EVAL_CONFIGS', list(cfgs)))
        p(patch('orchestrator.evals.eval_gate.campaign_usage_gate',
                functools.partial(eval_gate.campaign_usage_gate,
                                  factory=mock_build_gate)))
        error: BaseException | None = None
        try:
            cli._run_single_eval(
                Path('/fake/task.json'), 'all', base_config=MagicMock(),
            )
        except Exception as exc:  # surfaced to the caller below
            error = exc
    return capsys.readouterr().out, {
        'run_eval': mock_run_eval, 'run_arch': mock_run_arch,
        'build_gate': mock_build_gate, 'calls': calls, 'error': error,
    }


class TestCliArchitectDispatch:
    def _arch_cfg(self):
        from orchestrator.evals.configs import EvalConfig

        return EvalConfig(
            'architect-sonnet-high', 'claude', 'sonnet', 'high', role=RoleUnderTest.ARCHITECT,
        )

    def _impl_cfg(self):
        from orchestrator.evals.configs import EvalConfig

        return EvalConfig('opus-high', 'claude', 'opus', 'high')

    def test_architect_config_routes_to_run_architect_eval(self, capsys):
        _, run_eval, run_arch = _dispatch_single_eval(self._arch_cfg(), capsys)
        run_arch.assert_called_once()
        run_eval.assert_not_called()

    def test_architect_dispatch_surfaces_plan_quality(self, capsys):
        out, _, _ = _dispatch_single_eval(self._arch_cfg(), capsys)
        # The per-fixture plan-quality score is echoed to the operator.
        assert 'plan_quality' in out
        assert '0.75' in out

    def test_cap_tainted_cell_echo_names_the_taint(self, capsys):
        # An operator watching the run must see the infra refusal LIVE, not a
        # bare `plan_quality=None` that reads like a scoring quirk.
        out, _, _ = _dispatch_single_eval(
            self._arch_cfg(), capsys,
            arch_metrics={
                'role_under_test': 'architect',
                'plan_quality': None,
                'cap_tainted': True,
                'invocation_errors': [
                    {'stage': 'architect', 'cause': 'cap_hit', 'reason': REFUSAL_TEXT},
                ],
            },
        )
        # Scoped to the PER-CELL echo line ('plan_quality=' with the '='), not
        # the whole capture: the plan-quality table printed after the loop
        # already names the exclusion, and asserting over `out` would pass on
        # that alone while the live per-cell line still read a bare None.
        echo_line = next(
            ln for ln in out.splitlines() if 'plan_quality=' in ln
        )
        # Cause-neutral label + the record naming the ACTUAL cause: a permanent
        # config error must not read as a transient cap window.
        assert 'unmeasurable' in echo_line
        assert f'architect:cap_hit: {REFUSAL_TEXT}' in echo_line

    def test_model_not_found_echo_does_not_read_as_a_cap_window(self, capsys):
        # The same taint flag covers a PERMANENT candidate-configuration error.
        # An operator must not be told to "rerun after the cap resets" for a
        # model that does not exist (reviewer: design-coherence).
        out, _, _ = _dispatch_single_eval(
            self._arch_cfg(), capsys,
            arch_metrics={
                'role_under_test': 'architect',
                'plan_quality': None,
                'cap_tainted': True,
                'invocation_errors': [{
                    'stage': 'architect', 'cause': 'model_not_found',
                    'reason': 'no such model gpt-9',
                }],
            },
        )
        echo_line = next(
            ln for ln in out.splitlines() if 'plan_quality=' in ln
        )
        assert 'unmeasurable' in echo_line
        assert 'architect:model_not_found: no such model gpt-9' in echo_line
        assert 'cap' not in echo_line.lower()

    def test_implementer_config_still_routes_to_run_eval(self, capsys):
        _, run_eval, run_arch = _dispatch_single_eval(self._impl_cfg(), capsys)
        run_eval.assert_called_once()
        run_arch.assert_not_called()


class TestCliDispatchCampaignGate:
    """``_run_single_eval``'s config loop owns ONE gate (task 4427).

    A second, independent campaign loop the μ stage functions never pass
    through: ``orchestrator eval --config all`` walks every candidate in one
    process. Both dispatch branches must ride the SAME gate, or the architect
    configs and the implementer configs each keep their own cap-state view and
    the second half of the run re-discovers what the first half already paid to
    learn.
    """

    def _arch_cfg(self):
        from orchestrator.evals.configs import EvalConfig

        return EvalConfig(
            'architect-sonnet-high', 'claude', 'sonnet', 'high', role=RoleUnderTest.ARCHITECT,
        )

    def _impl_cfg(self):
        from orchestrator.evals.configs import EvalConfig

        return EvalConfig('opus-high', 'claude', 'opus', 'high')

    def _mixed(self):
        return [self._arch_cfg(), self._impl_cfg()]

    def test_one_gate_is_built_for_the_whole_loop(self, capsys):
        from shared.testing import make_gate_mock

        gate = make_gate_mock()
        _out, mocks = _dispatch_mixed_eval(self._mixed(), capsys, gate=gate)

        # ONE build for the loop, not one per config.
        mocks['build_gate'].assert_awaited_once()
        assert mocks['error'] is None

    def test_both_dispatch_branches_share_the_same_gate(self, capsys):
        from shared.testing import make_gate_mock

        gate = make_gate_mock()
        _out, mocks = _dispatch_mixed_eval(self._mixed(), capsys, gate=gate)

        # Both branches ran…
        assert {name for name, _g in mocks['calls']} == {
            'architect-sonnet-high', 'opus-high',
        }
        # …and each received the SAME object (identity, not equality).
        assert [g for _n, g in mocks['calls']] == [gate, gate]

    def test_gate_is_shut_down_once_after_the_loop(self, capsys):
        from shared.testing import make_gate_mock

        gate = make_gate_mock()
        _out, mocks = _dispatch_mixed_eval(self._mixed(), capsys, gate=gate)

        # The in-call assertion in the helper already proved it was still live
        # for every config; this pins that it comes down exactly once.
        assert len(mocks['calls']) == 2
        gate.shutdown.assert_awaited_once()

    def test_gate_is_shut_down_when_a_config_raises(self, capsys):
        """The teardown is in campaign_usage_gate's finally, so a blown cell
        cannot leak the probe loop into the rest of the process."""
        from shared.testing import make_gate_mock

        gate = make_gate_mock()
        _out, mocks = _dispatch_mixed_eval(
            self._mixed(), capsys, gate=gate, raise_on='architect-sonnet-high',
        )

        assert isinstance(mocks['error'], RuntimeError)
        gate.shutdown.assert_awaited_once()

    def test_a_degraded_campaign_threads_an_explicit_none(self, capsys):
        """Ungated is a DECISION every config must be told about explicitly.

        A config handed no argument would build its own gate — restoring the
        per-cell construction, and re-paying a probe-dir alloc plus a SIGHUP
        steal per config on the zero-account path.
        """
        _out, mocks = _dispatch_mixed_eval(self._mixed(), capsys, gate=None)

        mocks['build_gate'].assert_awaited_once()
        assert [g for _n, g in mocks['calls']] == [None, None]
