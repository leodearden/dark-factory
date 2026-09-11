"""Tests for orchestrator.evals.live_fixture — the live shadow-eval fixture builder.

Hermetic throughout, mirroring ``test_eval_task_sampler``'s style: every task
record and plan is an inline synthetic dict, every path is under ``tmp_path``,
and nothing here reads the live repo, the task store, the network or an LLM.
That is not merely test hygiene — C3's purity invariant (the builder reads
nothing but its arguments) is itself one of the properties under test, so the
suite would be unable to state it if it needed a real checkout to run.

Step map:
  step-01/02  ShadowShape — the four-value cell vocabulary and its plan rule
  step-03/04  build_live_fixture — the emitted key surface (and the omissions)
  step-05/06  build_live_fixture — the refusal table
  step-07/08  build_live_fixture — argument purity: no repo reads, no aliasing
  step-09/10  round trip through runner.load_task + build_eval_orch_config
  step-11/12  live_verify_commands / live_stratum — the task_sampler hookups
"""

from __future__ import annotations

import json

import pytest

from orchestrator.evals.live_fixture import ShadowShape


class TestShadowShape:
    """The closed cell-shape vocabulary (PRD C1) and the shape→plan rule."""

    def test_exactly_the_four_c1_values(self):
        assert [s.value for s in ShadowShape] == [
            'implementer',
            'architect',
            'architect-consequence',
            'end-to-end',
        ]

    def test_member_is_a_plain_str_and_json_round_trips(self):
        # C3 types `shape: str`; a StrEnum member satisfies that verbatim and
        # serialises as the bare value (no 'ShadowShape.IMPLEMENTER' leaking
        # into the shadow_cells.shape TEXT column or a persisted result).
        assert isinstance(ShadowShape.IMPLEMENTER, str)
        assert json.dumps(ShadowShape.IMPLEMENTER) == '"implementer"'
        assert json.loads(json.dumps(ShadowShape.IMPLEMENTER)) == 'implementer'

    @pytest.mark.parametrize(
        'raw,member',
        [
            ('implementer', ShadowShape.IMPLEMENTER),
            ('architect', ShadowShape.ARCHITECT),
            ('architect-consequence', ShadowShape.ARCHITECT_CONSEQUENCE),
            ('end-to-end', ShadowShape.END_TO_END),
        ],
    )
    def test_string_coercion_recovers_the_member(self, raw, member):
        # A caller holding a value read back from the shadow_cells TEXT column
        # coerces it without a lookup table of its own.
        assert ShadowShape(raw) is member

    @pytest.mark.parametrize(
        'shape,requires_plan',
        [
            (ShadowShape.IMPLEMENTER, True),
            (ShadowShape.ARCHITECT_CONSEQUENCE, True),
            (ShadowShape.ARCHITECT, False),
            (ShadowShape.END_TO_END, False),
        ],
    )
    def test_requires_plan_is_total_over_the_vocabulary(self, shape, requires_plan):
        assert shape.requires_plan is requires_plan
