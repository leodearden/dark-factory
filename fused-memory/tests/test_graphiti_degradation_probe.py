"""Tests for recon's Graphiti-degradation probe protocol (task 4644).

The protocol is DATA — a ladder of ``limit`` values, a minimum probe count,
a fan-out floor, two stat-key names and one verdict template — and the prompt
prose is rendered from it. Every assertion here is therefore about referential
integrity against those constants, never about the wording they render into:
rewording the guidance must keep this suite green, while dropping a ladder
rung, renaming a counter or weakening the verdict rule must turn it red.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from fused_memory.reconciliation.graphiti_degradation_probe import (
    GRAPHITI_DEGRADATION_REPRODUCED_STAT_KEY,
    GRAPHITI_MIXED_STORE_PROBES_RUN_STAT_KEY,
    HIGH_FANOUT_LIMIT_FLOOR,
    MIN_PROBES_PER_CYCLE,
    NEGATIVE_SET_VERDICT_TEMPLATE,
    PROBE_LIMIT_LADDER,
    render_graphiti_degradation_probe_section,
)

from fused_memory.reconciliation.prompts.stage1 import STAGE1_SYSTEM_PROMPT
from fused_memory.reconciliation.prompts.stage2 import (
    STAGE2_SYSTEM_PROMPT,
    build_stage2_system_prompt,
)
from fused_memory.reconciliation.prompts.stage3 import STAGE3_SYSTEM_PROMPT

_VERDICT = NEGATIVE_SET_VERDICT_TEMPLATE.format(n=len(PROBE_LIMIT_LADDER))
_PROBE_SECTION = render_graphiti_degradation_probe_section(runs_probes=True)
_READ_ONLY_SECTION = render_graphiti_degradation_probe_section(runs_probes=False)


class TestProbeLadderIsAControlledVariable:
    """Requirements 1 and 2: N>=3 probes, with ``limit`` spanned rather than
    silently held at the one value that produced the cd53b227 false negative."""

    def test_ladder_supplies_at_least_the_per_cycle_minimum(self):
        """Requirement 1 — the ladder IS the reason for the minimum, so it
        must be able to satisfy it on its own."""
        assert len(PROBE_LIMIT_LADDER) >= MIN_PROBES_PER_CYCLE

    def test_ladder_rungs_are_distinct(self):
        """Three probes at one ``limit`` would raise N without controlling the
        fan-out confound at all."""
        assert len(set(PROBE_LIMIT_LADDER)) == len(PROBE_LIMIT_LADDER)

    def test_ladder_reaches_the_high_fanout_floor(self):
        """Requirement 2 — at least one probe at the fan-out size that has
        demonstrated power to fire."""
        assert max(PROBE_LIMIT_LADDER) >= HIGH_FANOUT_LIMIT_FLOOR

    def test_ladder_keeps_the_limit_that_produced_the_false_negative(self):
        """limit=3 is the lone probe cd53b227's Stage 2 ran. It stays IN the
        ladder: controlling a variable means spanning it, including the value
        already known to miss. Dropping it would swap the confound, not fix it."""
        assert 3 in PROBE_LIMIT_LADDER

    def test_ladder_includes_the_limit_both_positive_sightings_used(self):
        """Both reproductions came from limit=8 mixed-store queries."""
        assert 8 in PROBE_LIMIT_LADDER

    def test_minimum_probe_count_is_three(self):
        assert MIN_PROBES_PER_CYCLE == 3

    def test_high_fanout_floor_is_eight(self):
        assert HIGH_FANOUT_LIMIT_FLOOR == 8

    def test_ladder_is_immutable(self):
        """A module-level list would let one importer mutate the protocol for
        every other."""
        assert isinstance(PROBE_LIMIT_LADDER, tuple)


class TestCounterKeyNames:
    """Requirement 3 — the denominator pair. These exact spellings are what a
    cycle-report reader greps for, so they are pinned as literals here and
    imported (never retyped) everywhere else."""

    def test_probes_run_key_spelling(self):
        assert GRAPHITI_MIXED_STORE_PROBES_RUN_STAT_KEY == 'graphiti_mixed_store_probes_run'

    def test_reproduced_key_spelling(self):
        assert GRAPHITI_DEGRADATION_REPRODUCED_STAT_KEY == 'graphiti_degradation_reproduced'

    def test_the_two_counters_are_distinct_keys(self):
        assert (
            GRAPHITI_MIXED_STORE_PROBES_RUN_STAT_KEY
            != GRAPHITI_DEGRADATION_REPRODUCED_STAT_KEY
        )


class TestVerdictTemplate:
    """Requirement 4's permitted wording, single-sourced."""

    def test_renders_the_mandated_sentence_for_the_ladder(self):
        assert NEGATIVE_SET_VERDICT_TEMPLATE.format(n=len(PROBE_LIMIT_LADDER)) == (
            '0 of 3 probes reproduced; the fault is intermittent and '
            'load-dependent, so a negative set does not clear it.'
        )

    def test_render_leaves_no_unfilled_placeholder(self):
        """The rendered verdict is interpolated into stage f-strings; a stray
        brace would be silently swallowed there rather than raising."""
        rendered = NEGATIVE_SET_VERDICT_TEMPLATE.format(n=len(PROBE_LIMIT_LADDER))
        assert '{' not in rendered
        assert '}' not in rendered


def test_module_is_import_light():
    """The leaf sits on the prompt-import path and is imported by the equally
    import-light ``recon_self_model``, so it must stay stdlib-only.

    Probed in a FRESH interpreter: this test process has already imported the
    heavy modules transitively, so an in-process ``sys.modules`` check would
    pass vacuously. The constant round-trip at the end keeps the probe itself
    from silently no-opping if the import ever stops resolving.
    """
    probe = (
        'import sys; '
        'import fused_memory.reconciliation.graphiti_degradation_probe as p; '
        "assert 'mem0' not in sys.modules, sorted(k for k in sys.modules if 'mem0' in k); "
        "assert 'fused_memory.config.schema' not in sys.modules; "
        "assert 'fused_memory.reconciliation.harness' not in sys.modules; "
        'assert p.MIN_PROBES_PER_CYCLE == 3'
    )
    result = subprocess.run(
        [sys.executable, '-c', probe], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr


class TestBothBranchesCarryTheVerdictRule:
    """Requirement 4's mandated wording is shared between the two stages that
    assess store health, so it must exist exactly once and reach both."""

    @pytest.mark.parametrize('runs_probes', [True, False])
    def test_verdict_appears_exactly_once(self, runs_probes):
        rendered = render_graphiti_degradation_probe_section(runs_probes=runs_probes)
        assert rendered.count(_VERDICT) == 1


class TestProbeBranchRendersTheLadder:
    """Stage 2 runs the probes and reports the counters."""

    def test_every_ladder_rung_is_named(self):
        rendered = render_graphiti_degradation_probe_section(runs_probes=True)
        for limit in PROBE_LIMIT_LADDER:
            assert str(limit) in rendered

    def test_per_cycle_minimum_is_named(self):
        rendered = render_graphiti_degradation_probe_section(runs_probes=True)
        assert str(MIN_PROBES_PER_CYCLE) in rendered

    def test_the_denominator_pair_is_never_advertised_apart(self):
        """Requirement 3: `graphiti_degradation_reproduced` is unreadable
        without its denominator beside it, so no edit may leave one behind."""
        rendered = render_graphiti_degradation_probe_section(runs_probes=True)
        assert GRAPHITI_MIXED_STORE_PROBES_RUN_STAT_KEY in rendered
        assert GRAPHITI_DEGRADATION_REPRODUCED_STAT_KEY in rendered


class TestReadOnlyBranchDoesNotOrderProbes:
    """Stage 3 is read-only: it inherits the verdict rule and the cross-stage
    caveat, and nothing that would have it run a ladder or emit stats it does
    not own. Naming a counter to a stage is an instruction, not decoration."""

    @pytest.mark.parametrize(
        'stat_key',
        [
            GRAPHITI_MIXED_STORE_PROBES_RUN_STAT_KEY,
            GRAPHITI_DEGRADATION_REPRODUCED_STAT_KEY,
        ],
    )
    def test_counter_is_absent(self, stat_key):
        rendered = render_graphiti_degradation_probe_section(runs_probes=False)
        assert stat_key not in rendered


class TestBranchesDiffer:
    def test_capability_flag_is_load_bearing(self):
        """A renderer that ignored its keyword argument would satisfy every
        positive assertion above."""
        assert render_graphiti_degradation_probe_section(
            runs_probes=True
        ) != render_graphiti_degradation_probe_section(runs_probes=False)


class TestRenderIsSafeToInterpolate:
    """Both stage prompts are module-level f-strings. An interpolated value is
    not re-scanned, but a brace authored into the section would break the day
    someone inlines it -- the hazard `prompts/__init__.py` documents."""

    @pytest.mark.parametrize('runs_probes', [True, False])
    def test_render_carries_no_braces(self, runs_probes):
        rendered = render_graphiti_degradation_probe_section(runs_probes=runs_probes)
        assert '{' not in rendered
        assert '}' not in rendered


class TestSectionIsWiredIntoBothStages:
    """Pin the WIRING -- the section is embedded verbatim, once, in each stage
    that needs it. Prose may be reworded freely; the wiring may not silently
    break. Shape lifted from ``test_recon_gate_closure_guidance.py``.
    """

    def test_probing_stage_embeds_the_probe_section_once(self):
        assert STAGE2_SYSTEM_PROMPT.count(_PROBE_SECTION) == 1

    def test_read_only_stage_embeds_the_read_only_section_once(self):
        assert STAGE3_SYSTEM_PROMPT.count(_READ_ONLY_SECTION) == 1

    @pytest.mark.parametrize('project_id', ['dark_factory', 'autopilot_video'])
    def test_section_survives_both_runtime_builder_branches(self, project_id):
        """``autopilot_video`` splices a contamination guardrail in ahead of the
        ``## Available Tools`` sentinel; that injection must not displace or
        truncate this section."""
        assert _PROBE_SECTION in build_stage2_system_prompt(project_id)


class TestNoCrossContamination:
    """The capability split has to hold at the wiring level too, not just at
    the renderer's."""

    def test_read_only_stage_is_never_told_to_run_the_ladder(self):
        assert _PROBE_SECTION not in STAGE3_SYSTEM_PROMPT

    def test_probing_stage_does_not_receive_the_weaker_clause(self):
        assert _READ_ONLY_SECTION not in STAGE2_SYSTEM_PROMPT


class TestCounterNamesReachTheProbingStage:
    """The user-observable signal, end to end: the counter names an operator
    reads out of a cycle's stats are the same strings the stage was instructed
    to emit -- resolved through the constants, never retyped."""

    @pytest.mark.parametrize(
        'stat_key',
        [
            GRAPHITI_MIXED_STORE_PROBES_RUN_STAT_KEY,
            GRAPHITI_DEGRADATION_REPRODUCED_STAT_KEY,
        ],
    )
    def test_counter_is_named_in_the_probing_stage_prompt(self, stat_key):
        assert stat_key in STAGE2_SYSTEM_PROMPT


class TestStage1IsDeliberatelyExcluded:
    """Stage 1 is the memory-consolidation stage: it ran no probes in the
    incident and asserts nothing about store health. Its exclusion is a
    decision, pinned here so a later editor does not "fix" it by accident."""

    @pytest.mark.parametrize('section', [_PROBE_SECTION, _READ_ONLY_SECTION])
    def test_section_is_absent_from_stage1(self, section):
        assert section not in STAGE1_SYSTEM_PROMPT
