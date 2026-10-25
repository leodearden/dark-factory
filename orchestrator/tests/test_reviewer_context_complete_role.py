"""A second reviewer generation whose Rule 2 blocks only on a stated ground.

The context-complete (D2) generation is selected per dispatch by
``ReviewBriefing``. Each role is resolved through the production resolve
path (``PromptArtifactStore.resolve``, the call
``workflow.py::TaskWorkflow._resolve_role_system_prompt`` makes), so a pinned
artifact or a harness-version mix-up shows up exactly as it would live.
"""

from __future__ import annotations

from pathlib import Path

from shared.prompt_artifact import ArtifactProvenance, PromptArtifactStore

from orchestrator.agents.review_briefing import ReviewBriefing
from orchestrator.agents.roles import (
    ALL_REVIEWERS,
    CODE_QUALITY_GUIDANCE,
    REVIEWER_COMPREHENSIVE,
    AgentRole,
    build_reviewer_prompt_spec,
    build_reviewer_role,
    reviewers_for,
)

SPECIALIZATION_FOOTER = '## Your Specialization: '


def _context_complete() -> AgentRole:
    (role,) = reviewers_for(ReviewBriefing.CONTEXT_COMPLETE)
    return role


def _resolve(store: PromptArtifactStore, role: AgentRole) -> str:
    assert role.prompt_spec is not None
    assert role.prompt_harness_version is not None
    return store.resolve(
        role.prompt_spec, executor_model='opus', harness_version=role.prompt_harness_version,
    ).text


def _without_rule_2(heuristics: str) -> str:
    start = heuristics.index('\n2. ')
    end = heuristics.index('\n3. ')
    return heuristics[:start] + heuristics[end:]


class TestRoster:
    def test_diff_only_is_the_existing_roster(self):
        assert reviewers_for(ReviewBriefing.DIFF_ONLY) == ALL_REVIEWERS

    def test_context_complete_keeps_the_verdict_identity(self):
        role = _context_complete()

        assert role.name == REVIEWER_COMPREHENSIVE.name == 'reviewer_comprehensive'
        assert role.prompt_harness_version == 'reviewer-d2-v1'
        assert REVIEWER_COMPREHENSIVE.prompt_harness_version == 'reviewer-v1'


class TestPromptSplit:
    def test_the_contract_half_is_shared(self):
        role = _context_complete()
        assert role.prompt_spec is not None
        assert REVIEWER_COMPREHENSIVE.prompt_spec is not None

        assert role.prompt_spec.contract == REVIEWER_COMPREHENSIVE.prompt_spec.contract

    def test_the_generations_differ_only_in_rule_2(self):
        role = _context_complete()
        assert role.prompt_spec is not None
        assert REVIEWER_COMPREHENSIVE.prompt_spec is not None

        assert _without_rule_2(role.prompt_spec.baseline_heuristics) == _without_rule_2(
            REVIEWER_COMPREHENSIVE.prompt_spec.baseline_heuristics,
        )
        assert role.prompt_spec.baseline_heuristics != (
            REVIEWER_COMPREHENSIVE.prompt_spec.baseline_heuristics
        )


class TestResolvedText:
    def test_the_context_complete_role_resolves_to_its_own_generation(self, tmp_path: Path):
        store = PromptArtifactStore(tmp_path / 'artifacts')
        role = _context_complete()
        incumbent = _resolve(store, REVIEWER_COMPREHENSIVE)
        assert role.prompt_spec is not None
        assert REVIEWER_COMPREHENSIVE.prompt_spec is not None

        text = _resolve(store, role)

        assert text == role.prompt_spec.in_code_constant
        assert text != REVIEWER_COMPREHENSIVE.prompt_spec.in_code_constant
        assert CODE_QUALITY_GUIDANCE in text
        assert incumbent[incumbent.index(SPECIALIZATION_FOOTER):] in text

    def test_the_incumbent_resolves_to_its_own_generation(self, tmp_path: Path):
        store = PromptArtifactStore(tmp_path / 'artifacts')
        context_complete = _context_complete()
        assert REVIEWER_COMPREHENSIVE.prompt_spec is not None
        assert context_complete.prompt_spec is not None

        text = _resolve(store, REVIEWER_COMPREHENSIVE)

        assert text == REVIEWER_COMPREHENSIVE.prompt_spec.in_code_constant
        assert text != context_complete.prompt_spec.in_code_constant

    def test_an_incumbent_pin_never_reaches_the_new_generation(self, tmp_path: Path):
        store = PromptArtifactStore(tmp_path / 'artifacts')
        role = _context_complete()
        before = _resolve(store, role)
        assert REVIEWER_COMPREHENSIVE.prompt_spec is not None

        store.pin(
            REVIEWER_COMPREHENSIVE.prompt_spec.prompt_id, 'opus', 'reviewer-v1',
            heuristics='PINNED: an optimised incumbent heuristics block.',
            provenance=ArtifactProvenance(
                optimizer_model='claude-opus-4',
                corpus_hash='sha256:deadbeef',
                split_seed=42,
                held_out_TEST_score=0.9,
                accept_delta=0.05,
                git_sha='abc1234',
                date='2026-10-18',
                harness_version='reviewer-v1',
            ),
        )

        assert 'PINNED: an optimised' in _resolve(store, REVIEWER_COMPREHENSIVE)
        assert _resolve(store, role) == before


def test_each_generation_resolves_to_the_public_builders_text(tmp_path: Path):
    store = PromptArtifactStore(tmp_path / 'artifacts')

    texts = {
        briefing: _resolve(store, build_reviewer_role('x', 'spec', briefing))
        for briefing in ReviewBriefing
    }

    for briefing, text in texts.items():
        assert text == build_reviewer_prompt_spec('x', 'spec', briefing).in_code_constant
    assert texts[ReviewBriefing.DIFF_ONLY] != texts[ReviewBriefing.CONTEXT_COMPLETE]


def test_build_reviewer_role_shares_the_incumbent_tooling():
    role = build_reviewer_role('x', 'spec', ReviewBriefing.CONTEXT_COMPLETE)

    assert role.name == 'reviewer_x'
    assert role.allowed_tools == REVIEWER_COMPREHENSIVE.allowed_tools
    assert role.disallowed_tools == REVIEWER_COMPREHENSIVE.disallowed_tools
    assert role.mcp_families == REVIEWER_COMPREHENSIVE.mcp_families
    assert role.prompt_harness_version == 'reviewer-d2-v1'
