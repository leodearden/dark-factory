"""Reviewer panel variant definitions for the trial.

Each variant defines a different panel composition to evaluate against
the production baseline (5x sonnet specialists).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from orchestrator.agents.roles import (
    _READ_ONLY_TOOLS,
    _REVIEWER_PROMPT_HARNESS_VERSION,
    _VERDICT_TOOLS,
    AgentRole,
    build_reviewer_prompt_spec,
)

# ---------------------------------------------------------------------------
# Spec + config types
# ---------------------------------------------------------------------------

@dataclass
class ReviewerSpec:
    """Specification for a single reviewer in a trial panel."""

    name: str
    model: str                 # "opus" | "sonnet" | cross-family model id (e.g. "gpt-5.4")
    specialization: str        # combined specialization prompt text
    budget: float = 2.0
    effort: str = 'high'
    # Cross-family dispatch (default-off keeps every existing spec byte-identical):
    backend: str = 'claude'                        # 'claude' | 'codex' | 'gemini'
    env_overrides: dict[str, str] | None = None    # merged into the invoke_agent subprocess env
    oauth_token_env: str | None = None             # env var whose value becomes invoke_agent's oauth_token


@dataclass
class VariantConfig:
    """A complete panel configuration to evaluate."""

    name: str
    description: str
    reviewers: list[ReviewerSpec] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Trial reviewer role builder — built from the SAME reviewer PromptSpec as
# production (roles.build_reviewer_prompt_spec: frozen CONTRACT + editable
# HEURISTICS, composed), so the trial reviewer's prompt AND verdict-tools
# transport match the live path exactly (task 2493 trial-parity decision:
# prompt parity forces transport parity, since the post-2484 CONTRACT tells
# the agent to CALL submit_review_verdict rather than emit JSON/prose — an
# output-schema-only capture would leave a prompt-parity reviewer with no
# way to emit). Replaces the drifted, pre-2484 _REVIEWER_SYSTEM_TEMPLATE.
# ---------------------------------------------------------------------------

_JCODEMUNCH_TOOLS = ['mcp__jcodemunch__*']


def build_trial_reviewer_role(spec_in: ReviewerSpec) -> AgentRole:
    """Build an AgentRole for a trial reviewer.

    Builds from the same reviewer ``PromptSpec`` as production
    (``roles.build_reviewer_prompt_spec``), so the trial reviewer's system
    prompt and live verdict-tools transport match production exactly.
    ``role.name`` is ``reviewer_{spec_in.name}`` (not ``trial_``-prefixed)
    because the verdict-tools server's ``reviewer == --verdict-role``
    identity check validates the emitted verdict's ``reviewer`` field
    against ``role.name``, and the frozen CONTRACT instructs the agent to
    emit ``reviewer_{name}``. Model/budget are still per-spec overridable;
    effort is consumed directly from ``spec_in.effort`` by the trial runner
    (AgentRole has no effort field).
    """
    spec = build_reviewer_prompt_spec(spec_in.name, spec_in.specialization)
    return AgentRole(
        name=f'reviewer_{spec_in.name}',
        system_prompt=spec.in_code_constant,
        allowed_tools=[*_READ_ONLY_TOOLS, *_VERDICT_TOOLS, *_JCODEMUNCH_TOOLS],
        disallowed_tools=['Edit', 'Write'],
        default_model=spec_in.model,
        default_budget=spec_in.budget,
        default_max_turns=30,
        mcp_families=frozenset({'verdict_tools'}),
        prompt_spec=spec,
        prompt_harness_version=_REVIEWER_PROMPT_HARNESS_VERSION,
    )


# ---------------------------------------------------------------------------
# Production specialization texts (copied from roles.py for reference)
# ---------------------------------------------------------------------------

_SPEC_TEST_ANALYST = (
    'Test coverage and quality. Are the right behaviors tested? Meaningful assertions? '
    'Untested failure modes? Edge cases? Do tests test what they claim?'
)

_SPEC_REUSE_AUDITOR = (
    'Code reuse and duplication. Is there code duplication? Missed existing utilities? '
    'Unnecessary new abstractions? Over-engineering?'
)

_SPEC_ARCHITECT = (
    'Architecture and design coherence. Consistent with system design? Good naming? '
    'Correct module boundaries? SOLID principles? Pattern consistency?'
)

_SPEC_PERFORMANCE = (
    'Performance and efficiency. Algorithmic complexity? N+1 queries? Unnecessary allocations? '
    'Hot path considerations? Resource cleanup?'
)

_SPEC_ROBUSTNESS = (
    'Robustness and error handling. Error handling at boundaries? Failure modes? '
    'Race conditions? Resource leaks? Graceful degradation?'
)


# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------

VARIANT_BASELINE = VariantConfig(
    name='baseline',
    description='Production panel: 5x sonnet specialists',
    reviewers=[
        ReviewerSpec(name='test_analyst', model='sonnet', specialization=_SPEC_TEST_ANALYST),
        ReviewerSpec(name='reuse_auditor', model='sonnet', specialization=_SPEC_REUSE_AUDITOR),
        ReviewerSpec(name='architect_reviewer', model='sonnet', specialization=_SPEC_ARCHITECT),
        ReviewerSpec(name='performance', model='sonnet', specialization=_SPEC_PERFORMANCE),
        ReviewerSpec(name='robustness', model='sonnet', specialization=_SPEC_ROBUSTNESS),
    ],
)

_SPEC_COMPREHENSIVE = (
    'Comprehensive code review covering ALL of the following areas:\n\n'
    '1. **Test coverage and quality**: Are the right behaviors tested? '
    'Meaningful assertions? Untested failure modes? Edge cases? '
    'Do tests test what they claim?\n\n'
    '2. **Code reuse and duplication**: Is there code duplication? '
    'Missed existing utilities? Unnecessary new abstractions? Over-engineering?\n\n'
    '3. **Architecture and design coherence**: Consistent with system design? '
    'Good naming? Correct module boundaries? SOLID principles? Pattern consistency?\n\n'
    '4. **Performance and efficiency**: Algorithmic complexity? N+1 queries? '
    'Unnecessary allocations? Hot path considerations? Resource cleanup?\n\n'
    '5. **Robustness and error handling**: Error handling at boundaries? '
    'Failure modes? Race conditions? Resource leaks? Graceful degradation?\n\n'
    'You are responsible for ALL five areas above. Produce findings under each.'
)


VARIANT_A = VariantConfig(
    name='variant_a',
    description='1x opus generalist — depth replaces breadth',
    reviewers=[
        ReviewerSpec(
            name='comprehensive_reviewer',
            model='opus',
            specialization=_SPEC_COMPREHENSIVE,
            budget=5.0,
            effort='high',
        ),
    ],
)

VARIANT_A_MEDIUM = VariantConfig(
    name='variant_a_medium',
    description='1x opus generalist @ medium effort',
    reviewers=[
        ReviewerSpec(
            name='comprehensive_reviewer',
            model='opus',
            specialization=_SPEC_COMPREHENSIVE,
            budget=5.0,
            effort='medium',
        ),
    ],
)

VARIANT_A_MAX = VariantConfig(
    name='variant_a_max',
    description='1x opus generalist @ max effort',
    reviewers=[
        ReviewerSpec(
            name='comprehensive_reviewer',
            model='opus',
            specialization=_SPEC_COMPREHENSIVE,
            budget=10.0,
            effort='max',
        ),
    ],
)

VARIANT_B = VariantConfig(
    name='variant_b',
    description='2x opus (bug_hunter + design_critic) — depth + diversity',
    reviewers=[
        ReviewerSpec(
            name='opus_bug_hunter',
            model='opus',
            specialization=(
                'Bug hunting and robustness. You are responsible for:\n\n'
                '1. **Test coverage and quality**: Are the right behaviors tested? '
                'Meaningful assertions? Untested failure modes? Edge cases? '
                'Do tests test what they claim?\n\n'
                '2. **Robustness and error handling**: Error handling at boundaries? '
                'Failure modes? Race conditions? Resource leaks? Graceful degradation?\n\n'
                '3. **Performance and efficiency**: Algorithmic complexity? N+1 queries? '
                'Unnecessary allocations? Hot path considerations? Resource cleanup?\n\n'
                'Focus on finding bugs, runtime failures, and correctness issues.'
            ),
            budget=4.0,
            effort='high',
        ),
        ReviewerSpec(
            name='opus_design_critic',
            model='opus',
            specialization=(
                'Design and structure quality. You are responsible for:\n\n'
                '1. **Architecture and design coherence**: Consistent with system design? '
                'Good naming? Correct module boundaries? SOLID principles? Pattern consistency?\n\n'
                '2. **Code reuse and duplication**: Is there code duplication? '
                'Missed existing utilities? Unnecessary new abstractions? Over-engineering?\n\n'
                'Focus on structural quality, maintainability, and design coherence.'
            ),
            budget=4.0,
            effort='high',
        ),
    ],
)

VARIANT_C = VariantConfig(
    name='variant_c',
    description='1x opus cross-cutting + 2x sonnet specialists',
    reviewers=[
        ReviewerSpec(
            name='opus_strategic',
            model='opus',
            specialization=(
                'Strategic cross-cutting review. You are responsible for:\n\n'
                '1. **Architecture and design coherence**: Consistent with system design? '
                'Good naming? Correct module boundaries? SOLID principles? Pattern consistency?\n\n'
                '2. **Code reuse and duplication**: Is there code duplication? '
                'Missed existing utilities? Unnecessary new abstractions? Over-engineering?\n\n'
                '3. **Performance and efficiency**: Algorithmic complexity? N+1 queries? '
                'Unnecessary allocations? Hot path considerations? Resource cleanup?\n\n'
                'Focus on high-level structural quality and cross-cutting concerns.'
            ),
            budget=4.0,
            effort='high',
        ),
        ReviewerSpec(name='sonnet_test_analyst', model='sonnet', specialization=_SPEC_TEST_ANALYST),
        ReviewerSpec(name='sonnet_robustness', model='sonnet', specialization=_SPEC_ROBUSTNESS),
    ],
)

VARIANT_D = VariantConfig(
    name='variant_d',
    description='3x sonnet (data-driven consolidated trim)',
    reviewers=[
        ReviewerSpec(
            name='sonnet_test_analyst',
            model='sonnet',
            specialization=_SPEC_TEST_ANALYST,
        ),
        ReviewerSpec(
            name='sonnet_bug_hunter',
            model='sonnet',
            specialization=(
                'Bug hunting and runtime correctness. You are responsible for:\n\n'
                '1. **Robustness and error handling**: Error handling at boundaries? '
                'Failure modes? Race conditions? Resource leaks? Graceful degradation?\n\n'
                '2. **Performance and efficiency**: Algorithmic complexity? N+1 queries? '
                'Unnecessary allocations? Hot path considerations? Resource cleanup?\n\n'
                '3. **Runtime architecture**: Does the implementation correctly handle '
                'async/await, concurrency, timeouts, and resource lifecycle? '
                'Are runtime invariants maintained?\n\n'
                'Focus on anything that can break at runtime.'
            ),
        ),
        ReviewerSpec(
            name='sonnet_design_critic',
            model='sonnet',
            specialization=(
                'Design quality and structural coherence. You are responsible for:\n\n'
                '1. **Code reuse and duplication**: Is there code duplication? '
                'Missed existing utilities? Unnecessary new abstractions? Over-engineering?\n\n'
                '2. **Structural architecture**: Consistent with system design? '
                'Good naming? Correct module boundaries? SOLID principles? '
                'Pattern consistency? Proper abstractions?\n\n'
                'Focus on maintainability, clarity, and structural soundness.'
            ),
        ),
    ],
)

ALL_VARIANTS = [VARIANT_BASELINE, VARIANT_A, VARIANT_B, VARIANT_C, VARIANT_D]

# Effort sweep — 1x opus at different thinking levels (variant_a is 'high')
EFFORT_SWEEP_VARIANTS = [VARIANT_A_MEDIUM, VARIANT_A, VARIANT_A_MAX]


# ---------------------------------------------------------------------------
# Eval-revival κ refresh set (task 2476, PRD decision 13)
#
# Isolate the MODEL as the sole independent variable: each candidate mirrors
# VARIANT_A's single comprehensive-generalist structure (effort='high',
# budget=5.0, _SPEC_COMPREHENSIVE), swapping ONLY model/backend, so quality
# and cost deltas are attributable to the model rather than panel shape. The
# incumbent is VARIANT_A itself (the Apr-8 trial winner + production reviewer).
# Deliberately NOT added to ALL_VARIANTS so `full`/`sweep` stay unchanged.
# ---------------------------------------------------------------------------

VARIANT_SONNET5_SOLO = VariantConfig(
    name='variant_sonnet5_solo',
    description='1x sonnet generalist (Sonnet 5) — model swap vs the 1x opus incumbent',
    reviewers=[
        ReviewerSpec(
            name='comprehensive_reviewer',
            model='sonnet',
            specialization=_SPEC_COMPREHENSIVE,
            budget=5.0,
            effort='high',
        ),
    ],
)

VARIANT_CROSS_FAMILY = VariantConfig(
    name='variant_cross_family',
    description='1x codex generalist (gpt-5.4) — cross-family model swap vs the 1x opus incumbent',
    reviewers=[
        ReviewerSpec(
            name='comprehensive_reviewer',
            model='gpt-5.4',
            specialization=_SPEC_COMPREHENSIVE,
            budget=5.0,
            effort='high',
            backend='codex',
        ),
    ],
)

# Refresh leaderboard order: 1×Opus incumbent first, then the two candidates.
REVIEWER_REFRESH_VARIANTS = [VARIANT_A, VARIANT_SONNET5_SOLO, VARIANT_CROSS_FAMILY]
