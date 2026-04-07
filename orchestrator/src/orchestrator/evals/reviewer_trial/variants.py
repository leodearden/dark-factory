"""Reviewer panel variant definitions for the trial.

Each variant defines a different panel composition to evaluate against
the production baseline (5x sonnet specialists).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from orchestrator.agents.roles import AgentRole


# ---------------------------------------------------------------------------
# Spec + config types
# ---------------------------------------------------------------------------

@dataclass
class ReviewerSpec:
    """Specification for a single reviewer in a trial panel."""

    name: str
    model: str                 # "opus" | "sonnet"
    specialization: str        # combined specialization prompt text
    budget: float = 2.0
    effort: str = 'high'


@dataclass
class VariantConfig:
    """A complete panel configuration to evaluate."""

    name: str
    description: str
    reviewers: list[ReviewerSpec] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Shared prompt template (mirrors roles._reviewer_role)
# ---------------------------------------------------------------------------

_REVIEWER_SYSTEM_TEMPLATE = """\
You are a code reviewer specializing in: **{specialization}**

## Your Task

Review the code diff provided and produce a structured JSON review.

## Output Schema

You MUST output ONLY valid JSON matching this schema:

```json
{{
  "reviewer": "{name}",
  "verdict": "PASS or ISSUES_FOUND",
  "issues": [
    {{
      "severity": "blocking or suggestion",
      "location": "src/foo.py:42",
      "category": "descriptive_category",
      "description": "Clear description of the issue",
      "suggested_fix": "How to fix it"
    }}
  ],
  "summary": "One paragraph summary"
}}
```

## Rules

1. **Be specific.** Every issue must have a file location and concrete description.
2. **Blocking means broken.** Use `blocking` ONLY for issues that will cause runtime errors,
   data corruption, security vulnerabilities, or API contract violations **within the scope
   of this task**. Do not block on:
   - Design concerns that are valid but outside this task's scope
   - Edge cases that cannot occur given the task's stated constraints
   - Missing features that belong in a follow-up task
   - Style, naming, or structural preferences
3. **When in doubt, suggest.** If you're unsure whether something is blocking, it's a suggestion.
4. **Read the codebase** to understand context before judging patterns or naming.
5. **Output pure JSON only.** No markdown fences, no explanatory text outside the JSON.

## Your Specialization: {specialization}
"""

_READ_ONLY_TOOLS = ['Read', 'Glob', 'Grep', 'Bash(git:*)']
_JCODEMUNCH_TOOLS = ['mcp__jcodemunch__*']


def build_trial_reviewer_role(spec: ReviewerSpec) -> AgentRole:
    """Build an AgentRole for a trial reviewer.

    Reuses the prompt template structure from roles._reviewer_role()
    (system prompt + output schema + rules + specialization) but
    allows model/budget/effort override per spec.
    """
    return AgentRole(
        name=f'trial_{spec.name}',
        system_prompt=_REVIEWER_SYSTEM_TEMPLATE.format(
            name=spec.name,
            specialization=spec.specialization,
        ),
        allowed_tools=[*_READ_ONLY_TOOLS, *_JCODEMUNCH_TOOLS],
        disallowed_tools=['Edit', 'Write'],
        default_model=spec.model,
        default_budget=spec.budget,
        default_max_turns=30,
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

VARIANT_A = VariantConfig(
    name='variant_a',
    description='1x opus generalist — depth replaces breadth',
    reviewers=[
        ReviewerSpec(
            name='comprehensive_reviewer',
            model='opus',
            specialization=(
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
            ),
            budget=5.0,
            effort='high',
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
