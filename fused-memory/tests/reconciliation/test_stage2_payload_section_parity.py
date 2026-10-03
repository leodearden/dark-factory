"""Parity guard: every Stage-2 payload builder must emit every INFERENCE-BEARING section.

Task 5113, the Stage-2 mirror of task 4708's Stage-1 guard
(``test_stage1_payload_section_parity.py``). Both guards share one structural
predicate, ``reconciliation/payload_section_parity.py``.

THE INVARIANT, stated precisely
-------------------------------
A section is REQUIRED in a Stage-2 payload **iff the shipped Stage-2 prompt
tells the model to draw an inference from that section's ABSENCE**. Today that
set has exactly one member, ``### Live-Workflow Signals``, because
``prompts/stage2.py`` says:

    "If `### Live-Workflow Signals` is absent from the payload, no task is live
     this cycle — neither through a per-task signal nor through the project-wide
     lock — and none has landing evidence; no live-workflow suppression applies."

A builder that omits the section therefore makes the model conclude something
FALSE, which is what makes the section mandatory.

THE STAGE-2 SECTION CENSUS — why every other conditional section stays out:

* Known Projects — keyed on per-PROJECT membership; the prompt asserts the
  local project is a member even when the section is absent, and its no-match
  branch is report-only.
* Stale Flags Requiring Escalation — presence-conditional only; nothing is
  inferred when it is missing.
* Done-Task Completion-Memory Audit, Proactive Task Sample, Tasks Needing
  Memory Hint Attention — omitted on remediation passes BY DESIGN.
* Done-task Provenance — every done task carries an explicit label, so absence
  is never read as evidence.
* Unconditional literal headers — always present; nothing to guard.

ONE BUILDER, ONE STAGE
----------------------
Today ``TaskKnowledgeSync`` has ONE payload builder, ``assemble_payload``, and it
serves both the full and the remediation pass via ``remediation_mode``. The
guard still earns its keep: a second builder, a second payload-returning
branch, or a conditional interpolation is each caught by construction.
``IntegrityCheck.assemble_payload`` lives in the same file but is STAGE 3 and
out of scope — ``STAGE3_SYSTEM_PROMPT`` carries no Live-Workflow
absence-inference — which is why discovery is scoped to the
``TaskKnowledgeSync`` class body rather than to the module.

NO PROMPT-REGEX CROSS-CHECK. The prompt <-> registry correspondence is
maintained by hand and pinned as a VALUE, for the reason recorded in the
Stage-1 guard's module docstring; it is not restated here.
"""

from __future__ import annotations

from fused_memory.reconciliation.stages.base import RequiredSection
from fused_memory.reconciliation.stages.task_knowledge_sync import TaskKnowledgeSync


class TestRequiredSectionsRegistry:
    """``TaskKnowledgeSync.REQUIRED_SECTIONS`` is the single declared section set."""

    def test_registry_is_a_non_empty_tuple_of_required_sections(self):
        registry = TaskKnowledgeSync.REQUIRED_SECTIONS
        assert isinstance(registry, tuple), (
            f'TaskKnowledgeSync.REQUIRED_SECTIONS must be a tuple (immutable — it '
            f'is a class-level declaration read by every Stage-2 payload builder), '
            f'got {type(registry).__name__}.'
        )
        assert registry, (
            'TaskKnowledgeSync.REQUIRED_SECTIONS is empty. An empty registry makes '
            'the Stage-2 aggregator a no-op and every parity assertion in this file '
            "vacuous; at least '### Live-Workflow Signals' is required by "
            'prompts/stage2.py.'
        )
        for section in registry:
            assert isinstance(section, RequiredSection), (
                f'TaskKnowledgeSync.REQUIRED_SECTIONS member {section!r} is a '
                f'{type(section).__name__}, not a stages.base.RequiredSection. Both '
                f'stages declare their registries with that ONE shared type; do not '
                f'define a Stage-2 copy.'
            )
            assert isinstance(section.header, str) and section.header.startswith('### '), (
                f'TaskKnowledgeSync.REQUIRED_SECTIONS member {section!r} has header '
                f'{section.header!r}; it must be the exact markdown header '
                f"prompts/stage2.py names, which is a level-3 heading ('### …')."
            )
            assert isinstance(section.renderer, str), (
                f'TaskKnowledgeSync.REQUIRED_SECTIONS member {section!r} has renderer '
                f'{section.renderer!r}; it must be the NAME of a TaskKnowledgeSync '
                f'method (a str resolved via getattr), not the method object — the '
                f'aggregator dispatches by name.'
            )

    def test_registry_contains_the_live_workflow_section(self):
        registry = TaskKnowledgeSync.REQUIRED_SECTIONS
        expected = RequiredSection('### Live-Workflow Signals', '_build_live_workflow_section')
        assert expected in registry, (
            f'TaskKnowledgeSync.REQUIRED_SECTIONS must contain {expected!r}; got '
            f'{[(s.header, s.renderer) for s in registry]!r}. This is the one '
            f'section prompts/stage2.py draws an absence-inference from, so a '
            f'Stage-2 payload omitting it makes the model conclude no task is live '
            f'or landed this cycle and that no live-workflow suppression applies.'
        )

    def test_every_registry_renderer_resolves_to_a_method(self):
        for section in TaskKnowledgeSync.REQUIRED_SECTIONS:
            renderer = getattr(TaskKnowledgeSync, section.renderer, None)
            assert callable(renderer), (
                f'TaskKnowledgeSync.REQUIRED_SECTIONS member {section.header!r} names '
                f'renderer {section.renderer!r}, which is not a callable attribute of '
                f'TaskKnowledgeSync. A typo in the renderer string must fail HERE, '
                f'not as an AttributeError at Stage-2 payload-assembly time in '
                f'production.'
            )
