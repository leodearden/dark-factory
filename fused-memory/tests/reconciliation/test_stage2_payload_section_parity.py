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
* Done-Task Completion-Memory Audit, Blocked Gate-Task Review Audit, Proactive
  Task Sample, Tasks Needing Memory Hint Attention — omitted on remediation
  passes BY DESIGN.
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

The aggregator's ASSEMBLY contract — registry order, no separator, the
payload's resolved tree reaching every renderer — is pinned through the public
``assemble_payload`` in ``tests/test_stages.py::TestAssemblePayloadRequiredSections``
rather than by calling the private aggregator from here.
"""

from __future__ import annotations

import pathlib

import pytest
from _ast_guard import calls_named

import fused_memory.reconciliation.stages.task_knowledge_sync as task_knowledge_sync_module
from fused_memory.reconciliation.stages.base import RequiredSection
from fused_memory.reconciliation.stages.task_knowledge_sync import TaskKnowledgeSync
from reconciliation.payload_section_parity import (
    AGGREGATOR,
    PAYLOAD_HEADER_PREFIX,
    branches_missing_the_aggregator,
    discover_payload_builders,
    registry_shape_violations,
)

TKS_SRC = pathlib.Path(task_knowledge_sync_module.__file__)
STAGE2_CLASS = 'TaskKnowledgeSync'

STAGE2_PAYLOAD_BUILDERS = discover_payload_builders(TKS_SRC, STAGE2_CLASS)


class TestRequiredSectionsRegistry:
    """``TaskKnowledgeSync.REQUIRED_SECTIONS`` is the single declared section set."""

    def test_registry_is_well_formed(self):
        violations = registry_shape_violations(TaskKnowledgeSync)
        assert not violations, '\n'.join(violations)

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


class TestDiscoveryItself:
    """The predicate must keep finding the Stage-2 builder we verified by hand.

    Without this floor, a predicate that silently stopped matching — the
    payload f-string hoisted to a module constant, the top-level header
    reworded — would parametrize over an EMPTY set and report green having
    checked nothing. Same anti-vacuity semantics as the Stage-1 guard's
    ``TestDiscoveryItself``.

    The shared predicate (``reconciliation/payload_section_parity.py``) was
    re-verified against ``task_knowledge_sync.py``:

    * SELECTS ``TaskKnowledgeSync.assemble_payload``
      ('## Stage 2: Task-Knowledge Sync').
    * REJECTS ``_format_known_projects_section`` (returns a ``Constant`` or a
      ``BinOp``), ``_build_live_workflow_section`` (returns an ``Await``) and
      ``_render_required_sections`` (returns a ``Call``).
    * Never scans ``IntegrityCheck.assemble_payload`` ('## Stage 3: …'),
      because discovery is scoped to the ``TaskKnowledgeSync`` class body.
    """

    # A LOWER bound, never an equality: a second builder must be pulled INTO
    # scope automatically, not reported here as a floor violation to edit away.
    VERIFIED_STAGE2_PAYLOAD_BUILDERS = {'assemble_payload'}

    def test_discovery_finds_the_verified_payload_builders(self):
        discovered = set(STAGE2_PAYLOAD_BUILDERS)
        missing = self.VERIFIED_STAGE2_PAYLOAD_BUILDERS - discovered
        assert not missing, (
            f'Stage-2 payload-builder discovery no longer finds {sorted(missing)} '
            f'(found {sorted(discovered) or "nothing"}). The predicate in '
            f'payload_section_parity.discover_payload_builders has gone stale — '
            f'most likely a payload f-string was hoisted to a module constant or '
            f'built by concatenation, or the top-level {PAYLOAD_HEADER_PREFIX!r} '
            f'header was reworded. Fix the PREDICATE; do NOT shrink this floor, '
            f'and do NOT turn it into an equality: an introspective guard that '
            f'discovers nothing passes having checked nothing.'
        )


@pytest.mark.parametrize(
    'builder_name', sorted(STAGE2_PAYLOAD_BUILDERS), ids=lambda name: name
)
class TestEveryPayloadBuilderRendersTheRequiredSections:
    """Every DISCOVERED Stage-2 payload builder routes through the one aggregator.

    AST, never string grep: a docstring that merely mentions
    ``_render_required_sections`` must not satisfy the guard.
    """

    def test_builder_interpolates_the_required_sections_aggregator(self, builder_name):
        """EVERY payload-returning branch of the builder must carry the sections.

        A conditional interpolation (an ``IfExp`` around the aggregator call)
        never matches the aggregator-call predicate, so it fails here too.
        """
        builder = STAGE2_PAYLOAD_BUILDERS[builder_name]

        offending_lines = branches_missing_the_aggregator(builder)

        assert not offending_lines, (
            f'{STAGE2_CLASS}.{builder_name} returns a Stage-2 payload that never '
            f'interpolates self.{AGGREGATOR}(filtered) at '
            f'{TKS_SRC.name}:{",".join(str(n) for n in offending_lines)} '
            f'(of {len(builder.payload_returns)} payload-returning branch(es) in '
            f'this builder), so that branch can omit '
            f'{sorted(s.header for s in TaskKnowledgeSync.REQUIRED_SECTIONS)} — '
            f"and prompts/stage2.py's absence-inference then makes the model "
            f'conclude no task is live. Fix: interpolate '
            f'{{await self.{AGGREGATOR}(filtered)}} into the returned f-string, or '
            f'bind it to a local and interpolate that local.'
        )

    def test_builder_does_not_call_a_registry_renderer_directly(self, builder_name):
        """No builder may keep its own hand-wired call to a registered renderer.

        A builder that still called ``self._build_live_workflow_section(…)``
        itself would satisfy the test above while making "adding a section is
        one edit" false again. The aggregator dispatches via
        ``getattr(self, section.renderer)(filtered)``, which is not a named call
        and so is unaffected.
        """
        builder = STAGE2_PAYLOAD_BUILDERS[builder_name]

        for section in TaskKnowledgeSync.REQUIRED_SECTIONS:
            assert not calls_named(builder.node, section.renderer), (
                f'{STAGE2_CLASS}.{builder_name} calls self.{section.renderer}() '
                f'directly. Registered sections must be reached ONLY through '
                f'self.{AGGREGATOR}(filtered), or adding the next section is once '
                f'again one edit per builder rather than one edit to '
                f'REQUIRED_SECTIONS. Delete the direct call and the local it feeds.'
            )
