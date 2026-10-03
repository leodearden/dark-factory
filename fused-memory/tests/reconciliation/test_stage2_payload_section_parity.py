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

import pathlib
from unittest.mock import AsyncMock

import pytest
from _ast_guard import calls_named

import fused_memory.reconciliation.stages.task_knowledge_sync as task_knowledge_sync_module
from fused_memory.config.schema import ReconciliationConfig
from fused_memory.models.reconciliation import StageId
from fused_memory.models.scope import ProjectId, ProjectRoot, ProjectScope
from fused_memory.reconciliation.stages.base import RequiredSection
from fused_memory.reconciliation.stages.task_knowledge_sync import TaskKnowledgeSync
from fused_memory.reconciliation.task_filter import FilteredTaskTree
from reconciliation.payload_section_parity import (
    AGGREGATOR,
    PAYLOAD_HEADER_PREFIX,
    branches_missing_the_aggregator,
    discover_payload_builders,
)

TKS_SRC = pathlib.Path(task_knowledge_sync_module.__file__)
STAGE2_CLASS = 'TaskKnowledgeSync'

STAGE2_PAYLOAD_BUILDERS = discover_payload_builders(TKS_SRC, STAGE2_CLASS)


def _make_stage() -> TaskKnowledgeSync:
    """A mock-backed Stage-2 stage, built here rather than imported from test_stages.py."""
    return TaskKnowledgeSync(
        StageId.task_knowledge_sync,
        memory_service=AsyncMock(),
        taskmaster=AsyncMock(),
        journal=AsyncMock(),
        config=ReconciliationConfig(),
        scope=ProjectScope(ProjectId('test_project'), ProjectRoot('/project')),
    )


def _make_tree(tasks: list[dict]) -> FilteredTaskTree:
    """Build a FilteredTaskTree with the given tasks as active_tasks."""
    return FilteredTaskTree(
        active_tasks=tasks,
        done_tasks=[],
        cancelled_tasks=[],
        done_count=0,
        cancelled_count=0,
        other_count=0,
        total_count=len(tasks),
        max_task_id=max((t.get('id', 0) for t in tasks), default=0),
    )


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


class TestRenderRequiredSections:
    """``_render_required_sections(filtered)`` is the one aggregator every Stage-2 builder consumes.

    Driven two ways, as the Stage-1 guard drives its aggregator:

    * WHAT IT DISPATCHES TO — a real live-workflow fixture (the detector
      monkeypatched at its home namespace in ``live_workflow_section``), so the
      assertion runs against real renderer output.
    * HOW IT ASSEMBLES — a two-member STAND-IN registry of stub renderers,
      because order, the absence of a separator, and the pass-through of
      *filtered* are not falsifiable against a real registry holding one member.
    """

    @pytest.mark.asyncio
    async def test_renders_every_registry_header_when_all_sections_apply(self, monkeypatch):
        import fused_memory.reconciliation.live_workflow_section as lws_module
        from fused_memory.services.live_workflow_detector import WorkflowLiveness

        live_task_id = '4321'

        async def _fake_detect(task_id, project_root, **kwargs):
            return WorkflowLiveness(
                is_live=str(task_id) == live_task_id,
                worktree_registered=str(task_id) == live_task_id,
                recent_commit=False,
                orchestrator_live=False,
                branch=f'task/{task_id}',
                last_commit_at=None,
            )

        monkeypatch.setattr(lws_module, 'detect_live_workflow', _fake_detect)
        stage = _make_stage()
        filtered = _make_tree(
            [
                {'id': int(live_task_id), 'title': 'Live task', 'status': 'in-progress'},
                {'id': 100, 'title': 'Other task', 'status': 'blocked'},
            ]
        )

        rendered = await stage._render_required_sections(filtered)

        for section in TaskKnowledgeSync.REQUIRED_SECTIONS:
            assert section.header in rendered, (
                f'TaskKnowledgeSync._render_required_sections(filtered) omitted '
                f'{section.header!r} even though its renderer {section.renderer!r} '
                f'applies. Either the registry header no longer matches what the '
                f'renderer emits, or the aggregator is not dispatching to it. '
                f'Rendered:\n{rendered!r}'
            )

    @pytest.mark.asyncio
    async def test_output_is_the_concatenation_of_the_registry_renderers_in_order(self, monkeypatch):
        """Registry order, no separator, no post-processing, *filtered* passed through.

        Driven through a TWO-member STAND-IN registry of stub renderers rather
        than the real one. Against the real single-member registry "in order" is
        vacuous, and the natural expected value would be the production line
        copied verbatim, agreeing with any implementation written the same way.
        Two distinguishable sentinels make the assembly contract falsifiable,
        and recording each stub's argument pins Stage 2's extra contract over
        Stage 1's zero-arg aggregator: every renderer receives the SAME resolved
        tree the builder handed to the aggregator.
        """
        stage = _make_stage()
        received: dict[str, FilteredTaskTree] = {}

        async def _stub_alpha(filtered: FilteredTaskTree) -> str:
            received['alpha'] = filtered
            return '\n### Alpha\nfirst\n'

        async def _stub_beta(filtered: FilteredTaskTree) -> str:
            received['beta'] = filtered
            return '\n### Beta\nsecond\n'

        # Instance attributes, so getattr(self, section.renderer)(filtered)
        # dispatches to them exactly as it does to real bound methods.
        monkeypatch.setattr(stage, '_stub_alpha', _stub_alpha, raising=False)
        monkeypatch.setattr(stage, '_stub_beta', _stub_beta, raising=False)
        monkeypatch.setattr(
            TaskKnowledgeSync,
            'REQUIRED_SECTIONS',
            (
                RequiredSection('### Alpha', '_stub_alpha'),
                RequiredSection('### Beta', '_stub_beta'),
            ),
        )
        filtered = _make_tree([{'id': 7, 'title': 'Any task', 'status': 'pending'}])

        rendered = await stage._render_required_sections(filtered)

        expected = '\n### Alpha\nfirst\n\n### Beta\nsecond\n'
        assert rendered == expected, (
            "The Stage-2 aggregator must be exactly its registry renderers' output "
            'concatenated in REGISTRY ORDER — it may not add a separator, '
            'reorder, or post-process. Each renderer already owns its own '
            f'leading newline.\n  got:      {rendered!r}\n'
            f'  expected: {expected!r}'
        )
        assert received.get('alpha') is filtered and received.get('beta') is filtered, (
            'Every Stage-2 registry renderer must receive the SAME FilteredTaskTree '
            'the payload builder passed to _render_required_sections — the '
            "builder's resolved tree may be self-fetched rather than "
            f'self.filtered_task_tree. Received: {received!r}'
        )

    @pytest.mark.asyncio
    async def test_returns_empty_string_when_no_section_applies(self):
        stage = _make_stage()

        rendered = await stage._render_required_sections(_make_tree([]))

        assert rendered == '', (
            "TaskKnowledgeSync._render_required_sections must return '' for a tree "
            "with no active tasks — each renderer keeps its own conditional-empty "
            f"contract and '' is a normal result. Got: {rendered!r}"
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
