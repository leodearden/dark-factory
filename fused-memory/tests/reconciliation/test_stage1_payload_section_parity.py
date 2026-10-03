"""Parity guard: every Stage-1 payload builder must emit every INFERENCE-BEARING section.

Task 4708. This closes a defect that has now recurred three times — tasks 2150,
2552 and 3839 each hand-fixed ONE missing (builder × section) cell and each left
the mechanism intact. ``MemoryConsolidator`` has three payload builders
(``assemble_payload``, ``_format_assembled_payload``,
``_assemble_remediation_payload``) and several section builders; every new
section had to be wired into all three by hand, and nothing noticed when it was
not.

THE INVARIANT, stated precisely
-------------------------------
A section is REQUIRED in a Stage-1 payload **iff the shipped Stage-1 prompt
tells the model to draw an inference from that section's ABSENCE**. Today that
set has exactly one member, ``### Live-Workflow Signals``, because
``prompts/stage1.py`` says:

    "If `### Live-Workflow Signals` is absent from the payload, no task is live
     this cycle — neither through a per-task signal nor through the project-wide
     lock — and none has landing evidence; no live-workflow suppression applies …"

That sentence makes absence load-bearing: a builder that omits the section is
not merely terser, it makes the model conclude something FALSE. That — not
symmetry for its own sake — is what makes the section mandatory.

That correspondence between prompt and registry is maintained BY HAND — a
human-checked inclusion criterion, recorded once in the ``REQUIRED_SECTIONS``
comment in ``memory_consolidator.py`` — and is deliberately NOT
machine-asserted. Parsing the prompt for the sentence above pins its WORDING,
not its contract: the sentence can be reworded with the header identifier and
the payload contract fully intact ("… does not appear in the payload …") and
any such pattern then reports spurious drift, while a genuinely new
absence-inference phrased differently is silently missed. An earlier revision
of this file asserted exactly that, in both directions, and it was deleted for
this reason. Do not re-add it, and do not replace it with a tighter pattern —
that is the same defect with a smaller blast radius. The required set is
pinned instead as a VALUE, by
:meth:`TestRequiredSectionsRegistry.test_registry_contains_the_live_workflow_section`.

"All builders emit the same set" is the WRONG invariant and would force real
regressions. Deliberately NOT swept in:

* ``_build_task_tree_section`` — 2 of 3 by design. Adding it to the
  findings-only remediation payload would dump the whole task tree into a
  payload whose entire point is to be focused.
* ``_build_task_count_census_section`` — 2 of 3 by design. Its own contract
  returns ``''`` when ``task_count_verification is None``, which is exactly the
  remediation-pass state, so "adding" it would be a no-op dressed as a fix.
* ``_build_project_root_directive`` — this one IS required in all three
  builders, but for a different reason: it is an unconditional directive with
  no absence-inference attached anywhere in the prompts. Registering it would
  satisfy an intuition about "things all builders need" while falling outside
  this registry's stated inclusion criterion — the registry would hold a
  member the prompt never infers from. It stays covered by its own dedicated
  tests in ``tests/reconciliation/test_stage1.py`` (task 2552).

WHY THIS IS NOT A FOURTH INSTANCE OF THE DEFECT
-----------------------------------------------
The three prior fixes each added a per-builder assertion naming one builder and
one section by hand, so the next new builder or section was invisible until
someone remembered. Here:

* the SECTION set is DECLARED ONCE in ``MemoryConsolidator.REQUIRED_SECTIONS``
  and consumed by all builders through one aggregator — adding a section is a
  single edit, not one edit per builder;
* the BUILDER set is DERIVED by AST introspection over the class body
  (``reconciliation/payload_section_parity.py``, shared with the Stage-2
  guard), so a fourth builder is in scope the day it is added with no edit
  here — and
  EVERY payload-returning branch of each discovered builder is checked, not
  just the first one found, so a second return added to an existing builder
  is in scope the same way.

The only hand-written literals are the registry itself (the intended single
edit point) and a discovery FLOOR, which can only ever be too small — and being
too small fails loudly.
"""

from __future__ import annotations

import pathlib

import pytest
from _ast_guard import calls_named

import fused_memory.reconciliation.stages.memory_consolidator as consolidator_module
from fused_memory.reconciliation.stages.base import RequiredSection
from fused_memory.reconciliation.stages.memory_consolidator import MemoryConsolidator
from fused_memory.reconciliation.task_filter import FilteredTaskTree
from reconciliation.consolidator_fixtures import make_consolidator
from reconciliation.payload_section_parity import (
    AGGREGATOR,
    PAYLOAD_HEADER_PREFIX,
    branches_missing_the_aggregator,
    discover_payload_builders,
    registry_shape_violations,
)

CONSOLIDATOR_SRC = pathlib.Path(consolidator_module.__file__)
CONSOLIDATOR_CLASS = 'MemoryConsolidator'

STAGE1_PAYLOAD_BUILDERS = discover_payload_builders(CONSOLIDATOR_SRC, CONSOLIDATOR_CLASS)


class TestRequiredSectionsRegistry:
    """``MemoryConsolidator.REQUIRED_SECTIONS`` is the single declared section set."""

    def test_registry_is_well_formed(self):
        violations = registry_shape_violations(MemoryConsolidator)
        assert not violations, '\n'.join(violations)

    def test_registry_contains_the_live_workflow_section(self):
        registry = MemoryConsolidator.REQUIRED_SECTIONS
        assert any(
            s.header == '### Live-Workflow Signals'
            and s.renderer == '_build_live_workflow_section'
            for s in registry
        ), (
            f'REQUIRED_SECTIONS must contain '
            f"RequiredSection('### Live-Workflow Signals', '_build_live_workflow_section'); "
            f'got {[(s.header, s.renderer) for s in registry]!r}. This is the one '
            f'section prompts/stage1.py draws an absence-inference from, so a builder '
            f'omitting it makes the model conclude no task is live or landed.'
        )


class TestRenderRequiredSections:
    """``_render_required_sections`` is the one aggregator all builders consume.

    Two complementary drives, because the aggregator has two separable
    contracts:

    * WHAT IT DISPATCHES TO — driven with a real live-workflow fixture (the
      detector monkeypatched at its home namespace in ``live_workflow_section``,
      the established spelling) so the assertion runs against real renderer
      output rather than the empty strings every renderer returns when its
      guard fails.
    * HOW IT ASSEMBLES — driven with a two-member STAND-IN registry of stub
      renderers, because order and the absence of a separator are not
      falsifiable against a real registry holding one member.
    """

    def _make_tree(self, tasks: list[dict]) -> FilteredTaskTree:
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

    def _make_live_stage(self, monkeypatch) -> MemoryConsolidator:
        """A consolidator whose every registry section actually renders."""
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

        stage = make_consolidator(project_root='/project')
        stage.filtered_task_tree = self._make_tree(
            [
                {'id': int(live_task_id), 'title': 'Live task', 'status': 'in-progress'},
                {'id': 100, 'title': 'Other task', 'status': 'blocked'},
            ]
        )
        return stage

    @pytest.mark.asyncio
    async def test_renders_every_registry_header_when_all_sections_apply(self, monkeypatch):
        stage = self._make_live_stage(monkeypatch)

        rendered = await stage._render_required_sections()

        for section in MemoryConsolidator.REQUIRED_SECTIONS:
            assert section.header in rendered, (
                f'_render_required_sections() omitted {section.header!r} even though '
                f'its renderer {section.renderer!r} applies. Either the registry '
                f'header no longer matches what the renderer emits, or the '
                f'aggregator is not dispatching to it. Rendered:\n{rendered!r}'
            )

    @pytest.mark.asyncio
    async def test_output_is_the_concatenation_of_the_registry_renderers_in_order(self, monkeypatch):
        """Registry order, no separator, no post-processing — pinned as literal text.

        Driven through a TWO-member STAND-IN registry of stub renderers rather
        than the real one. Against the real single-member registry "in order" is
        vacuous and "no separator" only incidentally covered, and the natural
        expected value — ``''.join(getattr(stage, s.renderer)() for s in
        REQUIRED_SECTIONS)`` — is the production line copied verbatim, so it
        would agree with any implementation written the same way and check
        nothing. Two distinguishable sentinels make the contract falsifiable:
        a separator, a reorder, a strip or any other post-processing each
        produces a string different from the one asserted below.

        The stand-in is deliberately NOT the real registry: this test owns the
        aggregator's ASSEMBLY contract. That the real registry's members render
        their real headers is pinned by
        :meth:`test_renders_every_registry_header_when_all_sections_apply`.
        """
        stage = make_consolidator(project_root='/project')
        # Instance attributes, so getattr(self, section.renderer)() dispatches to
        # them exactly as it does to real bound methods.
        async def _stub_alpha() -> str:
            return '\n### Alpha\nfirst\n'

        async def _stub_beta() -> str:
            return '\n### Beta\nsecond\n'

        monkeypatch.setattr(stage, '_stub_alpha', _stub_alpha, raising=False)
        monkeypatch.setattr(stage, '_stub_beta', _stub_beta, raising=False)
        monkeypatch.setattr(
            MemoryConsolidator,
            'REQUIRED_SECTIONS',
            (
                RequiredSection('### Alpha', '_stub_alpha'),
                RequiredSection('### Beta', '_stub_beta'),
            ),
        )

        rendered = await stage._render_required_sections()

        expected = '\n### Alpha\nfirst\n\n### Beta\nsecond\n'
        assert rendered == expected, (
            "The aggregator must be exactly its registry renderers' output "
            'concatenated in REGISTRY ORDER — it may not add a separator, '
            'reorder, or post-process. Each renderer already owns its own '
            f'leading newline.\n  got:      {rendered!r}\n'
            f'  expected: {expected!r}'
        )

    @pytest.mark.asyncio
    async def test_returns_empty_string_when_no_section_applies(self):
        # filtered_task_tree left at the make_consolidator default (None), so
        # every registry renderer's guard fails.
        stage = make_consolidator(project_root='/project')

        rendered = await stage._render_required_sections()

        assert rendered == '', (
            "_render_required_sections() must return '' when no section applies — "
            "each renderer keeps its own conditional-empty contract and '' is a "
            f'normal result, keeping the payload tight. Got: {rendered!r}'
        )


class TestDiscoveryItself:
    """The predicate must keep finding the builders we verified by hand.

    Without this floor, a guard whose predicate silently stopped matching — a
    payload f-string hoisted to a module constant, the top-level header
    reworded — would parametrize over an EMPTY set and report green having
    checked nothing. That silent failure is strictly worse than the
    hand-maintained assertions this guard replaces, so the floor converts it
    into a loud one. Mirrors ``tests/test_falkor_index_barrier_guard.py``'s
    ``TestDiscoveryItself`` / ``VERIFIED_LIVE_INDEX_MODULES`` so the two guards
    agree on anti-vacuity semantics by shared precedent.

    The shared predicate (``reconciliation/payload_section_parity.py``) — a
    method directly in the class body returning an f-string that opens with
    ``'## '`` — was verified against ``memory_consolidator.py``:

    * SELECTS ``assemble_payload`` ('## Reconciliation Run — Stage 1: …'),
      ``_format_assembled_payload`` (same header) and
      ``_assemble_remediation_payload`` ('## Remediation Run — Stage 1: …').
    * REJECTS the section builders. ``_build_task_count_census_section``
      ('\\n### Task Count Census …') and ``_build_project_root_directive``
      ('\\nUse project_root="…') do return f-strings, but neither opens with
      ``'## '``; ``_build_task_tree_section`` and
      ``_build_live_workflow_section`` return ``BinOp`` and cannot match at all;
      the aggregator returns a ``Call``.
    """

    # A LOWER bound, never an equality: a fourth builder must be pulled INTO
    # scope automatically, not reported here as a floor violation to edit away.
    VERIFIED_STAGE1_PAYLOAD_BUILDERS = {
        'assemble_payload',
        '_format_assembled_payload',
        '_assemble_remediation_payload',
    }

    def test_discovery_finds_the_verified_payload_builders(self):
        discovered = set(STAGE1_PAYLOAD_BUILDERS)
        missing = self.VERIFIED_STAGE1_PAYLOAD_BUILDERS - discovered
        assert not missing, (
            f'Stage-1 payload-builder discovery no longer finds {sorted(missing)} '
            f'(found {sorted(discovered) or "nothing"}). The predicate in '
            f'payload_section_parity.discover_payload_builders has gone stale — most likely a '
            f'payload f-string was hoisted to a module constant or built by '
            f'concatenation, or the top-level {PAYLOAD_HEADER_PREFIX!r} header was '
            f'reworded. Fix the PREDICATE; do NOT shrink this floor, and do NOT '
            f'turn it into an equality: an introspective guard that discovers '
            f'nothing passes having checked nothing, which is the failure this '
            f'floor exists to prevent.'
        )


@pytest.mark.parametrize(
    'builder_name', sorted(STAGE1_PAYLOAD_BUILDERS), ids=lambda name: name
)
class TestEveryPayloadBuilderRendersTheRequiredSections:
    """Every DISCOVERED payload builder routes through the one aggregator.

    One assertion per builder rather than one per (builder × section) pair —
    that combinatorial growth is what produced four hand-maintained assertion
    classes across tasks 2150/2552/3839 and still left cells uncovered.

    AST, never string grep: a docstring that merely mentions
    ``_render_required_sections`` must not satisfy the guard.
    """

    def test_builder_interpolates_the_required_sections_aggregator(self, builder_name):
        """EVERY payload-returning branch of the builder must carry the sections.

        Not just one of them: a builder that grows a second payload return —
        an early-return short payload, say — must have that branch checked too,
        or it becomes a fresh place for a required section to go missing.
        """
        builder = STAGE1_PAYLOAD_BUILDERS[builder_name]

        offending_lines = branches_missing_the_aggregator(builder)

        assert not offending_lines, (
            f'{CONSOLIDATOR_CLASS}.{builder_name} returns a Stage-1 payload that '
            f'never interpolates self.{AGGREGATOR}() at '
            f'{CONSOLIDATOR_SRC.name}:{",".join(str(n) for n in offending_lines)} '
            f'(of {len(builder.payload_returns)} payload-returning branch(es) in this '
            f'builder), so that branch can omit '
            f'{sorted(s.header for s in MemoryConsolidator.REQUIRED_SECTIONS)} — '
            f"and the prompt's absence-inference then makes the model conclude "
            f'something FALSE rather than merely reading a terser payload. Fix: '
            f'interpolate {{self.{AGGREGATOR}()}} into the returned f-string '
            f'(the spelling already used for '
            f'{{self._build_project_root_directive()}} in this file); assigning it '
            f'to a local and interpolating that local is equally accepted.'
        )

    def test_builder_does_not_call_a_registry_renderer_directly(self, builder_name):
        """No builder may keep its own hand-wired call to a registered renderer.

        This closes the obvious bypass. A builder that still called
        ``self._build_live_workflow_section()`` itself would satisfy the test
        above while silently making "adding a section is one edit" false again —
        that IS the 2150/2552/3839 drift, re-expressed. The aggregator dispatches
        via ``getattr(self, section.renderer)()``, which is not a named call and
        so is unaffected; it is also not a discovered builder.
        """
        builder = STAGE1_PAYLOAD_BUILDERS[builder_name]

        for section in MemoryConsolidator.REQUIRED_SECTIONS:
            assert not calls_named(builder.node, section.renderer), (
                f'{CONSOLIDATOR_CLASS}.{builder_name} calls '
                f'self.{section.renderer}() directly. Registered sections must be '
                f'reached ONLY through self.{AGGREGATOR}(), or adding the next '
                f'section is once again one edit per builder rather than one edit '
                f'to REQUIRED_SECTIONS — the exact drift behind tasks 2150, 2552 '
                f'and 3839. Delete the direct call and the local it feeds.'
            )
