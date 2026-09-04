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

    "If `### Live-Workflow Signals` is absent from the payload, all three
     signals are False for every task; no live-workflow suppression applies …"

That sentence makes absence load-bearing: a builder that omits the section is
not merely terser, it makes the model conclude something FALSE. That — not
symmetry for its own sake — is what makes the section mandatory.

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
  satisfy an intuition about "things all builders need" while making
  :class:`TestRegistryMatchesPromptAbsenceInference` unsatisfiable by
  construction — the registry would hold a member the prompt never infers
  from. It stays covered by its own dedicated tests in
  ``tests/reconciliation/test_stage1.py`` (task 2552).

WHY THIS IS NOT A FOURTH INSTANCE OF THE DEFECT
-----------------------------------------------
The three prior fixes each added a per-builder assertion naming one builder and
one section by hand, so the next new builder or section was invisible until
someone remembered. Here:

* the SECTION set is DECLARED ONCE in ``MemoryConsolidator.REQUIRED_SECTIONS``
  and consumed by all builders through one aggregator — adding a section is a
  single edit, not one edit per builder;
* the BUILDER set is DERIVED by AST introspection over the class body, so a
  fourth builder is in scope the day it is added with no edit here;
* the section set is cross-checked against the prompt that makes it required,
  in both directions.

The only hand-written literals are the registry itself (the intended single
edit point) and a discovery FLOOR, which can only ever be too small — and being
too small fails loudly.

Scope note: :data:`STAGE1_SYSTEM_PROMPT` only. ``prompts/stage2.py`` carries
the byte-identical absence-inference sentence, but it describes Stage 2's own
payload, built by two separate ``assemble_payload`` methods in
``task_knowledge_sync.py`` that this task deliberately does not govern.
Asserting over stage2 here would couple Stage-1's registry to Stage-2 prompt
drift.
"""

from __future__ import annotations

import re

from fused_memory.reconciliation.prompts.stage1 import STAGE1_SYSTEM_PROMPT
from fused_memory.reconciliation.stages.memory_consolidator import MemoryConsolidator
from fused_memory.reconciliation.task_filter import FilteredTaskTree
from reconciliation.test_stage1 import _make_consolidator

# The machine-readable shape of the prompt's absence-inference sentence. Applied
# to the IMPORTED runtime string rather than raw source, so the prompt's
# backslash line-continuations are already joined and the pattern matches the
# text the model actually receives. Verified to yield exactly
# ['### Live-Workflow Signals'] against the current prompt.
_ABSENCE_INFERENCE = re.compile(r'`(###[^`\n]{1,80})`[^.`]{0,80}?\bis absent from the payload')

_REGEX_REPAIR_HINT = (
    f'The pattern is {_ABSENCE_INFERENCE.pattern!r}, applied to the imported '
    f'STAGE1_SYSTEM_PROMPT. Two repairs are possible and they are NOT '
    f'interchangeable: if the prompt sentence was merely REWORDED, fix the '
    f'pattern in this file; if a NEW absence-inference was added to the prompt, '
    f'add that section to MemoryConsolidator.REQUIRED_SECTIONS so the inference '
    f'is actually sound.'
)


def _prompt_absence_inference_headers() -> set[str]:
    """Every ``### …`` header STAGE1_SYSTEM_PROMPT attaches an absence-inference to."""
    return set(_ABSENCE_INFERENCE.findall(STAGE1_SYSTEM_PROMPT))


class TestRequiredSectionsRegistry:
    """``MemoryConsolidator.REQUIRED_SECTIONS`` is the single declared section set."""

    def test_registry_is_a_non_empty_tuple_of_required_sections(self):
        registry = MemoryConsolidator.REQUIRED_SECTIONS
        assert isinstance(registry, tuple), (
            f'REQUIRED_SECTIONS must be a tuple (immutable — it is a class-level '
            f'declaration read by every payload builder), got {type(registry).__name__}.'
        )
        assert registry, (
            'REQUIRED_SECTIONS is empty. An empty registry makes the aggregator a '
            'no-op and every parity assertion in this file vacuous; at least '
            "'### Live-Workflow Signals' is required by prompts/stage1.py."
        )
        for section in registry:
            assert isinstance(section.header, str) and section.header.startswith('### '), (
                f'REQUIRED_SECTIONS member {section!r} has header {section.header!r}; '
                f"it must be the exact markdown header the shipped prompt names, "
                f"which is a level-3 heading ('### …')."
            )
            assert isinstance(section.renderer, str), (
                f'REQUIRED_SECTIONS member {section!r} has renderer '
                f'{section.renderer!r}; it must be the NAME of a MemoryConsolidator '
                f'method (a str resolved via getattr), not the method object — the '
                f'aggregator dispatches by name.'
            )

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
            f'omitting it makes the model conclude all three liveness signals are False.'
        )

    def test_every_registry_renderer_resolves_to_a_method(self):
        for section in MemoryConsolidator.REQUIRED_SECTIONS:
            renderer = getattr(MemoryConsolidator, section.renderer, None)
            assert callable(renderer), (
                f'REQUIRED_SECTIONS member {section.header!r} names renderer '
                f'{section.renderer!r}, which is not a callable attribute of '
                f'MemoryConsolidator. A typo in the renderer string must fail HERE, '
                f'not as an AttributeError at payload-assembly time in production.'
            )


class TestRenderRequiredSections:
    """``_render_required_sections`` is the one aggregator all builders consume.

    Driven with a real live-workflow fixture (the detector monkeypatched at its
    home namespace in ``task_knowledge_sync``, the established spelling) so
    these assertions run against real renderer output rather than the empty
    strings every renderer returns when its guard fails.
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
        import fused_memory.reconciliation.stages.task_knowledge_sync as tks_module
        from fused_memory.services.live_workflow_detector import WorkflowLiveness

        live_task_id = '4321'

        def _fake_detect(task_id, project_root, **kwargs):
            return WorkflowLiveness(
                is_live=str(task_id) == live_task_id,
                worktree_registered=str(task_id) == live_task_id,
                recent_commit=False,
                orchestrator_live=False,
                branch=f'task/{task_id}',
                last_commit_at=None,
            )

        monkeypatch.setattr(tks_module, 'detect_live_workflow', _fake_detect)

        stage = _make_consolidator(project_root='/project')
        stage.filtered_task_tree = self._make_tree(
            [
                {'id': int(live_task_id), 'title': 'Live task', 'status': 'in-progress'},
                {'id': 100, 'title': 'Other task', 'status': 'blocked'},
            ]
        )
        return stage

    def test_renders_every_registry_header_when_all_sections_apply(self, monkeypatch):
        stage = self._make_live_stage(monkeypatch)

        rendered = stage._render_required_sections()

        for section in MemoryConsolidator.REQUIRED_SECTIONS:
            assert section.header in rendered, (
                f'_render_required_sections() omitted {section.header!r} even though '
                f'its renderer {section.renderer!r} applies. Either the registry '
                f'header no longer matches what the renderer emits, or the '
                f'aggregator is not dispatching to it. Rendered:\n{rendered!r}'
            )

    def test_output_is_the_concatenation_of_the_registry_renderers_in_order(self, monkeypatch):
        stage = self._make_live_stage(monkeypatch)

        rendered = stage._render_required_sections()

        expected = ''.join(
            getattr(stage, section.renderer)()
            for section in MemoryConsolidator.REQUIRED_SECTIONS
        )
        assert rendered == expected, (
            'The aggregator must be exactly the concatenation of its registry '
            "renderers' output, in registry order — it may not add a separator, "
            'reorder, or post-process. Each renderer already owns its own leading '
            f'newline.\n  got:      {rendered!r}\n  expected: {expected!r}'
        )

    def test_returns_empty_string_when_no_section_applies(self):
        # filtered_task_tree left at the _make_consolidator default (None), so
        # every registry renderer's guard fails.
        stage = _make_consolidator(project_root='/project')

        rendered = stage._render_required_sections()

        assert rendered == '', (
            "_render_required_sections() must return '' when no section applies — "
            "each renderer keeps its own conditional-empty contract and '' is a "
            f'normal result, keeping the payload tight. Got: {rendered!r}'
        )


class TestRegistryMatchesPromptAbsenceInference:
    """The registry and the shipped prompt must agree on what is inference-bearing.

    Both containments are asserted because they catch OPPOSITE defects, and only
    one of them is the defect this task closes:

    * prompt → registry catches a new absence-inference with no enforcement
      behind it, i.e. an inference the model is told to make that is unsound by
      construction. That is the gate-3833 defect itself.
    * registry → prompt catches a registry drifting wider than its own stated
      inclusion criterion, which would quietly turn this guard into "all
      builders emit the same set" — the invariant the task ruled out.

    ``prompts/stage2.py`` carries the symmetric sentence for Stage 2's own
    builders and is deliberately NOT asserted here (see the module docstring).
    """

    def test_prompt_absence_inferences_are_all_registered(self):
        registered = {s.header for s in MemoryConsolidator.REQUIRED_SECTIONS}
        inferred = _prompt_absence_inference_headers()

        unenforced = inferred - registered
        assert not unenforced, (
            f'STAGE1_SYSTEM_PROMPT tells the model to infer something from the '
            f'ABSENCE of {sorted(unenforced)}, but no MemoryConsolidator.'
            f'REQUIRED_SECTIONS member covers it — so a payload builder may omit '
            f'the section and the model will confidently conclude something false. '
            f'Registered today: {sorted(registered)}. {_REGEX_REPAIR_HINT}'
        )

    def test_registry_has_no_section_the_prompt_never_infers_from(self):
        registered = {s.header for s in MemoryConsolidator.REQUIRED_SECTIONS}
        inferred = _prompt_absence_inference_headers()

        unjustified = registered - inferred
        assert not unjustified, (
            f'MemoryConsolidator.REQUIRED_SECTIONS registers {sorted(unjustified)}, '
            f'but STAGE1_SYSTEM_PROMPT draws no inference from those sections being '
            f'absent — so the registry has drifted wider than its own inclusion '
            f'criterion and is on its way to "all builders emit the same set", the '
            f'invariant task 4708 explicitly ruled out. Sections that all builders '
            f'happen to need but that carry no absence-inference (e.g. the '
            f'project_root directive) belong in their own dedicated tests, not '
            f'here. Prompt infers from: {sorted(inferred)}. {_REGEX_REPAIR_HINT}'
        )
