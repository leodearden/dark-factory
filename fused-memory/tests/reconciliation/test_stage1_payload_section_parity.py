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
* the BUILDER set is DERIVED by AST introspection over the class body, so a
  fourth builder is in scope the day it is added with no edit here.

The only hand-written literals are the registry itself (the intended single
edit point) and a discovery FLOOR, which can only ever be too small — and being
too small fails loudly.
"""

from __future__ import annotations

import ast
import pathlib

import pytest
from _ast_guard import calls_named, parse_python_module

import fused_memory.reconciliation.stages.memory_consolidator as consolidator_module
from fused_memory.reconciliation.stages.memory_consolidator import MemoryConsolidator
from fused_memory.reconciliation.task_filter import FilteredTaskTree
from reconciliation.test_stage1 import _make_consolidator

CONSOLIDATOR_SRC = pathlib.Path(consolidator_module.__file__)
AGGREGATOR = '_render_required_sections'
CONSOLIDATOR_CLASS = 'MemoryConsolidator'

# A payload builder returns a WHOLE payload, which in this module always means an
# f-string opening with the top-level markdown header. See
# _discover_stage1_payload_builders for the verified discrimination.
_PAYLOAD_HEADER_PREFIX = '## '


def _returns_a_whole_payload(node: ast.AST) -> ast.Return | None:
    """The ``Return`` under *node* that returns a whole Stage-1 payload, if any.

    A whole payload is an f-string whose LEADING literal chunk opens with
    ``'## '`` — the top-level markdown header that makes a string an entire
    payload rather than one section of one. Returns the node itself (not just a
    bool) so callers can assert over the f-string's interpolations.
    """
    for descendant in ast.walk(node):
        if not isinstance(descendant, ast.Return):
            continue
        if not isinstance(descendant.value, ast.JoinedStr):
            continue
        values = descendant.value.values
        lead = values[0] if values else None
        if (
            isinstance(lead, ast.Constant)
            and isinstance(lead.value, str)
            and lead.value.startswith(_PAYLOAD_HEADER_PREFIX)
        ):
            return descendant
    return None


def _discover_stage1_payload_builders() -> dict[str, tuple[ast.AST, ast.Return]]:
    """Every Stage-1 payload builder on ``MemoryConsolidator``, by introspection.

    DERIVED, not hand-listed — that is the whole point. A fourth payload builder
    is pulled into scope the day it is added, with no edit to this file. A
    hand-maintained list of three names would pin regression in the builders
    tasks 2150/2552/3839 already fixed and stay blind to the next one, which is
    precisely the defect task 4708 closes.

    Predicate: a method directly in the ``MemoryConsolidator`` class body that
    returns an f-string opening with ``'## '``. Its discrimination was verified
    against this module:

    * SELECTS ``assemble_payload`` ('## Reconciliation Run — Stage 1: …'),
      ``_format_assembled_payload`` (same header) and
      ``_assemble_remediation_payload`` ('## Remediation Run — Stage 1: …').
    * REJECTS the section builders. ``_build_task_count_census_section``
      ('\\n### Task Count Census …') and ``_build_project_root_directive``
      ('\\nUse project_root="…') do return f-strings, but neither opens with
      ``'## '``; ``_build_task_tree_section`` and
      ``_build_live_workflow_section`` return ``BinOp`` and cannot match at all;
      the aggregator returns a ``Call``.

    The class body is iterated DIRECTLY rather than via ``ast.walk``: a helper
    function nested inside a builder is not itself a builder.
    """
    tree = parse_python_module(CONSOLIDATOR_SRC)
    class_def = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == CONSOLIDATOR_CLASS
        ),
        None,
    )
    assert class_def is not None, (
        f'{CONSOLIDATOR_SRC.name} no longer defines a top-level class '
        f'{CONSOLIDATOR_CLASS!r}. Builder discovery cannot run; fix the class name '
        f'here rather than letting this guard check nothing.'
    )
    builders: dict[str, tuple[ast.AST, ast.Return]] = {}
    for node in class_def.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        payload_return = _returns_a_whole_payload(node)
        if payload_return is not None:
            builders[node.name] = (node, payload_return)
    return builders


STAGE1_PAYLOAD_BUILDERS = _discover_stage1_payload_builders()


def _aggregator_call(node: ast.AST) -> bool:
    """True when *node* is a call to ``self._render_required_sections()``."""
    return bool(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == AGGREGATOR
    )


def _names_bound_to_the_aggregator(builder: ast.AST) -> set[str]:
    """Locals in *builder* assigned the aggregator's output.

    Lets the parity check accept the indirect spelling
    ``x = self._render_required_sections()`` … ``f"…{x}…"`` as well as direct
    interpolation, so the guard proves the rendered text reaches the returned
    payload without pinning one brittle spelling.
    """
    bound: set[str] = set()
    for descendant in ast.walk(builder):
        if isinstance(descendant, ast.Assign) and _aggregator_call(descendant.value):
            bound.update(
                target.id for target in descendant.targets if isinstance(target, ast.Name)
            )
        elif (
            isinstance(descendant, ast.AnnAssign)
            and isinstance(descendant.target, ast.Name)
            # A bare annotation (``x: str``) has ``value=None`` and binds nothing.
            and descendant.value is not None
            and _aggregator_call(descendant.value)
        ):
            bound.add(descendant.target.id)
    return bound


def _payload_interpolates_the_aggregator(builder: ast.AST, payload_return: ast.Return) -> bool:
    """True when the returned payload f-string carries the aggregator's output."""
    bound = _names_bound_to_the_aggregator(builder)
    assert isinstance(payload_return.value, ast.JoinedStr)
    for value in payload_return.value.values:
        if not isinstance(value, ast.FormattedValue):
            continue
        if _aggregator_call(value.value):
            return True
        if isinstance(value.value, ast.Name) and value.value.id in bound:
            return True
    return False


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
            f'_discover_stage1_payload_builders has gone stale — most likely a '
            f'payload f-string was hoisted to a module constant or built by '
            f'concatenation, or the top-level {_PAYLOAD_HEADER_PREFIX!r} header was '
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
        builder, payload_return = STAGE1_PAYLOAD_BUILDERS[builder_name]

        assert _payload_interpolates_the_aggregator(builder, payload_return), (
            f'{CONSOLIDATOR_CLASS}.{builder_name} returns a Stage-1 payload that '
            f'never interpolates self.{AGGREGATOR}(), so it can omit '
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
        builder, _ = STAGE1_PAYLOAD_BUILDERS[builder_name]

        for section in MemoryConsolidator.REQUIRED_SECTIONS:
            assert not calls_named(builder, section.renderer), (
                f'{CONSOLIDATOR_CLASS}.{builder_name} calls '
                f'self.{section.renderer}() directly. Registered sections must be '
                f'reached ONLY through self.{AGGREGATOR}(), or adding the next '
                f'section is once again one edit per builder rather than one edit '
                f'to REQUIRED_SECTIONS — the exact drift behind tasks 2150, 2552 '
                f'and 3839. Delete the direct call and the local it feeds.'
            )
