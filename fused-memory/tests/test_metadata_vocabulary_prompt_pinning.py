"""Drift pin: every metadata example embedded in a PROMPT must agree with the
registry that defines the vocabulary (task 3202, PRD
``docs/prds/memory-metadata-vocabulary.md`` leaf ι, invariant INV-5 — one
normative copy, everything else a pointer).

``fused_memory.memory_metadata`` (leaf β / task 3195) is that normative copy.
Two prompt surfaces quote it, and both are pinned here:

ARM 1 — the writer instructions. ``METADATA_VOCABULARY_INSTRUCTIONS`` in
``orchestrator/src/orchestrator/agents/roles.py`` documents the reserved keys to
every memory-writing agent role. Adding a key to ``RESERVED_VOCABULARY_KEYS``
without documenting it, or renaming one, fails here.

ARM 2 — the agent-facing surfaces that show a memory kind: the reconciliation
stage prompts and the ``count_memories_by_metadata`` tool docstring. Every kind
literal they show — in the ``'kind': '<literal>'`` filter-dict spelling AND in
the ``metadata.kind = "<literal>"`` / ``record_kind = "<literal>"`` schema
spelling the stage prompts' schema sections use — must name a kind that is in
``KIND_REGISTRY``, so a kind rename cannot silently strand agents emitting
filters that match nothing.

ARM 2 is deliberately NOT exhaustive over every possible spelling: it pins the
surfaces and syntaxes that were measured to carry kinds, and a novel spelling
introduced later would be invisible to it. Extend the regex and
``_PROMPT_SOURCES`` when one appears rather than assuming the coverage is total.

WHY THIS FILE LIVES IN fused-memory/tests: this suite's pytest config already
declares ``pythonpath = ["src", "../orchestrator/src"]``, so BOTH
``fused_memory.memory_metadata`` and ``orchestrator.agents.roles`` are hard
imports here. The orchestrator suite cannot import ``fused_memory``, and a
guard written there would need a ``pytest.importorskip`` — i.e. a drift test
that silently becomes a no-op the day the path breaks.

SCOPE DISCIPLINE (review, task 3202). This file pins REGISTRY-DERIVED facts
only. It does NOT assert on prompt WORDING: no bullet/em-dash/backtick shape,
no verbatim substring of surrounding prose, no non-emptiness of a literal, no
``endswith`` splice-order check. Those all go red on a cosmetic reflow while
catching no functional regression. Reflowing, reordering, or rewording the
vocabulary section is expected to keep this file green — only a genuine
registry-vs-prompt divergence turns it red. Do not add wording assertions, and
do not add tests that exercise a helper defined in this file rather than
production code.

SEQUENCING SEAM — task 3131 (dep-gated behind 3169) rewrites the
write-eagerness prose inside ``_MEMORY_INSTRUCTIONS``. Nothing here pins any of
those sentences, so that edit and this pin cannot collide.
"""

from __future__ import annotations

import ast
import importlib.util
import re
from pathlib import Path

import pytest
from orchestrator.agents.roles import _MEMORY_INSTRUCTIONS, METADATA_VOCABULARY_INSTRUCTIONS, ROLES

from fused_memory.memory_metadata import (
    KIND_REGISTRY,
    RESERVED_VOCABULARY_KEYS,
    TOPIC_SLUG_MAX_LEN,
)
from fused_memory.reconciliation.prompts.stage1 import STAGE1_SYSTEM_PROMPT
from fused_memory.reconciliation.prompts.stage2 import (
    STAGE2_SYSTEM_PROMPT,
    build_stage2_system_prompt,
)
from fused_memory.reconciliation.prompts.stage3 import STAGE3_SYSTEM_PROMPT
from fused_memory.reconciliation.recon_pool_map import CYCLE_SUMMARY_KIND
from fused_memory.reconciliation.recon_self_model import render_cycle_summary_section

# Roles are DERIVED from the splice, never restated as a literal tuple: adding a
# tenth role must not fail a test about the metadata vocabulary.
_MEMORY_ROLES = sorted(
    name for name, role in ROLES.items() if _MEMORY_INSTRUCTIONS in role.system_prompt
)


def _defines(text: str, key: str) -> bool:
    """Does *text* carry a line whose LEADING reserved key is *key*?

    A bare ``key in text`` substring check is vacuous for the deletion case on
    two of the five keys (review, task 3202): ``topic`` also occurs inside the
    ``canonical`` bullet ("requires ``topic``") and ``kind`` inside the
    ``parent_id`` bullet ("kinds ``amendment``…"), so deleting either bullet
    outright leaves a bare check green while writers lose the documentation.

    "Leads a line" is the weakest structure that separates a key's own
    DEFINITION from an incidental cross-reference to it: wherever the section
    defines a key it names that key before it names any other, and wherever it
    merely cites one, some other key came first. Deliberately NOT a bullet /
    backtick / em-dash shape pin — an earlier such regex was removed in review
    for going red on cosmetic reflow — so markers, punctuation, ordering and
    wording all stay free to change.

    Known and accepted limit: if a reflow ever wrapped a cross-reference onto
    its own continuation line, that line would "lead" with the cited key and
    this would go falsely green for it. That is strictly better than the bare
    substring check it replaces, and it never goes falsely RED.
    """
    for line in text.splitlines():
        positions = {k: line.find(k) for k in RESERVED_VOCABULARY_KEYS}
        present = {k: i for k, i in positions.items() if i >= 0}
        if key in present and present[key] == min(present.values()):
            return True
    return False


class TestReservedKeysReachTheWriters:
    """ARM 1: the writer-facing text names every key the write path validates.

    Registry-derived and reflow-proof: the key set is read from
    ``RESERVED_VOCABULARY_KEYS`` and located structurally (see ``_defines``),
    so the section can be reworded or relaid-out freely. What it does catch is
    the PRD §6 gap this task closes — a key that the writer path enforces but
    no agent was ever told about — and, now, a documented key that later gets
    deleted.
    """

    @pytest.mark.parametrize('key', sorted(RESERVED_VOCABULARY_KEYS))
    def test_every_reserved_key_is_documented(self, key: str) -> None:
        assert _defines(METADATA_VOCABULARY_INSTRUCTIONS, key), (
            f'reserved key {key!r} is validated on write but has no line of its own '
            f'in the writer instructions — writers cannot use a key they are not '
            f'told about, and a passing mention elsewhere is not documentation'
        )

    def test_topic_slug_cap_resolves_to_the_registry_value(self) -> None:
        """The one registry SCALAR the writer prose quotes must be the live one.

        The orchestrator package cannot import ``fused_memory``, so the cap
        cannot be interpolated into ``METADATA_VOCABULARY_INSTRUCTIONS`` and is
        hand-written there. Raising ``TOPIC_SLUG_MAX_LEN`` without updating the
        prose would leave every agent briefing quietly stating the wrong limit
        — the exact miscommunication this leaf exists to close (review, task
        3202). A value-RESOLUTION check, not a wording pin: any reflow that
        keeps the number stays green.
        """
        assert str(TOPIC_SLUG_MAX_LEN) in METADATA_VOCABULARY_INSTRUCTIONS, (
            f'the writer instructions do not state the live topic-slug cap '
            f'({TOPIC_SLUG_MAX_LEN}); writers cannot obey a cap they are not told'
        )

    @pytest.mark.parametrize('role_name', _MEMORY_ROLES)
    @pytest.mark.parametrize('key', sorted(RESERVED_VOCABULARY_KEYS))
    def test_every_reserved_key_reaches_each_memory_role(self, key: str, role_name: str) -> None:
        """The task's user-observable signal, asserted end-to-end on the
        RENDERED prompt: a memory-writing role's assembled briefing names every
        reserved key (including ``supersedes``)."""
        assert key in ROLES[role_name].system_prompt

    def test_some_role_actually_splices_the_memory_block(self) -> None:
        """Guards the derivation above: if the splice were renamed away, the
        parametrized role checks would vacuously collapse to zero cases."""
        assert _MEMORY_ROLES


#: Both spellings an agent-facing surface uses to show a memory kind:
#:
#: * the MAPPING form ``'kind': '<value>'`` / ``"kind": "<value>"`` — the filter
#:   dict a recon prompt or a tool docstring tells the agent to emit;
#: * the ASSIGNMENT form ``metadata.kind = "<value>"`` / ``record_kind =
#:   "<value>"`` — how the stage prompts' schema sections state the kind a
#:   record must carry (``render_suppression_schema_section``,
#:   ``render_entity_standing_decision_schema_section``,
#:   ``render_investigation_outcome_section``, all spliced into STAGE1/STAGE2).
#:   The assignment form was invisible to this pin until review (task 3202), so
#:   a rename of ``stage1_flag_suppression`` / ``entity_standing_decision`` /
#:   ``investigation_outcome`` was unguarded.
#:
#: The ``kind`` token must be quote- or dot-/underscore-prefixed, which is what
#: keeps ``task_kind='deterministic'`` and ``task_kind='predicate'`` out: those
#: are TASK metadata (see ``docs/task-authoring.md``) and are deliberately not
#: KIND_REGISTRY members, so matching them would fail this pin spuriously.
_KIND_LITERAL_RE = re.compile(
    r"""(?:
          'kind':\s*'(?P<sq_mapping>[^']+)'
        | "kind":\s*"(?P<dq_mapping>[^"]+)"
        | (?:metadata\.|record_)kind\s*=\s*'(?P<sq_assign>[^']+)'
        | (?:metadata\.|record_)kind\s*=\s*"(?P<dq_assign>[^"]+)"
        )""",
    re.VERBOSE,
)


def _tool_docstring(module_name: str, func_name: str) -> str:
    """Return the docstring of the tool function *func_name* in *module_name*.

    Parsed out of the module SOURCE with ``ast`` rather than read off an
    attribute, because MCP tools are nested inside the registration function
    and only exist once a server has been constructed — a dependency this pin
    must not acquire. ``find_spec`` locates the file without executing it.

    Scoped to the one function on purpose. Scanning the whole module was
    measured and is wrong: ``server/tools.py`` also documents
    ``done_provenance``'s ``'kind': 'merged'``, which is TASK metadata and
    correctly absent from ``KIND_REGISTRY``.
    """
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        raise RuntimeError(f'cannot locate source of {module_name!r} to pin its tool docstrings')
    tree = ast.parse(Path(spec.origin).read_text(encoding='utf-8'))
    defs = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name
    ]
    if len(defs) != 1:
        raise RuntimeError(
            f'expected exactly one def of {func_name!r} in {module_name}, found {len(defs)} — '
            f'the tool was renamed or duplicated; update this pin rather than dropping it'
        )
    doc = ast.get_docstring(defs[0])
    if not doc:
        raise RuntimeError(f'{func_name} in {module_name} has no docstring left to pin')
    return doc


#: Agent-facing surfaces carrying kind examples. "Prompt" here means anything
#: an agent reads and copies from: the recon stage prompts AND the MCP tool
#: docstrings, which reach an agent through the tool listing just as directly.
_PROMPT_SOURCES = {
    'STAGE1_SYSTEM_PROMPT': STAGE1_SYSTEM_PROMPT,
    'STAGE2_SYSTEM_PROMPT': STAGE2_SYSTEM_PROMPT,
    "build_stage2_system_prompt('dark_factory')": build_stage2_system_prompt('dark_factory'),
    'STAGE3_SYSTEM_PROMPT': STAGE3_SYSTEM_PROMPT,
    'render_cycle_summary_section()': render_cycle_summary_section(),
    'count_memories_by_metadata.__doc__': _tool_docstring(
        'fused_memory.server.tools', 'count_memories_by_metadata'
    ),
}


def _kind_literals(text: str) -> set[str]:
    """Extract the memory-kind literals an agent-facing surface shows.

    This matches the literal syntax the agent is being told to emit or expect —
    not prose formatting — which is why it is a fair thing to pin.
    """
    return {
        value
        for match in _KIND_LITERAL_RE.finditer(text)
        for value in match.groups()
        if value is not None
    }


class TestReconPromptKindLiteralsPinned:
    """ARM 2: every kind example an agent-facing surface shows is REGISTERED.

    This is the INV-5 guard for ``render_cycle_summary_section``: if it re-typed
    the literal instead of interpolating ``CYCLE_SUMMARY_KIND``, a rename in
    ``recon_pool_map`` would leave the rendered text on the old value and
    strand every recon agent that copied the filter.

    An ``inspect.getsource`` pin on the *identifier text* was removed in review
    (task 3202): it passed for a function that merely MENTIONS the name in a
    comment, and failed correct refactors — ``import ... as CSK``, a
    module-qualified ``recon_pool_map.CYCLE_SUMMARY_KIND``, or extracting the
    f-string into a helper. Do NOT re-add it; assert on rendered output instead.
    """

    @pytest.mark.parametrize('source_name', sorted(_PROMPT_SOURCES))
    def test_every_kind_literal_is_registered(self, source_name: str) -> None:
        literals = _kind_literals(_PROMPT_SOURCES[source_name])
        assert literals, (
            'no kind literal extracted — the filter-example syntax changed, so '
            'this guard would silently pass against nothing'
        )
        outside = literals - KIND_REGISTRY
        assert not outside, (
            f'{source_name} shows kind literals that are not in KIND_REGISTRY: '
            f'{sorted(outside)} — recon agents copying these filters would match nothing'
        )

    @pytest.mark.parametrize('source_name', sorted(_PROMPT_SOURCES))
    def test_cycle_summary_survives_rendering(self, source_name: str) -> None:
        """Each pinned surface really does carry the shared constant's value."""
        assert CYCLE_SUMMARY_KIND in _kind_literals(_PROMPT_SOURCES[source_name])

    def test_cycle_summary_constant_is_registered(self) -> None:
        """The constant<->registry edge itself: the shared kind constant the
        writers use must be a member of the closed registry."""
        assert CYCLE_SUMMARY_KIND in KIND_REGISTRY
