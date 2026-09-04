"""Anti-rot parity guard: ARCHITECTURE.md section 3.1 vs
``shared.task_transitions.TRANSITIONS`` (task 4535).

The task-status lifecycle diagram in ``ARCHITECTURE.md`` section 3.1 is
documentation, hand-drawn from the real transition table
(``shared/src/shared/task_transitions.py::TRANSITIONS``) and prone to
drifting out of sync with it — the section's own prose records that the
diagram had already lost 19 edges once before this guard existed. This
module (together with its ``architecture_doc_transitions`` helper sibling)
pins the two artifacts together: it parses the diagram out of the doc,
reads the table, and asserts the edge sets are identical.

Built up TDD-pair by TDD-pair, mirroring the helper's own structure:

* pair 1 (this step) — the doc reader + section-3.1 extractor.
* pair 2 — the mermaid edge parser.
* pair 3 — the table-side reader + the diff comparator.
* pair 4 — the failure-message renderer.
* pair 5 — the mutation / failure-direction proofs, and the real gate.

Mirrors the established scanner+gate triad in this directory (see
``silent_fallthrough_scan.py`` + ``test_silent_fallthrough_gate.py``;
``capability_manifest_corpus.py`` + ``test_capability_manifest.py``):
``architecture_doc_transitions.py`` is the pure scanner/comparator, this
file holds the assertions. Importable bare
(``from architecture_doc_transitions import ...``) because
``shared/tests/conftest.py`` inserts this ``tests/`` directory onto
``sys.path``.
"""

from __future__ import annotations

from pathlib import Path

import architecture_doc_transitions
import pytest
from architecture_doc_transitions import (
    DOC_PATH,
    diff_transition_edges,
    extract_lifecycle_mermaid,
    format_parity_failure,
    parse_state_diagram_edges,
    read_architecture_doc,
    table_transition_edges,
)

from shared import task_transitions
from shared.task_statuses import TaskStatus
from shared.task_transitions import TRANSITIONS, ActorClass

# shared/tests/test_architecture_doc_transition_parity.py -> parents[0]=shared/tests,
# parents[1]=shared, parents[2]=repo root. Same idiom as
# shared/tests/conftest.py:23 and every sibling gate test in this directory.
_REPO_ROOT = Path(__file__).resolve().parents[2]

# A minimal but structurally real synthetic doc: a section 3.1 heading, one
# mermaid fence with exactly one edge, followed by the next section heading.
_MINIMAL_DOC = (
    "# ARCHITECTURE\n\n"
    "## 3. Task lifecycle\n\n"
    "### 3.1 Status vocabulary\n\n"
    "Some prose about statuses.\n\n"
    "```mermaid\n"
    "stateDiagram-v2\n"
    "    pending --> in_progress\n"
    "```\n\n"
    "### 3.2 Submit\n\n"
    "More prose that must not leak into the extracted body.\n"
)


class TestDocPath:
    """``DOC_PATH`` must resolve to the real repo-root ``ARCHITECTURE.md`` —
    including from inside a ``.worktrees/<id>`` checkout, which is where this
    test itself runs."""

    def test_doc_path_is_repo_root_architecture_md(self):
        assert DOC_PATH == _REPO_ROOT / 'ARCHITECTURE.md'

    def test_doc_path_exists(self):
        assert DOC_PATH.is_file()


class TestReadArchitectureDoc:
    def test_raises_loudly_when_doc_missing(self, tmp_path):
        # tmp_path has no ARCHITECTURE.md at all — must raise, never
        # pytest.skip past the missing file (that would silently disable the
        # guard instead of reporting the broken precondition).
        with pytest.raises(FileNotFoundError):
            read_architecture_doc(tmp_path)

    def test_reads_the_real_doc_by_default(self):
        text = read_architecture_doc()
        assert '### 3.1 Status vocabulary' in text


class TestExtractLifecycleMermaid:
    def test_returns_fence_body_only(self):
        body = extract_lifecycle_mermaid(_MINIMAL_DOC)
        assert body.strip() == 'stateDiagram-v2\n    pending --> in_progress'
        assert '```' not in body
        assert 'Some prose' not in body
        assert 'More prose' not in body

    def test_anchored_to_section_3_1_not_an_earlier_fence(self):
        # The real doc has THREE mermaid fences (process topology ~line 59,
        # lifecycle ~257, escalation ladder ~774) — an unanchored "first
        # fence in the doc" extractor would silently grab the wrong one.
        doc = (
            "# ARCHITECTURE\n\n"
            "### 2.1 Process topology\n\n"
            "```mermaid\n"
            "graph TD\n"
            "    X --> Y\n"
            "```\n\n"
            "## 3. Task lifecycle\n\n"
            "### 3.1 Status vocabulary\n\n"
            "```mermaid\n"
            "stateDiagram-v2\n"
            "    pending --> in_progress\n"
            "```\n\n"
            "### 3.2 Submit\n\n"
        )
        body = extract_lifecycle_mermaid(doc)
        assert 'pending --> in_progress' in body
        assert 'X --> Y' not in body

    def test_raises_when_section_heading_absent(self):
        doc = "# ARCHITECTURE\n\n### 3.2 Submit\n\nNo 3.1 heading anywhere in this doc.\n"
        with pytest.raises(RuntimeError):
            extract_lifecycle_mermaid(doc)

    def test_raises_when_no_mermaid_fence_in_section(self):
        doc = (
            "### 3.1 Status vocabulary\n\n"
            "Prose with no fence at all.\n\n"
            "### 3.2 Submit\n\n"
        )
        with pytest.raises(RuntimeError):
            extract_lifecycle_mermaid(doc)

    def test_raises_when_two_mermaid_fences_in_section(self):
        doc = (
            "### 3.1 Status vocabulary\n\n"
            "```mermaid\n"
            "stateDiagram-v2\n"
            "    pending --> in_progress\n"
            "```\n\n"
            "```mermaid\n"
            "stateDiagram-v2\n"
            "    blocked --> done\n"
            "```\n\n"
            "### 3.2 Submit\n\n"
        )
        with pytest.raises(RuntimeError):
            extract_lifecycle_mermaid(doc)

    def test_real_doc_body_contains_expected_markers(self):
        text = read_architecture_doc()
        body = extract_lifecycle_mermaid(text)
        assert 'stateDiagram-v2' in body
        assert 'in_progress' in body


class TestParseStateDiagramEdges:
    """``parse_state_diagram_edges`` — strict mermaid-line -> TaskStatus-pair
    parser. Strictness is the anti-rot core: any line it cannot interpret,
    or any node id outside the TaskStatus vocabulary, must raise rather than
    be silently dropped — a dropped line is exactly how a new diagram edge
    could vanish and leave the parity gate green through real drift."""

    def test_header_blank_and_full_line_comments_are_ignored(self):
        body = (
            "stateDiagram-v2\n"
            "\n"
            "    %% a full-line comment, and the blank line above\n"
            "    pending --> in_progress\n"
        )
        edges = parse_state_diagram_edges(body)
        assert edges == frozenset({(TaskStatus.PENDING, TaskStatus.IN_PROGRESS)})

    def test_trailing_comment_on_edge_line_is_stripped_not_kept(self):
        body = "stateDiagram-v2\n    pending --> in_progress %% dispatch note\n"
        edges = parse_state_diagram_edges(body)
        assert edges == frozenset({(TaskStatus.PENDING, TaskStatus.IN_PROGRESS)})

    def test_label_suffix_is_discarded(self):
        body = "stateDiagram-v2\n    pending --> in_progress: dispatch\n"
        edges = parse_state_diagram_edges(body)
        assert edges == frozenset({(TaskStatus.PENDING, TaskStatus.IN_PROGRESS)})

    def test_all_three_hyphenated_status_ids_round_trip(self):
        body = (
            "stateDiagram-v2\n"
            "    pending --> in_progress\n"
            "    in_progress --> merge_deferred\n"
            "    in_progress --> infra_hold\n"
        )
        edges = parse_state_diagram_edges(body)
        assert edges == frozenset(
            {
                (TaskStatus.PENDING, TaskStatus.IN_PROGRESS),
                (TaskStatus.IN_PROGRESS, TaskStatus.MERGE_DEFERRED),
                (TaskStatus.IN_PROGRESS, TaskStatus.INFRA_HOLD),
            }
        )

    def test_pseudo_state_entry_and_exit_edges_are_excluded(self):
        body = (
            "stateDiagram-v2\n"
            "    [*] --> pending\n"
            "    [*] --> deferred\n"
            "    done --> [*]\n"
            "    cancelled --> [*]\n"
            "    pending --> in_progress\n"
        )
        edges = parse_state_diagram_edges(body)
        assert edges == frozenset({(TaskStatus.PENDING, TaskStatus.IN_PROGRESS)})

    def test_raises_on_unknown_node_id(self):
        body = "stateDiagram-v2\n    foo --> pending\n"
        with pytest.raises(ValueError) as exc_info:
            parse_state_diagram_edges(body)
        assert 'foo' in str(exc_info.value)

    def test_raises_on_pipe_label_arrow_form(self):
        # `A -->|label| B` is valid mermaid but not a form this strict parser
        # supports — it must raise, not silently drop the edge.
        body = "stateDiagram-v2\n    pending -->|dispatch| done\n"
        with pytest.raises(ValueError) as exc_info:
            parse_state_diagram_edges(body)
        assert 'pending -->|dispatch| done' in str(exc_info.value)

    def test_raises_on_note_line(self):
        body = "stateDiagram-v2\n    note right of pending: some note\n"
        with pytest.raises(ValueError) as exc_info:
            parse_state_diagram_edges(body)
        assert 'note right of pending' in str(exc_info.value)

    def test_real_diagram_body_parses_to_nonempty_taskstatus_pairs(self):
        text = read_architecture_doc()
        body = extract_lifecycle_mermaid(text)
        edges = parse_state_diagram_edges(body)
        assert edges
        assert all(
            isinstance(frm, TaskStatus) and isinstance(to, TaskStatus) for frm, to in edges
        )
        assert (TaskStatus.PENDING, TaskStatus.IN_PROGRESS) in edges


class TestTableTransitionEdges:
    """``table_transition_edges()`` — the table side of the parity check."""

    def test_returns_union_over_all_actors(self):
        # Recomputed from the imported TRANSITIONS rather than pinning a
        # literal count, so this test does not itself become stale the next
        # time an edge is added to the table.
        expected = frozenset().union(*TRANSITIONS.values())
        assert table_transition_edges() == expected

    def test_every_element_is_a_taskstatus_pair(self):
        edges = table_transition_edges()
        assert edges
        assert all(
            isinstance(frm, TaskStatus) and isinstance(to, TaskStatus) for frm, to in edges
        )

    def test_reads_transitions_lazily_not_at_import_time(self, monkeypatch):
        # LOAD-BEARING, not incidental: this is what makes the step-9
        # mutation proofs mean anything. An import-time snapshot
        # (`from shared.task_transitions import TRANSITIONS` bound at module
        # scope, or an lru_cache) would survive this monkeypatch untouched
        # and keep returning the real ~37-edge union — silently making every
        # later mutation test vacuous.
        mutated = {ActorClass.HUMAN: frozenset({(TaskStatus.PENDING, TaskStatus.REVIEW)})}
        monkeypatch.setattr(task_transitions, 'TRANSITIONS', mutated)
        assert table_transition_edges() == frozenset({(TaskStatus.PENDING, TaskStatus.REVIEW)})


class TestDiffTransitionEdges:
    """``diff_transition_edges(doc, table)`` — the set-difference comparator.

    Every case asserts the ORIENTATION explicitly via the named fields
    (``.missing_from_doc`` / ``.extra_in_doc``), not just set equality of
    the pair — a comparator that silently swapped the two halves would
    otherwise pass an equality-only check.
    """

    def test_identical_inputs_yield_empty_diff(self):
        edges = frozenset({(TaskStatus.PENDING, TaskStatus.IN_PROGRESS)})
        diff = diff_transition_edges(edges, edges)
        assert diff.missing_from_doc == frozenset()
        assert diff.extra_in_doc == frozenset()

    def test_table_only_edge_is_missing_from_doc(self):
        doc_edges: frozenset[tuple[TaskStatus, TaskStatus]] = frozenset()
        table_edges = frozenset({(TaskStatus.PENDING, TaskStatus.IN_PROGRESS)})
        diff = diff_transition_edges(doc_edges, table_edges)
        assert diff.missing_from_doc == frozenset({(TaskStatus.PENDING, TaskStatus.IN_PROGRESS)})
        assert diff.extra_in_doc == frozenset()

    def test_doc_only_edge_is_extra_in_doc(self):
        doc_edges = frozenset({(TaskStatus.PENDING, TaskStatus.IN_PROGRESS)})
        table_edges: frozenset[tuple[TaskStatus, TaskStatus]] = frozenset()
        diff = diff_transition_edges(doc_edges, table_edges)
        assert diff.extra_in_doc == frozenset({(TaskStatus.PENDING, TaskStatus.IN_PROGRESS)})
        assert diff.missing_from_doc == frozenset()

    def test_both_sided_divergence_reports_each_edge_on_exactly_one_side(self):
        doc_edges = frozenset({(TaskStatus.PENDING, TaskStatus.REVIEW)})
        table_edges = frozenset({(TaskStatus.PENDING, TaskStatus.IN_PROGRESS)})
        diff = diff_transition_edges(doc_edges, table_edges)
        assert diff.missing_from_doc == frozenset({(TaskStatus.PENDING, TaskStatus.IN_PROGRESS)})
        assert diff.extra_in_doc == frozenset({(TaskStatus.PENDING, TaskStatus.REVIEW)})


class TestFormatParityFailure:
    """``format_parity_failure`` — the user-observable failure text. The
    task's own success signal is that the guard "fails naming the specific
    divergent edge", so this text is real behavior, not incidental logging,
    and gets pinned like any other output."""

    def test_missing_from_doc_names_edge_direction_and_both_sources(self):
        msg = format_parity_failure(
            frozenset({(TaskStatus.IN_PROGRESS, TaskStatus.REVIEW)}),
            frozenset(),
        )
        assert 'in-progress -> review' in msg
        assert 'NOT drawn in ARCHITECTURE.md section 3.1' in msg
        assert 'ARCHITECTURE.md section 3.1' in msg
        assert 'shared/src/shared/task_transitions.py' in msg

    def test_extra_in_doc_names_the_opposite_direction(self):
        msg = format_parity_failure(
            frozenset(),
            frozenset({(TaskStatus.IN_PROGRESS, TaskStatus.REVIEW)}),
        )
        assert 'in-progress -> review' in msg
        assert 'NOT in TRANSITIONS' in msg
        assert 'ARCHITECTURE.md section 3.1' in msg
        assert 'shared/src/shared/task_transitions.py' in msg

    def test_the_two_directions_produce_distinguishable_messages(self):
        edge = frozenset({(TaskStatus.IN_PROGRESS, TaskStatus.REVIEW)})
        missing_msg = format_parity_failure(edge, frozenset())
        extra_msg = format_parity_failure(frozenset(), edge)
        assert missing_msg != extra_msg

    def test_multiple_edges_all_listed_in_deterministic_sorted_order(self):
        edges = frozenset(
            {
                (TaskStatus.REVIEW, TaskStatus.DONE),
                (TaskStatus.IN_PROGRESS, TaskStatus.BLOCKED),
            }
        )
        msg = format_parity_failure(edges, frozenset())
        assert 'in-progress -> blocked' in msg
        assert 'review -> done' in msg
        assert msg.index('in-progress -> blocked') < msg.index('review -> done')

    def test_empty_and_empty_renders_no_edge_lines(self):
        msg = format_parity_failure(frozenset(), frozenset())
        assert '->' not in msg


class TestCheckArchitectureTransitionParity:
    """Mutation / failure-direction proofs for ``check_architecture_transition_parity``.

    This is the step the task exists for: "a parity test that passes under
    an injected mutation is worthless, and this repo has a recorded case of
    four assertions staying green through a full semantic reversal." Each
    case here monkeypatches ``shared.task_transitions.TRANSITIONS`` to a
    one-edge-mutated COPY of the real table (never the real frozensets
    themselves) and calls the exact function the real gate calls, with
    ARCHITECTURE.md left untouched throughout.

    ``check_architecture_transition_parity`` is deliberately referenced via
    ``architecture_doc_transitions.check_architecture_transition_parity``
    (module-attribute access), not a top-level ``from ... import``: it does
    not exist yet, and a top-level import would fail the whole module at
    collection (ImportError), taking every already-green test above down
    with it. Attribute access instead fails each of THIS class's tests
    individually with ``AttributeError``, which is this step's RED.
    """

    def test_control_unmutated_table_passes_and_returns_nonempty_edges(self, monkeypatch):
        # Guards against a guard that always raises: a byte-identical copy
        # of the real table must NOT be reported as divergent.
        unmutated = {actor: frozenset(edges) for actor, edges in TRANSITIONS.items()}
        monkeypatch.setattr(task_transitions, 'TRANSITIONS', unmutated)
        result = architecture_doc_transitions.check_architecture_transition_parity()
        assert result

    def test_added_edge_raises_naming_exactly_that_edge(self, monkeypatch):
        added_edge = (TaskStatus.REVIEW, TaskStatus.MERGE_DEFERRED)
        real_union = frozenset().union(*TRANSITIONS.values())
        assert added_edge not in real_union, 'test fixture must pick a genuinely absent edge'

        mutated = {actor: frozenset(edges) | {added_edge} for actor, edges in TRANSITIONS.items()}
        monkeypatch.setattr(task_transitions, 'TRANSITIONS', mutated)
        with pytest.raises(AssertionError) as exc_info:
            architecture_doc_transitions.check_architecture_transition_parity()
        message = str(exc_info.value)
        assert 'review -> merge-deferred' in message
        assert 'NOT drawn in ARCHITECTURE.md section 3.1' in message
        assert message.count('->') == 1, f'expected exactly one named edge, got: {message!r}'

    def test_removed_edge_raises_naming_exactly_that_edge(self, monkeypatch):
        removed_edge = (TaskStatus.PENDING, TaskStatus.IN_PROGRESS)
        mutated = {
            actor: frozenset(e for e in edges if e != removed_edge)
            for actor, edges in TRANSITIONS.items()
        }
        monkeypatch.setattr(task_transitions, 'TRANSITIONS', mutated)
        with pytest.raises(AssertionError) as exc_info:
            architecture_doc_transitions.check_architecture_transition_parity()
        message = str(exc_info.value)
        assert 'pending -> in-progress' in message
        assert 'NOT in TRANSITIONS' in message
        assert message.count('->') == 1, f'expected exactly one named edge, got: {message!r}'

    def test_added_and_removed_edge_messages_are_different(self, monkeypatch):
        # A semantic reversal that reported both directions identically
        # would satisfy every substring check above while proving nothing —
        # this is the check that rules that out.
        added_edge = (TaskStatus.REVIEW, TaskStatus.MERGE_DEFERRED)
        added_mutated = {
            actor: frozenset(edges) | {added_edge} for actor, edges in TRANSITIONS.items()
        }
        monkeypatch.setattr(task_transitions, 'TRANSITIONS', added_mutated)
        with pytest.raises(AssertionError) as added_exc:
            architecture_doc_transitions.check_architecture_transition_parity()

        removed_edge = (TaskStatus.PENDING, TaskStatus.IN_PROGRESS)
        removed_mutated = {
            actor: frozenset(e for e in edges if e != removed_edge)
            for actor, edges in TRANSITIONS.items()
        }
        monkeypatch.setattr(task_transitions, 'TRANSITIONS', removed_mutated)
        with pytest.raises(AssertionError) as removed_exc:
            architecture_doc_transitions.check_architecture_transition_parity()

        assert str(added_exc.value) != str(removed_exc.value)


class TestArchitectureDiagramGate:
    """The real, unmutated anti-rot gate: ARCHITECTURE.md section 3.1 must
    draw exactly the edges in shared.task_transitions.TRANSITIONS, today,
    on the real files -- no monkeypatching, no synthetic doc."""

    def test_architecture_diagram_matches_transitions_table(self):
        doc_edges = architecture_doc_transitions.check_architecture_transition_parity()
        # A parser regression that returned frozenset() would make BOTH diff
        # sides empty and the bare parity call above would pass silently --
        # the exact vacuous-green failure this task exists to prevent.
        # Cardinality is derived from the table, not pinned to the literal
        # 37, so this assertion does not itself become the next stale pin.
        assert doc_edges
        assert len(doc_edges) == len(table_transition_edges())
