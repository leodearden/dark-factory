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

import pytest
from architecture_doc_transitions import (
    DOC_PATH,
    extract_lifecycle_mermaid,
    parse_state_diagram_edges,
    read_architecture_doc,
)

from shared.task_statuses import TaskStatus

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
