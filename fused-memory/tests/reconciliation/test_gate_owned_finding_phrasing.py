"""Tests for gate-owned finding phrasing — Stage 1 norm + code gate (task 4814).

Stage 1 (``memory_consolidator``) sometimes writes a finding's
``suggested_action`` in words that read as authorizing Stage 2 to DECIDE
("Stage 2 or operator should decide", "enumerate", "extend") when the
finding's cited task is a HUMAN GATE (``metadata.operational_mode == 'gate'``
and/or ``metadata.always_escalates == true``).  Stage 2 holds
``update_task``/``set_task_status``, so that phrasing invites a stage to close
a question only a human may answer.  Evidence: autopilot_video run 8b2d3371,
Stage 1 findings 640d4ceb (task 645) and 66af9601 (task 652) — Stage 2 caught
both that cycle, and the recurring cost is that it must catch them EVERY
cycle.

Covers:
- extract_human_gated_task_ids: pure selector over
  ``FilteredTaskTree.active_tasks`` returning the sorted, deduped str ids of
  human-gate-owned tasks.  Deliberately a WIDER population than its sibling
  ``curator_gate_resolution_sweep.extract_open_gate_task_ids`` (which filters
  on mode alone), and deliberately STRICTER than a truthy read.
- CANONICAL_HUMAN_GATE_ACTION / GATE_OWNED_ACTION_NORM_HEADING /
  render_gate_owned_action_norm: the ONE sentence both halves quote, its
  exported heading, and the Stage-1 prompt section that carries it.
"""

from __future__ import annotations

from fused_memory.reconciliation.curator_gate_resolution_sweep import (
    GATE_RESOLUTION_FLAG_TYPE,
)
from fused_memory.reconciliation.gate_owned_finding_phrasing import (
    CANONICAL_HUMAN_GATE_ACTION,
    GATE_OWNED_ACTION_NORM_HEADING,
    extract_human_gated_task_ids,
    render_gate_owned_action_norm,
)
from fused_memory.reconciliation.task_filter import FilteredTaskTree


class TestExtractHumanGatedTaskIds:
    """extract_human_gated_task_ids(tasks) selects human-gate-owned tasks.

    Inputs are built as ``FilteredTaskTree(active_tasks=[...])`` and the
    ``active_tasks`` field is passed, mirroring how the sibling selector
    ``extract_open_gate_task_ids`` is exercised in
    ``test_curator_gate_resolution_sweep.py`` — that suite is the live proof
    that ``task['metadata']`` is reachable as a dict on those dicts.

    The predicate is ``operational_mode == 'gate'`` OR ``always_escalates is
    True``, mirroring the strictness of the canonical gate predicate
    ``TaskInterceptor._is_gate_metadata`` (which this module deliberately does
    not import — see the module docstring).  Strictness errs fail-safe: a
    strict predicate can only UNDER-select, so a non-gate finding is never
    rewritten.
    """

    def test_selects_operational_mode_gate_task(self):
        """metadata.operational_mode == 'gate' selects (the run-8b2d3371 task 645 shape)."""
        tree = FilteredTaskTree(active_tasks=[
            {'id': 645, 'status': 'blocked', 'metadata': {'operational_mode': 'gate'}},
        ])

        result = extract_human_gated_task_ids(tree.active_tasks)

        assert result == ['645'], (
            f'operational_mode == "gate" must select the task, got {result!r}'
        )

    def test_selects_always_escalates_task_without_operational_mode(self):
        """always_escalates=True with NO operational_mode selects (the task 652 shape).

        This is the population the shipped sibling ``extract_open_gate_task_ids``
        provably misses — it filters on mode alone — which is why task 4814
        needs its own selector rather than reusing that one.
        """
        tree = FilteredTaskTree(active_tasks=[
            {'id': 652, 'status': 'blocked', 'metadata': {'always_escalates': True}},
        ])

        result = extract_human_gated_task_ids(tree.active_tasks)

        assert result == ['652'], (
            'always_escalates is True must select even with no operational_mode '
            f'key — the operational-routing coercion produces exactly that shape; got {result!r}'
        )

    def test_task_carrying_both_markers_is_selected_exactly_once(self):
        """A task with both gate markers contributes one id, not two (the return is deduped)."""
        tree = FilteredTaskTree(active_tasks=[
            {
                'id': 645,
                'status': 'blocked',
                'metadata': {'operational_mode': 'gate', 'always_escalates': True},
            },
        ])

        result = extract_human_gated_task_ids(tree.active_tasks)

        assert result == ['645'], (
            f'a task matching both clauses must appear exactly once, got {result!r}'
        )

    def test_non_gate_task_is_not_selected(self):
        """operational_mode='llm' with no always_escalates is not a gate."""
        tree = FilteredTaskTree(active_tasks=[
            {'id': 4242, 'status': 'pending', 'metadata': {'operational_mode': 'llm'}},
            {'id': 4243, 'status': 'pending', 'metadata': {'always_escalates': False}},
            {'id': 4244, 'status': 'pending', 'metadata': {'execution_class': 'operational'}},
        ])

        result = extract_human_gated_task_ids(tree.active_tasks)

        assert result == [], (
            'a non-gate task must never be selected — over-selection would rewrite '
            f'the suggested_action of a finding no human owns; got {result!r}'
        )

    def test_execution_class_operational_alone_is_not_a_gate(self):
        """execution_class='operational' is the THIRD _is_gate_metadata clause, deliberately not adopted.

        ``TaskInterceptor._is_gate_metadata`` answers True on that clause too.
        Task 4814 asks only for ``operational_mode='gate'`` and/or
        ``always_escalates=true``, so adopting it would over-select
        operational-class tasks that are not human gates.
        """
        tree = FilteredTaskTree(active_tasks=[
            {'id': 99, 'metadata': {'execution_class': 'operational'}},
        ])

        assert extract_human_gated_task_ids(tree.active_tasks) == [], (
            'execution_class == "operational" must NOT select — that third '
            '_is_gate_metadata clause is deliberately not adopted by task 4814'
        )

    def test_matches_are_value_and_type_sensitive(self):
        """'GATE' is not 'gate', and the string 'true' is not True.

        Mirrors ``TaskInterceptor._is_gate_metadata``'s stated rationale:
        ``bool('false')`` is True, so a loose read would silently accept the
        opposite of the caller's intent.
        """
        tree = FilteredTaskTree(active_tasks=[
            {'id': 1, 'metadata': {'operational_mode': 'GATE'}},
            {'id': 2, 'metadata': {'operational_mode': None}},
            {'id': 3, 'metadata': {'always_escalates': 'true'}},
            {'id': 4, 'metadata': {'always_escalates': 'false'}},
            {'id': 5, 'metadata': {'always_escalates': 1}},
        ])

        result = extract_human_gated_task_ids(tree.active_tasks)

        assert result == [], (
            'the operational_mode match must be exact == "gate" and the '
            'always_escalates match must be strict `is True`, so non-bool '
            f'truthy values do not satisfy the gate; got {result!r}'
        )

    def test_skips_non_dict_tasks_metadata_and_none_ids_without_raising(self):
        """Non-dict elements, non-dict metadata, and None ids are all skipped."""
        tree = FilteredTaskTree(active_tasks=[
            'not-a-dict',  # type: ignore[list-item]
            None,  # type: ignore[list-item]
            {'id': 1, 'metadata': 'operational_mode=gate'},
            {'id': 2, 'metadata': ['gate']},
            {'id': 3, 'metadata': None},
            {'id': 4},
            {'id': None, 'metadata': {'operational_mode': 'gate'}},
            {'metadata': {'always_escalates': True}},
            {'id': 12, 'metadata': {'operational_mode': 'gate'}},
        ])

        result = extract_human_gated_task_ids(tree.active_tasks)

        assert result == ['12'], (
            'malformed elements must be skipped without raising — a spurious id '
            f'would rewrite an unrelated finding; got {result!r}'
        )

    def test_coerces_int_ids_and_returns_sorted_deduped(self):
        """int ids coerce to str; the result is sorted and deduped."""
        tree = FilteredTaskTree(active_tasks=[
            {'id': 652, 'metadata': {'always_escalates': True}},
            {'id': '645', 'metadata': {'operational_mode': 'gate'}},
            {'id': 645, 'metadata': {'operational_mode': 'gate'}},
        ])

        result = extract_human_gated_task_ids(tree.active_tasks)

        assert result == ['645', '652'], (
            'int and str spellings of one id must collapse, and the result must '
            f'be sorted; got {result!r}'
        )

    def test_empty_input_returns_empty_list(self):
        """No active tasks -> []."""
        assert extract_human_gated_task_ids(FilteredTaskTree().active_tasks) == []
        assert extract_human_gated_task_ids([]) == []


class TestCanonicalHumanGateAction:
    """The one sentence both halves quote, its heading, and its prompt section.

    ONE OWNER (INV-5).  The prompt tells Stage 1 what to write; the
    post-processor writes exactly that same sentence when Stage 1 does not.
    If the two texts could drift, an operator reading a corrected finding
    would see wording the stage was never given — so the constant is asserted
    to be embedded VERBATIM in the rendered section, not paraphrased into it.

    These are wiring/contract assertions on a code-owned constant, not prose
    pins.  ``test_recon_gate_closure_guidance.py``'s
    TestReconStageEscalationServerIdentity(c) records a prose substring pin
    that was tried and deliberately REMOVED because it passed green on a
    reworded mis-instruction and failed red on a harmless reword.
    """

    def test_canonical_action_is_a_nonempty_str_with_the_mandated_phrase(self):
        """The sentence exists and carries the phrase the norm is named for."""
        assert isinstance(CANONICAL_HUMAN_GATE_ACTION, str), (
            f'CANONICAL_HUMAN_GATE_ACTION must be a str, got {type(CANONICAL_HUMAN_GATE_ACTION)!r}'
        )
        assert CANONICAL_HUMAN_GATE_ACTION.strip(), (
            'CANONICAL_HUMAN_GATE_ACTION must be non-empty — it is prepended '
            'verbatim onto corrected findings'
        )
        assert 'awaiting human-operator sign-off only' in CANONICAL_HUMAN_GATE_ACTION, (
            'the canonical sentence must carry the mandated phrase "awaiting '
            'human-operator sign-off only" — that phrase is what tells a '
            'Stage-2 reader the finding is evidence, not an instruction; got '
            f'{CANONICAL_HUMAN_GATE_ACTION!r}'
        )

    def test_canonical_action_carries_no_braces(self):
        """No brace in any form: it is interpolated into an f-string prompt.

        ``STAGE1_SYSTEM_PROMPT`` is an f-string, so a bare brace fails at
        import; and ``test_stage1_consolidation_guidance.py`` additionally
        asserts that ``{{``/``}}`` never survive into the RENDERED text, so
        over-escaping fails too.  Carrying no brace at all satisfies both.
        """
        assert '{' not in CANONICAL_HUMAN_GATE_ACTION and '}' not in CANONICAL_HUMAN_GATE_ACTION, (
            'the canonical sentence must contain no brace at all: a bare brace '
            'breaks the STAGE1_SYSTEM_PROMPT f-string at import, and a doubled '
            'brace breaks test_stage1_consolidation_guidance.py'
        )

    def test_heading_constant_is_exported_and_well_formed(self):
        """The section heading is a constant so tests locate it without pinning prose.

        Follows the ``EXECUTING_A_CLUSTER_FOLD_HEADING`` precedent in
        ``prompts/stage1.py``: a pure rename with a byte-identical body must
        not turn the wiring pins red for no behavioural reason.
        """
        assert isinstance(GATE_OWNED_ACTION_NORM_HEADING, str), (
            'GATE_OWNED_ACTION_NORM_HEADING must be a str'
        )
        assert GATE_OWNED_ACTION_NORM_HEADING.startswith('## '), (
            'the heading must be a top-level markdown section heading so it '
            'sits at a prompt section boundary; got '
            f'{GATE_OWNED_ACTION_NORM_HEADING!r}'
        )
        assert GATE_OWNED_ACTION_NORM_HEADING.strip() != '##', (
            'the heading must carry a title, not just the marker'
        )

    def test_render_returns_a_section_starting_with_its_heading(self):
        """The renderer emits one section, headed by the exported constant."""
        section = render_gate_owned_action_norm()

        assert isinstance(section, str) and section.strip(), (
            'render_gate_owned_action_norm() must return a non-empty str'
        )
        assert section.startswith(GATE_OWNED_ACTION_NORM_HEADING), (
            'the rendered section must start with GATE_OWNED_ACTION_NORM_HEADING '
            'so it lands at a prompt section boundary; got '
            f'{section[:80]!r}'
        )

    def test_render_embeds_the_canonical_sentence_verbatim_exactly_once(self):
        """One owner: the section quotes the constant, it does not reword it."""
        section = render_gate_owned_action_norm()

        assert section.count(CANONICAL_HUMAN_GATE_ACTION) == 1, (
            'the rendered norm must embed CANONICAL_HUMAN_GATE_ACTION verbatim '
            'exactly once — a reworded second copy is exactly the drift the '
            'single-owner design (INV-5) exists to prevent, and would make the '
            'deterministic normalizer write wording the stage was never given'
        )

    def test_render_names_both_gate_metadata_keys(self):
        """The stage must be told which structured fact makes a task gate-owned."""
        section = render_gate_owned_action_norm()

        for key in ('operational_mode', 'always_escalates'):
            assert key in section, (
                f'the rendered norm must name metadata.{key} — those two keys '
                'are the whole selection predicate, and a stage told only "a '
                'gate task" cannot tell which findings the rule covers'
            )

    def test_render_names_the_carve_out_flag_type(self):
        """The carve-out is named by the IMPORTED constant, never re-spelled."""
        section = render_gate_owned_action_norm()

        assert GATE_RESOLUTION_FLAG_TYPE in section, (
            'the rendered norm must name the carve-out flag type '
            f'({GATE_RESOLUTION_FLAG_TYPE!r}) so the prompt half agrees with '
            'the code half: build_gate_resolution_flag legitimately asks Stage '
            '2 to transcribe a ruling a human curator already recorded, and a '
            'blanket rule would contradict that live sibling flag on its first '
            'cycle'
        )

    def test_render_carries_no_token_banned_from_stage1(self):
        """Every token a live suite bans from the assembled Stage-1 prompt.

        Each assertion names the suite that bans it, so a future editor of
        this text sees WHY the token is unavailable rather than discovering it
        as an unexplained red elsewhere.
        """
        section = render_gate_owned_action_norm()

        banned = [
            ('escalate_blocker',
             'test_stages.py bans the bare substring in the Stage 1/3 prompts'),
            ('{',
             'STAGE1_SYSTEM_PROMPT is an f-string (a bare brace fails at import) '
             'and test_stage1_consolidation_guidance.py bans doubled braces in '
             'the rendered text'),
            ('}',
             'STAGE1_SYSTEM_PROMPT is an f-string (a bare brace fails at import) '
             'and test_stage1_consolidation_guidance.py bans doubled braces in '
             'the rendered text'),
            ('source_finding_id',
             'test_finding_provenance_prompt_guidance.py pins a DERIVED-COUNT '
             'invariant that any new mention of this key breaks'),
            ('related_memory_ids',
             'test_finding_provenance_prompt_guidance.py pins a DERIVED-COUNT '
             'invariant that any new mention of this key breaks'),
            ('as_submit_task_kwargs',
             'test_recon_self_model.py asserts this against the whole Stage-1 '
             'prompt precisely so a future section cannot reintroduce it'),
            ('write_entity_standing_decision',
             'test_recon_report_guidance_drift.py bans this Stage-1-denied '
             'stage-gated tool from the Stage-1 prompt'),
        ]
        for token, why in banned:
            assert token not in section, (
                f'the rendered norm must not contain {token!r}: {why}'
            )

    def test_render_carries_no_recon_report_tool_call_example(self):
        """No ``add_finding(``-shaped example at all.

        ``test_recon_report_guidance_drift.py`` runs a balanced-paren scan over
        every ``add_finding(``-shaped example in the assembled prompt and
        requires ``run_id`` in the args — and its extractor HARD-FAILS on
        unbalanced parens.  A phrasing rule needs no call example, which
        sidesteps the trap entirely.
        """
        section = render_gate_owned_action_norm()

        assert 'add_finding(' not in section, (
            'the norm states a phrasing rule and needs no recon-report call '
            'example; adding one puts this section under '
            "test_recon_report_guidance_drift.py's balanced-paren run_id scan "
            'for no benefit'
        )
