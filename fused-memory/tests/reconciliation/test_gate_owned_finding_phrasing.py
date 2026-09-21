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
- normalize_gate_owned_suggested_actions: the deterministic post-processor
  that prepends that sentence to every gate-owned finding, whatever the model
  wrote — the half that actually removes Stage 2's per-cycle catch.
"""

from __future__ import annotations

import copy

from fused_memory.reconciliation.curator_gate_resolution_sweep import (
    GATE_RESOLUTION_FLAG_TYPE,
    build_gate_resolution_flag,
)
from fused_memory.reconciliation.gate_owned_finding_phrasing import (
    CANONICAL_HUMAN_GATE_ACTION,
    CANONICAL_HUMAN_GATE_ACTION_MARKER,
    CURATOR_GATE_SWEEP_FLAG_KEY,
    GATE_OWNED_ACTION_NORM_HEADING,
    extract_human_gated_task_ids,
    normalize_gate_owned_suggested_actions,
    render_gate_owned_action_norm,
    stamp_curator_gate_sweep_provenance,
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

    def test_canonical_action_is_a_nonempty_str(self):
        """The sentence exists and is non-blank."""
        assert isinstance(CANONICAL_HUMAN_GATE_ACTION, str), (
            f'CANONICAL_HUMAN_GATE_ACTION must be a str, got {type(CANONICAL_HUMAN_GATE_ACTION)!r}'
        )
        assert CANONICAL_HUMAN_GATE_ACTION.strip(), (
            'CANONICAL_HUMAN_GATE_ACTION must be non-empty — it is prepended '
            'verbatim onto corrected findings'
        )

    def test_canonical_action_opens_with_the_exported_marker(self):
        """The constant is built from the marker, and the normalizer relies on it.

        Referential, not a prose pin: both sides are exported identifiers.
        ``normalize_gate_owned_suggested_actions`` tests compliance by looking
        for the MARKER alone, so a canonical sentence that stopped opening with
        it would make the pass non-idempotent — it would re-prepend the
        sentence onto its own output every cycle.
        """
        assert CANONICAL_HUMAN_GATE_ACTION.startswith(
            CANONICAL_HUMAN_GATE_ACTION_MARKER,
        ), (
            'CANONICAL_HUMAN_GATE_ACTION must open with '
            f'{CANONICAL_HUMAN_GATE_ACTION_MARKER!r}; got '
            f'{CANONICAL_HUMAN_GATE_ACTION[:60]!r}'
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


#: The ambiguous phrasing observed in autopilot_video run 8b2d3371 (Stage 1
#: finding 640d4ceb, cited task 645) — the exact defect this normalizer exists
#: to correct.
_AMBIGUOUS_ACTION = (
    'Stage 2 or operator should decide whether to enumerate the remaining '
    'members and extend the gate.'
)


def _gate_owned_flag(**overrides):
    """A minimal LLM-shaped finding citing gate task 645, plus *overrides*."""
    flag = {
        'description': 'Gate 645 has an unresolved membership question.',
        'severity': 'moderate',
        'actionable': True,
        'task_id': '645',
        'flag_type': 'task_memory_divergence',
        'category': 'task_memory_mismatch',
        'suggested_action': _AMBIGUOUS_ACTION,
    }
    flag.update(overrides)
    return flag


class TestNormalizeGateOwnedSuggestedActions:
    """normalize_gate_owned_suggested_actions(flags, gate_task_ids) -> (flags, count).

    Half B of task 4814 — the deterministic half, and the one that actually
    removes Stage 2's per-cycle catch.  It keys on the STRUCTURED gate fact
    rather than on the model's prose: "is this phrasing ambiguous?" has no
    reliable regex, while gate-ownership is a metadata fact already sitting in
    ``filtered_task_tree.active_tasks``.  Keying on the fact means the
    correction fires on every gate-owned finding regardless of how the model
    phrased it — including phrasings nobody has seen yet — where a wording
    detector would only catch the three offenders the task enumerates.

    It never drops, never reorders, and never rewrites the model's own text:
    the canonical sentence is PREPENDED and the original follows, which keeps
    the evidence intact and makes the transformation reviewable and
    idempotent.
    """

    def test_prepends_canonical_action_for_a_gate_owned_task_id(self):
        """A flag whose top-level task_id is a gate id is corrected and counted."""
        result, count = normalize_gate_owned_suggested_actions(
            [_gate_owned_flag()], ['645'],
        )

        assert count == 1, f'one gate-owned flag must count 1, got {count!r}'
        action = result[0]['suggested_action']
        assert action.startswith(CANONICAL_HUMAN_GATE_ACTION), (
            'the canonical sentence must lead the suggested_action so a Stage-2 '
            f'reader meets it first; got {action[:120]!r}'
        )
        assert _AMBIGUOUS_ACTION in action, (
            "the model's original text must be PRESERVED after the prefix — it "
            'is the finding evidence, and dropping it would make the correction '
            'lossy and unreviewable'
        )
        assert result[0]['gate_owned_action_normalized'] is True, (
            'a corrected flag must carry gate_owned_action_normalized=True so '
            'the divergence from the durable recon_report row is explicit and '
            'greppable rather than silent'
        )

    def test_normalizes_via_cited_tasks_when_task_id_is_none(self):
        """The run-8b2d3371 shape: gate id reachable only through cited_tasks."""
        flag = _gate_owned_flag(
            task_id=None,
            cited_tasks=[
                {'project_id': 'autopilot_video', 'task_id': '645', 'title': 'Gate'},
            ],
        )

        result, count = normalize_gate_owned_suggested_actions([flag], ['645'])

        assert count == 1, (
            'cited_tasks is the AUTHORITATIVE dedup key and a finding may carry '
            'its task only there — reading task_id alone would miss the exact '
            f'shape observed in run 8b2d3371; got {count!r}'
        )
        assert result[0]['suggested_action'].startswith(CANONICAL_HUMAN_GATE_ACTION)

    def test_matches_a_comma_joined_composite_task_id(self):
        """FINDING_ITEM_SCHEMA documents a comma-joined multi-id task_id shape."""
        result, count = normalize_gate_owned_suggested_actions(
            [_gate_owned_flag(task_id='645,652')], ['652'],
        )

        assert count == 1, (
            'a composite task_id must match when EITHER component is a gate id '
            '— the finding still cites a human gate; got '
            f'{count!r}'
        )
        assert result[0]['suggested_action'].startswith(CANONICAL_HUMAN_GATE_ACTION)

    def test_leaves_a_non_gate_finding_byte_identical(self):
        """A finding citing only non-gate tasks is returned untouched."""
        flag = _gate_owned_flag(task_id='4242')
        original = copy.deepcopy(flag)

        result, count = normalize_gate_owned_suggested_actions([flag], ['645', '652'])

        assert count == 0, f'no gate-owned flag means count 0, got {count!r}'
        assert result[0] == original, (
            'a non-gate finding must be returned byte-identical — over-selection '
            f'would rewrite a finding no human owns; got {result[0]!r}'
        )
        assert 'gate_owned_action_normalized' not in result[0], (
            'the marker must not be stamped on an untouched flag'
        )

    def test_is_idempotent_across_repeated_passes(self):
        """Feeding the output back in changes nothing and counts 0.

        Matters because the normalizer runs once per cycle over findings that
        can persist across cycles; a non-idempotent prefix would accrete
        copies of the sentence until the suggested_action was unreadable.
        """
        once, first_count = normalize_gate_owned_suggested_actions(
            [_gate_owned_flag()], ['645'],
        )
        twice, second_count = normalize_gate_owned_suggested_actions(once, ['645'])

        assert first_count == 1 and second_count == 0, (
            'the second pass must be a no-op; got '
            f'{first_count!r} then {second_count!r}'
        )
        assert twice == once, 'a re-run must not alter an already-corrected flag'
        assert twice[0]['suggested_action'].count(CANONICAL_HUMAN_GATE_ACTION) == 1, (
            'the canonical sentence must appear exactly once, never doubled'
        )

    def test_sets_the_action_when_it_is_missing_none_or_blank(self):
        """A missing/None/empty suggested_action is SET to the canonical sentence."""
        for label, flag in (
            ('missing', _gate_owned_flag()),
            ('none', _gate_owned_flag(suggested_action=None)),
            ('blank', _gate_owned_flag(suggested_action='   ')),
        ):
            if label == 'missing':
                flag.pop('suggested_action')

            result, count = normalize_gate_owned_suggested_actions([flag], ['645'])

            assert count == 1, f'{label} suggested_action must still count 1, got {count!r}'
            assert result[0]['suggested_action'] == CANONICAL_HUMAN_GATE_ACTION, (
                f'a {label} suggested_action must become the canonical sentence '
                f'ALONE, with no dangling separator; got '
                f'{result[0]["suggested_action"]!r}'
            )

    def test_carves_out_the_real_curator_gate_resolution_flag(self):
        """The REAL build_gate_resolution_flag output is left untouched.

        Not a hand-written fixture: pinning against the real builder — through
        the real ``stamp_curator_gate_sweep_provenance``, as the Stage-1 call
        site does — means a future edit to either side is caught at the seam.
        That flag's suggested_action DELIBERATELY tells Stage 2 to set a gate
        task's status — but only to transcribe a ruling a human curator already
        recorded in Mem0, and it carries its own dismiss branch for the
        merely-curated case.  Without this carve-out, task 4814 would ship a
        rule that contradicts a live, deliberately-designed sibling flag on
        its very first cycle.
        """
        flag = stamp_curator_gate_sweep_provenance(
            [build_gate_resolution_flag('645', [{'id': 'mem-a'}])],
        )[0]
        original = copy.deepcopy(flag)

        assert flag['flag_type'] == GATE_RESOLUTION_FLAG_TYPE, (
            'guard on the fixture itself: this test only proves the carve-out '
            'if the real builder still emits the carved-out flag_type'
        )

        result, count = normalize_gate_owned_suggested_actions([flag], ['645'])

        assert count == 0, (
            f'the {GATE_RESOLUTION_FLAG_TYPE} carve-out must not count, got {count!r}'
        )
        assert result[0] == original, (
            'the curator-gate-resolution flag must be returned byte-identical — '
            'prepending "no stage may record a decision" to a flag whose whole '
            'purpose is to transcribe a recorded human ruling would contradict it'
        )

    def test_llm_authored_flag_with_the_carve_out_flag_type_is_still_normalized(self):
        """The carve-out needs sweep PROVENANCE, not just the flag_type string.

        ``GATE_RESOLUTION_FLAG_TYPE`` is the generic value
        ``'task_completed_not_reflected'`` and ``flag_type`` is free-form per
        FINDING_ITEM_SCHEMA — the model picks it — so keying the exemption on
        the string alone would let any LLM finding that happened to choose
        that natural-language-shaped value escape the whole correction while
        citing a human gate.
        """
        flag = _gate_owned_flag(flag_type=GATE_RESOLUTION_FLAG_TYPE)
        assert CURATOR_GATE_SWEEP_FLAG_KEY not in flag, (
            'guard on the fixture: this only tests the provenance requirement '
            'if the LLM-shaped flag carries no sweep stamp'
        )

        result, count = normalize_gate_owned_suggested_actions([flag], ['645'])

        assert count == 1, (
            'a finding merely CARRYING the carved-out flag_type, with no sweep '
            'provenance, must still be corrected — the carve-out exists for '
            "the sweep's own output, whose evidence is a ruling a human "
            f'curator already recorded; got {count!r}'
        )
        assert result[0]['suggested_action'].startswith(CANONICAL_HUMAN_GATE_ACTION)

    def test_foreign_project_citation_colliding_with_a_local_gate_id_is_not_normalized(self):
        """A cited task in ANOTHER project is not this project's gate.

        The gate ids come from the LOCAL project's ``active_tasks``, while
        ``cited_tasks`` entries carry their own required ``project_id`` and
        foreign citations are routine (see
        ``flag_dedup._resolve_live_cross_project_fix_task``).  Task ids are
        small per-project integers, so a collision is a matter of time — and
        prefixing "no reconciliation stage may decide this" onto a foreign,
        legitimately actionable task is exactly the over-selection the
        selector's strictness is meant to rule out.
        """
        flag = _gate_owned_flag(
            task_id=None,
            cited_tasks=[
                {'project_id': 'know_live', 'task_id': '645', 'title': 'Not a gate'},
            ],
        )
        original = copy.deepcopy(flag)

        result, count = normalize_gate_owned_suggested_actions(
            [flag], ['645'], project_id='autopilot_video',
        )

        assert count == 0, (
            "a foreign project's task 645 is not the local gate 645; got "
            f'{count!r}'
        )
        assert result[0] == original, (
            f'the foreign-citing finding must be byte-identical; got {result[0]!r}'
        )

    def test_local_and_project_less_citations_both_still_match(self):
        """Scoping must not cost the in-project or the project-less citation.

        A ``cited_tasks`` entry naming the running project matches, and so does
        one that omits ``project_id`` — the latter matching the top-level
        ``task_id`` channel, which carries no project at all.
        """
        for label, entry in (
            ('same project', {'project_id': 'autopilot_video', 'task_id': '645',
                              'title': 'Gate'}),
            ('no project_id', {'task_id': '645', 'title': 'Gate'}),
        ):
            result, count = normalize_gate_owned_suggested_actions(
                [_gate_owned_flag(task_id=None, cited_tasks=[entry])],
                ['645'],
                project_id='autopilot_video',
            )

            assert count == 1, f'the {label} citation must match; got {count!r}'
            assert result[0]['suggested_action'].startswith(CANONICAL_HUMAN_GATE_ACTION)

    def test_omitting_project_id_keeps_the_unscoped_behaviour(self):
        """No project argument means no project filtering — fail-open."""
        flag = _gate_owned_flag(
            task_id=None,
            cited_tasks=[
                {'project_id': 'know_live', 'task_id': '645', 'title': 'Foreign'},
            ],
        )

        result, count = normalize_gate_owned_suggested_actions([flag], ['645'])

        assert count == 1, (
            'a caller that passes no project_id must get the pre-scoping '
            f'behaviour, not silent under-correction; got {count!r}'
        )
        assert result[0]['suggested_action'].startswith(CANONICAL_HUMAN_GATE_ACTION)

    def test_top_level_task_id_is_matched_regardless_of_project(self):
        """The documented residual: the top-level channel carries no project.

        ``task_id`` is a bare (possibly comma-joined) id with nowhere to name a
        project, so it cannot be scoped.  Pinned so the residual is a recorded
        contract rather than an unexamined gap.
        """
        result, count = normalize_gate_owned_suggested_actions(
            [_gate_owned_flag(task_id='645')], ['645'], project_id='autopilot_video',
        )

        assert count == 1, (
            'the top-level task_id channel names no project, so it is matched '
            f'as a local id; got {count!r}'
        )

    def test_leaves_an_action_already_opening_with_the_marker_alone(self):
        """Compliance is judged on the MARKER, not on verbatim reproduction.

        The prompt asks the model to reproduce a 70-word sentence exactly —
        the least likely model outcome.  Testing for the whole sentence would
        answer a compliant paraphrase by prepending a near-duplicate of itself,
        spending ~470 characters of the shared ``_FLAGGED_ITEMS_CHAR_BUDGET``
        that ``_format_flagged`` drops later findings against.
        """
        paraphrase = (
            CANONICAL_HUMAN_GATE_ACTION_MARKER + ' task 645 is awaiting an '
            'operator ruling; no stage may settle it. Evidence follows.'
        )
        flag = _gate_owned_flag(suggested_action=paraphrase)
        original = copy.deepcopy(flag)

        result, count = normalize_gate_owned_suggested_actions([flag], ['645'])

        assert count == 0, (
            'an action already opening with the marker carries the signal, so '
            f'it must not be re-prefixed; got {count!r}'
        )
        assert result[0] == original, (
            f'the compliant finding must be byte-identical; got {result[0]!r}'
        )

    def test_passes_non_dict_elements_through_without_raising(self):
        """A malformed element must not cost the whole findings list."""
        flags = ['not-a-dict', None, _gate_owned_flag()]

        result, count = normalize_gate_owned_suggested_actions(flags, ['645'])

        assert count == 1, f'the one real gate-owned flag still counts, got {count!r}'
        assert result[0] == 'not-a-dict' and result[1] is None, (
            f'non-dict elements must pass through unchanged, got {result[:2]!r}'
        )

    def test_does_not_mutate_the_callers_input_dicts(self):
        """The in-scope flag is REPLACED by a shallow copy, never mutated in place.

        Load-bearing: ``report.items_flagged``'s ``_pre_filter_flags`` snapshot
        is a shallow list copy that ALIASES these same dicts, so an in-place
        rewrite would retroactively alter the pre-filter snapshot.
        """
        flag = _gate_owned_flag()
        original = copy.deepcopy(flag)

        result, _count = normalize_gate_owned_suggested_actions([flag], ['645'])

        assert flag == original, (
            'the caller\'s dict must be untouched; got '
            f'{flag!r}'
        )
        assert result[0] is not flag, (
            'the corrected flag must be a new dict, not the caller\'s object'
        )

    def test_empty_inputs_are_identity(self):
        """Empty gate ids and empty flags are both no-ops."""
        flag = _gate_owned_flag()

        result, count = normalize_gate_owned_suggested_actions([flag], [])
        assert count == 0 and result[0] == flag, (
            'no gate tasks in the tree means nothing to normalize'
        )

        result, count = normalize_gate_owned_suggested_actions([], ['645'])
        assert result == [] and count == 0

    def test_preserves_input_order_and_length(self):
        """The normalizer never drops and never reorders.

        It is a phrasing correction, not a filter — unlike every other member
        of the Stage-1 post-processor chain it sits in.
        """
        flags = [
            _gate_owned_flag(task_id='4242'),
            _gate_owned_flag(task_id='645'),
            _gate_owned_flag(task_id='9999'),
            _gate_owned_flag(task_id='652'),
        ]

        result, count = normalize_gate_owned_suggested_actions(flags, ['645', '652'])

        assert len(result) == len(flags), (
            f'length must be preserved, got {len(result)} from {len(flags)}'
        )
        assert [f['task_id'] for f in result] == ['4242', '645', '9999', '652'], (
            'input order must be preserved'
        )
        assert count == 2, f'exactly the two gate-owned flags count, got {count!r}'

    def test_leaves_the_dedup_signature_fields_untouched(self):
        """task_id, flag_type and cited_tasks are never modified.

        Those three ARE ``compute_flag_signature``'s key, so leaving them
        untouched is what makes running this after ``dedup_flags`` safe.
        """
        flag = _gate_owned_flag(
            cited_tasks=[
                {'project_id': 'autopilot_video', 'task_id': '645', 'title': 'Gate'},
            ],
        )

        result, _count = normalize_gate_owned_suggested_actions([flag], ['645'])

        for key in ('task_id', 'flag_type', 'cited_tasks'):
            assert result[0][key] == flag[key], (
                f'{key} must be untouched — it is part of compute_flag_signature, '
                'and altering it would perturb cross-cycle dedup, suppression '
                'and the stage1_flag_markers_acknowledged diff'
            )


class TestStampCuratorGateSweepProvenance:
    """stamp_curator_gate_sweep_provenance(flags) -> copies marked as sweep output.

    The provenance half of the carve-out.  It is called at the ONE Stage-1
    call site that appends ``sweep_resolved_curator_gates``' flags, which is
    the only place that knows where those dicts came from.
    """

    def test_stamps_the_key_on_a_copy_without_touching_the_input(self):
        """Copies rather than mutates, matching the normalizer's own pass."""
        flag = build_gate_resolution_flag('645', [{'id': 'mem-a'}])
        original = copy.deepcopy(flag)

        stamped = stamp_curator_gate_sweep_provenance([flag])

        assert stamped[0][CURATOR_GATE_SWEEP_FLAG_KEY] is True, (
            f'the sweep flag must carry {CURATOR_GATE_SWEEP_FLAG_KEY!r}=True; '
            f'got {stamped[0]!r}'
        )
        assert flag == original, "the caller's dict must be untouched"
        assert stamped[0] is not flag, 'the stamped flag must be a new dict'

    def test_passes_non_dict_elements_through_without_raising(self):
        """A malformed element must not cost the whole sweep batch."""
        stamped = stamp_curator_gate_sweep_provenance(['not-a-dict', None])

        assert stamped == ['not-a-dict', None], (
            f'non-dict elements must pass through unchanged; got {stamped!r}'
        )

    def test_empty_input_returns_empty_list(self):
        """The common case — a sweep that found nothing."""
        assert stamp_curator_gate_sweep_provenance([]) == []
