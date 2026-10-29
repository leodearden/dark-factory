"""Tests for scripts/legibility/reingestion.py -- the provenance predicates
that decide whether a carrier's text, or a record's provenance, is material
other than this session's own dialogue.

The gold-turn and signal-detector consequences of these predicates are
pinned where those readers live (test_legibility_digest.py,
test_legibility_detectors.py); this file pins the predicates alone.
"""
from __future__ import annotations

import json

import pytest
from legibility import coder, reingestion
from legibility_carrier_fixtures import (
    CENSUS_MEMORY_CONTEXT_TURN,
    CENSUS_QUOTE_OF_THE_ESCALATION_PAYLOAD,
    HUMAN_TURNS_NAMING_AN_INJECTED_PROMPT,
    NON_HUMAN_ORIGIN_IDS,
    NON_HUMAN_ORIGIN_TURNS,
    TRICKLE_CODER_PREAMBLE,
    UNKNOWN_PROVENANCE_TEXT,
    WATCHER_ROTATION_PROMPT,
    briefing_text,
    coder_judgment,
    json_fenced,
    merger_prompt,
    pre_5356_replan_prompt,
    prior_round_adjudicator_prompt,
    recon_run_review_text,
    steward_continuation_prompt,
    user_record,
    with_origin,
)
from shared.cli_invoke import CAP_HIT_RESUME_PROMPT, CRASH_RECOVERY_RESUME_PROMPT

# ---------------------------------------------------------------------------
# is_coder_judgment_payload -- the re-ingestion content classifier's
# machine-answer half (task 5685). A prior trickle-coder answer re-enters a
# transcript as ONE carrier holding the coder's whole {"matches",
# "candidates"} object (scripts/legibility/coder.py::build_prompt), and its
# notes quote the digest being coded, so every signal literal they carry
# used to fire as this session's own. The classifier PARSES the whole
# carrier, bare or inside one outer ```json fence; it never substring-matches
# or brace-slices, so prose that merely quotes the schema stays dialogue.
# ---------------------------------------------------------------------------

class TestCoderJudgmentPayloadClassifier:
    @pytest.mark.parametrize(
        'text',
        [
            coder_judgment(),
            '{"matches": [], "candidates": []}',
            json_fenced(coder_judgment()),
            f'\n\n  {coder_judgment()}  \n\n',
            json.dumps({'matches': [], 'candidates': [], 'rationale': 'nothing fits'}),
        ],
        ids=['bare', 'empty_judgment', 'json_fenced', 'surrounding_whitespace', 'extra_key'],
    )
    def test_whole_carrier_judgment_is_recognised(self, text):
        assert reingestion.is_coder_judgment_payload(text) is True

    def test_the_reply_the_coder_prompt_prescribes_is_recognised_lockstep(self):
        """LOCKSTEP with scripts/legibility/coder.py::build_prompt, the one
        copy of the response schema CODER_JUDGMENT_KEYS restates. The reply
        is read out of the prompt, never hand-built: a hand-built reply stays
        green when that schema is renamed, and a rename is the drift this
        pins."""
        prompt = coder.build_prompt(digest_text='', codebook_index='', invariant_slugs=())
        [prescribed_reply] = [line for line in prompt.splitlines() if line.startswith('{')]

        assert reingestion.is_coder_judgment_payload(prescribed_reply) is True

    def test_prose_quoting_the_schema_inline_is_not_a_payload(self):
        # A genuine self-correction that merely MENTIONS the schema is this
        # session's own dialogue and must never be suppressed.
        text = (
            'The coder answers with {"matches": [], "candidates": []} '
            "— that's wrong, it should be a single object per session."
        )

        assert reingestion.is_coder_judgment_payload(text) is False

    @pytest.mark.parametrize(
        'text',
        [
            '{"matches": []}',
            '{"candidates": []}',
            '[{"matches": [], "candidates": []}]',
            '{"results": [{"id": "ccf73ca4", "content": "Task 1470 wired /audit."}]}',
            'not json at all',
            '',
        ],
        ids=[
            'matches_only', 'candidates_only', 'top_level_array',
            'unrelated_object', 'non_json', 'empty',
        ],
    )
    def test_anything_but_a_judgment_object_is_not_a_payload(self, text):
        assert reingestion.is_coder_judgment_payload(text) is False

    def test_pathologically_nested_object_answers_false_without_raising(self):
        # json.loads raises RecursionError, which is not a ValueError, on
        # this input under CPython 3.13. The predicate runs on every carrier
        # of arbitrary transcripts, tool_results included, so it must
        # answer rather than abort the whole digest.
        text = '{"a":' * 200_000 + '1' + '}' * 200_000

        assert reingestion.is_coder_judgment_payload(text) is False


# ---------------------------------------------------------------------------
# is_reingested_content -- the ONE predicate every signal bucket consults
# (task 5685). It answers "is this carrier machine content re-ingested as
# session material rather than this session's own dialogue?", and its two
# known members are a prior coder judgment and a harness-injected prompt or
# briefing.
# ---------------------------------------------------------------------------

class TestReingestedContentClassifier:
    @pytest.mark.parametrize(
        'text', [coder_judgment(), json_fenced(coder_judgment())],
        ids=['bare', 'json_fenced'],
    )
    def test_a_coder_judgment_is_reingested(self, text):
        assert reingestion.is_reingested_content(text) is True

    @pytest.mark.parametrize(
        'text', [briefing_text(), CENSUS_MEMORY_CONTEXT_TURN, TRICKLE_CODER_PREAMBLE],
        ids=['briefing', 'census_memory_context', 'trickle_coder_prompt'],
    )
    def test_a_harness_prompt_or_briefing_is_reingested(self, text):
        assert reingestion.is_reingested_content(text) is True

    @pytest.mark.parametrize(
        'resume_prompt', [CAP_HIT_RESUME_PROMPT, CRASH_RECOVERY_RESUME_PROMPT],
        ids=['usage_limit', 'crash_recovery'],
    )
    def test_resume_prompt_is_reingested_lockstep(self, resume_prompt):
        # LOCKSTEP, as in test_resume_prompt_is_excluded_lockstep: asserted
        # against each canonical constant, never a restated literal.
        assert reingestion.is_reingested_content(resume_prompt) is True

    @pytest.mark.parametrize(
        'text',
        ['please fix the bug in the merge worker', "Actually, that's wrong — use the other branch", ''],
        ids=['ordinary_turn', 'genuine_self_correction', 'empty'],
    )
    def test_dialogue_is_not_reingested(self, text):
        assert reingestion.is_reingested_content(text) is False


# ---------------------------------------------------------------------------
# is_harness_injected_turn -- the briefing anchor+corroborator rule, the
# all-of heading sets, the prose preamble markers and the lone memory-context
# block. Its consequence for the gold section (iter_user_turns) is pinned in
# test_legibility_digest.py.
# ---------------------------------------------------------------------------

class TestHarnessInjectedTurnFilter:
    def test_census_sighted_memory_context_turn_is_excluded(self):
        # The 07-31 census cluster 1.1(b) sighting, verbatim
        # (plans/confusion-census-2026-07-31.md:85): a context-ONLY briefing
        # injection -- '# Context' + '## Project Context' + the memory-search
        # JSON dump. It carries NEITHER '## Agent Identity' NOR '# Task', so
        # the old all-three rule let it through into the gold section.
        assert reingestion.is_harness_injected_turn(CENSUS_MEMORY_CONTEXT_TURN) is True

    def test_human_double_hash_context_plus_conventions_is_retained(self):
        # '## Context' is NOT a briefing anchor: briefing.py emits only the
        # single-hash '# Context', while '## Context' + '## Conventions' is a
        # common human-authored markdown shape in this repo's own plans/ and
        # docs/ prose. Treating it as an anchor silently drops a genuine
        # human correction from the gold user_corrections section -- the
        # exact pathology the instrument exists to prevent.
        text = (
            '## Context\n\nWe are debugging the merge worker.\n\n'
            '## Conventions\n\nplease stop using tabs, this is wrong'
        )

        assert reingestion.is_harness_injected_turn(text) is False

    def test_human_task_plus_action_prose_turn_is_retained(self):
        # Two SINGLE-hash top-level headings is a human spec-writing shape,
        # not a briefing shape. Every role prompt template composes
        # '{context}\n\n{identity}\n\n# Task ... # Action ...', so a real
        # briefing always carries the '# Context' anchor and almost always
        # '## Agent Identity' too -- it is never just these two. Corroborators
        # are therefore '##'-level sub-blocks only; see
        # HARNESS_BRIEFING_SUBHEADINGS for the recall corner this declines.
        text = (
            '# Task\n\nRewrite the retry loop so it backs off.\n\n'
            '# Action\n\nStart with the tests, then the loop itself.\n'
        )

        assert reingestion.is_harness_injected_turn(text) is False

    def test_every_corroborator_is_a_double_hash_subblock(self):
        # Structural guard, so a future agent does not re-add a top-level
        # heading to the corroborator tuple: a '# '-level corroborator pairs
        # with the '# task' anchor to clear the >=2 threshold on its own,
        # which is exactly the human shape above.
        assert all(
            heading.startswith('## ') for heading in reingestion.HARNESS_BRIEFING_SUBHEADINGS
        )

    @pytest.mark.parametrize(
        'resume_prompt', [CAP_HIT_RESUME_PROMPT, CRASH_RECOVERY_RESUME_PROMPT],
        ids=['usage_limit', 'crash_recovery'],
    )
    def test_resume_prompt_is_excluded_lockstep(self, resume_prompt):
        # LOCKSTEP: asserted against each canonical constant, not a
        # restated literal, so a harness rewording of either resume prompt
        # turns this suite red instead of silently drifting out of
        # coverage. One parametrize row per resume prompt -- a future
        # resume prompt is covered by adding one more row.
        assert reingestion.is_harness_injected_turn(resume_prompt) is True

    def test_merger_prompt_is_excluded_lockstep(self, tmp_path):
        # LOCKSTEP: built by the REAL prompt builder, not a hand-copied
        # literal, the same way test_resume_prompt_is_excluded_lockstep
        # pins the cli_invoke constants -- so a future merger-prompt
        # rewording turns this red instead of silently dropping the role
        # out of coverage.
        #
        # The merger is the one dispatched role with NO memory block
        # (task 3659, D7), so it emits no '# Context' anchor and the
        # briefing anchor+corroborator rule cannot see it at all. Before
        # 3659 it was classified via that anchor; MERGER_HEADINGS is what
        # keeps merge-conflict dispatches (7 per 14 days, measured) out of
        # the digest's gold user_corrections section.
        assert reingestion.is_harness_injected_turn(merger_prompt(tmp_path)) is True

    def test_steward_continuation_prompt_is_excluded_lockstep(self, tmp_path):
        # LOCKSTEP via the REAL builder, like test_merger_prompt_is_excluded_lockstep.
        # The sighted records (reify 77fb6617 records 237 and 289) carry no
        # origin and no isMeta, so provenance cannot exclude them; only this
        # text rule can.
        assert reingestion.is_harness_injected_turn(steward_continuation_prompt(tmp_path)) is True

    def test_prior_round_adjudicator_prompt_is_excluded_lockstep(self):
        # LOCKSTEP via the REAL builder. Like the steward's, the sighted
        # records carry no origin and no isMeta, so only this text rule sees them.
        assert reingestion.is_harness_injected_turn(prior_round_adjudicator_prompt()) is True

    def test_pre_5356_replan_prompt_is_excluded(self):
        # A RETIRED shape held for the archived corpus, so it is restated, not built.
        assert reingestion.is_harness_injected_turn(pre_5356_replan_prompt()) is True

    def test_watcher_rotation_prompt_is_excluded(self):
        assert reingestion.is_harness_injected_turn(WATCHER_ROTATION_PROMPT) is True

    @pytest.mark.parametrize('text', HUMAN_TURNS_NAMING_AN_INJECTED_PROMPT)
    def test_human_turn_naming_an_injected_prompt_is_retained(self, text):
        assert reingestion.is_harness_injected_turn(text) is False

    def test_human_turn_quoting_the_census_sighting_is_retained(self):
        assert reingestion.is_harness_injected_turn(CENSUS_QUOTE_OF_THE_ESCALATION_PAYLOAD) is False

    def test_human_turn_with_one_merger_heading_is_retained(self):
        # The all-of guard MERGER_HEADINGS is matched under: a human turn
        # writing '# Action' (an ordinary spec-writing heading, emitted by
        # every role template and deliberately NOT a briefing corroborator
        # since task 3610) must stay gold.
        human = (
            '# Action\n\n'
            'Please resolve the conflict in briefing.py by hand -- the '
            'merge queue keeps picking the wrong side.\n'
        )

        assert reingestion.is_harness_injected_turn(human) is False


# ---------------------------------------------------------------------------
# The reconciliation judge's run-review prompt (confusion census 2026-07-31
# §1.1 facet (b)) is harness-INJECTED, exactly like the briefing block: one
# source class, two injectors, so its heading set lives in
# HARNESS_HEADING_SETS rather than behind a separate "pasted report"
# predicate that would assert a human typed it.
# ---------------------------------------------------------------------------

class TestRunReviewPromptTurnFilter:
    def test_run_review_prompt_is_classified_as_harness_injected(self):
        # The run-review block IS harness-injected -- the judge's own prompt,
        # recorded as the session's sole user record by `claude --print`
        # (verified on all five census sightings) -- so it belongs to
        # is_harness_injected_turn rather than to a separate predicate whose
        # name would assert a human typed it.
        assert reingestion.is_harness_injected_turn(recon_run_review_text()) is True
        assert reingestion.is_harness_injected_turn(briefing_text()) is True

    def test_each_harness_heading_set_is_matched_independently(self):
        # HARNESS_HEADING_SETS is an any()-over-sets fan-out with two real
        # entries: a turn carrying ONE set's headings is excluded without
        # needing the other's. Cross-set heading mixtures are not a match.
        assert reingestion.HARNESS_BRIEFING_HEADINGS in reingestion.HARNESS_HEADING_SETS
        assert reingestion.RECON_RUN_REVIEW_HEADINGS in reingestion.HARNESS_HEADING_SETS

        crossed = '\n\n'.join(
            ('# Context', '### Stage Reports', 'so what should I do here?'),
        )
        assert reingestion.is_harness_injected_turn(crossed) is False

    def test_both_heading_sets_use_the_same_line_anchoring(self):
        # One shared normalization, so the sets cannot silently diverge on
        # what counts as a heading line: leading and trailing whitespace and
        # case are stripped for all of them alike.
        for headings in reingestion.HARNESS_HEADING_SETS:
            padded = '\n'.join(f'  {heading.upper()}  ' for heading in headings)

            assert reingestion.is_harness_injected_turn(padded) is True
            assert reingestion.is_harness_injected_turn(' | '.join(headings)) is False


# ---------------------------------------------------------------------------
# has_non_human_origin (task 5956). Claude Code stamps a queued prompt with a
# structured top-level ``origin`` dict, and a background task's completion or
# failure arrives as an ordinary user record with isMeta ABSENT and
# origin={'kind': 'task-notification'} -- so no text rule saw it.
# ---------------------------------------------------------------------------

class TestNonHumanOriginFilter:
    @pytest.mark.parametrize(('kind', 'text'), NON_HUMAN_ORIGIN_TURNS, ids=NON_HUMAN_ORIGIN_IDS)
    def test_every_non_human_origin_kind_is_excluded(self, kind, text):
        # The text alone is ordinary dialogue to the content classifier, so
        # only the record's provenance can exclude it -- and the rule is
        # "anything but human", never an allowlist of known machine kinds.
        rec = with_origin(user_record(text), kind)

        assert reingestion.is_reingested_content(text) is False
        assert reingestion.has_non_human_origin(rec) is True

    def test_human_origin_record_quoting_a_notification_mid_prose_is_kept(self):
        text = 'why did the <task-notification> for the lint run say failed? please look again'
        rec = with_origin(user_record(text), 'human')

        assert reingestion.has_non_human_origin(rec) is False

    @pytest.mark.parametrize(
        'record',
        [
            user_record(UNKNOWN_PROVENANCE_TEXT),
            {**user_record(UNKNOWN_PROVENANCE_TEXT), 'origin': 'task-notification'},
            {**user_record(UNKNOWN_PROVENANCE_TEXT), 'origin': {}},
            {**user_record(UNKNOWN_PROVENANCE_TEXT), 'origin': {'kind': None}},
            {**user_record(UNKNOWN_PROVENANCE_TEXT), 'origin': {'body': 'x'}},
        ],
        ids=['absent', 'bare_string', 'empty_dict', 'none_kind', 'no_kind_key'],
    )
    def test_unknown_provenance_falls_through_to_text_rules(self, record):
        assert reingestion.has_non_human_origin(record) is False
