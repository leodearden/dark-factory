"""Tests for scripts/legibility/detectors.py -- the per-record detector layer
both the sampler and the digest reduce: the signal vocabulary, the carrier
cut, the text-pattern tables, the structured-error classifier and
record_hits.
"""
from __future__ import annotations

import dataclasses
import json
from datetime import date
from pathlib import Path

import pytest
from legibility import coder, detectors, digest, inventory, sampling
from legibility.detectors import AgentClass, CarrierKind, SignalClass
from legibility_carrier_fixtures import (
    PROSE_QUOTING_THE_SCHEMA,
    TRICKLE_CODER_PREAMBLE,
    assistant_record,
    briefing_text,
    coder_judgment,
    json_fenced,
    text_block,
    tool_result_record,
    tool_use_block,
    user_record,
    with_origin,
)

_CEILING_DECLARATION = 'WATCHER_REARM_OUTCOME: CEILING exit=124'
"""The sighted line verbatim (plans/confusion-census-2026-07-31.md:97)."""

_TEXT_CLASSES = (
    SignalClass.SELF_CORRECT, SignalClass.NOT_FOUND, SignalClass.DF_GUARD, SignalClass.INTERRUPT,
)


def _patterns(record, signal):
    return [hit.pattern for hit in detectors.record_hits(record).patterns[signal]]


# ---------------------------------------------------------------------------
# (a) The signal vocabulary, and the digest tables derived from it.
# ---------------------------------------------------------------------------

class TestSignalVocabulary:
    def test_signal_classes_are_the_frontmatter_contract_in_order(self):
        assert [c.value for c in SignalClass] == [
            'tool_error', 'self_correct', 'not_found', 'df_guard', 'interrupt',
            'designed_outcome',
        ]

    def test_confusion_classes_are_every_class_but_designed_outcome(self):
        expected = frozenset(SignalClass) - {SignalClass.DESIGNED_OUTCOME}

        assert expected == detectors.CONFUSION_CLASSES

    def test_digest_signal_count_keys_are_the_vocabulary(self):
        assert tuple(SignalClass) == digest.SIGNAL_COUNT_KEYS

    def test_signal_weights_cover_exactly_the_confusion_classes(self):
        assert set(digest.SIGNAL_WEIGHTS) == detectors.CONFUSION_CLASSES
        assert all(weight > 0 for weight in digest.SIGNAL_WEIGHTS.values())

    def test_user_turn_weight_is_gold(self):
        assert digest.USER_TURN_WEIGHT == 5.0

    def test_score_signals_values_are_unchanged(self):
        counts = {
            'tool_error': 2, 'self_correct': 1, 'not_found': 3,
            'df_guard': 1, 'interrupt': 1, 'designed_outcome': 7,
        }

        assert digest.score_signals(counts, 2) == 22.0
        assert digest.score_signals(dict.fromkeys(counts, 0), 0) == 0.0


# ---------------------------------------------------------------------------
# (b) record_hits: one positive per class, each from the digest's tables.
# ---------------------------------------------------------------------------

class TestRecordHitsPerClass:
    def test_self_correct_in_assistant_text(self):
        hits = detectors.record_hits(assistant_record(text_block("Hmm, that's wrong.")))

        [hit] = hits.patterns[SignalClass.SELF_CORRECT]
        assert hit == detectors.PatternHit(
            pattern="that's wrong", carrier=CarrierKind.ASSISTANT_TEXT, line="Hmm, that's wrong.",
        )

    def test_not_found_in_a_tool_result(self):
        hits = detectors.record_hits(
            tool_result_record('tu-1', 'cat: x.py: No such file or directory'),
        )

        [hit] = hits.patterns[SignalClass.NOT_FOUND]
        assert hit.pattern == 'no such file or directory'
        assert hit.carrier is CarrierKind.TOOL_RESULT

    @pytest.mark.parametrize(
        ('record', 'carrier'),
        [
            (tool_result_record('tu-1', 'BLOCKED: merge gate refused'), CarrierKind.TOOL_RESULT),
            (assistant_record(text_block('BLOCKED: merge gate refused')), CarrierKind.ASSISTANT_TEXT),
            (user_record('BLOCKED: merge gate refused'), CarrierKind.USER_TEXT),
        ],
        ids=['tool_result', 'assistant_text', 'user_text'],
    )
    def test_df_guard_in_each_of_its_three_carriers(self, record, carrier):
        [hit] = detectors.record_hits(record).patterns[SignalClass.DF_GUARD]

        assert hit.pattern == 'blocked:'
        assert hit.carrier is carrier

    def test_interrupt_as_a_user_turn(self):
        hits = detectors.record_hits(user_record('[Request interrupted by user]'))

        [hit] = hits.patterns[SignalClass.INTERRUPT]
        assert hit.pattern == 'request interrupted by user'
        assert hit.carrier is CarrierKind.USER_TEXT

    def test_tool_error_from_a_structured_error(self):
        hits = detectors.record_hits(
            tool_result_record('tu-1', 'boom, exit code 2', is_error=True),
        )

        assert hits.errors == (detectors.ErrorHit(
            tool_use_id='tu-1', content='boom, exit code 2', exit_code=2, designed_outcome=None,
        ),)
        assert hits.errors[0].signal_class is SignalClass.TOOL_ERROR
        assert hits.count(SignalClass.TOOL_ERROR) == 1
        assert hits.count(SignalClass.DESIGNED_OUTCOME) == 0

    def test_designed_outcome_from_a_declared_ceiling(self):
        hits = detectors.record_hits(
            tool_result_record('tu-1', _CEILING_DECLARATION, is_error=True),
        )

        [error] = hits.errors
        assert error.designed_outcome == 'watcher-rearm-declared'
        assert error.exit_code == 124
        assert error.signal_class is SignalClass.DESIGNED_OUTCOME
        assert hits.count(SignalClass.DESIGNED_OUTCOME) == 1
        assert hits.count(SignalClass.TOOL_ERROR) == 0

    def test_an_is_error_false_result_is_no_structured_error(self):
        hits = detectors.record_hits(tool_result_record('tu-1', 'boom, exit code 2'))

        assert hits.errors == ()


# ---------------------------------------------------------------------------
# (c) Carrier scoping and filtering.
# ---------------------------------------------------------------------------

class TestCarrierScoping:
    def test_not_found_reads_tool_results_only(self):
        record = assistant_record(text_block('that module was not found in the tree: does not exist'))

        assert detectors.record_hits(record).count(SignalClass.NOT_FOUND) == 0

    def test_self_correct_reads_assistant_text_only(self):
        record = tool_result_record('tu-1', "test output: that's wrong, my mistake")

        assert detectors.record_hits(record).count(SignalClass.SELF_CORRECT) == 0

    def test_a_sidechain_user_turn_is_no_interrupt_or_guard_carrier(self):
        record = user_record(
            '[Request interrupted by user]\nBLOCKED: merge gate refused', isSidechain=True,
        )
        hits = detectors.record_hits(record)

        assert hits.count(SignalClass.INTERRUPT) == 0
        assert hits.count(SignalClass.DF_GUARD) == 0

    def test_a_tool_use_input_never_yields_a_hit(self):
        literals = (
            "that's wrong / No such file or directory / BLOCKED: / "
            '[Request interrupted by user] / exit code 2'
        )
        record = assistant_record(
            tool_use_block('Write', {'file_path': 'fixture.txt', 'content': literals}),
        )
        hits = detectors.record_hits(record)

        assert all(hits.count(c) == 0 for c in SignalClass)

    def test_a_decoy_marker_suppresses_only_its_own_line(self):
        record = tool_result_record('tu-1', (
            'cat: a: No such file or directory  # decoy-fail\n'
            'cat: b: No such file or directory'
        ))

        [hit] = detectors.record_hits(record).patterns[SignalClass.NOT_FOUND]
        assert hit.line == 'cat: b: No such file or directory'

    def test_a_carrier_holding_only_decoy_lines_yields_nothing(self):
        record = tool_result_record('tu-1', 'BLOCKED: fixture data  # DECOY-FAIL')

        assert detectors.record_hits(record).count(SignalClass.DF_GUARD) == 0

    @pytest.mark.parametrize(
        'record',
        [
            assistant_record(text_block(coder_judgment(
                note="that's wrong: BLOCKED: no such file or directory, request interrupted by user",
            ))),
            user_record(briefing_text(
                body_filler='BLOCKED: merge gate refused\n[Request interrupted by user]',
            )),
            user_record(f'{TRICKLE_CODER_PREAMBLE}\n- (turn 3) BLOCKED: refused'),
            with_origin(
                user_record('BLOCKED: merge gate refused\n[Request interrupted by user]'),
                'task-notification',
            ),
        ],
        ids=['coder_judgment', 'briefing', 'trickle_coder_prompt', 'non_human_origin'],
    )
    def test_a_non_dialogue_carrier_yields_no_text_hit(self, record):
        hits = detectors.record_hits(record)

        assert all(hits.count(c) == 0 for c in _TEXT_CLASSES)

    def test_a_non_human_origin_record_still_yields_its_structured_error(self):
        record = with_origin(
            tool_result_record('tu-1', 'BLOCKED: refused, exit code 2', is_error=True),
            'task-notification',
        )
        hits = detectors.record_hits(record)

        assert hits.count(SignalClass.DF_GUARD) == 0
        assert hits.count(SignalClass.TOOL_ERROR) == 1

    def test_one_hit_per_pattern_per_carrier_on_its_first_line(self):
        record = tool_result_record('tu-1', (
            'step 1: BLOCKED: gate a\n'
            'step 2: BLOCKED: gate b\n'
            'step 3: done_gate_missing_files'
        ))

        hits = detectors.record_hits(record).patterns[SignalClass.DF_GUARD]

        assert [(h.pattern, h.line) for h in hits] == [
            ('blocked:', 'step 1: BLOCKED: gate a'),
            ('done_gate_missing_files', 'step 3: done_gate_missing_files'),
        ]

    def test_each_block_of_a_record_is_its_own_carrier(self):
        record = assistant_record(
            text_block('My mistake.'), text_block('my mistake again.'),
        )

        assert _patterns(record, SignalClass.SELF_CORRECT) == ['my mistake', 'my mistake']

    def test_count_is_the_hit_count_for_every_class(self):
        record = user_record([
            {'type': 'tool_result', 'tool_use_id': 'tu-1', 'is_error': True,
             'content': 'BLOCKED: x\nNo such file or directory\nexit code 2'},
            {'type': 'tool_result', 'tool_use_id': 'tu-2', 'is_error': True,
             'content': _CEILING_DECLARATION},
            {'type': 'text', 'text': '[Request interrupted by user]'},
        ])
        hits = detectors.record_hits(record)

        for signal in _TEXT_CLASSES:
            assert hits.count(signal) == len(hits.patterns[signal])
        assert hits.count(SignalClass.TOOL_ERROR) == sum(
            e.signal_class is SignalClass.TOOL_ERROR for e in hits.errors
        )
        assert hits.count(SignalClass.DESIGNED_OUTCOME) == sum(
            e.signal_class is SignalClass.DESIGNED_OUTCOME for e in hits.errors
        )
        assert [hits.count(c) for c in SignalClass] == [1, 0, 1, 1, 1, 1]

    def test_record_hits_is_frozen(self):
        hits = detectors.record_hits(user_record('hello'))
        field = dataclasses.fields(hits)[0].name

        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(hits, field, ())


# ---------------------------------------------------------------------------
# (d) is_non_dialogue_carrier -- the ONE composite question (provenance +
# content) every reader asks of a carrier before it counts as dialogue.
# ---------------------------------------------------------------------------

_ORDINARY_DIALOGUE = 'please fix the bug in the merge worker'
_PROSE_SELF_CORRECTION = "Actually, that's wrong — use the other branch"


class TestNonDialogueCarrierPredicate:
    @pytest.mark.parametrize(
        ('record', 'text'),
        [
            (with_origin(user_record(_ORDINARY_DIALOGUE), 'task-notification'), _ORDINARY_DIALOGUE),
            (assistant_record(text_block(coder_judgment())), coder_judgment()),
            (
                assistant_record(text_block(json_fenced(coder_judgment()))),
                json_fenced(coder_judgment()),
            ),
            (user_record(briefing_text()), briefing_text()),
            (user_record(TRICKLE_CODER_PREAMBLE), TRICKLE_CODER_PREAMBLE),
        ],
        ids=[
            'non_human_origin_alone_decides', 'coder_judgment', 'json_fenced_coder_judgment',
            'briefing', 'trickle_coder_prompt',
        ],
    )
    def test_a_non_dialogue_carrier_is_rejected(self, record, text):
        assert detectors.is_non_dialogue_carrier(record, text) is True

    @pytest.mark.parametrize(
        ('record', 'text'),
        [
            (with_origin(user_record(_ORDINARY_DIALOGUE), 'human'), _ORDINARY_DIALOGUE),
            (assistant_record(text_block(_PROSE_SELF_CORRECTION)), _PROSE_SELF_CORRECTION),
            (assistant_record(text_block(PROSE_QUOTING_THE_SCHEMA)), PROSE_QUOTING_THE_SCHEMA),
        ],
        ids=['human_origin', 'genuine_self_correction', 'prose_quoting_the_schema_inline'],
    )
    def test_this_sessions_own_dialogue_is_kept(self, record, text):
        assert detectors.is_non_dialogue_carrier(record, text) is False


# ---------------------------------------------------------------------------
# (e) classify_error_content -- the 07-31 census cluster 1.3 finding
# (plans/confusion-census-2026-07-31.md:97): watcher-rearm.sh's bounded-wait
# CEILING is a DESIGNED loop-continuation outcome that the extractor counted
# as a tool_error, inflating session a189558e's error signal 4x and burying
# the one genuine failure among false positives. Recognition is two-tier:
# the machine-readable self-declaration first (scripts/watcher-rearm.sh), the
# bare exit code second.
# ---------------------------------------------------------------------------

class TestDesignedOutcomeRecognition:
    @pytest.mark.parametrize(
        ('text', 'expected'),
        [
            ('failed, exit=2', (2, None)),
            ('command failed with exit code 2', (2, None)),
            ('exit status 137', (137, None)),
            ('Exit Code 2', (2, None)),
            ('something went wrong', (None, None)),
            ('exit code 2\nthe retry then died, exit status 137', (2, None)),
        ],
        ids=[
            'exit_equals_form', 'exit_code_prose_form', 'exit_status_prose_form',
            'case_insensitive', 'no_code_present', 'first_code_wins',
        ],
    )
    def test_extracts_the_exit_code_of_an_undeclared_failure(self, text, expected):
        assert detectors.classify_error_content(text) == expected

    def test_declared_ceiling_is_designed(self):
        exit_code, label = detectors.classify_error_content(_CEILING_DECLARATION)

        assert exit_code == 124
        assert label is not None

    def test_declared_fired_is_designed(self):
        _, label = detectors.classify_error_content('WATCHER_REARM_OUTCOME: FIRED exit=0')

        assert label is not None

    def test_declared_fired_reports_its_own_zero_exit_code(self):
        # exit=0 is falsy: the deciding-marker code must be selected with an
        # explicit None check, never a truthiness fallback.
        exit_code, _ = detectors.classify_error_content('WATCHER_REARM_OUTCOME: FIRED exit=0')

        assert exit_code == 0

    def test_declared_killed_is_a_genuine_failure(self):
        # scripts/watcher-rearm.sh classes 137|143|144 as KILLED -- a REAL
        # failure the census should still see, not a designed outcome.
        assert detectors.classify_error_content(
            'WATCHER_REARM_OUTCOME: KILLED exit=137',
        ) == (137, None)

    def test_declared_error_is_a_genuine_failure(self):
        # scripts/watcher-rearm.sh's catch-all ERROR branch.
        assert detectors.classify_error_content(
            'WATCHER_REARM_OUTCOME: ERROR exit=2',
        ) == (2, None)

    def test_bare_exit_124_is_a_designed_bounded_wait(self):
        _, label = detectors.classify_error_content('command timed out after 600s, exit=124')

        assert label is not None

    def test_ordinary_failure_is_not_designed(self):
        assert detectors.classify_error_content('boom, exit code 2') == (2, None)

    def test_empty_content_with_no_exit_code_is_not_designed(self):
        assert detectors.classify_error_content('') == (None, None)

    def test_declaration_and_bare_124_labels_are_distinct(self):
        # Which rule fired must stay legible to a digest reader.
        _, declared = detectors.classify_error_content(_CEILING_DECLARATION)
        _, bare = detectors.classify_error_content('timed out, exit=124')

        assert declared != bare

    def test_accepts_a_raw_block_list_content_field(self):
        # The public entry point takes a tool_result 'content' field as-is
        # (str OR list of blocks), so no caller needs a text flattener of
        # its own.
        blocks = [{'type': 'text', 'text': _CEILING_DECLARATION}]

        assert (
            detectors.classify_error_content(blocks)
            == detectors.classify_error_content(_CEILING_DECLARATION)
        )

    def test_bounded_wait_exit_code_constant_is_timeout_ceiling(self):
        assert detectors.BOUNDED_WAIT_EXIT_CODE == 124


class TestCoMingledDesignedAndGenuineContent:
    """A designed classification must hold for the WHOLE content.

    One tool_result can carry both a designed declaration and a real
    failure: a Bash call that loops scripts/watcher-rearm.sh, or that
    tails its log beside a failing command. The script writes its marker
    to STDERR, so it co-mingles with any other stderr in the same result.
    Classifying on the first designed marker alone deletes the genuine
    failure from BOTH ``signal_counts['tool_error']`` and the Error
    Neighborhoods section, and renders it under Designed Outcomes with the
    wrong exit code -- exactly the under-count that
    DESIGNED_OUTCOME_PATTERNS' own docstring refuses to trade for.
    """

    def test_ceiling_followed_by_an_error_marker_is_genuine(self):
        text = (
            'WATCHER_REARM_OUTCOME: CEILING exit=124\n'
            'WATCHER_REARM_OUTCOME: ERROR exit=2'
        )

        assert detectors.classify_error_content(text) == (2, None)

    def test_fired_followed_by_a_killed_marker_is_genuine(self):
        text = (
            'WATCHER_REARM_OUTCOME: FIRED exit=0\n'
            'WATCHER_REARM_OUTCOME: KILLED exit=137'
        )

        assert detectors.classify_error_content(text) == (137, None)

    def test_a_dissenting_marker_decides_even_when_it_comes_first(self):
        text = (
            'WATCHER_REARM_OUTCOME: ERROR exit=2\n'
            'WATCHER_REARM_OUTCOME: CEILING exit=124'
        )

        assert detectors.classify_error_content(text) == (2, None)

    def test_repeated_designed_markers_stay_designed(self):
        text = _CEILING_DECLARATION + '\nWATCHER_REARM_OUTCOME: CEILING exit=124'

        exit_code, label = detectors.classify_error_content(text)

        assert exit_code == 124
        assert label is not None

    def test_unaccounted_failure_code_beside_a_ceiling_is_genuine(self):
        # A declaration speaks only for its OWN invocation: a fatal code
        # that no declaration accounts for outvotes it.
        text = (
            'tail of watcher.log: WATCHER_REARM_OUTCOME: CEILING exit=124\n'
            'fatal: repo corrupt, exit code 128'
        )

        assert detectors.classify_error_content(text) == (128, None)

    def test_bare_ceiling_code_beside_a_failure_code_is_genuine(self):
        # The same rule one tier down: a bare 124 classifies only when it
        # is the ONLY outcome code in the content.
        text = 'poll timed out, exit=124\nthe retry then died, exit code 2'

        assert detectors.classify_error_content(text) == (2, None)


# ---------------------------------------------------------------------------
# (f) user_turn_text -- the one flattening of a user record's own text.
# ---------------------------------------------------------------------------

class TestUserTurnText:
    def test_string_content_is_the_text(self):
        assert detectors.user_turn_text(user_record('please redo it')) == 'please redo it'

    def test_text_blocks_are_joined_by_newlines(self):
        record = user_record([text_block('first'), text_block('second')])

        assert detectors.user_turn_text(record) == 'first\nsecond'

    def test_a_tool_result_only_record_has_no_turn_text(self):
        assert detectors.user_turn_text(tool_result_record('tu-1', 'output')) is None


# ---------------------------------------------------------------------------
# Boundary row 1: the sampler and the digest agree on class presence. Each
# committed transcript holds ONLY its stated class; the first eight are the
# discriminating cases the two scorers used to disagree on.
# ---------------------------------------------------------------------------

_CORPUS = Path(__file__).parent / 'fixtures' / 'legibility_detectors'

EXPECTED_PRESENT: dict[str, frozenset[SignalClass]] = {
    'bare_scope_violation.jsonl': frozenset(),
    'phantom_done.jsonl': frozenset(),
    'guard_tool_use.jsonl': frozenset(),
    'sampler_only_self_correct.jsonl': frozenset(),
    'not_found_outside_tool_result.jsonl': frozenset(),
    'interrupt_in_tool_result.jsonl': frozenset(),
    'reingested_coder_judgment.jsonl': frozenset(),
    'harness_briefing.jsonl': frozenset(),
    'tool_error.jsonl': frozenset({SignalClass.TOOL_ERROR}),
    'self_correct.jsonl': frozenset({SignalClass.SELF_CORRECT}),
    'not_found.jsonl': frozenset({SignalClass.NOT_FOUND}),
    'df_guard.jsonl': frozenset({SignalClass.DF_GUARD}),
    'interrupt.jsonl': frozenset({SignalClass.INTERRUPT}),
    'designed_outcome.jsonl': frozenset({SignalClass.DESIGNED_OUTCOME}),
}


class TestSamplerAndDigestAgreeOnTheCorpus:
    def test_the_corpus_is_exactly_the_table(self):
        assert {path.name for path in _CORPUS.glob('*.jsonl')} == set(EXPECTED_PRESENT)

    @pytest.mark.parametrize('signal', list(SignalClass), ids=str)
    @pytest.mark.parametrize('name', sorted(EXPECTED_PRESENT))
    def test_class_presence_agrees(self, name, signal):
        path = _CORPUS / name

        sampled = bool(sampling.scan_transcript(path).counts[signal])
        digested = bool(digest.signal_counts(digest.load_transcript(path))[signal])

        assert sampled == digested == (signal in EXPECTED_PRESENT[name])


# ---------------------------------------------------------------------------
# Boundary row 2: one agent-class classifier, read by the sampler's stratum
# and the digest's frontmatter alike.
# ---------------------------------------------------------------------------

_MAIN_DIR_PATH = Path('/root/-home-leo-src-dark-factory/sess.jsonl')

_WORKTREE_DIRS = (
    '-home-leo-src-dark-factory--worktrees-2573',
    '-home-leo-src-dark-factory--claude-worktrees-fix-foo',
    '-home-leo-src-reify--warm-lanes-worktrees-5187',
    '-home-leo-src-reify-build-worktrees-abc',
)


def _classify(*records):
    return detectors.classify_agent_class(_MAIN_DIR_PATH, list(records))


def _coder_prompt(digest_line):
    return coder.build_prompt(digest_text=digest_line, codebook_index='', invariant_slugs=())


_READ_OF_MARKER_SOURCE = (
    user_record('why did the sampler restratify my session?'),
    assistant_record(tool_use_block(
        'Read', {'file_path': '/home/leo/src/dark-factory/scripts/legibility/detectors.py'}, 'tu-read',
    )),
    tool_result_record('tu-read', (
        "    # orchestrator/src/orchestrator/dry_run_unblock.py injects the\n"
        "    # 'Task ID: {task_id}' / 'Worktree: {worktree}' preamble; its two lines\n"
        "    AgentMarker(AgentClass.RECON, ('operating in sleep mode', 'memory_consolidator')),\n"
    )),
)


class TestAgentClassVocabulary:
    def test_agent_classes_are_the_strata_plus_unknown(self):
        assert [c.value for c in AgentClass] == [
            'recon', 'curator-classifier', 'watcher', 'orchestrated-task', 'interactive',
            'legibility-instrument', 'unknown',
        ]

    def test_the_sampler_strata_are_every_class_but_unknown(self):
        assert sampling.STRATA == (
            'recon', 'curator-classifier', 'watcher', 'orchestrated-task', 'interactive',
            'legibility-instrument',
        )


class TestClassifyAgentClass:
    @pytest.mark.parametrize('path', [_MAIN_DIR_PATH, Path(f'/root/{_WORKTREE_DIRS[0]}/s.jsonl')])
    def test_empty_records_are_unknown_whatever_the_dir(self, path):
        assert detectors.classify_agent_class(path, []) is AgentClass.UNKNOWN

    @pytest.mark.parametrize('encoded', _WORKTREE_DIRS)
    def test_a_worktree_encoded_dir_wins_over_any_content_marker(self, encoded):
        records = [
            user_record('## Reconciliation Run\nStage 1: sync\n'),
            assistant_record(text_block('You are the task curator for the dark-factory orchestrator.')),
        ]

        assert detectors.classify_agent_class(
            Path(f'/root/{encoded}/sess.jsonl'), records,
        ) is AgentClass.ORCHESTRATED_TASK

    @pytest.mark.parametrize(
        ('text', 'expected'),
        [
            ('## Reconciliation Run\nStage 1: sync\n', AgentClass.RECON),
            ('## Stage 2: Task-Knowledge Sync\n## Project: know_live\n', AgentClass.RECON),
            ('Invoking memory_consolidator for this pass.', AgentClass.RECON),
            ('<command-name>recon-escalation-watcher</command-name>', AgentClass.WATCHER),
            (
                'You are a review suggestion classifier. You receive a numbered list...',
                AgentClass.CURATOR_CLASSIFIER,
            ),
        ],
        ids=[
            'reconciliation_run', 'stage_2_sync', 'memory_consolidator',
            'recon_escalation_watcher', 'review_suggestion_classifier',
        ],
    )
    def test_a_first_turn_marker_classifies(self, text, expected):
        assert _classify(user_record(text)) is expected

    @pytest.mark.parametrize(
        ('record', 'expected'),
        [
            (
                assistant_record(text_block('You are a Memory Consolidator agent operating in sleep mode.')),
                AgentClass.RECON,
            ),
            (
                tool_result_record('tu-1', 'resolved_by=orchestrator-escalation-watcher-auto'),
                AgentClass.WATCHER,
            ),
            (
                assistant_record(text_block('You are the task curator for the dark-factory orchestrator.')),
                AgentClass.CURATOR_CLASSIFIER,
            ),
            (
                assistant_record(text_block('You are a code module classifier.')),
                AgentClass.CURATOR_CLASSIFIER,
            ),
            (
                user_record(
                    'Task ID: 2572\nWorktree: /home/leo/src/dark-factory/.worktrees/2572\n',
                    isMeta=True,
                ),
                AgentClass.ORCHESTRATED_TASK,
            ),
        ],
        ids=['sleep_mode', 'watcher_auto', 'task_curator', 'code_module_classifier', 'task_preamble_in_meta'],
    )
    def test_a_marker_anywhere_in_the_raw_carriers_classifies(self, record, expected):
        assert _classify(user_record('please look at this'), record) is expected

    def test_the_first_turn_is_the_first_non_meta_user_turn(self):
        records = [
            user_record('<command-name>recon-escalation-watcher</command-name>', isMeta=True),
            user_record('You are a review suggestion classifier.'),
        ]

        assert _classify(*records) is AgentClass.CURATOR_CLASSIFIER

    def test_a_first_turn_marker_wins_over_a_different_haystack_marker(self):
        records = [
            user_record('<command-name>recon-escalation-watcher</command-name>'),
            assistant_record(text_block('You are a Memory Consolidator agent operating in sleep mode.')),
        ]

        assert _classify(*records) is AgentClass.WATCHER

    def test_a_harness_injected_first_turn_is_read_raw(self):
        record = user_record(briefing_text(body_filler='Invoking memory_consolidator for this pass.'))

        assert _classify(record) is AgentClass.RECON

    def test_matching_is_a_lowercase_substring_on_every_row(self):
        assert _classify(user_record('Context first.\n## RECONCILIATION RUN\n')) is AgentClass.RECON

    def test_no_marker_falls_back_to_interactive(self):
        assert _classify(user_record('just a normal conversation, please fix this bug')) is (
            AgentClass.INTERACTIVE
        )

    def test_a_lone_worktree_mention_stays_interactive(self):
        # Both task-preamble lines must co-occur: 'Worktree:' alone is
        # ordinary prose in this project.
        record = assistant_record(text_block(
            'The worktree: /home/leo/src/dark-factory/.worktrees/2572 is where this runs.'
        ))

        assert _classify(user_record('where does it run?'), record) is AgentClass.INTERACTIVE

    def test_a_require_all_marker_may_be_spread_across_carriers(self):
        records = [
            user_record('please look at this'),
            tool_result_record('tu-1', 'Task ID: 2572'),
            assistant_record(text_block('Worktree: /home/leo/src/dark-factory/.worktrees/2572')),
        ]

        assert _classify(*records) is AgentClass.ORCHESTRATED_TASK

    def test_a_read_of_source_quoting_the_marker_table_classifies_by_the_quote(self):
        """Decided, not incidental. The raw carriers include tool_results
        because a watcher's identity reaches its own session only as one
        (``resolved_by=orchestrator-escalation-watcher-auto``). The price
        is this case: an interactive main-checkout session that Reads a
        module quoting the markers is restratified by the quote. Telling a
        session's own markers from foreign material it reads is task
        5785's, which already names a session restratified by a Read of
        digest.py."""
        assert _classify(*_READ_OF_MARKER_SOURCE) is AgentClass.ORCHESTRATED_TASK

    def test_records_are_read_once_in_order_from_any_iterable(self):
        records = iter([
            user_record('please look at this'),
            assistant_record(text_block('You are a Memory Consolidator agent operating in sleep mode.')),
        ])

        assert detectors.classify_agent_class(_MAIN_DIR_PATH, records) is AgentClass.RECON


class TestAgentClassEvidence:
    def test_folding_records_one_at_a_time_is_classify_agent_class(self):
        records = [
            user_record('<command-name>recon-escalation-watcher</command-name>', isMeta=True),
            user_record('please check the queue'),
            tool_result_record('tu-1', 'resolved_by=orchestrator-escalation-watcher-auto'),
        ]
        evidence = detectors.AgentClassEvidence.for_path(_MAIN_DIR_PATH)

        for record in records:
            evidence = evidence.observe(record)

        assert evidence.agent_class is _classify(*records) is AgentClass.WATCHER
        assert evidence.first_turn_text == 'please check the queue'

    def test_no_record_yet_is_unknown_and_no_first_turn(self):
        evidence = detectors.AgentClassEvidence.for_path(_MAIN_DIR_PATH)

        assert evidence.agent_class is AgentClass.UNKNOWN
        assert evidence.first_turn_text is None

    def test_a_first_turn_without_text_is_empty_text(self):
        evidence = detectors.AgentClassEvidence.for_path(_MAIN_DIR_PATH).observe(
            tool_result_record('tu-1', 'ok'),
        )

        assert evidence.first_turn_text == ''

    def test_observing_returns_new_evidence_and_leaves_the_old_unchanged(self):
        before = detectors.AgentClassEvidence.for_path(_MAIN_DIR_PATH)

        after = before.observe(user_record('## Reconciliation Run\n'))

        assert before.agent_class is AgentClass.UNKNOWN
        assert after.agent_class is AgentClass.RECON


_ORCHESTRATED_TASK_PREAMBLE = 'Task ID: 6925\nWorktree: /home/leo/src/dark-factory/.worktrees/6925'


class TestLegibilityInstrumentSessions:
    @pytest.mark.parametrize(
        'digest_line',
        [
            _ORCHESTRATED_TASK_PREAMBLE,
            '<command-message>recon-escalation-watcher</command-message>',
            'You are a Memory Consolidator agent operating in sleep mode.',
            'You are the task curator for the dark-factory orchestrator.',
        ],
        ids=['orchestrated_task', 'watcher', 'recon', 'curator_classifier'],
    )
    def test_a_trickle_coder_prompt_is_classified_by_its_opener_not_its_embedded_digest(
        self, digest_line,
    ):
        assert _classify(user_record(_coder_prompt(digest_line))) is AgentClass.LEGIBILITY_INSTRUMENT

    def test_later_carriers_do_not_reclassify_an_instrument_session(self):
        records = [
            user_record(_coder_prompt('nothing quoted')),
            assistant_record(text_block(coder_judgment(
                note='resolved_by=orchestrator-escalation-watcher-auto',
            ))),
            tool_result_record('tu-1', 'operating in sleep mode'),
        ]

        assert _classify(*records) is AgentClass.LEGIBILITY_INSTRUMENT

    def test_leading_whitespace_before_the_opener_is_ignored(self):
        assert _classify(user_record('\n  ' + TRICKLE_CODER_PREAMBLE)) is (
            AgentClass.LEGIBILITY_INSTRUMENT
        )

    def test_the_opener_quoted_inside_a_first_turn_is_foreign(self):
        record = user_record(f'why is this prompt restratified?\n\n{TRICKLE_CODER_PREAMBLE}')

        assert _classify(record) is AgentClass.INTERACTIVE

    def test_the_opener_in_a_tool_result_is_foreign(self):
        records = [
            user_record('please read reingestion.py'),
            tool_result_record(
                'tu-1', "    'you are the trickle coder for the dark-factory agent-confusion codebook',",
            ),
        ]

        assert _classify(*records) is AgentClass.INTERACTIVE

    def test_the_opener_in_a_later_user_turn_is_not_the_first_turn(self):
        assert _classify(user_record('hello'), user_record(TRICKLE_CODER_PREAMBLE)) is (
            AgentClass.INTERACTIVE
        )

    def test_a_worktree_encoded_dir_still_wins_over_an_instrument_opener(self):
        path = Path(f'/root/{_WORKTREE_DIRS[0]}/s.jsonl')

        assert detectors.classify_agent_class(path, [user_record(TRICKLE_CODER_PREAMBLE)]) is (
            AgentClass.ORCHESTRATED_TASK
        )


_SHAPES = [
    pytest.param(_WORKTREE_DIRS[0], [user_record('please fix it')], 'orchestrated-task', id='worktree_path'),
    pytest.param(
        '-home-leo-src-dark-factory', [user_record('## Reconciliation Run\nStage 1: sync\n')],
        'recon', id='first_turn_recon',
    ),
    pytest.param(
        '-home-leo-src-dark-factory',
        [
            user_record('<command-name>recon-escalation-watcher</command-name>'),
            assistant_record(text_block('You are a Memory Consolidator agent operating in sleep mode.')),
        ],
        'watcher', id='first_turn_beats_haystack',
    ),
    pytest.param(
        '-home-leo-src-dark-factory',
        [
            user_record('please check the escalation queue'),
            tool_result_record('tu-1', 'resolved_by=orchestrator-escalation-watcher-auto'),
        ],
        'watcher', id='haystack_only_watcher',
    ),
    pytest.param(
        '-home-leo-src-dark-factory', list(_READ_OF_MARKER_SOURCE), 'orchestrated-task',
        id='read_of_marker_source',
    ),
    pytest.param(
        '-home-leo-src-dark-factory', [user_record('please fix the parser')], 'interactive',
        id='no_marker',
    ),
    pytest.param('-home-leo-src-dark-factory', [], 'unknown', id='empty'),
    pytest.param(
        '-home-leo-src-dark-factory',
        [
            user_record(_coder_prompt(_ORCHESTRATED_TASK_PREAMBLE)),
            assistant_record(text_block(coder_judgment())),
        ],
        'legibility-instrument', id='trickle_coder_session',
    ),
]


@pytest.mark.parametrize(('encoded', 'records', 'expected'), _SHAPES)
def test_sampler_stratum_and_digest_agent_class_agree(tmp_path, encoded, records, expected):
    directory = tmp_path / encoded
    directory.mkdir()
    path = directory / 'sess.jsonl'
    path.write_text(''.join(json.dumps(record) + '\n' for record in records))
    session = inventory.SessionRecord(
        path=path, encoded_dir=encoded, cwd='/home/leo/src/dark-factory',
        date=date(2026, 10, 11), size_bytes=path.stat().st_size,
    )

    stratum = sampling.score_session(session).stratum
    rendered = coder.parse_frontmatter(digest.build_digest(path))['agent_class']

    assert stratum == rendered == expected
