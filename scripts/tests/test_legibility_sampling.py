"""Tests for scripts/legibility/sampling.py — zero-LLM signal scorer +
stratified budget sampler (PRD §5.2 point 2, contract §7.4, boundary test §8.4).

Self-contained: does not import task α's ``digest.py``. Imported as
``from legibility import sampling`` (PEP-420 namespace package; see
test_legibility_config.py's module docstring for the import mechanics).
"""
from __future__ import annotations

import json
import textwrap
from datetime import date as dt_date
from pathlib import Path

from legibility import config as config_mod
from legibility import inventory
from legibility import sampling as mod

MAIN_CWD = '/home/leo/src/dark-factory'


def _write_transcript(path: Path, records: list[dict]) -> Path:
    path.write_text('\n'.join(json.dumps(r) for r in records) + '\n')
    return path


def _tool_error_record() -> dict:
    return {
        'type': 'user',
        'timestamp': '2026-07-13T10:00:00.000Z',
        'message': {
            'content': [
                {'type': 'tool_result', 'tool_use_id': 't1', 'is_error': True, 'content': 'boom'},
            ]
        },
    }


def _not_found_record() -> dict:
    return {
        'type': 'user',
        'timestamp': '2026-07-13T10:01:00.000Z',
        'message': {
            'content': [
                {
                    'type': 'tool_result',
                    'tool_use_id': 't2',
                    'is_error': False,
                    'content': 'cat: /tmp/x: No such file or directory',
                },
            ]
        },
    }


def _self_correct_record() -> dict:
    return {
        'type': 'assistant',
        'timestamp': '2026-07-13T10:02:00.000Z',
        'message': {
            'content': [
                {'type': 'text', 'text': "Wait, that's wrong -- let me reconsider my approach."},
            ]
        },
    }


def _df_guard_record() -> dict:
    # mcp__plan-tools__report_false_premise is a real dark_factory guard
    # tool (orchestrator/src/orchestrator/mcp/plan_tools.py) — a structural
    # tool_use.name match, not a text/substring guess.
    return {
        'type': 'assistant',
        'timestamp': '2026-07-13T10:03:00.000Z',
        'message': {
            'content': [
                {
                    'type': 'tool_use',
                    'id': 'tu1',
                    'name': 'mcp__plan-tools__report_false_premise',
                    'input': {'task_id': '2573', 'reason': 'premise invalid'},
                },
            ]
        },
    }


def _interrupt_record() -> dict:
    return {
        'type': 'user',
        'timestamp': '2026-07-13T10:04:00.000Z',
        'message': {'content': '[Request interrupted by user]'},
    }


def _clean_record(i: int) -> dict:
    return {
        'type': 'user',
        'timestamp': f'2026-07-13T10:{10 + i:02d}:00.000Z',
        'message': {'content': f'Please do the thing #{i}.'},
    }


class TestScoreSignalsAllClasses:
    """One planted marker per class scores exactly 1 for that class."""

    def _build(self, tmp_path: Path) -> Path:
        return _write_transcript(
            tmp_path / 'sess.jsonl',
            [
                _tool_error_record(),
                _not_found_record(),
                _self_correct_record(),
                _df_guard_record(),
                _interrupt_record(),
            ],
        )

    def test_tool_error(self, tmp_path):
        assert mod.score_signals(self._build(tmp_path)).tool_error == 1

    def test_not_found(self, tmp_path):
        assert mod.score_signals(self._build(tmp_path)).not_found == 1

    def test_self_correct(self, tmp_path):
        assert mod.score_signals(self._build(tmp_path)).self_correct == 1

    def test_df_guard(self, tmp_path):
        assert mod.score_signals(self._build(tmp_path)).df_guard == 1

    def test_interrupt(self, tmp_path):
        assert mod.score_signals(self._build(tmp_path)).interrupt == 1

    def test_total_signal_is_sum_of_classes(self, tmp_path):
        counts = mod.score_signals(self._build(tmp_path))
        assert counts.total_signal == (
            counts.tool_error
            + counts.not_found
            + counts.self_correct
            + counts.df_guard
            + counts.interrupt
        )
        assert counts.total_signal == 5


class TestScoreSignalsClean:
    def test_clean_transcript_scores_zero(self, tmp_path):
        path = _write_transcript(tmp_path / 'clean.jsonl', [_clean_record(i) for i in range(3)])
        counts = mod.score_signals(path)
        assert counts.total_signal == 0
        assert counts.tool_error == 0
        assert counts.not_found == 0
        assert counts.self_correct == 0
        assert counts.df_guard == 0
        assert counts.interrupt == 0


def _user_turn(text: str) -> dict:
    return {'type': 'user', 'isSidechain': False, 'message': {'content': text}}


class TestClassifyAgentClass:
    """classify_agent_class(record, path) maps to the 5 strata. *record* is
    the session's already-located first non-sidechain, non-meta user turn
    (or None); *path*'s PARENT DIRECTORY NAME is the encoded dir — its
    shape is checked first, before any content marker.
    """

    MAIN_DIR_PATH = Path('/root/-home-leo-src-dark-factory/sess.jsonl')

    def test_worktrees_encoded_dir_is_orchestrated_task(self):
        path = Path('/root/-home-leo-src-dark-factory--worktrees-2573/sess.jsonl')
        assert mod.classify_agent_class(_user_turn('anything'), path) == 'orchestrated-task'

    def test_claude_worktrees_variant_is_orchestrated_task(self):
        path = Path('/root/-home-leo-src-dark-factory--claude-worktrees-fix-foo/sess.jsonl')
        assert mod.classify_agent_class(_user_turn('anything'), path) == 'orchestrated-task'

    def test_reify_warm_lane_worktree_is_orchestrated_task(self):
        # Faithful encoding of /home/leo/src/reify/.warm-lanes/worktrees/5187
        # (encode_cwd maps '/', '.' and '_' all to '-'), embedding the
        # '-warm-lanes-worktrees-' substring (task 2612).
        path = Path('/root/-home-leo-src-reify--warm-lanes-worktrees-5187/sess.jsonl')
        assert mod.classify_agent_class(_user_turn('anything'), path) == 'orchestrated-task'

    def test_reify_build_worktree_is_orchestrated_task(self):
        # Embeds the '-reify-build-worktrees-' substring (task 2612).
        path = Path('/root/-home-leo-src-reify-build-worktrees-abc/sess.jsonl')
        assert mod.classify_agent_class(_user_turn('anything'), path) == 'orchestrated-task'

    def test_reconciliation_run_header_is_recon(self):
        text = '## Reconciliation Run\nStage 1: sync\n'
        assert mod.classify_agent_class(_user_turn(text), self.MAIN_DIR_PATH) == 'recon'

    def test_stage_2_task_knowledge_sync_header_is_recon(self):
        # Real marker observed in ~/.claude/projects transcripts (task 2573 premise research).
        text = '## Stage 2: Task-Knowledge Sync\n## Project: know_live\n'
        assert mod.classify_agent_class(_user_turn(text), self.MAIN_DIR_PATH) == 'recon'

    def test_memory_consolidator_marker_is_recon(self):
        text = 'Invoking memory_consolidator for this pass.'
        assert mod.classify_agent_class(_user_turn(text), self.MAIN_DIR_PATH) == 'recon'

    def test_recon_escalation_watcher_marker_is_watcher(self):
        # skills/recon-escalation-watcher is a real dark_factory skill.
        text = '<command-name>recon-escalation-watcher</command-name>'
        assert mod.classify_agent_class(_user_turn(text), self.MAIN_DIR_PATH) == 'watcher'

    def test_curator_classifier_marker_is_curator_classifier(self):
        # Verbatim opening of TRIAGE_SYSTEM_PROMPT
        # (orchestrator/src/orchestrator/agents/triage.py:131).
        text = 'You are a review suggestion classifier. You receive a numbered list...'
        assert mod.classify_agent_class(_user_turn(text), self.MAIN_DIR_PATH) == 'curator-classifier'

    def test_main_dir_freeform_human_turn_is_interactive(self):
        text = 'Can you help me fix this bug in the parser?'
        assert mod.classify_agent_class(_user_turn(text), self.MAIN_DIR_PATH) == 'interactive'

    def test_none_record_defaults_to_interactive(self):
        assert mod.classify_agent_class(None, self.MAIN_DIR_PATH) == 'interactive'

    def test_worktree_shape_wins_over_content_markers(self):
        # Encoded-dir shape is checked FIRST — even a recon-marker turn in
        # a worktree dir classifies as orchestrated-task, not recon.
        path = Path('/root/-home-leo-src-dark-factory--worktrees-2573/sess.jsonl')
        text = '## Reconciliation Run\n'
        assert mod.classify_agent_class(_user_turn(text), path) == 'orchestrated-task'


def _scored(session_id, stratum, counts, first_turn_text, size_bytes=1000):
    session = inventory.SessionRecord(
        path=Path(f'/root/-home-leo-src-dark-factory/{session_id}.jsonl'),
        encoded_dir='-home-leo-src-dark-factory',
        cwd=MAIN_CWD,
        date=dt_date(2026, 7, 13),
        size_bytes=size_bytes,
    )
    return mod.ScoredRecord(
        session=session, stratum=stratum, counts=counts, first_turn_text=first_turn_text
    )


class TestShapeFingerprint:
    """shape_fingerprint is a cheap hashable key: (stratum, signal-shape,
    normalized first-turn skeleton). Signal-SHAPE is a boolean
    presence-per-class pattern, not exact counts — recon clones fire the
    same classes night after night even when exact counts drift."""

    def test_clones_share_a_fingerprint(self):
        a = _scored('a', 'recon', mod.SignalCounts(tool_error=1), '## Reconciliation Run\nDate: 2026-07-10\n')
        b = _scored('b', 'recon', mod.SignalCounts(tool_error=3), '## Reconciliation Run\nDate: 2026-07-12\n')
        assert mod.shape_fingerprint(a) == mod.shape_fingerprint(b)

    def test_distinct_content_differs(self):
        a = _scored('a', 'recon', mod.SignalCounts(tool_error=1), '## Reconciliation Run\nDate: 2026-07-10\n')
        distinct = _scored(
            'distinct', 'recon',
            mod.SignalCounts(self_correct=1, not_found=1),
            'A completely different opening about something else entirely.',
        )
        assert mod.shape_fingerprint(a) != mod.shape_fingerprint(distinct)

    def test_different_stratum_differs_even_with_same_text_and_shape(self):
        a = _scored('a', 'recon', mod.SignalCounts(tool_error=1), 'same text')
        b = _scored('b', 'watcher', mod.SignalCounts(tool_error=1), 'same text')
        assert mod.shape_fingerprint(a) != mod.shape_fingerprint(b)


class TestDedupeShapes:
    """dedupe_shapes collapses near-duplicate clones to the highest-scoring
    representative per fingerprint; a structurally distinct record in the
    same stratum survives independently."""

    def _build(self):
        clone_low = _scored(
            'recon-a', 'recon', mod.SignalCounts(tool_error=1),
            '## Reconciliation Run\nDate: 2026-07-10\n',
        )
        clone_mid = _scored(
            'recon-b', 'recon', mod.SignalCounts(tool_error=2),
            '## Reconciliation Run\nDate: 2026-07-11\n',
        )
        clone_high = _scored(
            'recon-c', 'recon', mod.SignalCounts(tool_error=3),
            '## Reconciliation Run\nDate: 2026-07-12\n',
        )
        distinct = _scored(
            'recon-distinct', 'recon',
            mod.SignalCounts(self_correct=1, not_found=1),
            'A completely different opening about something else entirely.',
        )
        return [clone_low, clone_mid, clone_high, distinct]

    def test_clones_collapse_to_highest_scoring(self):
        deduped = mod.dedupe_shapes(self._build())
        assert {r.path.stem for r in deduped} == {'recon-c', 'recon-distinct'}

    def test_kept_clone_representative_is_the_highest_scoring_one(self):
        deduped = mod.dedupe_shapes(self._build())
        kept_clone = next(r for r in deduped if r.path.stem.startswith('recon-') and r.path.stem != 'recon-distinct')
        assert kept_clone.path.stem == 'recon-c'
        assert kept_clone.score == 3


class TestSampleResultAccounting:
    """``SampleResult`` accounts for EVERY record the sampler was handed.

    Task 3270. Two facts the operator-facing summary line (and the
    total-suppression predicate ``not selected and budget_skipped > 0``)
    rest on:

      (i) ``total_records`` is the number of records handed to the sampler
          — one per enumerated session in both production callers, which is
          what lets the summary label it "enumerated".
      (ii) CANDIDATE CONSERVATION: every record that reached the budget
           phase ends up in exactly one of ``selected`` or
           ``budget_skipped``. That equality is what makes ``selected == []
           and budget_skipped > 0`` provably mean "candidates existed and
           were ALL discarded on budget" rather than "there was nothing to
           sample" — so the predicate's premise is machine-checked here
           instead of assumed at the call site.
      (iii) TOTAL CONSERVATION: ``total_records`` equals the four drop
            counts plus ``len(selected)``, with NO unaccounted remainder,
            so the operator line balances.

    The main batch is deliberately mixed so every accounting bucket is
    non-zero: 2 zero-signal records dropped, 1 clone collapsed, 2
    candidates selected and 2 budget-skipped, out of 7 handed in. Costs are
    injected via ``cost_fn`` (the same convention as the sibling
    budget-algebra classes) so the squeeze is explicit and hand-derivable
    rather than dependent on real digest sizes.

    That batch is built so every stratum's survivors ALL reach the
    candidate stage, which is exactly why it cannot pin (iii) on its own —
    it never exercises the ``top_fraction``/``per_stratum_min`` narrowing.
    ``_build_narrowed_records`` is the second fixture that does, and the
    reason ``below_sampling_cut`` exists: with it missing, 14 of 17 records
    fell out of the accounting entirely and the summary line under-reported
    its own denominator (reviewer_comprehensive/observability-accuracy,
    task 3270 amendment pass).
    """

    _COST = staticmethod(lambda r: r.size_bytes)

    def _config(self, max_bytes):
        return config_mod.LegibilityConfig(
            project_id='dark_factory',
            project_root='/home/leo/src/dark-factory',
            escalation_port=8103,
            cwd_prefixes=['/home/leo/src/dark-factory'],
            budgets=config_mod.Budgets(max_daily_digest_bytes=max_bytes),
            sampling=config_mod.Sampling(top_fraction=0.12, per_stratum_min=2),
        )

    def _build_records(self):
        """7 records; every stratum's survivors all reach the candidate stage.

        - 'recon' (3): two clones (collapse to the higher scorer) + one
          structurally distinct record -> 2 survivors, both candidates,
          both reserved. Priced at 200_000 each, so the group cannot fit.
        - 'interactive' (4): two zero-signal records (dropped before the
          budget phase) + two distinct real-signal records -> 2 survivors,
          both candidates, both reserved. Priced at 1_000 each, so the
          group fits.
        """
        return [
            _scored(
                'recon-clone-a', 'recon', mod.SignalCounts(tool_error=3),
                '## Reconciliation Run\nDate: 2026-07-10\n', 200_000,
            ),
            _scored(
                'recon-clone-b', 'recon', mod.SignalCounts(tool_error=2),
                '## Reconciliation Run\nDate: 2026-07-11\n', 200_000,
            ),
            _scored(
                'recon-distinct', 'recon', mod.SignalCounts(self_correct=1, not_found=4),
                'A completely different opening about something else entirely.', 200_000,
            ),
            _scored('quiet-a', 'interactive', mod.SignalCounts(), 'nothing went wrong here', 1_000),
            _scored('quiet-b', 'interactive', mod.SignalCounts(), 'all green, no surprises', 1_000),
            _scored('human-a', 'interactive', mod.SignalCounts(tool_error=9), 'human turn one', 1_000),
            _scored('human-b', 'interactive', mod.SignalCounts(not_found=8), 'human turn two', 1_000),
        ]

    def _build_narrowed_records(self):
        """17 distinct real-signal records in ONE stratum, so the sampling
        cut — not the budget — is what excludes most of them.

        At the stock ``top_fraction=0.12`` the cut is
        ``max(ceil(0.12 * 17), per_stratum_min=2) == 3``, so 3 compete for
        budget and 14 never do. Every first turn is structurally distinct
        and every score non-zero, so neither the zero-signal filter nor
        ``dedupe_shapes`` removes any of them: ``below_sampling_cut`` is the
        ONLY bucket that can absorb the other 14.

        The openers differ by LEADING WORD, not by an index number:
        ``_normalize_first_turn`` rewrites every digit run to '#' before
        fingerprinting (that is how it absorbs a recon clone's date drift),
        so numbered variants of one sentence collapse to a single shape and
        would silently land in ``dedupe_collapsed`` instead.
        """
        openers = (
            'Kafka', 'Postgres', 'Redis', 'Terraform', 'Kubernetes', 'Envoy',
            'Bazel', 'Rust', 'Grafana', 'Airflow', 'Vitess', 'Ceph',
            'Nomad', 'Pulsar', 'Debezium', 'Flink', 'Clickhouse',
        )
        return [
            _scored(
                f'distinct-{index:02d}', 'interactive',
                mod.SignalCounts(tool_error=index),
                f'{opener} is behaving oddly and I need help working out why.',
                1_000,
            )
            for index, opener in enumerate(openers, start=1)
        ]

    def test_total_records_counts_every_record_handed_to_the_sampler(self):
        records = self._build_records()
        result = mod.stratified_sample(records, self._config(3_000), cost_fn=self._COST)
        assert result.total_records == len(records) == 7

    def test_every_candidate_is_either_selected_or_budget_skipped(self):
        records = self._build_records()
        result = mod.stratified_sample(records, self._config(3_000), cost_fn=self._COST)

        # The squeeze really did bite (otherwise the conservation law below
        # would hold trivially with budget_skipped == 0).
        assert result.zero_signal_dropped == 2
        assert result.dedupe_collapsed == 1
        assert {r.path.stem for r in result.selected} == {'human-a', 'human-b'}
        assert result.budget_skipped == 2

        # Derived through below_sampling_cut, not around it. This fixture
        # happens to narrow nothing, and asserting that explicitly is what
        # stops the derivation from being accidentally right.
        assert result.below_sampling_cut == 0
        candidates = (
            result.total_records
            - result.zero_signal_dropped
            - result.dedupe_collapsed
            - result.below_sampling_cut
        )
        assert candidates == 4
        assert len(result.selected) + result.budget_skipped == candidates

    def test_records_below_the_sampling_cut_are_counted_not_silently_dropped(self):
        records = self._build_narrowed_records()
        # Budget deliberately generous (17 x 1_000 would all fit): the ONLY
        # thing excluding records here is the stratum sampling cut, so a
        # non-zero budget_skipped would mean the fixture is not isolating
        # what it claims to.
        result = mod.stratified_sample(records, self._config(300_000), cost_fn=self._COST)

        assert result.total_records == 17
        assert result.zero_signal_dropped == 0
        assert result.dedupe_collapsed == 0
        assert result.budget_skipped == 0
        assert len(result.selected) == 3, 'ceil(0.12 * 17) == 3 candidates'
        assert result.below_sampling_cut == 14, (
            'the 14 survivors the cut left out must land in a bucket — before '
            'this counter existed they landed in none, and the operator line '
            'reported enumerated=17 against selected=3 with no explanation'
        )

    def test_the_summary_line_balances_when_the_sampling_cut_bites(self):
        """TOTAL CONSERVATION, on the fixture that actually narrows.

        This is the property an operator relies on when they read the line
        and ask "where did the other 14 go?": the five numbers must add up,
        or the gap reads as a second, undiagnosed suppression mode.
        """
        for records, max_bytes in (
            (self._build_narrowed_records(), 300_000),  # cut bites, budget does not
            (self._build_narrowed_records(), 2_500),    # both bite
            (self._build_records(), 3_000),             # dedupe + zero-signal + budget
            (self._build_records(), 10),                # total suppression
        ):
            result = mod.stratified_sample(records, self._config(max_bytes), cost_fn=self._COST)
            accounted = (
                result.zero_signal_dropped
                + result.dedupe_collapsed
                + result.below_sampling_cut
                + result.budget_skipped
                + len(result.selected)
            )
            assert accounted == result.total_records, (
                f'{result.total_records - accounted} record(s) unaccounted at '
                f'max_bytes={max_bytes}'
            )

    def test_the_sampling_cut_never_hides_a_night_from_the_predicate(self):
        """``below_sampling_cut`` must not open a NEW silent-suppression hole.

        The failure this task exists to end is "real signal found, nothing
        digested, night looks quiet". A third drop bucket is a fresh chance
        to reintroduce it: if the cut could swallow an entire night's
        survivors, the run would report ``selected=0 budget_skipped=0`` --
        exactly what a genuine no-change night reports -- and
        ``nightly._report_sample_outcome`` would stay silent.

        It cannot, structurally: ``candidate_count`` is floored at
        ``min(per_stratum_min, len(survivors))``, so ANY survivor yields at
        least one candidate, and every candidate lands in ``selected`` or
        ``budget_skipped``. Swept rather than argued, over stratum sizes
        0..17 x eight budget regimes including 0.
        """
        base = self._build_narrowed_records()
        for size in range(len(base) + 1):
            for max_bytes in (0, 1, 999, 1_000, 1_001, 2_000, 3_000, 300_000):
                result = mod.stratified_sample(
                    base[:size], self._config(max_bytes), cost_fn=self._COST,
                )
                looks_like_nothing_to_sample = (
                    not result.selected and result.budget_skipped == 0
                )
                had_real_signal = (
                    result.total_records
                    - result.zero_signal_dropped
                    - result.dedupe_collapsed
                ) > 0
                assert not (looks_like_nothing_to_sample and had_real_signal), (
                    f'n={size} max_bytes={max_bytes}: {result.below_sampling_cut} '
                    f'record(s) of real signal, yet the night reads as a genuine '
                    f'no-change night'
                )

    def test_the_operator_line_reports_the_sampling_cut(self):
        """The counter is worthless if it never reaches the operator."""
        result = mod.stratified_sample(
            self._build_narrowed_records(), self._config(300_000), cost_fn=self._COST,
        )
        line = mod.format_sample_summary_line(result, max_bytes=300_000)
        assert 'below_sampling_cut=14' in line

    def test_total_suppression_is_distinguishable_from_nothing_to_sample(self):
        """The predicate itself, on the two states it must tell apart."""
        records = self._build_records()

        # Budget squeezed below even the cheap group: real candidates existed
        # and every one was discarded on budget.
        suppressed = mod.stratified_sample(records, self._config(10), cost_fn=self._COST)
        assert suppressed.selected == []
        assert suppressed.budget_skipped == 4
        assert (not suppressed.selected and suppressed.budget_skipped > 0) is True

        # Nothing but zero-signal records: no candidate ever reached the
        # budget phase, so the SAME empty selection must NOT read as
        # suppression.
        quiet = [r for r in records if r.score == 0]
        quiet_result = mod.stratified_sample(quiet, self._config(300_000), cost_fn=self._COST)
        assert quiet_result.selected == []
        assert quiet_result.budget_skipped == 0
        assert quiet_result.total_records == 2
        assert (not quiet_result.selected and quiet_result.budget_skipped > 0) is False


class TestStratifiedSampleBoundary:
    """§8.4 boundary fixture: ~100x size variance, clone shapes, several
    zero-signal records, and one TINY stratum. Drives the PURE
    stratified_sample(records, config).

    Three strata, each isolating one property:
      - "recon" (N=17: 2 huge high-score/high-size + 3 mid + 12 zero-signal)
        tests zero-signal dropping and supplies the huge sessions that
        would otherwise starve other strata under a naive whole-pool
        greedy-by-score fill.
      - "watcher" (4 near-identical clones, scores 5/6/7/8) tests dedup:
        only the highest scorer (clone index 3, score 8) survives.
      - "interactive" (exactly 2 records) is the TINY stratum that must
        survive the byte budget despite recon's huge high-score sessions.

    Hand-derived expected outcome (see task 2573 step-15 design notes):
    recon's candidates = {600, 500, 29} (top max(ceil(0.12*17), 2)=3 of 5
    survivors) -> reserve = top min(per_stratum_min=2, 3) = {600, 500}
    (400_000 bytes), leftover = {29} (5_000 bytes). watcher's candidates =
    {8} (capped at 1 survivor after dedup) -> reserve = {8} (2_000 bytes),
    no leftover. interactive's candidates = {4, 3} -> reserve = both
    (1_700 bytes), no leftover. Total reserved = 403_700 bytes. Budget is
    405_000, leaving only 1_300 for the greedy-fill phase — not enough for
    recon's leftover 29-scorer (5_000 bytes), so it is excluded and the
    final selection is exactly the 5 reserved records.

    These tests are about the sampler's budget ALGEBRA — cheapest-floor-
    first, the overall cap holding, big sessions being unable to evict a
    whole stratum, per-stratum-min floors — which is exactly what task
    3268's units fix does NOT change. So they inject
    ``cost_fn=lambda r: r.size_bytes``: the fixture's ``size_bytes`` numbers
    now stand in as an explicit SYNTHETIC per-record cost rather than
    riding on the production default, and every hand-derived number above
    survives verbatim. (The production default is now a digest-byte basis;
    :class:`TestStratifiedSampleRealWorldSizes` covers that.)
    """

    _COST = staticmethod(lambda r: r.size_bytes)

    def _build_records(self):
        records = [
            _scored('recon-huge-1', 'recon', mod.SignalCounts(tool_error=500), 'recon run A', 200_000),
            _scored('recon-huge-2', 'recon', mod.SignalCounts(tool_error=600), 'recon run B', 200_000),
            _scored('recon-mid-1', 'recon', mod.SignalCounts(tool_error=25), 'recon mid run 1', 5_000),
            _scored('recon-mid-2', 'recon', mod.SignalCounts(tool_error=27), 'recon mid run 2', 5_000),
            _scored('recon-mid-3', 'recon', mod.SignalCounts(tool_error=29), 'recon mid run 3', 5_000),
        ]
        for i in range(12):
            records.append(
                _scored(f'recon-zero-{i}', 'recon', mod.SignalCounts(), f'recon zero {i}', 500)
            )
        for i, score in enumerate((5, 6, 7, 8)):
            records.append(
                _scored(
                    f'watcher-clone-{i}', 'watcher', mod.SignalCounts(tool_error=score),
                    'watcher polling cycle', 2_000,
                )
            )
        records.append(_scored('interactive-1', 'interactive', mod.SignalCounts(tool_error=3), 'human turn A', 800))
        records.append(_scored('interactive-2', 'interactive', mod.SignalCounts(tool_error=4), 'human turn B', 900))
        return records

    def _config(self):
        return config_mod.LegibilityConfig(
            project_id='dark_factory',
            project_root='/home/leo/src/dark-factory',
            escalation_port=8103,
            cwd_prefixes=['/home/leo/src/dark-factory'],
            budgets=config_mod.Budgets(max_daily_digest_bytes=405_000),
            sampling=config_mod.Sampling(top_fraction=0.12, per_stratum_min=2),
        )

    def _selected_ids(self, result):
        return {r.path.stem for r in result.selected}

    def test_zero_signal_dropped(self):
        result = mod.stratified_sample(self._build_records(), self._config(), cost_fn=self._COST)
        assert not any(sid.startswith('recon-zero-') for sid in self._selected_ids(result))

    def test_zero_signal_drop_count(self):
        result = mod.stratified_sample(self._build_records(), self._config(), cost_fn=self._COST)
        assert result.zero_signal_dropped == 12

    def test_clones_deduped_to_single_highest_scorer(self):
        result = mod.stratified_sample(self._build_records(), self._config(), cost_fn=self._COST)
        watcher_selected = [r for r in result.selected if r.stratum == 'watcher']
        assert len(watcher_selected) == 1
        assert watcher_selected[0].path.stem == 'watcher-clone-3'

    def test_no_duplicate_fingerprints_in_selection(self):
        result = mod.stratified_sample(self._build_records(), self._config(), cost_fn=self._COST)
        fingerprints = [mod.shape_fingerprint(r) for r in result.selected]
        assert len(fingerprints) == len(set(fingerprints))

    def test_each_nonempty_stratum_retains_per_stratum_min_or_all_survivors(self):
        result = mod.stratified_sample(self._build_records(), self._config(), cost_fn=self._COST)
        per_stratum = {}
        for r in result.selected:
            per_stratum.setdefault(r.stratum, []).append(r)
        assert len(per_stratum['recon']) >= 2
        assert len(per_stratum['watcher']) >= 1  # only 1 survivor exists after dedup
        assert len(per_stratum['interactive']) >= 2

    def test_total_bytes_within_budget(self):
        result = mod.stratified_sample(self._build_records(), self._config(), cost_fn=self._COST)
        assert sum(r.size_bytes for r in result.selected) <= 405_000
        assert result.bytes_used == sum(r.size_bytes for r in result.selected)

    def test_tiny_stratum_survives_despite_huge_high_score_sessions_elsewhere(self):
        result = mod.stratified_sample(self._build_records(), self._config(), cost_fn=self._COST)
        selected_ids = self._selected_ids(result)
        assert 'interactive-1' in selected_ids
        assert 'interactive-2' in selected_ids
        # The huge, high-scoring recon sessions really are present too —
        # this is exactly what would otherwise have starved the tiny
        # stratum under a naive whole-pool greedy-by-score fill.
        assert 'recon-huge-1' in selected_ids
        assert 'recon-huge-2' in selected_ids

    def test_budget_cap_excludes_recon_leftover_candidate(self):
        # Without per-stratum reserve, recon's mid-tier candidate (score 29)
        # would consume the leftover budget ahead of lower-scoring strata;
        # the tight budget here must exclude it.
        result = mod.stratified_sample(self._build_records(), self._config(), cost_fn=self._COST)
        assert 'recon-mid-3' not in self._selected_ids(result)

    def test_selected_ordered_by_score_desc(self):
        result = mod.stratified_sample(self._build_records(), self._config(), cost_fn=self._COST)
        scores = [r.score for r in result.selected]
        assert scores == sorted(scores, reverse=True)

    def test_per_stratum_counts_accounting(self):
        result = mod.stratified_sample(self._build_records(), self._config(), cost_fn=self._COST)
        assert result.per_stratum_counts == {'recon': 2, 'watcher': 1, 'interactive': 2}


class TestStratifiedSampleReserveExceedsBudget:
    """When per-stratum reserve floors collectively exceed the byte budget
    (a real ~/.claude/projects scenario found via manual acceptance
    testing: session sizes can dwarf a conservative daily budget), the
    OVERALL cap must still hold — cheapest stratum floor first, so budget
    pressure falls on the priciest stratum rather than the cap being
    silently blown.

    Like :class:`TestStratifiedSampleBoundary`, this tests the sampler's
    budget ALGEBRA, which task 3268's units fix does not change, so it
    injects ``cost_fn=lambda r: r.size_bytes`` to keep the fixture's
    ``size_bytes`` numbers as an explicit synthetic per-record cost and its
    hand-derived expectations verbatim."""

    _COST = staticmethod(lambda r: r.size_bytes)

    def _config(self, max_bytes):
        return config_mod.LegibilityConfig(
            project_id='dark_factory',
            project_root='/home/leo/src/dark-factory',
            escalation_port=8103,
            cwd_prefixes=['/home/leo/src/dark-factory'],
            budgets=config_mod.Budgets(max_daily_digest_bytes=max_bytes),
            sampling=config_mod.Sampling(top_fraction=0.12, per_stratum_min=2),
        )

    def _build_records(self):
        return [
            # "recon" floor: expensive (200_000 bytes each -> 400_000 total).
            _scored('recon-1', 'recon', mod.SignalCounts(tool_error=10), 'recon a', 200_000),
            _scored('recon-2', 'recon', mod.SignalCounts(tool_error=9), 'recon b', 200_000),
            # "interactive" floor: cheap (1_000 bytes each -> 2_000 total).
            _scored('interactive-1', 'interactive', mod.SignalCounts(tool_error=2), 'human a', 1_000),
            _scored('interactive-2', 'interactive', mod.SignalCounts(tool_error=1), 'human b', 1_000),
        ]

    def test_total_never_exceeds_budget_even_when_floors_alone_would(self):
        # Both floors together (402_000) exceed a 3_000-byte budget.
        result = mod.stratified_sample(self._build_records(), self._config(3_000), cost_fn=self._COST)
        assert sum(r.size_bytes for r in result.selected) <= 3_000
        assert result.bytes_used <= 3_000

    def test_cheap_stratum_preferred_when_budget_cannot_fit_both(self):
        result = mod.stratified_sample(self._build_records(), self._config(3_000), cost_fn=self._COST)
        selected_ids = {r.path.stem for r in result.selected}
        assert {'interactive-1', 'interactive-2'} <= selected_ids
        assert 'recon-1' not in selected_ids
        assert 'recon-2' not in selected_ids


class TestStratifiedSampleRealWorldSizes:
    """Realistically-sized sessions must still be selectable under the STOCK
    daily budget, with NO cost_fn injected.

    This pins the live defect. ``budgets.max_daily_digest_bytes`` is 300_000
    — a DIGEST-output budget sized for ~19 of the §7.2 15KB digests a night
    — but the sampler charged it against the raw ``.jsonl`` transcript size.
    Real reify sessions run 0.5-6.5MB, so EVERY ONE of the records below
    individually exceeds the entire nightly budget when charged at raw
    transcript size: each reserve group is skipped whole, the greedy
    leftover fill halts at its first record, and the selection comes back
    empty. That is exactly why the live nightly trickle selected nothing at
    all from 2026-07-16 to 2026-07-29.

    Charged in the right units the same sessions are trivially affordable:
    a measured 879,254-byte transcript of this shape renders to a
    15,123-byte digest, so the per-stratum floor of 2 costs at most 2 x
    15360 = 30_720 against 300_000. A non-empty selection is therefore
    guaranteed by construction, not hoped for.
    """

    def _stock_config(self):
        """The shipped budget/sampling numbers, verbatim — the point of these
        tests is that the STOCK configuration works on real session sizes."""
        return config_mod.LegibilityConfig(
            project_id='dark_factory',
            project_root='/home/leo/src/dark-factory',
            escalation_port=8103,
            cwd_prefixes=['/home/leo/src/dark-factory'],
            budgets=config_mod.Budgets(max_daily_digest_bytes=300_000),
            sampling=config_mod.Sampling(top_fraction=0.12, per_stratum_min=2),
        )

    def _single_stratum(self):
        # Distinct WORD stems, never digits: _normalize_first_turn collapses
        # digit runs to '#', so 'session 1'/'session 2' would fingerprint
        # identically and dedupe away.
        names = ('alpha', 'beta', 'gamma', 'delta', 'epsilon')
        return [
            _scored(
                f'orch-{name}', 'orchestrated-task',
                mod.SignalCounts(tool_error=score),
                f'orchestrated task session {name}',
                6_000_000,
            )
            for name, score in zip(names, (40, 35, 30, 25, 20))
        ]

    def _two_strata(self):
        records = self._single_stratum()[:4]
        for name, score in zip(('one', 'two', 'three'), (18, 14, 11)):
            records.append(
                _scored(
                    f'recon-{name}', 'recon', mod.SignalCounts(not_found=score),
                    f'reconciliation run {name}', 4_500_000,
                )
            )
        return records

    def test_multi_mb_sessions_are_still_selected(self):
        result = mod.stratified_sample(self._single_stratum(), self._stock_config())
        assert result.selected != []
        assert len(result.selected) >= 2
        assert result.budget_skipped == 0
        assert result.bytes_used <= 300_000

    def test_multi_mb_sessions_do_not_starve_a_second_stratum(self):
        result = mod.stratified_sample(self._two_strata(), self._stock_config())
        assert set(result.per_stratum_counts) == {'orchestrated-task', 'recon'}
        assert result.per_stratum_counts['orchestrated-task'] >= 2
        assert result.per_stratum_counts['recon'] >= 2
        assert result.budget_skipped == 0
        assert result.bytes_used <= 300_000

    def test_bytes_used_is_not_the_raw_transcript_size(self):
        # The units claim, stated directly: bytes_used describes digest
        # output, so it must be nowhere near the 30MB of raw transcript the
        # selection's sessions occupy on disk.
        result = mod.stratified_sample(self._single_stratum(), self._stock_config())
        raw_total = sum(r.size_bytes for r in result.selected)
        assert raw_total >= 12_000_000
        assert result.bytes_used < raw_total


class TestStratifiedSampleCostSeam:
    """``stratified_sample`` charges the byte budget through an injectable
    per-record ``cost_fn`` seam, not through ``record.size_bytes``.

    The daily budget (``budgets.max_daily_digest_bytes``) is a DIGEST-OUTPUT
    budget, but the sampler historically charged it against the RAW
    transcript size — a 20x-500x over-charge that made a single multi-MB
    session unaffordable against the whole night's cap. The seam lets a
    caller inject the real digest-byte cost while keeping
    ``stratified_sample`` itself pure (all I/O confined to the injected
    callable), which is what the PRD §8.4 boundary test depends on.

    Every record here carries a 6MB ``size_bytes`` — individually larger
    than the configured budget — so any assertion that yields a non-empty
    selection can only pass if the injected cost, not ``size_bytes``, is
    what was charged.
    """

    _COSTS = {
        'recon-a': 1_000,
        'recon-b': 2_000,
        'interactive-a': 3_000,
        'interactive-b': 4_000,
    }

    def _config(self, max_bytes=100_000):
        return config_mod.LegibilityConfig(
            project_id='dark_factory',
            project_root='/home/leo/src/dark-factory',
            escalation_port=8103,
            cwd_prefixes=['/home/leo/src/dark-factory'],
            budgets=config_mod.Budgets(max_daily_digest_bytes=max_bytes),
            sampling=config_mod.Sampling(top_fraction=0.12, per_stratum_min=2),
        )

    def _build_records(self):
        """Four candidates across two strata, each 6MB of raw transcript."""
        return [
            _scored('recon-a', 'recon', mod.SignalCounts(tool_error=9), 'recon alpha opening', 6_000_000),
            _scored('recon-b', 'recon', mod.SignalCounts(not_found=8), 'recon beta opening', 6_000_000),
            _scored('interactive-a', 'interactive', mod.SignalCounts(tool_error=5), 'human turn one', 6_000_000),
            _scored('interactive-b', 'interactive', mod.SignalCounts(self_correct=4), 'human turn two', 6_000_000),
        ]

    def _build_records_with_noise(self):
        """The four candidates plus one zero-signal record and a clone pair
        that ``dedupe_shapes`` collapses — records that must never be costed."""
        records = self._build_records()
        records.append(
            _scored('recon-zero', 'recon', mod.SignalCounts(), 'recon zero opening', 6_000_000)
        )
        records.append(
            _scored(
                'recon-clone-1', 'recon', mod.SignalCounts(tool_error=1),
                'recon clone cycle 2026-07-10', 6_000_000,
            )
        )
        records.append(
            _scored(
                'recon-clone-2', 'recon', mod.SignalCounts(tool_error=2),
                'recon clone cycle 2026-07-11', 6_000_000,
            )
        )
        return records

    def _cost(self, record):
        return self._COSTS[record.path.stem]

    def test_injected_cost_is_charged_instead_of_size_bytes(self):
        # Each record's size_bytes (6MB) alone blows the 100_000 budget, so a
        # non-empty selection is only reachable via the injected cost.
        result = mod.stratified_sample(
            self._build_records(), self._config(), cost_fn=self._cost,
        )
        assert result.selected != []
        assert {r.path.stem for r in result.selected} == set(self._COSTS)
        assert result.budget_skipped == 0

    def test_bytes_used_sums_the_injected_cost(self):
        result = mod.stratified_sample(
            self._build_records(), self._config(), cost_fn=self._cost,
        )
        assert result.bytes_used == sum(self._cost(r) for r in result.selected)
        assert result.bytes_used == sum(self._COSTS.values())

    def test_injected_cost_governs_the_budget_cut(self):
        # Cheapest-floor-first, priced by the INJECTED cost: recon's floor
        # costs 1_000+2_000 = 3_000 and interactive's 3_000+4_000 = 7_000, so
        # a 5_000 budget admits recon's floor and skips interactive's whole.
        # By raw size_bytes both floors are identical (12MB), so this
        # ordering can only come from the injected cost.
        result = mod.stratified_sample(
            self._build_records(), self._config(5_000), cost_fn=self._cost,
        )
        assert {r.path.stem for r in result.selected} == {'recon-a', 'recon-b'}
        assert result.bytes_used == 3_000
        assert result.budget_skipped == 2

    def test_cost_fn_invoked_at_most_once_per_record(self):
        # The reserve phase reads each group's total twice (sort key, then
        # charge), so an unmemoized I/O-backed cost_fn would render every
        # candidate transcript twice. Memoization is part of the contract.
        calls: dict[str, int] = {}

        def counting_cost(record):
            calls[record.path.stem] = calls.get(record.path.stem, 0) + 1
            return self._cost(record)

        mod.stratified_sample(self._build_records(), self._config(), cost_fn=counting_cost)

        assert calls, 'cost_fn was never invoked'
        assert max(calls.values()) == 1

    def test_only_candidates_are_costed(self):
        # Zero-signal drops and dedupe-collapsed clones never reach the
        # budget phase, so they must never trigger an (expensive) costing.
        costed: list[str] = []

        def recording_cost(record):
            costed.append(record.path.stem)
            return self._COSTS.get(record.path.stem, 1_000)

        mod.stratified_sample(
            self._build_records_with_noise(), self._config(), cost_fn=recording_cost,
        )

        assert 'recon-zero' not in costed
        assert 'recon-clone-1' not in costed
        assert set(costed) == set(self._COSTS)


class TestDigestByteCostFn:
    """``digest_byte_cost_fn`` is the production cost basis: what a record
    will ACTUALLY cost the daily budget is the size of the digest it renders
    to, so the factory renders it and charges the real bytes.

    The charge must be computed with the same ``max_bytes`` the nightly
    pipeline will later render with, so ``SampleResult.bytes_used``
    describes the digests that get produced rather than a parallel estimate
    that can drift from them.
    """

    def _record(self, size_bytes=6_000_000):
        return _scored(
            'cost-sess', 'orchestrated-task', mod.SignalCounts(tool_error=7),
            'orchestrated task session alpha', size_bytes,
        )

    def test_charges_utf8_byte_length_not_character_count(self):
        # 'é' is one character but two UTF-8 bytes — the budget is a BYTE
        # budget, so len(str) would silently under-charge every digest that
        # quotes a non-ASCII transcript.
        rendered = 'digest with a multibyte char: é'
        assert len(rendered.encode('utf-8')) != len(rendered)

        cost = mod.digest_byte_cost_fn(max_bytes=15360, build=lambda *a, **k: rendered)
        assert cost(self._record()) == len(rendered.encode('utf-8'))

    def test_build_is_called_with_the_records_stratum_and_the_factorys_max_bytes(self):
        calls = []

        def fake_build(path, **kwargs):
            calls.append((path, kwargs))
            return 'x' * 100

        record = self._record()
        mod.digest_byte_cost_fn(max_bytes=9_999, build=fake_build)(record)

        assert calls == [(
            record.path,
            {'agent_class_override': 'orchestrated-task', 'max_bytes': 9_999},
        )]

    def test_reuses_a_cached_render_rather_than_rendering_twice(self):
        """The closure's cache exists to avoid a second RENDER, which is the
        expensive half — ``stratified_sample`` already memoizes ``cost_fn``
        per record, so a bare byte-count memo here would be dead weight in
        the path that uses it (reviewer_comprehensive, task 3268 amendment
        pass)."""
        calls = []

        def fake_build(path, **kwargs):
            calls.append(path)
            return 'y' * 50

        cost = mod.digest_byte_cost_fn(max_bytes=15360, build=fake_build)
        record = self._record()
        assert cost(record) == 50
        assert cost(record) == 50
        assert len(calls) == 1

    def test_populates_a_callers_render_cache_with_the_digest_text(self):
        """The cache holds the rendered TEXT, not just its byte count, so a
        later pipeline stage (``nightly.build_digests``) can EMIT the digest
        instead of paying to render it again."""
        cache: dict = {}

        def fake_build(path, **kwargs):
            return 'the rendered digest'

        record = self._record()
        cost = mod.digest_byte_cost_fn(max_bytes=15360, build=fake_build, rendered=cache)
        assert cost(record) == len('the rendered digest')

        key = mod.render_cache_key(record.path, 15360, fake_build)
        assert cache == {key: 'the rendered digest'}

    def test_a_failed_render_is_never_cached(self):
        """A failure must NOT be memoized as an entry: ``build_digests``
        has to re-attempt it, raise again, and report it through
        ``extractor_failures``. A cached failure would silently swallow the
        only structured signal the operator gets."""
        cache: dict = {}

        def exploding_build(path, **kwargs):
            raise RuntimeError('transcript unreadable')

        cost = mod.digest_byte_cost_fn(
            max_bytes=15360, build=exploding_build, rendered=cache,
        )
        assert cost(self._record()) == 0
        assert cache == {}

    def test_cache_key_isolates_a_different_cap_or_builder(self):
        """A cache entry must never be served to a stage rendering with a
        different soft cap or a different builder — a mismatch is a
        structural MISS and an honest re-render, not a wrong digest."""
        record = self._record()

        def build_a(path, **kwargs):
            return 'aaa'

        def build_b(path, **kwargs):
            return 'bbbbbb'

        cache: dict = {}
        assert mod.digest_byte_cost_fn(
            max_bytes=15360, build=build_a, rendered=cache)(record) == 3
        # Same path + same builder, different cap -> miss.
        assert mod.digest_byte_cost_fn(
            max_bytes=2048, build=build_a, rendered=cache)(record) == 3
        # Same path + same cap, different builder -> miss.
        assert mod.digest_byte_cost_fn(
            max_bytes=15360, build=build_b, rendered=cache)(record) == 6
        assert len(cache) == 3

    def test_build_failure_charges_zero_and_warns(self, caplog):
        def exploding_build(path, **kwargs):
            raise RuntimeError('transcript unreadable')

        cost = mod.digest_byte_cost_fn(max_bytes=15360, build=exploding_build)
        record = self._record()

        with caplog.at_level('WARNING'):
            charged = cost(record)

        # Zero, not the flat max_bytes estimate. A render that raises
        # produces no digest bytes, so zero is the accurate charge — and it
        # is the charge that keeps the record affordable enough to reach
        # nightly.build_digests, which raises again and routes the failure
        # through the EXISTING structured extractor_failures -> escalation
        # channel (PRD decision 8). Charging the flat max_bytes made a
        # failed costing the MOST expensive candidate and therefore the one
        # most likely to be cut by the budget, which silently destroyed that
        # report (reviewer_comprehensive, task 3268 amendment pass). A bare
        # traceback out of the sampler would instead kill the whole night.
        assert charged == 0
        assert 'cost-sess' in caplog.text
        assert 'transcript unreadable' in caplog.text

    def test_build_failure_charge_does_not_starve_the_budget(self):
        """The zero charge must actually BUY the survival its docstring
        promises: a candidate whose costing render raises must still be
        selected when the budget is otherwise exhausted, because only a
        selected record reaches ``build_digests`` and gets reported.

        Under the previous flat-``max_bytes`` failure charge this exact
        arrangement dropped the failing record — the most expensive charge
        available made a failed costing the first thing the greedy fill
        halted on, so no ``extractor_failures`` entry and no escalation were
        ever produced (reviewer_comprehensive, task 3268 amendment pass).
        """
        def build(path, **kwargs):
            if path.stem == 'boom':
                raise RuntimeError('transcript unreadable')
            return 'x' * 40_000

        records = [
            # Higher score -> reserved first, and it eats the whole budget.
            _scored('ok-1', 'interactive', mod.SignalCounts(tool_error=9),
                    'healthy session alpha', 900),
            # Lower score -> reaches the budget phase in the leftover fill,
            # where the strict greedy halt is unforgiving of an over-charge.
            _scored('boom', 'interactive', mod.SignalCounts(tool_error=8),
                    'failing session beta', 900),
        ]
        cfg = config_mod.LegibilityConfig(
            project_id='dark_factory',
            project_root='/home/leo/src/dark-factory',
            escalation_port=8103,
            cwd_prefixes=['/home/leo/src/dark-factory'],
            budgets=config_mod.Budgets(max_daily_digest_bytes=40_000),
            sampling=config_mod.Sampling(top_fraction=1.0, per_stratum_min=1),
        )

        result = mod.stratified_sample(
            records, cfg, cost_fn=mod.digest_byte_cost_fn(max_bytes=15360, build=build),
        )

        # 40_000 (ok-1) + 0 (boom) == the budget exactly, so both fit.
        # Charged the old flat 15_360, boom needed 55_360 and was skipped.
        assert {r.path.stem for r in result.selected} == {'ok-1', 'boom'}
        assert result.budget_skipped == 0
        assert result.bytes_used == 40_000

    def test_default_build_renders_a_real_transcript(self, tmp_path):
        from legibility import digest as digest_mod

        path = _write_transcript(
            tmp_path / 'real-sess.jsonl',
            [
                {
                    'type': 'user', 'isSidechain': False, 'isMeta': False,
                    'cwd': MAIN_CWD, 'timestamp': '2026-07-13T09:00:00.000Z',
                    'message': {'content': 'Please help with the thing'},
                },
                _tool_error_record(),
            ],
        )
        session = inventory.SessionRecord(
            path=path, encoded_dir='-home-leo-src-dark-factory', cwd=MAIN_CWD,
            date=dt_date(2026, 7, 13), size_bytes=path.stat().st_size,
        )
        record = mod.ScoredRecord(
            session=session, stratum='interactive',
            counts=mod.SignalCounts(tool_error=1), first_turn_text='Please help with the thing',
        )

        expected = len(
            digest_mod.build_digest(
                path, agent_class_override='interactive', max_bytes=15360,
            ).encode('utf-8')
        )
        assert mod.digest_byte_cost_fn(max_bytes=15360)(record) == expected
        assert expected > 0

    def test_max_bytes_defaults_to_the_module_constant(self):
        calls = []

        def fake_build(path, **kwargs):
            calls.append(kwargs['max_bytes'])
            return 'z'

        mod.digest_byte_cost_fn(build=fake_build)(self._record())
        assert calls == [mod.DEFAULT_DIGEST_MAX_BYTES]

    def test_module_constant_matches_the_renderers_own_default(self):
        """The whole point of this change is that the COST basis and the
        RENDER basis are provably the same number. ``digest.py`` is outside
        task 3268's locked scope, so its ``max_bytes=15360`` literal stays a
        third independent definition of the §7.2 cap — this pins it equal to
        ours so a drift fails here instead of silently mis-charging the
        budget (reviewer_comprehensive, task 3268 amendment pass).
        """
        import inspect

        from legibility import digest as digest_mod

        renderer_default = (
            inspect.signature(digest_mod.build_digest).parameters['max_bytes'].default
        )
        assert mod.DEFAULT_DIGEST_MAX_BYTES == renderer_default


class TestRenderManifest:
    def test_emits_one_json_object_per_line(self):
        records = [
            _scored('sess-a', 'recon', mod.SignalCounts(tool_error=5), 'text', 1000),
            _scored('sess-b', 'watcher', mod.SignalCounts(tool_error=3), 'text', 2000),
        ]
        lines = mod.render_manifest(records).splitlines()
        assert len(lines) == 2
        parsed = [json.loads(line) for line in lines]
        assert parsed[0] == {
            'session': str(records[0].path), 'stratum': 'recon', 'score': 5, 'size': 1000,
        }
        assert parsed[1] == {
            'session': str(records[1].path), 'stratum': 'watcher', 'score': 3, 'size': 2000,
        }

    def test_empty_selection_yields_empty_string(self):
        assert mod.render_manifest([]) == ''


def _write_full_session(dir_path, session_id, cwd, timestamp, first_turn_text, include_tool_error=False):
    dir_path.mkdir(parents=True, exist_ok=True)
    lines = [
        {
            'type': 'user', 'isSidechain': False, 'isMeta': False, 'cwd': cwd, 'timestamp': timestamp,
            'message': {'content': first_turn_text},
        },
    ]
    if include_tool_error:
        lines.append({
            'type': 'user', 'cwd': cwd, 'timestamp': timestamp,
            'message': {
                'content': [
                    {'type': 'tool_result', 'tool_use_id': 't1', 'is_error': True, 'content': 'boom'},
                ]
            },
        })
    path = dir_path / f'{session_id}.jsonl'
    path.write_text('\n'.join(json.dumps(line) for line in lines) + '\n')
    return path


class TestMainCLI:
    """main(argv) wires load_config -> enumerate_sessions -> score_signals ->
    classify_agent_class -> stratified_sample -> render_manifest end-to-end
    against a tmp config + tmp projects_root (never live ~/.claude)."""

    def test_main_prints_manifest_and_summary_and_returns_zero(self, tmp_path, capsys):
        projects_root = tmp_path / 'projects'
        main_dir = projects_root / '-home-leo-src-dark-factory'
        _write_full_session(
            main_dir, 'sess-1', MAIN_CWD, '2026-07-13T09:00:00.000Z',
            'Please help with X', include_tool_error=True,
        )
        _write_full_session(
            main_dir, 'sess-2', MAIN_CWD, '2026-07-13T10:00:00.000Z',
            'Please help with Y', include_tool_error=True,
        )

        config_path = tmp_path / 'legibility.yaml'
        config_path.write_text(textwrap.dedent(f"""\
            project_id: dark_factory
            project_root: /home/leo/src/dark-factory
            escalation_port: 8103
            cwd_prefixes: [{MAIN_CWD}]
            budgets: {{max_daily_digest_bytes: 300000}}
            sampling: {{top_fraction: 0.12, per_stratum_min: 2}}
            """))

        ret = mod.main([
            '--config', str(config_path),
            '--projects-root', str(projects_root),
            '--date', '2026-07-13',
        ])

        captured = capsys.readouterr()
        assert ret == 0

        manifest_lines = [line for line in captured.out.splitlines() if line.strip()]
        parsed = [json.loads(line) for line in manifest_lines]
        assert {Path(p['session']).stem for p in parsed} == {'sess-1', 'sess-2'}

        assert 'zero-signal' in captured.err.lower()
        assert 'bytes used' in captured.err.lower()

    def test_main_reports_real_digest_bytes_not_raw_transcript_bytes(self, tmp_path, capsys):
        """The CLI acceptance surface and the nightly pipeline must never
        disagree about what the budget was spent on.

        ``main`` is the diagnostic an operator reads to decide whether the
        budget is the reason a session was skipped, so its ``bytes used``
        line has to be the same quantity ``nightly`` charges: the real
        rendered digest bytes at ``DEFAULT_DIGEST_MAX_BYTES``.
        """
        from legibility import digest as digest_mod

        projects_root = tmp_path / 'projects'
        main_dir = projects_root / '-home-leo-src-dark-factory'
        for session_id, first_turn in (
            ('sess-1', 'Please help with X'), ('sess-2', 'Please help with Y'),
        ):
            _write_full_session(
                main_dir, session_id, MAIN_CWD, '2026-07-13T09:00:00.000Z',
                first_turn, include_tool_error=True,
            )

        config_path = tmp_path / 'legibility.yaml'
        config_path.write_text(textwrap.dedent(f"""\
            project_id: dark_factory
            project_root: /home/leo/src/dark-factory
            escalation_port: 8103
            cwd_prefixes: [{MAIN_CWD}]
            budgets: {{max_daily_digest_bytes: 300000}}
            sampling: {{top_fraction: 0.12, per_stratum_min: 2}}
            """))

        ret = mod.main([
            '--config', str(config_path),
            '--projects-root', str(projects_root),
            '--date', '2026-07-13',
        ])
        assert ret == 0

        captured = capsys.readouterr()
        selected = [
            json.loads(line) for line in captured.out.splitlines() if line.strip()
        ]
        assert {Path(p['session']).stem for p in selected} == {'sess-1', 'sess-2'}

        reported = next(
            int(line.rsplit(':', 1)[1].split('/')[0].strip())
            for line in captured.err.splitlines()
            if 'bytes used' in line.lower()
        )

        expected = sum(
            len(
                digest_mod.build_digest(
                    Path(entry['session']),
                    agent_class_override=entry['stratum'],
                    max_bytes=mod.DEFAULT_DIGEST_MAX_BYTES,
                ).encode('utf-8')
            )
            for entry in selected
        )
        assert reported == expected

        # The two WRONG bases really are distinguishable on this fixture, so
        # the assertion above cannot pass by coincidence:
        #   - raw transcript size (the original defect): 768 bytes measured
        #   - the flat DEFAULT_DIGEST_MAX_BYTES estimate (the step-4
        #     fallback, and what main reported before this wiring): 30_720
        # against 744 real digest bytes. The flat-estimate margin is 41x;
        # the raw-size margin on a 2-line fixture is only ~24 bytes (the
        # digest's fixed frontmatter nearly equals the whole transcript at
        # this size), so this asserts inequality rather than a fragile
        # ordering — padding the fixture to widen that margin would instead
        # saturate the 15KB cap and collapse the flat-estimate margin.
        raw_total = sum(entry['size'] for entry in selected)
        assert reported != raw_total
        assert reported != len(selected) * mod.DEFAULT_DIGEST_MAX_BYTES

    def test_main_threads_resolved_archive_roots_into_enumerate(self, tmp_path, monkeypatch):
        # Wiring proof (no operator flip): main() resolves the config's
        # agent_transcript_roots against cfg.project_root and passes them to
        # enumerate_sessions, so the shipped archive root is actually read.
        # Patches sampling.enumerate_sessions (bare-imported into this module's
        # namespace) to capture the kwargs the call site supplies.
        config_path = tmp_path / 'legibility.yaml'
        config_path.write_text(textwrap.dedent(f"""\
            project_id: dark_factory
            project_root: /home/leo/src/dark-factory
            escalation_port: 8103
            cwd_prefixes: [{MAIN_CWD}]
            agent_transcript_roots:
              - data/orchestrator/agent-transcripts
            """))

        captured = []

        def fake_enumerate_sessions(projects_root, cwd_prefixes, target_date, **kwargs):
            captured.append(kwargs)
            return []

        monkeypatch.setattr(mod, 'enumerate_sessions', fake_enumerate_sessions)

        ret = mod.main([
            '--config', str(config_path),
            '--projects-root', str(tmp_path / 'projects'),
            '--date', '2026-07-13',
        ])

        assert ret == 0
        assert len(captured) == 1
        cfg = config_mod.load_config(config_path)
        expected = inventory.resolve_agent_transcript_roots(
            cfg.project_root, cfg.agent_transcript_roots
        )
        assert expected == [Path('/home/leo/src/dark-factory') / 'data/orchestrator/agent-transcripts']
        assert captured[0]['agent_transcript_roots'] == expected


class TestFormatSampleSummaryLine:
    """``format_sample_summary_line`` is the ONE greppable accounting line,
    shared by this module's CLI stderr block and the nightly trickle's
    journal INFO line (task 3270).

    One source for both, so the diagnostic an operator runs by hand and the
    line the systemd timer writes to the journal can never disagree about
    what the budget was spent on — the drift that let a budget-suppressed
    night look exactly like a genuine no-change night for 14 nights.
    """

    def _result(self):
        """Every field a DISTINCT non-zero value, so a field wired to the
        wrong attribute is caught (an all-zeros fixture would not be)."""
        return mod.SampleResult(
            selected=[
                _scored('sess-a', 'recon', mod.SignalCounts(tool_error=5), 'text a'),
                _scored('sess-b', 'interactive', mod.SignalCounts(not_found=3), 'text b'),
            ],
            per_stratum_counts={'recon': 1, 'interactive': 1},
            zero_signal_dropped=3,
            bytes_used=6,
            dedupe_collapsed=4,
            budget_skipped=5,
            below_sampling_cut=8,
            total_records=11,
        )

    def test_carries_every_field_in_stable_key_value_form(self):
        line = mod.format_sample_summary_line(self._result(), max_bytes=7)
        assert 'enumerated=11' in line
        assert 'zero_signal_dropped=3' in line
        assert 'dedupe_collapsed=4' in line
        assert 'below_sampling_cut=8' in line
        assert 'budget_skipped=5' in line
        assert 'selected=2' in line
        assert 'bytes_used=6/7' in line

    def test_is_a_single_line_an_operator_can_grep(self):
        # It is what `journalctl | grep` and a `grep` over the CLI's stderr
        # both have to return as ONE unit, so it must carry no newline of
        # its own (callers own their prefix and their line terminator).
        line = mod.format_sample_summary_line(self._result(), max_bytes=7)
        assert '\n' not in line
        assert line == line.strip()

    def test_main_stderr_carries_that_exact_line_verbatim(self, tmp_path, capsys, monkeypatch):
        """The CLI's summary block and the shared formatter are ONE source.

        Spies on the real sampler rather than replacing it, so the line is
        asserted against the numbers a genuine two-session run produced.
        """
        projects_root = tmp_path / 'projects'
        main_dir = projects_root / '-home-leo-src-dark-factory'
        for session_id, first_turn in (
            ('sess-1', 'Please help with X'), ('sess-2', 'Please help with Y'),
        ):
            _write_full_session(
                main_dir, session_id, MAIN_CWD, '2026-07-13T09:00:00.000Z',
                first_turn, include_tool_error=True,
            )

        config_path = tmp_path / 'legibility.yaml'
        config_path.write_text(textwrap.dedent(f"""\
            project_id: dark_factory
            project_root: /home/leo/src/dark-factory
            escalation_port: 8103
            cwd_prefixes: [{MAIN_CWD}]
            budgets: {{max_daily_digest_bytes: 300000}}
            sampling: {{top_fraction: 0.12, per_stratum_min: 2}}
            """))

        seen = []
        real_sample = mod.stratified_sample

        def spy_stratified_sample(records, config, *, cost_fn=None):
            result = real_sample(records, config, cost_fn=cost_fn)
            seen.append((result, config.budgets.max_daily_digest_bytes))
            return result

        monkeypatch.setattr(mod, 'stratified_sample', spy_stratified_sample)

        ret = mod.main([
            '--config', str(config_path),
            '--projects-root', str(projects_root),
            '--date', '2026-07-13',
        ])
        assert ret == 0
        assert len(seen) == 1

        result, max_bytes = seen[0]
        expected = mod.format_sample_summary_line(result, max_bytes=max_bytes)
        err_lines = capsys.readouterr().err.splitlines()
        assert expected in err_lines, (
            'the CLI must emit the shared summary line verbatim, on its own line'
        )
        # Real numbers, not a zeros template: both fixture sessions were
        # enumerated and both were affordable.
        assert 'enumerated=2' in expected
        assert 'selected=2' in expected
        assert 'budget_skipped=0' in expected
