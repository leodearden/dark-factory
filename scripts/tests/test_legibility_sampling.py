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
    """

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
        result = mod.stratified_sample(self._build_records(), self._config())
        assert not any(sid.startswith('recon-zero-') for sid in self._selected_ids(result))

    def test_zero_signal_drop_count(self):
        result = mod.stratified_sample(self._build_records(), self._config())
        assert result.zero_signal_dropped == 12

    def test_clones_deduped_to_single_highest_scorer(self):
        result = mod.stratified_sample(self._build_records(), self._config())
        watcher_selected = [r for r in result.selected if r.stratum == 'watcher']
        assert len(watcher_selected) == 1
        assert watcher_selected[0].path.stem == 'watcher-clone-3'

    def test_no_duplicate_fingerprints_in_selection(self):
        result = mod.stratified_sample(self._build_records(), self._config())
        fingerprints = [mod.shape_fingerprint(r) for r in result.selected]
        assert len(fingerprints) == len(set(fingerprints))

    def test_each_nonempty_stratum_retains_per_stratum_min_or_all_survivors(self):
        result = mod.stratified_sample(self._build_records(), self._config())
        per_stratum = {}
        for r in result.selected:
            per_stratum.setdefault(r.stratum, []).append(r)
        assert len(per_stratum['recon']) >= 2
        assert len(per_stratum['watcher']) >= 1  # only 1 survivor exists after dedup
        assert len(per_stratum['interactive']) >= 2

    def test_total_bytes_within_budget(self):
        result = mod.stratified_sample(self._build_records(), self._config())
        assert sum(r.size_bytes for r in result.selected) <= 405_000
        assert result.bytes_used == sum(r.size_bytes for r in result.selected)

    def test_tiny_stratum_survives_despite_huge_high_score_sessions_elsewhere(self):
        result = mod.stratified_sample(self._build_records(), self._config())
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
        result = mod.stratified_sample(self._build_records(), self._config())
        assert 'recon-mid-3' not in self._selected_ids(result)

    def test_selected_ordered_by_score_desc(self):
        result = mod.stratified_sample(self._build_records(), self._config())
        scores = [r.score for r in result.selected]
        assert scores == sorted(scores, reverse=True)

    def test_per_stratum_counts_accounting(self):
        result = mod.stratified_sample(self._build_records(), self._config())
        assert result.per_stratum_counts == {'recon': 2, 'watcher': 1, 'interactive': 2}


class TestStratifiedSampleReserveExceedsBudget:
    """When per-stratum reserve floors collectively exceed the byte budget
    (a real ~/.claude/projects scenario found via manual acceptance
    testing: session sizes can dwarf a conservative daily budget), the
    OVERALL cap must still hold — cheapest stratum floor first, so budget
    pressure falls on the priciest stratum rather than the cap being
    silently blown."""

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
        result = mod.stratified_sample(self._build_records(), self._config(3_000))
        assert sum(r.size_bytes for r in result.selected) <= 3_000
        assert result.bytes_used <= 3_000

    def test_cheap_stratum_preferred_when_budget_cannot_fit_both(self):
        result = mod.stratified_sample(self._build_records(), self._config(3_000))
        selected_ids = {r.path.stem for r in result.selected}
        assert {'interactive-1', 'interactive-2'} <= selected_ids
        assert 'recon-1' not in selected_ids
        assert 'recon-2' not in selected_ids


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
