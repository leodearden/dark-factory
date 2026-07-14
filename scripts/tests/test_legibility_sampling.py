"""Tests for scripts/legibility/sampling.py — zero-LLM signal scorer +
stratified budget sampler (PRD §5.2 point 2, contract §7.4, boundary test §8.4).

Self-contained: does not import task α's ``digest.py``. Imported as
``from legibility import sampling`` (PEP-420 namespace package; see
test_legibility_config.py's module docstring for the import mechanics).
"""
from __future__ import annotations

import json
from datetime import date as dt_date
from pathlib import Path

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
