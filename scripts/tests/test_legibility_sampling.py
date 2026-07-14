"""Tests for scripts/legibility/sampling.py — zero-LLM signal scorer +
stratified budget sampler (PRD §5.2 point 2, contract §7.4, boundary test §8.4).

Self-contained: does not import task α's ``digest.py``. Imported as
``from legibility import sampling`` (PEP-420 namespace package; see
test_legibility_config.py's module docstring for the import mechanics).
"""
from __future__ import annotations

import json
from pathlib import Path

from legibility import sampling as mod


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
