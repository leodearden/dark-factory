"""Unit tests for the pre-triage module (orchestrator.agents.triage)."""

from __future__ import annotations

from unittest.mock import MagicMock

from orchestrator.agents.triage import (
    build_triage_prompt,
    format_pretriaged_detail,
    parse_triage_result,
)

# ---------------------------------------------------------------------------
# build_triage_prompt
# ---------------------------------------------------------------------------

class TestBuildTriagePrompt:
    def test_includes_all_suggestions_numbered(self):
        suggestions = [
            {'reviewer': 'test_analyst', 'location': 'a.py:1',
             'category': 'coverage', 'description': 'Missing test',
             'suggested_fix': 'Add test'},
            {'reviewer': 'style_cop', 'location': 'b.py:10',
             'category': 'style', 'description': 'Bad name',
             'suggested_fix': 'Rename'},
            {'reviewer': 'arch_auditor', 'location': 'c.py:5',
             'category': 'architecture', 'description': 'Duplication',
             'suggested_fix': 'Extract'},
        ]
        task = {'id': '42', 'title': 'Test Task', 'description': 'A test'}
        prompt = build_triage_prompt(suggestions, task)

        assert '[0]' in prompt
        assert '[1]' in prompt
        assert '[2]' in prompt
        assert 'Missing test' in prompt
        assert 'Bad name' in prompt
        assert 'Duplication' in prompt
        assert '3 items' in prompt

    def test_includes_task_context(self):
        task = {'id': '99', 'title': 'My Feature', 'description': 'Add foo'}
        prompt = build_triage_prompt([], task)
        assert 'Task 99' in prompt
        assert 'My Feature' in prompt

    def test_handles_missing_fields_gracefully(self):
        suggestions = [{'description': 'something'}]
        task = {}
        prompt = build_triage_prompt(suggestions, task)
        assert '[0]' in prompt
        assert 'something' in prompt


# ---------------------------------------------------------------------------
# parse_triage_result
# ---------------------------------------------------------------------------

class TestParseTriageResult:
    def test_valid_structured_output(self):
        result = MagicMock()
        result.structured_output = {
            'accepted': [{'index': 0, 'suggestion': 'x', 'reason': 'y',
                          'files': ['a.py'], 'proposed_task_title': 'Fix x'}],
            'skipped': [{'index': 1, 'suggestion': 'z', 'reason': 'n/a'}],
            'proposed_task_groups': [{'title': 'Fix x', 'description': 'do it',
                                      'accepted_indices': [0]}],
        }
        result.success = True
        parsed = parse_triage_result(result)
        assert parsed is not None
        assert len(parsed['accepted']) == 1
        assert len(parsed['skipped']) == 1

    def test_returns_none_on_missing_keys(self):
        result = MagicMock()
        result.structured_output = {'accepted': []}
        result.success = True
        assert parse_triage_result(result) is None

    def test_returns_none_on_no_structured_output(self):
        result = MagicMock()
        result.structured_output = None
        result.success = False
        assert parse_triage_result(result) is None

    def test_returns_none_on_non_dict(self):
        result = MagicMock()
        result.structured_output = 'not a dict'
        result.success = True
        assert parse_triage_result(result) is None


# ---------------------------------------------------------------------------
# format_pretriaged_detail
# ---------------------------------------------------------------------------

class TestFormatPretriagedDetail:
    def test_contains_header(self):
        triage_result = {
            'accepted': [{'index': 0, 'suggestion': 'x', 'reason': 'y',
                          'files': ['a.py'], 'proposed_task_title': 'Fix x'}],
            'skipped': [],
            'proposed_task_groups': [{'title': 'Fix x', 'description': 'do it',
                                      'accepted_indices': [0]}],
        }
        detail = format_pretriaged_detail(triage_result, [{'desc': 'original'}])
        assert detail.startswith('## Pre-Triaged Results')

    def test_includes_task_groups(self):
        triage_result = {
            'accepted': [
                {'index': 0, 'suggestion': 'a', 'reason': 'r',
                 'files': ['x.py'], 'proposed_task_title': 'Fix a'},
                {'index': 1, 'suggestion': 'b', 'reason': 'r',
                 'files': ['y.py'], 'proposed_task_title': 'Fix b'},
            ],
            'skipped': [],
            'proposed_task_groups': [
                {'title': 'Combined Fix', 'description': 'Fix a and b',
                 'accepted_indices': [0, 1]},
            ],
        }
        detail = format_pretriaged_detail(triage_result, [])
        assert 'Combined Fix' in detail
        assert 'x.py' in detail
        assert 'y.py' in detail

    def test_includes_skipped_items(self):
        triage_result = {
            'accepted': [],
            'skipped': [{'index': 0, 'suggestion': 'noise', 'reason': 'meritless'}],
            'proposed_task_groups': [],
        }
        detail = format_pretriaged_detail(triage_result, [])
        assert 'Skipped' in detail
        assert 'noise' in detail
        assert 'meritless' in detail

    def test_includes_original_suggestions_as_reference(self):
        originals = [{'description': 'test', 'location': 'foo.py:1'}]
        triage_result = {
            'accepted': [], 'skipped': [], 'proposed_task_groups': [],
        }
        detail = format_pretriaged_detail(triage_result, originals)
        assert 'Original Suggestions' in detail
        assert 'foo.py:1' in detail

    def test_summary_counts(self):
        triage_result = {
            'accepted': [
                {'index': i, 'suggestion': f's{i}', 'reason': 'r',
                 'files': [], 'proposed_task_title': f't{i}'}
                for i in range(3)
            ],
            'skipped': [
                {'index': i, 'suggestion': f'k{i}', 'reason': 'r'}
                for i in range(2)
            ],
            'proposed_task_groups': [
                {'title': 'g', 'description': 'd', 'accepted_indices': [0, 1, 2]},
            ],
        }
        detail = format_pretriaged_detail(triage_result, [])
        assert '3 accepted' in detail
        assert '2 skipped' in detail
        assert '1 task group' in detail
