"""Unit tests for the pre-triage module (orchestrator.agents.triage)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from orchestrator.agents.triage import (
    _combine_suggestion_hashes,
    build_triage_prompt,
    format_pretriaged_detail,
    parse_triage_result,
    sha256_16,
    suggestion_hash,
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

    def test_instructs_agent_to_call_submit_triage(self):
        """The prompt must name the submit_triage tool and its three params
        so the agent knows to emit its verdict via the verdict-tools MCP
        server instead of (the now-removed) --json-schema structured output.
        """
        suggestions = [
            {'reviewer': 'test_analyst', 'location': 'a.py:1',
             'category': 'coverage', 'description': 'Missing test',
             'suggested_fix': 'Add test'},
        ]
        task = {'id': '42', 'title': 'Test Task', 'description': 'A test'}
        prompt = build_triage_prompt(suggestions, task)

        assert 'submit_triage' in prompt
        assert 'accepted' in prompt
        assert 'skipped' in prompt
        assert 'proposed_task_groups' in prompt


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
# extract_triage_verdict — verdict-tools artifact envelope unwrapping
#
# Local (function-scope) imports below, matching the codebase convention
# (e.g. test_workflow_verdict_tools_injection.py) of importing not-yet-built
# symbols inside the test body rather than at module level, so a RED failure
# here does not cascade into an ImportError for every other test in this file.
# ---------------------------------------------------------------------------

class TestExtractTriageVerdict:
    def test_valid_envelope_returns_verdict_payload(self):
        from orchestrator.agents.triage import extract_triage_verdict

        envelope = {
            'role': 'triage',
            'schema_version': 1,
            'session_id': 'sess-1',
            'emitted_at': '2026-07-14T00:00:00+00:00',
            'verdict': {
                'accepted': [{'index': 0, 'suggestion': 'x', 'reason': 'y',
                              'files': ['a.py'], 'proposed_task_title': 'Fix x'}],
                'skipped': [{'index': 1, 'suggestion': 'z', 'reason': 'n/a'}],
                'proposed_task_groups': [{'title': 'Fix x', 'description': 'do it',
                                          'accepted_indices': [0]}],
            },
        }
        result = extract_triage_verdict(envelope)
        assert result == envelope['verdict']

    def test_none_envelope_returns_none(self):
        from orchestrator.agents.triage import extract_triage_verdict

        assert extract_triage_verdict(None) is None

    def test_missing_verdict_key_returns_none(self):
        from orchestrator.agents.triage import extract_triage_verdict

        envelope = {'role': 'triage', 'schema_version': 1, 'session_id': 's'}
        assert extract_triage_verdict(envelope) is None

    def test_verdict_missing_required_key_returns_none(self):
        from orchestrator.agents.triage import extract_triage_verdict

        # 'proposed_task_groups' is absent from the verdict payload.
        envelope = {'verdict': {'accepted': [], 'skipped': []}}
        assert extract_triage_verdict(envelope) is None

    def test_non_dict_envelope_returns_none(self):
        from orchestrator.agents.triage import extract_triage_verdict

        assert extract_triage_verdict('not a dict') is None

    def test_non_dict_verdict_returns_none(self):
        from orchestrator.agents.triage import extract_triage_verdict

        envelope = {'verdict': 'not a dict'}
        assert extract_triage_verdict(envelope) is None


# ---------------------------------------------------------------------------
# TRIAGE AgentRole — verdict-tools submit_triage tool grant
#
# Local (function-scope) import, matching the TestExtractTriageVerdict
# convention above: TRIAGE does not exist yet, so importing it at module
# level would turn this RED test into a collection-time ImportError for
# every other test in the file.
# ---------------------------------------------------------------------------

class TestTriageRole:
    def test_name_is_triage(self):
        from orchestrator.agents.triage import TRIAGE

        assert TRIAGE.name == 'triage'

    def test_allowed_tools_include_submit_triage_and_read_tools(self):
        from orchestrator.agents.triage import TRIAGE

        assert 'mcp__verdict-tools__submit_triage' in TRIAGE.allowed_tools
        assert 'Read' in TRIAGE.allowed_tools
        assert 'Glob' in TRIAGE.allowed_tools
        assert 'Grep' in TRIAGE.allowed_tools
        assert 'mcp__fused-memory__get_tasks' in TRIAGE.allowed_tools
        assert 'mcp__fused-memory__search' in TRIAGE.allowed_tools

    def test_mcp_families_satisfy_post_init_capability_assertion(self):
        # Importing TRIAGE constructs the module-level AgentRole immediately,
        # so AgentRole.__post_init__ already ran by the time this import
        # returns. A wrong mcp_families declaration (missing 'orchestrator'
        # for the fused-memory tools, or 'verdict_tools' for submit_triage)
        # would have raised ValueError at import time, failing this test
        # with an error rather than an assertion failure.
        from orchestrator.agents.triage import TRIAGE

        assert TRIAGE.mcp_families == frozenset({'orchestrator', 'verdict_tools'})


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

    # ── R4: idempotency stamping ─────────────────────────────────────

    def test_escalation_id_embeds_stamps_and_instructions(self):
        triage_result = {
            'accepted': [
                {'index': 0, 'suggestion': 's0', 'reason': 'r',
                 'files': ['x.py'], 'proposed_task_title': 't0'},
            ],
            'skipped': [],
            'proposed_task_groups': [
                {'title': 'Fix 0', 'description': 'd',
                 'accepted_indices': [0]},
            ],
        }
        originals = [{
            'reviewer': 'arch_auditor', 'location': 'x.py:10',
            'category': 'design', 'description': 'Fix the thing',
        }]
        detail = format_pretriaged_detail(
            triage_result, originals, escalation_id='esc-1912-179',
        )
        assert 'Task Idempotency Stamps' in detail
        assert 'esc-1912-179' in detail
        # Per-group suggestion_hash rendered deterministically
        expected_hash = suggestion_hash(originals[0])
        assert expected_hash in detail
        # Steward-facing instruction present
        assert 'escalation_id' in detail
        assert 'suggestion_hash' in detail
        assert 'interceptor will' in detail
        # submit_task call must show the metadata= kwarg form
        assert 'submit_task' in detail
        assert 'metadata=' in detail
        # One-step submit: emitted detail must NOT name resolve_ticket — the
        # steward fires-and-forgets submit_task; the curator's combine
        # decision lands in tasks.json asynchronously.
        assert 'resolve_ticket' not in detail, (
            'Pre-triaged block must not direct the steward to call '
            'resolve_ticket — the janitor surfaces failures asynchronously'
        )
        assert 'combined' in detail, (
            "Pre-triaged block must still describe the 'combined' outcome "
            'so the steward understands R4 idempotency-hit semantics'
        )

    def test_escalation_id_absent_keeps_legacy_format(self):
        triage_result = {
            'accepted': [], 'skipped': [], 'proposed_task_groups': [],
        }
        detail = format_pretriaged_detail(triage_result, [])
        assert 'Task Idempotency Stamps' not in detail
        assert 'suggestion_hash' not in detail

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


# ---------------------------------------------------------------------------
# R4: suggestion_hash determinism
# ---------------------------------------------------------------------------

class TestSuggestionHash:
    """R4 requires deterministic hashes so steward re-queues produce the
    same ``(escalation_id, suggestion_hash)`` tuple across retries.
    """

    def test_same_suggestion_same_hash(self):
        s = {
            'reviewer': 'arch_auditor', 'location': 'x.py:10',
            'category': 'design', 'description': 'Fix it',
        }
        assert suggestion_hash(s) == suggestion_hash(s)

    def test_differs_on_description_change(self):
        base = {
            'reviewer': 'arch_auditor', 'location': 'x.py:10',
            'category': 'design', 'description': 'Fix it',
        }
        variant = {**base, 'description': 'Something else'}
        assert suggestion_hash(base) != suggestion_hash(variant)

    def test_ignores_unrelated_fields(self):
        base = {
            'reviewer': 'arch_auditor', 'location': 'x.py:10',
            'category': 'design', 'description': 'Fix it',
            'suggested_fix': 'v1',
        }
        variant = {**base, 'suggested_fix': 'rephrased v2'}
        # suggested_fix is not part of the identity tuple.
        assert suggestion_hash(base) == suggestion_hash(variant)

    def test_hash_length_is_16(self):
        s = {'reviewer': 'r', 'location': 'l', 'category': 'c', 'description': 'd'}
        assert len(suggestion_hash(s)) == 16

    def test_combine_sorted_deterministically(self):
        a = _combine_suggestion_hashes(['bbb', 'aaa', 'ccc'])
        b = _combine_suggestion_hashes(['ccc', 'aaa', 'bbb'])
        assert a == b

    def test_combine_single_returns_self(self):
        assert _combine_suggestion_hashes(['abcd1234abcd1234']) == 'abcd1234abcd1234'


# ---------------------------------------------------------------------------
# sha256_16 — canonical 16-char sha256-hex helper
# ---------------------------------------------------------------------------

class TestSha256_16:
    """sha256_16 is the shared 16-char sha256 helper that both suggestion_hash
    and the escalation-watcher skill's cleanup_needed snippet depend on.
    """

    def test_length_is_16(self):
        assert len(sha256_16('anything')) == 16

    def test_deterministic(self):
        assert sha256_16('hello') == sha256_16('hello')

    def test_differs_across_inputs(self):
        assert sha256_16('hello') != sha256_16('world')

    def test_empty_string_raises(self):
        """Empty input raises ValueError so blank-detail callers fail loudly at call time."""
        with pytest.raises(ValueError, match="non-empty"):
            sha256_16('')
