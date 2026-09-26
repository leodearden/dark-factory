"""Tests for scripts/sitting/payloads.py — the apply vocabulary the sitting hands the agent (task 5376)."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
from escalation.shadow_ruling import SHADOW_RULING_MARKER
from sitting import payloads as mod
from sitting.inventory import OpenItem, escalation_key

REPO_ROOT = Path(__file__).resolve().parents[2]
ESCALATION_SERVER = REPO_ROOT / 'escalation' / 'src' / 'escalation' / 'server.py'
FUSED_MEMORY_TOOLS = REPO_ROOT / 'fused-memory' / 'src' / 'fused_memory' / 'server' / 'tools.py'
SKILL = REPO_ROOT / 'skills' / 'escalation-watcher' / 'SKILL.md'

#: tool -> (source file, the server factory the tool is nested in)
TOOL_SOURCES = {
    'resolve_issue': (ESCALATION_SERVER, 'create_server'),
    'update_task': (FUSED_MEMORY_TOOLS, 'create_mcp_server'),
    'add_dependency': (FUSED_MEMORY_TOOLS, 'create_mcp_server'),
    'submit_task': (FUSED_MEMORY_TOOLS, 'create_mcp_server'),
}

PROJECT_ROOT = '/home/leo/src/dark-factory'
RULING = mod.Ruling(esc_id='esc-4803-2', action='resume', text='Leo: option B', resolved_by='leo',
                    at='2026-09-26T09:00:00+00:00')
FINDING = mod.Finding(escalation_id='esc-6798-1', summary='rebase hazard', detail='use rerere.enabled=false')


def _tool_parameters(tool: str) -> tuple[set[str], set[str]]:
    """(every parameter, the parameters without a default) of the MCP tool's public definition."""
    source, factory = TOOL_SOURCES[tool]
    tree = ast.parse(source.read_text(encoding='utf-8'))
    (outer,) = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == factory]
    (fn,) = [n for n in ast.walk(outer) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == tool]
    positional = [*fn.args.posonlyargs, *fn.args.args]
    first_defaulted = len(positional) - len(fn.args.defaults)
    required = {a.arg for i, a in enumerate(positional) if i < first_defaulted}
    required |= {a.arg for a, d in zip(fn.args.kwonlyargs, fn.args.kw_defaults, strict=True) if d is None}
    return {a.arg for a in [*positional, *fn.args.kwonlyargs]}, required


def _server_resolve_actions() -> set[str]:
    tree = ast.parse(ESCALATION_SERVER.read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node, ast.AnnAssign):
            targets, value = [node.target], node.value
        elif isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        else:
            continue
        if value is not None and any(isinstance(t, ast.Name) and t.id == 'RESOLVE_ACTIONS' for t in targets):
            return set(ast.literal_eval(value))
    raise AssertionError('RESOLVE_ACTIONS assignment not found in escalation/server.py')


def _args(payload: mod.ApplyPayload) -> dict:
    assert isinstance(payload.args, dict), payload
    return payload.args


def _argvs(payload: mod.ApplyPayload) -> list[list[str]]:
    assert isinstance(payload.args, list), payload
    return payload.args


def _all_builder_payloads() -> list[mod.ApplyPayload]:
    return [
        mod.resolve_issue_payload('esc-4803-2', 'ruled B', 'resume', resolved_by='leo', resolution_turns=2),
        mod.update_task_payload('4803', PROJECT_ROOT, RULING, existing=None),
        mod.append_details_payload('5255', PROJECT_ROOT, FINDING),
        mod.add_dependency_payload('5255', '4803', PROJECT_ROOT),
        mod.submit_task_payload(PROJECT_ROOT, 'title', 'description', metadata={'source': 'sitting-preparer'}),
    ]


class TestSignatureContract:
    @pytest.mark.parametrize('payload', _all_builder_payloads(), ids=lambda p: p.tool)
    def test_payload_keys_fit_the_public_tool_definition(self, payload):
        accepted, required = _tool_parameters(payload.tool)

        assert set(_args(payload)) <= accepted
        assert required <= set(_args(payload))

    def test_every_mcp_tool_has_a_builder(self):
        assert {p.tool for p in _all_builder_payloads()} == set(TOOL_SOURCES)


class TestResolveIssuePayload:
    def test_actions_mirror_the_server(self):
        assert set(mod.RESOLVE_ACTIONS) == _server_resolve_actions()

    @pytest.mark.parametrize('action', sorted(_server_resolve_actions()))
    def test_every_server_action_is_accepted(self, action):
        payload = mod.resolve_issue_payload('esc-1-1', 'r', action, resolved_by='leo', resolution_turns=1)

        assert _args(payload)['action'] == action

    def test_unknown_action_is_refused(self):
        with pytest.raises(ValueError):
            mod.resolve_issue_payload('esc-1-1', 'r', 'dismiss', resolved_by='leo', resolution_turns=1)

    def test_resolution_turns_is_carried_as_an_int(self):
        payload = mod.resolve_issue_payload('esc-1-1', 'r', 'resume', resolved_by='leo', resolution_turns=3)

        assert _args(payload)['resolution_turns'] == 3
        assert type(_args(payload)['resolution_turns']) is int

    @pytest.mark.parametrize('turns', ['3', True, 0, -1, 2.0])
    def test_non_int_or_non_positive_turns_are_refused(self, turns):
        with pytest.raises(ValueError):
            mod.resolve_issue_payload('esc-1-1', 'r', 'resume', resolved_by='leo', resolution_turns=turns)


class TestUpdateTaskPayload:
    def test_shape_is_a_shallow_merge_of_one_p3_6_ruling(self):
        payload = mod.update_task_payload('4803', PROJECT_ROOT, RULING, existing=None)

        assert payload.tool == 'update_task'
        assert _args(payload) == {
            'id': '4803',
            'project_root': PROJECT_ROOT,
            'metadata': {mod.X_RULING_KEY: {
                'esc_id': 'esc-4803-2', 'action': 'resume', 'text': 'Leo: option B', 'resolved_by': 'leo',
                'at': '2026-09-26T09:00:00+00:00',
            }},
            'metadata_mode': 'merge',
        }
        assert mod.X_RULING_KEY == 'x_ruling'
        assert payload.supersedes == ''

    def test_long_ruling_text_is_capped_with_a_pointer(self):
        long = mod.Ruling(esc_id='esc-4803-2', action='resume', text='x' * (mod.X_RULING_TEXT_CAP * 2),
                          resolved_by='leo', at='2026-09-26T09:00:00+00:00')

        text = _args(mod.update_task_payload('4803', PROJECT_ROOT, long, existing=None))['metadata']['x_ruling']['text']

        assert len(text) <= mod.X_RULING_TEXT_CAP
        assert text.startswith('xxx')
        assert text.endswith('esc-4803-2]')

    @pytest.mark.parametrize(('existing', 'quoted'), [
        ('esc-5580-4 / Leo 2026-09-21: action D', 'esc-5580-4 / Leo 2026-09-21: action D'),
        ({'esc_id': 'esc-1-1', 'action': 'park'}, '{"esc_id": "esc-1-1", "action": "park"}'),
    ])
    def test_a_prior_value_is_disclosed_verbatim(self, existing, quoted):
        payload = mod.update_task_payload('5089', PROJECT_ROOT, RULING, existing=existing)

        assert payload.supersedes == quoted


class TestAddDependencyPayload:
    def test_shape(self):
        payload = mod.add_dependency_payload('5255', '4803', PROJECT_ROOT)

        assert _args(payload) == {'id': '5255', 'depends_on': '4803', 'project_root': PROJECT_ROOT}

    def test_self_dependency_is_refused(self):
        with pytest.raises(ValueError):
            mod.add_dependency_payload('5255', '5255', PROJECT_ROOT)


class TestRouteFindingToOwner:
    def test_pending_owner_gets_an_append_to_details(self):
        payload = mod.route_finding_to_owner('5255', 'pending', FINDING, PROJECT_ROOT)

        assert payload.tool == 'update_task'
        assert _args(payload)['id'] == '5255'
        assert _args(payload)['append'] is True
        assert FINDING.detail in _args(payload)['details']
        assert 'esc-6798-1' in _args(payload)['details']
        assert 'metadata' not in _args(payload)
        assert not payload.put_to_leo

    @pytest.mark.parametrize('status', ['in-progress', 'blocked', 'done', 'deferred', 'cancelled', 'review', ''])
    def test_any_other_owner_status_files_a_follow_up_put_to_leo(self, status):
        payload = mod.route_finding_to_owner('5255', status, FINDING, PROJECT_ROOT)

        assert payload.tool == 'submit_task'
        assert payload.put_to_leo
        assert _args(payload)['project_root'] == PROJECT_ROOT
        metadata = _args(payload)['metadata']
        assert metadata['spawned_from'] == '5255'
        assert metadata['escalation_id'] == 'esc-6798-1'
        assert metadata['source'] == 'sitting-preparer'


class TestCloseDecisionArgv:
    def test_close_argv_targets_the_registry_cli(self):
        payload = mod.close_decision_argv('df-esc-4803-2', 'answered', 'gate evidence', create=None)

        assert payload.tool == 'cli:session_registry'
        (argv,) = _argvs(payload)
        assert Path(argv[1]).name == 'session_registry.py'
        assert argv[2:] == ['close-decision', '--id', 'df-esc-4803-2', '--state', 'answered',
                            '--evidence', 'gate evidence']

    def test_create_prepends_a_write_decision_with_its_queue(self):
        item = OpenItem(key=escalation_key('/q', 'esc-4803-2'), escalation_id='esc-4803-2', queue_dir='/q',
                        project='dark_factory', task_id='4803', severity='blocking', text='the question')

        payload = mod.close_decision_argv('esc-4803-2', 'dropped', 'Leo: drop it', create=item)

        write, close = _argvs(payload)
        assert write[2] == 'write-decision'
        assert write[write.index('--escalations-dir') + 1] == '/q'
        assert write[write.index('--id') + 1] == 'esc-4803-2'
        assert write[write.index('--escalation-id') + 1] == 'esc-4803-2'
        assert close[2] == 'close-decision'

    @pytest.mark.parametrize('evidence', ['', '   '])
    def test_empty_evidence_is_refused(self, evidence):
        with pytest.raises(ValueError):
            mod.close_decision_argv('d1', 'answered', evidence, create=None)

    def test_open_is_not_a_closing_state(self):
        with pytest.raises(ValueError):
            mod.close_decision_argv('d1', 'open', 'evidence', create=None)


PREPARED = mod.PreparedMarker(recommendation='B', no_lean_reason='', sitting_id='nightly-2026-09-26',
                              prepared_at='2026-09-26T05:30:00+00:00')
AGREED = mod.AgreedMarker(answer='B', agreed=True, answer_rounds=1, at='2026-09-26T09:00:00+00:00')


class TestMarkers:
    def test_prepared_marker_is_one_sorted_json_line_that_round_trips(self):
        line = mod.render_prepared_marker(PREPARED)

        assert '\n' not in line
        assert line.startswith(f'{mod.PREPARED_MARKER} ')
        payload = line[len(mod.PREPARED_MARKER) + 1:]
        assert payload == json.dumps(json.loads(payload), sort_keys=True)
        assert mod.parse_prepared_marker(line) == PREPARED

    def test_agreed_marker_round_trips(self):
        line = mod.render_agreed_marker(AGREED)

        assert line.startswith(f'{mod.AGREED_MARKER} ')
        assert mod.parse_agreed_marker(line) == AGREED

    def test_agreement_is_computed_from_the_prepared_recommendation(self):
        assert mod.AgreedMarker.for_answer(PREPARED, 'B', answer_rounds=2, at='t').agreed is True
        assert mod.AgreedMarker.for_answer(PREPARED, 'C', answer_rounds=1, at='t').agreed is False
        no_lean = mod.PreparedMarker(recommendation='', no_lean_reason='both reversible', sitting_id='s',
                                     prepared_at='t')
        assert mod.AgreedMarker.for_answer(no_lean, 'C', answer_rounds=1, at='t').agreed is None

    @pytest.mark.parametrize('fields', [
        {'recommendation': '', 'no_lean_reason': ''},
        {'recommendation': 'B', 'no_lean_reason': 'also a reason'},
    ])
    def test_prepared_needs_exactly_one_of_recommendation_or_no_lean(self, fields):
        with pytest.raises(ValueError):
            mod.PreparedMarker(sitting_id='s', prepared_at='t', **fields)

    @pytest.mark.parametrize('line', [
        'x_prepared: {not json',
        'x_prepared: ["a list"]',
        'x_prepared: {"recommendation": "B"}',
        'x_prepared: {"no_lean_reason": "", "prepared_at": "t", "recommendation": "", "sitting_id": "s"}',
    ])
    def test_malformed_payload_is_a_rejected_marker_not_an_exception(self, line):
        parsed = mod.parse_prepared_marker(line)

        assert isinstance(parsed, mod.RejectedMarker)
        assert parsed.marker == mod.PREPARED_MARKER
        assert parsed.line == line

    def test_absent_marker_parses_to_none(self):
        assert mod.parse_prepared_marker('verified 2026-09-25: predicate holds') is None
        assert mod.parse_agreed_marker('') is None


class TestAppendMarkers:
    def test_existing_note_survives_verbatim_and_repeats_append(self):
        existing = 'verified 2026-09-25: task 5255 still pending (probe: get_task)'
        first = mod.append_markers(existing, mod.render_prepared_marker(PREPARED))
        later = mod.PreparedMarker(recommendation='C', no_lean_reason='', sitting_id='s2', prepared_at='t2')

        note = mod.append_markers(first, mod.render_prepared_marker(later), mod.render_agreed_marker(AGREED))

        assert note.startswith(existing + '\n')
        assert note.splitlines()[0] == existing
        assert sum(line.startswith(mod.PREPARED_MARKER) for line in note.splitlines()) == 2
        assert mod.parse_prepared_marker(note) == later
        assert mod.parse_agreed_marker(note) == AGREED

    def test_empty_note_gets_just_the_markers(self):
        line = mod.render_agreed_marker(AGREED)

        assert mod.append_markers('', line) == line

    def test_a_multi_line_marker_is_refused(self):
        with pytest.raises(ValueError):
            mod.append_markers('note', 'x_agreed: {}\nsmuggled')


class TestMarkerVocabulary:
    def test_markers_are_distinct_from_the_shadow_ruling_marker(self):
        assert len({mod.PREPARED_MARKER, mod.AGREED_MARKER, SHADOW_RULING_MARKER}) == 3

    @pytest.mark.parametrize('marker', [mod.PREPARED_MARKER, mod.AGREED_MARKER])
    def test_the_watcher_skill_names_each_marker_slug(self, marker):
        assert marker.removesuffix(':') in SKILL.read_text(encoding='utf-8')
