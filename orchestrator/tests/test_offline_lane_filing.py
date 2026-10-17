"""The offline lane's fix-task payloads (PRD 5.5)."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from orchestrator.offline_lane_filing import (
    CountedPassEvidence,
    EvidenceNote,
    build_lane_command_task_arguments,
    build_lane_fix_task_arguments,
)
from orchestrator.offline_lane_filing import TestEvidence as _Evidence

COMMAND = 'nightly-deep'
NODE = 'tests/test_corpus_smoke.py::test_live_population_smoke'
FAILURE = "AssertionError: 3 rows carried '[unverified_claim]' ..."
FIRST_RED = 'e935a2f0c4b1d8a7961354abcdef0123456789ab'
LAST_GREEN = '0a1b2c3d4e5f60718293a4b5c6d7e8f901234567'
T0 = datetime(2026, 10, 1, 13, 21, 4, tzinfo=UTC)

WORKED_DIGEST_ENTRY = """\
tests/test_corpus_smoke.py::test_live_population_smoke   kind: failed
  pass 20261001T132104-e935a2 detect : AssertionError: 3 rows carried '[unverified_claim]' ...
  pass 20261001T132104-e935a2 confirm: AssertionError: 3 rows carried '[unverified_claim]' ...
  pass 20261001T140922-7c41d0 detect : AssertionError: 3 rows carried '[unverified_claim]' ...
  pass 20261001T140922-7c41d0 confirm: AssertionError: 3 rows carried '[unverified_claim]' ..."""


def _counted(pass_id: str, started_at: datetime | None, head: str = FIRST_RED, **overrides):
    fields = {
        'pass_id': pass_id,
        'head': head,
        'started_at': started_at,
        'mode': 'narrowed',
        'detect_first_line': FAILURE,
        'confirm_first_line': FAILURE,
        'record_dir': f'/records/{COMMAND}/{pass_id}',
    }
    fields.update(overrides)
    return CountedPassEvidence(**fields)


SECOND_START = T0 + timedelta(minutes=48)
FIRST_PASS = _counted('20261001T132104-e935a2', T0)
SECOND_PASS = _counted('20261001T140922-7c41d0', SECOND_START)
PSEUDO_NODE = f'{COMMAND}::unconfirmable'


def _test(node: str = NODE, passes=(FIRST_PASS, SECOND_PASS), **overrides) -> _Evidence:
    fields = {
        'node': node,
        'kind': 'failed',
        'passes': tuple(passes),
        'first_red_head': FIRST_RED,
        'last_green_head': LAST_GREEN,
        'notes': (),
        'prior_task_ids': (),
    }
    fields.update(overrides)
    return _Evidence(**fields)


def _build(tests=None, **overrides):
    arguments = {
        'command': COMMAND,
        'leg': 'generic',
        'priority': 'medium',
        'tests': [_test()] if tests is None else tests,
        'reproduce': f'cd /repo && pytest {NODE}',
    }
    arguments.update(overrides)
    return build_lane_fix_task_arguments(**arguments)


def _many_tests(count: int) -> list[_Evidence]:
    return [_test(node=f'tests/test_many.py::test_{i:02d}') for i in range(count)]


def test_the_digest_holds_the_prds_worked_entry_for_a_two_pass_episode():
    assert WORKED_DIGEST_ENTRY in _build()['details']


def test_the_digest_entry_leads_the_details():
    assert _build()['details'].startswith(WORKED_DIGEST_ENTRY)


def test_a_failure_line_is_cut_at_three_hundred_characters():
    long_pass = _counted('p-long', T0, detect_first_line='x' * 500, confirm_first_line=None)

    details = _build([_test(passes=(long_pass,))])['details']

    assert f'  pass p-long detect : {"x" * 300}\n' in details
    assert 'x' * 301 not in details


def test_a_phase_with_no_failure_line_renders_no_line():
    bare = _counted('p-bare', T0, confirm_first_line=None)

    details = _build([_test(passes=(bare,))])['details']

    assert 'pass p-bare detect' in details
    assert 'pass p-bare confirm' not in details


def test_a_counted_pass_with_no_failure_line_in_either_phase_still_renders():
    silent = _counted('p-silent', T0, detect_first_line=None, confirm_first_line=None)

    details = _build([_test(passes=(silent,))])['details']

    assert '  pass p-silent detect : (no failure line)' in details
    assert '  pass p-silent confirm: (no failure line)' in details


def test_the_title_counts_tests_and_distinct_passes_and_names_the_first_test():
    other = _test(node='tests/test_other.py::test_two')

    title = _build([_test(), other])['title']

    assert title == (
        f'offline-lane[{COMMAND}]: 2 test(s) red on 2 passes since {FIRST_RED[:12]} — {NODE}'
    )


def test_a_pass_shared_by_several_tests_counts_once():
    other = _test(node='tests/test_other.py::test_two', passes=(FIRST_PASS,))

    arguments = _build([_test(), other])

    assert 'red on 2 passes' in arguments['title']
    assert arguments['metadata']['x_offline_lane']['pass_ids'] == [
        FIRST_PASS.pass_id,
        SECOND_PASS.pass_id,
    ]
    assert arguments['details'].count('| narrowed') == 2


def test_the_description_names_command_leg_tests_range_and_reproduce_line():
    description = _build()['description']

    assert COMMAND in description
    assert 'generic' in description
    assert f'  - {NODE}' in description
    assert f'{LAST_GREEN}..{FIRST_RED}' in description
    assert f'Reproduce: cd /repo && pytest {NODE}' in description


def test_the_reproduce_line_is_printed_as_given_and_never_replaced_by_the_command_name():
    whole = 'cd integration && make check'

    description = _build(reproduce=whole)['description']

    assert f'Reproduce: {whole}' in description
    assert f'Reproduce: {COMMAND}' not in description


def test_a_reproduce_line_is_required():
    arguments = {
        'command': COMMAND, 'leg': 'generic', 'priority': 'medium', 'tests': [_test()],
    }

    with pytest.raises(TypeError):
        build_lane_fix_task_arguments(**arguments)


def test_a_range_with_no_green_head_says_so():
    description = _build([_test(last_green_head=None)])['description']

    assert f'unknown..{FIRST_RED}' in description


def test_ten_tests_are_shown_and_the_rest_counted():
    arguments = _build(_many_tests(13))

    assert arguments['description'].count('  - tests/test_many.py::test_') == 10
    assert '  - and 3 more' in arguments['description']
    assert 'tests/test_many.py::test_09' in arguments['details']
    assert 'tests/test_many.py::test_10' not in arguments['details']
    assert 'and 3 more' in arguments['details']
    assert len(arguments['metadata']['x_offline_lane']['nodes']) == 13


def test_exactly_ten_tests_have_no_and_more_line():
    arguments = _build(_many_tests(10))

    assert 'and 0 more' not in arguments['description']
    assert ' more' not in arguments['details']


def test_the_suspect_range_comes_from_the_earliest_first_red_episode():
    later_pass = _counted('p-late', T0 + timedelta(days=1), head='bbbb' * 10)
    later = _test(
        node='tests/test_later.py::test_x',
        passes=(later_pass,),
        first_red_head='bbbb' * 10,
        last_green_head='cccc' * 10,
    )

    arguments = _build([later, _test()])

    assert f'{LAST_GREEN}..{FIRST_RED}' in arguments['description']
    assert f'since {FIRST_RED[:12]}' in arguments['title']
    lane = arguments['metadata']['x_offline_lane']
    assert (lane['first_red_head'], lane['last_green_head']) == (FIRST_RED, LAST_GREEN)


def test_the_metadata_carries_the_lane_keys_and_none_of_the_retired_ones():
    metadata = _build()['metadata']

    assert metadata['spawn_context'] == 'offline_lane_red'
    assert metadata['merge_lane'] == 'normal'
    assert metadata['x_offline_lane'] == {
        'command': COMMAND,
        'leg': 'generic',
        'nodes': [NODE],
        'first_red_head': FIRST_RED,
        'last_green_head': LAST_GREEN,
        'pass_ids': [FIRST_PASS.pass_id, SECOND_PASS.pass_id],
        'prior_task_ids': [],
    }
    assert set(metadata) == {'spawn_context', 'merge_lane', 'x_offline_lane'}
    for retired in ('failing_tests', 'suspect_ranges', 'offline_lane_fingerprint'):
        assert retired not in metadata


def test_the_block_is_planning_mode_with_no_project_root_and_no_files():
    arguments = _build()

    assert arguments['planning_mode'] is True
    assert arguments['priority'] == 'medium'
    assert 'project_root' not in arguments
    assert 'status' not in arguments
    assert 'files' not in arguments['metadata']


def test_the_block_is_json_serialisable_and_deterministic():
    assert json.dumps(_build()) == json.dumps(_build())


def test_the_table_lists_each_counted_pass_with_head_start_and_mode():
    details = _build()['details']

    assert f'  {FIRST_PASS.pass_id} | {FIRST_RED} | {T0.isoformat()} | narrowed' in details
    assert f'| {SECOND_START.isoformat()} | narrowed' in details


def test_a_pass_whose_start_is_unknown_renders_unknown_and_sorts_last():
    unread = _counted('p-unread', None, head='unknown', mode='unknown')

    details = _build([_test(passes=(unread, FIRST_PASS))])['details']

    assert '  p-unread | unknown | unknown | unknown' in details
    assert details.index(f'  {FIRST_PASS.pass_id} | ') < details.index('  p-unread | ')


def test_hook_notes_are_rendered_with_their_source_and_node():
    named = EvidenceNote(source='verify-cache', node=NODE, text='cache hit on a stale key')
    unnamed = EvidenceNote(source='audit', node=None, text='suite-wide leak record')

    details = _build([_test(notes=(named, unnamed))])['details']

    assert f'  [verify-cache] {NODE}: cache hit on a stale key' in details
    assert '  [audit] (all reproduced tests): suite-wide leak record' in details


def test_prior_task_ids_are_rendered_and_recorded():
    arguments = _build([_test(prior_task_ids=('6001', '6002'))])

    assert f'  {NODE}: 6001, 6002' in arguments['details']
    assert arguments['metadata']['x_offline_lane']['prior_task_ids'] == ['6001', '6002']


def test_a_live_deflake_owner_is_named_in_the_details():
    details = _build(deflake_owners={NODE: '7777', 'tests/unrelated.py::t': '1'})['details']

    assert f'  {NODE}: task 7777' in details
    assert 'tests/unrelated.py::t' not in details


def test_sections_with_nothing_to_say_are_absent():
    details = _build()['details']

    for header in ('Hook notes', 'Prior lane tasks', 'Live de-flake owners'):
        assert header not in details


def test_the_record_directories_are_listed_once_each():
    details = _build()['details']

    assert details.count(f'  - {FIRST_PASS.record_dir}') == 1
    assert f'  - {SECOND_PASS.record_dir}' in details


@pytest.mark.parametrize('tests', [[], [replace(_test(), passes=())]])
def test_a_task_with_no_counted_pass_is_refused(tests):
    with pytest.raises(ValueError):
        _build(tests)


def _command_level(**overrides):
    arguments = {
        'command': COMMAND,
        'leg': 'generic',
        'priority': 'high',
        'head': FIRST_RED,
        'passes': [SECOND_PASS, FIRST_PASS, FIRST_PASS],
        'node': PSEUDO_NODE,
        'reason': 'no_detect_record',
        'reproduce': 'cd integration && make check',
        'prior_task_ids': [],
        'notes': [],
    }
    arguments.update(overrides)
    return build_lane_command_task_arguments(**arguments)


def test_the_command_level_title_names_the_head_not_a_test():
    assert _command_level()['title'] == (
        f'offline-lane[{COMMAND}]: exits non-zero with no reproducible failing test '
        f'since {FIRST_RED[:12]}'
    )


def test_the_command_level_description_names_command_leg_head_and_reason():
    description = _command_level()['description']

    assert COMMAND in description
    assert 'generic' in description
    assert FIRST_RED in description
    assert 'no_detect_record' in description
    assert 'Reproduce: cd integration && make check' in description


def test_the_command_level_details_list_each_pass_once_in_start_order():
    details = _command_level()['details']

    assert details.index(FIRST_PASS.pass_id) < details.index(SECOND_PASS.pass_id)
    assert details.count(f'pass {FIRST_PASS.pass_id} head {FIRST_RED}') == 1
    assert f'  pass {FIRST_PASS.pass_id} detect : {FAILURE}' in details
    assert f'  - {FIRST_PASS.record_dir}' in details


def test_the_command_level_task_renders_its_hook_notes_and_prior_tasks_as_a_test_task_does():
    unnamed = EvidenceNote(source='hook', node=None, text='suite-wide leak record')

    arguments = _command_level(prior_task_ids=['6001', '6002'], notes=[unnamed])

    assert 'Hook notes:\n  [hook] (all reproduced tests): suite-wide leak record' in arguments['details']
    assert f'Prior lane tasks for these tests:\n  {PSEUDO_NODE}: 6001, 6002' in arguments['details']
    assert arguments['metadata']['x_offline_lane']['prior_task_ids'] == ['6001', '6002']


def test_the_command_level_block_carries_the_lane_metadata_naming_its_pseudo_test():
    arguments = _command_level()

    assert arguments['metadata'] == {
        'spawn_context': 'offline_lane_red',
        'merge_lane': 'normal',
        'x_offline_lane': {
            'command': COMMAND,
            'leg': 'generic',
            'nodes': [PSEUDO_NODE],
            'first_red_head': FIRST_RED,
            'last_green_head': None,
            'pass_ids': [FIRST_PASS.pass_id, SECOND_PASS.pass_id],
            'prior_task_ids': [],
        },
    }
    assert arguments['planning_mode'] is True
    assert arguments['priority'] == 'high'
    assert 'project_root' not in arguments
