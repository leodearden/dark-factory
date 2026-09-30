"""Validate the COMMITTED entities-omission corpus-repair artifacts (task 6060).

Task 6060 amends live Mem0 records in place, so its deliverable is evidence,
not code. This module reads only the committed files under
``docs/entities-omission-advice-repair/`` and never touches a live store:
the corpus takes concurrent writes, and a store outage must not turn into a
spurious task failure. Same stance as ``test_toolcall_xml_leak_sweep_artifacts.py``.

``census.json`` is the pre-mutation capture: every target, its verbatim
pre-image, and the searches that found it. ``phase1-apply.json`` is the
mutation record: each amendment's raw ``update_memory`` reply, its raw
``get_memory_by_id`` readback, and the census searches re-run afterwards.

The referent-attribution checks deliberately run the frozen correction text
through the PRODUCTION ``resolve_referents``, not a recorded snapshot of its
output: what matters is how today's scanner reads the corrections. Task 6059
changes that scanner, and the phase-2 follow-up (task 6067) revisits them.
"""

from __future__ import annotations

import difflib
import json
import uuid
from pathlib import Path

import pytest

from fused_memory.utils.canonical_labels import Referent
from fused_memory.utils.referent_resolution import resolve_referents

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = REPO_ROOT / 'docs' / 'entities-omission-advice-repair'

REQUIRED_PROJECTS = {
    'dark_factory',
    'reify',
    'know_live',
    'autopilot_video',
    'solar_challenge_platform',
}
TASK_NAMED_QUERIES = (
    'add_memory rejects declared entities DeclaredReferentConflictRejected; omitting entities fixes it',
    'entities parameter conflict rejected task referent declaration gotcha',
)
TASK_NAMED_TARGETS = {
    ('solar_challenge_platform', '4214c151-8776-4efa-99dd-8a12f02906e1'),
    ('dark_factory', '525ad930-15d7-415f-a701-b155cf86cfce'),
    ('dark_factory', '454d5a87-bea6-4965-97e9-cf762c9d9e21'),
    ('dark_factory', 'd3d753ef-dba4-4ac8-a712-69610ce7a65a'),
}

NOT_TARGET = 'reviewed_not_target'
RECORD_DISPOSITIONS = {'teaches_omission', 'frames_defect_as_rule', NOT_TARGET}
AUTO_MEMORY_DISPOSITIONS = {'teaches_omission', 'already_corrected'}

FIX_TASK_NUMBER = '6059'
LOCAL_PROJECT = 'dark_factory'
REPLY_FAILURE_KEYS = ('error', 'error_type', '_mcp_is_error', '_raw')
OBSERVATION_KEPT_MIN_RATIO = 0.85


def _load(name: str) -> dict:
    path = ARTIFACT_DIR / name
    if not path.is_file():
        pytest.fail(f'{path} is missing: the artifact must be committed, not skipped')
    return json.loads(path.read_text())


@pytest.fixture(scope='module')
def census() -> dict:
    return _load('census.json')


@pytest.fixture(scope='module')
def phase1() -> dict:
    return _load('phase1-apply.json')


@pytest.fixture(scope='module')
def census_by_key(census) -> dict[tuple[str, str], dict]:
    return {_record_key(r): r for r in census['records']}


def _record_key(record: dict) -> tuple[str, str]:
    return record['project_id'], record['memory_id']


def _census_targets(census) -> set[tuple[str, str]]:
    return {_record_key(r) for r in census['records'] if r['disposition'] != NOT_TARGET}


def _found_ids(hits_by_query: dict) -> set[str]:
    found: set[str] = set()
    for hits in hits_by_query.values():
        assert isinstance(hits, list), hits
        found.update(hits)
    return found


def _readback_content(readback: dict, memory_id: str) -> str:
    assert readback['found'] is True, (memory_id, readback)
    assert readback['memory_id'] == memory_id
    return readback['content']


def _observation_body(amended_content: str, marker: str) -> str:
    assert amended_content.startswith(marker)
    _correction, blank_line, body = amended_content.partition('\n\n')
    assert blank_line, 'no blank line separates the correction block from the observation'
    return body


def test_census_names_the_repair_task_and_the_fix_task(census):
    assert census['task']['project_id'] == 'dark_factory'
    assert census['task']['id'] == '6060'
    assert census['fix_task']['project_id'] == 'dark_factory'
    assert census['fix_task']['id'] == '6059'
    assert isinstance(census['fix_task']['status_at_census'], str)
    assert census['fix_task']['status_at_census']


def test_every_census_record_is_well_formed_and_unique(census):
    records = census['records']
    assert records
    for record in records:
        uuid.UUID(record['memory_id'])
        assert record['project_id'] in census['projects']
        assert record['disposition'] in RECORD_DISPOSITIONS
        assert isinstance(record['before_content'], str)
        assert record['before_content'].strip()
    keys = [_record_key(r) for r in records]
    assert len(keys) == len(set(keys))


def test_census_searched_every_project_with_every_query(census):
    assert set(census['projects']) >= REQUIRED_PROJECTS
    for query in TASK_NAMED_QUERIES:
        assert query in census['queries']
    for project in census['projects']:
        hits_by_query = census['search_hits'][project]
        for query in census['queries']:
            assert query in hits_by_query, (project, query)


def test_task_named_targets_are_census_targets(census):
    dispositions = {_record_key(r): r['disposition'] for r in census['records']}
    for key in TASK_NAMED_TARGETS:
        assert key in dispositions, key
        assert dispositions[key] != NOT_TARGET, key


def test_every_census_record_was_found_by_a_recorded_search(census):
    for record in census['records']:
        found = _found_ids(census['search_hits'][record['project_id']])
        assert record['memory_id'] in found, _record_key(record)


def test_every_auto_memory_hit_is_classified_with_its_excerpt(census):
    for entry in census['auto_memory_files']:
        assert entry['disposition'] in AUTO_MEMORY_DISPOSITIONS
        assert isinstance(entry['excerpt'], str)
        assert entry['excerpt'].strip()
        assert entry['path']
        assert isinstance(entry['line'], int)


def test_phase1_amended_exactly_the_census_targets_once_each(census, phase1):
    keys = [_record_key(a) for a in phase1['amendments']]
    assert len(keys) == len(set(keys))
    assert set(keys) == _census_targets(census)


def test_every_phase1_update_reply_is_a_success_naming_its_target(phase1):
    for amendment in phase1['amendments']:
        reply = amendment['update_reply']
        for key in REPLY_FAILURE_KEYS:
            assert key not in reply, (amendment['memory_id'], key, reply)
        assert reply['status'] == 'updated'
        assert reply['id'] == amendment['memory_id']
        assert amendment['reason'].strip()


def test_every_amendment_landed_in_place_and_changed_the_text(phase1, census_by_key):
    for amendment in phase1['amendments']:
        landed = _readback_content(amendment['readback'], amendment['memory_id'])
        assert landed == amendment['amended_content']
        before = census_by_key[_record_key(amendment)]['before_content']
        assert amendment['amended_content'] != before


def test_every_amendment_leads_with_the_task_neutral_correction_marker(phase1):
    marker = phase1['correction_marker']
    assert marker.strip()
    for project_id in {a['project_id'] for a in phase1['amendments']}:
        marker_scan = resolve_referents(
            declared=None, metadata=None, content=marker, group_id=project_id,
        )
        assert marker_scan.referents == ()
    for amendment in phase1['amendments']:
        assert amendment['amended_content'].startswith(marker)


def test_every_amendment_keeps_the_original_observation(phase1, census_by_key):
    marker = phase1['correction_marker']
    for amendment in phase1['amendments']:
        before = census_by_key[_record_key(amendment)]['before_content']
        body = _observation_body(amendment['amended_content'], marker)
        kept = difflib.SequenceMatcher(None, before, body, autojunk=False).ratio()
        assert kept >= OBSERVATION_KEPT_MIN_RATIO, (amendment['memory_id'], kept)


def test_every_correction_attributes_the_fix_task_to_dark_factory(phase1):
    for amendment in phase1['amendments']:
        project_id = amendment['project_id']
        resolution = resolve_referents(
            declared=None,
            metadata=None,
            content=amendment['amended_content'],
            group_id=project_id,
        )
        qualifier = '' if project_id == LOCAL_PROJECT else LOCAL_PROJECT
        expected = Referent(number=FIX_TASK_NUMBER, kind='task', project_id=qualifier)
        assert expected in resolution.referents, (project_id, amendment['memory_id'])
        assert all(r.number != FIX_TASK_NUMBER for r in resolution.ambiguous)
        if project_id != LOCAL_PROJECT:
            local_fix = Referent(number=FIX_TASK_NUMBER, kind='task', project_id='')
            assert local_fix not in resolution.referents, amendment['memory_id']


def test_phase1_rerun_repeated_every_census_search(census, phase1):
    rerun = phase1['search_hits_after']
    assert set(rerun) == set(census['projects'])
    for project_id in census['projects']:
        assert set(rerun[project_id]) == set(census['queries']), project_id


def test_amended_records_stay_findable_and_no_duplicate_correction_appeared(census, phase1):
    marker = phase1['correction_marker']
    rerun = phase1['search_hits_after']
    targets = _census_targets(census)
    for project_id, memory_id in targets:
        assert memory_id in _found_ids(rerun[project_id]), memory_id
    target_ids = {memory_id for _, memory_id in targets}
    for project_id in census['projects']:
        for query in census['queries']:
            before = census['search_hits'][project_id][query]
            new_ids = set(rerun[project_id][query]) - set(before) - target_ids
            for memory_id in new_ids:
                readback = phase1['new_hit_readbacks'][memory_id]
                assert marker not in _readback_content(readback, memory_id), memory_id
