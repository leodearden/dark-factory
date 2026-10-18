"""Validate the COMMITTED entities-omission corpus-repair artifacts (tasks 6060, 6067).

The repair amends live Mem0 records in place, so its deliverable is evidence,
not code. This module reads only the committed files under
``docs/entities-omission-advice-repair/`` and never touches a live store:
the corpus takes concurrent writes, and a store outage must not turn into a
spurious task failure. Same stance as ``test_toolcall_xml_leak_sweep_artifacts.py``.
Its one outside read is a read-only ``git log`` of the fix commit phase 2 cites.

``census.json`` is the pre-mutation capture: every target, its verbatim
pre-image, and the searches that found it. Each ``phaseN-apply.json`` is one
mutation record: each amendment's raw ``update_memory`` reply, its raw
``get_memory_by_id`` readback, and the census searches re-run afterwards.
Phase 1 (task 6060) said the rejection was a defect awaiting task 6059;
phase 2 (task 6067) replaced that correction block to say 6059 fixed it.

The referent-attribution checks run the correction text through the
PRODUCTION ``resolve_referents``, not a recorded snapshot of its output, and
only for ``CURRENT_APPLY``: what matters is how today's scanner reads the text
that is in the store today. Superseded phase-1 text is no longer in the store.
"""

from __future__ import annotations

import difflib
import json
import re
import subprocess
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

APPLY_FILES = ('phase1-apply.json', 'phase2-apply.json')
CURRENT_APPLY = 'phase2-apply.json'

FIX_TASK_NUMBER = '6059'
FIX_MERGE_SUBJECT = 'Merge task/6059 into main'
PHASE2_TASK_ID = '6067'
LOCAL_PROJECT = 'dark_factory'
REPLY_FAILURE_KEYS = ('error', 'error_type', '_mcp_is_error', '_raw')
ZERO_UUID = '00000000-0000-0000-0000-000000000000'
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
def phase2() -> dict:
    return _load('phase2-apply.json')


@pytest.fixture(scope='module')
def current() -> dict:
    return _load(CURRENT_APPLY)


@pytest.fixture(scope='module', params=APPLY_FILES, ids=('phase1', 'phase2'))
def apply_artifact(request) -> dict:
    return _load(request.param)


@pytest.fixture(scope='module')
def correction_markers(phase1, phase2) -> set[str]:
    return {phase1['correction_marker'], phase2['correction_marker']}


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


def _correction_block(amended_content: str, marker: str) -> str:
    assert amended_content.startswith(marker)
    correction, _blank_line, _body = amended_content.partition('\n\n')
    return correction


def _scan(content: str, project_id: str):
    return resolve_referents(declared=None, metadata=None, content=content, group_id=project_id)


def _fix_task_qualifier(project_id: str) -> str:
    return '' if project_id == LOCAL_PROJECT else LOCAL_PROJECT


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


def test_apply_amended_exactly_the_census_targets_once_each(census, apply_artifact):
    keys = [_record_key(a) for a in apply_artifact['amendments']]
    assert len(keys) == len(set(keys))
    assert set(keys) == _census_targets(census)


def test_every_update_reply_is_a_success_naming_its_target(apply_artifact):
    assert apply_artifact['agent_id'].startswith('curator-')
    for amendment in apply_artifact['amendments']:
        reply = amendment['update_reply']
        for key in REPLY_FAILURE_KEYS:
            assert key not in reply, (amendment['memory_id'], key, reply)
        assert reply['status'] == 'updated'
        assert reply['id'] == amendment['memory_id']
        assert amendment['reason'].strip()


def test_every_amendment_landed_in_place(apply_artifact):
    for amendment in apply_artifact['amendments']:
        landed = _readback_content(amendment['readback'], amendment['memory_id'])
        assert landed == amendment['amended_content']


def test_every_amendment_leads_with_its_phase_correction_marker(apply_artifact):
    marker = apply_artifact['correction_marker']
    assert marker.strip()
    for amendment in apply_artifact['amendments']:
        assert amendment['amended_content'].startswith(marker), amendment['memory_id']


def test_apply_rerun_repeated_every_census_search(census, apply_artifact):
    rerun = apply_artifact['search_hits_after']
    assert set(rerun) == set(census['projects'])
    for project_id in census['projects']:
        assert set(rerun[project_id]) == set(census['queries']), project_id


def test_amended_records_stay_findable_and_no_new_hit_carries_a_correction_marker(
    census, apply_artifact, correction_markers,
):
    rerun = apply_artifact['search_hits_after']
    targets = _census_targets(census)
    for project_id, memory_id in targets:
        assert memory_id in _found_ids(rerun[project_id]), memory_id
    target_ids = {memory_id for _, memory_id in targets}
    for project_id in census['projects']:
        for query in census['queries']:
            before = census['search_hits'][project_id][query]
            new_ids = set(rerun[project_id][query]) - set(before) - target_ids
            for memory_id in new_ids:
                readback = apply_artifact['new_hit_readbacks'][memory_id]
                content = _readback_content(readback, memory_id)
                for marker in correction_markers:
                    assert marker not in content, (memory_id, marker)


def test_every_phase1_amendment_changed_the_text(phase1, census_by_key):
    for amendment in phase1['amendments']:
        before = census_by_key[_record_key(amendment)]['before_content']
        assert amendment['amended_content'] != before


def test_every_phase1_amendment_keeps_the_original_observation(phase1, census_by_key):
    marker = phase1['correction_marker']
    for amendment in phase1['amendments']:
        before = census_by_key[_record_key(amendment)]['before_content']
        body = _observation_body(amendment['amended_content'], marker)
        kept = difflib.SequenceMatcher(None, before, body, autojunk=False).ratio()
        assert kept >= OBSERVATION_KEPT_MIN_RATIO, (amendment['memory_id'], kept)


def test_current_correction_marker_derives_no_referent(current):
    marker = current['correction_marker']
    for project_id in {a['project_id'] for a in current['amendments']}:
        assert _scan(marker, project_id).referents == (), project_id


def test_every_current_correction_attributes_the_fix_task_to_dark_factory(current):
    for amendment in current['amendments']:
        project_id = amendment['project_id']
        resolution = _scan(amendment['amended_content'], project_id)
        expected = Referent(
            number=FIX_TASK_NUMBER, kind='task', project_id=_fix_task_qualifier(project_id),
        )
        assert expected in resolution.referents, (project_id, amendment['memory_id'])
        assert all(r.number != FIX_TASK_NUMBER for r in resolution.ambiguous)
        if project_id != LOCAL_PROJECT:
            local_fix = Referent(number=FIX_TASK_NUMBER, kind='task', project_id='')
            assert local_fix not in resolution.referents, amendment['memory_id']


def test_every_current_correction_block_cites_only_dark_factory_tasks(current):
    marker = current['correction_marker']
    for amendment in current['amendments']:
        project_id = amendment['project_id']
        block = _correction_block(amendment['amended_content'], marker)
        resolution = _scan(block, project_id)
        assert resolution.ambiguous == (), (amendment['memory_id'], resolution.ambiguous)
        assert resolution.referents, amendment['memory_id']
        for referent in resolution.referents:
            assert referent.kind == 'task', (amendment['memory_id'], referent)
            assert referent.project_id == _fix_task_qualifier(project_id), (
                amendment['memory_id'], referent,
            )


def test_phase2_names_its_task_and_the_measured_fix(phase2):
    assert phase2['task'] == {'id': PHASE2_TASK_ID, 'project_id': LOCAL_PROJECT}
    fix = phase2['fix_task']
    assert fix['id'] == FIX_TASK_NUMBER
    assert fix['project_id'] == LOCAL_PROJECT
    assert fix['status_at_apply'] == 'done'
    merge_commit = fix['merge_commit']
    assert re.fullmatch(r'[0-9a-f]{40}', merge_commit), merge_commit
    measured = subprocess.run(
        ['git', '-C', str(REPO_ROOT), 'log', '-1', '--format=%H%x00%cI%x00%s', merge_commit],
        capture_output=True, text=True,
    )
    if measured.returncode != 0:
        pytest.fail(
            f'fix merge commit {merge_commit} is not in the local history of {REPO_ROOT} '
            f'(shallow clone or source export without .git?): {measured.stderr.strip()}'
        )
    sha, merged_at, subject = measured.stdout.rstrip('\n').split('\x00')
    assert (sha, merged_at, subject) == (merge_commit, fix['merged_at'], FIX_MERGE_SUBJECT)


def test_every_phase2_correction_cites_the_fix_merge_commit_and_date(phase2):
    marker = phase2['correction_marker']
    merge_short = phase2['fix_task']['merge_commit'][:10]
    merged_date = phase2['fix_task']['merged_at'][:10]
    for amendment in phase2['amendments']:
        block = _correction_block(amendment['amended_content'], marker)
        assert merge_short in block, amendment['memory_id']
        assert merged_date in block, amendment['memory_id']


def test_phase2_replaced_the_phase1_correction_and_kept_the_body(phase1, phase2):
    phase1_marker = phase1['correction_marker']
    phase2_marker = phase2['correction_marker']
    assert phase2_marker != phase1_marker
    phase1_landed = {_record_key(a): a['amended_content'] for a in phase1['amendments']}
    for amendment in phase2['amendments']:
        memory_id = amendment['memory_id']
        pre_read = _readback_content(amendment['pre_read'], memory_id)
        assert pre_read == phase1_landed[_record_key(amendment)], memory_id
        amended = amendment['amended_content']
        assert phase1_marker not in amended, memory_id
        assert _observation_body(amended, phase2_marker) == _observation_body(
            pre_read, phase1_marker,
        ), memory_id


def test_phase2_authority_probe_was_a_not_found_under_the_curator_identity(phase2):
    probe = phase2['authority_probe']
    arguments = probe['arguments']
    assert arguments['memory_id'] == ZERO_UUID
    assert arguments['store'] == 'mem0'
    assert arguments['agent_id'] == phase2['agent_id']
    assert 'content' in arguments
    assert 'metadata_patch' in arguments
    assert probe['reply']['error_type'] == 'MemoryNotFound', probe['reply']
