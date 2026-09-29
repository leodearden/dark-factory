"""Validate the COMMITTED entities-omission corpus-repair artifacts (task 6060).

Task 6060 amends live Mem0 records in place, so its deliverable is evidence,
not code. This module reads only the committed files under
``docs/entities-omission-advice-repair/`` and never touches a live store:
the corpus takes concurrent writes, and a store outage must not turn into a
spurious task failure. Same stance as ``test_toolcall_xml_leak_sweep_artifacts.py``.

``census.json`` is the pre-mutation capture: every target, its verbatim
pre-image, and the searches that found it.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

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


def _load(name: str) -> dict:
    path = ARTIFACT_DIR / name
    if not path.is_file():
        pytest.fail(f'{path} is missing: the artifact must be committed, not skipped')
    return json.loads(path.read_text())


@pytest.fixture(scope='module')
def census() -> dict:
    return _load('census.json')


def _record_key(record: dict) -> tuple[str, str]:
    return record['project_id'], record['memory_id']


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
        hits_by_query = census['search_hits'][record['project_id']]
        found = {
            memory_id
            for hits in hits_by_query.values()
            if isinstance(hits, list)
            for memory_id in hits
        }
        assert record['memory_id'] in found, _record_key(record)


def test_every_auto_memory_hit_is_classified_with_its_excerpt(census):
    for entry in census['auto_memory_files']:
        assert entry['disposition'] in AUTO_MEMORY_DISPOSITIONS
        assert isinstance(entry['excerpt'], str)
        assert entry['excerpt'].strip()
        assert entry['path']
        assert isinstance(entry['line'], int)
