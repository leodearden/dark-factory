"""The committed E1 registry's canonical keys agree with the records they were keyed from.

``fixtures/memory_eval_registry_live_canonicals.json`` freezes, verbatim, the
live ``canonical: true`` record each re-keyed topic was keyed to. These tests
assert, through the probe's own matcher, that the registry finds each frozen
record by content hash, falls back to its id, and anchors its report on that
record's opening. In the other direction, every ``curator_gate`` canonical
must be keyed from committed content: its calibration row or its frozen
record.

Pure: no network, no store, no OPENAI_API_KEY.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import NamedTuple

import pytest

from fused_memory.eval import retrieval_probe
from fused_memory.eval.topic_registry import load_topic_registry

REGISTRY_PATH = Path(__file__).parent / 'fixtures' / 'memory_eval_topic_registry.json'
SNAPSHOT_PATH = Path(__file__).parent / 'fixtures' / 'memory_eval_registry_live_canonicals.json'
CALIBRATION_PATH = Path(__file__).parent / 'fixtures' / 'write_triage_calibration.jsonl'


def _snapshot_records() -> list[dict]:
    return json.loads(SNAPSHOT_PATH.read_text(encoding='utf-8'))['records']


def _calibration_canonical_content_by_cluster() -> dict[str, str]:
    rows = (
        json.loads(line)
        for line in CALIBRATION_PATH.read_text(encoding='utf-8').splitlines()
        if line.strip()
    )
    return {row['cluster_id']: row['content'] for row in rows if row['label'] == 'canonical'}


class _Hit(NamedTuple):
    id: str
    content: str


per_record = pytest.mark.parametrize(
    'record', _snapshot_records(), ids=lambda record: record['topic'],
)


@pytest.fixture(scope='module')
def registry():
    return load_topic_registry(REGISTRY_PATH)


@pytest.fixture(scope='module')
def snapshot() -> list[dict]:
    return _snapshot_records()


def test_every_record_is_what_mem0_stored(snapshot):
    for record in snapshot:
        digest = hashlib.md5(record['content'].encode('utf-8')).hexdigest()
        assert digest == record['mem0_hash'], record['topic']


def test_records_name_distinct_registry_topics_in_their_own_project(snapshot, registry):
    topics = [record['topic'] for record in snapshot]
    assert topics
    assert len(set(topics)) == len(topics)
    for record in snapshot:
        entry = registry.by_topic.get(record['topic'])
        assert entry is not None, record['topic']
        assert record['project_id'] == entry.project_id, record['topic']


@per_record
def test_the_probe_finds_the_live_canonical_by_content_hash(record, registry):
    canonical = registry.by_topic[record['topic']].canonical
    outcome = retrieval_probe.canonical_hit([_Hit(record['id'], record['content'])], canonical, 1)
    assert outcome == retrieval_probe.MatchOutcome(hit=True, rank=1, matched_by=retrieval_probe.MATCHED_BY_CONTENT_HASH)


@per_record
def test_a_reworded_canonical_falls_back_to_its_live_id(record, registry):
    canonical = registry.by_topic[record['topic']].canonical
    outcome = retrieval_probe.canonical_hit([_Hit(record['id'], 'reworded since the snapshot')], canonical, 1)
    assert outcome.matched_by == retrieval_probe.MATCHED_BY_LAST_KNOWN_ID


@per_record
def test_the_report_anchor_opens_the_live_canonical(record, registry):
    prefix = registry.by_topic[record['topic']].canonical.content_prefix
    assert prefix
    assert ' '.join(record['content'].split()).startswith(prefix)


def test_every_supersedes_pair_points_at_its_entrys_canonical(registry):
    misdirected = [
        (entry.topic, pair.superseded_hash, pair.successor_hash)
        for entry in registry.entries
        for pair in entry.supersedes_pairs
        if pair.successor_hash != entry.canonical.content_hash
    ]
    assert misdirected == []


def test_every_curator_gate_canonical_is_keyed_from_committed_content(snapshot, registry):
    content_key = retrieval_probe.content_key
    calibration = _calibration_canonical_content_by_cluster()
    frozen = {record['topic']: record['content'] for record in snapshot}
    curator_gates = [e for e in registry.entries if e.derived_from == 'curator_gate']
    unauditable = [
        entry.topic
        for entry in curator_gates
        if entry.canonical.content_hash not in {
            content_key(content)
            for content in (calibration.get(entry.provenance['cluster_id']), frozen.get(entry.topic))
            if content is not None
        }
    ]
    assert curator_gates
    assert unauditable == []
