"""Tests for PER-PROJECT write-triage calibration (task 5547).

A project other than reify is calibrated on its OWN curator-labelled fixture
and its cutoffs are written to their own config section, never pooled with
reify's. This module pins the dark_factory fixture's contract (the R1
provenance mapping and the R2 label rule), the project-config writer and
its CLI flag, and the committed artifacts that bind config <-> report <->
fixture.

The script is loaded lazily via ``_mod()``, as in
test_calibrate_write_triage.py, and the fixtures are parsed with the stdlib,
so a loader bug cannot mask a data defect. No test here needs
OPENAI_API_KEY, network or Qdrant.
"""
from __future__ import annotations

import functools
import json
import types
from collections import defaultdict
from pathlib import Path

import pytest
from _fm_helpers import load_script_module

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'calibrate_write_triage.py'
FIXTURES_DIR = Path(__file__).parent / 'fixtures'
REIFY_FIXTURE_PATH = FIXTURES_DIR / 'write_triage_calibration.jsonl'
DARK_FACTORY_FIXTURE_PATH = FIXTURES_DIR / 'write_triage_calibration_dark_factory.jsonl'

TARGET_CATEGORIES = frozenset({'preferences_and_norms', 'observations_and_summaries'})
LIVE_STORE_SOURCE = 'live_store_topic_scroll'


@functools.cache
def _mod() -> types.ModuleType:
    return load_script_module(SCRIPT_PATH, mod_name='calibrate_write_triage')


def _parse_fixture(path: Path) -> list[dict]:
    assert path.exists(), f'fixture missing: {path}'
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.fixture(scope='module')
def dark_factory_records() -> list[dict]:
    return _parse_fixture(DARK_FACTORY_FIXTURE_PATH)


def _canonicals_by_cluster(records: list[dict]) -> dict[str, dict]:
    return {r['cluster_id']: r for r in records if r['label'] == 'canonical'}


class TestDarkFactoryFixtureContract:
    """The dark_factory corpus is read from the live store's retain-and-tag topics.

    R1: there is no transcript — the store record IS the artifact — and the
    adjudicating sitting is the canonical's author. R2: a label is read off
    the sitting's own text, recorded verbatim as ``provenance.ruling``.
    """

    def test_every_record_was_read_from_the_live_store_with_no_transcript(
        self, dark_factory_records: list[dict],
    ) -> None:
        bad = {
            r['memory_id']: (r['provenance'].get('source'), r['provenance'].get('transcript_line'))
            for r in dark_factory_records
            if r['provenance'].get('source') != LIVE_STORE_SOURCE
            or r['provenance'].get('transcript_line') is not None
        }
        assert not bad, (
            f'every record must carry source={LIVE_STORE_SOURCE!r} and a null '
            f'transcript_line (a line number would be fabricated); got {bad}'
        )

    def test_every_record_names_its_topic_and_a_cluster_shares_one(
        self, dark_factory_records: list[dict],
    ) -> None:
        topics: dict[str, set] = defaultdict(set)
        for r in dark_factory_records:
            topic = r['provenance'].get('topic')
            assert isinstance(topic, str) and topic.strip(), (
                f'{r["memory_id"]}: provenance.topic must be a non-empty str'
            )
            topics[r['cluster_id']].add(topic)
        split = {cluster: sorted(t) for cluster, t in topics.items() if len(t) != 1}
        assert not split, f'a cluster is one topic; these span several: {split}'

    def test_gate_id_is_the_adjudicating_sittings_agent_id(
        self, dark_factory_records: list[dict],
    ) -> None:
        canonicals = _canonicals_by_cluster(dark_factory_records)
        bad = {
            r['memory_id']: (r['provenance'].get('gate_id'), canonicals[r['cluster_id']]['agent_id'])
            for r in dark_factory_records
            if r['provenance'].get('gate_id') != canonicals[r['cluster_id']]['agent_id']
        }
        assert not bad, f'provenance.gate_id must equal the canonical agent_id; got {bad}'

    def test_every_peer_carries_the_ruling_that_decided_its_label(
        self, dark_factory_records: list[dict],
    ) -> None:
        bad = [
            r['memory_id'] for r in dark_factory_records
            if r['label'] != 'canonical'
            and not (isinstance(r['provenance'].get('ruling'), str)
                     and r['provenance']['ruling'].strip())
        ]
        assert not bad, f'a non-canonical record must quote its ruling verbatim; got {bad}'

    def test_labels_are_only_those_the_r2_rule_can_produce(
        self, dark_factory_records: list[dict],
    ) -> None:
        labels = {r['label'] for r in dark_factory_records}
        assert labels <= {'canonical', 'duplicate', 'distinct'}, (
            'R2 rules only duplicate vs distinct; pseudo_contradiction has no rule '
            f'in this corpus. Got {sorted(labels)}'
        )

    def test_every_record_is_in_a_target_category(
        self, dark_factory_records: list[dict],
    ) -> None:
        off = {r['memory_id']: r['category'] for r in dark_factory_records
               if r['category'] not in TARGET_CATEGORIES}
        assert not off, f'off-category members are excluded and tallied; got {off}'

    def test_both_target_categories_are_present(
        self, dark_factory_records: list[dict],
    ) -> None:
        assert {r['category'] for r in dark_factory_records} == TARGET_CATEGORIES

    def test_the_corpus_shares_no_memory_with_the_reify_fixture(
        self, dark_factory_records: list[dict],
    ) -> None:
        reify_ids = {r['memory_id'] for r in _parse_fixture(REIFY_FIXTURE_PATH)}
        shared = sorted(reify_ids & {r['memory_id'] for r in dark_factory_records})
        assert not shared, f'the two corpora must be disjoint (R3); shared: {shared}'
