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
import logging
import sys
import types
from collections import defaultdict
from pathlib import Path

import pytest
import yaml
from _fm_helpers import load_script_module

from fused_memory.config.schema import WriteTriageProjectCalibration

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


# ---------------------------------------------------------------------------
# The project-config writer
# ---------------------------------------------------------------------------

SECTION = 'write_triage_project_calibrations'

# Restated, not imported: a test module never reaches into a sibling's fixtures.
# The shipped config.yaml's shape — write_triage is the LAST block, its hand-set
# knobs sit above the calibration fence, and hand-written prose trails it.
POOLED_YAML = """\
# Leading comment that must survive.
server:
  host: localhost

write_triage:
  # OPERATOR KNOBS — hand-set, NOT calibration outputs.
  enabled: false
  candidate_k: 32
  # CALIBRATION OUTPUT — do not hand-edit.
  t_high: 0.11
  t_low: 0.05
  calibration_report_path: old/report.json
  t_high_by_category:
    procedural_knowledge: 0.12
  # EVIDENCE STATUS (hand-added prose — no numbers here).
  # - procedural_knowledge: has its own cutoff above.
"""

OTHER = {'calibration_report_path': 'calibration/other.json',
         't_high_by_category': {'observations_and_summaries': 0.7}}


def _write(text: str, project: str = 'dark_factory', cutoffs: dict | None = None,
           report: str = 'calibration/df.json') -> str:
    return _mod().write_project_calibration_block(
        text, project, report,
        cutoffs if cutoffs is not None else {'preferences_and_norms': 0.9},
    )


def _section(text: str) -> dict:
    return yaml.safe_load(text)[SECTION]


def _block_lines(text: str, key: str) -> list[str]:
    """The lines of top-level *key*'s block: its key line up to the next column-0 key.

    Trailing blank lines are separators between sections, not block content.
    """
    lines = text.splitlines()
    start = lines.index(key)
    end = next(
        (i for i in range(start + 1, len(lines))
         if lines[i].strip() and not lines[i][0].isspace() and not lines[i].startswith('#')),
        len(lines),
    )
    while not lines[end - 1].strip():
        end -= 1
    return lines[start:end]


class TestDerivedCategoryCutoffs:
    def test_only_entries_with_a_derived_t_high_are_returned(self) -> None:
        report = {'per_category': {
            'procedural_knowledge': {'t_high': 0.84, 't_low': 0.57, 'reason': None},
            'observations_and_summaries': {
                't_high': None, 't_low': None, 'reason': 'insufficient_pairs: 4 duplicate',
            },
            'preferences_and_norms': {
                't_high': None, 't_low': None, 'reason': 'no_judge_band: every measured',
            },
        }}
        assert _mod().derived_category_cutoffs(report) == {'procedural_knowledge': 0.84}

    @pytest.mark.parametrize('report', [{'per_category': {}}, {}], ids=['empty', 'missing'])
    def test_no_per_category_section_derives_nothing(self, report: dict) -> None:
        assert _mod().derived_category_cutoffs(report) == {}


class TestWriteProjectCalibrationBlock:
    def test_an_absent_section_is_appended_and_every_existing_line_survives(self) -> None:
        out = _write(POOLED_YAML)
        assert out.startswith(POOLED_YAML)
        assert _section(out) == {'dark_factory': {
            'calibration_report_path': 'calibration/df.json',
            't_high_by_category': {'preferences_and_norms': 0.9},
        }}

    def test_only_the_named_project_entry_is_replaced(self) -> None:
        with_other = _write(POOLED_YAML, 'other_project', OTHER['t_high_by_category'],
                            OTHER['calibration_report_path'])
        out = _write(with_other)
        assert _section(out)['other_project'] == OTHER
        assert _section(out)['dark_factory']['t_high_by_category'] == {
            'preferences_and_norms': 0.9,
        }

    def test_rewriting_a_project_replaces_its_entry_never_merges(self) -> None:
        first = _write(POOLED_YAML, cutoffs={
            'preferences_and_norms': 0.9, 'observations_and_summaries': 0.8,
        })
        out = _write(first, cutoffs={'observations_and_summaries': 0.85})
        assert _section(out)['dark_factory']['t_high_by_category'] == {
            'observations_and_summaries': 0.85,
        }

    def test_output_is_sorted_and_idempotent(self) -> None:
        out = _write(
            _write(POOLED_YAML, 'z_project', {'b_cat': 0.6, 'a_cat': 0.5}),
            'a_project', {'d_cat': 0.7, 'c_cat': 0.8},
        )
        lines = _block_lines(out, f'{SECTION}:')
        assert lines.index('  a_project:') < lines.index('  z_project:')
        assert lines.index('      a_cat: 0.5') < lines.index('      b_cat: 0.6')
        assert lines.index('      c_cat: 0.8') < lines.index('      d_cat: 0.7')
        again = _write(out, 'a_project', {'d_cat': 0.7, 'c_cat': 0.8})
        assert again == out

    def test_an_empty_map_is_refused(self) -> None:
        with pytest.raises(ValueError, match='evidences nothing'):
            _write(POOLED_YAML, cutoffs={})

    def test_every_written_entry_validates_as_the_schema_model(self) -> None:
        out = _write(_write(POOLED_YAML, 'other_project', {'observations_and_summaries': 0.7}))
        for entry in _section(out).values():
            WriteTriageProjectCalibration.model_validate(entry)

    def test_the_section_comments_belong_to_the_section(self) -> None:
        """Indented, so the pooled writer's walk-back cannot claim them as a header."""
        lines = _block_lines(_write(POOLED_YAML), f'{SECTION}:')
        comments = [line for line in lines if line.lstrip().startswith('#')]
        assert comments, 'the section must carry its machine-owned fence'
        assert all(line.startswith('  #') for line in comments)
        assert 'machine-owned' in comments[0]


class TestTheTwoWritersAreDisjoint:
    """Neither writer can touch the other's bytes."""

    def test_the_pooled_writer_leaves_the_project_section_intact(self) -> None:
        text = _write(_write(POOLED_YAML, 'other_project', {'observations_and_summaries': 0.7}))
        out = _mod().write_triage_config_block(
            text, 0.2, 0.1, 'calibration/new.json',
            t_high_by_category={'procedural_knowledge': 0.3},
        )
        assert _block_lines(out, f'{SECTION}:') == _block_lines(text, f'{SECTION}:')
        assert yaml.safe_load(out)['write_triage']['t_high'] == 0.2

    @pytest.mark.parametrize('already_written', [False, True], ids=['append', 'replace'])
    def test_the_project_writer_leaves_the_pooled_block_intact(
        self, already_written: bool,
    ) -> None:
        text = _write(POOLED_YAML, cutoffs={'observations_and_summaries': 0.5}) \
            if already_written else POOLED_YAML
        out = _write(text)
        assert _block_lines(out, 'write_triage:') == _block_lines(POOLED_YAML, 'write_triage:')


class TestWriteProjectConfigCli:
    @pytest.mark.parametrize('other', [['--write-config'], ['--bands-from', 'base.json']],
                             ids=['write-config', 'bands-from'])
    def test_write_project_config_excludes_the_other_outcomes(
        self, other: list[str], monkeypatch, capsys,
    ) -> None:
        monkeypatch.setattr(sys, 'argv', [
            'calibrate_write_triage.py', '--write-project-config', *other,
        ])
        with pytest.raises(SystemExit) as excinfo:
            _mod().main()
        assert excinfo.value.code == 2
        assert 'not allowed with argument' in capsys.readouterr().err


def _report(per_category: dict) -> dict:
    return {'chosen_t_high': 0.99, 'chosen_t_low': 0.01, 'per_category': per_category}


class TestWriteProjectConfigHelper:
    """`_write_project_config`: the CLI's one write, and its refusal."""

    def test_no_derived_category_aborts_and_writes_nothing(self, tmp_path: Path, caplog) -> None:
        config = tmp_path / 'config.yaml'
        config.write_text(POOLED_YAML)
        report = _report({'preferences_and_norms': {
            't_high': None, 't_low': None, 'reason': 'insufficient_pairs: 10 duplicate',
        }})
        with caplog.at_level(logging.ERROR):
            rc = _mod()._write_project_config(report, config, 'dark_factory', 'r.json')
        assert rc == 1
        assert config.read_text() == POOLED_YAML
        assert 'ABORT' in caplog.text and 'insufficient_pairs' in caplog.text

    def test_a_derived_category_is_written_and_the_pooled_bands_are_not(
        self, tmp_path: Path,
    ) -> None:
        config = tmp_path / 'config.yaml'
        config.write_text(POOLED_YAML)
        report_path = SCRIPT_PATH.resolve().parents[1] / 'calibration' / 'df.json'
        report = _report({
            'observations_and_summaries': {'t_high': 0.87, 't_low': 0.6, 'reason': None},
            'preferences_and_norms': {
                't_high': None, 't_low': None, 'reason': 'insufficient_pairs: 10 duplicate',
            },
        })
        rc = _mod()._write_project_config(report, config, 'dark_factory', str(report_path))
        out = config.read_text()
        assert rc == 0
        assert _section(out) == {'dark_factory': {
            'calibration_report_path': 'calibration/df.json',
            't_high_by_category': {'observations_and_summaries': 0.87},
        }}
        assert _block_lines(out, 'write_triage:') == _block_lines(POOLED_YAML, 'write_triage:')
