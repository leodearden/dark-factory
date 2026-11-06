"""Tests for PER-PROJECT write-triage calibration (task 5547).

A project other than reify is calibrated on its OWN curator-labelled fixture
and its cutoffs are written to their own config section, never pooled with
reify's. This module pins every live-store fixture's contract (the R1
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
from _jsonl_fixture import parse_jsonl_fixture
from _write_triage_calibration_fixtures import (
    CALIBRATION_FIXTURES,
    DARK_FACTORY_ADMITTED_CATEGORIES,
    DARK_FACTORY_FIXTURE_PATH,
    LIVE_STORE_FIXTURES,
    PACKAGE_ROOT,
    LiveStoreFixture,
)

from fused_memory.config.schema import FusedMemoryConfig, WriteTriageProjectCalibration

SCRIPT_PATH = PACKAGE_ROOT / 'scripts' / 'calibrate_write_triage.py'
LIVE_STORE_SOURCE = 'live_store_topic_scroll'


@functools.cache
def _mod() -> types.ModuleType:
    return load_script_module(SCRIPT_PATH, mod_name='calibrate_write_triage')


@pytest.fixture(scope='module')
def dark_factory_records() -> list[dict]:
    return parse_jsonl_fixture(DARK_FACTORY_FIXTURE_PATH)


@pytest.fixture(scope='module', params=LIVE_STORE_FIXTURES, ids=lambda f: f.project_id)
def live_store(request: pytest.FixtureRequest) -> LiveStoreFixture:
    return request.param


@pytest.fixture(scope='module')
def live_records(live_store: LiveStoreFixture) -> list[dict]:
    return parse_jsonl_fixture(live_store.path)


def _canonicals_by_cluster(records: list[dict]) -> dict[str, dict]:
    return {r['cluster_id']: r for r in records if r['label'] == 'canonical'}


class TestLiveStoreFixtureContract:
    """Each per-project corpus is read from its live store's retain-and-tag topics.

    R1: there is no transcript — the store record IS the artifact — and the
    adjudicating sitting is the canonical's author. R2: a label is read off
    the sitting's own text, recorded verbatim as ``provenance.ruling``.
    """

    def test_every_record_was_read_from_the_live_store_with_no_transcript(
        self, live_records: list[dict],
    ) -> None:
        bad = {
            r['memory_id']: (r['provenance'].get('source'), r['provenance'].get('transcript_line'))
            for r in live_records
            if r['provenance'].get('source') != LIVE_STORE_SOURCE
            or r['provenance'].get('transcript_line') is not None
        }
        assert not bad, (
            f'every record must carry source={LIVE_STORE_SOURCE!r} and a null '
            f'transcript_line (a line number would be fabricated); got {bad}'
        )

    def test_every_record_names_its_topic_and_a_cluster_shares_one(
        self, live_records: list[dict],
    ) -> None:
        topics: dict[str, set] = defaultdict(set)
        for r in live_records:
            topic = r['provenance'].get('topic')
            assert isinstance(topic, str) and topic.strip(), (
                f'{r["memory_id"]}: provenance.topic must be a non-empty str'
            )
            topics[r['cluster_id']].add(topic)
        split = {cluster: sorted(t) for cluster, t in topics.items() if len(t) != 1}
        assert not split, f'a cluster is one topic; these span several: {split}'

    def test_gate_id_is_the_adjudicating_sittings_agent_id(
        self, live_records: list[dict],
    ) -> None:
        canonicals = _canonicals_by_cluster(live_records)
        bad = {
            r['memory_id']: (r['provenance'].get('gate_id'), canonicals[r['cluster_id']]['agent_id'])
            for r in live_records
            if r['provenance'].get('gate_id') != canonicals[r['cluster_id']]['agent_id']
        }
        assert not bad, f'provenance.gate_id must equal the canonical agent_id; got {bad}'

    def test_every_peer_carries_the_ruling_that_decided_its_label(
        self, live_records: list[dict],
    ) -> None:
        bad = [
            r['memory_id'] for r in live_records
            if r['label'] != 'canonical'
            and not (isinstance(r['provenance'].get('ruling'), str)
                     and r['provenance']['ruling'].strip())
        ]
        assert not bad, f'a non-canonical record must quote its ruling verbatim; got {bad}'

    def test_labels_are_only_those_the_r2_rule_can_produce(
        self, live_records: list[dict],
    ) -> None:
        labels = {r['label'] for r in live_records}
        assert labels <= {'canonical', 'duplicate', 'distinct'}, (
            'R2 rules only duplicate vs distinct; pseudo_contradiction has no rule '
            f'in this corpus. Got {sorted(labels)}'
        )

    def test_every_record_is_in_an_admitted_category(
        self, live_store: LiveStoreFixture, live_records: list[dict],
    ) -> None:
        off = {r['memory_id']: r['category'] for r in live_records
               if r['category'] not in live_store.admitted_categories}
        assert not off, f'off-category members are excluded and tallied; got {off}'

    def test_the_corpus_shares_no_memory_with_any_other_calibration_fixture(
        self, live_store: LiveStoreFixture, live_records: list[dict],
    ) -> None:
        own_ids = {r['memory_id'] for r in live_records}
        shared = {
            project: sorted(own_ids & {r['memory_id'] for r in parse_jsonl_fixture(other)})
            for project, other in CALIBRATION_FIXTURES.items() if other != live_store.path
        }
        assert not any(shared.values()), f'calibration corpora must be disjoint (R3); shared: {shared}'


class TestDarkFactoryFixtureScope:
    def test_both_target_categories_are_present(
        self, dark_factory_records: list[dict],
    ) -> None:
        assert {r['category'] for r in dark_factory_records} == DARK_FACTORY_ADMITTED_CATEGORIES


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


class TestProjectSectionScalarsRoundTrip:
    """What the writer renders, yaml.safe_load reads back as the same str/float values."""

    @pytest.mark.parametrize('project', ['123', 'yes', 'on', 'null', '0.5'])
    def test_a_yaml_ambiguous_project_id_stays_a_string_key(self, project: str) -> None:
        section = _section(_write(POOLED_YAML, project))
        assert list(section) == [project]
        WriteTriageProjectCalibration.model_validate(section[project])

    @pytest.mark.parametrize('report', [
        'calibration/a: b.json', 'calibration/a #b.json', '123', 'yes',
    ])
    def test_a_report_path_carrying_yaml_syntax_is_read_back_verbatim(self, report: str) -> None:
        entry = _section(_write(POOLED_YAML, report=report))['dark_factory']
        assert entry['calibration_report_path'] == report

    def test_a_rewritten_shipped_config_still_loads(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        text = CONFIG_PATH.read_text()
        for project in ('123', 'yes'):
            text = _write(text, project)
        rewritten = tmp_path / 'config.yaml'
        rewritten.write_text(text)
        monkeypatch.setenv('CONFIG_PATH', str(rewritten))
        loaded = FusedMemoryConfig().write_triage_project_calibrations
        assert {'123', 'yes'} <= set(loaded)

    def test_an_existing_non_string_project_key_is_refused_by_name(self) -> None:
        broken = POOLED_YAML + (
            f'\n{SECTION}:\n  123:\n    calibration_report_path: calibration/x.json\n'
            '    t_high_by_category:\n      preferences_and_norms: 0.5\n'
        )
        with pytest.raises(ValueError, match='123'):
            _write(broken)


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
    @pytest.fixture(autouse=True)
    def _no_live_store(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A refusal that regresses must fail here, never fall through to a live run.

        Unguarded, a live run would overwrite the committed reports and config.yaml.
        """
        def no_store(*_args, **_kwargs):
            raise AssertionError('the refusal must come before the store is touched')

        monkeypatch.setattr('fused_memory.services.memory_service.MemoryService', no_store)

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

    @pytest.mark.parametrize('given', [
        ['--fixture', 'f.jsonl'], ['--report-path', 'r.json'], [],
    ], ids=['no-report-path', 'no-fixture', 'neither'])
    def test_write_project_config_requires_an_explicit_fixture_and_report_path(
        self, given: list[str], monkeypatch, capsys,
    ) -> None:
        """The defaults are the pooled reify corpus and its report of record."""
        monkeypatch.setattr(sys, 'argv', [
            'calibrate_write_triage.py', '--project-id', 'dark_factory',
            '--write-project-config', *given,
        ])
        with pytest.raises(SystemExit) as excinfo:
            _mod().main()
        assert excinfo.value.code == 2
        err = capsys.readouterr().err
        assert '--fixture' in err and '--report-path' in err

    @pytest.mark.parametrize('project, report', [
        ('reify', 'write_triage_calibration_report_other.json'),
        ('dark_factory', 'write_triage_calibration_report.json'),
    ], ids=['pooled-project', 'pooled-report'])
    def test_a_run_that_would_clobber_or_shadow_the_pooled_corpus_aborts_before_any_io(
        self, project: str, report: str, monkeypatch, caplog,
    ) -> None:
        monkeypatch.setattr(sys, 'argv', [
            'calibrate_write_triage.py', '--project-id', project, '--write-project-config',
            '--fixture', str(DARK_FACTORY_FIXTURE_PATH),
            '--report-path', str(PACKAGE_ROOT / 'calibration' / report),
        ])
        with caplog.at_level(logging.ERROR):
            assert _mod().main() == 1
        assert 'ABORT' in caplog.text


class TestPooledCorpusConflict:
    """`pooled_corpus_conflict`: a project entry may neither overwrite nor shadow reify's."""

    POOLED_REPORT = Path('/pkg/calibration/pooled.json')

    def _conflict(self, project: str, report: str) -> str | None:
        return _mod().pooled_corpus_conflict(
            project, report, pooled_report=self.POOLED_REPORT, pooled_project_id='reify',
        )

    def test_another_project_with_its_own_report_has_no_conflict(self) -> None:
        assert self._conflict('dark_factory', '/pkg/calibration/df.json') is None

    def test_the_pooled_project_is_refused(self) -> None:
        assert 'reify' in str(self._conflict('reify', '/pkg/calibration/df.json'))

    def test_the_pooled_report_path_is_refused(self) -> None:
        assert 'pooled.json' in str(self._conflict('dark_factory', str(self.POOLED_REPORT)))

    def test_an_uncalibrated_pool_conflicts_with_nothing(self) -> None:
        assert _mod().pooled_corpus_conflict(
            'reify', '/pkg/calibration/df.json', pooled_report=None, pooled_project_id=None,
        ) is None


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


# ---------------------------------------------------------------------------
# The committed artifacts: config <-> report <-> fixture, no network
# ---------------------------------------------------------------------------

CONFIG_PATH = PACKAGE_ROOT / 'config' / 'config.yaml'


@functools.cache
def _committed_report(project_id: str) -> dict:
    path = next(f.report_path for f in LIVE_STORE_FIXTURES if f.project_id == project_id)
    assert path.exists(), f'report missing: {path}'
    return json.loads(path.read_text())


@functools.cache
def _shipped_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def _pooled_report() -> dict:
    named = _shipped_config()['write_triage']['calibration_report_path']
    return json.loads((PACKAGE_ROOT / named).read_text())


def _project_entries() -> dict:
    return _shipped_config().get(SECTION) or {}


def _max_negative(entry: dict) -> float:
    maxima = [entry['distributions'][cls]['max'] for cls in ('unrelated', 'hard_negative')]
    measured = [m for m in maxima if m is not None]
    assert measured, 'a derived cutoff needs at least one measured negative'
    return max(measured)


def _category_outcome_defect(entry: dict) -> str | None:
    """Why *entry* is neither a separating cutoff nor a measured refusal, else None."""
    if entry['t_high'] is not None:
        if not 0.0 <= entry['t_high'] <= 1.0:
            return f't_high {entry["t_high"]} is not a cosine'
        if entry['t_high'] <= _max_negative(entry):
            return 'a derived cutoff must admit no in-category negative'
        return None
    if entry['t_low'] is not None:
        return 'a refused category must leave t_low None too'
    code, _, detail = str(entry['reason']).partition(':')
    refusals = {
        _mod().REASON_EMPTY_CLASS, _mod().REASON_NOT_SEPARABLE, _mod().REASON_INSUFFICIENT_PAIRS,
    }
    if code not in refusals:
        return f'unknown refusal {entry["reason"]!r}'
    if not detail.strip():
        return 'a refusal must state what was measured'
    return None


def _pair_counts_by_category(records: list[dict]) -> dict[str, dict[str, int]]:
    buckets = _mod().partition_pairs_by_category(records)['by_category']
    return {
        category: {cls.removesuffix('_pairs'): len(pairs) for cls, pairs in bucket.items()}
        for category, bucket in buckets.items()
    }


class TestCommittedLiveStoreCalibration:
    def test_the_report_pair_is_committed(self, live_store: LiveStoreFixture) -> None:
        assert live_store.report_path.exists()
        assert live_store.report_path.with_suffix('.md').exists()

    def test_provenance_names_the_projects_own_corpus(self, live_store: LiveStoreFixture) -> None:
        provenance = _committed_report(live_store.project_id)['provenance']
        assert provenance['project_id'] == live_store.project_id
        assert provenance['fixture_path'] == live_store.package_relative_path
        assert provenance['retrieval_mode'] == 'production'

    def test_the_record_count_is_the_committed_fixtures(
        self, live_store: LiveStoreFixture, live_records: list[dict],
    ) -> None:
        report = _committed_report(live_store.project_id)
        assert report['provenance']['record_count'] == len(live_records)

    def test_the_pair_counts_are_recomputed_from_the_committed_fixture(
        self, live_store: LiveStoreFixture, live_records: list[dict],
    ) -> None:
        """A fixture edit without a re-run fails here."""
        report = _committed_report(live_store.project_id)
        assert report['provenance']['per_category_pair_counts'] == _pair_counts_by_category(
            live_records,
        )

    def test_every_admitted_category_that_forms_pairs_has_a_measured_outcome(
        self, live_store: LiveStoreFixture, live_records: list[dict],
    ) -> None:
        per_category = _committed_report(live_store.project_id)['per_category']
        forming = {
            category for category, counts in _pair_counts_by_category(live_records).items()
            if category in live_store.admitted_categories and any(counts.values())
        }
        assert forming, 'a fixture that forms no pair calibrates nothing'
        missing = sorted(forming - set(per_category))
        assert not missing, f'categories with pairs but no outcome: {missing}'
        defects = {c: _category_outcome_defect(per_category[c]) for c in sorted(forming)}
        assert not any(defects.values()), defects

    def test_the_corpus_is_isolated_from_the_pooled_report_of_record(
        self, live_store: LiveStoreFixture,
    ) -> None:
        pooled_named = _shipped_config()['write_triage']['calibration_report_path']
        assert (PACKAGE_ROOT / pooled_named).resolve() != live_store.report_path.resolve()
        pooled = _pooled_report()['provenance']
        assert pooled['project_id'] == 'reify'
        assert pooled['fixture_path'] != live_store.package_relative_path


class TestCommittedProjectCalibrationsAreTraceable:
    def test_every_entry_validates_as_the_schema_model(self) -> None:
        for entry in _project_entries().values():
            WriteTriageProjectCalibration.model_validate(entry)

    def test_every_entry_names_a_committed_report_for_its_own_project(self) -> None:
        for project, entry in _project_entries().items():
            named = Path(entry['calibration_report_path'])
            assert not named.is_absolute(), 'a worktree path dangles once merged'
            report = json.loads((PACKAGE_ROOT / named).read_text())
            assert report['provenance']['project_id'] == project

    def test_every_cutoff_is_its_reports_derived_value_and_nothing_else(self) -> None:
        for entry in _project_entries().values():
            report = json.loads((PACKAGE_ROOT / entry['calibration_report_path']).read_text())
            derived = _mod().derived_category_cutoffs(report)
            assert entry['t_high_by_category'] == derived, (
                'a derived cutoff is wired and a refused category is absent, '
                'never hand-edited'
            )

    def test_no_entry_shadows_the_pooled_corpus_of_record(self) -> None:
        assert _pooled_report()['provenance']['project_id'] not in _project_entries()

    def test_a_project_entry_exists_exactly_when_a_cutoff_was_derived(
        self, live_store: LiveStoreFixture,
    ) -> None:
        derived = _mod().derived_category_cutoffs(_committed_report(live_store.project_id))
        assert (live_store.project_id in _project_entries()) == bool(derived)

    def test_rewriting_every_shipped_entry_is_a_byte_for_byte_no_op(self) -> None:
        shipped = CONFIG_PATH.read_text()
        changed = [
            project for project, entry in _project_entries().items()
            if _mod().write_project_calibration_block(
                shipped, project, entry['calibration_report_path'], entry['t_high_by_category'],
            ) != shipped
        ]
        assert not changed, f'a rewrite of these shipped entries changes config.yaml: {changed}'
