"""The ONE place the suite reads the committed plans/memory-metadata-* census artifacts.

Every other test feeds its reader a fixture, so damage on main fails here by
name rather than scattering red across unrelated branches (task 5231).
"""
from __future__ import annotations

import functools
import json
import types
from pathlib import Path

import pytest
from _fm_helpers import load_script_module

_SCRIPTS_DIR = Path(__file__).parent.parent / 'scripts'


@functools.cache
def _probe() -> types.ModuleType:
    return load_script_module(
        _SCRIPTS_DIR / 'memory_eval_retrieval_probe.py', mod_name='memory_eval_retrieval_probe',
    )


@functools.cache
def _census() -> types.ModuleType:
    return load_script_module(
        _SCRIPTS_DIR / 'census_memory_metadata.py', mod_name='census_memory_metadata',
    )


# ---------------------------------------------------------------------------
# Falsifiability: each detector catches the damage it exists for
# ---------------------------------------------------------------------------

_STUB_COMMITTED_ON_2026_08_31 = '{"baseline": true}\n'
_HEALTHY_ROW = {'value': 'a-topic', 'count': 3}


def _census_text(*rows: object) -> str:
    return json.dumps({'grand_total': {'topic': {'entries': list(rows)}}})


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding='utf-8')
    return path


class TestCensusReportDamageIsDetected:
    @pytest.mark.parametrize('text', [
        pytest.param(_STUB_COMMITTED_ON_2026_08_31, id='the-2026-08-31-stub'),
        pytest.param('', id='empty-file'),
        pytest.param('not json', id='not-json'),
        pytest.param('[]', id='top-level-list'),
        pytest.param('{"grand_total": "nope"}', id='grand-total-not-an-object'),
        pytest.param('{"grand_total": {"topic": "nope"}}', id='topic-not-an-object'),
        pytest.param('{"grand_total": {"topic": {"entries": {}}}}', id='entries-not-a-list'),
        pytest.param(_census_text(), id='empty-topic-table'),
        pytest.param(
            _census_text(_HEALTHY_ROW, {'value': '', 'count': 2}), id='row-with-empty-value',
        ),
        pytest.param(
            _census_text(_HEALTHY_ROW, {'value': 'b-topic', 'count': '2'}),
            id='row-with-string-count',
        ),
    ])
    def test_a_damaged_report_is_reported(self, tmp_path, text):
        assert _census_report_damage(_write(tmp_path / 'census.json', text)) is not None

    def test_an_absent_report_is_reported(self, tmp_path):
        assert _census_report_damage(tmp_path / 'absent.json') is not None

    def test_a_well_formed_report_is_not(self, tmp_path):
        text = _census_text(_HEALTHY_ROW, {'value': 'b-topic', 'count': 1})
        assert _census_report_damage(_write(tmp_path / 'census.json', text)) is None


class TestCoverageHistoryDamageIsDetected:
    @pytest.mark.parametrize('text', [
        pytest.param(_STUB_COMMITTED_ON_2026_08_31, id='the-2026-08-31-stub'),
        pytest.param('{"schema_version": 2, "runs": []}', id='unknown-schema-version'),
        pytest.param('{"schema_version": 1}', id='no-runs-list'),
        pytest.param('not json', id='not-json'),
    ])
    def test_a_damaged_history_is_reported(self, tmp_path, text):
        assert _coverage_history_damage(_write(tmp_path / 'history.json', text)) is not None

    def test_an_absent_history_is_reported_though_the_census_would_start_afresh(self, tmp_path):
        assert _coverage_history_damage(tmp_path / 'absent.json') is not None

    def test_a_history_the_census_itself_saved_is_not(self, tmp_path):
        path = tmp_path / 'history.json'
        _census().save_coverage_history(_census().empty_coverage_history(), str(path))
        assert _coverage_history_damage(path) is None


class TestTheGuardMessage:
    def test_it_names_the_artifact_the_reason_and_the_restore_remedy(self):
        message = _damaged_artifact_message(Path(_census().DEFAULT_HISTORY_OUT), 'the reason')
        assert 'the reason' in message
        assert 'git log -- plans/memory-metadata-coverage-history.json' in message
        assert 'not a defect' in message.lower()


# ---------------------------------------------------------------------------
# The live guards: one per committed artifact
# ---------------------------------------------------------------------------

class TestTheCommittedArtifacts:
    def test_the_probe_reads_the_report_the_census_writes(self):
        assert _probe().DEFAULT_CENSUS_PATH == Path(_census().DEFAULT_JSON_OUT)

    def test_the_committed_census_report_is_one_the_probe_can_derive_from(self):
        path = _probe().DEFAULT_CENSUS_PATH
        reason = _census_report_damage(path)
        assert reason is None, _damaged_artifact_message(path, reason)

    def test_the_committed_coverage_history_loads_under_the_current_schema(self):
        path = Path(_census().DEFAULT_HISTORY_OUT)
        reason = _coverage_history_damage(path)
        assert reason is None, _damaged_artifact_message(path, reason)
