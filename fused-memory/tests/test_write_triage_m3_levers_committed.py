"""The committed M3 lever replay re-derives from git alone (task 7519).

The report ``calibration/write_triage_m3_levers_report.json`` is scored from
committed files only: the soak items, verdicts and strata, the text-free
replay features, the hand labels of the M3 hits, and the arm rows under
``calibration/write_triage_m3_lever_arms/``. None of the gitignored soak
texts is needed. The checks are self-consistency only. Nothing here binds to
the live ``JUDGE_SYSTEM_PROMPT``, so a follow-up that ships a lever leaves
this file alone.
"""
from __future__ import annotations

import functools
import json
import re
import types
from pathlib import Path
from typing import Any

import pytest
from _fm_helpers import load_script_module

SCRIPTS = Path(__file__).parent.parent / 'scripts'
CALIBRATION = Path(__file__).parent.parent / 'calibration'
REPORT = CALIBRATION / 'write_triage_m3_levers_report.json'
ARMS_DIR = CALIBRATION / 'write_triage_m3_lever_arms'
ARM_KEYS = ('prompt_sha256', 'model', 'effort', 'cases_sha256')


@functools.cache
def _mod() -> types.ModuleType:
    return load_script_module(
        SCRIPTS / 'score_write_triage_m3_levers.py', mod_name='score_write_triage_m3_levers',
    )


@functools.cache
def _report() -> dict[str, Any]:
    return json.loads(REPORT.read_text())


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@functools.cache
def _features() -> tuple[Any, ...]:
    cases = load_script_module(
        SCRIPTS / 'write_triage_m3_replay_cases.py', mod_name='write_triage_m3_replay_cases',
    )
    return cases.read_features(CALIBRATION / 'write_triage_m3_replay_cases.jsonl')


def _arm_rows(name: str) -> list[dict[str, Any]]:
    return _jsonl(ARMS_DIR / f'{name}.jsonl')


@functools.cache
def _inputs() -> dict[str, Any]:
    return _mod().committed_inputs()


def _scoring_inputs() -> dict[str, Any]:
    return {key: value for key, value in _inputs().items() if key != 'labels'}


def _committed_arm_names() -> list[str]:
    return list(_report()['arm_order'])


class TestTheCommittedLeverReport:
    """Re-scoring reproduces the committed report, one arm per test.

    Each arm costs one or two full ``score_reader`` passes, so the arms are
    split across tests to spread them over workers. Together the tests
    re-derive every key of the report.
    """

    def test_the_report_holds_exactly_the_tested_parts(self) -> None:
        report = _report()
        assert set(report) == {
            'rubric', 'baseline', 'arm_order', 'provenance', 'strata', 'soak',
            'classification', 'arms', 'repro', 'partial',
        }
        assert sorted(report['arm_order']) == sorted(report['arms'])

    def test_everything_but_the_arms_reproduces(self) -> None:
        report = _report()
        baseline = report['baseline']
        rescored = _mod().build_report(
            **_inputs(), arms={baseline: _arm_rows(baseline)}, baseline=baseline,
            partial=report['partial'],
            provenance={'shipped_prompt_commit': report['provenance']['shipped_prompt_commit']},
            repro={name: _arm_rows(name) for name in report['repro']},
        )
        for key in set(report) - {'arms', 'arm_order'}:
            assert rescored[key] == report[key], key

    @pytest.mark.parametrize('name', _committed_arm_names() if REPORT.exists() else ['missing'])
    def test_each_arm_reproduces(self, name: str) -> None:
        report = _report()
        baseline = report['baseline']
        rescored = _mod().arm_report(
            _arm_rows(name), baseline_rows=None if name == baseline else _arm_rows(baseline),
            **_scoring_inputs(),
        )
        committed = json.loads(json.dumps(report['arms'][name]))
        assert json.loads(json.dumps(rescored, sort_keys=True)) == committed

    def test_the_markdown_is_the_rendered_report(self) -> None:
        rendered = _mod().render_markdown(_report())
        assert (CALIBRATION / 'write_triage_m3_levers_report.md').read_text() == rendered

    def test_every_scored_arm_is_complete_over_its_scope(self) -> None:
        every_case = {feature.entry_id for feature in _features()}
        judge_band = {feature.entry_id for feature in _features() if feature.soak_band == 'judge'}
        for name in _report()['arms']:
            assert {row['entry_id'] for row in _arm_rows(name)} == every_case, name
            assert len(_arm_rows(name)) == len(every_case), name
        for name in _report()['repro']:
            assert {row['entry_id'] for row in _arm_rows(name)} == judge_band, name

    def test_each_arms_rows_carry_one_provenance_matching_the_report(self) -> None:
        report = _report()
        named = {**report['arms'], **report['repro']}
        assert named, 'the report scores no arm'
        for name, arm in named.items():
            rows = _arm_rows(name)
            for key in ARM_KEYS:
                assert {row[key] for row in rows} == {arm['provenance'][key]}, (name, key)
            assert arm['provenance']['rows'] == len(rows)
            assert {row['arm'] for row in rows} == {name}

    def test_no_arm_is_left_partial(self) -> None:
        """The plan doc reads complete arms only; the partial path is pinned synthetically."""
        assert _report()['partial'] == {}

    def test_the_labels_cover_exactly_the_reports_m3_hits(self) -> None:
        report = _report()
        labels = _jsonl(CALIBRATION / 'write_triage_m3_hidden_conflicts.jsonl')
        hits = {hit['entry_id'] for hit in report['classification']['by_hit']}
        assert {label['entry_id'] for label in labels} == hits
        assert len(labels) == len(hits) == report['soak']['m3']['hits']

    def test_the_baseline_arms_ran_the_recorded_shipped_prompt(self) -> None:
        provenance = _report()['provenance']
        assert re.fullmatch(r'[0-9a-f]{10,40}', provenance['shipped_prompt_commit'])
        assert re.fullmatch(r'[0-9a-f]{64}', provenance['shipped_prompt_sha256'])
        for name in ('f3p', 'f3p-repeat'):
            assert {row['prompt_sha256'] for row in _arm_rows(name)} == {
                provenance['shipped_prompt_sha256'],
            }, name

    def test_every_arm_file_on_disk_is_named_by_the_report(self) -> None:
        report = _report()
        named = {*report['arms'], *report['repro'], *report['partial']}
        assert {path.stem for path in ARMS_DIR.glob('*.jsonl')} <= named
