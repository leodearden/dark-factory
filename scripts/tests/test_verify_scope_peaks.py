"""Tests for verify_scope_peaks.py — the poll tick, gate grouping, percentiles and the split recommendation.

Fixture: project 'p' has two merge gates, each with two legs.
  gate 1 (t=0..100):   legs 4 GiB and 10 GiB
  gate 2 (t=1000..1100): legs 2 GiB and 20 GiB
so per-gate max-leg = [10, 20] -> p50 = 15, p90 = 19, max = 20; sum-of-legs max = 22.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import shutil
import time

import pytest
import verify_scope_peaks as mod

GIB = mod.GIB
MEMORY = mod.HostMemory(total_gib=125.0, available_gib=90.0)
# The shape orchestrator/src/orchestrator/verify.py::_verify_scope_name builds.
UNIT = 'df-verify-dark-factory-1a2b3c4d-0123456789ab.scope'
UTC_MIDNIGHT_2026_10_09 = 1791504000


def _rec(slug, start, end, gib, role='merge'):
    return mod.ScopeRecord(
        unit=f'df-verify-{slug}-aaaaaaaa-{start}', slug=slug, role=role, first_seen=start, last_seen=end,
        peak_bytes=int(gib * GIB), mem_total_gib=125.0, mem_available_gib_at_start=90.0,
        threads={'PYTEST_XDIST_AUTO_NUM_WORKERS': '16'},
    )


def _fixture():
    return [_rec('p', 0, 50, 4), _rec('p', 10, 100, 10), _rec('p', 1000, 1050, 2), _rec('p', 1010, 1100, 20)]


def _write_jsonl(path, records):
    path.write_text(''.join(json.dumps(mod.asdict(r)) + '\n' for r in records))


def _obs(unit, gib, cwd='_merge-verify'):
    return mod.ScopeObservation(unit=unit, peak_bytes=int(gib * GIB), cwd_basename=cwd, threads={})


def _fake_scope(slice_dir, unit, gib):
    scope = slice_dir / unit
    scope.mkdir()
    (scope / 'memory.peak').write_text(f'{int(gib * GIB)}\n')
    (scope / 'cgroup.procs').write_text(f'{os.getpid()}\n')
    return scope


def test_classify_role_by_cwd_basename():
    assert mod.classify_role('_merge-verify') == 'merge'
    assert mod.classify_role('_merge-1a2b3c') == 'merge'
    assert mod.classify_role('_lane-07') == 'task'
    assert mod.classify_role('src') == 'other'
    assert mod.classify_role(None) == 'other'


def test_scope_name_parses_the_orchestrator_unit_shape():
    match = mod.SCOPE_NAME.match(UNIT)
    assert match is not None
    assert match['slug'] == 'dark-factory'


def test_tick_keeps_the_max_peak_and_flushes_a_vanished_scope_with_its_last_sighting():
    live, ended = mod.tick({}, [_obs(UNIT, 4)], 10.0, MEMORY)
    live, ended = mod.tick(live, [_obs(UNIT, 9, cwd='src')], 15.0, MEMORY)
    live, ended = mod.tick(live, [_obs(UNIT, 2)], 20.0, MEMORY)
    assert ended == []
    live, ended = mod.tick(live, [], 25.0, MEMORY)
    assert live == {}
    (rec,) = ended
    assert (rec.first_seen, rec.last_seen, rec.peak_bytes, rec.truncated) == (10.0, 20.0, 9 * GIB, False)
    assert (rec.slug, rec.role, rec.cwd_basename) == ('dark-factory', 'merge', '_merge-verify')
    assert (rec.mem_total_gib, rec.mem_available_gib_at_start) == (125.0, 90.0)


def test_tick_ignores_names_outside_the_verify_scope_shape():
    assert mod.tick({}, [_obs('df-verify-legacy.scope', 4)], 10.0, MEMORY) == ({}, [])


def test_scope_root_is_the_process_whose_parent_is_outside_the_scope():
    assert mod.scope_root({50: 900, 900: 1}) == 900
    assert mod.scope_root({}) is None


def test_poll_flushes_an_ended_scope_and_truncates_live_ones_on_exit(tmp_path, monkeypatch):
    slice_dir = tmp_path / 'app.slice'
    slice_dir.mkdir()
    ending = _fake_scope(slice_dir, 'df-verify-p-1a2b3c4d-000000000001.scope', 4)
    staying = _fake_scope(slice_dir, 'df-verify-p-1a2b3c4d-000000000002.scope', 1)
    (tmp_path / '_merge-verify').mkdir()
    monkeypatch.chdir(tmp_path / '_merge-verify')
    sleeps = []

    def end_one_scope_then_stop(seconds):
        sleeps.append(seconds)
        if len(sleeps) > 1:
            raise SystemExit(143)
        shutil.rmtree(ending)
        (staying / 'memory.peak').write_text(f'{3 * GIB}\n')

    monkeypatch.setattr(time, 'sleep', end_one_scope_then_stop)
    out = tmp_path / 'peaks.jsonl'
    with pytest.raises(SystemExit):
        mod.poll(out, 5.0, str(slice_dir / 'df-verify-*.scope'))
    rows = {row['unit']: row for row in map(json.loads, out.read_text().splitlines())}
    assert (rows[ending.name]['peak_bytes'], rows[ending.name]['truncated']) == (4 * GIB, False)
    assert (rows[staying.name]['peak_bytes'], rows[staying.name]['truncated']) == (3 * GIB, True)
    assert {(row['role'], row['cwd_basename']) for row in rows.values()} == {('merge', '_merge-verify')}


def test_group_gates_splits_on_gap_and_ignores_role_mixing():
    gates = mod.group_gates(_fixture() + [_rec('p', 5, 60, 30, role='task')])
    merge = [g for g in gates if g.role == 'merge']
    assert [len(g.legs) for g in merge] == [2, 2]
    assert [g.role for g in gates].count('task') == 1


def test_summarise_percentiles_headroom_and_scaling():
    (summary,) = mod.summarise(mod.group_gates(_fixture()), {'p': 2.0})
    assert (summary.gates, summary.p50, summary.p90, summary.max) == (2, 15.0, 19.0, 20.0)
    assert summary.sum_legs_max == 22.0
    assert summary.recommended_max_gib == math.ceil(19.0 * 1.25)
    assert summary.scaled_p90 == 38.0
    assert summary.threads == {'PYTEST_XDIST_AUTO_NUM_WORKERS': {'16'}}


def test_percentile_rejects_an_empty_sample():
    with pytest.raises(ValueError):
        mod.percentile([], 0.5)


def test_split_recommendation_flips_at_limit():
    low = mod.summarise(mod.group_gates(_fixture()), {'p': 1.0})
    high = mod.summarise(mod.group_gates(_fixture()), {'p': 3.0})
    assert mod.split_recommendation(low).split == '16/16'
    assert mod.split_recommendation(high).split == '8/8 for Reify'
    assert mod.split_recommendation(high).scaled_p90_sum_gib == 57.0


def test_records_round_trip_through_jsonl(tmp_path):
    path = tmp_path / 'peaks.jsonl'
    _write_jsonl(path, _fixture())
    loaded = mod.load_records(path, since=500)
    assert [r.first_seen for r in loaded] == [1000, 1010]
    assert '| p | 2 |' in mod.render_report(_fixture(), {'p': 1.0})


def test_load_records_folds_a_unit_split_by_a_poller_restart(tmp_path):
    before_restart = dataclasses.replace(_rec('p', 0, 50, 4), truncated=True)
    after_restart = dataclasses.replace(before_restart, first_seen=70, last_seen=100, peak_bytes=6 * GIB, truncated=False)
    path = tmp_path / 'peaks.jsonl'
    _write_jsonl(path, [before_restart, after_restart])
    (rec,) = mod.load_records(path, since=0)
    assert (rec.first_seen, rec.last_seen, rec.peak_bytes, rec.truncated) == (0, 100, 6 * GIB, False)


def test_report_flags_scopes_the_poller_truncated():
    clean = mod.render_report(_fixture(), {'p': 1.0})
    flagged = mod.render_report(_fixture()[:-1] + [dataclasses.replace(_fixture()[-1], truncated=True)], {'p': 1.0})
    assert 'lower bound' not in clean
    assert 'lower bound' in flagged


@pytest.fixture
def local_time_in_bst(monkeypatch):
    monkeypatch.setenv('TZ', 'Europe/London')
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


@pytest.mark.usefixtures('local_time_in_bst')
def test_report_since_is_utc_midnight_even_under_local_summer_time(tmp_path, capsys):
    path = tmp_path / 'peaks.jsonl'
    _write_jsonl(path, [
        _rec('p', UTC_MIDNIGHT_2026_10_09 - 1800, UTC_MIDNIGHT_2026_10_09 - 1700, 4),
        _rec('p', UTC_MIDNIGHT_2026_10_09 + 1800, UTC_MIDNIGHT_2026_10_09 + 1900, 8),
    ])
    assert mod.main(['report', str(path), '--since', '2026-10-09']) == 0
    assert '| p | 1 |' in capsys.readouterr().out
