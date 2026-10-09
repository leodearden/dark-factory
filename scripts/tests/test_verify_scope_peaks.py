"""Tests for verify_scope_peaks.py — gate grouping, percentiles and the split recommendation.

Fixture: project 'p' has two merge gates, each with two legs.
  gate 1 (t=0..100):   legs 4 GiB and 10 GiB
  gate 2 (t=1000..1100): legs 2 GiB and 20 GiB
so per-gate max-leg = [10, 20] -> p50 = 15, p90 = 19, max = 20; sum-of-legs max = 22.
"""
from __future__ import annotations

import json
import math

import verify_scope_peaks as mod

GIB = mod.GIB


def _rec(slug, start, end, gib, role='merge'):
    return mod.ScopeRecord(
        unit=f'df-verify-{slug}-aaaaaaaa-{start}', slug=slug, role=role, first_seen=start, last_seen=end,
        peak_bytes=int(gib * GIB), mem_total_gib=125.0, mem_available_gib_at_start=90.0,
        threads={'PYTEST_XDIST_AUTO_NUM_WORKERS': '16'},
    )


def _fixture():
    return [_rec('p', 0, 50, 4), _rec('p', 10, 100, 10), _rec('p', 1000, 1050, 2), _rec('p', 1010, 1100, 20)]


def test_classify_role_by_cwd_basename():
    assert mod.classify_role('_merge-verify') == 'merge'
    assert mod.classify_role('_merge-1a2b3c') == 'merge'
    assert mod.classify_role('_lane-07') == 'task'
    assert mod.classify_role('src') == 'other'


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


def test_split_recommendation_flips_at_limit():
    low = mod.summarise(mod.group_gates(_fixture()), {'p': 1.0})
    high = mod.summarise(mod.group_gates(_fixture()), {'p': 3.0})
    assert mod.split_recommendation(low).endswith('-> 16/16')
    assert mod.split_recommendation(high).endswith('-> 8/8 for Reify')


def test_records_round_trip_through_jsonl(tmp_path):
    path = tmp_path / 'peaks.jsonl'
    path.write_text(''.join(json.dumps(mod.asdict(r)) + '\n' for r in _fixture()))
    loaded = mod.load_records(path, since=500)
    assert [r.first_seen for r in loaded] == [1000, 1010]
    assert '| p | 2 |' in mod.render_report(_fixture(), {'p': 1.0})
