"""Dispatch scoring β (task 3819; plans/scheduler-dispatch-scoring-and-lock-layer-prd.md C2, I-4, I-6, row 4, row 9)."""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import orchestrator
from orchestrator.config import ScoringConfig
from orchestrator.pending_age import read_pending_age

REPO_ROOT = Path(orchestrator.__file__).resolve().parents[3]
FIXTURE_DIR = REPO_ROOT / 'orchestrator' / 'tests' / 'fixtures' / 'scheduler_candidates'
BASELINE_CSV = REPO_ROOT / 'plans' / 'evidence' / 'scheduler-scoring-2026-08-06' / 'candidates_scored.csv'


def load_candidate_fixture() -> dict:
    """The newest committed dump of scripts/dump_scheduler_candidate_fixture.py."""
    dumps = sorted(FIXTURE_DIR.glob('*.json'))
    assert dumps, f'no committed candidate fixture under {FIXTURE_DIR}'
    return json.loads(dumps[-1].read_text())


def fixture_captured_at(fixture: dict) -> datetime:
    return datetime.fromisoformat(fixture['provenance']['captured_at'])


def fixture_as_tasks(fixture: dict) -> list[dict]:
    """The fixture's candidates as scheduler task dicts."""
    return [
        {'id': c['id'], 'status': 'pending', 'priority': c['tier'],
         'metadata': {} if c['pending_since'] is None else {'pending_since': c['pending_since']}}
        for c in fixture['candidates']
    ]


def load_baseline_rows() -> list[dict]:
    with BASELINE_CSV.open(newline='') as fh:
        return list(csv.DictReader(fh))


class TestCandidateFixtureFloors:
    """Gate the fixture before anything is certified against it."""

    def test_fixture_is_a_realistic_armed_population(self):
        fixture = load_candidate_fixture()
        candidates = fixture['candidates']
        captured_at = fixture_captured_at(fixture)

        assert len(candidates) >= 250
        assert len({c['tier'] for c in candidates}) >= 3
        assert sum(c['transitive_count'] >= 1 for c in candidates) >= 20
        assert captured_at.utcoffset() is not None
        covered = sum(read_pending_age(t, captured_at).covered for t in fixture_as_tasks(fixture))
        assert covered / len(candidates) >= ScoringConfig().new_scoring_min_anchor_coverage


class TestBaselineEvidence:
    """The 2026-08-06 saturation the restructure exists to remove, derived from the committed CSV."""

    def test_baseline_saturation_is_what_the_csv_records(self):
        rows = load_baseline_rows()
        scores = Counter(float(r['score']) for r in rows)
        tie_score, tie_size = scores.most_common(1)[0]

        assert len(rows) == 294
        assert len(scores) == 47
        assert sum(r['capped'] == 'True' for r in rows) == 251
        assert (tie_score, tie_size) == (2999.0, 117)
