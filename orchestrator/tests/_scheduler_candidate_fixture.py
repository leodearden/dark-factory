"""Loaders for the committed live candidate dumps under ``fixtures/scheduler_candidates/``."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import orchestrator

FIXTURE_DIR = Path(orchestrator.__file__).resolve().parents[3] / 'orchestrator' / 'tests' / 'fixtures' / 'scheduler_candidates'


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
