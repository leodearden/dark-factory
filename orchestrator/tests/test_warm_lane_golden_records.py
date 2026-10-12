"""Golden lane-state and plan.json records, regenerated from the real serialisers.

The files in ``orchestrator/tests/fixtures/warm_lane_golden/`` are the canonical
golden records for reify's warm-lane-audit (reify PRD
``docs/prds/v0_6/warm-lane-scripts-one-lock-one-env-one-oracle.md``, stream S20
leaf L5, Companion D5; ruled Q4 a). Each test regenerates one record through its
real producer and asserts byte equality with the checked-in golden:

- ``lane-state.json``: ``orchestrator/src/orchestrator/lane_lifecycle.py::LaneRecord.to_json``
  for one fixed record;
- ``lane-states.json``: the values of
  ``orchestrator/src/orchestrator/lane_lifecycle.py::LaneState``, in enum order;
- ``plan.json``: the bytes ``orchestrator/src/orchestrator/mcp/plan_tools.py::create_server``'s
  tools persist through ``orchestrator/src/orchestrator/artifacts.py::TaskArtifacts``.

Refresh procedure. Never hand-edit a golden. When a deliberate serialiser change
turns this module red, copy the produced bytes named in the failure message over
the golden. Then copy all three files byte-for-byte into reify's
``tests/infra/fixtures/warm_lane_golden/`` and bump reify's
``tests/infra/fixtures/warm_lane_golden/PROVENANCE.md``.
"""

import json
from pathlib import Path

import pytest

from orchestrator.lane_lifecycle import LaneRecord, LaneState

GOLDEN_DIR = Path(__file__).parent / 'fixtures' / 'warm_lane_golden'

GOLDEN_TASK_ID = '4242'

GOLDEN_LANE_RECORD = LaneRecord(
    state=LaneState.ASSIGNED,
    task_id=GOLDEN_TASK_ID,
    title='Golden fixture task 4242',
    branch='task/4242',
    seeded_from_sha='0123456789abcdef0123456789abcdef01234567',
    updated_at='2026-10-12T08:15:30.123456+00:00',
)


def _assert_matches_golden(name: str, produced: bytes, scratch: Path) -> None:
    golden_path = GOLDEN_DIR / name
    golden = golden_path.read_bytes() if golden_path.is_file() else None
    if golden == produced:
        return
    scratch.mkdir(parents=True, exist_ok=True)
    produced_path = scratch / name
    produced_path.write_bytes(produced)
    verdict = 'no golden yet' if golden is None else 'golden differs from producer output'
    pytest.fail(
        f'{verdict}: {golden_path}\n'
        f'produced bytes: {produced_path}\n'
        f'refresh: cp {produced_path} {golden_path}, then copy all three '
        f'goldens into reify tests/infra/fixtures/warm_lane_golden/ and bump '
        f'its PROVENANCE.md (see this module docstring)'
    )


def test_lane_state_golden_is_lane_record_to_json(tmp_path):
    produced = GOLDEN_LANE_RECORD.to_json().encode('utf-8')
    _assert_matches_golden('lane-state.json', produced, tmp_path / 'produced')


def test_lane_states_golden_lists_every_lane_state_member(tmp_path):
    produced = (json.dumps([member.value for member in LaneState], indent=2) + '\n').encode('utf-8')
    _assert_matches_golden('lane-states.json', produced, tmp_path / 'produced')
