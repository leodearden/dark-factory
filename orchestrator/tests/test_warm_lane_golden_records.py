"""Golden lane-state and plan.json records, regenerated from the real serialisers.

The files in ``orchestrator/tests/fixtures/warm_lane_golden/`` are the canonical
golden records for reify's warm-lane-audit (reify PRD
``docs/prds/v0_6/warm-lane-scripts-one-lock-one-env-one-oracle.md``, stream S20
leaf L5, Companion D5; ruled Q4 a). Each test regenerates one record through its
real producer and asserts byte equality with the checked-in golden:

- ``lane-state.json``: ``orchestrator/src/orchestrator/lane_lifecycle.py::LaneRecord.to_json``
  for one fixed pinned (ASSIGNED) record with every optional field set;
- ``lane-state-released.json``: the same serialiser for one fixed RELEASED
  record with every optional field left null, the free-lane shape;
- ``lane-states.json``: the values of
  ``orchestrator/src/orchestrator/lane_lifecycle.py::LaneState``, in enum order,
  framed by this module as an indent-2 JSON list with a trailing newline;
- ``plan.json``: the bytes ``orchestrator/src/orchestrator/mcp/plan_tools.py::create_server``'s
  tools persist through ``orchestrator/src/orchestrator/artifacts.py::TaskArtifacts``.

Refresh procedure. Never hand-edit a golden. When a deliberate serialiser change
turns this module red, copy the produced bytes named in the failure message over
the golden. Then copy every golden byte-for-byte into reify's
``tests/infra/fixtures/warm_lane_golden/`` and bump reify's
``tests/infra/fixtures/warm_lane_golden/PROVENANCE.md``.
"""

import json
from pathlib import Path

import pytest
from fastmcp import Client

from orchestrator.artifacts import TaskArtifacts
from orchestrator.lane_lifecycle import LaneRecord, LaneState
from orchestrator.mcp import plan_markup_stamp, plan_tools

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

GOLDEN_RELEASED_LANE_RECORD = LaneRecord(
    state=LaneState.RELEASED,
    updated_at='2026-10-12T09:45:00.654321+00:00',
)

GOLDEN_PLAN_ANALYSIS = 'Fixed analysis for the golden plan record.'
GOLDEN_PLAN_FILES = ['orchestrator/src/example.py', 'orchestrator/tests/test_example.py']
PREREQ_COMMIT = 'a1' * 20
STEP_COMMIT = 'b2' * 20

GOLDEN_PLAN_CALLS: list[tuple[str, dict[str, object]]] = [
    (
        'create_plan',
        {
            'task_id': GOLDEN_TASK_ID,
            'title': GOLDEN_LANE_RECORD.title,
            'analysis': GOLDEN_PLAN_ANALYSIS,
            'files': GOLDEN_PLAN_FILES,
        },
    ),
    ('add_prerequisite', {'prereq_id': 'pre-1', 'description': 'Set up the example fixture.'}),
    ('add_plan_step', {'step_id': 'step-1', 'step_type': 'test', 'description': 'RED: pin the example.'}),
    ('add_plan_step', {'step_id': 'step-2', 'step_type': 'impl', 'description': 'GREEN: implement the example.'}),
    ('mark_step_done', {'step_id': 'pre-1', 'commit_sha': PREREQ_COMMIT}),
    ('mark_step_done', {'step_id': 'step-1', 'commit_sha': STEP_COMMIT}),
]


@pytest.fixture(autouse=True)
def _clear_pending_refusals():
    plan_markup_stamp.clear_pending()
    yield
    plan_markup_stamp.clear_pending()


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
        f'refresh: cp {produced_path} {golden_path}, then copy every '
        f'golden into reify tests/infra/fixtures/warm_lane_golden/ and bump '
        f'its PROVENANCE.md (see this module docstring)'
    )


async def _produce_plan_bytes(worktree: Path) -> bytes:
    worktree.mkdir()
    artifacts = TaskArtifacts(worktree)
    server = plan_tools.create_server(artifacts)
    async with Client(server) as client:
        for tool, arguments in GOLDEN_PLAN_CALLS:
            result = await client.call_tool(tool, arguments)
            assert result.data['status'] == 'ok', f'{tool} refused: {result.data!r}'
    return (artifacts.root / 'plan.json').read_bytes()


def test_lane_state_golden_is_lane_record_to_json(tmp_path):
    produced = GOLDEN_LANE_RECORD.to_json().encode('utf-8')
    _assert_matches_golden('lane-state.json', produced, tmp_path / 'produced')


def test_released_lane_state_golden_is_null_field_lane_record_to_json(tmp_path):
    produced = GOLDEN_RELEASED_LANE_RECORD.to_json().encode('utf-8')
    _assert_matches_golden('lane-state-released.json', produced, tmp_path / 'produced')


def test_lane_states_golden_lists_every_lane_state_member(tmp_path):
    produced = (json.dumps([member.value for member in LaneState], indent=2) + '\n').encode('utf-8')
    _assert_matches_golden('lane-states.json', produced, tmp_path / 'produced')


@pytest.mark.asyncio
async def test_plan_golden_is_plan_tools_writer_output(tmp_path):
    produced = await _produce_plan_bytes(tmp_path / 'lane')
    _assert_matches_golden('plan.json', produced, tmp_path / 'produced')
