"""The ordinary pass decides exactly what it decided before task 10476's read/write split.

The golden was recorded on base f8e633a9ae, before the split
(plans/early-start-on-predecessor-branch-prd.md §5.4: "unchanged on a recorded
pass"). It holds decisions and writes only, over the committed live dump of
1,480 pending tasks plus synthetic tasks that make every split step act.
Re-record only on purpose, with RECORD_SCHEDULER_PASS=1.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from _scheduler_pass_scenario import build_scenario, record_passes

pytestmark = [pytest.mark.usefixtures('code_default_config'), pytest.mark.asyncio]

GOLDEN = Path(__file__).parent / 'fixtures' / 'scheduler_recorded_pass' / 'ordinary_pass_decisions.json'
RECORD_SWITCH = 'RECORD_SCHEDULER_PASS'


async def test_the_ordinary_pass_decides_what_the_golden_recorded(tmp_path):
    scenario = build_scenario(tmp_path)
    try:
        record = await record_passes(scenario)
    finally:
        await scenario.scheduler.aclose()

    if os.environ.get(RECORD_SWITCH) == '1':
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(record.decisions, indent=1, sort_keys=True) + '\n')
        pytest.skip(f'{RECORD_SWITCH}=1: golden re-recorded at {GOLDEN}')
    assert GOLDEN.exists(), f'no golden at {GOLDEN}; record one on purpose with {RECORD_SWITCH}=1'
    assert record.decisions == json.loads(GOLDEN.read_text())
