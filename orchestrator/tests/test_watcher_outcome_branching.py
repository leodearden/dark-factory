"""Every consumer of `scripts/watcher-rearm.sh` branches on its real outcomes (task 3606).

The wrapper's exit-code -> `WATCHER_REARM_OUTCOME` mapping has one home: the
script itself. The vocabulary below is DERIVED by running that script over a
stub watcher (its `WATCHER_REARM_PYTHON` seam), never restated here, so a new
outcome fails every consumer at once.

Every assertion checks a consumer's outcome table against values derived by
running the producer.
"""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

from orchestrator.agents.skill_prompt import load_skill_system_prompt
from orchestrator.agents.watcher_rearm_outcome_guidance import WATCHER_REARM_OUTCOME_GUIDANCE

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REARM_SCRIPT = _REPO_ROOT / 'scripts' / 'watcher-rearm.sh'
_OUTCOME_MARKER = re.compile(r'WATCHER_REARM_OUTCOME: (\w+) exit=(\d+)')
_OUTCOME_TOKEN = re.compile(r'WATCHER_REARM_OUTCOME: (\w+)')
_PROBE_RCS = (0, 124, 137, 143, 144, 1, 3)
_CATCH_ALL_PROBE_RCS = (1, 3)

_ARTIFACTS: dict[str, Callable[[], str]] = {
    'auto-skill': lambda: load_skill_system_prompt('escalation-watcher-auto'),
    'l2-skill': lambda: load_skill_system_prompt('escalation-watcher'),
    'rotation-guidance': lambda: WATCHER_REARM_OUTCOME_GUIDANCE,
}


@dataclass(frozen=True)
class ProducerContract:
    outcomes: dict[str, frozenset[int]]
    catch_all: str
    usage_rc: int


def _run_rearm(args: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ['bash', str(_REARM_SCRIPT), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.fixture(scope='module')
def producer_contract(tmp_path_factory: pytest.TempPathFactory) -> ProducerContract:
    work = tmp_path_factory.mktemp('rearm-producer')
    stub = work / 'stub-watcher'
    stub.write_text('#!/bin/sh\nexit "$STUB_RC"\n')
    stub.chmod(0o755)
    tooling_root = work / 'df-root'
    tooling_root.mkdir()
    args = ['--queue-dir', str(work / 'queue'), '--level', '1', '--timeout', '1']

    outcome_by_rc: dict[int, str] = {}
    for rc in _PROBE_RCS:
        env = {
            'PATH': os.environ.get('PATH', ''),
            'DARK_FACTORY_ROOT': str(tooling_root),
            'WATCHER_REARM_PYTHON': str(stub),
            'STUB_RC': str(rc),
        }
        result = _run_rearm(args, env)
        match = _OUTCOME_MARKER.search(result.stderr)
        assert match is not None, f'probe rc={rc} printed no outcome marker: {result.stderr!r}'
        assert int(match.group(2)) == rc == result.returncode, (
            f'probe rc={rc}: marker {match.group(0)!r}, returncode {result.returncode}'
        )
        outcome_by_rc[rc] = match.group(1)

    outcomes: dict[str, set[int]] = {}
    for rc, outcome in outcome_by_rc.items():
        outcomes.setdefault(outcome, set()).add(rc)
    catch_all = outcome_by_rc[_CATCH_ALL_PROBE_RCS[0]]
    assert {outcome_by_rc[rc] for rc in _CATCH_ALL_PROBE_RCS} == {catch_all}, outcome_by_rc

    usage = _run_rearm(['--no-such-flag'], {'PATH': os.environ.get('PATH', '')})
    assert usage.returncode != 0, usage
    assert _OUTCOME_MARKER.search(usage.stderr) is None, (
        f'a usage error must print no outcome marker: {usage.stderr!r}'
    )

    return ProducerContract(
        outcomes={outcome: frozenset(rcs) for outcome, rcs in outcomes.items()},
        catch_all=catch_all,
        usage_rc=usage.returncode,
    )


def _branch_rows(text: str) -> dict[str, list[str]]:
    rows: dict[str, list[str]] = {}
    table_rows = (line for line in text.splitlines() if line.lstrip().startswith('|'))
    for row in table_rows:
        for token in _OUTCOME_TOKEN.findall(row):
            rows.setdefault(token, []).append(row)
    return rows


def _names_code(row: str, rc: int) -> bool:
    return re.search(rf'(?<!\d){rc}(?!\d)', row) is not None


@pytest.mark.parametrize('artifact_id', sorted(_ARTIFACTS))
class TestOutcomeBranchContract:
    def test_one_branch_row_per_producer_outcome(
        self, artifact_id: str, producer_contract: ProducerContract
    ) -> None:
        rows = _branch_rows(_ARTIFACTS[artifact_id]())
        miscounted = {
            outcome: len(rows.get(outcome, []))
            for outcome in producer_contract.outcomes
            if len(rows.get(outcome, [])) != 1
        }
        assert not miscounted, (
            f'{artifact_id}: each producer outcome needs exactly one table row '
            f'carrying its marker; row counts {miscounted}'
        )

    def test_row_names_every_exit_code_the_producer_maps_to_it(
        self, artifact_id: str, producer_contract: ProducerContract
    ) -> None:
        rows = _branch_rows(_ARTIFACTS[artifact_id]())
        unnamed: dict[str, list[int]] = {}
        for outcome, rcs in producer_contract.outcomes.items():
            if outcome == producer_contract.catch_all:
                continue
            missing = sorted(
                rc for rc in rcs if not any(_names_code(row, rc) for row in rows.get(outcome, []))
            )
            if missing:
                unnamed[outcome] = missing
        assert not unnamed, (
            f'{artifact_id}: outcome rows omit exit codes the wrapper maps to them: {unnamed}'
        )

    def test_catch_all_row_names_the_no_marker_usage_code(
        self, artifact_id: str, producer_contract: ProducerContract
    ) -> None:
        rows = _branch_rows(_ARTIFACTS[artifact_id]()).get(producer_contract.catch_all, [])
        assert any(_names_code(row, producer_contract.usage_rc) for row in rows), (
            f'{artifact_id}: the {producer_contract.catch_all} row must name exit '
            f'{producer_contract.usage_rc}, the usage error that prints no marker'
        )
