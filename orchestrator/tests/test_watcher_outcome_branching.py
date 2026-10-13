"""Every consumer of `scripts/watcher-rearm.sh` branches on its real outcomes (task 3606).

The wrapper's exit-code -> `WATCHER_REARM_OUTCOME` mapping has one home: the
script itself. The vocabulary below is DERIVED by running that script over a
stub watcher (its `WATCHER_REARM_PYTHON` seam) for every exit code 0-255, never
restated here, so a new outcome fails every consumer at once.

Every assertion checks a consumer's outcome table against values derived by
running the producer. The rotation prompt carries no table of its own: its
guidance points at the auto skill section that does, and that pointer must resolve.
"""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pytest

from orchestrator.agents.skill_prompt import load_skill_system_prompt
from orchestrator.agents.watcher_rearm_outcome_guidance import (
    WATCHER_REARM_OUTCOME_GUIDANCE,
    WATCHER_REARM_OUTCOME_SECTION,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REARM_SCRIPT = _REPO_ROOT / 'scripts' / 'watcher-rearm.sh'
_OUTCOME_MARKER = re.compile(r'WATCHER_REARM_OUTCOME: (\w+) exit=(\d+)')
_OUTCOME_TOKEN = re.compile(r'WATCHER_REARM_OUTCOME: (\w+)')
_PROBE_RCS = range(256)
_PROBE_WORKERS = 8
_EXIT_COLUMN = 'Exit'

pytestmark = pytest.mark.xdist_group('watcher-rearm-producer')

_ARTIFACTS: dict[str, Callable[[], str]] = {
    'auto-skill': lambda: load_skill_system_prompt('escalation-watcher-auto'),
    'l2-skill': lambda: load_skill_system_prompt('escalation-watcher'),
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

    def probe(rc: int) -> subprocess.CompletedProcess[str]:
        env = {
            'PATH': os.environ.get('PATH', ''),
            'DARK_FACTORY_ROOT': str(tooling_root),
            'WATCHER_REARM_PYTHON': str(stub),
            'STUB_RC': str(rc),
        }
        return _run_rearm(args, env)

    with ThreadPoolExecutor(max_workers=_PROBE_WORKERS) as pool:
        probes = dict(zip(_PROBE_RCS, pool.map(probe, _PROBE_RCS), strict=True))

    outcome_by_rc: dict[int, str] = {}
    for rc, result in probes.items():
        match = _OUTCOME_MARKER.search(result.stderr)
        assert match is not None, f'probe rc={rc} printed no outcome marker: {result.stderr!r}'
        assert int(match.group(2)) == rc == result.returncode, (
            f'probe rc={rc}: marker {match.group(0)!r}, returncode {result.returncode}'
        )
        outcome_by_rc[rc] = match.group(1)

    outcomes: dict[str, set[int]] = {}
    for rc, outcome in outcome_by_rc.items():
        outcomes.setdefault(outcome, set()).add(rc)
    catch_all = max(outcomes, key=lambda outcome: len(outcomes[outcome]))

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


def _cells(row: str) -> list[str]:
    return [cell.strip() for cell in row.strip().strip('|').split('|')]


def _exit_cells_by_outcome(text: str) -> dict[str, list[str]]:
    """The Exit cell of every markdown table row carrying each outcome marker."""
    exit_cells: dict[str, list[str]] = {}
    header: list[str] | None = None
    for line in text.splitlines():
        if not line.lstrip().startswith('|'):
            header = None
            continue
        if header is None:
            header = _cells(line)
            continue
        exit_cell = dict(zip(header, _cells(line), strict=False)).get(_EXIT_COLUMN, '')
        for token in _OUTCOME_TOKEN.findall(line):
            exit_cells.setdefault(token, []).append(exit_cell)
    return exit_cells


def _names_code(exit_cell: str, rc: int) -> bool:
    return re.search(rf'(?<!\d){rc}(?!\d)', exit_cell) is not None


def _heading_level(line: str) -> int:
    hashes = len(line) - len(line.lstrip('#'))
    return hashes if hashes and line[hashes:].startswith(' ') else 0


def _section_under(text: str, title: str) -> str | None:
    """The body of the markdown section titled `title`; '#' lines inside code fences are not headings."""
    body: list[str] | None = None
    level = 0
    in_fence = False
    for line in text.splitlines():
        if line.lstrip().startswith('```'):
            in_fence = not in_fence
        heading = 0 if in_fence else _heading_level(line)
        if body is None:
            if heading and line[heading:].strip() == title:
                body, level = [], heading
        elif 0 < heading <= level:
            break
        else:
            body.append(line)
    return None if body is None else '\n'.join(body)


@pytest.mark.parametrize('artifact_id', sorted(_ARTIFACTS))
class TestOutcomeBranchContract:
    def test_one_branch_row_per_producer_outcome(
        self, artifact_id: str, producer_contract: ProducerContract
    ) -> None:
        exit_cells = _exit_cells_by_outcome(_ARTIFACTS[artifact_id]())
        miscounted = {
            outcome: len(exit_cells.get(outcome, []))
            for outcome in producer_contract.outcomes
            if len(exit_cells.get(outcome, [])) != 1
        }
        assert not miscounted, (
            f'{artifact_id}: each producer outcome needs exactly one table row '
            f'carrying its marker; row counts {miscounted}'
        )

    def test_row_names_every_exit_code_the_producer_maps_to_it(
        self, artifact_id: str, producer_contract: ProducerContract
    ) -> None:
        exit_cells = _exit_cells_by_outcome(_ARTIFACTS[artifact_id]())
        unnamed: dict[str, list[int]] = {}
        for outcome, rcs in producer_contract.outcomes.items():
            if outcome == producer_contract.catch_all:
                continue
            missing = sorted(
                rc
                for rc in rcs
                if not any(_names_code(cell, rc) for cell in exit_cells.get(outcome, []))
            )
            if missing:
                unnamed[outcome] = missing
        assert not unnamed, (
            f'{artifact_id}: outcome rows\' {_EXIT_COLUMN} cells omit exit codes the wrapper '
            f'maps to them: {unnamed}'
        )

    def test_catch_all_row_names_the_no_marker_usage_code(
        self, artifact_id: str, producer_contract: ProducerContract
    ) -> None:
        exit_cells = _exit_cells_by_outcome(_ARTIFACTS[artifact_id]()).get(
            producer_contract.catch_all, []
        )
        assert any(_names_code(cell, producer_contract.usage_rc) for cell in exit_cells), (
            f'{artifact_id}: the {producer_contract.catch_all} row\'s {_EXIT_COLUMN} cell must '
            f'name exit {producer_contract.usage_rc}, the usage error that prints no marker'
        )


class TestRotationGuidancePointer:
    def test_points_at_the_auto_skill_section_holding_every_outcome_row(
        self, producer_contract: ProducerContract
    ) -> None:
        assert WATCHER_REARM_OUTCOME_SECTION in WATCHER_REARM_OUTCOME_GUIDANCE
        section = _section_under(
            load_skill_system_prompt('escalation-watcher-auto'), WATCHER_REARM_OUTCOME_SECTION
        )
        assert section is not None, (
            f'the rotation guidance points at a "{WATCHER_REARM_OUTCOME_SECTION}" section '
            'the auto skill does not have'
        )
        missing = sorted(set(producer_contract.outcomes) - set(_exit_cells_by_outcome(section)))
        assert not missing, (
            f'the auto skill\'s "{WATCHER_REARM_OUTCOME_SECTION}" section, which the rotation '
            f'guidance points at, has no table row for outcomes {missing}'
        )
