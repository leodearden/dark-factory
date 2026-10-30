"""The timeout-marker inversion guard, instantiated for tests/scripts.

The extractor, the band, the judgement, its failure messages and their
rationale live in shared/src/shared/testing_timeout_markers.py.  This module
supplies only this tree's inputs: its tests root, the verify test_command that
judges it, and its anti-vacuity floors.  There is no allowlist.  Two registered
module configs collect this tree, its own tests/scripts/orchestrator.yaml and
scripts/orchestrator.yaml, so the band's top edge is the LARGER of their
--timeout values: a marker below either one clamps that leg.  A config with no
--timeout fails the guard instead of dropping out of that comparison, because
its leg then runs under a budget this guard does not read.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml
from shared.testing_timeout_markers import (
    InversionVerdict,
    SweepFloors,
    TimeoutSite,
    TreeScan,
    judge_inversion_band,
    scan_timeout_marker_sites,
    verify_cli_timeout,
)

_TESTS_DIR = Path(__file__).resolve().parent
_COLLECTING_CONFIGS = (
    _TESTS_DIR / 'orchestrator.yaml',
    _TESTS_DIR.parents[1] / 'scripts' / 'orchestrator.yaml',
)

#: No named timeout constant is shared across this tree's modules, so a
#: marker spelled as a NAME reads as "no opinion".  The guard is a floor, with
#: the same stated limit as orchestrator's.
_SANCTIONED: Mapping[str, float] = {}

#: Set below the 128 files and 9 sites measured on 2026-10-30.
_FLOORS = SweepFloors(test_files=90, marker_sites=6)


def _collecting_test_commands() -> list[str]:
    return [
        yaml.safe_load(config.read_text(encoding='utf-8'))['test_command']
        for config in _COLLECTING_CONFIGS
    ]


def _judging_test_command(commands: Sequence[str]) -> str:
    unbudgeted = next(
        (command for command in commands if verify_cli_timeout(command) is None), None
    )
    if unbudgeted is not None:
        return unbudgeted
    return max(commands, key=lambda command: verify_cli_timeout(command) or 0)


@functools.cache
def _scan() -> TreeScan[tuple[str, TimeoutSite]]:
    return scan_timeout_marker_sites(_TESTS_DIR, _SANCTIONED)


def _judge(verify_test_command: str) -> InversionVerdict:
    return judge_inversion_band(
        _scan(),
        verify_test_command=verify_test_command,
        grandfathered=frozenset(),
        floors=_FLOORS,
    )


@functools.cache
def _verdict() -> InversionVerdict:
    return _judge(_judging_test_command(_collecting_test_commands()))


def test_a_collecting_config_without_a_timeout_fails_rather_than_dropping_out() -> None:
    failure = _judge(
        _judging_test_command(('pytest --timeout=300 tests/scripts/', 'pytest tests/scripts/'))
    ).budget_failure
    assert failure is not None and "'pytest tests/scripts/'" in failure, failure


def test_the_verify_budget_is_configured() -> None:
    failure = _verdict().budget_failure
    assert failure is None, failure


def test_no_new_timeout_marker_sits_in_the_inversion_band() -> None:
    failure = _verdict().new_offender_failure
    assert failure is None, failure


def test_the_marker_census_is_not_vacuous() -> None:
    failure = _verdict().census_failure
    assert failure is None, failure
