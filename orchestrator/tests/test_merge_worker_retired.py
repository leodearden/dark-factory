"""The serial ``MergeWorker`` is gone -- from production and from the tests.

MQ-refactor task ν (R7b) removed the serial worker from
``orchestrator.merge_queue`` but kept a frozen copy of it under
``orchestrator/tests`` so the tests built on it kept running. Task 5034 (PRD
``plans/merge-lane-quality-prd.md`` task δ, decision 4) discarded that copy and
re-homed the behaviours those tests checked onto the production lane,
``orchestrator.merge_lane.MergeLane``, driven through the fakes in
``_merge_lane_fakes.py``. A fallback that is never exercised is not a fallback
(INV-10), so neither half may come back.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from _orch_helpers import WHOLE_TREE_SCAN_TEST_TIMEOUT

import orchestrator.merge_lane as merge_lane
import orchestrator.merge_queue as mq

# test_no_copy_of_the_serial_worker_in_the_test_tree reads every *.py under
# this directory; test_whole_tree_scan_timeout_guard.py says why such a sweep
# carries this mark.
pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

_RETIRED_CLASS = 'MergeWorker'
_TESTS_DIR = Path(__file__).resolve().parent

#: Anti-vacuity floor (600+ files today): a sweep that silently reads nothing
#: must fail rather than report a clean tree.
_MIN_EXPECTED_TEST_FILES = 400


def _defines_retired_class(source: str) -> bool:
    """Whether *source* defines a class named ``MergeWorker`` at any depth.

    A mention in a comment or docstring is not a definition. The substring test
    only spares parsing the hundreds of files that cannot contain one.
    """
    if f'class {_RETIRED_CLASS}' not in source:
        return False
    return any(
        isinstance(node, ast.ClassDef) and node.name == _RETIRED_CLASS
        for node in ast.walk(ast.parse(source))
    )


def test_the_production_lane_is_the_only_merge_worker() -> None:
    assert not hasattr(mq, _RETIRED_CLASS), (
        'the serial MergeWorker was retired from orchestrator.merge_queue; '
        'the production worker is SpeculativeMergeWorker'
    )
    assert merge_lane.MergeLane is mq.SpeculativeMergeWorker


def test_detector_sees_a_definition_and_not_a_mention() -> None:
    assert _defines_retired_class('class Outer:\n    class MergeWorker:\n        pass\n')
    assert not _defines_retired_class(
        '"""Once there was a class MergeWorker here."""\n'
        '# class MergeWorker(_WipHaltMixin): was its header\n'
        'class SpeculativeMergeWorker:\n    pass\n'
    )


def test_no_copy_of_the_serial_worker_in_the_test_tree() -> None:
    sources = sorted(_TESTS_DIR.rglob('*.py'))
    assert len(sources) >= _MIN_EXPECTED_TEST_FILES, (
        f'swept only {len(sources)} files under {_TESTS_DIR}'
    )
    copies = [
        path.relative_to(_TESTS_DIR).as_posix()
        for path in sources
        if _defines_retired_class(path.read_text(encoding='utf-8'))
    ]
    assert copies == [], (
        f'a serial MergeWorker is defined again in {copies}. Drive the '
        'production lane instead: make_lane / merge_through_lane in '
        '_merge_lane_fakes.py.'
    )
