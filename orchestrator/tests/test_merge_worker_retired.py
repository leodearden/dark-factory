"""The serial ``MergeWorker`` is gone -- from production and from the tests.

MQ-refactor task ν (R7b) removed the serial worker from
``orchestrator.merge_lane.worker`` but kept a frozen copy of it under
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
from _tree_scan import (
    ORCHESTRATOR_SRC_DIR,
    TESTS_DIR,
    TREE_PASS_XDIST_GROUP,
    ParsedModule,
    TreeAnalysis,
    TreePass,
    python_files,
)

import orchestrator.merge_lane as merge_lane
import orchestrator.merge_lane.worker as mq

# The family mark and shared group: _orch_helpers.py::WHOLE_TREE_SCAN_TEST_TIMEOUT, _tree_scan.py.
pytestmark = [
    pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT),
    pytest.mark.xdist_group(TREE_PASS_XDIST_GROUP),
]

_RETIRED_CLASS = 'MergeWorker'
_TESTS_DIR = TESTS_DIR
_SRC_DIR = ORCHESTRATOR_SRC_DIR

#: Anti-vacuity floors, one per swept tree (630 test files and 155 source
#: files today): a sweep that silently reads nothing must fail rather than
#: report a clean tree.
_MIN_EXPECTED_FILES = {_TESTS_DIR: 400, _SRC_DIR: 100}


def _defines_retired_class(source: str) -> bool:
    """Whether *source* defines a class named ``MergeWorker`` at any depth.

    A mention in a comment or docstring is not a definition. The substring test
    only spares parsing the hundreds of files that cannot contain one.
    """
    return _may_define_retired_class(source) and _defines_retired_class_in(ast.parse(source))


def _may_define_retired_class(source: str) -> bool:
    return f'class {_RETIRED_CLASS}' in source


def _defines_retired_class_in(tree: ast.Module) -> bool:
    return any(
        isinstance(node, ast.ClassDef) and node.name == _RETIRED_CLASS
        for node in ast.walk(tree)
    )


def _retired_class_defined(module: ParsedModule) -> bool:
    return _may_define_retired_class(module.source) and _defines_retired_class_in(module.tree)


ANALYSIS = TreeAnalysis(
    name=Path(__file__).stem,
    files=lambda: python_files(_TESTS_DIR, _SRC_DIR),
    extract=_retired_class_defined,
)
TREE_ANALYSES = (ANALYSIS,)


def test_the_lane_facade_exports_the_production_worker() -> None:
    assert merge_lane.MergeLane is mq.SpeculativeMergeWorker


def test_detector_sees_a_definition_and_not_a_mention() -> None:
    assert _defines_retired_class('class Outer:\n    class MergeWorker:\n        pass\n')
    assert not _defines_retired_class(
        '"""Once there was a class MergeWorker here."""\n'
        '# class MergeWorker(_WipHaltMixin): was its header\n'
        'class SpeculativeMergeWorker:\n    pass\n'
    )


def test_no_copy_of_the_serial_worker_anywhere(tree_pass: TreePass) -> None:
    defines_retired_class = tree_pass.facts(ANALYSIS)
    for root, floor in _MIN_EXPECTED_FILES.items():
        swept = [path for path in defines_retired_class if path.is_relative_to(root)]
        assert len(swept) >= floor, f'swept only {len(swept)} files under {root}'
    copies = [path.as_posix() for path, defines in defines_retired_class.items() if defines]
    assert copies == [], (
        f'a serial MergeWorker is defined again in {copies}. Drive the '
        'production lane instead: make_lane / merge_through_lane in '
        '_merge_lane_fakes.py.'
    )
