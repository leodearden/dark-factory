"""The repo-wide guard-family census, instantiated for scripts/tests.

The detector, the judgement, its failure messages and their rationale live in
shared/src/shared/testing_repo_wide_tests.py.  This module supplies only this
package's inputs: its tests root and its anti-vacuity floors.  It is itself
repo-wide, so it carries the family mark it demands.
"""

from __future__ import annotations

import functools
from pathlib import Path

import pytest
from shared.testing_repo_wide_tests import RepoWideFloors, RepoWideVerdict, judge_repo_wide_family
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = [
    pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT),
    pytest.mark.xdist_group('test_repo_wide_guard_family'),
]

_TESTS_DIR = Path(__file__).resolve().parent

#: Set well below the 217 files and 126 repo-wide modules measured on
#: 2026-11-11, so a relocation of a few modules elsewhere cannot trip it.
_FLOORS = RepoWideFloors(test_files=130, repo_wide_modules=63)


@functools.cache
def _verdict() -> RepoWideVerdict:
    return judge_repo_wide_family(_TESTS_DIR, _FLOORS)


def test_every_repo_wide_test_module_is_a_guard_family_member() -> None:
    failure = _verdict().offender_failure
    assert failure is None, failure


def test_the_repo_wide_census_is_not_vacuous() -> None:
    failure = _verdict().census_failure
    assert failure is None, failure
