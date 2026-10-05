"""measure_python_tree counts, per package, how test files pin implementation detail."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest
import suite_census_pinning as pinning
from source_measures import private_reads_in_tree
from suite_census_fixtures import git_tree

MOD = '''\
REASON = 'the window was measured at 247 s on a loaded host today'
TAG = 'abc def'
_x = 1


def _helper():
    return 1


def public():
    return 2
'''

CONFTEST = '''\
import pkga.mod as mod
import pytest


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(mod, '_x', 2)
'''

TEST_M = '''\
import os
from unittest.mock import patch

import pkga.mod as mod
import pytest


def test_string_patch():
    with patch('pkga.mod._helper'):
        pass


def test_object_patch(monkeypatch):
    monkeypatch.setattr(mod, '_x', 3)


def test_os_patch(monkeypatch):
    monkeypatch.setattr(os, '_exit', print)


def test_public_patch():
    with patch('pkga.mod.public') as public:
        assert public


@patch('pkga.mod._helper')
class TestK:
    def test_one(self, helper):
        assert helper

    def test_two(self, helper):
        assert helper is not None


def test_prose(err):
    assert '247' in str(err)


def test_not_prose(msg):
    assert 'zz9' in msg


def test_short(s):
    assert 'abc' in s.upper()


@pytest.mark.slow
def test_dup_1():
    value = compute(3)
    assert value == 9


def test_dup_2():
    value = compute(3)
    assert value == 9


def test_shape_1():
    assert f(1) == 2


def test_shape_2():
    assert g(5) == 7


def test_shape_3():
    assert h(8) == 11


def test_private_reads(obj):
    assert obj._a
    obj._c = 1


class TestReads:
    def test_self(self):
        assert self._b
'''

TEST_N = '''\
def test_n_dup():
    value = compute(3)
    assert value == 9
'''


@pytest.fixture(scope='module')
def tree(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return git_tree(tmp_path_factory.mktemp('pinning'), {
        'pkga/src/pkga/__init__.py': '',
        'pkga/src/pkga/mod.py': MOD,
        'pkga/tests/conftest.py': CONFTEST,
        'pkga/tests/test_m.py': TEST_M,
        'pkgb/tests/test_n.py': TEST_N,
        'pkgb/tests/fixtures/broken.py': 'def broken(:\n',
    })


@pytest.fixture(scope='module')
def census(tree: Path) -> pinning.PythonPinningCensus:
    return pinning.measure_python_tree(tree)


@pytest.fixture(scope='module')
def pkga(census: pinning.PythonPinningCensus) -> pinning.PackageRow:
    (row,) = [row for row in census.rows if row.package == 'pkga']
    return row


class TestPackageRow:
    def test_files_and_functions(self, pkga):
        assert (pkga.test_files, pkga.test_functions) == (2, 16)

    def test_private_patches(self, pkga):
        assert pkga.tests_with_private_patch == 4
        assert pkga.private_patch_sites_outside_tests == 1
        assert pkga.distinct_private_targets == frozenset({'pkga.mod._helper', 'pkga.mod._x'})

    def test_private_reads_are_the_ratchet_measure(self, pkga):
        ratchet = sum(
            private_reads_in_tree(ast.parse(source))
            for source in (CONFTEST, TEST_M)
        )
        assert pkga.private_reads == ratchet == 2

    def test_prose_constant_assertions(self, pkga):
        assert (pkga.prose_assertion_sites, pkga.tests_with_prose_assertion) == (1, 1)

    def test_duplicates(self, pkga):
        assert (pkga.exact_dup_groups, pkga.exact_dup_redundant) == (1, 1)
        assert (pkga.structural_dup_groups, pkga.structural_dup_redundant) == (2, 3)
        assert pkga.largest_structural_group == (3, 'pkga/tests/test_m.py::test_shape_1')


class TestCensus:
    def test_rows_are_per_package_in_name_order(self, census):
        assert [row.package for row in census.rows] == ['pkga', 'pkgb']

    def test_duplicates_are_never_grouped_across_packages(self, census):
        (pkgb,) = [row for row in census.rows if row.package == 'pkgb']
        assert (pkgb.exact_dup_groups, pkgb.structural_dup_groups) == (0, 0)

    def test_an_unparseable_file_is_listed_and_the_census_incomplete(self, census):
        assert census.unreadable == ('pkgb/tests/fixtures/broken.py',)
        assert census.complete is False

    def test_totals_sum_the_rows_and_union_the_targets(self, census):
        totals = census.totals
        assert totals.test_functions == sum(row.test_functions for row in census.rows)
        assert totals.private_reads == sum(row.private_reads for row in census.rows)
        assert totals.exact_dup_redundant == sum(row.exact_dup_redundant for row in census.rows)
        assert totals.distinct_private_targets == frozenset().union(
            *(row.distinct_private_targets for row in census.rows)
        )


def test_render_has_one_row_per_package_plus_totals_and_names_the_unreadable(census):
    text = pinning.render_markdown(census)
    first_cells = [
        line.strip('|').split('|')[0].strip() for line in text.splitlines() if line.startswith('|')
    ]
    assert first_cells.count('pkga') == 1 and first_cells.count('pkgb') == 1
    assert first_cells.index('pkga') < first_cells.index('pkgb') < first_cells.index(
        census.totals.package
    )
    assert 'pkgb/tests/fixtures/broken.py' in text
