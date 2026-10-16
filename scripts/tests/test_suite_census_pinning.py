"""measure_python_tree counts, per package, how test files pin implementation detail."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest
import suite_census_pinning as pinning
from source_measures import MetricsError, private_reads_in_tree
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
from unittest.mock import patch


def test_n_dup():
    value = compute(3)
    assert value == 9


def test_namespace_package_patch():
    with patch('nsb.state._s'):
        pass
'''

SCRIPTS_TEST = '''\
from unittest.mock import patch
from legibility import ledger


def test_dotted_legibility_patch():
    with patch('legibility.ledger._x'):
        pass


def test_object_legibility_patch(monkeypatch):
    monkeypatch.setattr(ledger, '_x', 2)


def test_bare_legibility_name_patch():
    with patch('ledger._x'):
        pass


def test_scripts_root_patch():
    with patch('tool._y'):
        pass


def test_unimportable_script_name_patch():
    with patch('wait-for-port._z'):
        pass


def test_prose_through_a_dotted_import(line):
    assert 'one sighting per' in line
'''


@pytest.fixture(scope='module')
def tree(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = git_tree(tmp_path_factory.mktemp('pinning'), {
        'pyproject.toml': '[tool.uv.workspace]\nmembers = ["pkga", "pkgb"]\n',
        'pkga/src/pkga/__init__.py': '',
        'pkga/src/pkga/mod.py': MOD,
        'pkga/tests/conftest.py': CONFTEST,
        'pkga/tests/test_m.py': TEST_M,
        'pkgb/src/nsb/state.py': '_s = 1\n',
        'pkgb/tests/test_n.py': TEST_N,
        'pkgb/tests/fixtures/broken.py': 'def broken(:\n',
        'scripts/tool.py': '_y = 1\n',
        'scripts/wait-for-port.py': '_z = 1\n',
        'scripts/legibility/ledger.py': "_x = 1\nNOTE = 'a ledger line records one sighting per confusion'\n",
        'scripts/tests/test_scripts.py': SCRIPTS_TEST,
        'tests/test_top.py': 'def test_top():\n    pass\n',
        'hooks/tests/test_h.py': 'def test_h():\n    pass\n',
    })
    (root / 'pkga/tests/test_untracked.py').write_text('def test_u():\n    pass\n')
    return root


@pytest.fixture(scope='module')
def census(tree: Path) -> pinning.PythonPinningCensus:
    return pinning.measure_python_tree(tree)


@pytest.fixture(scope='module')
def pkga(census: pinning.PythonPinningCensus) -> pinning.PackageRow:
    (row,) = [row for row in census.rows if row.package == 'pkga']
    return row


@pytest.fixture(scope='module')
def pkgb(census: pinning.PythonPinningCensus) -> pinning.PackageRow:
    (row,) = [row for row in census.rows if row.package == 'pkgb']
    return row


@pytest.fixture(scope='module')
def scripts(census: pinning.PythonPinningCensus) -> pinning.PackageRow:
    (row,) = [row for row in census.rows if row.package == 'scripts']
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
        assert [row.package for row in census.rows] == ['pkga', 'pkgb', 'scripts', 'tests']

    def test_each_member_row_counts_only_that_members_tracked_test_files(self, census):
        assert {row.package: row.test_files for row in census.rows} == {
            'pkga': 2, 'pkgb': 1, 'scripts': 1, 'tests': 1,
        }

    def test_duplicates_are_never_grouped_across_packages(self, pkgb):
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


class TestFirstPartyIsTheDomain:
    def test_scripts_is_the_import_root_so_legibility_modules_are_dotted(self, scripts):
        assert scripts.distinct_private_targets == frozenset({'legibility.ledger._x', 'tool._y'})
        assert scripts.tests_with_private_patch == 3

    def test_a_bare_legibility_module_name_is_the_disclosed_undercount(self, scripts):
        assert 'ledger._x' not in scripts.distinct_private_targets

    def test_a_member_namespace_package_without_init_is_first_party(self, pkgb):
        assert pkgb.distinct_private_targets == frozenset({'nsb.state._s'})
        assert pkgb.tests_with_private_patch == 1

    def test_a_prose_constant_counts_through_a_dotted_legibility_import(self, scripts):
        assert (scripts.prose_assertion_sites, scripts.tests_with_prose_assertion) == (1, 1)

    def test_a_script_whose_name_is_not_an_identifier_is_never_first_party(self, scripts):
        assert 'wait-for-port._z' not in scripts.distinct_private_targets

    def test_a_tree_whose_pyproject_declares_no_members_is_refused(self, tmp_path):
        tree = git_tree(tmp_path, {
            'pyproject.toml': '[project]\nname = "x"\n',
            'pkga/tests/test_m.py': 'def test_m():\n    pass\n',
        })
        with pytest.raises(MetricsError, match=r'\[tool\.uv\.workspace\]\.members'):
            pinning.measure_python_tree(tree)


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
