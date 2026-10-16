"""The static escalation-category census, pinned mechanism by mechanism.

Each test writes a small synthetic tree mirroring the real
``<pkg-dir>/src/<pkg_mod>/`` layout under ``tmp_path`` and asserts exactly
which category values the census resolves, and from where.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from _category_census import CensusResult, run_census

from escalation.categories import EscalationCategory

REPO_ROOT = Path(__file__).resolve().parents[2]
_PRODUCER_ROOTS = (
    'escalation/src',
    'orchestrator/src',
    'fused-memory/src',
    'fused-memory/scripts',
    'shared/src',
    'dashboard/src',
    'cockpit/src',
    'sampler/src',
    'scripts',
    'skills',
)


def _write(root: Path, rel: str, source: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source))


def _pairs(result: CensusResult, root: Path) -> set[tuple[str, str]]:
    return {(site.path.relative_to(root).as_posix(), site.value) for site in result.sites}


def _values(result: CensusResult) -> set[str]:
    return {site.value for site in result.sites}


class TestResolution:
    def test_literal(self, tmp_path: Path):
        _write(tmp_path, 'pkg/src/pkg/a.py', """
            from escalation.models import Escalation
            Escalation(category='alpha')
        """)

        assert _pairs(run_census([tmp_path]), tmp_path) == {('pkg/src/pkg/a.py', 'alpha')}

    def test_aliased_escalation_import(self, tmp_path: Path):
        _write(tmp_path, 'pkg/src/pkg/a.py', """
            from escalation.models import Escalation as Esc
            Esc(category='alpha')
        """)

        assert _values(run_census([tmp_path])) == {'alpha'}

    def test_module_constant(self, tmp_path: Path):
        _write(tmp_path, 'pkg/src/pkg/a.py', """
            X = 'beta'
            Escalation(category=X)
        """)
        _write(tmp_path, 'pkg/src/pkg/b.py', """
            Y: str = 'beta'
            Escalation(category=Y)
        """)

        assert _pairs(run_census([tmp_path]), tmp_path) == {
            ('pkg/src/pkg/a.py', 'beta'),
            ('pkg/src/pkg/b.py', 'beta'),
        }

    def test_constant_imported_across_packages(self, tmp_path: Path):
        _write(tmp_path, 'fake-mem/src/fake_mem/consts.py', """
            C = 'gamma'
        """)
        _write(tmp_path, 'pkg/src/pkg/plain.py', """
            from fake_mem.consts import C
            Escalation(category=C)
        """)
        _write(tmp_path, 'pkg/src/pkg/aliased.py', """
            from fake_mem.consts import C as D
            Escalation(category=D)
        """)

        assert _pairs(run_census([tmp_path]), tmp_path) == {
            ('pkg/src/pkg/plain.py', 'gamma'),
            ('pkg/src/pkg/aliased.py', 'gamma'),
        }

    def test_class_constant(self, tmp_path: Path):
        _write(tmp_path, 'pkg/src/pkg/h.py', """
            class H:
                _C = 'delta'

                def file(self):
                    Escalation(category=self._C)
        """)

        assert _values(run_census([tmp_path])) == {'delta'}

    def test_enum_member_resolves_through_the_real_registry(self, tmp_path: Path):
        _write(tmp_path, 'pkg/src/pkg/a.py', """
            from escalation.categories import EscalationCategory
            Escalation(category=EscalationCategory.REVIEW_DUE)
        """)

        assert _values(run_census([tmp_path])) == {'review_due'}

    def test_function_local_assignments(self, tmp_path: Path):
        _write(tmp_path, 'pkg/src/pkg/w.py', """
            def _mark(reason, *, category):
                Escalation(category=category)

            def land(a):
                if a:
                    category = 'iota'
                else:
                    category = 'kappa'
                _mark('r', category=category)
        """)

        assert _values(run_census([tmp_path])) == {'iota', 'kappa'}


class TestWrapperDiscovery:
    def test_keyword_wrapper_and_its_default(self, tmp_path: Path):
        _write(tmp_path, 'pkg/src/pkg/w.py', """
            def _mark(reason, *, category='epsilon'):
                Escalation(category=category)

            _mark('r', category='zeta')
        """)

        result = run_census([tmp_path])

        assert _values(result) == {'epsilon', 'zeta'}
        assert '_mark' in result.wrappers

    def test_positional_method_wrapper_excludes_self(self, tmp_path: Path):
        _write(tmp_path, 'pkg/src/pkg/r.py', """
            class R:
                def _escalate(self, category, run_id):
                    Escalation(category=category)

                def tick(self, rid):
                    self._escalate('eta', rid)
        """)

        result = run_census([tmp_path])

        assert _values(result) == {'eta'}
        assert '_escalate' in result.wrappers

    def test_transitive_wrapper(self, tmp_path: Path):
        _write(tmp_path, 'pkg/src/pkg/w.py', """
            def _mark(reason, *, category='epsilon'):
                Escalation(category=category)

            def outer(category):
                _mark('r', category=category)

            outer(category='theta')
        """)

        result = run_census([tmp_path])

        assert 'outer' in result.wrappers
        assert _values(result) == {'epsilon', 'theta'}

    def test_reader_seeds(self, tmp_path: Path):
        _write(tmp_path, 'pkg/src/pkg/q.py', """
            def check(q, t, sha):
                q.has_open_l1(t, category='lambda')
                q.find_terminal_by_citation(t, 'mu', sha)
        """)

        assert _values(run_census([tmp_path])) == {'lambda', 'mu'}


class TestWhatIsNotASite:
    def test_dynamic_values_are_skipped(self, tmp_path: Path):
        _write(tmp_path, 'pkg/src/pkg/d.py', """
            def f(info, rec):
                Escalation(category=info['category'])
                Escalation(category=rec.get('category') or 'x')
                Escalation(category=some_call())
        """)

        result = run_census([tmp_path])

        assert result.sites == ()

    def test_non_escalation_calls_are_ignored(self, tmp_path: Path):
        _write(tmp_path, 'pkg/src/pkg/n.py', """
            add_memory(category='procedural_knowledge')
            VerifyResult(category='test_failure')
        """)

        assert run_census([tmp_path]).sites == ()

    def test_tests_are_not_walked_and_unparsable_files_are_reported(self, tmp_path: Path):
        _write(tmp_path, 'pkg/tests/helper.py', "Escalation(category='in_tests_dir')\n")
        _write(tmp_path, 'pkg/src/pkg/test_thing.py', "Escalation(category='in_test_file')\n")
        _write(tmp_path, 'pkg/src/pkg/broken.py', 'def (:\n')
        _write(tmp_path, 'pkg/src/pkg/ok.py', "Escalation(category='ok')\n")

        result = run_census([tmp_path])

        assert _values(result) == {'ok'}
        assert result.files_parsed == 1
        assert [Path(p).name for p in result.parse_failures] == ['broken.py']


def _existing_roots() -> list[Path]:
    return [REPO_ROOT / rel for rel in _PRODUCER_ROOTS if (REPO_ROOT / rel).is_dir()]


@pytest.fixture(scope='module')
def repository_census() -> CensusResult:
    return run_census(_existing_roots())


@pytest.mark.xdist_group('category_census')
class TestRepositoryCensus:
    def test_every_filed_category_is_registered(self, repository_census: CensusResult):
        unregistered = [s for s in repository_census.sites if s.value not in EscalationCategory]

        assert unregistered == [], (
            'unregistered escalation categories:\n'
            + '\n'.join(
                f'  {s.path.relative_to(REPO_ROOT)}:{s.line} {s.value}' for s in unregistered
            )
            + '\nregister it in escalation/src/escalation/categories.py::EscalationCategory, '
            'or, if this call is not an escalation filer, the census over-matched a wrapper '
            'name — see _category_census._discover_wrappers'
        )

    def test_census_reaches_the_tree(self, repository_census: CensusResult):
        assert repository_census.parse_failures == ()
        for root in _existing_roots():
            assert any(p.is_relative_to(root) for p in repository_census.parsed_paths), root
        site_paths = [s.path for s in repository_census.sites]
        assert any(p.is_relative_to(REPO_ROOT / 'orchestrator/src') for p in site_paths)
        assert any(p.is_relative_to(REPO_ROOT / 'fused-memory/src') for p in site_paths)
        assert repository_census.wrappers
