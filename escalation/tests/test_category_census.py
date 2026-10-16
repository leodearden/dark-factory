"""The static escalation-category census, pinned mechanism by mechanism.

Each test writes a small synthetic tree mirroring the real
``<pkg-dir>/src/<pkg_mod>/`` layout under ``tmp_path`` and asserts exactly
which category values the census resolves, and from where.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from _category_census import CensusResult, run_census


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
