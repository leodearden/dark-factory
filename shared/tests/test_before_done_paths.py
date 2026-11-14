"""Where a deterministic task's before_done script runs (tasks 6464, 6386, 9116).

Pure-function tests for ``resolve_before_done_paths``, the single entry point
both the run-time executor and the fused-memory submit guard call: a relative
script and a relative cwd resolve against the configured project_root, and
``before_done.script_root`` picks the script's base. Every refusal it raises
is a ``BeforeDonePathsRefused``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from shared.before_done_paths import (
    BeforeDonePaths,
    BeforeDonePathsRefused,
    CwdRefused,
    FactoryRootUnresolved,
    ScriptRootRefused,
    resolve_before_done_paths,
)

ROOT = Path('/srv/project')
FACTORY = Path('/srv/factory')
RELATIVE_SCRIPT = 'data/ops/deploy.sh'
FACTORY_PREDICATE = {'script': RELATIVE_SCRIPT, 'kind': 'predicate', 'script_root': 'factory'}


def _never_consulted() -> Path | None:
    raise AssertionError('the factory root resolver must not run here')


class TestResolveBeforeDonePaths:
    def test_relative_script_without_cwd_resolves_under_project_root(self) -> None:
        paths = resolve_before_done_paths({'script': RELATIVE_SCRIPT}, ROOT, _never_consulted)

        assert paths == BeforeDonePaths(
            script=ROOT / RELATIVE_SCRIPT, script_base=ROOT, cwd=ROOT,
        )

    @pytest.mark.parametrize('empty_cwd', ['', None])
    def test_empty_cwd_defaults_to_project_root(self, empty_cwd: str | None) -> None:
        paths = resolve_before_done_paths(
            {'script': RELATIVE_SCRIPT, 'cwd': empty_cwd}, ROOT, _never_consulted,
        )

        assert paths.cwd == ROOT

    def test_absolute_script_is_kept_unchanged(self) -> None:
        paths = resolve_before_done_paths(
            {'script': '/usr/local/bin/x.sh'}, ROOT, _never_consulted,
        )

        assert paths.script == Path('/usr/local/bin/x.sh')

    def test_absolute_cwd_is_kept_and_relative_script_still_resolves_under_root(
        self,
    ) -> None:
        paths = resolve_before_done_paths(
            {'script': RELATIVE_SCRIPT, 'cwd': '/srv/elsewhere'}, ROOT, _never_consulted,
        )

        assert paths == BeforeDonePaths(
            script=ROOT / RELATIVE_SCRIPT, script_base=ROOT, cwd=Path('/srv/elsewhere'),
        )

    def test_relative_cwd_resolves_under_project_root(self) -> None:
        paths = resolve_before_done_paths(
            {'script': RELATIVE_SCRIPT, 'cwd': 'sub/dir'}, ROOT, _never_consulted,
        )

        assert paths.cwd == ROOT / 'sub/dir'

    def test_relative_project_root_is_rejected(self) -> None:
        with pytest.raises(ValueError, match='rel/root'):
            resolve_before_done_paths(
                {'script': RELATIVE_SCRIPT}, Path('rel/root'), _never_consulted,
            )

    def test_factory_predicate_script_resolves_under_the_factory_root(self) -> None:
        paths = resolve_before_done_paths(FACTORY_PREDICATE, ROOT, lambda: FACTORY)

        assert paths == BeforeDonePaths(
            script=FACTORY / RELATIVE_SCRIPT, script_base=FACTORY, cwd=ROOT,
        )

    def test_factory_root_does_not_move_a_relative_cwd(self) -> None:
        paths = resolve_before_done_paths(
            {**FACTORY_PREDICATE, 'cwd': 'sub/dir'}, ROOT, lambda: FACTORY,
        )

        assert paths == BeforeDonePaths(
            script=FACTORY / RELATIVE_SCRIPT, script_base=FACTORY, cwd=ROOT / 'sub/dir',
        )

    def test_relative_script_base_is_rejected(self) -> None:
        with pytest.raises(ValueError, match='rel/factory'):
            resolve_before_done_paths(FACTORY_PREDICATE, ROOT, lambda: Path('rel/factory'))


class TestScriptBaseSelection:
    @pytest.mark.parametrize(
        'root_fields', [{'script_root': 'project'}, {}], ids=['explicit', 'default'],
    )
    @pytest.mark.parametrize('kind', ['deploy', 'predicate'])
    def test_project_rooted_script_resolves_under_project_root(
        self, root_fields: dict[str, str], kind: str,
    ) -> None:
        before_done = {'script': RELATIVE_SCRIPT, 'kind': kind, **root_fields}

        paths = resolve_before_done_paths(before_done, ROOT, _never_consulted)

        assert paths.script_base == ROOT

    @pytest.mark.parametrize(
        'kind_fields', [{'kind': 'deploy'}, {}], ids=['explicit-deploy', 'default-deploy'],
    )
    def test_factory_deploy_refused_before_the_resolver_runs(
        self, kind_fields: dict[str, str],
    ) -> None:
        before_done = {'script': RELATIVE_SCRIPT, 'script_root': 'factory', **kind_fields}

        with pytest.raises(ScriptRootRefused, match="requires kind='predicate'"):
            resolve_before_done_paths(before_done, ROOT, _never_consulted)

    def test_unknown_script_root_refused_naming_the_valid_values(self) -> None:
        before_done = {'script': RELATIVE_SCRIPT, 'kind': 'predicate', 'script_root': 'elsewhere'}

        with pytest.raises(ScriptRootRefused) as excinfo:
            resolve_before_done_paths(before_done, ROOT, _never_consulted)

        assert 'elsewhere' in str(excinfo.value)
        assert "'project'" in str(excinfo.value)
        assert "'factory'" in str(excinfo.value)

    def test_unresolvable_factory_root_refused_naming_the_script(self) -> None:
        with pytest.raises(FactoryRootUnresolved) as excinfo:
            resolve_before_done_paths(FACTORY_PREDICATE, ROOT, lambda: None)

        assert "script_root='factory'" in str(excinfo.value)
        assert 'dark-factory root could not be resolved' in str(excinfo.value)
        assert repr(RELATIVE_SCRIPT) in str(excinfo.value)


class TestRefusalFamily:
    @pytest.mark.parametrize(
        ('before_done', 'factory_root_resolver', 'refusal'),
        [
            ({'script': RELATIVE_SCRIPT, 'cwd': 5}, _never_consulted, CwdRefused),
            (
                {'script': RELATIVE_SCRIPT, 'kind': 'predicate', 'script_root': 'elsewhere'},
                _never_consulted,
                ScriptRootRefused,
            ),
            (FACTORY_PREDICATE, lambda: None, FactoryRootUnresolved),
        ],
        ids=['non-string-cwd', 'unknown-script-root', 'unresolvable-factory-root'],
    )
    def test_every_refusal_is_a_before_done_paths_refused_value_error(
        self, before_done: dict, factory_root_resolver, refusal: type[Exception],
    ) -> None:
        with pytest.raises(BeforeDonePathsRefused) as excinfo:
            resolve_before_done_paths(before_done, ROOT, factory_root_resolver)

        assert isinstance(excinfo.value, refusal)
        assert isinstance(excinfo.value, ValueError)

    def test_cwd_is_refused_before_script_root(self) -> None:
        before_done = {'script': RELATIVE_SCRIPT, 'cwd': 5, 'script_root': 'elsewhere'}

        with pytest.raises(CwdRefused):
            resolve_before_done_paths(before_done, ROOT, _never_consulted)


class TestBeforeDonePathsInvariant:
    @pytest.mark.parametrize(
        ('field', 'script', 'script_base', 'cwd'),
        [
            ('script', Path('relative/x.sh'), ROOT, ROOT),
            ('script_base', ROOT / 'x.sh', Path('relative/base'), ROOT),
            ('cwd', ROOT / 'x.sh', ROOT, Path('relative/dir')),
        ],
    )
    def test_relative_field_is_rejected_naming_field_and_value(
        self, field: str, script: Path, script_base: Path, cwd: Path,
    ) -> None:
        offending = {'script': script, 'script_base': script_base, 'cwd': cwd}[field]

        with pytest.raises(ValueError) as excinfo:
            BeforeDonePaths(script=script, script_base=script_base, cwd=cwd)

        assert f'BeforeDonePaths.{field} ' in str(excinfo.value)
        assert str(offending) in str(excinfo.value)
