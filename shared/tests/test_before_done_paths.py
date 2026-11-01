"""Where a deterministic task's before_done script runs (task 6464).

Pure-function tests for ``resolve_before_done_paths``: a relative script and a
relative cwd resolve against the configured project_root, the same root the
fused-memory submit guard validates the script under. ``resolve_script_base``
picks the script's base from ``before_done.script_root`` (task 6386).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from shared.before_done_paths import (
    BeforeDonePaths,
    ScriptRootRefused,
    resolve_before_done_paths,
    resolve_script_base,
)

ROOT = Path('/srv/project')
FACTORY = Path('/srv/factory')
RELATIVE_SCRIPT = 'data/ops/deploy.sh'


class TestResolveBeforeDonePaths:
    def test_relative_script_without_cwd_resolves_under_project_root(self) -> None:
        paths = resolve_before_done_paths({'script': RELATIVE_SCRIPT}, ROOT)

        assert paths == BeforeDonePaths(script=ROOT / RELATIVE_SCRIPT, cwd=ROOT)

    @pytest.mark.parametrize('empty_cwd', ['', None])
    def test_empty_cwd_defaults_to_project_root(self, empty_cwd: str | None) -> None:
        paths = resolve_before_done_paths(
            {'script': RELATIVE_SCRIPT, 'cwd': empty_cwd}, ROOT,
        )

        assert paths.cwd == ROOT

    def test_absolute_script_is_kept_unchanged(self) -> None:
        paths = resolve_before_done_paths({'script': '/usr/local/bin/x.sh'}, ROOT)

        assert paths.script == Path('/usr/local/bin/x.sh')

    def test_absolute_cwd_is_kept_and_relative_script_still_resolves_under_root(
        self,
    ) -> None:
        paths = resolve_before_done_paths(
            {'script': RELATIVE_SCRIPT, 'cwd': '/srv/elsewhere'}, ROOT,
        )

        assert paths == BeforeDonePaths(
            script=ROOT / RELATIVE_SCRIPT, cwd=Path('/srv/elsewhere'),
        )

    def test_relative_cwd_resolves_under_project_root(self) -> None:
        paths = resolve_before_done_paths(
            {'script': RELATIVE_SCRIPT, 'cwd': 'sub/dir'}, ROOT,
        )

        assert paths.cwd == ROOT / 'sub/dir'

    def test_relative_project_root_is_rejected(self) -> None:
        with pytest.raises(ValueError, match='rel/root'):
            resolve_before_done_paths({'script': RELATIVE_SCRIPT}, Path('rel/root'))

    def test_script_base_replaces_project_root_for_the_script_only(self) -> None:
        paths = resolve_before_done_paths(
            {'script': RELATIVE_SCRIPT}, ROOT, script_base=FACTORY,
        )

        assert paths == BeforeDonePaths(script=FACTORY / RELATIVE_SCRIPT, cwd=ROOT)

    def test_script_base_does_not_move_a_relative_cwd(self) -> None:
        paths = resolve_before_done_paths(
            {'script': RELATIVE_SCRIPT, 'cwd': 'sub/dir'}, ROOT, script_base=FACTORY,
        )

        assert paths == BeforeDonePaths(
            script=FACTORY / RELATIVE_SCRIPT, cwd=ROOT / 'sub/dir',
        )

    def test_relative_script_base_is_rejected(self) -> None:
        with pytest.raises(ValueError, match='rel/factory'):
            resolve_before_done_paths(
                {'script': RELATIVE_SCRIPT}, ROOT, script_base=Path('rel/factory'),
            )


def _never_consulted() -> Path | None:
    raise AssertionError('the factory root resolver must not run here')


class TestResolveScriptBase:
    @pytest.mark.parametrize(
        'root_fields', [{'script_root': 'project'}, {}], ids=['explicit', 'default'],
    )
    @pytest.mark.parametrize('kind', ['deploy', 'predicate'])
    def test_project_rooted_script_resolves_under_project_root(
        self, root_fields: dict[str, str], kind: str,
    ) -> None:
        before_done = {'script': RELATIVE_SCRIPT, 'kind': kind, **root_fields}

        assert resolve_script_base(before_done, ROOT, _never_consulted) == ROOT

    def test_factory_predicate_resolves_under_the_factory_root(self) -> None:
        before_done = {'script': RELATIVE_SCRIPT, 'kind': 'predicate', 'script_root': 'factory'}

        assert resolve_script_base(before_done, ROOT, lambda: FACTORY) == FACTORY

    def test_unresolvable_factory_root_is_none(self) -> None:
        before_done = {'script': RELATIVE_SCRIPT, 'kind': 'predicate', 'script_root': 'factory'}

        assert resolve_script_base(before_done, ROOT, lambda: None) is None

    @pytest.mark.parametrize(
        'kind_fields', [{'kind': 'deploy'}, {}], ids=['explicit-deploy', 'default-deploy'],
    )
    def test_factory_deploy_refused_before_the_resolver_runs(
        self, kind_fields: dict[str, str],
    ) -> None:
        before_done = {'script': RELATIVE_SCRIPT, 'script_root': 'factory', **kind_fields}

        with pytest.raises(ScriptRootRefused, match="requires kind='predicate'"):
            resolve_script_base(before_done, ROOT, _never_consulted)

    def test_unknown_script_root_refused_naming_the_valid_values(self) -> None:
        before_done = {'script': RELATIVE_SCRIPT, 'kind': 'predicate', 'script_root': 'elsewhere'}

        with pytest.raises(ScriptRootRefused) as excinfo:
            resolve_script_base(before_done, ROOT, _never_consulted)

        assert 'elsewhere' in str(excinfo.value)
        assert "'project'" in str(excinfo.value)
        assert "'factory'" in str(excinfo.value)


class TestBeforeDonePathsInvariant:
    @pytest.mark.parametrize(
        ('field', 'script', 'cwd'),
        [
            ('script', Path('relative/x.sh'), ROOT),
            ('cwd', ROOT / 'x.sh', Path('relative/dir')),
        ],
    )
    def test_relative_field_is_rejected_naming_field_and_value(
        self, field: str, script: Path, cwd: Path,
    ) -> None:
        offending = script if field == 'script' else cwd

        with pytest.raises(ValueError) as excinfo:
            BeforeDonePaths(script=script, cwd=cwd)

        assert field in str(excinfo.value)
        assert str(offending) in str(excinfo.value)
