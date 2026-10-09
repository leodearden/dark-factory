"""Where a deterministic task's before_done script runs (task 6464).

Pure-function tests for ``resolve_before_done_paths``: a relative script and a
relative cwd resolve against the configured project_root, the same root the
fused-memory submit guard validates the script under.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator.before_done_paths import BeforeDonePaths, resolve_before_done_paths

ROOT = Path('/srv/project')
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
