"""``before_done.cwd`` is a string path or absent, at run time too (task 6557).

The fused-memory submit guard enforces this rule when a task is filed, but a
stored record can still reach the run-time executor through ``update_task``,
whose ``BeforeDone`` write boundary only warns. ``resolve_before_done_paths``
therefore refuses any other cwd itself, with an error naming the field and the
offending type rather than a bare ``TypeError`` from the path join.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from shared.before_done_paths import (
    BeforeDonePathsRefused,
    CwdRefused,
    resolve_before_done_paths,
)

ROOT = Path('/srv/project')
NON_STRING_CWDS = pytest.mark.parametrize(
    'cwd',
    [5, True, ['scripts'], {'dir': 'x'}, 0, False, []],
    ids=['int', 'true', 'list', 'dict', 'zero', 'false', 'empty-list'],
)


def _never_consulted() -> Path | None:
    raise AssertionError('the factory root resolver must not run here')


@NON_STRING_CWDS
def test_resolver_refuses_non_string_cwd_naming_field_and_type(cwd: object) -> None:
    with pytest.raises(CwdRefused) as excinfo:
        resolve_before_done_paths({'script': 'deploy.sh', 'cwd': cwd}, ROOT, _never_consulted)

    assert 'before_done.cwd' in str(excinfo.value)
    assert repr(type(cwd).__name__) in str(excinfo.value)


@NON_STRING_CWDS
def test_cwd_refusal_is_a_before_done_paths_refused_value_error(cwd: object) -> None:
    with pytest.raises(CwdRefused) as excinfo:
        resolve_before_done_paths({'script': 'deploy.sh', 'cwd': cwd}, ROOT, _never_consulted)

    assert isinstance(excinfo.value, BeforeDonePathsRefused)
    assert isinstance(excinfo.value, ValueError)


@pytest.mark.parametrize(
    ('cwd', 'resolved'),
    [
        (None, ROOT),
        ('', ROOT),
        ('sub/dir', ROOT / 'sub/dir'),
        ('/srv/elsewhere', Path('/srv/elsewhere')),
    ],
)
def test_string_or_absent_cwd_resolves(cwd: str | None, resolved: Path) -> None:
    paths = resolve_before_done_paths({'script': 'deploy.sh', 'cwd': cwd}, ROOT, _never_consulted)

    assert paths.cwd == resolved


def test_missing_cwd_key_resolves_to_project_root() -> None:
    paths = resolve_before_done_paths({'script': 'deploy.sh'}, ROOT, _never_consulted)

    assert paths.cwd == ROOT
