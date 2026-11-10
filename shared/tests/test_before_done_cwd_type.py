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

from shared.before_done_paths import CwdRefused, before_done_cwd, resolve_before_done_paths

ROOT = Path('/srv/project')
NON_STRING_CWDS = pytest.mark.parametrize(
    'cwd',
    [5, True, ['scripts'], {'dir': 'x'}, 0, False, []],
    ids=['int', 'true', 'list', 'dict', 'zero', 'false', 'empty-list'],
)


@NON_STRING_CWDS
def test_resolver_refuses_non_string_cwd_naming_field_and_type(cwd: object) -> None:
    with pytest.raises(CwdRefused) as excinfo:
        resolve_before_done_paths({'script': 'deploy.sh', 'cwd': cwd}, ROOT)

    assert 'before_done.cwd' in str(excinfo.value)
    assert repr(type(cwd).__name__) in str(excinfo.value)


@NON_STRING_CWDS
def test_cwd_refusal_is_a_value_error(cwd: object) -> None:
    with pytest.raises(ValueError):
        before_done_cwd({'cwd': cwd})


@pytest.mark.parametrize('cwd', [None, '', 'sub/dir', '/srv/elsewhere'])
def test_string_or_absent_cwd_is_returned_unchanged(cwd: str | None) -> None:
    assert before_done_cwd({'cwd': cwd}) == cwd


def test_missing_cwd_key_is_none() -> None:
    assert before_done_cwd({'script': 'deploy.sh'}) is None
