"""Contract of ``shared.testing_landing_history.LandingRepo``: git runs only inside a ceiling at its root."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from shared.testing_landing_history import LandingRepo

_CEILING = 'GIT_CEILING_DIRECTORIES'


def _env(ceiling: str | None) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if key != _CEILING}
    if ceiling is not None:
        env[_CEILING] = ceiling
    return env


@pytest.mark.parametrize('ceiling', [None, '', '/elsewhere'])
def test_an_env_without_a_ceiling_at_the_root_is_refused_before_any_git_call(
    tmp_path: Path, ceiling: str | None,
) -> None:
    root = tmp_path / 'repo'

    with pytest.raises(ValueError, match=_CEILING):
        LandingRepo(root, env=_env(ceiling))

    assert not (root / '.git').exists()


def test_a_ceiling_among_others_builds_the_repo_on_its_branch(tmp_path: Path) -> None:
    root = tmp_path / 'repo'

    repo = LandingRepo(root, env=_env(f'/elsewhere:{tmp_path.resolve()}'), branch='trunk')

    merge_sha = repo.merge('task/1')

    assert (root / '.git' / 'refs' / 'heads' / 'trunk').read_text().strip() == merge_sha
