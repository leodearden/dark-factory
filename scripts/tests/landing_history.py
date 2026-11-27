"""This test root's contained ``LandingRepo``.

A plain module rather than a conftest fixture, the same convention as
``committed_git_repo``: conftest appends this directory to ``sys.path``, so
consumers import it by bare name. The repo and the event writers live in
``shared.testing_landing_history``; what this root adds is the git ceiling
(incident esc-3072-3), built with ``df_pytest_isolation.git_ceiling_value`` so
an existing ceiling, such as the basetemp one, is kept.
"""
from __future__ import annotations

import os
from pathlib import Path

from shared.testing_landing_history import LandingRepo

from df_pytest_isolation import git_ceiling_value

_CEILING = 'GIT_CEILING_DIRECTORIES'


def landing_repo(root: Path, *, branch: str = 'main') -> LandingRepo:
    """A LandingRepo at *root* on *branch* whose git calls cannot walk above *root*."""
    env = dict(os.environ)
    env[_CEILING] = git_ceiling_value(root.resolve().parent, existing=env.get(_CEILING))
    return LandingRepo(root, env=env, branch=branch)
