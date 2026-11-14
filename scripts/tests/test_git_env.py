"""Tests for scripts/git_env.py — the environment a script's git subprocess runs
with, minus ambient redirection (incident 2026-08-31).

``import git_env`` resolves via scripts/tests/conftest.py's sys.path insertion.

The poisoned mapping is DERIVED from ``df_pytest_isolation``'s own redirect
list, so a widening there poisons every test here too. The only hand-written
mapping is the small set of names the scrub must leave alone.
"""
from __future__ import annotations

import git_env

from df_pytest_isolation import (
    _GIT_REDIRECT_ENV,
    _GIT_REDIRECT_ENV_PREFIXES,
    git_redirect_env,
)

POISONED_GIT_ENV = {name: "/decoy" for name in _GIT_REDIRECT_ENV}
POISONED_GIT_ENV.update({f"{prefix}0": "/decoy" for prefix in _GIT_REDIRECT_ENV_PREFIXES})

# Names the scrub must LEAVE ALONE. The ceiling is a DIFFERENT defence's own
# mechanism (df_pytest_isolation._df_git_ceiling_at_basetemp, incident
# esc-3072-3); the identity vars change what a commit SAYS, never where it lands.
PRESERVED_GIT_ENV = {
    "GIT_CEILING_DIRECTORIES": "/tmp/pytest-basetemp",
    "GIT_AUTHOR_NAME": "Scrub Test",
    "GIT_COMMITTER_EMAIL": "test@example.com",
}


def test_scrubbed_git_env_drops_every_redirecting_name():
    """No redirecting name — exact or indexed-prefix — survives the scrub."""
    scrubbed = git_env.scrubbed_git_env(dict(POISONED_GIT_ENV))

    assert [name for name in POISONED_GIT_ENV if name in scrubbed] == []


def test_scrubbed_git_env_forces_lc_all_c_over_ambient_locale():
    """LC_ALL=C is FORCED (porcelain stability), overriding an ambient locale."""
    scrubbed = git_env.scrubbed_git_env({"LC_ALL": "en_US.UTF-8", "GIT_DIR": "/decoy/.git"})

    assert scrubbed["LC_ALL"] == "C"


def test_scrubbed_git_env_preserves_unrelated_vars():
    """A targeted removal, not a hermetic rebuild: the unit's PATH/HOME survive."""
    scrubbed = git_env.scrubbed_git_env(
        {"PATH": "/usr/bin:/bin", "HOME": "/home/leo", "GIT_DIR": "/decoy/.git"}
    )

    assert scrubbed["PATH"] == "/usr/bin:/bin"
    assert scrubbed["HOME"] == "/home/leo"


def test_scrubbed_git_env_preserves_ceiling_and_identity_vars():
    """GIT_CEILING_DIRECTORIES and the identity vars are DELIBERATELY kept."""
    scrubbed = git_env.scrubbed_git_env({**POISONED_GIT_ENV, **PRESERVED_GIT_ENV})

    for name, value in PRESERVED_GIT_ENV.items():
        assert scrubbed[name] == value, f"{name} must survive the scrub"


def test_scrubbed_git_env_does_not_mutate_input():
    """Pure: the caller's mapping (in production, ``os.environ``) is untouched."""
    environ = dict(POISONED_GIT_ENV)
    before = dict(environ)

    git_env.scrubbed_git_env(environ)

    assert environ == before


def test_scrubbed_git_env_matches_shared_redirect_classifier():
    """DRIFT PIN. git_env is STDLIB-ONLY by design (the systemd wrappers run
    plain `python3`, and df_pytest_isolation imports pytest), so the redirecting
    names are DUPLICATED rather than imported. This test pins the copy against
    the shared definition in BOTH directions: a name added to
    df_pytest_isolation's list that git_env does not scrub, or a name git_env
    scrubs that the shared classifier does not consider redirecting, fails here."""
    assert git_env.GIT_REDIRECT_ENV == _GIT_REDIRECT_ENV
    assert git_env.GIT_REDIRECT_ENV_PREFIXES == _GIT_REDIRECT_ENV_PREFIXES

    redirecting = git_redirect_env(POISONED_GIT_ENV)
    assert redirecting, "the shared classifier must classify the derived mapping"

    scrubbed = git_env.scrubbed_git_env(POISONED_GIT_ENV)
    assert [name for name in redirecting if name in scrubbed] == []
