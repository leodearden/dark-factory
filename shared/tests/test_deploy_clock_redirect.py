"""Regression test: the deploy-clock redirect (task 3797) is now suite-wide.

``shared/tests`` never carried any deploy-clock MITIGATION of its own -- only
the autouse ``_df_deploy_clocks_unwritten`` GUARD (``df_pytest_isolation.py``),
which watches the live-checkout deploy-clock paths and fails the run if either
one changes. Before task 5299 the guard was wired into all nine conftests that
import ``df_pytest_isolation``, but the mitigation -- redirecting
``ORCH_FLEET_DEPLOY_CLOCK`` and ``FM_DEPLOY_CLOCK`` at a tmp file so a spawner
never reaches the live path in the first place -- existed only in
``scripts/tests/conftest.py``. Task 4892 (esc-4892-7) hit exactly that gap: a
genuine external fleet redeploy fired while this rootdir's suite happened to
be running, and the guard -- correctly watching the live path, because nothing
in this directory pointed it elsewhere -- failed an unrelated task's
post-merge verify.

Task 5299 folded the redirect into ``_df_deploy_clocks_unwritten`` itself
(rather than adding a second fixture every conftest would need to import
separately), so every conftest that imports it -- this one included --
redirects both protected clock env vars for the whole session before any test
runs. This test pins that ``shared/tests`` specifically inherited the fix, so
a future refactor that re-narrows the redirect back to a single rootdir --
the exact regression this task exists to prevent -- fails HERE too, not only
in ``scripts/tests`` where the mitigation already worked.

Deliberately does not re-test the guard itself (``_df_deploy_clocks_unwritten``
snapshot/compare, or that it fires on a real falsification) -- that behaviour
already has its own coverage in ``tests/scripts/test_deploy_clock_isolation.py``
and is unchanged by this task. This file's only job is to prove the redirect
reaches a rootdir other than ``scripts/tests``.
"""
import os
from pathlib import Path

from df_pytest_isolation import (
    PROTECTED_DEPLOY_CLOCK_ENV_VARS,
    PROTECTED_DEPLOY_CLOCK_RELPATHS,
    deploy_clock_guard_roots,
)

# Same anchor _df_deploy_clocks_unwritten itself uses (df_pytest_isolation.py
# sits at the repo root): a test file two directories below the repo root
# (shared/tests/<this file>) reaches it the same way shared/tests/conftest.py's
# own REPO_ROOT does.
_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_every_protected_clock_env_var_is_redirected_away_from_the_live_checkout():
    """Both protected clock env vars must avoid every live checkout path.

    Reads ``os.environ`` directly rather than through a fixture: this test's
    whole point is that the redirect is now unconditional for every rootdir
    that imports ``df_pytest_isolation``, so it must hold with no extra wiring
    of its own beyond ``shared/tests/conftest.py``'s existing
    ``_df_deploy_clocks_unwritten`` import.
    """
    live_paths = {
        (root / relpath).resolve()
        for root in deploy_clock_guard_roots(_REPO_ROOT)
        for relpath in PROTECTED_DEPLOY_CLOCK_RELPATHS
    }
    for relpath, env_var in PROTECTED_DEPLOY_CLOCK_ENV_VARS.items():
        value = os.environ.get(env_var)
        assert value, (
            f'{env_var} is unset in shared/tests -- the suite-wide redirect in '
            'df_pytest_isolation._df_deploy_clocks_unwritten did not apply here, '
            'so a spawner resolving it via `${VAR:-...}` falls through to the '
            f'live checkout default ({relpath}).'
        )
        resolved = Path(value).resolve()
        assert resolved not in live_paths, (
            f'{env_var} resolves to {resolved}, one of the LIVE checkout '
            f'deploy-clock paths this session must never write to: {live_paths}. '
            'The suite-wide redirect (df_pytest_isolation.py, task 5299) is '
            'supposed to point every protected clock env var at a tmp file for '
            "the whole session -- this is the scripts-tests-only regression "
            'that task closed.'
        )
