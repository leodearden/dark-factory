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
runs. This file pins the two halves of that redirect's contract:

* it APPLIES in ``shared/tests`` specifically, so a future refactor that
  re-narrows it back to a single rootdir -- the exact regression this task
  exists to prevent -- fails HERE too, not only in ``scripts/tests`` where the
  mitigation already worked; and
* it is UNDONE exactly at session teardown, restoring an absent var to absent
  rather than to an empty string.

Deliberately does not re-test the guard's DETECTION half
(``deploy_clock_snapshot``/``deploy_clock_violation_reason``, or that a real
falsification fails the run) -- that behaviour already has its own coverage in
``tests/scripts/test_deploy_clock_isolation.py`` and is unchanged by this task.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import df_pytest_isolation
import pytest
from df_pytest_isolation import (
    PROTECTED_DEPLOY_CLOCK_ENV_VARS,
    deploy_clock_guard_roots,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]

_PROTECTED_ENV_VARS = tuple(PROTECTED_DEPLOY_CLOCK_ENV_VARS.values())


def test_every_protected_clock_env_var_is_redirected_away_from_the_live_checkout(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Both protected clock env vars must point into THIS session's tmp tree.

    Reads ``os.environ`` directly rather than through a fixture: this test's
    whole point is that the redirect is now unconditional for every rootdir
    that imports ``df_pytest_isolation``, so it must hold with no extra wiring
    of its own beyond ``shared/tests/conftest.py``'s existing
    ``_df_deploy_clocks_unwritten`` import.

    Asserting the value lands under ``getbasetemp()`` -- not merely that it is
    non-empty and unequal to the live paths -- is what keeps that check from
    passing vacuously. These vars are routinely forwarded INTO a pytest process
    by its parent (``scripts/orchestrator-watchdog.py`` passes
    ``--setenv=FM_DEPLOY_CLOCK=<path>`` to what it spawns, and verify runs
    under the orchestrator), so an inherited value pointing anywhere harmless
    would satisfy a weaker check with the redirect deleted. Only this session's
    own basetemp proves the fixture actually ran here.
    """
    basetemp = tmp_path_factory.getbasetemp().resolve()
    # Built from the ENV-VAR table, not from the narrower change-detected
    # PROTECTED_DEPLOY_CLOCK_RELPATHS tuple: the in-flight lease is
    # redirect-only (task 4755 review fix), so it is absent from that tuple,
    # and deriving from it here would silently stop covering ORCH_FLEET_LEASE
    # — the one var whose redirect this test exists to prove points away from
    # production.
    live_paths = {
        (root / relpath).resolve()
        for root in deploy_clock_guard_roots(_REPO_ROOT)
        for relpath in PROTECTED_DEPLOY_CLOCK_ENV_VARS
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
        assert basetemp in resolved.parents, (
            f'{env_var} resolves to {resolved}, which is outside this session\'s '
            f'pytest tmp tree ({basetemp}). It avoids the live paths, but nothing '
            'here shows the redirect RAN -- an ORCH_FLEET_DEPLOY_CLOCK / '
            'FM_DEPLOY_CLOCK inherited from whatever spawned this pytest would '
            'look exactly the same, so deleting the redirect would leave this '
            'test green.'
        )


# ---------------------------------------------------------------------------
# The other half of the redirect's contract: it is undone EXACTLY on teardown.
# ---------------------------------------------------------------------------
# Only a nested session can observe this. The restore lives in the session
# fixture's own `finally`, so nothing inside the session that installed it can
# see the state afterwards -- and the distinction that matters (absent vs. set
# to '') is invisible to the fixture's own callers by construction.
#
# The nested tree is driven by an in-process `pytest.main` inside a throwaway
# subprocess rather than by `python -m pytest`, purely so the post-session read
# needs no hook-ordering argument: once `pytest.main` returns, session-scoped
# teardown has provably finished, and `os.environ` in that same process is the
# state a later spawner would see. Both observations are handed back as JSON
# files rather than parsed out of stdout (heuristic 12).

# Minimal ini so the nested run's rootdir is the tmp tree and NOT this repo:
# without it pytest walks up looking for an inifile and would inherit this
# repo's addopts.
_NESTED_INI = '[pytest]\n'

_DURING_NAME = 'env-during-session.json'
_AFTER_NAME = 'env-after-session.json'

_NESTED_CONFTEST = '''\
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from df_pytest_isolation import _df_deploy_clocks_unwritten  # noqa: F401
'''

_NESTED_TEST = f'''\
import json
import os
from pathlib import Path

from df_pytest_isolation import PROTECTED_DEPLOY_CLOCK_ENV_VARS


def test_record_the_redirected_values():
    """PASSES. Records what the redirect resolved to, for the outer assertions."""
    record = {{
        var: os.environ.get(var)
        for var in PROTECTED_DEPLOY_CLOCK_ENV_VARS.values()
    }}
    (Path(__file__).resolve().parent / {_DURING_NAME!r}).write_text(json.dumps(record))
'''

_NESTED_DRIVER = f'''\
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from df_pytest_isolation import PROTECTED_DEPLOY_CLOCK_ENV_VARS

code = pytest.main(['-q', '-p', 'no:cacheprovider', str(ROOT)])
record = {{
    var: os.environ.get(var)
    for var in PROTECTED_DEPLOY_CLOCK_ENV_VARS.values()
}}
(ROOT / {_AFTER_NAME!r}).write_text(json.dumps(record))
sys.exit(int(code))
'''


def _nested_session_env(
    tmp_path: Path, name: str, start_env: dict[str, str | None],
) -> tuple[dict[str, str | None], dict[str, str | None]]:
    """Run a throwaway session under *start_env*; return its during/after env.

    The copied ``df_pytest_isolation.py`` sits at the tmp tree's root, so the
    guard's ``Path(__file__).resolve().parent`` resolves THERE and the protected
    clocks it watches are the tmp ones -- the real checkout is never involved.

    A ``None`` in *start_env* means "absent from the child's environment", which
    is the case the restore is easiest to get subtly wrong.
    """
    root = tmp_path / name
    root.mkdir()
    shutil.copy2(Path(df_pytest_isolation.__file__), root / 'df_pytest_isolation.py')
    (root / 'pytest.ini').write_text(_NESTED_INI)
    (root / 'conftest.py').write_text(_NESTED_CONFTEST)
    (root / 'test_record_env.py').write_text(_NESTED_TEST)
    (root / '_driver.py').write_text(_NESTED_DRIVER)

    env = os.environ.copy()
    # This outer run's own addopts would otherwise be inherited and applied to
    # the nested one, where its markers and plugins mean nothing.
    env.pop('PYTEST_ADDOPTS', None)
    for var, value in start_env.items():
        if value is None:
            env.pop(var, None)
        else:
            env[var] = value

    result = subprocess.run(
        [sys.executable, '_driver.py'],
        cwd=root, env=env, capture_output=True, text=True, timeout=300,
    )

    assert result.returncode == 0, (
        'the nested session did not pass, so it proves nothing about the '
        f'restore.\nstdout={result.stdout!r}\nstderr={result.stderr!r}'
    )
    return (
        json.loads((root / _DURING_NAME).read_text()),
        json.loads((root / _AFTER_NAME).read_text()),
    )


def test_teardown_restores_an_absent_var_to_absent_not_to_empty(tmp_path: Path) -> None:
    """The branch a refactor drops first: ``os.environ[var] = saved or ''``.

    An empty string is not "unset" to the shell scripts this defence exists for:
    ``CLOCK_FILE="${ORCH_FLEET_DEPLOY_CLOCK:-$REPO_DIR/<live path>}"`` treats
    empty as unset and falls straight through to the LIVE checkout -- while
    ``os.environ`` still reports the var as set, so the next reader in the
    process believes it is redirected. That is the silently-green shape the
    whole deploy-clock defence is built to avoid.
    """
    during, after = _nested_session_env(
        tmp_path, 'absent', dict.fromkeys(_PROTECTED_ENV_VARS, None),
    )

    for var in _PROTECTED_ENV_VARS:
        assert during[var], (
            f'{var} was empty INSIDE the nested session -- the redirect never '
            'applied there, so this run says nothing about its teardown.'
        )
        assert after[var] is None, (
            f'{var} was absent from the environment before the nested session '
            f'and is {after[var]!r} after it. Teardown must POP a key it did not '
            "find, not assign it back as an empty string: `${VAR:-<live "
            'default>}` reads empty as unset and resolves to the live checkout '
            'path, which is exactly what the redirect exists to prevent.'
        )


def test_teardown_restores_a_preset_var_to_its_exact_prior_value(tmp_path: Path) -> None:
    """The other branch: a caller that had deliberately aimed the var somewhere.

    A suite is entitled to point these vars at its own file before pytest starts
    (``scripts/orchestrator-watchdog.py`` forwards ``--setenv=FM_DEPLOY_CLOCK``
    to what it spawns). The redirect overrides that for the session's duration
    and must hand the caller's value back untouched afterwards.
    """
    presets = {
        var: str(tmp_path / f'preset-{var}.json') for var in _PROTECTED_ENV_VARS
    }

    during, after = _nested_session_env(tmp_path, 'preset', dict(presets))

    for var, preset in presets.items():
        assert during[var] != preset, (
            f'{var} still held the caller\'s value {preset!r} INSIDE the nested '
            'session -- the redirect did not override it, so this run says '
            'nothing about its teardown.'
        )
        assert after[var] == preset, (
            f'{var} was {preset!r} before the nested session and is '
            f'{after[var]!r} after it. Teardown must restore the exact prior '
            "value; leaving the session's tmp path behind aims the caller at a "
            'directory pytest is about to delete.'
        )
