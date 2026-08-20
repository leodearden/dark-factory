"""Standing guard: the operator's interim `verify_admission_pytest_n` cap is
DEPLOYED, and reaches exactly the roles it is meant to reach (task 4456).

Three facts this module pins, none of which any other test covers:

1. The committed ``dark-factory-orchestrator.yaml`` actually carries the cap.
   The knob is inert unless an operator sets it (the SHIPPED CODE DEFAULT is
   ``'auto'``, and stays ``'auto'`` — task 4456 changed no source), so nothing
   in the config or wiring suites can notice the deployed value going missing.
2. The cap reaches roles ``{task, background}``.
3. The cap NEVER reaches ``merge`` — deliberately, because merge bypasses
   admission slot-counting and is latency-critical.

PAIRED WITH TASK 3589, which is the other half of this and is NOT closed by
4456: 3589 puts ``-n 8`` in ``addopts`` (orchestrator/pyproject.toml,
fused-memory/pyproject.toml) and thereby ALSO covers merge verifies, the
offline lane, and agent-initiated pytest inside the implement phase — none of
which this knob reaches, since it lives entirely inside verify.py. When 3589
lands, reconcile the two to the same value and say in that commit which layer
is authoritative.

OVERRIDES A STANDING RECOMMENDATION — do not read this module alone.
``plans/verify-oversubscription-benchmark-2026-07-14.md`` (the T6 report,
task 2394) recommended KEEPING ``'auto'`` because no clean idle window was
obtainable on this host, and nominated ``-n 16`` as the next candidate to try.
The deployed ``"8"`` rests on task 3589's LATER loaded-host ladder, which
postdates that report. Anyone reopening the value question must reconcile BOTH
sources rather than cite either alone; the yaml comment block and that report's
dated addendum each point at the other.

Provenance: OPERATOR DECISION (Leo, 2026-08-19, esc-5984-3), paired with
dark_factory:3589 and reify:6018.

Helpers are duplicated from test_verify_admission_pytest_n.py rather than
imported or hoisted: each admission test file is deliberately self-contained,
and a conftest.py edit would trip verify.py's ``has_conftest`` and widen every
downstream verify to the full owning-package suite.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from orchestrator.config import ModuleConfig, OrchestratorConfig
from orchestrator.verify import run_verification

REPO_ROOT = Path(__file__).resolve().parents[2]
DF_CONFIG_PATH = REPO_ROOT / 'dark-factory-orchestrator.yaml'

#: The value the committed yaml is expected to carry.
#:
#: SINGLE-EDIT REVERSION: to retune or roll back, change this literal in the
#: SAME COMMIT as the yaml value — they are two halves of one decision, and a
#: commit moving only one of them is the failure this constant exists to make
#: loud. Rolling back means setting BOTH to ``'auto'`` (or deleting the yaml
#: key and setting this to ``'auto'``); the knob is green-tier hot-reloadable
#: (RELOADABLE_FIELDS, config.py:5250), so that reverts the fleet with no
#: restart and no code change.
EXPECTED_COMMITTED_PYTEST_N = '8'

# Module-local wiring fixtures (see module docstring for why they are copied).
_TEST_CMD = 'pytest tests/'
_LINT_CMD = 'ruff'
_TYPE_CMD = 'pyright'


def _leg_for_cmd(cmd: str) -> str:
    """Label which leg *cmd* belongs to, checking ``'pytest'``/``'tests/'`` as
    two SEPARATE substrings rather than the joined ``_TEST_CMD``: the whole
    point of this module is a ``-n`` cap that splices flags BETWEEN them
    (``pytest tests/`` -> ``pytest -n 8 tests/``), which breaks containment of
    the joined string.
    """
    if 'pytest' in cmd and 'tests/' in cmd:
        return 'test'
    if _LINT_CMD in cmd:
        return 'lint'
    if _TYPE_CMD in cmd:
        return 'type'
    return cmd


def _module_config(**overrides: Any) -> ModuleConfig:
    kwargs: dict[str, Any] = dict(
        prefix='pkg',
        test_command=_TEST_CMD,
        lint_command=_LINT_CMD,
        type_check_command=_TYPE_CMD,
        # Sequential so the three legs run strictly test -> lint -> type.
        concurrent_verify=False,
    )
    kwargs.update(overrides)
    return ModuleConfig(**kwargs)


def _inner_pytest_cmd(cmd: str) -> str:
    """Unwrap the nice/bash-c wrapper and return the INNER pytest command.

    An active admission gate wraps the test leg as
    ``<nice argv> /bin/bash -c <shlex.quote(cmd)>``, and EVERY role's nice tier
    itself contains a literal ``-n`` (``nice -n 5`` for merge, ``nice -n 15
    ionice -c2 -n7`` for task, ``nice -n 19`` for background — see
    ``shared.verify_admission._NICE_TIERS``).

    So asserting on ``-n`` against the WHOLE captured string is worthless in
    both directions: it would pass vacuously for task/background off the
    wrapper's own ``nice -n 15`` even if the cap never reached pytest at all,
    and it would fail spuriously for merge off ``nice -n 5`` even though merge
    is correctly un-capped. Every assertion below therefore runs against this
    unwrapped inner command.

    ``shlex.split(...)[-1]`` recovers it: the quoted inner command is the final
    token of the wrapped form. An unwrapped command (admission inactive, or a
    role with no nice tier) has no ``/bin/bash -c`` and is returned as-is.
    """
    if '/bin/bash -c ' not in cmd:
        return cmd
    return shlex.split(cmd)[-1]


def _argv(inner_cmd: str) -> list[str]:
    """Split the inner pytest command into argv TOKENS.

    Every ``-n`` assertion below runs against tokens, never a substring of the
    joined string: verify.py also injects ``--junitxml <path>`` into the test
    leg, and that path is a tmp/worktree path whose text is outside this test's
    control — a worktree or branch named e.g. ``df-nightly`` would make a
    substring check for ``'-n'`` match the PATH and silently invert the
    merge-role assertion. ``apply_pytest_numprocesses`` emits the flag as two
    adjacent tokens (``'-n'``, ``<n>``) in both its structured and raw
    branches, so tokens are also the exact shape being asserted about.
    """
    return shlex.split(inner_cmd)


def _load_committed_config(
    monkeypatch: pytest.MonkeyPatch, **overrides: Any
) -> OrchestratorConfig:
    """Load the COMMITTED dark-factory-orchestrator.yaml through the real model.

    Deliberately not a hand-parse of the YAML text: config layering means the
    LOADED ATTRIBUTE — not the file text — is what the daemon acts on, so a
    value that fails validation or is shadowed by a later layer must FAIL here
    rather than pass a naive text match. Same rationale as
    ``test_warm_lane_bash_bucket_placement._load_committed_config``.

    *overrides* are passed as constructor kwargs (the highest-priority settings
    layer), so a caller can redirect e.g. ``verify_admission_slots_dir`` at a
    tmp_path while every other value still comes from the committed file.
    """
    monkeypatch.setenv('ORCH_CONFIG_PATH', str(DF_CONFIG_PATH))
    return OrchestratorConfig(**overrides)


async def _captured_test_leg(config: OrchestratorConfig, worktree: Path, role: str) -> str:
    """Drive ``run_verification`` for *role* and return the INNER test-leg cmd."""
    captured_cmds: list[str] = []

    async def spy_run_cmd(cmd, cwd, timeout, env=None, log_path=None, **kwargs):
        captured_cmds.append(cmd)
        return 0, '', False

    with patch('orchestrator.verify._run_cmd', side_effect=spy_run_cmd):
        await run_verification(
            worktree=worktree,
            config=config,
            module_config=_module_config(),
            role=role,
            attempt_id=None,
        )

    test_cmd = next(c for c in captured_cmds if _leg_for_cmd(c) == 'test')
    return _inner_pytest_cmd(test_cmd)


def test_committed_config_carries_the_operator_cap(monkeypatch):
    config = _load_committed_config(monkeypatch)
    assert config.verify_admission_pytest_n == EXPECTED_COMMITTED_PYTEST_N, (
        f'{DF_CONFIG_PATH.name} must set verify_admission_pytest_n to '
        f'{EXPECTED_COMMITTED_PYTEST_N!r}; loaded '
        f'{config.verify_admission_pytest_n!r}.\n'
        '\n'
        'This value is a DELIBERATE OPERATOR DECISION (Leo, 2026-08-19, '
        'esc-5984-3), not a default that drifted. It overrides '
        'plans/verify-oversubscription-benchmark-2026-07-14.md, which '
        "recommended KEEPING 'auto' and nominated -n 16; the override rests on "
        "task 3589's later loaded-host ladder. Reconcile BOTH sources before "
        'changing it, never cite either alone.\n'
        '\n'
        'Any change here must also be reconciled with task 3589, which puts '
        '-n 8 in the pyproject addopts layer (covering merge, the offline '
        'lane, and agent-initiated pytest — none of which this knob reaches).\n'
        '\n'
        'The reversion is EXACTLY TWO EDITS IN ONE COMMIT: this module\'s '
        'EXPECTED_COMMITTED_PYTEST_N and the yaml value. If you are seeing '
        'this failure, you probably moved one without the other.'
    )


@pytest.mark.real_verify_admission
@pytest.mark.asyncio
@pytest.mark.parametrize('role', ['task', 'background'])
async def test_committed_cap_reaches_task_and_background_roles(monkeypatch, tmp_path, role):
    # slots_dir under tmp_path so concurrent xdist workers never contend on the
    # shared operator slots directory (same rationale as
    # test_verify_admission_pytest_n.py::TestPytestNWiring).
    config = _load_committed_config(
        monkeypatch, verify_admission_slots_dir=str(tmp_path / 'slots')
    )
    worktree = tmp_path / 'wt'
    worktree.mkdir()

    inner = await _captured_test_leg(config, worktree, role)

    # Derived from the LOADED config, not from EXPECTED_COMMITTED_PYTEST_N, so
    # this stays a real assertion about wiring under any retune of the value.
    expected_n = config.verify_admission_pytest_n
    tokens = _argv(inner)
    assert '-n' in tokens, (
        f'role={role!r} must carry the committed cap -n {expected_n} in its '
        f'pytest test leg; inner argv was {tokens!r}. The cap is what keeps a '
        f'fleet-wide verify from spawning ~32 workers per module at ~446 MB '
        f'RSS each.'
    )
    assert tokens[tokens.index('-n') + 1] == expected_n, (
        f'role={role!r} carries -n but with the wrong value: expected '
        f'{expected_n!r} from the committed config, got '
        f'{tokens[tokens.index("-n") + 1]!r}'
    )


@pytest.mark.real_verify_admission
@pytest.mark.asyncio
async def test_committed_cap_never_reaches_the_merge_role(monkeypatch, tmp_path):
    config = _load_committed_config(
        monkeypatch, verify_admission_slots_dir=str(tmp_path / 'slots')
    )
    worktree = tmp_path / 'wt'
    worktree.mkdir()

    inner = await _captured_test_leg(config, worktree, 'merge')

    tokens = _argv(inner)
    assert '-n' not in tokens, (
        f"merge's test leg must NEVER be -n-capped — it bypasses admission "
        f'slot-counting and is latency-critical — but the inner argv was '
        f'{tokens!r}. (Closing this gap is task 3589\'s job, via the pyproject '
        f'addopts layer, NOT this knob: verify.py gates the cap on '
        f"role in {{'task','background'}}.)"
    )
