"""Standing guard: the operator's interim `verify_admission_pytest_n` cap is
DEPLOYED, and reaches exactly the roles it is meant to reach (task 4456).

WHY THE DEPLOYED VALUE IS WHAT IT IS: the comment block directly above
``verify_admission_pytest_n`` in ``dark-factory-orchestrator.yaml`` is the
SINGLE SOURCE OF TRUTH for that decision — its interim status, the task 3589
pairing, the measured basis, and the T6-report recommendation it overrides.
Deliberately NOT restated here or in the assertion messages below: four
paraphrases of one rationale drift apart silently, which is the same failure
mode this guard exists to prevent, relocated into prose. Read the yaml block
before changing anything; this docstring records only what is local to this
module.

Three facts this module pins, none of which any other test covers:

1. The committed ``dark-factory-orchestrator.yaml`` actually carries the cap.
   The knob is inert unless an operator sets it (the SHIPPED CODE DEFAULT is
   ``'auto'``, and stays ``'auto'`` — task 4456 changed no source), so nothing
   in the config or wiring suites can notice the deployed value going missing.
2. The cap reaches roles ``{task, background}``.
3. The cap NEVER reaches ``merge`` — deliberately, because merge bypasses
   admission slot-counting and is latency-critical.

RETUNE OR ROLLBACK IS EXACTLY TWO EDITS IN ONE COMMIT: the yaml value and
``EXPECTED_COMMITTED_PYTEST_N`` below, which are two halves of one decision.
Nothing else has to move — INCLUDING a rollback all the way to ``'auto'``.
The role test derives its expectation from the LOADED config and asserts the
NO-OP contract (no ``-n`` injected at all) whenever that value is the
``''``/``'auto'`` sentinel, so both regimes stay covered and stay green.

Harness provenance: ``_leg_for_cmd``/``_module_config`` are cross-imported from
``test_verify_admission_pytest_n`` so the leg-labelling contract has exactly
ONE definition — the ``'pytest'``-and-``'tests/'`` two-substring form, which
the joined ``'pytest tests/'`` form silently breaks the moment a ``-n`` is
spliced between them (the very rewrite under test here). ``_load_committed_
config`` stays module-local instead: the ``test_warm_lane_bash_bucket_
placement`` copy takes no constructor overrides, and this module needs one
(``verify_admission_slots_dir`` redirected at ``tmp_path``).
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any, Literal
from unittest.mock import patch

import pytest
from test_verify_admission_pytest_n import _leg_for_cmd, _module_config

from orchestrator.config import OrchestratorConfig
from orchestrator.verify import run_verification

REPO_ROOT = Path(__file__).resolve().parents[2]
DF_CONFIG_PATH = REPO_ROOT / 'dark-factory-orchestrator.yaml'

#: The value the committed yaml is expected to carry.
#:
#: TWO-EDIT RETUNE: to change or roll back the deployed cap, move this literal
#: in the SAME COMMIT as the yaml value — they are two halves of one decision,
#: and a commit moving only one of them is the failure this constant exists to
#: make loud. Rolling back means setting BOTH to ``'auto'`` (or deleting the
#: yaml key and setting this to ``'auto'``); the role tests below cover that
#: no-op regime too, so no third edit is needed. Rationale for the value lives
#: in the yaml comment block, not here.
EXPECTED_COMMITTED_PYTEST_N = '8'

#: Bound on ``_inner_pytest_cmd``'s unwrap loop — high enough for every nesting
#: verify.py can produce (nice tier around cpu-governance around the command),
#: low enough that a pathological input fails the positive control below rather
#: than spinning.
_MAX_UNWRAP_DEPTH = 8


def _inner_pytest_cmd(cmd: str) -> str:
    """Unwrap the nice/bash-c wrapper(s) and return the INNER pytest command.

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

    ``shlex.split(...)[-1]`` recovers one layer: the quoted inner command is
    the final token of the wrapped form. This LOOPS rather than unwrapping
    once, because verify.py can nest two layers — ``_govern_cpu_str`` wraps as
    ``<governed exec> -- /bin/bash -c '<cmd>'`` and the nice tier then wraps
    THAT again. Unwrapping once there would hand back the governed-exec string,
    silently turning the merge assertion into a vacuous pass. (Today
    ``_resolve_governed_exec_path`` returns None for a tmp worktree, so only
    one layer is produced here — the loop is what keeps that an implementation
    detail rather than a load-bearing assumption.) An unwrapped command
    (admission inactive, or a role with no nice tier) has no ``/bin/bash -c``
    and is returned as-is.
    """
    for _ in range(_MAX_UNWRAP_DEPTH):
        if '/bin/bash -c ' not in cmd:
            return cmd
        inner = shlex.split(cmd)[-1]
        if inner == cmd:  # not actually a wrapper — stop rather than spin
            return cmd
        cmd = inner
    return cmd


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


async def _captured_test_leg(
    config: OrchestratorConfig,
    worktree: Path,
    role: Literal['merge', 'task', 'background'],
) -> str:
    """Drive ``run_verification`` for *role* and return the INNER test-leg cmd.

    Both guards below exist so that a drift in the leg-labelling heuristic or
    in the wrapper nesting fails LOUDLY and legibly here, instead of surfacing
    downstream as an unintelligible ``RuntimeError: coroutine raised
    StopIteration`` or — worse — as a negative ``-n`` assertion passing
    vacuously against a string that was never pytest's argv at all.
    """
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

    test_cmd = next((c for c in captured_cmds if _leg_for_cmd(c) == 'test'), None)
    assert test_cmd is not None, (
        f'no test leg captured for role={role!r}; captured={captured_cmds!r}. '
        f'Either the test leg was skipped, or _leg_for_cmd no longer labels it '
        f'(it matches "pytest" and "tests/" as SEPARATE substrings precisely '
        f'because the -n cap splices tokens between them).'
    )

    inner = _inner_pytest_cmd(test_cmd)
    tokens = shlex.split(inner)
    assert 'pytest' in tokens and 'tests/' in tokens, (
        f'unwrap did not recover the pytest argv for role={role!r}: got '
        f'{tokens!r} from {test_cmd!r}. Without this positive control the '
        f'negative "-n not in tokens" assertion for merge would pass '
        f'vacuously against a wrapper string.'
    )
    return inner


def test_committed_config_carries_the_operator_cap(monkeypatch):
    config = _load_committed_config(monkeypatch)
    assert config.verify_admission_pytest_n == EXPECTED_COMMITTED_PYTEST_N, (
        f'{DF_CONFIG_PATH.name} must set verify_admission_pytest_n to '
        f'{EXPECTED_COMMITTED_PYTEST_N!r}; loaded '
        f'{config.verify_admission_pytest_n!r}.\n'
        '\n'
        'This value is a DELIBERATE OPERATOR DECISION, not a default that '
        'drifted. Its full rationale — interim status, the task 3589 pairing, '
        'the measured basis, and the contrary T6 report it overrides — lives '
        'in the comment block directly above the key in '
        f'{DF_CONFIG_PATH.name}, which is the single source of truth. Read it '
        'before changing the value.\n'
        '\n'
        'The change is EXACTLY TWO EDITS IN ONE COMMIT: this module\'s '
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

    # Asserted against the LOADED config, never against
    # EXPECTED_COMMITTED_PYTEST_N, and BOTH regimes are covered: this stays a
    # real statement about wiring under any retune of the value, including a
    # rollback to the 'auto' no-op. That is what keeps the reversion two edits
    # rather than three (a numeric-only assertion here would go red on
    # rollback, leaving the whole orchestrator suite failing).
    expected_n = config.verify_admission_pytest_n
    tokens = _argv(inner)

    if expected_n in {'', 'auto'}:
        assert '-n' not in tokens, (
            f'verify_admission_pytest_n is {expected_n!r} — the no-op '
            f'sentinel — so role={role!r} must get NO -n spliced into its '
            f'pytest test leg at all (verify.py skips the rewrite entirely, '
            f'leaving whatever the pyproject addopts already carry). Inner '
            f'argv was {tokens!r}.'
        )
        return

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

    # _captured_test_leg has already positively controlled that `inner` IS
    # pytest's argv, so the negative assertion below cannot pass vacuously
    # against a wrapper string that merely happens to lack '-n'.
    inner = await _captured_test_leg(config, worktree, 'merge')

    tokens = _argv(inner)
    assert '-n' not in tokens, (
        f"merge's test leg must NEVER be -n-capped — it bypasses admission "
        f'slot-counting and is latency-critical — but the inner argv was '
        f'{tokens!r}. (Closing this gap is task 3589\'s job, via the pyproject '
        f'addopts layer, NOT this knob: verify.py gates the cap on '
        f"role in {{'task','background'}}.)"
    )
