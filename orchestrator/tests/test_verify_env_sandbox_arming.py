"""Standing guard: the real-kernel sandbox surface is ARMED in the COMMITTED
``dark-factory-orchestrator.yaml``, and the arming actually reaches verify
subprocesses (task 4635).

WHY THIS MODULE EXISTS. Task 4635's entire deliverable is one config key —
``verify_env: {DF_REQUIRE_SANDBOX_TESTS: "1"}``. Nothing executable anchored
it: deleting the key, renaming it (``DF_REQUIRE_SANDBOX_TEST``), or writing
its value as a YAML boolean rather than the string ``"1"`` the guards compare
against would silently return ``_skip_var_tmp()`` / ``_skip_landlock()`` to
dead-code status with no test going red — reinstating the exact
"green suite, enforcement surface not running" shape 4635 was filed to
eliminate. The rationale for the value lives in the yaml comment block above
the key (single source of truth); this module pins only that it is DEPLOYED
and REACHES the subprocess, and deliberately does not paraphrase the why.

Asserted through the REAL loader and the REAL resolver, never a text match on
the YAML: config layering means the loaded+resolved value — not the file text
— is what verify subprocesses actually receive, so a key that fails
validation, gets shadowed by a later layer, or is dropped by a module override
must fail HERE rather than pass a naive grep. (Precedent for asserting on the
committed config through the real model: ``test_verify_pytest_n_operator_cap``
and ``test_warm_lane_bash_bucket_placement``; precedent for asserting on a
real ``verify_env`` block: ``scripts/tests/test_flip_reify_gate_exclude_heavy``.)

ROLLBACK IS ONE EDIT PLUS ONE: disarming means deleting the yaml key AND
deleting this module — they are two halves of one decision. Do not weaken the
assertions in place to make a removal green.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest

from orchestrator.config import OrchestratorConfig, _discover_module_configs
from orchestrator.verify import _resolve_verify_env

REPO_ROOT = Path(__file__).resolve().parents[2]
DF_CONFIG_PATH = REPO_ROOT / 'dark-factory-orchestrator.yaml'

#: The variable read by ``orchestrator/tests/test_landlock.py::_skip_var_tmp``,
#: its ``_skip_landlock`` sibling, the verbatim copies in
#: ``test_sandbox_enforcement_matrix.py``, and
#: ``fused-memory/tests/reconciliation/test_recon_sandbox_guard.py``.
ARMING_VAR = 'DF_REQUIRE_SANDBOX_TESTS'

#: The value those guards compare against, with ``== '1'``. A STRING, not a
#: bool: ``verify_env`` is typed ``dict[str, str]``, so a YAML ``true`` fails
#: validation outright, but a plausible-looking ``"true"``/``"yes"``/``"0"``
#: would load fine and silently disarm every guard. That is what this literal
#: pins.
ARMED_VALUE = '1'

#: Every role ``_resolve_verify_env`` can be asked for. Checked exhaustively
#: because the arming must not depend on which lane runs the suite — a merge
#: verify that skipped the enforcement rows while task verify ran them would
#: be the same silent hole, gated on lane instead of on host.
_ROLES: tuple[Literal['merge', 'task', 'background'], ...] = (
    'merge', 'task', 'background',
)


@pytest.fixture
def committed_config(monkeypatch: pytest.MonkeyPatch) -> OrchestratorConfig:
    """The COMMITTED dark-factory-orchestrator.yaml, loaded through the real model.

    ``ORCH_CONFIG_PATH`` is monkeypatched (not just passed) because the yaml
    settings source reads it from the environment at construction time, and
    because ``load_config`` writes it back — monkeypatch is what restores the
    ambient value for every other test in the session.
    """
    monkeypatch.setenv('ORCH_CONFIG_PATH', str(DF_CONFIG_PATH))
    return OrchestratorConfig()


class TestSandboxTestsArmedInCommittedConfig:
    def test_committed_config_carries_the_arming_key(
        self, committed_config: OrchestratorConfig,
    ):
        """The deployed value is present and is exactly the string the guards test."""
        assert DF_CONFIG_PATH.is_file(), (
            f'{DF_CONFIG_PATH} missing — this guard cannot pass vacuously.'
        )
        verify_env = committed_config.verify_env or {}
        assert ARMING_VAR in verify_env, (
            f'{ARMING_VAR} is absent from the committed verify_env. The '
            f'real-kernel sandbox enforcement surface (30 tests in '
            f'test_landlock.py + test_sandbox_enforcement_matrix.py) can now '
            f'skip silently on a host where /var/tmp is unwritable or landlock '
            f'is unavailable. Restore the key or delete this module '
            f'deliberately — see {DF_CONFIG_PATH.name}, task 4635.'
        )
        assert verify_env[ARMING_VAR] == ARMED_VALUE, (
            f'{ARMING_VAR}={verify_env[ARMING_VAR]!r}, but the guards compare '
            f'with == {ARMED_VALUE!r}. Any other value disarms them silently.'
        )

    @pytest.mark.parametrize('role', _ROLES)
    def test_resolved_verify_env_reaches_every_verify_role(
        self, committed_config: OrchestratorConfig, role,
    ):
        """The resolver — not just the file — hands the var to the subprocess."""
        resolved = _resolve_verify_env(committed_config, None, role=role)
        assert resolved.get(ARMING_VAR) == ARMED_VALUE, (
            f'role={role!r} resolves {ARMING_VAR}={resolved.get(ARMING_VAR)!r}; '
            f'expected {ARMED_VALUE!r}.'
        )

    def test_no_module_config_shadows_the_arming_key(
        self, committed_config: OrchestratorConfig,
    ):
        """Task 4635's scope decision, made executable.

        One top-level key was chosen over per-package copies precisely because
        ``_resolve_verify_env`` starts from ``config.verify_env`` and merges
        ``module_config.verify_env`` ON TOP — so a module that later declares
        its OWN ``verify_env`` block without this key does not merely fail to
        add it, it can DROP it for that module's whole suite. That is the one
        way the single-key decision can silently stop holding, so it is checked
        against every module config actually discovered in this tree (which
        today includes both packages that read the variable: ``orchestrator``
        and ``fused-memory``).
        """
        modules = _discover_module_configs(REPO_ROOT)
        assert modules, (
            f'No module configs discovered under {REPO_ROOT} — this guard '
            f'would pass vacuously.'
        )
        disarmed = {
            prefix: _resolve_verify_env(committed_config, mc).get(ARMING_VAR)
            for prefix, mc in modules.items()
            if _resolve_verify_env(committed_config, mc).get(ARMING_VAR)
            != ARMED_VALUE
        }
        assert not disarmed, (
            f'These module configs shadow {ARMING_VAR} away from '
            f'{ARMED_VALUE!r}: {disarmed}. A module-level verify_env REPLACES '
            f'top-level keys it redeclares; re-add the key to that module or '
            f'drop its verify_env override.'
        )
