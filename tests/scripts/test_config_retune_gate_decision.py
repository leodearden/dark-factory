"""Executable premises of the config-retune gate decision (task 3886).

The decision and its rationale live in ``docs/config-retune-gate.md``. Each
test here asserts one of its premises on runtime state: the config as the
production loader resolves it, and the production predicates applied to it.
The premise that every orchestrator unit test is bound to the operational
config file is asserted where that binding runs, in
``orchestrator/tests/test_operational_config_binding.py``.

No assertion names a knob's value. Each reads the live value or derives its
fixture from it, so retuning a knob cannot falsify anything here. Nothing
skips: an unimportable production module fails the guard.
"""
from __future__ import annotations

import pathlib
from collections.abc import Callable

import pytest
import yaml
from orchestrator.config import ModuleConfig, OrchestratorConfig
from shared.locking import normalize_lock

from orchestrator import verify, verify_plan

DF_CONFIG_NAME = 'dark-factory-orchestrator.yaml'
DECISION_DOC = 'docs/config-retune-gate.md'


def test_merge_lane_already_gates_a_config_only_retune(root_config: OrchestratorConfig) -> None:
    """A config retune that enters the merge lane is verified against every module's suite."""
    assert root_config.merge_verify_breadth == 'full', (
        f'{DF_CONFIG_NAME} declares merge_verify_breadth='
        f'{root_config.merge_verify_breadth!r}, not "full". Task 3886 declined a '
        'new pre-merge gate for config retunes because this one already runs '
        "every registered module's suite on a merge. Re-take the decision in "
        f'{DECISION_DOC} rather than editing this assertion.'
    )
    assert verify_plan._merge_breadth_is_full(root_config) is True, (
        'verify_plan._merge_breadth_is_full rejects a config declaring '
        f'merge_verify_breadth={root_config.merge_verify_breadth!r}, so the '
        'declared value no longer drives the full merge gate.'
    )
    assert verify._merge_config_only_diff_forces_full_gate(root_config, [DF_CONFIG_NAME]) is True, (
        f'a config-only merge diff touching {DF_CONFIG_NAME} does not force the '
        'full gate: git.merge_config_only_full_gate_globs='
        f'{root_config.git.merge_config_only_full_gate_globs!r}. Restore the '
        f'entry in the git: block of {DF_CONFIG_NAME}; {DECISION_DOC} says why '
        'it is needed.'
    )


def test_a_config_file_layer_wins_over_the_package_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    """A value the file at ``ORCH_CONFIG_PATH`` declares beats the package-bundled default.

    The declared value is derived from the package default, so the two differ
    whatever either is tuned to.
    """
    monkeypatch.setenv('ORCH_CONFIG_PATH', str(tmp_path / 'absent.yaml'))
    package_default = OrchestratorConfig(project_root=tmp_path).lock_depth

    declared = package_default + 1
    config_file = tmp_path / DF_CONFIG_NAME
    config_file.write_text(yaml.safe_dump({'lock_depth': declared}))
    monkeypatch.setenv('ORCH_CONFIG_PATH', str(config_file))
    resolved = OrchestratorConfig(project_root=tmp_path).lock_depth

    assert resolved == declared, (
        f'the production loader resolved lock_depth={resolved!r} from a config '
        f'file declaring {declared!r}, over a package default of '
        f'{package_default!r}. The config-file layer no longer wins, so '
        'orchestrator unit tests no longer read operational values.'
    )


def test_every_discovered_module_config_is_reachable_at_the_operational_lock_depth(
    root_config: OrchestratorConfig,
    discover_module_configs: Callable[[], dict[str, ModuleConfig]],
) -> None:
    """No module config's prefix is truncated by the lock depth the scheduler applies.

    ``load_config`` only warns about such a config, which is then half-applied:
    its commands still run, but the scheduler and workflow see module paths
    through ``shared.locking.normalize_lock`` and never match its scheduling
    limits. A downward retune of ``lock_depth`` would cause that silently.
    """
    discovered = discover_module_configs()
    assert discovered, (
        'config._discover_module_configs found no module configs, so this guard '
        'would pass vacuously.'
    )

    seen_by_scheduler = {
        prefix: normalize_lock(prefix, root_config.lock_depth) for prefix in discovered
    }
    truncated = {prefix: seen for prefix, seen in seen_by_scheduler.items() if seen != prefix}
    assert not truncated, (
        f'at the operational lock_depth={root_config.lock_depth}, the scheduler '
        f'truncates these module-config prefixes (prefix -> seen as): {truncated!r}. '
        'Their max_per_module and module_overrides are silently ignored. Move '
        f'the orchestrator.yaml up or raise lock_depth in {DF_CONFIG_NAME}. '
        f'Discovered prefixes: {sorted(discovered)}'
    )
