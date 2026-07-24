"""Config-integrity test for dark-factory's own offline-lane instance.

Per plans/integration-test-lane-prd.md task beta: dark-factory-orchestrator.yaml
instantiates the generic config-driven off-hot-path offline lane (schema
delivered by task 2789/alpha) with a single ``qdrant-integration`` command
that runs ``pytest -m integration`` in ``fused-memory/`` — restoring the
qdrant-client/mem0 version-compat coverage that task 2773 removes from the
merge-verify hot path, on the offline lane instead of letting it lapse.

Loads the committed YAML THROUGH the real OrchestratorConfig pydantic model
rather than a raw yaml.safe_load: OrchestratorConfig is declared with
``extra='ignore'``, so a key typo here or a future field rename in config.py
would otherwise silently drop the value and revert to the field's default
with no error anywhere. Mirrors
tests/scripts/test_orchestrator_restart_config_drift.py::test_orchestrator_restart_config_round_trips_through_config_model.
"""

import pathlib

import pytest

from orchestrator.config import LaneCommand, OrchestratorConfig

REPO_ROOT = pathlib.Path(__file__).parents[2]
DF_CONFIG_PATH = REPO_ROOT / "dark-factory-orchestrator.yaml"


def test_offline_lane_qdrant_config_round_trips_through_config_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """dark-factory-orchestrator.yaml must instantiate the qdrant-integration
    offline-lane sub-run with the three gate flags plus its one LaneCommand.
    """
    monkeypatch.setenv("ORCH_CONFIG_PATH", str(DF_CONFIG_PATH))
    config = OrchestratorConfig()

    assert config.git.offline_lane_enabled is True, (
        "config.git.offline_lane_enabled did not bind to True from the "
        "committed YAML — check for a field rename/typo in config.py "
        "(OrchestratorConfig uses extra='ignore', so a mismatch silently "
        "reverts to the disabled-by-default default instead of raising)"
    )
    assert config.git.persistent_offline_deep_worktree is True, (
        "config.git.persistent_offline_deep_worktree did not bind to True "
        "from the committed YAML — the offline-deep lane worker cannot run "
        "without its dedicated worktree even if offline_lane_enabled is True "
        "(check for a field rename/typo in config.py)"
    )
    assert config.git.offline_lane_legacy_numeric_enabled is False, (
        "config.git.offline_lane_legacy_numeric_enabled did not bind to "
        "False from the committed YAML — dark-factory has no "
        "scripts/run-offline-deep.sh, so leaving this at its True default "
        "would make the offline-deep worker attempt a nonexistent script "
        "(check for a field rename/typo in config.py)"
    )

    commands = config.git.offline_lane_commands
    assert len(commands) == 1, (
        "config.git.offline_lane_commands did not round-trip exactly one "
        f"entry from the committed YAML (got {len(commands)}) — check for a "
        "field rename/typo in config.py"
    )
    (command,) = commands
    assert isinstance(command, LaneCommand), (
        "config.git.offline_lane_commands[0] is not a LaneCommand instance "
        f"(got {type(command)!r})"
    )
    assert command.name == "qdrant-integration", (
        "offline_lane_commands[0].name did not round-trip 'qdrant-integration' "
        "from the committed YAML — check for a field rename/typo in config.py"
    )
    assert command.command == "pytest -m integration", (
        "offline_lane_commands[0].command did not round-trip "
        "'pytest -m integration' from the committed YAML — check for a "
        "field rename/typo in config.py"
    )
    assert command.cwd == "fused-memory", (
        "offline_lane_commands[0].cwd did not round-trip 'fused-memory' from "
        "the committed YAML — check for a field rename/typo in config.py"
    )
    assert command.fix_task_priority == "medium", (
        "offline_lane_commands[0].fix_task_priority did not round-trip "
        "'medium' from the committed YAML — check for a field rename/typo "
        "in config.py"
    )
