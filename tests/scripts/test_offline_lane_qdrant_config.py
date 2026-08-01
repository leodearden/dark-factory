"""Config-integrity test for dark-factory's own offline-lane instance.

Per plans/integration-test-lane-prd.md task beta: dark-factory-orchestrator.yaml
instantiates the generic config-driven off-hot-path offline lane (schema
delivered by task 2789/alpha) with a ``qdrant-integration`` command that runs
``pytest -m integration`` in ``fused-memory/`` — restoring the
qdrant-client/mem0 version-compat coverage that task 2773 removes from the
merge-verify hot path, on the offline lane instead of letting it lapse.

Scoped to the ``qdrant-integration`` entry BY NAME, not by asserting the lane
holds exactly one command: the lane is a shared, growable list. Task 3349
(PRD plans/warm-lane-infra-repatriation-prd.md §11 q4, re-decided) added a
second ``warm-lane-bash`` entry, and further entries are expected. A total-count
pin here would fail every such addition while guarding nothing this test is
about — ``orchestrator/tests/test_warm_lane_bash_bucket_placement.py`` owns the
equivalent by-name guard for its own entry.

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
    offline-lane sub-run with the three gate flags plus its own LaneCommand.

    Other lane entries may coexist; this asserts only the qdrant one.
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
    matches = [c for c in commands if getattr(c, "name", None) == "qdrant-integration"]
    assert len(matches) == 1, (
        "config.git.offline_lane_commands did not round-trip exactly one "
        f"'qdrant-integration' entry from the committed YAML (got "
        f"{len(matches)}; all entries: "
        f"{[getattr(c, 'name', None) for c in commands]!r}) — check for a "
        "field rename/typo in config.py"
    )
    (command,) = matches
    assert isinstance(command, LaneCommand), (
        "the 'qdrant-integration' offline_lane_commands entry is not a "
        f"LaneCommand instance (got {type(command)!r})"
    )
    assert command.command == "pytest -m integration", (
        "the 'qdrant-integration' entry's .command did not round-trip "
        "'pytest -m integration' from the committed YAML — check for a "
        "field rename/typo in config.py"
    )
    assert command.cwd == "fused-memory", (
        "the 'qdrant-integration' entry's .cwd did not round-trip "
        "'fused-memory' from the committed YAML — check for a field "
        "rename/typo in config.py"
    )
    assert command.fix_task_priority == "medium", (
        "the 'qdrant-integration' entry's .fix_task_priority did not "
        "round-trip 'medium' from the committed YAML — check for a field "
        "rename/typo in config.py"
    )
    assert command.enabled is True, (
        "the 'qdrant-integration' entry's .enabled is not True — the "
        "offline-lane runner skips disabled commands entirely, so a committed "
        "'enabled: false' would silently drop the qdrant-integration "
        "coverage this test exists to guard (check the committed YAML and "
        "for a field rename/typo in config.py)"
    )
