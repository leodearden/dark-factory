"""Config-integrity drift test for the U2 orchestrator-restart-on-merge flip.

Per plans/orchestrator-fleet-staleness-prd.md task γ: dark-factory-orchestrator.yaml
(the canonical top-level config filename — see the module docstring convention
recorded in tests/scripts/conftest.py) declares orchestrator_restart_on_merge_enabled
for dark-factory's own daemon, pointing at scripts/restart-all-orchestrators.sh
with a set of watch prefixes.

The dormant U2 coordinator (Harness._build_orchestrator_restart_coordinator)
fails OPEN at fire time: a missing/non-executable script path raises
FileNotFoundError which the coordinator swallows down to a WARNING log and
clears the pending restart — so a typo'd path would silently no-op the whole
fleet restart with no hard failure anywhere in the running system. Most of
these tests assert directly on the *committed* dark-factory-orchestrator.yaml,
independent of the runtime config-loading / hot-reload path. One additional
test loads the file through the real OrchestratorConfig pydantic model to
guard the YAML-key-to-field binding itself: OrchestratorConfig uses
``extra='ignore'``, so a future field rename in config.py or a key typo here
would otherwise silently revert to the field's (disabled-by-default) default
with no error anywhere — passing the raw-YAML assertions below while the
runtime coordinator stays dark.

This file guards the committed ``orchestrator_restart_*`` block against
silent DRIFT — the YAML-key-to-pydantic-field binding
(``tests/scripts/test_orchestrator_restart_config_drift.py::_assert_key_binds``),
the restart script's existence, and the watch prefixes' existence — and
deliberately does NOT pin the VALUE of ``orchestrator_restart_on_merge_enabled``.
That flag is an operator lever, not a code contract: it was flipped ``true``
-> ``false`` on 2026-09-03 by the fleet deploy pause (see the ``DEPLOY PAUSE``
comment block above the key in dark-factory-orchestrator.yaml; task 5020 is
the gate that restores it).

Task 5088 removed a value pin that had been added here to survive that same
pause: it asserted on the PROSE of the YAML comment block above the key,
which can be checked for shape but never for truth — a reworded pause
comment with the identical declared intent would have turned it red, and any
comment containing the magic marker phrase (including one copy-pasted
forward from a long-lifted pause) would have turned it green. The binding
guard, ``::_assert_key_binds``, is value-agnostic and is paired with
``test_binding_guard_bites_on_a_typod_key_and_on_a_renamed_field`` proving it
still bites on a typo'd key or a renamed field.
"""

import pathlib
from types import SimpleNamespace

import pytest
import yaml
from orchestrator.config import OrchestratorConfig

REPO_ROOT = pathlib.Path(__file__).parents[2]
DF_CONFIG_PATH = REPO_ROOT / "dark-factory-orchestrator.yaml"


def _load_df_config() -> dict:
    return yaml.safe_load(DF_CONFIG_PATH.read_text(encoding="utf-8"))


def _assert_key_binds(cfg: dict, config: object, key: str) -> None:
    """Assert a committed YAML key still binds to the same-named OrchestratorConfig field.

    Three terms, each catching a distinct drift mode:

    1. The key must exist in the committed YAML. ``OrchestratorConfig`` is
       declared with ``extra='ignore'``, so a YAML key typo/removal reverts
       silently to the field's default with no error anywhere.
    2. The field must exist on the parsed config. A field rename in
       ``orchestrator/src/orchestrator/config.py`` would otherwise raise a
       bare ``AttributeError`` from a plain ``getattr``; ``hasattr`` plus an
       explanatory message names the failure instead.
    3. The parsed value must equal the raw committed value — the value
       actually round-tripped.

    Deliberately no ``is True`` / ``is False`` anywhere in this helper: the
    committed value is an operator lever, not a fixed policy this helper
    enforces.
    """
    assert key in cfg, (
        f"{key!r} is not a key in the committed dark-factory-orchestrator.yaml "
        "— a YAML key typo/removal silently reverts to the field's default "
        "under OrchestratorConfig's extra='ignore', with no error anywhere"
    )
    assert hasattr(config, key), (
        f"OrchestratorConfig has no field named {key!r} — check for a field "
        "rename in orchestrator/src/orchestrator/config.py"
    )
    assert getattr(config, key) == cfg[key], (
        f"OrchestratorConfig.{key} did not round-trip the committed YAML "
        f"value ({cfg[key]!r}) — check for a field rename/typo in "
        "orchestrator/src/orchestrator/config.py"
    )


def test_orchestrator_restart_script_exists_and_is_executable() -> None:
    """The configured restart script must exist and be executable on main.

    A typo'd path is not caught anywhere at runtime — the coordinator fails
    open (FileNotFoundError -> WARNING, pending cleared) at fire time. This
    is the only gate.
    """
    cfg = _load_df_config()
    script_rel = cfg.get("orchestrator_restart_script")
    assert script_rel, "orchestrator_restart_script must be set"

    script_path = REPO_ROOT / script_rel
    assert script_path.is_file(), (
        f"orchestrator_restart_script {script_rel!r} does not exist at "
        f"{script_path} — the U2 coordinator fails open (WARNING-only) on a "
        "missing script, silently no-op'ing the fleet restart"
    )
    assert script_path.stat().st_mode & 0o111, (
        f"orchestrator_restart_script {script_rel!r} at {script_path} is not "
        "executable"
    )


def test_orchestrator_restart_watch_prefixes_all_exist() -> None:
    """Every watch_prefixes entry must resolve to a real file or directory.

    diff_touches_watched_paths (service_restart.py) does pure string
    prefix-matching against changed_files with no filesystem check, so a
    typo'd prefix here would silently never match rather than error —
    the restart hook would just never fire for that subtree.

    This enforces a project convention that is deliberately STRICTER than
    that runtime semantics (see the WATCH-PREFIX CONVENTION note above
    ``orchestrator_restart_watch_prefixes`` in orchestrator/config.yaml):
    every configured entry must be a full, existing path. The runtime
    matcher would also accept a bare string prefix of a real path that is
    not itself on disk (e.g. a partial-filename prefix meant to match
    several sibling files); no configured entry needs that today, so the
    stricter existence check is used here to catch typos.
    """
    cfg = _load_df_config()
    prefixes = cfg.get("orchestrator_restart_watch_prefixes")
    assert prefixes, "orchestrator_restart_watch_prefixes must be a non-empty list"

    missing = [p for p in prefixes if not (REPO_ROOT / p).exists()]
    assert not missing, (
        f"orchestrator_restart_watch_prefixes entries missing on disk: {missing} "
        f"(checked relative to {REPO_ROOT})"
    )


def test_orchestrator_restart_config_round_trips_through_config_model(
    root_config: OrchestratorConfig,
) -> None:
    """The committed keys must still bind to real OrchestratorConfig fields.

    The raw-YAML tests above guard the committed bytes but say nothing about
    whether those keys still bind to a live pydantic field: OrchestratorConfig
    is declared with ``extra='ignore'``, so a field rename in config.py or a
    key typo here would silently drop the value and fall back to the field's
    default with no error anywhere — the exact silent-drift failure mode this
    file exists to prevent, and one the raw-YAML tests above cannot see.

    Asserts ``parsed == raw`` via ``_assert_key_binds``, not a hardcoded
    ``True``, so a DELIBERATE operator flip (the 2026-09-03 deploy pause;
    task 5020 is the gate that restores it) does not fail this test, while a
    rename/typo still does. Residual limitation, stated honestly: while the
    committed value equals the field's own default (as it does during the
    pause), the equality term alone is vacuous — a silently-dropped key would
    ALSO read back as the default. The two presence terms inside
    ``_assert_key_binds`` are what carry the guard in that case, which is why
    ``test_binding_guard_bites_on_a_typod_key_and_on_a_renamed_field`` exists:
    it proves those terms still fire on a typo'd key and a renamed field.
    """
    cfg = _load_df_config()
    _assert_key_binds(cfg, root_config, "orchestrator_restart_on_merge_enabled")
    _assert_key_binds(cfg, root_config, "orchestrator_restart_script")
    _assert_key_binds(cfg, root_config, "orchestrator_restart_watch_prefixes")

    # Self-redeploy rate cap (task 2371). The committed YAML does not set this
    # key, so it must round-trip to the field's default (8h). A rename/typo in
    # config.py would surface here as the attribute vanishing (AttributeError).
    assert root_config.orchestrator_restart_min_interval_secs == pytest.approx(28800.0), (
        "OrchestratorConfig.orchestrator_restart_min_interval_secs did not "
        "resolve to its 8h default — check for a field rename/typo in config.py"
    )


def test_binding_guard_bites_on_a_typod_key_and_on_a_renamed_field(
    root_config: OrchestratorConfig,
) -> None:
    """``_assert_key_binds`` must still fail on the two drift modes it exists for.

    Under the 2026-09-03 deploy pause (task 5088) the committed raw value
    (``false``) equals ``OrchestratorConfig.orchestrator_restart_on_merge_
    enabled``'s own field default (``Field(default=False)``), so the
    ``parsed == raw`` term ALONE cannot distinguish "bound correctly" from
    "silently dropped by ``extra='ignore'``" while the pause holds. The two
    presence terms — the key exists in the committed YAML, and the field
    exists on the parsed config — are what actually carry the guard during
    the pause, and this test is what proves they are still there.
    """
    typod_cfg = _load_df_config()
    typod_cfg["orchestrator_restart_on_merge_enabled_typo"] = typod_cfg.pop(
        "orchestrator_restart_on_merge_enabled"
    )
    with pytest.raises(AssertionError, match="orchestrator_restart_on_merge_enabled"):
        _assert_key_binds(typod_cfg, root_config, "orchestrator_restart_on_merge_enabled")

    cfg = _load_df_config()
    renamed_config = SimpleNamespace()
    with pytest.raises(AssertionError):
        _assert_key_binds(cfg, renamed_config, "orchestrator_restart_on_merge_enabled")
