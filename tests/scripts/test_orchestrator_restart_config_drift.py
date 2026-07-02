"""Config-integrity drift test for the U2 orchestrator-restart-on-merge flip.

Per plans/orchestrator-fleet-staleness-prd.md task γ: orchestrator/config.yaml
enables orchestrator_restart_on_merge_enabled for dark-factory's own daemon,
pointing at scripts/restart-all-orchestrators.sh with a set of watch prefixes.

The dormant U2 coordinator (Harness._build_orchestrator_restart_coordinator)
fails OPEN at fire time: a missing/non-executable script path raises
FileNotFoundError which the coordinator swallows down to a WARNING log and
clears the pending restart — so a typo'd path would silently no-op the whole
fleet restart with no hard failure anywhere in the running system. This test
makes that unshippable by asserting directly on the *committed*
orchestrator/config.yaml, independent of the runtime config-loading /
hot-reload path.
"""

import pathlib

import yaml

REPO_ROOT = pathlib.Path(__file__).parents[2]
DF_CONFIG_PATH = REPO_ROOT / "orchestrator" / "config.yaml"


def _load_df_config() -> dict:
    return yaml.safe_load(DF_CONFIG_PATH.read_text(encoding="utf-8"))


def test_orchestrator_restart_on_merge_enabled_is_true() -> None:
    """The U2 coordinator must be flipped on for the dark-factory daemon."""
    cfg = _load_df_config()
    assert cfg.get("orchestrator_restart_on_merge_enabled") is True, (
        "orchestrator/config.yaml must set orchestrator_restart_on_merge_enabled: "
        "true (PRD task γ) — the U2 coordinator is otherwise dormant"
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
    """
    cfg = _load_df_config()
    prefixes = cfg.get("orchestrator_restart_watch_prefixes")
    assert prefixes, "orchestrator_restart_watch_prefixes must be a non-empty list"

    missing = [p for p in prefixes if not (REPO_ROOT / p).exists()]
    assert not missing, (
        f"orchestrator_restart_watch_prefixes entries missing on disk: {missing} "
        f"(checked relative to {REPO_ROOT})"
    )
