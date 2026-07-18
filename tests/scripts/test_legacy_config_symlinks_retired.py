"""Repo-structure guard: the two transitional top-level orchestrator-config
symlinks are retired, and the load-bearing structure they were never allowed to
touch is preserved (task 2719 — the deferred step-9 cleanup from task 2698).

Task 2698 canonicalized the operational orchestrator config to the top-level
``dark-factory-orchestrator.yaml`` and left two transitional symlinks pointing
at it (both tracked as git mode-120000 blobs):

    orchestrator.yaml         -> dark-factory-orchestrator.yaml
    orchestrator/config.yaml  -> ../dark-factory-orchestrator.yaml

Task 2719 removes them once the live fleet has restarted onto the canonical
``--config`` path. This is a runtime-state (filesystem / tracked-tree)
assertion — NOT a ``__doc__``/introspection meta-test — so it has a proper
RED (both symlinks present on the base tree) -> GREEN (removed) home and
guards against the symlinks being re-introduced.

Each assertion also checks ``is_symlink()`` so a *dangling*-symlink leftover
(``exists()`` would report ``False`` for one, silently passing) is still
caught.

HARD CONSTRAINT (inherited from task 2698): ONLY the two top-level mode-120000
symlinks are removed. The orchestrator module config
``orchestrator/orchestrator.yaml`` (a real mode-100644 file), the
``orchestrator/src/orchestrator`` package, and the canonical
``dark-factory-orchestrator.yaml`` MUST all survive.
"""
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_toplevel_orchestrator_yaml_symlink_retired() -> None:
    """The top-level ``orchestrator.yaml`` transitional symlink is gone."""
    p = REPO_ROOT / "orchestrator.yaml"
    assert not p.exists() and not p.is_symlink(), (
        f"task 2719: the transitional top-level symlink {p} "
        "(orchestrator.yaml -> dark-factory-orchestrator.yaml) must be retired; "
        "it is still present (or is a dangling-symlink leftover)"
    )


def test_orchestrator_config_yaml_symlink_retired() -> None:
    """The ``orchestrator/config.yaml`` transitional symlink is gone."""
    p = REPO_ROOT / "orchestrator" / "config.yaml"
    assert not p.exists() and not p.is_symlink(), (
        f"task 2719: the transitional symlink {p} "
        "(orchestrator/config.yaml -> ../dark-factory-orchestrator.yaml) must be "
        "retired; it is still present (or is a dangling-symlink leftover)"
    )


def test_hard_constraint_module_config_preserved() -> None:
    """HARD CONSTRAINT: the orchestrator MODULE config (a real file) survives."""
    p = REPO_ROOT / "orchestrator" / "orchestrator.yaml"
    assert p.is_file(), (
        f"task 2719 HARD CONSTRAINT VIOLATED: the orchestrator module config {p} "
        "(mode-100644, NOT one of the two removed transitional symlinks) must be "
        "preserved"
    )


def test_hard_constraint_orchestrator_package_preserved() -> None:
    """HARD CONSTRAINT: the orchestrator source package survives."""
    p = REPO_ROOT / "orchestrator" / "src" / "orchestrator"
    assert p.is_dir(), (
        f"task 2719 HARD CONSTRAINT VIOLATED: the orchestrator source package {p} "
        "must be preserved — only the two top-level config symlinks are removed"
    )


def test_canonical_config_preserved() -> None:
    """The canonical top-level orchestrator config (the symlinks' target) survives."""
    p = REPO_ROOT / "dark-factory-orchestrator.yaml"
    assert p.is_file(), (
        f"task 2719: the canonical operational config {p} — the target the two "
        "retired symlinks pointed at, and the path all consumers were repointed "
        "to — must be preserved"
    )
