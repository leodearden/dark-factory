"""Behavioural tests for skills/factory-init/scripts/find_escalation_port.py.

WHY THIS FILE LIVES HERE AND NOT UNDER skills/. A ``skills/**/tests/`` directory
or a ``skills/**/test_*.py`` file would trip ``test_skills_has_no_tests_of_its_own``
in ``test_skills_module_config_decision.py`` — the deliberate REVISIT TRIGGER task
3554 installed, whose six pathspecs cover both the nested ``skills/**/...`` and the
direct-child ``skills/...`` spellings (so even a bare ``skills/test_probe.py`` fires
it). ``tests/scripts/`` is collected by TWO registered module configs' test_commands
(``tests/scripts`` and ``scripts``), so a module placed here actually runs inside
``verify.run_full_verification``'s gather rather than sitting uncollected.

WHY THESE TESTS NEED A MEASURED RED. This is characterization work over code that is
already correct, so every assertion below is green at HEAD BY CONSTRUCTION and a green
run proves nothing about whether the guard bites. Each test therefore records the
failure text observed against a NAMED scratch mutation of
``skills/factory-init/scripts/find_escalation_port.py``, reverted before commit —
the convention ``test_skills_module_config_decision.py`` and
``test_root_lint_covers_nonmember_py.py`` already follow in this same directory.
Every MEASURED RED below was observed at base main ``fc6f048b55``.

MUST NOT SKIP. There is no ``pytest.importorskip`` and no try/except-and-skip
anywhere in this module. The script under test is a tracked file in this repo — the
only tracked ``.py`` under ``skills/`` — so a missing script, a broken loader, or a
module that stops importing is a REAL regression and must FAIL loudly rather than
turn this suite green-by-vacuity.

TWO MACHINE DEPENDENCIES, BOTH MEASURED, BOTH PINNED HERE. ``--df-root`` alone does
NOT make this script hermetic:
  1. ``known_project_roots()`` shells out to ``systemctl --user show
     fused-memory.service -p Environment`` and merges DASHBOARD_KNOWN_PROJECT_ROOTS.
     MEASURED: running the script with ``--df-root`` pointed at a /tmp tree still
     surveyed the real machine's roots (know-live, solar-challenge,
     solar-challenge-platform, pump-web-ui, ...).
  2. ``is_bound()`` binds a REAL 127.0.0.1 socket. MEASURED with systemctl
     neutralised: ``--base 8100`` over a tmp tree returned 8104 rather than 8102, and
     ``--base 8002`` returned 8004 rather than 8003, because live escalation servers
     hold those ports on this host.
So the ``no_systemd`` fixture and the ``is_bound`` rebinding are load-bearing for
hermeticity, not conveniences. The repo-root ``conftest.py``'s session isolation
fixtures (``_df_fleet_dir_redirect``, ``_df_no_leaked_drain_processes``,
``_df_no_synthetic_heartbeats_in_live_fleet``; tasks 3798/3799) govern drain-script
processes, ORCH_FLEET_DIR and fleet heartbeat files — none of them constrains socket
binding or subprocess patching, so no interaction with this module is expected.
"""

import importlib.util
import pathlib
import subprocess
import types

import pytest  # pyright: ignore[reportMissingImports]

REPO_ROOT = pathlib.Path(__file__).parents[2]
SCRIPT_PATH = REPO_ROOT / "skills" / "factory-init" / "scripts" / "find_escalation_port.py"


def _load_find_escalation_port() -> types.ModuleType:
    """Load the script by file path.

    It is not on any package path and its directory has no ``__init__.py``, so a
    plain ``import`` cannot reach it. Mirrors ``_load_watchdog`` in
    ``test_orchestrator_watchdog.py``.
    """
    assert SCRIPT_PATH.is_file(), f"script under test is missing: {SCRIPT_PATH}"
    spec = importlib.util.spec_from_file_location("find_escalation_port", SCRIPT_PATH)
    assert spec is not None, f"could not build spec from {SCRIPT_PATH}"
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def fep() -> types.ModuleType:
    """A FRESHLY loaded module per test.

    Deliberately not module-scoped: several tests below rebind ``fep.is_bound``
    (main() resolves it as a module global at call time), and a shared instance
    would leak that rebinding into siblings.
    """
    return _load_find_escalation_port()


@pytest.fixture
def no_systemd(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutralise ``known_project_roots``' systemctl branch.

    MEASURED: without this, running the script with ``--df-root`` pointed at a /tmp
    tree still surveyed the real machine's roots (know-live, solar-challenge, ...),
    because the live unit's DASHBOARD_KNOWN_PROJECT_ROOTS is merged in ahead of the
    sibling glob. Patching the GLOBAL ``subprocess.run`` is the precedent
    ``test_orchestrator_watchdog.py`` already sets; monkeypatch restores it.

    FileNotFoundError is one of the two failure modes the script's bare
    ``except Exception`` swallows (the binary missing); step-3 covers the other
    (``subprocess.TimeoutExpired``, the hung-systemd case the ``timeout=10`` implies).
    """

    def _no_systemctl(*args, **kwargs):
        raise FileNotFoundError("systemctl neutralised for hermetic test")

    monkeypatch.setattr(subprocess, "run", _no_systemctl)


def _project_tree(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """Build a hermetic parent-of-projects tree; return ``(parent, df_root)``.

    Layout — deliberately falsifiable in both directions:
        <tmp>/dark-factory/        the checkout itself (a project root too)
        <tmp>/proj-a/              a sibling project
        <tmp>/proj-a/nested/       a GRANDCHILD — discovery must not recurse into it
        <tmp>/target/              another sibling (the --exclude-root subject)
        <tmp>/loose.txt            a NON-directory sibling — must be excluded
    """
    parent = tmp_path
    df_root = parent / "dark-factory"
    for d in ("dark-factory", "proj-a", "proj-a/nested", "target"):
        (parent / d).mkdir(parents=True, exist_ok=True)
    (parent / "loose.txt").write_text("not a directory\n")
    return parent, df_root


def test_module_loads_with_its_documented_surface(fep: types.ModuleType) -> None:
    """Non-vacuity: the loader really loaded THIS script, not an empty module.

    Asserts on the KEYS of RESERVED and on callability — never on the reserved
    ports' prose, which is documentation and free to be reworded.
    """
    for name in ("known_project_roots", "find_config", "escalation_port", "is_bound", "main"):
        assert callable(getattr(fep, name, None)), f"{name} missing or not callable"

    assert set(fep.RESERVED) == {8002, 8103}
    assert len(fep.CONFIG_NAMES) > 0
    assert fep.CONFIG_NAMES[0] == "dark-factory-orchestrator.yaml"
