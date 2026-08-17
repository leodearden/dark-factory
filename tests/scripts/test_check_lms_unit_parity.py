"""Tests for scripts/check_lms_unit_parity.py.

Every test runs against tmp_path trees and an INJECTED effective-config probe
— NEVER the host's real ~/.config/systemd/user/ and never a real systemd user
manager — mirroring the rule tests/scripts/test_check_dashboard_unit_parity.py
and tests/scripts/test_check_orchestrator_unit_parity.py each state in their
own docstrings.

That rule is load-bearing here for a sharper reason than portability. This
checker's whole subject is a drop-in that survives reinstallation, and the
instance that prompted it (~/.config/systemd/user/lms-arm@.service.d/
10-worktree-3713.conf, pinning WorkingDirectory at a landed worktree) has
since been removed from this host by scripts/remove-lms-arm-worktree-dropin.sh.
A test asserting the live host has that drop-in would have been red the moment
the removal ran; one asserting it does not would flip red the next time an
operator runs `systemctl --user edit`. Either encodes host state rather than
checker behaviour, so every drop-in here is built under tmp_path.

The only real-tree read is REPO-side: the committed
scripts/local-model-serving/lms-arm@.service, copied into each fixture repo so
the compared values are the ones actually shipped rather than a restatement
that could drift away from them.

Module loading: scripts/ is not a package, so the checker is loaded via
importlib.util.spec_from_file_location, mirroring
tests/scripts/test_check_dashboard_unit_parity.py::_load_checker.
"""

import importlib.util
import pathlib
import types

REPO_ROOT = pathlib.Path(__file__).parents[2]
CHECKER_PATH = REPO_ROOT / "scripts" / "check_lms_unit_parity.py"

UNIT_NAME = "lms-arm@.service"
REPO_RELPATH = "scripts/local-model-serving/lms-arm@.service"

# The committed template, read once. Fixtures copy THIS rather than restating
# it: a restated fixture that drifted away from the shipped unit would make
# every assertion below a claim about the fixture instead of about the unit.
COMMITTED_TEMPLATE = (REPO_ROOT / REPO_RELPATH).read_text(encoding="utf-8")


def _load_checker() -> types.ModuleType:
    """Load scripts/check_lms_unit_parity.py by file path."""
    spec = importlib.util.spec_from_file_location("check_lms_unit_parity", CHECKER_PATH)
    assert spec is not None, f"Could not build spec from {CHECKER_PATH}"
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_repo(tmp_path: pathlib.Path, *, text: str | None = None) -> pathlib.Path:
    """Build a fake repo root holding the committed lms-arm@ template."""
    repo_root = tmp_path / "repo"
    unit_path = repo_root / REPO_RELPATH
    unit_path.parent.mkdir(parents=True, exist_ok=True)
    unit_path.write_text(
        COMMITTED_TEMPLATE if text is None else text, encoding="utf-8"
    )
    return repo_root


def _installed_from(
    tmp_path: pathlib.Path,
    repo_root: pathlib.Path,
    *,
    text: str | None = None,
) -> pathlib.Path:
    """Build an installed dir holding a copy of *repo_root*'s template."""
    installed_dir = tmp_path / "installed"
    installed_dir.mkdir(parents=True, exist_ok=True)
    source = (repo_root / REPO_RELPATH).read_text(encoding="utf-8")
    (installed_dir / UNIT_NAME).write_text(
        source if text is None else text, encoding="utf-8"
    )
    return installed_dir


def _declared_working_directory(repo_root: pathlib.Path) -> str:
    """Return the ``[Service] WorkingDirectory`` the fixture template declares.

    Read from the fixture rather than hardcoded, so a clean probe answer is
    clean BY CONSTRUCTION: a literal here could drift away from the committed
    template and fabricate agreement the checker would then believe.
    """
    for line in (repo_root / REPO_RELPATH).read_text(encoding="utf-8").splitlines():
        if line.startswith("WorkingDirectory="):
            return line.partition("=")[2].strip()
    raise AssertionError("fixture template declares no WorkingDirectory")


def _clean_probe(repo_root: pathlib.Path):
    """A probe runner reporting the template as the effective configuration."""
    working_directory = _declared_working_directory(repo_root)

    def _runner(argv):
        return 0, f"WorkingDirectory={working_directory}\nDropInPaths=\n"

    return _runner


# ---------------------------------------------------------------------------
# The FILE layer  (step-3 / step-4)
# ---------------------------------------------------------------------------
#
# Every test in this section injects a CLEAN probe, so what it exercises is
# purely the repo-vs-installed file comparison. The effective-config layer has
# its own section below.


def test_identical_copies_are_parity(tmp_path: pathlib.Path, capsys):
    """Byte-identical repo and installed copies, no drop-in => parity."""
    checker = _load_checker()
    repo_root = _fake_repo(tmp_path)
    installed_dir = _installed_from(tmp_path, repo_root)

    rc = checker.main(
        ["--repo-root", str(repo_root), "--installed-dir", str(installed_dir)],
        probe_runner=_clean_probe(repo_root),
    )

    out = capsys.readouterr().out
    assert rc == 0
    assert "[ok] parity" in out


def test_a_drifted_timeout_is_reported_with_both_values(
    tmp_path: pathlib.Path, capsys
):
    """A drifted TimeoutStopSec is drift, and BOTH values reach the operator.

    TimeoutStopSec is value-compared, not presence-only, precisely so this
    case fires: a presence-only check would wave it through, since the key is
    declared on both sides. Reporting only "they differ" without the values
    sends the operator back to the two files to work out what changed.
    """
    checker = _load_checker()
    repo_root = _fake_repo(tmp_path)
    installed_dir = _installed_from(
        tmp_path,
        repo_root,
        text=COMMITTED_TEMPLATE.replace("TimeoutStopSec=120", "TimeoutStopSec=90"),
    )

    rc = checker.main(
        ["--repo-root", str(repo_root), "--installed-dir", str(installed_dir)],
        probe_runner=_clean_probe(repo_root),
    )

    out = capsys.readouterr().out
    assert rc == 1
    assert "[drift]" in out
    assert "TimeoutStopSec" in out
    assert "120" in out
    assert "90" in out
    assert "[ok] parity" not in out


def test_an_absent_installed_unit_is_the_benign_skip(
    tmp_path: pathlib.Path, capsys
):
    """Not installed on this host => exit 2 and a [skip] naming the path.

    Exit 2 is the benign branch setup-host.sh's vocabulary depends on: most
    hosts never install the arms, and warning there would train operators to
    ignore the gate.
    """
    checker = _load_checker()
    repo_root = _fake_repo(tmp_path)
    installed_dir = tmp_path / "installed"
    installed_dir.mkdir()

    rc = checker.main(
        ["--repo-root", str(repo_root), "--installed-dir", str(installed_dir)],
        probe_runner=_clean_probe(repo_root),
    )

    out = capsys.readouterr().out
    assert rc == 2
    assert "[skip]" in out
    assert str(installed_dir / UNIT_NAME) in out
    assert "[ok] parity" not in out


def test_a_vanished_committed_unit_is_never_parity(
    tmp_path: pathlib.Path, capsys
):
    """No committed template => [vanished] and exit 1, never 0.

    The committed template is this checker's source of truth. Without it
    nothing was compared, and a run that compared nothing must never claim
    parity — that would be a green report covering exactly zero verification.
    """
    checker = _load_checker()
    repo_root = tmp_path / "repo"
    (repo_root / REPO_RELPATH).parent.mkdir(parents=True)
    installed_dir = tmp_path / "installed"
    installed_dir.mkdir()
    (installed_dir / UNIT_NAME).write_text(COMMITTED_TEMPLATE, encoding="utf-8")

    def _runner(argv):
        raise AssertionError("probe must not run when there is nothing to compare")

    rc = checker.main(
        ["--repo-root", str(repo_root), "--installed-dir", str(installed_dir)],
        probe_runner=_runner,
    )

    out = capsys.readouterr().out
    assert rc == 1
    assert "[vanished]" in out
    assert str(repo_root / REPO_RELPATH) in out
    assert "[ok] parity" not in out


def test_a_run_that_compares_nothing_never_reports_parity(
    tmp_path: pathlib.Path, capsys
):
    """Neither zero-compare path may print the success line.

    Both ways of comparing nothing — the committed copy gone, and the
    installed copy absent — are exercised together here because the invariant
    is one invariant: the success line is a claim about units this run
    actually read, so a run that read none may not print it under any exit
    code.
    """
    checker = _load_checker()

    # Nothing on either side at all.
    repo_root = tmp_path / "repo"
    (repo_root / REPO_RELPATH).parent.mkdir(parents=True)
    installed_dir = tmp_path / "installed"
    installed_dir.mkdir()

    rc = checker.main(
        ["--repo-root", str(repo_root), "--installed-dir", str(installed_dir)],
        probe_runner=_clean_probe(_fake_repo(tmp_path / "reference")),
    )

    out = capsys.readouterr().out
    assert rc != 0
    assert "[ok] parity" not in out
