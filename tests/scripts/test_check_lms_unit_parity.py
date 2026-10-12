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


def test_a_lost_present_only_directive_is_drift(tmp_path: pathlib.Path, capsys):
    """An installed copy that LOST ExecStop is drift, naming the missing side.

    This is the half of the UnitSpec design the value comparison structurally
    cannot reach. ExecStop embeds an absolute host path (/usr/bin/docker), so
    value-comparing it would fire on every machine whose docker lives
    elsewhere — a gate that is red everywhere gets switched off. Presence is
    still a real claim though: without ExecStop the arm's container keeps the
    GPU after the unit stops, with no unit left to show for it.

    The report must say which SIDE lost the directive. "ExecStop differs" sends
    the operator to diff two files; "absent from the installed copy" tells them
    the installed copy is the stale one.
    """
    checker = _load_checker()
    repo_root = _fake_repo(tmp_path)
    installed_dir = _installed_from(
        tmp_path,
        repo_root,
        text="\n".join(
            line
            for line in COMMITTED_TEMPLATE.splitlines()
            if not line.startswith("ExecStop=")
        ),
    )

    rc = checker.main(
        ["--repo-root", str(repo_root), "--installed-dir", str(installed_dir)],
        probe_runner=_clean_probe(repo_root),
    )

    out = capsys.readouterr().out
    assert rc == 1
    assert "[drift]" in out
    assert "ExecStop" in out
    assert "<absent>" in out
    assert "absent from the installed copy" in out
    assert "[ok] parity" not in out


def test_a_present_only_directive_the_committed_copy_lacks_is_drift(
    tmp_path: pathlib.Path, capsys
):
    """The SYMMETRIC case: the installed copy declares one the template does not.

    Comparison is symmetric on purpose. An installed-only directive is how a
    unit acquires behaviour that is in no committed file — the same class of
    invisibility as a drop-in, just written into the unit itself — so it must
    be reported, and reported as belonging to the INSTALLED side.
    """
    checker = _load_checker()
    without_documentation = "\n".join(
        line
        for line in COMMITTED_TEMPLATE.splitlines()
        if not line.startswith("Documentation=")
    )
    repo_root = _fake_repo(tmp_path, text=without_documentation)
    # The installed copy keeps the Documentation= line the template lost.
    installed_dir = _installed_from(tmp_path, repo_root, text=COMMITTED_TEMPLATE)

    rc = checker.main(
        ["--repo-root", str(repo_root), "--installed-dir", str(installed_dir)],
        probe_runner=_clean_probe(repo_root),
    )

    out = capsys.readouterr().out
    assert rc == 1
    assert "[drift]" in out
    assert "Documentation" in out
    assert "absent from the committed copy" in out
    assert "[ok] parity" not in out


def test_present_only_directives_with_different_values_are_not_drift(
    tmp_path: pathlib.Path, capsys
):
    """A differently-spelled host path is NOT drift. This is the whole point.

    Every present_only key embeds an absolute host path — the uv binary and the
    repo root in ExecStart, the docker binary in ExecStop, the PRD file:// URL
    in Documentation. Value-comparing them would report drift on every machine
    that is not this one, and a gate that fires unconditionally is a gate that
    gets switched off, taking the real findings with it.

    So this asserts the PROPERTY that distinguishes present_only from compared,
    which is the only thing making the two lists a design rather than two
    spellings of one list.
    """
    checker = _load_checker()
    repo_root = _fake_repo(tmp_path)
    other_host = (
        COMMITTED_TEMPLATE.replace(
            "ExecStart=/home/leo/.local/bin/uv", "ExecStart=/usr/local/bin/uv"
        )
        .replace("ExecStop=/usr/bin/docker", "ExecStop=/opt/bin/docker")
        .replace("Documentation=file:///home/leo", "Documentation=file:///srv")
    )
    installed_dir = _installed_from(tmp_path, repo_root, text=other_host)

    # Precondition: the fixture really did change all three values.
    assert other_host != COMMITTED_TEMPLATE
    for changed in ("/usr/local/bin/uv", "/opt/bin/docker", "file:///srv"):
        assert changed in other_host

    rc = checker.main(
        ["--repo-root", str(repo_root), "--installed-dir", str(installed_dir)],
        probe_runner=_clean_probe(repo_root),
    )

    out = capsys.readouterr().out
    assert rc == 0, out
    assert "[drift]" not in out
    assert "[ok] parity" in out


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


# ---------------------------------------------------------------------------
# The DROP-IN layer  (step-5 / step-6)
# ---------------------------------------------------------------------------
#
# These encode the task's core finding. In every test below the two unit files
# are BYTE-IDENTICAL, so the file layer above has nothing to say — which is
# exactly the silent pass that let an override survive reinstallation
# unreported.

# The real observed instance, verbatim in shape: a drop-in pinning
# WorkingDirectory at a worktree that has since landed and been deleted, so the
# arms would serve a frozen tree while everyone believed they ran merged main.
_WORKTREE_DROPIN = (
    "[Service]\nWorkingDirectory=/home/leo/src/dark-factory/.worktrees/3713\n"
)


def _plant_dropin(
    installed_dir: pathlib.Path, name: str, text: str
) -> pathlib.Path:
    """Write a drop-in under ``<installed-dir>/lms-arm@.service.d/``."""
    dropin_dir = installed_dir / f"{UNIT_NAME}.d"
    dropin_dir.mkdir(parents=True, exist_ok=True)
    dropin = dropin_dir / name
    dropin.write_text(text, encoding="utf-8")
    return dropin


def test_a_dropin_over_an_identical_unit_is_not_parity(
    tmp_path: pathlib.Path, capsys
):
    """BYTE-IDENTICAL unit files plus a drop-in must NOT report parity.

    This is the behavioural claim the whole task rests on. `systemctl --user
    edit` never modifies the unit file; it writes `<unit>.d/override.conf`,
    which systemd merges OVER the unit at load time. So every compared
    directive can match character for character while the EFFECTIVE
    WorkingDirectory is a frozen worktree — the exact silent pass the old
    file-presence self-verify gave.
    """
    checker = _load_checker()
    repo_root = _fake_repo(tmp_path)
    installed_dir = _installed_from(tmp_path, repo_root)
    dropin = _plant_dropin(installed_dir, "10-worktree-3713.conf", _WORKTREE_DROPIN)

    # Precondition: the file layer genuinely has nothing to report.
    assert (installed_dir / UNIT_NAME).read_bytes() == (
        repo_root / REPO_RELPATH
    ).read_bytes()

    rc = checker.main(
        ["--repo-root", str(repo_root), "--installed-dir", str(installed_dir)],
        probe_runner=_clean_probe(repo_root),
    )

    out = capsys.readouterr().out
    assert rc == 1
    assert "[override]" in out
    assert str(dropin) in out
    assert "[ok] parity" not in out


def test_every_applying_dropin_is_named_by_path(tmp_path: pathlib.Path, capsys):
    """EVERY applying drop-in is named, not just a count and not just the first.

    systemd merges all of them, so a report naming one leaves the operator
    fixing half the problem and re-running into the same red.
    """
    checker = _load_checker()
    repo_root = _fake_repo(tmp_path)
    installed_dir = _installed_from(tmp_path, repo_root)
    first = _plant_dropin(installed_dir, "10-worktree-3713.conf", _WORKTREE_DROPIN)
    second = _plant_dropin(
        installed_dir, "20-limits.conf", "[Service]\nTimeoutStartSec=60\n"
    )

    rc = checker.main(
        ["--repo-root", str(repo_root), "--installed-dir", str(installed_dir)],
        probe_runner=_clean_probe(repo_root),
    )

    out = capsys.readouterr().out
    assert rc == 1
    assert str(first) in out
    assert str(second) in out


def test_dropin_report_is_worded_apart_from_drift(tmp_path: pathlib.Path, capsys):
    """The [override] block is not phrased as a directive diff.

    Same wording discipline check_dashboard_unit_parity applies, for the same
    reason: an operator sent hunting for a directive diff that does not exist
    wastes the trip. The unit files matched; what could not be established is
    that the EFFECTIVE configuration matches. So the block must say that, and
    must name how to inspect the merged result.
    """
    checker = _load_checker()
    repo_root = _fake_repo(tmp_path)
    installed_dir = _installed_from(tmp_path, repo_root)
    _plant_dropin(installed_dir, "10-worktree-3713.conf", _WORKTREE_DROPIN)

    checker.main(
        ["--repo-root", str(repo_root), "--installed-dir", str(installed_dir)],
        probe_runner=_clean_probe(repo_root),
    )

    out = capsys.readouterr().out
    override_lines = [line for line in out.splitlines() if "[override]" in line]
    assert override_lines

    override_text = "\n".join(override_lines)
    assert "EFFECTIVE" in override_text
    assert "systemctl --user cat" in override_text
    # The unit files DID match, so the override block must not be phrased as a
    # directive difference between them.
    assert "[drift]" not in out


def test_a_dropin_is_never_removed_by_the_checker(tmp_path: pathlib.Path, capsys):
    """Reporting an override never deletes it. Fail loud, do not "fix".

    Task 3750's finding is that the observed drop-in was LOAD-BEARING while
    its worktree was unmerged: removing it then would have pointed every arm
    at a directory with no launcher. Removal has a correct owner already,
    scripts/remove-lms-arm-worktree-dropin.sh, which gates it behind
    preconditions a general-purpose parity checker has no business
    re-implementing.
    """
    checker = _load_checker()
    repo_root = _fake_repo(tmp_path)
    installed_dir = _installed_from(tmp_path, repo_root)
    dropin = _plant_dropin(installed_dir, "10-worktree-3713.conf", _WORKTREE_DROPIN)

    rc = checker.main(
        ["--repo-root", str(repo_root), "--installed-dir", str(installed_dir)],
        probe_runner=_clean_probe(repo_root),
    )
    capsys.readouterr()

    assert rc == 1
    assert dropin.is_file()
    assert dropin.read_text(encoding="utf-8") == _WORKTREE_DROPIN
    assert dropin.parent.is_dir()


# ---------------------------------------------------------------------------
# The EFFECTIVE-CONFIG layer  (step-7 / step-8)
# ---------------------------------------------------------------------------
#
# What the file layer structurally CANNOT see. Globbing <installed-dir>/
# lms-arm@.service.d/ reads one directory; systemd also merges drop-ins from
# /etc/systemd/user/ and /run/systemd/user/, so asking systemd is strictly
# stronger. Every test drives the probe through an INJECTED runner, so nothing
# here needs a live systemd user manager.

# The exact argv this checker must issue. Pinned as a literal rather than
# rebuilt from the checker's own constants: a test that derives the command
# from the code under test would follow it wherever it went, which is not a
# pin at all.
_EXPECTED_PROBE_ARGV = [
    "systemctl",
    "--user",
    "show",
    "-p",
    "WorkingDirectory",
    "-p",
    "DropInPaths",
    "lms-arm@probe.service",
]


def _recording_probe(returncode: int, stdout: str):
    """Return ``(runner, calls)`` — a probe runner that records its argv."""
    calls: list[list[str]] = []

    def _runner(argv):
        calls.append(list(argv))
        return returncode, stdout

    return _runner, calls


def test_probe_command_is_the_documented_one(tmp_path: pathlib.Path, capsys):
    """The probe argv is pinned EXACTLY, including the absence of --value.

    This deliberately diverges from scripts/remove-lms-arm-worktree-dropin.sh,
    which issues two separate `show -p <KEY> --value` calls. The combined
    no---value form yields self-labelling KEY=VALUE lines, is one subprocess
    instead of two, and cannot mis-attribute a value to the wrong key.

    Pinning it stops a later refactor from quietly adding --value — which
    strips the very keys the parser matches on, so the parse would silently
    yield nothing and the run would report "unverifiable" forever — or from
    narrowing to a single -p and losing a whole half of the check.
    """
    checker = _load_checker()
    repo_root = _fake_repo(tmp_path)
    installed_dir = _installed_from(tmp_path, repo_root)
    runner, calls = _recording_probe(
        0, f"WorkingDirectory={_declared_working_directory(repo_root)}\nDropInPaths=\n"
    )

    checker.main(
        ["--repo-root", str(repo_root), "--installed-dir", str(installed_dir)],
        probe_runner=runner,
    )
    capsys.readouterr()

    assert calls == [_EXPECTED_PROBE_ARGV]


def test_probe_instance_is_derived_from_the_template_name(
    tmp_path: pathlib.Path, capsys
):
    """The probed unit is an INSTANCE, not the bare `lms-arm@.service` template.

    `systemctl show` on a bare template does not resolve %i-dependent state the
    way an instance does. scripts/remove-lms-arm-worktree-dropin.sh already
    established the `${TEMPLATE}probe.service` spelling; reusing it means both
    tools name the same unit, so an operator reading either output sees the
    same vocabulary.
    """
    checker = _load_checker()
    repo_root = _fake_repo(tmp_path)
    installed_dir = _installed_from(tmp_path, repo_root)
    runner, calls = _recording_probe(
        0, f"WorkingDirectory={_declared_working_directory(repo_root)}\nDropInPaths=\n"
    )

    checker.main(
        ["--repo-root", str(repo_root), "--installed-dir", str(installed_dir)],
        probe_runner=runner,
    )
    capsys.readouterr()

    probed = calls[0][-1]
    assert probed != UNIT_NAME
    assert probed.startswith("lms-arm@")
    assert probed.endswith(".service")
    # An instance name has something between the '@' and the suffix.
    assert probed[len("lms-arm@") : -len(".service")]


def test_clean_effective_config_is_parity(tmp_path: pathlib.Path, capsys):
    """A clean probe is parity, and the success case is a POSITIVE claim.

    "No complaint" and "verified clean" look identical in a log until the
    checker breaks, at which point silence reads as success. So the clean path
    states what it resolved, naming the effective WorkingDirectory.
    """
    checker = _load_checker()
    repo_root = _fake_repo(tmp_path)
    installed_dir = _installed_from(tmp_path, repo_root)
    declared = _declared_working_directory(repo_root)

    rc = checker.main(
        ["--repo-root", str(repo_root), "--installed-dir", str(installed_dir)],
        probe_runner=lambda argv: (0, f"WorkingDirectory={declared}\nDropInPaths=\n"),
    )

    out = capsys.readouterr().out
    assert rc == 0
    assert "[ok] parity" in out
    assert declared in out
    assert "[effective]" in out


def test_dropinpaths_reported_by_systemd_fails_the_run(
    tmp_path: pathlib.Path, capsys
):
    """systemd naming a drop-in fails the run even with an EMPTY filesystem glob.

    This is the case the tmp_path file layer cannot reach: systemd also merges
    drop-ins from /etc/systemd/user/ and /run/systemd/user/, so DropInPaths is
    a superset of globbing one directory. A checker trusting only the glob
    would report parity on a unit systemd is actively overriding.
    """
    checker = _load_checker()
    repo_root = _fake_repo(tmp_path)
    installed_dir = _installed_from(tmp_path, repo_root)
    declared = _declared_working_directory(repo_root)
    elsewhere = "/home/leo/.config/systemd/user/lms-arm@.service.d/10-worktree-3713.conf"

    # Precondition: the filesystem layer sees nothing at all.
    assert not (installed_dir / f"{UNIT_NAME}.d").exists()

    rc = checker.main(
        ["--repo-root", str(repo_root), "--installed-dir", str(installed_dir)],
        probe_runner=lambda argv: (
            0,
            f"WorkingDirectory={declared}\nDropInPaths={elsewhere}\n",
        ),
    )

    out = capsys.readouterr().out
    assert rc == 1
    assert elsewhere in out
    assert "[ok] parity" not in out


def test_a_dropin_systemd_spells_differently_is_named_once(
    tmp_path: pathlib.Path, capsys
):
    """One file found by BOTH layers is reported ONCE, not once per spelling.

    The two layers spell paths differently by construction: find_dropins builds
    them from --installed-dir verbatim, systemd reports its own canonical path.
    So the same file arrives twice whenever the two spellings differ while
    denoting one file — a symlinked $HOME being the live case on any host whose
    home is a symlink.

    Listing it twice damages precisely the property the [override] block exists
    for: every applying drop-in, named exactly once, so the operator fixes all
    of them and can tell how many there are. A duplicate reads as two files.
    """
    checker = _load_checker()
    repo_root = _fake_repo(tmp_path)
    installed_dir = _installed_from(tmp_path, repo_root)
    dropin = _plant_dropin(installed_dir, "10-worktree-3713.conf", _WORKTREE_DROPIN)
    declared = _declared_working_directory(repo_root)

    # The SAME file, reached through a symlinked parent — a different string,
    # the same inode. Nothing lexical about it, so a normalisation that only
    # collapsed '//' and '/./' would still double-report.
    link = tmp_path / "installed-via-link"
    link.symlink_to(installed_dir, target_is_directory=True)
    systemd_spelling = str(link / f"{UNIT_NAME}.d" / dropin.name)
    assert systemd_spelling != str(dropin)
    assert pathlib.Path(systemd_spelling).resolve() == dropin.resolve()

    rc = checker.main(
        ["--repo-root", str(repo_root), "--installed-dir", str(installed_dir)],
        probe_runner=lambda argv: (
            0,
            f"WorkingDirectory={declared}\nDropInPaths={systemd_spelling}\n",
        ),
    )

    out = capsys.readouterr().out
    assert rc == 1
    assert "[override]" in out
    naming = [line for line in out.splitlines() if dropin.name in line]
    assert len(naming) == 1, out


def test_a_partial_probe_answer_is_unverifiable_not_parity(
    tmp_path: pathlib.Path, capsys
):
    """A reply carrying only WorkingDirectory answered HALF the question.

    DropInPaths absent from the reply is not "no drop-ins" — it is no answer.
    The two probed keys are two separate questions, and the drop-in half is the
    one this whole task exists for, so a run that got no answer to it must not
    print the affirmative "merges no drop-in" line. Treating a missing key as
    its clean default would be a green claim over a question never asked: this
    task's own bug, one level up.

    Both one-sided replies are exercised, because the two halves fail
    differently — the missing-DropInPaths one still has a MATCHING
    WorkingDirectory, so it is the shape that would otherwise reach parity.
    """
    checker = _load_checker()
    repo_root = _fake_repo(tmp_path)
    installed_dir = _installed_from(tmp_path, repo_root)
    declared = _declared_working_directory(repo_root)

    partials = {
        "WorkingDirectory only": f"WorkingDirectory={declared}\n",
        "DropInPaths only": "DropInPaths=\n",
    }

    for label, stdout in partials.items():
        rc = checker.main(
            ["--repo-root", str(repo_root), "--installed-dir", str(installed_dir)],
            probe_runner=lambda argv, _stdout=stdout: (0, _stdout),
        )
        out = capsys.readouterr().out
        assert rc == 1, label
        assert "could not verify" in out.lower(), label
        assert "[ok] parity" not in out, label
        assert "merges no drop-in" not in out, label


def test_effective_workingdirectory_elsewhere_fails_the_run(
    tmp_path: pathlib.Path, capsys
):
    """A resolved WorkingDirectory that is not the committed one fails, loudly.

    Both values reach the operator: "they differ" without them sends someone
    back to systemctl to work out what it actually resolved to.
    """
    checker = _load_checker()
    repo_root = _fake_repo(tmp_path)
    installed_dir = _installed_from(tmp_path, repo_root)
    declared = _declared_working_directory(repo_root)
    actual = "/home/leo/src/dark-factory/.worktrees/3713"

    rc = checker.main(
        ["--repo-root", str(repo_root), "--installed-dir", str(installed_dir)],
        probe_runner=lambda argv: (0, f"WorkingDirectory={actual}\nDropInPaths=\n"),
    )

    out = capsys.readouterr().out
    assert rc == 1
    assert actual in out
    assert declared in out
    assert "[ok] parity" not in out


def test_a_failed_probe_is_unverifiable_not_parity(tmp_path: pathlib.Path, capsys):
    """A probe that did not answer has established NOTHING. Never rc 0.

    No silent fail-soft: a checker that shrugs when systemctl is missing and
    reports parity would reproduce this task's own bug one level up — a green
    claim covering nothing. All four failure shapes are exercised, because they
    arrive by different routes and only the last is an exception.
    """
    checker = _load_checker()
    repo_root = _fake_repo(tmp_path)
    installed_dir = _installed_from(tmp_path, repo_root)

    def _missing_systemctl(argv):
        raise FileNotFoundError(2, "No such file or directory", "systemctl")

    failures = {
        "non-zero exit": lambda argv: (1, ""),
        # Exactly what the fake systemctl in scripts/tests/test_lms_ctl.py
        # returns for an unhandled subcommand, so this is not hypothetical.
        "empty stdout": lambda argv: (0, ""),
        "garbage stdout": lambda argv: (0, "Failed to get properties: no such unit\n"),
        "systemctl absent": _missing_systemctl,
    }

    for label, runner in failures.items():
        rc = checker.main(
            ["--repo-root", str(repo_root), "--installed-dir", str(installed_dir)],
            probe_runner=runner,
        )
        out = capsys.readouterr().out
        assert rc == 1, label
        assert "could not verify" in out.lower(), label
        assert "[ok] parity" not in out, label


def test_expected_workingdirectory_comes_from_the_committed_template_not_repo_root(
    tmp_path: pathlib.Path, capsys
):
    """The expected value is read from the TEMPLATE, never from --repo-root.

    This is the worktree case, and the single most likely place to get it
    wrong. The committed template hardcodes the main checkout, so running from
    .worktrees/3775 gives a --repo-root that is NOT the declared
    WorkingDirectory. Keying the assertion off --repo-root would report a false
    red on every worktree run — and a gate that is red every time gets switched
    off, taking the real drift with it.

    The claim this gate actually makes is "the committed template is the
    effective configuration", which is exactly the template-sourced comparison.
    """
    checker = _load_checker()
    repo_root = _fake_repo(tmp_path)
    installed_dir = _installed_from(tmp_path, repo_root)
    declared = _declared_working_directory(repo_root)

    # The fixture repo deliberately is NOT where the template says to run.
    assert str(repo_root) != declared

    rc = checker.main(
        ["--repo-root", str(repo_root), "--installed-dir", str(installed_dir)],
        probe_runner=lambda argv: (0, f"WorkingDirectory={declared}\nDropInPaths=\n"),
    )

    out = capsys.readouterr().out
    assert rc == 0, out
    assert "[ok] parity" in out

    # And the converse: matching --repo-root instead of the template is NOT
    # parity, which is what pins the value's source rather than its shape.
    rc_repo_root = checker.main(
        ["--repo-root", str(repo_root), "--installed-dir", str(installed_dir)],
        probe_runner=lambda argv: (
            0,
            f"WorkingDirectory={repo_root}\nDropInPaths=\n",
        ),
    )
    capsys.readouterr()
    assert rc_repo_root == 1
