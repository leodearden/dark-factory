#!/usr/bin/env python3
"""Verify the installed lms-arm@ unit against its committed template.

The third ``check_*_unit_parity.py`` sibling, and the first that asks systemd
what the EFFECTIVE configuration actually is rather than only comparing two
files.

Why the effective configuration, not file presence
--------------------------------------------------
``scripts/local-model-serving/install-lms-units.sh`` used to `cp` the template,
`daemon-reload`, and then self-verify by checking the file was THERE. That is
blind to the mechanism that actually redirects a unit: ``systemctl --user edit``
never modifies the unit file, it writes ``<unit>.d/override.conf`` beside it,
and systemd merges that over the unit at load time. The observed instance was
``~/.config/systemd/user/lms-arm@.service.d/10-worktree-3713.conf`` pinning
``WorkingDirectory`` at a worktree — so every byte of the unit file matched the
committed template while the arms served a frozen tree. Reinstalling did not
clear it and nothing reported it. This checker closes both halves: it compares
the files AND asks systemd what will really be applied.

Exit codes
----------
The family's codes are defined once, in
``systemd_unit_parity.py::ParityExit``. This checker's probe adds two findings:
``[effective]`` means systemd resolves the unit to a WorkingDirectory other
than the committed template's, and ``[unverifiable]`` means the probe could not
get an answer from systemd. A
run that could not ask systemd has established nothing. Not installed is the
common case, because most hosts never install the arms.

Read-only, and NO --fix
-----------------------
This checker never removes a drop-in, and offers no flag that would. Task
3750's finding is that the observed drop-in was LOAD-BEARING while its worktree
was unmerged — removing it then would have pointed every arm at a directory
with no launcher. Removal already has a correct owner,
``scripts/remove-lms-arm-worktree-dropin.sh``, which gates it behind three
preconditions (launcher present on the merged path, template still installed,
launcher compiles) that a general-purpose parity checker has no business
re-implementing. Fail loud, name every drop-in by path, and leave it alone.

Why this file is at top-level scripts/ and NOT under scripts/local-model-serving/
--------------------------------------------------------------------------------
Deliberate, and enforced from both sides by
``scripts/tests/test_lms_marker_contract.py``. That suite asserts (a) every
git-tracked file under ``scripts/local-model-serving/`` carries task 3713's
serving marker, and (b) that the delivered-check grep for that marker finds
EXACTLY the files under ``scripts/local-model-serving/`` plus
``scripts/tests/test_lms_*.py`` — nothing else. So a module placed in the
serving directory would need a marker this task is forbidden to write, and a
marked module outside it would break (b). Top-level placement satisfies both,
and is independently where this belongs: beside its ``check_*_unit_parity.py``
siblings, which is the family this module is a member of. Do NOT "tidy" it into
the serving directory, and do not write that marker literal here — either move
turns that suite red.

Testing note
------------
tests/scripts/test_check_lms_unit_parity.py drives every path against tmp_path
trees and an INJECTED probe runner. Nothing here may be tested against the
host's real ~/.config/systemd/user/ or a live systemd user manager: the drop-in
this checker exists to catch has already been removed from this host, so a test
keyed on live host state would encode host state rather than checker behaviour.
"""

import argparse
import dataclasses
import functools
import pathlib
import subprocess
import sys
from collections.abc import Sequence

# The parser, the drift record and the parity driver live in
# scripts/systemd_unit_parity.py, imported by bare name, which resolves
# under CLI (scripts/ lands at sys.path[0]), under pytest (tests/scripts/
# conftest.py inserts scripts/ explicitly, load-bearing because pyproject sets
# --import-mode=importlib) and under pyright ([tool.pyright] extraPaths).
#
# What stays per-checker, and why: plans/review-all-s25-scripts-instruments-prd.md
# (finding fk-3042baa5983e).
from systemd_unit_parity import (
    ABSENT,
    BlockText,
    Drift,
    ParityRun,
    Probe,
    ProbeFinding,
    ProbeOutcome,
    ReportText,
    UnitRef,
    parse_unit_directives,
    run_parity,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Prefixed onto every line this script prints, matching
# [dashboard_unit_parity] and [orchestrator_unit_parity]. setup-host.sh routes
# operators to a detailed report BY TAG rather than by position, so an untagged
# line in a long bring-up run has no reliable way to point at its own output.
LOG_TAG = "lms_unit_parity"

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_DEFAULT_INSTALLED_DIR = pathlib.Path.home() / ".config" / "systemd" / "user"

# The instance suffix probed in place of the bare template. `systemctl show`
# on `lms-arm@.service` does not resolve %i-dependent state the way an
# instance does, so the probe names one. The spelling matches
# scripts/remove-lms-arm-worktree-dropin.sh's PROBE="${TEMPLATE}probe.service",
# so both tools name the same unit and an operator reading either output sees
# identical vocabulary.
_PROBE_INSTANCE = "probe"

# The two properties the probe asks for, in the order the argv spells them.
_PROBE_KEYS = ("WorkingDirectory", "DropInPaths")

# Bounded so a wedged systemd user manager cannot hang an install or a
# host bring-up run. A timeout lands on the UNVERIFIABLE path, which is the
# honest answer: the probe did not come back.
_PROBE_TIMEOUT_SECONDS = 20


# ---------------------------------------------------------------------------
# Drift records and unit specs
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class UnitSpec:
    """What to compare for one unit, and where its committed copy lives.

    Expected VALUES are never stated here — they are read from the committed
    template at run time. Only the KEY registry is curated. Restating the
    values would create a THIRD site that must agree with the committed unit
    and the installed one, reintroducing — in the tool built to close it —
    exactly the lockstep duplication this check exists to catch.
    """

    name: str
    repo_relpath: str
    # (section, key) pairs whose VALUES must agree. Host-INVARIANT literals
    # only — anything carrying a host path belongs on present_only.
    compared: tuple[tuple[str, str], ...] = ()
    # (section, key) pairs whose PRESENCE must agree but whose values cannot
    # be compared, because they embed an absolute host path. Value-comparing
    # them would report drift on every machine that is not this one, and a
    # gate that fires unconditionally is a gate that gets switched off.
    present_only: tuple[tuple[str, str], ...] = ()


# ---------------------------------------------------------------------------
# The unit registry
# ---------------------------------------------------------------------------

# One unit, one entry, and every key names WHY it is on its list.
#
# Description= and After= are deliberately absent from both lists: they are
# cosmetic here, and comparing them would spend the gate's credibility on
# nothing — the same rule check_dashboard_unit_parity.UNITS states.
UNITS: dict[str, UnitSpec] = {
    "lms-arm@.service": UnitSpec(
        name="lms-arm@.service",
        repo_relpath="scripts/local-model-serving/lms-arm@.service",
        compared=(
            # Type=exec means systemd considers the unit started only once the
            # binary has actually been executed. A drift to `simple` would make
            # a failed arm look started.
            ("Service", "Type"),
            # Restart=no is a deliberate safety property the template argues
            # for at length: a restart policy on a GPU-holding unit thrashes
            # the shared 3090 when an arm OOMs, starving whisper-writer and
            # corrupting any concurrent arm's latency measurements. A drifted
            # value here is a real regression, not cosmetic.
            ("Service", "Restart"),
            # ~14 GB of AWQ weights plus CUDA graph capture is minutes, not
            # seconds. A shortened start timeout kills an arm that was loading
            # CORRECTLY and reports it as a startup failure.
            ("Service", "TimeoutStartSec"),
            # The stop side matters for the same GPU reason: a too-short stop
            # timeout SIGKILLs the launcher before `docker stop` returns, and
            # the container keeps the card with no unit to show for it.
            ("Service", "TimeoutStopSec"),
            # Both streams go to the journal; that IS the arm's observability.
            # A drift to `null` makes a failing arm silent.
            ("Service", "StandardOutput"),
            ("Service", "StandardError"),
            # Dropping [Install] would leave the unit un-enableable while
            # looking like a working unit file.
            ("Install", "WantedBy"),
        ),
        present_only=(
            # Every one of these embeds an absolute host path: the uv binary
            # and the repo root in ExecStart, the checkout in
            # WorkingDirectory, the docker binary in ExecStop, and the PRD
            # file:// URL in Documentation. Presence is still asserted, so a
            # copy that LOST one is reported.
            ("Service", "ExecStart"),
            ("Service", "WorkingDirectory"),
            ("Service", "ExecStop"),
            ("Unit", "Documentation"),
        ),
    ),
}


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def compare_unit(spec: UnitSpec, repo_text: str, installed_text: str) -> list[Drift]:
    """Return every drift between the committed and installed copies of *spec*.

    Comparison is SYMMETRIC over the curated key set: a compared directive the
    installed copy declares and the committed one does not is drift just as
    much as the reverse. A missing key is treated as ``ABSENT`` on its side,
    so both asymmetric cases fall out of the same equality test.

    The full values LIST is compared, not just the first value: systemd
    applies every occurrence, so the checker must see every occurrence.

    ``spec.present_only`` keys are checked for PRESENCE only — a drift is
    emitted when one copy declares the directive and the other does not, never
    on a value difference. See UnitSpec.present_only for why.
    """
    repo = parse_unit_directives(repo_text)
    installed = parse_unit_directives(installed_text)
    drifts: list[Drift] = []

    for section, key in spec.compared:
        repo_values = repo.get(section, {}).get(key)
        installed_values = installed.get(section, {}).get(key)
        if repo_values == installed_values:
            continue
        drifts.append(
            Drift(
                unit=spec.name,
                section=section,
                key=key,
                repo_value=", ".join(repo_values) if repo_values else ABSENT,
                installed_value=(
                    ", ".join(installed_values) if installed_values else ABSENT
                ),
                reason="value differs between the committed and installed copies",
            )
        )

    for section, key in spec.present_only:
        in_repo = key in repo.get(section, {})
        in_installed = key in installed.get(section, {})
        if in_repo == in_installed:
            continue
        drifts.append(
            Drift(
                unit=spec.name,
                section=section,
                key=key,
                repo_value="<declared>" if in_repo else ABSENT,
                installed_value="<declared>" if in_installed else ABSENT,
                reason=(
                    f"{key} declared in the installed copy, absent from the "
                    "committed copy"
                    if in_installed
                    else f"{key} declared in the committed copy, absent from "
                    "the installed copy"
                ),
            )
        )

    return drifts


# ---------------------------------------------------------------------------
# The effective configuration
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class EffectiveConfig:
    """What systemd says it would actually apply, or why we could not tell."""

    # None when the probe did not answer usably. Distinct from "" (a key
    # systemd reported as empty, which is the CLEAN answer for DropInPaths).
    working_directory: str | None
    dropin_paths: tuple[str, ...]
    # Set iff the probe established nothing. Rendered verbatim to the operator.
    unverifiable: str | None = None


def probe_instance_name(template_name: str) -> str:
    """Return the instance to probe for *template_name*.

    ``lms-arm@.service`` -> ``lms-arm@probe.service``. Built from the template
    rather than hardcoded so a renamed unit takes its probe with it.
    """
    stem, _, suffix = template_name.rpartition(".")
    return f"{stem}{_PROBE_INSTANCE}.{suffix}"


def probe_argv(template_name: str) -> list[str]:
    """Return the exact ``systemctl show`` argv this checker issues.

    Deliberately the SINGLE combined form, and deliberately WITHOUT
    ``--value``. scripts/remove-lms-arm-worktree-dropin.sh issues two separate
    ``show -p <KEY> --value`` calls; this diverges because the combined
    no-``--value`` form yields self-labelling ``KEY=VALUE`` lines, is one
    subprocess instead of two, and cannot mis-attribute a value to the wrong
    key. Adding ``--value`` would strip the very keys the parser matches on, so
    every run would parse nothing and report "unverifiable" forever — which is
    why the argv is pinned by a test.
    """
    argv = ["systemctl", "--user", "show"]
    for key in _PROBE_KEYS:
        argv += ["-p", key]
    argv.append(probe_instance_name(template_name))
    return argv


def _default_probe_runner(argv: Sequence[str]) -> tuple[int, str]:
    """Run *argv* and return ``(returncode, stdout)``.

    ``check=False`` because a non-zero exit is data here, not an exception:
    it routes to the UNVERIFIABLE report rather than a traceback.
    """
    completed = subprocess.run(
        list(argv),
        capture_output=True,
        text=True,
        check=False,
        timeout=_PROBE_TIMEOUT_SECONDS,
    )
    return completed.returncode, completed.stdout


def read_effective_config(template_name: str, probe_runner) -> EffectiveConfig:
    """Ask systemd what it would actually apply for *template_name*.

    Every failure shape lands on ``unverifiable`` rather than on a plausible
    default: a probe that did not run has established nothing, and a checker
    that shrugged and reported parity would reproduce this task's own bug one
    level up — a green claim covering nothing.
    """
    argv = probe_argv(template_name)
    rendered = " ".join(argv)
    try:
        returncode, stdout = probe_runner(argv)
    except FileNotFoundError:
        # systemctl absent — a container, or a host with no systemd at all.
        return EffectiveConfig(None, (), f"`{rendered}` failed: systemctl not found")
    except subprocess.TimeoutExpired:
        return EffectiveConfig(None, (), f"`{rendered}` timed out")
    except OSError as exc:
        return EffectiveConfig(None, (), f"`{rendered}` failed: {exc}")

    if returncode != 0:
        return EffectiveConfig(None, (), f"`{rendered}` exited {returncode}")

    working_directory: str | None = None
    dropin_paths: tuple[str, ...] = ()
    # Tracked PER KEY, not as one "saw something" flag. Each probed key is a
    # separate question, and an answer to one establishes nothing about the
    # other: DropInPaths defaults to (), which is indistinguishable from
    # systemd's CLEAN answer, so a reply carrying only WorkingDirectory would
    # otherwise parse "successfully" and — on a matching WorkingDirectory —
    # print the affirmative "merges no drop-in" line for a question systemd
    # never answered. That is a green claim over an unasked half, which is this
    # task's own bug one level up. Reachable in practice: a systemd version
    # that does not expose one property, truncated stdout, or a later argv edit
    # that narrows to a single -p (test_probe_command_is_the_documented_one
    # pins the argv, but a pin on the SEND side cannot vouch for the reply).
    seen: set[str] = set()
    for line in stdout.splitlines():
        key, sep, value = line.partition("=")
        key = key.strip()
        if not sep or key not in _PROBE_KEYS:
            continue
        seen.add(key)
        if key == "WorkingDirectory":
            working_directory = value.strip()
        else:
            # systemd renders DropInPaths as a space-separated list, empty when
            # nothing is merged.
            dropin_paths = tuple(value.split())

    unanswered = [key for key in _PROBE_KEYS if key not in seen]
    if unanswered:
        # Includes the empty-stdout case (exactly what an unhandled subcommand
        # returns from the fake systemctl in the installer tests) and every
        # PARTIAL answer. Both are UNVERIFIABLE, never a default.
        return EffectiveConfig(
            None, (), f"`{rendered}` did not report {' or '.join(unanswered)}"
        )

    return EffectiveConfig(working_directory, dropin_paths)


EFFECTIVE = BlockText(
    "effective",
    "{count} unit(s) resolve to a WorkingDirectory that is NOT the one the committed "
    "template declares:",
    (
        "An arm started now would run from the resolved path, not the committed one. "
        "Inspect with: systemctl --user cat lms-arm@<arm>.service",
    ),
)

UNVERIFIABLE = BlockText(
    "unverifiable",
    "could not verify the effective configuration for {count} unit(s):",
    (
        "The unit files may well match; what this run could NOT establish is what "
        "systemd would actually apply. Check that a systemd user manager is running "
        "for this user (systemctl --user is-system-running).",
    ),
)


def _effective_outcome(probe_runner, unit: UnitRef, repo_text: str) -> ProbeOutcome:
    """Ask systemd what it would apply for *unit* and judge it against the committed text.

    EXPECTED comes from the committed template's own WorkingDirectory, never
    from --repo-root: the template hardcodes the main checkout, so keying off
    --repo-root would false-red on every worktree run.
    """
    effective = read_effective_config(unit.name, probe_runner)
    if effective.unverifiable is not None:
        return ProbeOutcome(
            findings=(ProbeFinding(UNVERIFIABLE, (f"  {unit.name}: {effective.unverifiable}",)),)
        )
    dropins = tuple(pathlib.Path(path) for path in effective.dropin_paths)
    declared = parse_unit_directives(repo_text).get("Service", {}).get("WorkingDirectory")
    expected = declared[0] if declared else None
    if expected is None:
        reason = "the committed template declares no [Service] WorkingDirectory"
        finding = ProbeFinding(UNVERIFIABLE, (f"  {unit.name}: {reason}",))
        return ProbeOutcome(findings=(finding,), dropins=dropins)
    if effective.working_directory != expected:
        finding = ProbeFinding(
            EFFECTIVE,
            (
                f"  {unit.name}",
                f"      committed template: {expected}",
                f"      systemd resolved:   {effective.working_directory or ABSENT}",
            ),
        )
        return ProbeOutcome(findings=(finding,), dropins=dropins)
    if dropins:
        return ProbeOutcome(dropins=dropins)
    # An affirmative claim, not the absence of a complaint: silence would read
    # as success the day the probe broke.
    return ProbeOutcome(
        confirmation=(
            f"[effective] {unit.name}: the committed template IS the effective "
            f"configuration — systemd resolves WorkingDirectory={expected} and merges "
            "no drop-in."
        )
    )


def _compare(unit: UnitRef, repo_text: str, installed_text: str) -> list[Drift]:
    return compare_unit(UNITS[unit.name], repo_text, installed_text)


_REPORT = ReportText(
    drift=BlockText(
        "drift",
        "{count} directive(s) differ between the committed and installed units:",
        (
            "To propagate the committed template to this host, run: "
            "scripts/local-model-serving/install-lms-units.sh  (this checker is "
            "read-only by design — see the module docstring for why there is no --fix)",
        ),
    ),
    vanished=dataclasses.replace(
        ReportText.vanished,
        tail=(
            "The committed template is this checker's source of truth. Check "
            "--repo-root, and whether the unit was renamed or moved (the paths live "
            "in UNITS in this script).",
        ),
    ),
    override=BlockText(
        "override",
        "{count} unit(s) have drop-in overrides — the unit files were compared, but "
        "the EFFECTIVE configuration was NOT verified by that comparison:",
        (
            "systemd merges these over the unit at load time, so a directive set here "
            "silently wins over the committed value. Inspect the merged result with: "
            "systemctl --user cat lms-arm@<arm>.service",
            "Nothing was removed: this checker is read-only BY DESIGN, because a "
            "drop-in can be load-bearing (task 3750). To remove the known worktree "
            "drop-in, run scripts/remove-lms-arm-worktree-dropin.sh, which gates "
            "removal behind the preconditions that make it safe.",
        ),
    ),
    repo_label="committed",
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str], *, probe_runner=None) -> int:
    """Parse args and run the parity check.

    Exit codes: see ``systemd_unit_parity.ParityExit``.

    *probe_runner* is the injection seam for the effective-config probe: a
    callable taking an argv list and returning ``(returncode, stdout)``. Tests
    pass a stub so no test ever needs a live systemd user manager.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Verify parity between the committed and installed lms-arm@ units, "
            "including the effective configuration systemd would apply."
        )
    )
    parser.add_argument(
        "--installed-dir",
        type=pathlib.Path,
        default=_DEFAULT_INSTALLED_DIR,
        help="Directory holding the installed units (default: %(default)s)",
    )
    parser.add_argument(
        "--repo-root",
        type=pathlib.Path,
        default=_REPO_ROOT,
        help="Repo root holding the committed units (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    probe = Probe(
        (EFFECTIVE, UNVERIFIABLE),
        functools.partial(_effective_outcome, probe_runner or _default_probe_runner),
    )
    units = [
        UnitRef(name, args.repo_root / UNITS[name].repo_relpath, args.installed_dir / name)
        for name in sorted(UNITS)
    ]
    return run_parity(ParityRun(LOG_TAG, units, _compare, _REPORT, (probe,)), out=sys.stdout)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
