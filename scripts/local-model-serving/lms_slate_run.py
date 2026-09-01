"""Run the whole arm slate, one arm at a time, in a transient systemd --user unit.

PRD-MARKER:local-memory-models-eval serving

Task 4301 of `plans/local-memory-models-eval-prd.md`.

WHAT THIS IS.  The committed, reproducible driver for a full slate run: it
sweeps every arm the manifest commissions, writes one per-arm report part,
and then asks `lms_healthcheck --merge` to assemble
`verification/health-report.json` from them.

WHY IT EXISTS.  The 2026-08-06 slate run was driven BY HAND -- a person typed
the `lms_ctl` and `lms_healthcheck` invocations one arm at a time.  So no
compliant invocation for that run exists anywhere in the repo, and every
future re-measure re-derives the hazard-compliant form from scratch.  A
re-derivation is where `--collect` and `--working-directory` get dropped, and
where an operator reaches for `nohup ... &` because it is shorter.  A script
makes those errors impossible rather than merely fixed.

PRD hazard 11 is binding: "long runs in transient `systemd --user` units,
never bare background shells".  A full slate is ~30 minutes of model loads;
through a background shell it is unsupervised, unloggable, and dies with the
invoking session -- losing every arm measured so far.

TWO LAYERS, and the split is the point:

  * SUBMIT (the default).  `slate_argv` is a PURE function returning the
    transient-unit argv; `_submit` prints it and runs it.  Being pure, the
    compliant form is fully assertable offline -- the one property a live
    30-minute run can never conveniently prove.
  * IN-UNIT (`--in-unit`).  `run_slate` performs the sweep.  The flag is also
    the guard that stops a unit from recursively re-submitting itself.

The driver SHELLS OUT to `lms_ctl.py` and `lms_healthcheck.py` rather than
importing them.  Three reasons.  (1) `lms_ctl.start`'s LIBRARY default is
`exclusive=False` while its CLI default is `exclusive=True`; an import-based
driver one forgotten keyword from starting a second arm on a card that fits
one is exactly the hazard `start` exists to refuse.  Shelling out inherits the
safe default by construction.  (2) The driver then runs the same commands the
README documents an operator running by hand, so the script and the prose
cannot describe two different procedures.  (3) A wedged HTTP client or a
180s completion timeout stays in its own process instead of hanging the sweep.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import lms_fetch_weights
import lms_vram
from lms_ctl import DEFAULT_READY_TIMEOUT_S, EXIT_MANIFEST_ERROR
from lms_healthcheck import HealthReport
from lms_manifest import load_arms
from lms_serve import REPO_ROOT

#: The transient unit the whole sweep runs in.  Quoted verbatim by the
#: README's `journalctl --user -u lms-slate-run -f` follow line, so a drift
#: here sends an operator to a unit that does not exist.
SLATE_UNIT_NAME = 'lms-slate-run'

#: This file, absolute.  The payload re-invokes it by absolute path: the unit
#: runs with a minimal PATH and none of the caller's venv, so neither a
#: relative path nor a bare module name resolves inside it.
MODULE_PATH = Path(__file__).resolve()

#: The two sibling CLIs the sweep drives, by absolute path for the same reason
#: the payload names `sys.executable`: the unit inherits a minimal PATH and
#: none of the caller's venv.
CTL_PATH = MODULE_PATH.parent / 'lms_ctl.py'
HEALTHCHECK_PATH = MODULE_PATH.parent / 'lms_healthcheck.py'


class CompletedLike(Protocol):
    """The one thing this module reads off a finished subprocess."""

    @property
    def returncode(self) -> int: ...


#: The subprocess seam.  Narrowed to "takes an argv, has a returncode" rather
#: than pinned to `subprocess.run` itself, so a test can inject a recorder and
#: keep the whole suite offline: no arm started, no card touched.
Runner = Callable[[list[str]], CompletedLike]

#: The caller variables that reach the unit, as an ALLOWLIST rather than a copy
#: of `os.environ`.
#:
#: `LMS_BASELINE_DIR` earns its place: `lms_ctl start` writes the VRAM baseline
#: through `lms_vram.baseline_dir()`, which reads this variable, and
#: `lms_healthcheck` reads the baseline back through the same function.  If the
#: two disagree the healthcheck exits 8 (`EXIT_STALE_BASELINE`) and writes no
#: file at all -- so a ~30 minute sweep produces nothing, and the cause is a
#: variable nobody mentioned.  Pinned to `lms_vram`'s own constant so a rename
#: there cannot leave this driver propagating a dead name.
#:
#: A whitelist and not a blanket copy, because `systemd-run --setenv` records
#: the value in the unit's systemd properties and the journal: copying
#: `os.environ` would push `OPENAI_API_KEY`, `HF_TOKEN` and `VIRTUAL_ENV` into
#: both, a secret-leak surface for zero benefit.  `lms_fetch_weights` already
#: sets exactly the two variables it needs, deliberately; this follows it.
PROPAGATED_ENV_KEYS = (lms_vram.BASELINE_DIR_ENV,)


def _setenv_flags(env: dict[str, str]) -> list[str]:
    """One `--setenv=K=V` per allowlisted key PRESENT in *env*.

    An absent key emits NOTHING rather than an empty flag.  `''` is not
    "unset" to a consumer's `os.environ.get(...)` fallback: an empty
    `LMS_BASELINE_DIR` would send `lms_vram.baseline_dir()` to `Path('')`
    instead of the `$XDG_RUNTIME_DIR` default it was supposed to fall back to.
    """
    return [f'--setenv={key}={env[key]}' for key in PROPAGATED_ENV_KEYS if key in env]


#: The slate artifact.  Written by `lms_healthcheck --merge`, never by this
#: module -- see the merge step for why that separation is load-bearing.
DEFAULT_ARTIFACT = (
    REPO_ROOT / 'scripts' / 'local-model-serving' / 'verification' / 'health-report.json'
)


def slate_argv(
    parts_dir: str | Path,
    output: str | Path,
    *,
    ready_timeout: float | None = None,
    force: bool = False,
    env: dict[str, str] | None = None,
) -> list[str]:
    """The transient-unit argv that runs the whole slate.

    Paths are resolved to ABSOLUTE here, in the submit layer, and passed
    explicitly in the payload.  Deriving them again inside the unit is how the
    two layers end up reading different directories and a resume silently does
    nothing: `systemd --user` propagates no caller environment, so an
    `XDG_RUNTIME_DIR`-derived path is not the same path on both sides.

    *env* is the caller environment the `--setenv=` allowlist is read from,
    injectable so the allowlist is assertable offline.
    """
    environment = dict(os.environ) if env is None else env

    payload = [
        sys.executable, str(MODULE_PATH),
        '--in-unit',
        '--parts-dir', str(Path(parts_dir).resolve()),
        '--output', str(Path(output).resolve()),
    ]
    if ready_timeout is not None:
        payload += ['--ready-timeout', str(ready_timeout)]
    if force:
        payload.append('--force')

    return lms_fetch_weights.transient_unit_prefix(
        SLATE_UNIT_NAME, _setenv_flags(environment),
    ) + payload


# ---------------------------------------------------------------------------
# the in-unit sweep
# ---------------------------------------------------------------------------


def part_path(parts_dir: str | Path, arm_id: str) -> Path:
    """Where one arm's report part lives.

    A "part" is not a distinct format: it is a full `HealthReport` JSON whose
    `arms` list holds exactly one row, which is what
    `lms_healthcheck --arm X --output p` already writes.  Reusing that shape
    means the resume unit and the `--merge` input are the same file.
    """
    return Path(parts_dir) / f'{arm_id}.json'


def part_is_complete(
    path: str | Path,
    arm_id: str,
    *,
    served_model_name: str | None = None,
) -> bool:
    """True only when *path* is a usable part for *arm_id* -- a PASSING one.

    Validated through the PRODUCER'S OWN pydantic model rather than an ad-hoc
    key check, so a part and `lms_healthcheck` can never disagree about what a
    report is: if the producer's schema changes, an old part stops validating
    here and the arm is re-measured instead of silently reused.

    The row's `arm_id` is checked against the one asked for rather than trusted
    from the FILE NAME -- a mis-copied part would otherwise stand in for an arm
    that was never measured.  Exactly one row, because a multi-row report is a
    merged artifact, not a part; resuming off one would skip an arm whose row
    came from somewhere else entirely.

    A FAIL ROW IS NOT A RESUME POINT.  `lms_healthcheck` writes its report
    BEFORE returning the verdict's exit code, so an arm whose probe failed
    still leaves a fully valid part on disk.  Accepting it would mean an
    operator who FIXED that arm and re-ran the driver gets a byte-identical
    artifact carrying the stale FAIL row, with nothing but a `SKIPPED` line to
    say why -- and the only escape would be `--force`, which re-measures all
    seven arms and so costs exactly the ~30 minutes resuming exists to save.
    Re-measuring the arms that FAILED is the cheap direction of that trade: a
    red arm costs one model load to re-check, a red slate costs the sweep.

    *served_model_name*, when given, is checked against the row for the same
    reason: an `arms.yaml` edit that changes what an arm serves invalidates its
    part, which would otherwise be reused and leave the artifact describing a
    model the manifest no longer commissions under that id.  It cannot catch
    EVERY manifest edit -- a `model_ref` or `quant` change that keeps the
    served name is invisible from a report row -- so `--force` remains the
    answer after a manifest edit that does not move the served name.

    Any failure returns False (re-run) and never propagates.  The common case
    is the ordinary first run, where the file simply does not exist; the
    interesting one is a half-written part from a killed sweep, which is what
    a resume must fall through on rather than trust.
    """
    try:
        report = HealthReport.model_validate_json(Path(path).read_text())
    except (OSError, ValueError):
        return False
    if len(report.arms) != 1:
        return False
    row = report.arms[0]
    if served_model_name is not None and row.served_model_name != served_model_name:
        return False
    return row.arm_id == arm_id and row.verdict == 'PASS'


def ctl_argv(verb: str, arm_id: str | None = None, *extra: str) -> list[str]:
    """One `lms_ctl.py` invocation.

    *arm_id* is optional because `stop-all` takes none -- `lms_ctl` declares it
    `nargs='?'` and errors out on the verbs that need one, so passing an empty
    string rather than omitting it would be an arm id of `''`.

    Note what is NOT here: `--no-exclusive`.  `lms_ctl start` is exclusive by
    default and REFUSES (exit 4) when another arm holds the card rather than
    evicting it, and the sweep depends on that -- it is what turns "two arms
    overlapped" from a silently-degraded measurement into a loud refusal.  The
    sweep never needs the flag because it stops each arm before starting the
    next.
    """
    subject = [] if arm_id is None else [arm_id]
    return [sys.executable, str(CTL_PATH), verb, *subject, *extra]


def healthcheck_argv(*args: str) -> list[str]:
    """One `lms_healthcheck.py` invocation."""
    return [sys.executable, str(HEALTHCHECK_PATH), *args]


#: The subject the pre-sweep release reports a failure under.  Deliberately
#: not an arm id and deliberately not parseable as one: `stop-all` acts on
#: every arm, and attributing its failure to one of them would send an operator
#: to debug an arm that is fine.
RELEASE_SUBJECT = '(all arms)'


def sweep_arms(
    parts_dir: str | Path,
    *,
    ready_timeout: float = DEFAULT_READY_TIMEOUT_S,
    force: bool = False,
    runner: Runner = subprocess.run,
) -> list[tuple[str, str, int]]:
    """Sweep every arm the manifest commissions, one at a time.

    Returns the per-arm failures as `(arm_id, stage, returncode)` rather than
    raising on the first one: the loop must reach every arm, and a bare
    non-zero exit would send an operator back through ~30 minutes of journal to
    find out which arm and which stage.

    Arm order and identity come from `load_arms().arms`, never a hardcoded
    list, so an eighth arm added to `arms.yaml` is swept without touching this
    file.

    RESUMABLE: an arm that already has a valid PASSING part on disk is skipped
    entirely, so a sweep killed at arm six re-measures one arm and not seven.
    *force* re-measures every arm regardless.

    THE CARD IS RELEASED BEFORE THE FIRST ARM.  `lms_ctl start` is exclusive
    and REFUSES (exit 4) while any `lms-arm@` unit is active; it never evicts.
    So a single arm left running by an earlier hand-driven session -- the very
    workflow this script replaces -- would turn the whole sweep into seven
    refusals, no parts, and a merge that refuses on coverage.  The `finally`
    below defends only against arms THIS sweep started, so the leading
    `stop-all` is what makes the sweep survive a card it did not leave clean.
    It touches `lms-arm@` units only, so whisper-writer and anything else on
    the card is unaffected.

    *runner* is the seam that keeps the tests offline: it defaults to
    `subprocess.run` and is injected as a recorder in `test_lms_slate_run.py`,
    so no test can start an arm or touch the card.
    """
    parts = Path(parts_dir)
    parts.mkdir(parents=True, exist_ok=True)
    failures: list[tuple[str, str, int]] = []

    released = runner(ctl_argv('stop-all')).returncode
    if released:
        # Recorded rather than swallowed, and NOT fatal: if nothing was
        # actually held the sweep runs perfectly well from here, and if
        # something was, the per-arm refusals below name it.  Reporting it
        # under a subject that is plainly not an arm id keeps the failure line
        # honest about what failed.
        failures.append((RELEASE_SUBJECT, 'stop-all', released))

    for arm in load_arms().arms:
        existing = part_path(parts, arm.arm_id)
        if not force and part_is_complete(
            existing, arm.arm_id, served_model_name=arm.served_model_name,
        ):
            # Named, not silent: a resumed sweep that quietly does less looks
            # identical to one that measured everything.
            print(f'\n=== {arm.arm_id} === SKIPPED, reusing {existing}', flush=True)
            continue

        print(f'\n=== {arm.arm_id} ===', flush=True)
        try:
            started = runner(ctl_argv('start', arm.arm_id)).returncode
            if started:
                failures.append((arm.arm_id, 'start', started))
                continue

            ready = runner(ctl_argv(
                'wait-ready', arm.arm_id, '--timeout', str(ready_timeout),
            )).returncode
            if ready:
                # Deliberately NOT probed.  A healthcheck against an arm that
                # never came ready records a FAIL row blaming the model for
                # never having loaded, which is worse than the absent row a
                # skip leaves -- an absent row is what `merge_reports`'
                # coverage check refuses on, loudly and by name.
                failures.append((arm.arm_id, 'wait-ready', ready))
                continue

            probed = runner(healthcheck_argv(
                '--arm', arm.arm_id,
                '--output', str(part_path(parts, arm.arm_id)),
            )).returncode
            if probed:
                failures.append((arm.arm_id, 'healthcheck', probed))
        finally:
            # try/FINALLY, not `if rc`.  `lms_ctl start` is exclusive by
            # default and REFUSES (exit 4) when a sibling holds the card
            # instead of evicting it, so ONE arm left running poisons every
            # arm after it: one bad arm becomes six spurious refusals.  The
            # release therefore cannot be conditional on correctly guessing
            # which failures left the card held -- including the raising
            # kind (a missing interpreter, a KeyboardInterrupt mid-sweep).
            #
            # `stop` deliberately leaves this arm's VRAM baseline file behind.
            # Per-arm baselines accumulate in the runtime dir by design, and
            # the merge reads REPORTS, not baselines.
            runner(ctl_argv('stop', arm.arm_id))

    return failures


def existing_parts(parts_dir: str | Path) -> list[str]:
    """The part files that EXIST on disk, in manifest order.

    Only the ones that exist.  Handing the merge a manifest-derived path that
    is missing would route the refusal through an `OSError` on a nonexistent
    file; handing it what exists routes it through `merge_reports`'
    manifest-coverage check instead, which NAMES the uncovered arms.  Both
    refuse and neither writes -- but only one explains.
    """
    parts = Path(parts_dir)
    return [
        str(part_path(parts, arm.arm_id))
        for arm in load_arms().arms
        if part_path(parts, arm.arm_id).exists()
    ]


def placeholder_arm_ids() -> list[str]:
    """The manifest arms that still carry TBD placeholders, if any."""
    return [arm.arm_id for arm in load_arms().arms if arm.is_placeholder]


#: Why a placeholder arm makes the WHOLE slate unassemblable, measured on this
#: branch rather than assumed.  Each leg was checked:
#:
#:   * `lms_ctl start` refuses a placeholder BEFORE touching the card
#:     (`lms_ctl.preflight`'s first check) -- exit 4, nothing started.
#:   * `lms_healthcheck --arm <placeholder>` cannot stand in for it either.
#:     `run_healthcheck` reads the VRAM baseline for every arm it is given
#:     BEFORE probing, and the only writer of a baseline is `lms_ctl start`.
#:     With no start there is no baseline, so it raises `StaleBaselineError`
#:     -> exit 8 and writes NO file -- confirmed by direct call.  So the
#:     PLACEHOLDER_ARM refusal row `_placeholder_refusal` would produce never
#:     reaches disk, and no part exists for the arm.
#:   * `merge_reports` requires a row for every id in `load_arms().arm_ids()`,
#:     placeholders included, and refuses without one.
#:
#: A hand-run `lms_healthcheck --all` hits the same baseline wall, so this is
#: not a limitation the driver introduces -- it is one the driver can only
#: report EARLY instead of after ~30 minutes of sweeping the other arms for an
#: artifact that could never have been written.
PLACEHOLDER_REFUSAL = (
    'lms_slate_run: refusing to sweep: arms {arms} still carry TBD '
    'placeholders, and no slate artifact can be assembled while they do. '
    '`lms_ctl start` refuses a placeholder arm (exit 4), and '
    '`lms_healthcheck --arm` cannot cover it either: with no start there is no '
    'VRAM baseline, so it exits 8 having written nothing -- while the merge '
    'requires a row for EVERY manifest arm. Resolve the PRD open question '
    'that owns them, or drop them from arms.yaml, before running the slate.'
)


def run_slate(
    parts_dir: str | Path,
    output: str | Path,
    *,
    ready_timeout: float = DEFAULT_READY_TIMEOUT_S,
    force: bool = False,
    runner: Runner = subprocess.run,
) -> int:
    """Sweep the slate, then assemble the artifact from the parts it produced.

    THE ARTIFACT IS NEVER WRITTEN HERE.  `lms_healthcheck --merge` writes it,
    and its `merge_reports` manifest-coverage check is THE enforcement point
    against an artifact assembled from a partial slate -- it refuses by name
    ("arms [...] carry no row ... would describe a NARROWER slate than the
    manifest commissions while reading as a complete one") and writes nothing.

    That check is deliberately NOT duplicated here.  A second coverage check in
    this driver could drift from the manifest, and would then either block a
    legitimate merge or -- much worse -- permit a short artifact that reads as
    a complete slate.  For the same reason this module never imports
    `merge_reports` itself: its `expected_arm_ids` parameter defaults to
    `None`, which SKIPS the coverage check entirely, so a library call is one
    forgotten keyword away from writing exactly the artifact that must not
    exist.

    The merge is issued even when NO part exists.  Skipping it on a totally
    failed sweep would end the run with no refusal recorded anywhere.

    The ONE thing that returns before the merge is a manifest still carrying
    TBD placeholders -- see `PLACEHOLDER_REFUSAL` for why such a slate cannot
    be assembled by this driver OR by hand.  Sweeping anyway would spend the
    full ~30 minutes to arrive at a coverage refusal that was decidable from
    the manifest alone, before the card was touched.
    """
    unresolved = placeholder_arm_ids()
    if unresolved:
        print(PLACEHOLDER_REFUSAL.format(arms=unresolved), file=sys.stderr)
        return EXIT_MANIFEST_ERROR

    failures = sweep_arms(
        parts_dir, ready_timeout=ready_timeout, force=force, runner=runner,
    )
    for arm_id, stage, code in failures:
        print(f'lms_slate_run: {arm_id}: {stage} failed (exit {code})', file=sys.stderr)

    merged = runner(healthcheck_argv(
        '--merge', *existing_parts(parts_dir), '--output', str(output),
    )).returncode

    # A failed arm counts even when the merge succeeded: `--merge` can only
    # judge the parts it was handed, so an arm that failed and left no part is
    # invisible to a merge handed a complete set from a previous run.
    return merged or (1 if failures else 0)


# ---------------------------------------------------------------------------
# the submit layer
# ---------------------------------------------------------------------------


def default_parts_dir(env: dict[str, str] | None = None) -> Path:
    """Where per-arm parts live by default.

    `$XDG_RUNTIME_DIR` by preference, matching `lms_vram.baseline_dir()`: parts
    belong to one boot's run, and a tmpfs that empties on reboot says so by
    construction.  Resolved ONCE, here in the submit layer, and passed to the
    unit as an explicit absolute path -- `systemd --user` propagates no caller
    environment, so deriving it again inside the unit is how the two layers end
    up reading different directories and a resume silently does nothing.
    """
    environment = dict(os.environ) if env is None else env
    runtime = environment.get('XDG_RUNTIME_DIR')
    root = Path(runtime) if runtime else Path(tempfile.gettempdir())
    return root / 'lms-slate-parts'


def _submit(
    argv: list[str],
    *,
    dry_run: bool,
    runner: Runner = subprocess.run,
) -> int:
    """Echo the compliant command, then run it unless this is a dry run."""
    print(' '.join(argv), flush=True)
    if dry_run:
        # Deliberately NO `journalctl` follow hint here.  Pointing an operator
        # at a unit that was never created sends them to watch nothing,
        # indefinitely.
        return 0

    code = runner(argv).returncode
    if code == 0:
        print(f'\nfollow this transient unit with:\n'
              f'    journalctl --user -u {SLATE_UNIT_NAME} -f')
    return code


def main(
    argv: list[str] | None = None,
    *,
    runner: Runner = subprocess.run,
) -> int:
    parser = argparse.ArgumentParser(
        prog='lms_slate_run',
        description='Run the whole arm slate in a transient systemd --user unit.',
    )
    parser.add_argument(
        '--in-unit', action='store_true',
        help='run the sweep HERE rather than submitting a unit. Passed by the '
             'unit\'s own payload; also the guard that stops a unit from '
             'recursively submitting another one.',
    )
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument(
        '--force', action='store_true',
        help='re-measure every arm, even one that already has a valid part',
    )
    parser.add_argument('--parts-dir', help='where per-arm report parts live')
    parser.add_argument('--output', help='the slate artifact to assemble')
    parser.add_argument('--ready-timeout', type=float, default=DEFAULT_READY_TIMEOUT_S)
    args = parser.parse_args(argv)

    parts_dir = Path(args.parts_dir) if args.parts_dir else default_parts_dir()
    output = Path(args.output) if args.output else DEFAULT_ARTIFACT

    if args.in_unit:
        return run_slate(
            parts_dir, output,
            ready_timeout=args.ready_timeout, force=args.force, runner=runner,
        )

    return _submit(
        slate_argv(
            parts_dir, output,
            ready_timeout=args.ready_timeout, force=args.force,
        ),
        dry_run=args.dry_run,
        runner=runner,
    )


if __name__ == '__main__':  # pragma: no cover - process entry point
    raise SystemExit(main())
