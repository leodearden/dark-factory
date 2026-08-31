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

import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import lms_fetch_weights
import lms_vram
from lms_ctl import DEFAULT_READY_TIMEOUT_S
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


def ctl_argv(verb: str, arm_id: str, *extra: str) -> list[str]:
    """One `lms_ctl.py` invocation.

    Note what is NOT here: `--no-exclusive`.  `lms_ctl start` is exclusive by
    default and REFUSES (exit 4) when another arm holds the card rather than
    evicting it, and the sweep depends on that -- it is what turns "two arms
    overlapped" from a silently-degraded measurement into a loud refusal.  The
    sweep never needs the flag because it stops each arm before starting the
    next.
    """
    return [sys.executable, str(CTL_PATH), verb, arm_id, *extra]


def healthcheck_argv(*args: str) -> list[str]:
    """One `lms_healthcheck.py` invocation."""
    return [sys.executable, str(HEALTHCHECK_PATH), *args]


def sweep_arms(
    parts_dir: str | Path,
    *,
    ready_timeout: float = DEFAULT_READY_TIMEOUT_S,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> list[tuple[str, str, int]]:
    """Sweep every arm the manifest commissions, one at a time.

    Returns the per-arm failures as `(arm_id, stage, returncode)` rather than
    raising on the first one: the loop must reach every arm, and a bare
    non-zero exit would send an operator back through ~30 minutes of journal to
    find out which arm and which stage.

    Arm order and identity come from `load_arms().arms`, never a hardcoded
    list, so an eighth arm added to `arms.yaml` is swept without touching this
    file.

    *runner* is the seam that keeps the tests offline: it defaults to
    `subprocess.run` and is injected as a recorder in `test_lms_slate_run.py`,
    so no test can start an arm or touch the card.
    """
    parts = Path(parts_dir)
    parts.mkdir(parents=True, exist_ok=True)
    failures: list[tuple[str, str, int]] = []

    for arm in load_arms().arms:
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


def run_slate(
    parts_dir: str | Path,
    output: str | Path,
    *,
    ready_timeout: float = DEFAULT_READY_TIMEOUT_S,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> int:
    """Sweep the slate, then assemble the artifact from the parts it produced.

    *output* is the slate artifact path.  It is never written by this module --
    see the merge step for why that separation is load-bearing.
    """
    failures = sweep_arms(parts_dir, ready_timeout=ready_timeout, runner=runner)
    for arm_id, stage, code in failures:
        print(f'lms_slate_run: {arm_id}: {stage} failed (exit {code})', file=sys.stderr)
    return 0
