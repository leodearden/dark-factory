"""Lifecycle control for `lms-arm@<arm_id>.service` units.

PRD-MARKER:local-memory-models-eval serving

Task 3713 (LME-alpha) of `plans/local-memory-models-eval-prd.md`.

The VRAM pre-flight in :func:`start` runs BEFORE any systemctl call, and that
ordering is the whole design.  A refusal issued after `systemctl start` has
already handed the unit to systemd; on a single 24 GB card shared with
whisper-writer (which Leo requires resident, PRD D10) that means a CUDA OOM
that disturbs a process the eval must not disturb, and a unit that must then be
cleaned up.  Refusing first costs nothing and cannot go wrong.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

import lms_vram
from lms_manifest import ArmEntry, ArmManifestError, load_arms

UNIT_PREFIX = 'lms-arm@'
UNIT_SUFFIX = '.service'

DEFAULT_READY_TIMEOUT_S = 900.0
DEFAULT_READY_INTERVAL_S = 5.0
#: Per-request timeout for a readiness probe.  A plain float, never an
#: httpx.Timeout object: the shared test fake exposes only get/post.
READY_PROBE_TIMEOUT_S = 5.0


class ArmPreflightError(Exception):
    """This arm must not be started, and nothing has been started."""


def unit_name(arm: ArmEntry | str) -> str:
    arm_id = arm.arm_id if isinstance(arm, ArmEntry) else arm
    return f'{UNIT_PREFIX}{arm_id}{UNIT_SUFFIX}'


def arm_id_from_unit(unit: str) -> str:
    return unit[len(UNIT_PREFIX):-len(UNIT_SUFFIX)]


def _systemctl(*args: str, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ['systemctl', '--user', *args],
        capture_output=capture, text=True, check=False, timeout=120,
    )


def active_arms() -> set[str]:
    """Arm ids whose unit is ACTIVE and RUNNING.

    `failed` and `inactive` rows are excluded: a failed arm holds no GPU and
    reporting it as active would make `stop_all` claim work it did not do and
    make the exclusivity check refuse a start for no reason.
    """
    result = _systemctl(
        'list-units', f'{UNIT_PREFIX}*{UNIT_SUFFIX}', '--no-legend', '--plain',
        capture=True,
    )
    running: set[str] = set()
    for line in (result.stdout or '').splitlines():
        fields = line.split()
        if len(fields) < 4:
            continue
        unit, _load, active, sub = fields[0], fields[1], fields[2], fields[3]
        if not (unit.startswith(UNIT_PREFIX) and unit.endswith(UNIT_SUFFIX)):
            continue
        if active == 'active' and sub == 'running':
            running.add(arm_id_from_unit(unit))
    return running


def preflight(
    arm: ArmEntry,
    gpu: lms_vram.GpuReading,
    *,
    exclusive: bool = False,
) -> None:
    """Raise :class:`ArmPreflightError` if this arm must not start.

    Called before any side effect.  Never starts anything.
    """
    if arm.is_placeholder:
        raise ArmPreflightError(
            f'arm {arm.arm_id!r} still carries TBD placeholders '
            f'(model_ref={arm.model_ref!r}, quant={arm.quant!r}); resolve the '
            'PRD open question that owns it before starting a unit'
        )
    if not lms_vram.arm_fits(arm, gpu.free_gib):
        raise ArmPreflightError(lms_vram.arm_fit_reason(arm, gpu.free_gib))
    if exclusive:
        others = active_arms() - {arm.arm_id}
        if others:
            raise ArmPreflightError(
                f'arms {sorted(others)} are already running and hold GPU memory. '
                "The PRD's funnel does not run all units simultaneously; stop "
                'them first (lms_ctl stop-all) or pass --no-exclusive knowingly'
            )


def start(
    arm: ArmEntry,
    gpu: lms_vram.GpuReading | None = None,
    *,
    exclusive: bool = False,
) -> None:
    reading = gpu if gpu is not None else lms_vram.probe_gpu()
    preflight(arm, reading, exclusive=exclusive)
    # The reading the pre-flight just admitted this arm on IS the "immediately
    # before it started" baseline the budget verdict subtracts (esc-3713-6).
    # Recorded here, after the refusal path and before the side effect, so a
    # refused arm leaves no baseline behind for another arm's report to pick up.
    lms_vram.record_baseline(arm.arm_id, reading)
    _systemctl('start', unit_name(arm))


def stop(arm: ArmEntry | str) -> None:
    _systemctl('stop', unit_name(arm))


def status(arm: ArmEntry | str) -> int:
    """Return systemctl's own exit code (0 active, 3 inactive, 4 unknown unit).

    Not collapsed to a boolean: swallowing a 3 would make a dead arm read as a
    live one in a report.
    """
    return _systemctl('status', unit_name(arm), capture=True).returncode


def unit_has_failed(arm: ArmEntry | str) -> bool:
    """True when this arm's unit has already given up.

    Read from `is-active`, and treat ONLY the literal `failed` as terminal.

    `inactive` is deliberately NOT a failure, even though a cleanly-exited unit
    reports it: `systemctl is-active` also answers `inactive` for a unit that
    was never started and for one that is not installed at all.  Folding those
    in would make this function abort a wait for reasons that have nothing to do
    with the arm dying, and would turn any caller that polls before starting
    into an instant false negative.  `failed` is unambiguous — the unit ran and
    exited non-zero — and it is what the measured incident produced
    (`Result=exit-code`, `ExecMainStatus=1`).

    `activating` and `active` are likewise not failures: an arm can sit in
    `activating` for seven minutes while vLLM loads weights, and treating that
    as dead would abort every legitimate slow start on the slate.

    Fails SAFE: any systemctl error, timeout, or unrecognised state reads as
    "not failed", so a broken probe makes `wait_ready` fall back to its ordinary
    deadline instead of aborting a healthy arm on bad information.
    """
    try:
        result = _systemctl('is-active', unit_name(arm), capture=True)
    except Exception:
        return False
    return (result.stdout or '').strip() == 'failed'


def stop_all() -> list[str]:
    stopped = sorted(active_arms())
    for arm_id in stopped:
        stop(arm_id)
    return stopped


def wait_ready(
    arm: ArmEntry,
    timeout_s: float = DEFAULT_READY_TIMEOUT_S,
    interval_s: float = DEFAULT_READY_INTERVAL_S,
) -> bool:
    """Poll until the arm is up AND is serving the model the manifest names.

    Returns False on timeout rather than raising: a slow or dead arm is an
    ordinary outcome here, and the caller decides what it means.

    Identity is checked, not assumed.  A `/health` 200 alone once made a
    DIFFERENT model on a colliding port look healthy and mis-attributed a whole
    eval run (the 2026-04-08 404 bug, scripts/run_vllm_eval.py:541-553).  In a
    rig that starts and stops units repeatedly on a fixed port block, that is
    not a hypothetical.

    A DEAD unit short-circuits the wait.  The units are ``Restart=no``, so once
    the container has exited the endpoint can never come up and every remaining
    poll is spent on a foregone conclusion.  Measured 2026-08-06:
    mistral-small-3.2-24b's container exited 82 s in and `wait_ready` polled the
    dead port for its full 900 s default, burning 14 minutes of a 39-minute
    slate run on an arm that had already failed.
    """
    import httpx  # lazy: keeps import cost off every consumer of this module

    deadline = time.monotonic() + timeout_s
    while True:
        if unit_has_failed(arm):
            return False
        try:
            health = httpx.get(f'{arm.base_url}/health', timeout=READY_PROBE_TIMEOUT_S)
            if health.status_code == 200:
                models = httpx.get(
                    f'{arm.base_url}/v1/models', timeout=READY_PROBE_TIMEOUT_S,
                )
                if models.status_code == 200:
                    served = {
                        entry.get('id')
                        for entry in (models.json() or {}).get('data', [])
                    }
                    if arm.served_model_name in served:
                        return True
        except Exception:
            # Connection refused / DNS / malformed body while the server is
            # still coming up. Ordinary during startup; the deadline decides.
            pass
        if time.monotonic() >= deadline:
            return False
        time.sleep(interval_s)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog='lms_ctl', description='Control local model serving arms (LME-alpha).',
    )
    parser.add_argument(
        'verb',
        choices=['start', 'stop', 'stop-all', 'status', 'active', 'wait-ready'],
    )
    parser.add_argument('arm_id', nargs='?')
    parser.add_argument(
        '--no-exclusive', action='store_true',
        help='allow starting while another arm holds the GPU (off by default: '
             "the PRD's funnel does not run all units simultaneously)",
    )
    parser.add_argument('--timeout', type=float, default=DEFAULT_READY_TIMEOUT_S)
    args = parser.parse_args(argv)

    if args.verb in {'stop-all', 'active'}:
        if args.verb == 'active':
            for arm_id in sorted(active_arms()):
                print(arm_id)
        else:
            for arm_id in stop_all():
                print(f'stopped {arm_id}')
        return 0

    if not args.arm_id:
        parser.error(f'{args.verb} needs an arm_id')

    try:
        arm = load_arms().by_id(args.arm_id)
    except ArmManifestError as exc:
        print(f'lms_ctl: {exc}', file=sys.stderr)
        return 2

    if args.verb == 'start':
        try:
            start(arm, exclusive=not args.no_exclusive)
        except (ArmPreflightError, lms_vram.VramProbeError) as exc:
            print(f'lms_ctl: refusing to start {arm.arm_id}: {exc}', file=sys.stderr)
            return 4
        print(f'started {unit_name(arm)}')
        return 0

    if args.verb == 'stop':
        stop(arm)
        print(f'stopped {unit_name(arm)}')
        return 0

    if args.verb == 'status':
        return status(arm)

    ready = wait_ready(arm, timeout_s=args.timeout)
    print(f'{arm.arm_id}: {"ready" if ready else "NOT ready"}')
    return 0 if ready else 1


if __name__ == '__main__':  # pragma: no cover - process entry point
    raise SystemExit(main())
