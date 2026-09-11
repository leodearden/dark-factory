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

# Exit codes, named here rather than spelled as bare literals at the `return`,
# so the meaning an operator acts on lives beside the number and a test can
# assert the NAME.  Same pattern (and, for 2, the same number) as
# `lms_healthcheck`'s block -- the two CLIs are read side by side in the README
# and a code that meant one thing in one and another in the other would be
# worse than no convention at all.
EXIT_OK = 0
#: `wait-ready` timed out, or the arm never served the model the manifest names.
#: Only that verb returns it; nothing was started or stopped.
EXIT_NOT_READY = 1
#: The manifest, or the arm id asked for, is wrong.  Matches lms_healthcheck.
EXIT_MANIFEST_ERROR = 2
#: This arm must not be started AS ASKED: it does not fit the measured free
#: VRAM, or another arm already holds the card and `--no-exclusive` was not
#: given.  The fix is a smaller arm, a freed sibling, or a different flag.
EXIT_ARM_REFUSED = 4
#: Another PROCESS is holding the card, so a baseline taken now would charge
#: its memory to this arm.  Deliberately NOT `EXIT_ARM_REFUSED`: these two have
#: OPPOSITE fixes, and collapsing them would send an operator to shrink an arm
#: that fits perfectly well on a card somebody else is sitting on.  Nothing was
#: started and no baseline file was written.
EXIT_CARD_HELD = 5


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
    consumers: list[lms_vram.GpuConsumer] | None = None,
) -> list[str]:
    """Raise :class:`ArmPreflightError` if this arm must not start.

    Called before any side effect.  Never starts anything.  Returns the arms
    a non-exclusive start knowingly excused, for the caller to record.

    THE ORDER OF THESE FOUR CHECKS IS THE DESIGN, and it is authored HERE ONCE
    so no caller can compose them differently.  Every one of them is a correct
    refusal on a sufficiently bad card; the question is only which one the
    operator is TOLD, and they have different fixes:

    1. PLACEHOLDER -- a manifest problem.  Needs neither card nor systemd, so
       it is decided before anything is probed.
    2. EXCLUSIVITY -- "another ARM is running": stop it, or pass
       --no-exclusive.  This must precede pollution, because a co-resident
       arm's containerised vLLM appears in the inventory as a ``python``
       holding several GiB, which exceeds whisper-writer's 6144 MiB ceiling
       and so reads to the strict allowlist as a foreign intruder.  Checking
       pollution first therefore answered "free the card" (exit 5) for an arm
       the operator can see they started, and sent them into a retry loop
       against a message about ollama.  Ours-versus-foreign is decidable HERE,
       from systemd, and nowhere downstream.
    3. POLLUTION -- "a FOREIGN process is holding the card": free it and start
       the same arm again.  Before the fit check, because an intruder over
       ``POLLUTION_FLOOR_MIB`` is by construction eating the free VRAM the fit
       check measures, so checking fit first makes this refusal unreachable in
       exactly the case it exists for and sends the operator off to shrink an
       arm that fits perfectly well once the intruder releases the card.
       Measured 2026-08-06: ollama holding 10314 MiB left 9.97 GiB free and
       qwen3.5-9b declares 12.0 + 0.5, so the fit check fired.
    4. FIT -- "this arm is too big for THIS card": use a smaller arm, or a
       bigger budget.  Last, because it is the only one whose remedy is the
       arm itself rather than the card, and every earlier refusal changes the
       number it would measure.

    Skipped when *consumers* is None: an inventory is required to say anything
    about pollution, and inventing a clean one would be the silent default
    this package refuses.
    """
    if arm.is_placeholder:
        raise ArmPreflightError(
            f'arm {arm.arm_id!r} still carries TBD placeholders '
            f'(model_ref={arm.model_ref!r}, quant={arm.quant!r}); resolve the '
            'PRD open question that owns it before starting a unit'
        )
    coresident: list[str] = []
    if exclusive:
        others = active_arms() - {arm.arm_id}
        if others:
            raise ArmPreflightError(
                f'arms {sorted(others)} are already running and hold GPU memory. '
                "The PRD's funnel does not run all units simultaneously; stop "
                'them first (lms_ctl stop-all) or pass --no-exclusive knowingly'
            )
    elif consumers is not None:
        coresident = _coresident_excuse(arm, consumers)
    if consumers is not None:
        lms_vram.assert_clean_baseline(
            consumers,
            context=f'refusing to start arm {arm.arm_id!r}',
            coresident_arms=coresident,
        )
    if not lms_vram.arm_fits(arm, gpu.free_gib):
        raise ArmPreflightError(lms_vram.arm_fit_reason(arm, gpu.free_gib))
    return coresident


def _coresident_excuse(
    arm: ArmEntry,
    consumers: list[lms_vram.GpuConsumer],
) -> list[str]:
    """Which running arms, if any, may be holding this card at baseline.

    Reached only on the NON-exclusive path: :func:`preflight` has already
    refused an exclusive start that had any other arm running, so by the time
    this is called there is either nothing to excuse or the operator asked for
    exactly this.

    ASKS SYSTEMD ONLY WHEN THE ANSWER CAN CHANGE THE VERDICT, and that laziness
    is deliberate rather than an optimisation.  On a card that is clean under
    the strict rule, and on one whose only offender is a KNOWN FOREIGN process
    (ollama, which no ``--no-exclusive`` may excuse), the outcome is already
    settled -- and ``start``'s contract is that a refused arm issues NO
    systemctl call at all.  A ``list-units`` is read-only and starts nothing,
    but paying for one on every refusal blurs a boundary the whole module is
    built to keep sharp.  (An EXCLUSIVE start always pays for that query, in
    :func:`preflight`, because there the answer always changes the verdict.)
    """
    offenders = lms_vram.unexpected_baseline_consumers(consumers)
    could_be_an_arm = any(
        lms_vram.matching_foreign_pattern(offender) is None for offender in offenders
    )
    if not could_be_an_arm:
        return []
    return sorted(active_arms() - {arm.arm_id})


def start(
    arm: ArmEntry,
    gpu: lms_vram.GpuReading | None = None,
    *,
    consumers: list[lms_vram.GpuConsumer] | None = None,
    exclusive: bool = False,
) -> None:
    """Start one arm, capturing the baseline the budget verdict will subtract.

    Probes WHO holds the card alongside HOW MUCH is held, through
    :func:`lms_vram.probe_baseline_capture`, which brackets the reading between
    two inventories and refuses a capture whose two halves disagree.  A reading
    taken at one instant paired with an inventory from another would describe a
    card that never existed, and its most dangerous form is silent: a holder
    that is resident for the reading and gone by the inventory inflates the
    baseline and UNDER-charges every arm measured against it afterwards.  That
    narrows the window rather than abolishing it -- the residual is named on
    :func:`lms_vram.unstable_capture_consumers`.  An INJECTED *gpu* or
    *consumers* skips the bracket: the pairing is then the caller's own, which
    is what the tests want and what the check has nothing to say about.

    Ordering is refuse-then-record-then-start throughout.  A polluted card
    raises before any systemctl call, so an arm is never launched onto a card
    it will then be measured against unfairly -- every number such a run
    produced would be uninterpretable, and nothing downstream would say so.

    EVERY REFUSAL LIVES IN :func:`preflight`, in one deliberate order, and
    this function composes none of them itself.  An earlier revision hoisted
    the pollution check up here, ahead of the pre-flight, and that split
    ownership immediately produced its own operator-misdirection bug: with
    exclusivity still inside ``preflight``, a co-resident arm tripped the
    pollution guard first and reported "free the card" (exit 5) for an arm the
    operator had started on purpose.  Two orderings in two places cannot be
    kept consistent by review; one ordering in one place can.  See
    ``preflight`` for what each check means and why it sits where it does.

    ``--no-exclusive`` is a supported escape hatch (README "One arm at a
    time"), and the excuse it produces is returned by the pre-flight and
    recorded with the baseline, so the healthcheck later applies the same rule
    to the same file rather than re-deriving a stricter one.

    :func:`lms_vram.record_baseline` runs the pollution check AGAIN at the
    write.  That is not redundancy: the pre-flight's fixes operator
    MISDIRECTION, and the one inside ``record_baseline`` is the DATA INTEGRITY
    backstop that holds no matter which caller reaches it or in what order.
    """
    if gpu is None and consumers is None:
        reading, held_by = lms_vram.probe_baseline_capture()
    else:
        reading = gpu if gpu is not None else lms_vram.probe_gpu()
        held_by = consumers if consumers is not None else lms_vram.probe_gpu_consumers()
    coresident = preflight(
        arm, reading, exclusive=exclusive, consumers=held_by,
    )
    # The reading the pre-flight just admitted this arm on IS the "immediately
    # before it started" baseline the budget verdict subtracts (esc-3713-6).
    # Recorded here, after the refusal path and before the side effect, so a
    # refused arm leaves no baseline behind for another arm's report to pick up.
    # This call also RAISES on a polluted card, and does so before it writes.
    lms_vram.record_baseline(
        arm.arm_id, reading, consumers=held_by, coresident_arms=coresident,
    )
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
        return EXIT_OK

    if not args.arm_id:
        parser.error(f'{args.verb} needs an arm_id')

    try:
        arm = load_arms().by_id(args.arm_id)
    except ArmManifestError as exc:
        print(f'lms_ctl: {exc}', file=sys.stderr)
        return EXIT_MANIFEST_ERROR

    if args.verb == 'start':
        try:
            start(arm, exclusive=not args.no_exclusive)
        # BEFORE the VramProbeError branch it subclasses.  What each code means
        # and why the two must not collapse is on the constants themselves.
        except lms_vram.PollutedBaselineError as exc:
            print(f'lms_ctl: refusing to start {arm.arm_id}: {exc}', file=sys.stderr)
            return EXIT_CARD_HELD
        except (ArmPreflightError, lms_vram.VramProbeError) as exc:
            print(f'lms_ctl: refusing to start {arm.arm_id}: {exc}', file=sys.stderr)
            return EXIT_ARM_REFUSED
        print(f'started {unit_name(arm)}')
        return EXIT_OK

    if args.verb == 'stop':
        stop(arm)
        print(f'stopped {unit_name(arm)}')
        return EXIT_OK

    if args.verb == 'status':
        return status(arm)

    ready = wait_ready(arm, timeout_s=args.timeout)
    print(f'{arm.arm_id}: {"ready" if ready else "NOT ready"}')
    return EXIT_OK if ready else EXIT_NOT_READY


if __name__ == '__main__':  # pragma: no cover - process entry point
    raise SystemExit(main())
