"""VRAM probing and the measured operating budget.

PRD-MARKER:local-memory-models-eval serving

Task 3713 (LME-alpha) of `plans/local-memory-models-eval-prd.md`.

THE MEASUREMENT.  PRD D10 budgets "~19-20GB (24GB minus ~4GB whisper-writer)".
Measured on this host on 2026-08-05, with the desktop in its ordinary state::

    $ nvidia-smi --query-gpu=memory.total,memory.used,memory.free \\
          --format=csv,noheader,nounits
    24576, 7362, 16761

    $ nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
    pid, process_name, used_gpu_memory [MiB]
    7575, python, 4050 MiB

whisper-writer (pid 7575, resident since 2026-07-03) is the only *compute*
application at 4050 MiB.  The remaining ~3.3 GB is the KDE/X11 desktop's
graphics contexts -- Xorg, plasmashell, kwin_x11, obs, an Electron app and ~40
small KDE clients -- which `--query-compute-apps` does not list at all.

So the real operating budget is **16.4 GiB, not 19-20**.  That is not a
contradiction of the PRD: "within 19-20GB" is an upper bound a smaller
footprint still satisfies.  But it binds, because the MoE stretch arm's nominal
~17 GiB does not fit alongside the desktop.  Hence this module treats the
budget as a live measurement and reports BOTH figures, and
:func:`arm_fits` refuses an arm whose declared footprint exceeds measured free
VRAM *before* the launch rather than after the OOM.

Nothing here touches the GPU except through an injected runner.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from lms_manifest import ArmEntry
from pydantic import BaseModel, ConfigDict, ValidationError

MIB_PER_GIB = 1024

#: PRD D10's nominal ceiling (midpoint of its "~19-20GB").  Reported alongside
#: every measurement so the deviation stays visible in the artifact rather than
#: living in a commit message.
NOMINAL_CEILING_GIB = 19.5

#: Free VRAM measured on this host 2026-08-05 (16761 MiB).  A DOCUMENTED
#: REFERENCE VALUE, not the operative number: every verdict uses the free
#: reading taken live at that arm's own baseline (esc-3713-6).  A frozen budget
#: would judge an arm against capacity the card did not have at the time.
MEASURED_OPERATING_BUDGET_GIB = 16.37

#: The non-arm baseline the figure above is the complement of: 4050 MiB
#: whisper-writer plus ~3312 MiB of KDE/X11 graphics contexts.  Also a
#: REFERENCE VALUE only -- see :func:`read_baseline`.  Subtracting a frozen
#: baseline from a live reading misattributes desktop drift to the arm, and in
#: the fabrication-relevant direction: a desktop that shrank since 2026-08-05
#: would hand the arm a discount it did not earn.
MEASURED_BASELINE_GIB = 7.19

#: An arm sized to the last byte of free VRAM OOMs on the first allocation
#: spike (CUDA graph capture, a sampler warmup buffer, a long prompt's KV).
#: The margin is what makes "fits" mean "runs".
SAFETY_MARGIN_GIB = 0.5

_NVIDIA_SMI_QUERY = [
    'nvidia-smi',
    '--query-gpu=memory.total,memory.used,memory.free',
    '--format=csv,noheader,nounits',
]

#: Deliberately NOT `nounits`: neither field is numeric, and a driver version
#: is a dotted string that `nounits` would leave untouched anyway.
_NVIDIA_SMI_IDENTITY_QUERY = [
    'nvidia-smi',
    '--query-gpu=name,driver_version',
    '--format=csv,noheader',
]

_INTEGER = re.compile(r'^\d+$')

#: used + free never sums exactly to total (driver/ECC reserve ~450 MiB here),
#: but a wild mismatch means the fields were transposed or the query changed
#: shape.  5% of total is far above the observed reserve and far below any
#: transposition.
_COHERENCE_TOLERANCE = 0.05

Verdict = Literal['PASS', 'FAIL']

GpuRunner = Callable[[list[str]], str]


class VramProbeError(Exception):
    """The GPU reading is missing, unparseable, or incoherent.

    Typed and always raised -- never a zero-valued reading.  A silent
    ``used_mib = 0`` would report a *passing* budget with maximal headroom off
    a broken probe, which is exactly the reading an operator would trust.
    """


class GpuReading(BaseModel):
    model_config = ConfigDict(frozen=True)

    total_mib: int
    used_mib: int
    free_mib: int

    @property
    def total_gib(self) -> float:
        return round(self.total_mib / MIB_PER_GIB, 2)

    @property
    def used_gib(self) -> float:
        return round(self.used_mib / MIB_PER_GIB, 2)

    @property
    def free_gib(self) -> float:
        return round(self.free_mib / MIB_PER_GIB, 2)


class GpuIdentity(BaseModel):
    """Which card and driver a measurement was taken on.

    Not decoration.  Every verdict in a health report is relative to a specific
    GPU -- the same manifest on a different card gives different answers -- so
    an artifact that does not name its hardware cannot be compared with another
    or trusted a month later.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    driver_version: str


class GpuSnapshot(BaseModel):
    """One atomic "what the GPU is and what it currently holds"."""

    model_config = ConfigDict(frozen=True)

    identity: GpuIdentity
    reading: GpuReading


class BudgetVerdict(BaseModel):
    """One structured budget answer, carrying both reference figures.

    A report that showed only the applied budget would hide the finding this
    task uncovered; a report that showed only the PRD's nominal ceiling would
    assert capacity this host does not have.

    ``arm_footprint_mib`` (``used - baseline``) is the SUBJECT of ``verdict``
    and ``budget_mib`` is the ceiling it is judged against.  Every raw number
    the verdict was computed from travels with it, so a downstream consumer can
    re-derive the answer instead of trusting it.
    """

    model_config = ConfigDict(frozen=True)

    verdict: Verdict
    reason: str
    used_mib: int
    total_mib: int
    baseline_mib: int
    arm_footprint_mib: int
    budget_mib: int
    used_gib: float
    total_gib: float
    baseline_gib: float
    arm_footprint_gib: float
    budget_gib: float
    headroom_gib: float
    #: PRD D10's figure.  REPORTED, NON-GATING -- kept so the deviation this
    #: task measured stays legible in the artifact (esc-3713-6).
    nominal_ceiling_gib: float = NOMINAL_CEILING_GIB
    operating_budget_gib: float = MEASURED_OPERATING_BUDGET_GIB


def parse_nvidia_smi_csv(text: str) -> GpuReading:
    """Parse one `--format=csv,noheader,nounits` memory row.

    Raises :class:`VramProbeError` on anything else at all: an empty capture, a
    header row (the caller dropped ``noheader``, which changes what the numbers
    mean), a non-numeric or ``[N/A]`` field, the wrong field count, units left
    in, more than one GPU, or a row whose parts do not add up.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise VramProbeError(
            'nvidia-smi returned no memory rows; refusing to report a budget '
            'from an empty probe'
        )
    if len(lines) > 1:
        raise VramProbeError(
            f'nvidia-smi returned {len(lines)} rows ({lines!r}); this rig assumes a '
            'single GPU and "the budget" is ambiguous across more than one'
        )

    fields = [field.strip() for field in lines[0].split(',')]
    if len(fields) != 3:
        raise VramProbeError(
            f'expected 3 comma-separated fields (total, used, free), got '
            f'{len(fields)} in {lines[0]!r}'
        )
    if not all(_INTEGER.match(field) for field in fields):
        raise VramProbeError(
            f'nvidia-smi memory row {lines[0]!r} carries a non-integer field. '
            'A [N/A], a units suffix or a header row all land here; none may be '
            'coerced to 0, which would read as a passing budget'
        )

    total_mib, used_mib, free_mib = (int(field) for field in fields)
    if total_mib <= 0:
        raise VramProbeError(f'nvidia-smi reports total memory {total_mib} MiB')

    drift = abs(total_mib - (used_mib + free_mib))
    if drift > total_mib * _COHERENCE_TOLERANCE:
        raise VramProbeError(
            f'incoherent reading total={total_mib} used={used_mib} free={free_mib}: '
            f'used+free is {drift} MiB off total, beyond the '
            f'{_COHERENCE_TOLERANCE:.0%} driver-reserve tolerance'
        )

    return GpuReading(total_mib=total_mib, used_mib=used_mib, free_mib=free_mib)


def _default_runner(argv: list[str]) -> str:
    return subprocess.check_output(argv, text=True, timeout=30)


def probe_gpu(runner: GpuRunner | None = None) -> GpuReading:
    """Read the live GPU memory state.

    The subprocess is injected so every test in this suite runs on a box with
    no NVIDIA driver at all, and so a probe failure is exercised rather than
    assumed.
    """
    run = runner if runner is not None else _default_runner
    try:
        output = run(list(_NVIDIA_SMI_QUERY))
    except Exception as exc:
        raise VramProbeError(
            f'could not run {" ".join(_NVIDIA_SMI_QUERY)}: {exc}'
        ) from exc
    return parse_nvidia_smi_csv(output)


def parse_nvidia_smi_identity_csv(text: str) -> GpuIdentity:
    """Parse one `--query-gpu=name,driver_version --format=csv,noheader` row.

    Held to the same standard as the memory parser: an unreadable identity
    raises rather than defaulting to ``'unknown'``.  A health artifact whose
    provenance says "unknown" is worse than one that does not exist -- the same
    manifest measured on a different card produces different verdicts, so an
    unattributed report invites exactly the cross-host comparison it cannot
    support.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise VramProbeError(
            'nvidia-smi returned no GPU identity row; refusing to attribute a '
            'health report to an unidentified card'
        )
    if len(lines) > 1:
        raise VramProbeError(
            f'nvidia-smi returned {len(lines)} GPU identity rows ({lines!r}); this '
            'rig assumes a single GPU'
        )

    fields = [field.strip() for field in lines[0].split(',')]
    if len(fields) != 2:
        raise VramProbeError(
            f'expected 2 comma-separated fields (name, driver_version), got '
            f'{len(fields)} in {lines[0]!r}'
        )
    if not all(fields) or any(field.startswith('[N/A') for field in fields):
        raise VramProbeError(
            f'nvidia-smi GPU identity row {lines[0]!r} carries an empty or [N/A] '
            'field; the report would then name no card at all'
        )

    return GpuIdentity(name=fields[0], driver_version=fields[1])


def probe_gpu_identity(runner: GpuRunner | None = None) -> GpuIdentity:
    """Read which card and driver this host is measuring on."""
    run = runner if runner is not None else _default_runner
    try:
        output = run(list(_NVIDIA_SMI_IDENTITY_QUERY))
    except Exception as exc:
        raise VramProbeError(
            f'could not run {" ".join(_NVIDIA_SMI_IDENTITY_QUERY)}: {exc}'
        ) from exc
    return parse_nvidia_smi_identity_csv(output)


def probe_gpu_snapshot(runner: GpuRunner | None = None) -> GpuSnapshot:
    """Identity plus a live memory reading, as one value.

    Two nvidia-smi calls rather than one wide query, so the strict three-field
    memory parser -- the one whose failure would misreport the budget -- keeps
    its exact shape and its existing coverage.
    """
    return GpuSnapshot(identity=probe_gpu_identity(runner), reading=probe_gpu(runner))


def gpu_memory_utilization_for(budget_gib: float, total_gib: float) -> float:
    """Derive vLLM's ``--gpu-memory-utilization`` from the measured budget.

    Deliberately NOT the 0.95 pod-era default: that figure came from dedicated
    96 GB eval pods (docs/vllm-eval-status.md:1037), and on this shared 24 GB
    card it would hand vLLM ~23 GiB and evict whisper-writer, which Leo
    requires resident (PRD D10).
    """
    if total_gib <= 0:
        raise VramProbeError(f'total VRAM must be positive, got {total_gib}')
    if budget_gib <= 0:
        raise VramProbeError(
            f'budget must be positive, got {budget_gib}; a zero or negative '
            'budget would silently disable the cap'
        )
    if budget_gib > total_gib:
        raise VramProbeError(
            f'budget {budget_gib} GiB exceeds the card\'s {total_gib} GiB'
        )
    return round(budget_gib / total_gib, 3)


def evaluate_budget(
    used_mib: int,
    total_mib: int,
    *,
    baseline_mib: int,
    baseline_free_mib: int,
) -> BudgetVerdict:
    """Judge THE ARM'S footprint against the free VRAM measured before it started.

    The subject was corrected on 2026-08-06 (esc-3713-6).  This used to compare
    TOTAL card usage with PRD D10's nominal 19.5 GiB ceiling, which charged
    every arm a second time for the ~7.3 GiB desktop+whisper baseline D10 had
    already subtracted.  The effect was not a big-arm technicality: a 9B AWQ
    serving schema-constrained completions correctly measured 21.75 GiB total
    and FAILED.  PRD l.165/l.192 derive 19.5 GiB as the allowance *to the arm*,
    so the arm is what gets judged.

    *baseline_mib* / *baseline_free_mib* come from an nvidia-smi reading taken
    immediately BEFORE this arm started (see :func:`read_baseline`), never from
    :data:`MEASURED_BASELINE_GIB`.  Both reference figures still travel with the
    verdict as reported, non-gating fields.

    :data:`SAFETY_MARGIN_GIB` is deliberately NOT re-added here.  It is applied
    once, where it actually prevents an OOM -- at allocation time, in
    ``lms_serve._memory_share_for``.  Charging it twice would fail an arm that
    fits its own allocation exactly, which is a knife edge and not a budget.
    """
    if total_mib <= 0:
        raise VramProbeError(f'total VRAM must be positive, got {total_mib} MiB')
    if used_mib < 0:
        raise VramProbeError(f'used VRAM cannot be negative, got {used_mib} MiB')
    if used_mib > total_mib:
        raise VramProbeError(
            f'used {used_mib} MiB exceeds total {total_mib} MiB — incoherent reading'
        )
    if baseline_mib <= 0:
        raise VramProbeError(
            f'baseline VRAM must be positive, got {baseline_mib} MiB. A zero '
            'baseline means the pre-start probe never ran, and subtracting it '
            "would credit the desktop's memory to the arm"
        )
    if baseline_mib > used_mib:
        raise VramProbeError(
            f'baseline {baseline_mib} MiB exceeds used {used_mib} MiB: the arm '
            'appears to have FREED memory, so this baseline was not taken '
            'before this run'
        )
    if not 0 < baseline_free_mib <= total_mib:
        raise VramProbeError(
            f'free VRAM at baseline must be in (0, {total_mib}], got '
            f'{baseline_free_mib} MiB'
        )

    footprint_mib = used_mib - baseline_mib
    footprint_gib_raw = footprint_mib / MIB_PER_GIB
    budget_gib_raw = baseline_free_mib / MIB_PER_GIB
    headroom_raw = budget_gib_raw - footprint_gib_raw
    passed = headroom_raw >= 0

    if passed:
        reason = (
            f'the arm took {footprint_gib_raw:.2f} GiB '
            f'({used_mib} - {baseline_mib} MiB), within the '
            f'{budget_gib_raw:.2f} GiB budget free before it started'
        )
    else:
        reason = (
            f'the arm took {footprint_gib_raw:.2f} GiB '
            f'({used_mib} - {baseline_mib} MiB), EXCEEDING the '
            f'{budget_gib_raw:.2f} GiB budget free before it started by '
            f'{-headroom_raw:.2f} GiB'
        )

    return BudgetVerdict(
        verdict='PASS' if passed else 'FAIL',
        reason=reason,
        used_mib=used_mib,
        total_mib=total_mib,
        baseline_mib=baseline_mib,
        arm_footprint_mib=footprint_mib,
        budget_mib=baseline_free_mib,
        used_gib=round(used_mib / MIB_PER_GIB, 2),
        total_gib=round(total_mib / MIB_PER_GIB, 2),
        baseline_gib=round(baseline_mib / MIB_PER_GIB, 2),
        arm_footprint_gib=round(footprint_gib_raw, 2),
        budget_gib=round(budget_gib_raw, 2),
        headroom_gib=round(headroom_raw, 2),
    )


# ---------------------------------------------------------------------------
# The per-arm baseline
# ---------------------------------------------------------------------------

#: Override for the baseline store.  Exists so tests never touch a real runtime
#: directory, and so an operator can point a run at a scratch dir.
BASELINE_DIR_ENV = 'LMS_BASELINE_DIR'


def baseline_dir() -> Path:
    """Where per-arm baselines live.

    ``$XDG_RUNTIME_DIR`` by preference: a baseline is valid only for the boot
    that produced it, and a tmpfs that empties on reboot says so by
    construction rather than leaving a stale file to be read next week.
    """
    override = os.environ.get(BASELINE_DIR_ENV)
    if override:
        return Path(override)
    runtime = os.environ.get('XDG_RUNTIME_DIR')
    root = Path(runtime) if runtime else Path(tempfile.gettempdir())
    return root / 'lms-baselines'


def baseline_path(arm_id: str) -> Path:
    return baseline_dir() / f'{arm_id}.json'


def record_baseline(arm_id: str, reading: GpuReading) -> Path:
    """Persist the pre-start GPU reading for *arm_id*.

    Called by ``lms_ctl.start`` between the pre-flight and ``systemctl start``.
    Recording it AT THE START EVENT, rather than accepting it as a healthcheck
    flag, is the anti-fabrication property: the number is produced by the act
    of starting the arm and cannot be typed in afterwards to make a report fit.
    """
    path = baseline_path(arm_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'arm_id': arm_id,
        'measured_at': datetime.now(UTC).isoformat(),
        'total_mib': reading.total_mib,
        'used_mib': reading.used_mib,
        'free_mib': reading.free_mib,
    }
    path.write_text(json.dumps(payload, indent=2) + '\n')
    return path


def read_baseline(arm_id: str) -> GpuReading:
    """The reading taken immediately before *arm_id* started.

    Raises rather than falling back to :data:`MEASURED_BASELINE_GIB`.  A default
    here would silently reintroduce the frozen baseline esc-3713-6 ruled out,
    and the artifact would look identical either way -- which is exactly the
    class of wrong answer this package refuses to emit.
    """
    path = baseline_path(arm_id)
    if not path.exists():
        raise VramProbeError(
            f'no baseline recorded for arm {arm_id!r} at {path}. The budget '
            'verdict needs the nvidia-smi reading taken before this arm '
            'started; start it through `lms_ctl start` so the baseline is '
            'captured, and do not substitute the frozen '
            f'{MEASURED_BASELINE_GIB} GiB reference value'
        )
    try:
        payload = json.loads(path.read_text())
        return GpuReading(
            total_mib=payload['total_mib'],
            used_mib=payload['used_mib'],
            free_mib=payload['free_mib'],
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValidationError) as exc:
        raise VramProbeError(f'baseline file {path} is unreadable: {exc}') from exc


def read_baselines(arm_ids: Sequence[str]) -> GpuReading:
    """One baseline for a set of arms probed together.

    The LOWEST prior usage wins.  With several arms up, that reading attributes
    the MOST memory to them collectively, so the choice can only ever be
    unflattering -- the opposite of the direction a fabricated artifact wants.
    """
    if not arm_ids:
        raise VramProbeError(
            'no arms to read a baseline for; a budget verdict over zero arms '
            'would describe nothing'
        )
    readings = [read_baseline(arm_id) for arm_id in arm_ids]
    return min(readings, key=lambda r: r.used_mib)


def clear_baseline(arm_id: str) -> None:
    """Drop a recorded baseline.  Idempotent."""
    baseline_path(arm_id).unlink(missing_ok=True)


def arm_fits(arm: ArmEntry, free_gib: float) -> bool:
    """Does this arm's declared footprint fit measured free VRAM, with margin?

    This is where PRD D10's nominal budget stops being the operative number.
    The MoE stretch arm's ~17 GiB fits 19.5 and does not fit 16.4, so on this
    host the answer is no -- and the caller must refuse before launching, not
    discover it from a CUDA OOM that also disturbs whisper-writer.
    """
    return arm.est_vram_gib + SAFETY_MARGIN_GIB <= free_gib


def arm_fit_reason(arm: ArmEntry, free_gib: float) -> str:
    """Human-readable refusal, or ``''`` when the arm fits."""
    if arm_fits(arm, free_gib):
        return ''
    return (
        f'arm {arm.arm_id!r} declares {arm.est_vram_gib} GiB + '
        f'{SAFETY_MARGIN_GIB} GiB safety margin, which exceeds the {free_gib} GiB '
        f'of measured free VRAM. PRD D10\'s nominal ceiling is '
        f'{NOMINAL_CEILING_GIB} GiB, but this host runs whisper-writer '
        f'({MEASURED_BASELINE_GIB} GiB baseline including the KDE/X11 desktop), '
        'so the operating budget is smaller than the PRD assumed'
    )
