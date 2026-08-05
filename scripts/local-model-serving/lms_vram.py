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

import re
import subprocess
from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, ConfigDict

from lms_manifest import ArmEntry

MIB_PER_GIB = 1024

#: PRD D10's nominal ceiling (midpoint of its "~19-20GB").  Reported alongside
#: every measurement so the deviation stays visible in the artifact rather than
#: living in a commit message.
NOMINAL_CEILING_GIB = 19.5

#: Free VRAM measured on this host 2026-08-05 (16761 MiB).  This is what an arm
#: may actually take while whisper-writer and the desktop stay up.
MEASURED_OPERATING_BUDGET_GIB = 16.37

#: The non-arm baseline the figure above is the complement of: 4050 MiB
#: whisper-writer plus ~3312 MiB of KDE/X11 graphics contexts.
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


class BudgetVerdict(BaseModel):
    """One structured budget answer, carrying both reference figures.

    A report that showed only the applied budget would hide the finding this
    task uncovered; a report that showed only the PRD's nominal ceiling would
    assert capacity this host does not have.
    """

    model_config = ConfigDict(frozen=True)

    verdict: Verdict
    reason: str
    used_mib: int
    total_mib: int
    used_gib: float
    total_gib: float
    budget_gib: float
    headroom_gib: float
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
    budget_gib: float = NOMINAL_CEILING_GIB,
) -> BudgetVerdict:
    """Judge a live reading against the budget, reporting both reference figures.

    *budget_gib* is the ceiling on TOTAL used VRAM and defaults to the PRD's
    nominal ceiling, because that is the figure the PRD's user-observable
    signal is stated in ("nvidia-smi within the 19-20GB budget").  The measured
    operating budget travels alongside it on every verdict, and is what
    :func:`arm_fits` enforces per arm.
    """
    if total_mib <= 0:
        raise VramProbeError(f'total VRAM must be positive, got {total_mib} MiB')
    if used_mib < 0:
        raise VramProbeError(f'used VRAM cannot be negative, got {used_mib} MiB')
    if used_mib > total_mib:
        raise VramProbeError(
            f'used {used_mib} MiB exceeds total {total_mib} MiB — incoherent reading'
        )
    if budget_gib <= 0:
        raise VramProbeError(f'budget must be positive, got {budget_gib} GiB')

    used_gib_raw = used_mib / MIB_PER_GIB
    headroom_raw = budget_gib - used_gib_raw
    passed = headroom_raw >= 0

    if passed:
        reason = (
            f'{used_gib_raw:.2f} GiB used is within the {budget_gib:.2f} GiB budget'
        )
    else:
        reason = (
            f'{used_gib_raw:.2f} GiB used EXCEEDS the {budget_gib:.2f} GiB budget by '
            f'{-headroom_raw:.2f} GiB'
        )

    return BudgetVerdict(
        verdict='PASS' if passed else 'FAIL',
        reason=reason,
        used_mib=used_mib,
        total_mib=total_mib,
        used_gib=round(used_gib_raw, 2),
        total_gib=round(total_mib / MIB_PER_GIB, 2),
        budget_gib=budget_gib,
        headroom_gib=round(headroom_raw, 2),
    )


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
