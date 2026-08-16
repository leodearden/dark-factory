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
from enum import StrEnum
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
#: REFERENCE VALUE only -- see :func:`read_baseline_record`.  Subtracting a frozen
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

#: WHO holds the card, as opposed to HOW MUCH of it is held.  `memory.used`
#: above cannot distinguish an arm's own allocation from a foreign process
#: that happened to be resident at the same moment (task 3755).
_NVIDIA_SMI_COMPUTE_APPS_QUERY = [
    'nvidia-smi',
    '--query-compute-apps=pid,process_name,used_memory',
    '--format=csv,noheader,nounits',
]

#: What the consumer list is, and — more importantly — what it is not.
#: Travels into the health artifact beside every inventory so a downstream
#: reader cannot mistake it for a balance sheet of the card.
CONSUMER_INVENTORY_NOTE = (
    'INVENTORY, not an accounting: `nvidia-smi --query-compute-apps` lists only '
    'CUDA compute applications. It does not list the KDE/X11 graphics contexts '
    '(Xorg, plasmashell, kwin_x11, and ~40 small KDE clients) that hold the '
    'remaining ~3.3 GiB on this host — which is exactly why the measured '
    f'operating budget is ~{MEASURED_OPERATING_BUDGET_GIB} GiB rather than '
    f'{NOMINAL_CEILING_GIB}. These entries name who else held the card; they do '
    'not sum to memory.used.'
)

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


class PollutedBaselineError(VramProbeError):
    """A foreign process held the card when a baseline was to be recorded.

    A :class:`VramProbeError` subclass on purpose, so every existing
    ``except VramProbeError`` handler already catches it and a polluted
    baseline cannot escape a caller that was written before this guard.
    Callers that want the distinct exit code catch this one FIRST.

    Carries :attr:`consumers` so a caller can print WHICH process to deal with
    rather than only that something was wrong.
    """

    def __init__(self, consumers: Sequence[GpuConsumer], context: str = ''):
        self.consumers = list(consumers)
        offenders = '; '.join(
            f'pid {c.pid} {c.process_name} holding {c.used_mib} MiB '
            f'({c.used_gib} GiB)'
            for c in self.consumers
        )
        lead = context or 'the GPU baseline is polluted'
        super().__init__(
            f'{lead}: {offenders}. Nothing but whisper-writer should hold this '
            'card before an arm starts, so a baseline taken now would be '
            "subtracted from the arm's footprint and misattribute another "
            "process's memory. This tool does not stop or police those "
            'processes (ollama self-expires on keep_alive); free the card and '
            'start the arm again'
        )


class StaleBaselineError(VramProbeError):
    """No usable baseline was recorded before this arm started.

    Covers BOTH an absent file and one written before the consumer inventory
    existed.  Those are one condition to an operator -- nothing usable was
    recorded -- differing only in how far back the omission goes, and the remedy
    never varies: ``lms_ctl start`` the arm again so a baseline is captured.

    A :class:`VramProbeError` subclass for the same reason as
    :class:`PollutedBaselineError`: every existing handler still catches it.
    And it exists for the same reason too.  Routed through the generic branch
    this reads as "the GPU probe failed", which sends someone to debug
    nvidia-smi on a perfectly healthy card -- the exact operator-misdirection
    class exit 7 was introduced to end.  The likeliest way to meet it is the
    most ordinary one: an operator whose ``$XDG_RUNTIME_DIR`` still holds a
    pre-guard baseline from the CURRENT boot, immediately after this lands.

    Carries :attr:`path` so a caller can name the file, and :attr:`arm_id` so it
    can name the command, rather than only that something was wrong.
    """

    def __init__(self, arm_id: str, path: Path, message: str):
        self.arm_id = arm_id
        self.path = path
        super().__init__(message)


class PollutedMeasurementError(VramProbeError):
    """The card changed under the measurement, so the budget arithmetic is void.

    Distinct from :class:`PollutedBaselineError`, which says the card was dirty
    BEFORE the arm started.  This one says a non-arm consumer moved DURING the
    run and drove the readings somewhere no subtraction can interpret --
    typically the SHRINK/VANISH direction, where a baseline holder exits and
    ``used`` lands below ``baseline``.

    It exists because that case is otherwise indistinguishable, to a caller,
    from a broken nvidia-smi: :func:`evaluate_budget` raises a plain
    ``VramProbeError`` for it, and the CLI would report "the GPU probe failed"
    (exit 4) for a probe that worked perfectly on a card that was polluted
    (exit 7).  Same operator-misdirection class the ``lms_ctl`` refusal order
    exists to prevent, and the same remedy: name the condition the operator can
    actually act on.
    """

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(
            f'the card changed under this measurement, so the budget '
            f'arithmetic is void rather than merely unflattering: {reason}. '
            'The GPU probe itself is fine; nothing about the arm was learned, '
            'so no report was produced'
        )


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


class GpuConsumer(BaseModel):
    """One CUDA compute application holding VRAM, as nvidia-smi names it.

    The unit of :data:`CONSUMER_INVENTORY_NOTE`.  ``memory.used`` answers how
    much of the card is held; this answers by whom, which is the only thing
    that can tell an arm's own allocation apart from a foreign process that
    happened to be resident at the same moment.
    """

    model_config = ConfigDict(frozen=True)

    pid: int
    process_name: str
    used_mib: int

    @property
    def used_gib(self) -> float:
        return round(self.used_mib / MIB_PER_GIB, 2)


class PollutionState(StrEnum):
    """Whether a VRAM measurement can be attributed to the arm at all.

    A StrEnum so it serialises as a plain string under
    ``model_dump(mode='json')`` and a downstream filter can match on it without
    knowing this module.
    """

    #: Nobody but the expected residents held the card across the measurement.
    CLEAN = 'CLEAN'
    #: A foreign process moved on the card. The budget arithmetic is void — not
    #: merely unflattering. Nothing was learned about the arm.
    POLLUTED = 'POLLUTED'
    #: Nobody looked. Emitted by NO code path in this package; it exists only
    #: so an artifact written before this guard still parses, and reads
    #: honestly as "unmeasured" rather than defaulting to CLEAN.
    UNMEASURED = 'UNMEASURED'


class ConsumerPattern(BaseModel):
    """A named process signature, matched against ``process_name``."""

    model_config = ConfigDict(frozen=True)

    label: str
    name_pattern: str

    def matches(self, process_name: str) -> bool:
        return re.search(self.name_pattern, process_name) is not None


class ExpectedConsumer(ConsumerPattern):
    """One process this host is KNOWN to run on the GPU at baseline.

    A ceiling, not just a name: an allowlist keyed on the process name alone
    would let anything calling itself ``python`` sit in a "clean" baseline at
    any size.
    """

    ceiling_mib: int


#: Below this a consumer cannot move a budget verdict, and flagging it would
#: make the guard cry wolf until an operator learns to ignore it.  The task's
#: "~1 GiB".
POLLUTION_FLOOR_MIB = 1024

#: The ONLY process this host is expected to be running on the GPU when an arm
#: has not started yet.  PRD D10 requires whisper-writer resident, so it is
#: the allowlist and not an exception to it.
#:
#: The ceiling is DERIVED, not guessed: every whisper-writer reading recorded on
#: this host is 4050 MiB, and 6144 sits far above that (leaving room for reload
#: drift) and far below the 10314 MiB ollama holding that motivated this guard.
#: Its only failure direction is a false REFUSAL, which is loud and gets fixed.
#:
#: Deliberately a module constant with NO environment override.  An env var
#: could silence this guard invisibly, leaving no trace in any diff or
#: artifact — the same class of hole the frozen baseline esc-3713-6 removed.
#: Changing what this host is expected to run is a code change with a
#: reviewable diff.  (`LMS_BASELINE_DIR` is not a counterexample: it redirects
#: WHERE a measurement is stored, not what counts as a passing one.)
#:
#: The NAME pattern admits the versioned interpreter spellings nvidia-smi
#: actually reports -- `python`, `python3`, `python3.12`, and any of those with
#: a path -- because whisper-writer under a venv or after a distro interpreter
#: bump is a ROUTINE event, not an exotic one.  An anchor of `python3?$` alone
#: would turn each of those restarts into an exit-5 refusal on a perfectly
#: clean card, pointing the operator at ollama.  A false refusal is the safe
#: direction for a genuine intruder; it is not a safe direction for the one
#: process this host is required to be running.  The `$` and the `(?:^|/)`
#: still hold the line: `pythonish` and `/opt/not-python` match nothing.
EXPECTED_CONSUMERS: tuple[ExpectedConsumer, ...] = (
    ExpectedConsumer(
        label='whisper-writer (PRD D10 requires it resident)',
        name_pattern=r'(?:^|/)python(?:3(?:\.\d+)?)?$',
        ceiling_mib=6144,
    ),
)

#: Processes known to hold this card that are definitely NOT an arm.  Used only
#: at probe time, where the positive allowlist above cannot be applied.
#:
#: ``ollama.service`` is a persistent unit on this host and holds its model on
#: ``keep_alive``; on 2026-08-06 it held 10314 MiB through a whole slate run.
#: The ``/usr/local/lib/ollama/`` path is what makes it decidable -- a
#: containerised vLLM arm never matches it, so this cannot fire on a healthy
#: run.  Same no-environment-override discipline as :data:`EXPECTED_CONSUMERS`.
KNOWN_FOREIGN_CONSUMERS: tuple[ConsumerPattern, ...] = (
    ConsumerPattern(
        label='ollama (persistent unit; holds its model on keep_alive)',
        name_pattern=r'(?:^|/)ollama/',
    ),
)


class GpuBaseline(BaseModel):
    """What the card held immediately before one arm started.

    The reading and the inventory travel together as ONE value because they
    describe one moment.  A reading taken at one instant paired with an
    inventory from another would describe a card that never existed, and
    nothing downstream could detect the mismatch.
    """

    model_config = ConfigDict(frozen=True)

    reading: GpuReading
    consumers: list[GpuConsumer]
    measured_at: datetime
    #: Arms the operator KNOWINGLY left resident when this baseline was taken
    #: (`lms_ctl start --no-exclusive`).  Empty means the card was expected to
    #: hold nothing of ours, which is the strict reading and the default: a
    #: record that does not say otherwise excuses nobody.  See
    #: :func:`unexpected_baseline_consumers` for what a non-empty list relaxes.
    coresident_arms: list[str] = []


class GpuSnapshot(BaseModel):
    """One atomic "what the GPU is, what it holds, and who is holding it".

    *consumers* is REQUIRED, not optional.  An optional inventory would let a
    call site build a snapshot carrying no evidence about who else held the
    card at the moment of the reading, and downstream that absence reads as
    "nobody else was there".
    """

    model_config = ConfigDict(frozen=True)

    identity: GpuIdentity
    reading: GpuReading
    consumers: list[GpuConsumer]


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


def parse_nvidia_smi_compute_apps_csv(text: str) -> list[GpuConsumer]:
    """Parse `--query-compute-apps=pid,process_name,used_memory` rows.

    EMPTY OUTPUT IS A LEGITIMATE EMPTY LIST, and that is the one place this
    parser is deliberately laxer than :func:`parse_nvidia_smi_csv`.  A card
    with no compute application on it prints nothing at all, and that is
    precisely the reading a clean baseline is supposed to produce; raising
    there would make the healthiest possible slate the one this tooling
    refuses to record.  A row that is PRESENT but unreadable is the opposite
    case and always raises: a blank, ``[N/A]``, non-integer or unit-suffixed
    field, a header row (the caller dropped ``noheader``), or fewer than three
    fields.

    One bad row poisons the whole inventory rather than being skipped.  A
    partial list that reads as complete is exactly what would let an unlisted
    holder pass for absent — the failure this guard exists to prevent.

    Each row is split on its FIRST and LAST comma, not on every comma: a
    process path may legitimately contain one, and treating that as a
    field-count error would turn a real (possibly large) GPU holder into a
    blanket refusal.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    consumers: list[GpuConsumer] = []
    for line in lines:
        first = line.find(',')
        last = line.rfind(',')
        if first == -1 or last == first:
            raise VramProbeError(
                f'expected 3 comma-separated fields (pid, process_name, '
                f'used_memory) in compute-apps row {line!r}'
            )

        pid_field = line[:first].strip()
        process_name = line[first + 1:last].strip()
        used_field = line[last + 1:].strip()

        if not _INTEGER.match(pid_field) or not _INTEGER.match(used_field):
            raise VramProbeError(
                f'nvidia-smi compute-apps row {line!r} carries a non-integer pid '
                'or used_memory. A blank, a [N/A], a units suffix (the caller '
                'dropped `nounits`) or a header row all land here; none may be '
                'coerced to 0, which would hide a process holding the card'
            )
        if not process_name or process_name.startswith('[N/A'):
            raise VramProbeError(
                f'nvidia-smi compute-apps row {line!r} names no process; an '
                'anonymous holder cannot be told apart from the arm itself'
            )

        consumers.append(GpuConsumer(
            pid=int(pid_field),
            process_name=process_name,
            used_mib=int(used_field),
        ))

    return consumers


def probe_gpu_consumers(runner: GpuRunner | None = None) -> list[GpuConsumer]:
    """Read who currently holds VRAM on this card.

    Same injected-runner discipline as :func:`probe_gpu`: the subprocess never
    runs in this suite, and a probe failure raises rather than yielding an
    empty list.  An empty list means "nobody holds the card", which is a
    materially different claim from "the probe did not run".
    """
    run = runner if runner is not None else _default_runner
    try:
        output = run(list(_NVIDIA_SMI_COMPUTE_APPS_QUERY))
    except Exception as exc:
        raise VramProbeError(
            f'could not run {" ".join(_NVIDIA_SMI_COMPUTE_APPS_QUERY)}: {exc}'
        ) from exc
    return parse_nvidia_smi_compute_apps_csv(output)


def matching_foreign_pattern(consumer: GpuConsumer) -> ConsumerPattern | None:
    """The :data:`KNOWN_FOREIGN_CONSUMERS` entry this consumer wears, if any.

    Authored once because three call sites need the SAME answer and would
    otherwise each re-derive it: the baseline guard, the probe-time
    classifier, and ``lms_ctl.start`` deciding whether a refusal could
    possibly be a co-resident arm.  A pattern added here must not become
    decidable in one of them and not the others.
    """
    return next(
        (p for p in KNOWN_FOREIGN_CONSUMERS if p.matches(consumer.process_name)),
        None,
    )


def unexpected_baseline_consumers(
    consumers: Sequence[GpuConsumer],
    *,
    coresident_arms: Sequence[str] = (),
) -> list[GpuConsumer]:
    """Which of these has no business holding the card before an arm starts?

    A POSITIVE ALLOWLIST, and that choice is the crux of the design.
    ``lms_ctl.start`` records the baseline BEFORE ``systemctl start``, so at
    that instant nothing of ours is on the card: any consumer over
    :data:`POLLUTION_FLOOR_MIB` that is not in :data:`EXPECTED_CONSUMERS` is a
    surprise.  There is no false-positive direction to protect, so the strictest
    possible rule is also the safest one — it catches ollama and equally
    catches whatever nobody anticipated.

    *coresident_arms* is the ONE case where that premise does not hold, and it
    is a documented, supported one: ``lms_ctl start --no-exclusive`` starts an
    arm while another arm's unit is deliberately left running ("you have
    checked the arithmetic yourself", README "One arm at a time").  Then
    something of ours IS legitimately on the card at baseline, and
    ``--query-compute-apps=pid,process_name,used_memory`` cannot tell that
    containerised vLLM ``python`` from any other ``python`` -- the same
    undecidability :func:`classify_pollution` documents.  So a NON-EMPTY
    *coresident_arms* narrows this to the negative rule that is still decidable:
    a consumer over the floor offends only if it matches
    :data:`KNOWN_FOREIGN_CONSUMERS`.  ollama is still refused; an unrecognised
    holder is excused rather than misdiagnosed as one.

    That relaxation is deliberately NOT free, and the caller pays for it: the
    excused consumers are recorded verbatim in the baseline inventory, the
    arm ids that bought the excuse are recorded beside them
    (:attr:`GpuBaseline.coresident_arms`), and :func:`classify_pollution` still
    catches any DRIFT in those pids between baseline and probe.  An empty
    *coresident_arms* -- the default, and what an older baseline file with no
    such key reads as -- excuses nobody.

    :func:`classify_pollution` deliberately CANNOT use the strict rule at all.
    At probe time the arm's own container is legitimately a new consumer, so
    this allowlist there would flag every healthy run.  The asymmetry is
    intended.

    Size matters as much as name: an allowlist keyed on the process name alone
    would let a leftover containerised vLLM sit in a "clean" baseline under the
    name ``python`` and be subtracted from the next arm's footprint.
    """
    offenders: list[GpuConsumer] = []
    for consumer in consumers:
        if consumer.used_mib < POLLUTION_FLOOR_MIB:
            continue
        if coresident_arms:
            if matching_foreign_pattern(consumer) is not None:
                offenders.append(consumer)
            continue
        expected = any(
            entry.matches(consumer.process_name)
            and consumer.used_mib <= entry.ceiling_mib
            for entry in EXPECTED_CONSUMERS
        )
        if not expected:
            offenders.append(consumer)
    return offenders


def classify_pollution(
    baseline_consumers: Sequence[GpuConsumer],
    probe_consumers: Sequence[GpuConsumer],
) -> tuple[PollutionState, str]:
    """Can the memory delta between these two readings be attributed to the arm?

    NOT the same rule as :func:`unexpected_baseline_consumers`, and the
    asymmetry is the crux of the design.  Arms run as docker containers,
    nvidia-smi reports HOST pids, and
    ``--query-compute-apps=pid,process_name,used_memory`` cannot tell a
    containerised vLLM ``python`` from any other ``python``.  So at probe time
    the arm's own container is an unrecognisable newcomer, and a negative
    allowlist would mark every healthy run POLLUTED.

    Two things ARE decidable, and this function uses only those:

    1. A NEWCOMER matching :data:`KNOWN_FOREIGN_CONSUMERS` over the floor.
       ollama's ``/usr/local/lib/ollama/`` path is not something an arm wears.
    2. Any change over the floor in a consumer PRESENT AT BOTH readings,
       matched by pid.  Such a consumer is non-arm by construction: the arm was
       not running when the baseline was taken.

    BOTH DIRECTIONS of (2) are pollution.  Growth over-charges the arm.  A
    shrink -- or a vanish -- UNDER-charges it, leaving ``used - baseline``
    smaller than the arm truly took, and that flattering direction is precisely
    the one a fabricated artifact wants.  Passing for the wrong reason is not a
    pass.

    THE NAMED RESIDUAL: a newcomer that matches no foreign pattern and is not
    the arm cannot be discriminated from ``--query-compute-apps`` alone.  It is
    recorded verbatim in the report's inventory as the audit trail rather than
    silently attributed to anybody.  Resolving it exactly would need
    ``docker top`` or cgroup introspection to map host pids back to
    ``lms-arm-<arm_id>`` containers; that buys a new docker dependency and a
    resolver whose own failure would mark every real run POLLUTED, so it is
    deliberately out of scope.  Stating the limit is honest; implying full
    attribution would not be.
    """
    reasons: list[str] = []

    baseline_by_pid = {c.pid: c for c in baseline_consumers}
    probe_by_pid = {c.pid: c for c in probe_consumers}

    for consumer in probe_consumers:
        if consumer.pid in baseline_by_pid:
            continue
        if consumer.used_mib < POLLUTION_FLOOR_MIB:
            continue
        foreign = matching_foreign_pattern(consumer)
        if foreign is not None:
            reasons.append(
                f'{foreign.label} arrived while the arm ran: pid {consumer.pid} '
                f'{consumer.process_name} holding {consumer.used_mib} MiB '
                f'({consumer.used_gib} GiB). Its memory is inside the probe '
                "reading and is charged to the arm by `used - baseline`"
            )

    for pid, before in baseline_by_pid.items():
        after = probe_by_pid.get(pid)
        now_mib = after.used_mib if after is not None else 0
        drift = now_mib - before.used_mib
        if abs(drift) < POLLUTION_FLOOR_MIB:
            continue
        # Present before the arm started, so by construction not the arm.
        if drift > 0:
            reasons.append(
                f'pid {pid} {before.process_name} GREW from {before.used_mib} to '
                f'{now_mib} MiB while the arm ran (+{drift} MiB). It was on the '
                'card before the arm started, so that growth is not the arm, '
                'but `used - baseline` OVER-CHARGES the arm for it'
            )
        else:
            gone = ' (it is gone from the probe reading entirely)' if after is None else ''
            reasons.append(
                f'pid {pid} {before.process_name} SHRANK from {before.used_mib} to '
                f'{now_mib} MiB while the arm ran ({drift} MiB){gone}. This is the '
                'FLATTERING direction: `used - baseline` now UNDER-CHARGES the '
                'arm by that much, so a PASS here would be a pass for the wrong '
                'reason'
            )

    if not reasons:
        return PollutionState.CLEAN, ''
    return PollutionState.POLLUTED, '; '.join(reasons)


def probe_gpu_snapshot(runner: GpuRunner | None = None) -> GpuSnapshot:
    """Identity, a live memory reading, and who is holding it, as one value.

    Three nvidia-smi calls rather than one wide query, so the strict three-field
    memory parser -- the one whose failure would misreport the budget -- keeps
    its exact shape and its existing coverage.
    """
    return GpuSnapshot(
        identity=probe_gpu_identity(runner),
        reading=probe_gpu(runner),
        consumers=probe_gpu_consumers(runner),
    )


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
    immediately BEFORE this arm started (see :func:`read_baseline_record`), never from
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


def assert_clean_baseline(
    consumers: Sequence[GpuConsumer],
    *,
    context: str = '',
    coresident_arms: Sequence[str] = (),
) -> None:
    """Raise :class:`PollutedBaselineError` if anything foreign holds the card.

    *coresident_arms* is passed straight through to
    :func:`unexpected_baseline_consumers`; see there for what a non-empty list
    narrows the rule to, and why only ``--no-exclusive`` may supply one.

    Exists so the offending-consumer message is authored ONCE and every caller
    raises the identical, pid-naming error.  There are two callers on purpose,
    and they are doing different jobs:

    * ``lms_ctl.start`` calls it FIRST, ahead of the pre-flight, to fix
      operator MISDIRECTION -- see that function for why the order is
      load-bearing.
    * :func:`record_baseline` calls it again at the write, as the DATA
      INTEGRITY backstop: no polluted baseline is ever persisted, whatever a
      caller did or did not check first.
    """
    offenders = unexpected_baseline_consumers(
        consumers, coresident_arms=coresident_arms,
    )
    if offenders:
        raise PollutedBaselineError(offenders, context=context)


def record_baseline(
    arm_id: str,
    reading: GpuReading,
    *,
    consumers: Sequence[GpuConsumer],
    coresident_arms: Sequence[str] = (),
) -> Path:
    """Persist the pre-start GPU reading AND inventory for *arm_id*.

    Called by ``lms_ctl.start`` between the pre-flight and ``systemctl start``.
    Recording it AT THE START EVENT, rather than accepting it as a healthcheck
    flag, is the anti-fabrication property: the number is produced by the act
    of starting the arm and cannot be typed in afterwards to make a report fit.

    REFUSES a polluted baseline, and refuses it BEFORE any write, so no
    polluted file ever exists for a later reader to pick up and no
    truncate-then-fail can destroy a good measurement.  This is the same
    refusal already sited in ``lms_ctl.start`` for a failed pre-flight: a
    refused arm leaves nothing behind for another arm's report to find.
    Writing-and-flagging would not do -- the file would still be the number the
    next healthcheck subtracts, and the flag only helps a reader who thought to
    look.

    *consumers* is REQUIRED and keyword-only, not optional.  An optional
    inventory would let a caller record a baseline carrying no evidence about
    who else held the card, and downstream that absence reads as "clean".

    *coresident_arms* names the arms a ``--no-exclusive`` start knowingly left
    on the card.  It is PERSISTED, not merely consulted: every later reader of
    this file -- the healthcheck included -- must apply the same relaxed rule
    to the same inventory, and a relaxation that lived only in the caller's
    memory would make the identical file mean two different things depending
    on who read it.
    """
    consumers = list(consumers)
    coresident_arms = list(coresident_arms)
    assert_clean_baseline(
        consumers,
        context=f'refusing to record a baseline for arm {arm_id!r}',
        coresident_arms=coresident_arms,
    )

    path = baseline_path(arm_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'arm_id': arm_id,
        'measured_at': datetime.now(UTC).isoformat(),
        'total_mib': reading.total_mib,
        'used_mib': reading.used_mib,
        'free_mib': reading.free_mib,
        'consumers': [consumer.model_dump(mode='json') for consumer in consumers],
        'coresident_arms': coresident_arms,
    }
    path.write_text(json.dumps(payload, indent=2) + '\n')
    return path


def read_baseline_record(arm_id: str) -> GpuBaseline:
    """The reading AND inventory taken immediately before *arm_id* started.

    Raises rather than falling back to :data:`MEASURED_BASELINE_GIB`.  A default
    here would silently reintroduce the frozen baseline esc-3713-6 ruled out,
    and the artifact would look identical either way -- which is exactly the
    class of wrong answer this package refuses to emit.

    A payload with no ``consumers`` key -- one written before this guard
    existed -- raises for the same reason.  Defaulting it to ``[]`` would make
    the one baseline we know NOTHING about look like the cleanest possible
    reading.  Baselines live in ``$XDG_RUNTIME_DIR`` and are per-boot, so the
    staleness window is bounded and the fix is one command.

    Both of those raise :class:`StaleBaselineError` specifically, so the CLI can
    say "re-take the baseline" instead of "the GPU probe failed".  An
    UNREADABLE file does not: corrupt JSON is a genuine defect in the store, not
    a missing capture, and telling an operator to re-run the same command that
    just wrote garbage would be the wrong advice.
    """
    path = baseline_path(arm_id)
    if not path.exists():
        raise StaleBaselineError(
            arm_id, path,
            f'no baseline recorded for arm {arm_id!r} at {path}. The budget '
            'verdict needs the nvidia-smi reading taken before this arm '
            f'started; start it through `lms_ctl start {arm_id}` so the '
            'baseline is captured, and do not substitute the frozen '
            f'{MEASURED_BASELINE_GIB} GiB reference value',
        )
    try:
        payload = json.loads(path.read_text())
        if 'consumers' not in payload:
            raise StaleBaselineError(
                arm_id, path,
                f'baseline file {path} records no GPU consumer inventory, so '
                'there is no evidence about who else held the card when it was '
                'taken. Treating that as an empty (clean) list would flatter '
                'the one baseline nothing is known about; re-take it with '
                f'`lms_ctl start {arm_id}`',
            )
        return GpuBaseline(
            reading=GpuReading(
                total_mib=payload['total_mib'],
                used_mib=payload['used_mib'],
                free_mib=payload['free_mib'],
            ),
            consumers=[GpuConsumer(**entry) for entry in payload['consumers']],
            measured_at=datetime.fromisoformat(payload['measured_at']),
            # Absent means NOBODY was excused, which is the strict reading and
            # the safe direction: an older file cannot silently acquire an
            # excuse it never earned.  Unlike `consumers`, whose absence is
            # fatal, a missing key here cannot flatter anything -- it can only
            # make the guard stricter than the writer intended.
            coresident_arms=list(payload.get('coresident_arms', [])),
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError,
            ValidationError) as exc:
        raise VramProbeError(f'baseline file {path} is unreadable: {exc}') from exc


def read_baseline_records(arm_ids: Sequence[str]) -> GpuBaseline:
    """One baseline record for a set of arms probed together.

    The LOWEST prior usage wins.  With several arms up, that reading attributes
    the MOST memory to them collectively, so the choice can only ever be
    unflattering -- the opposite of the direction a fabricated artifact wants.

    The whole RECORD is chosen, never a reading from one arm and an inventory
    from another: that pair would describe a card that never existed, and the
    mismatch would be invisible in the artifact.
    """
    if not arm_ids:
        raise VramProbeError(
            'no arms to read a baseline for; a budget verdict over zero arms '
            'would describe nothing'
        )
    records = [read_baseline_record(arm_id) for arm_id in arm_ids]
    return min(records, key=lambda record: record.reading.used_mib)


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
