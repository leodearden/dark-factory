"""The orchestrator watchdog's persisted-state primitives: atomic JSON state files and the provenance-stamped clocks read and gated on top of them.

Pure in its inputs: the caller passes its journal sink, so every line these
primitives emit lands on the CALLER's log, and nothing here reads a module
global of scripts/orchestrator-watchdog.py. Stdlib-only, like its importer.

Import-resolution contract, the same as scripts/_watched_module_removals.py's:
this stays a flat sibling in scripts/, which a directly executed script finds
because its own directory is sys.path[0].
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import time
from collections.abc import Callable

LogSink = Callable[[str], None]

# Provenance recorded in every clock the watchdog stamps (task 4823). These
# MIRROR df_pytest_isolation.py::CLOCK_PROVENANCE_SOURCE_KEY /
# ::CLOCK_PROVENANCE_SESSION_KEY / ::PYTEST_SESSION_TOKEN_ENV, which
# df_pytest_isolation.py::deploy_clock_change_report reads and is the one place
# that explains what they are for. They cannot be imported: the watchdog is
# stdlib-only and imports no first-party package (the same constraint that
# forced the four-way FLEET_DEPLOY_CLOCK_RELPATH mirror in
# scripts/orchestrator-watchdog.py). Both mirrors are pinned together by
# tests/scripts/test_watchdog_state.py. CLOCK_SOURCE names the writer
# tier deploy_clock_change_report attributes, not this file.
CLOCK_SOURCE = "orchestrator-watchdog.py"
PYTEST_SESSION_TOKEN_ENV = "DF_PYTEST_SESSION_TOKEN"


def atomic_write_json(path: str, payload: dict, log: LogSink) -> bool:
    """Atomically write *payload* as JSON to *path*; return True iff it landed.

    The shared write primitive behind every piece of persisted watchdog state
    (every clock, and the fm liveness streak) — extracted
    in task 3764 so the mkdir/mktemp/write/rename dance is defined ONCE rather
    than copy-pasted per state file. Python analogue of
    restart-all-orchestrators.sh's stamp_fleet_deploy_clock: mkdir -p, mktemp a
    sibling in the SAME directory (so os.replace is a same-filesystem atomic
    rename), write, rename over the target.

    Fail-soft: makedirs / temp-file / rename errors are logged and swallowed
    (the temp file, if created, is unlinked) rather than raised. A persistence
    hiccup must never crash the Type=oneshot watchdog or abort the rest of the
    tick. Every caller's contract must therefore tolerate the write silently
    not having happened — see
    scripts/orchestrator-watchdog.py::_record_fm_liveness_failure, where a
    failed write means the streak counter cannot accumulate and so fails SAFE
    (toward never restarting).

    RETURNS True only after the rename succeeded. Fail-soft is not the same as
    fail-safe: for most callers a missed write is benign, but for one it is
    not — scripts/orchestrator-watchdog.py::_stamp_fm_liveness_restart_clock
    ARMS a cap, so a silently-missed write leaves the liveness revive
    unbounded. Such callers must inspect this value and say so loudly; the log
    line here alone is one line among healthy ticks and is not sufficient
    signal on its own.
    """
    tmp_path: str | None = None
    try:
        target_dir = os.path.dirname(path)
        os.makedirs(target_dir, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix="." + os.path.basename(path) + ".", dir=target_dir
        )
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
        os.replace(tmp_path, path)
        tmp_path = None  # renamed away — nothing to clean up
        return True
    except Exception as exc:  # noqa: BLE001
        if tmp_path is not None:
            # Best-effort cleanup: if the temp file cannot be removed there is
            # nothing further to do, and raising here would mask *exc* (the
            # actual write failure) that the log line below reports.
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
        log(f"atomic_write_json: failed to write {path}: {exc!r}")
        return False


def read_json_state(path: str, log: LogSink) -> dict | None:
    """Return the parsed JSON dict at *path*, or None if it cannot be read.

    The shared fail-open read primitive paired with atomic_write_json
    (task 3764). Returns None — never raises — when the file is missing,
    corrupt, truncated (a torn concurrent write), or unreadable.

    A MISSING file is the normal "no state yet" case (fresh checkout, never
    stamped, streak just cleared) and is deliberately SILENT: logging it would
    spam the journal on every 60s tick. Corrupt/unreadable files ARE logged —
    those are genuine degradations, and swallowing them silently is exactly the
    silent-degradation failure mode the project's norms forbid.

    A well-formed but NON-OBJECT body (``[1, 2]``, ``"hello"``, ``null`` — a
    partially-overwritten or hand-edited file) is logged for the same reason:
    it is indistinguishable in consequence from a corrupt one, since every
    caller then loses its state. Before the task-3764 extraction this case
    raised TypeError inside
    scripts/orchestrator-watchdog.py::_read_last_fm_deploy_epoch and WAS
    logged; letting it become a silent None would have made the deploy
    backstop and the liveness cap both vanish with zero journal evidence.
    """
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            log(
                f"ignoring non-object watchdog state at {path}: "
                f"{type(raw).__name__}"
            )
            return None
        return raw
    except FileNotFoundError:
        return None
    except (OSError, ValueError, TypeError) as exc:
        log(f"ignoring unreadable/corrupt watchdog state at {path}: {exc!r}")
        return None


def read_clock_epoch(path: str, label: str, log: LogSink) -> float | None:
    """Return the ``ts`` epoch from the ``{ts, iso}`` clock file at *path*, or None.

    The shared CLOCK-layer read primitive, one level above read_json_state
    (which owns the missing/corrupt/non-object branches) — extracted so the
    watchdog's clocks cannot drift apart in their fail-open contracts.
    Every one writes and reads the same ``{ts, iso, source, pytest_session}``
    schema that restart-all-orchestrators.sh's ``stamp_fleet_deploy_clock`` and
    ``StaleServiceRestartCoordinator._load_last_fire_wall`` agree on, so a
    single ``float(ts)`` extraction serves every tier. The two provenance keys
    (task 4823) are read ONLY by the pytest-side deploy-clock guard; nothing
    here extracts them, and a reader added later should not start.

    *label* names the clock in the log line only ("fleet-deploy clock",
    "fm-deploy clock", ...) so a journal reader can tell WHICH clock
    degraded; nothing branches on it.

    FAIL-OPEN, uniformly: returns None — never raises — when the file is
    missing (nothing has ever been stamped, or a fresh checkout with no
    data/ yet), corrupt, unreadable, or carrying an unusable ``ts``. Every
    caller must treat None as "the min-interval cap does not apply" (fail
    toward letting the restart/backstop run, never toward silencing it
    indefinitely on the strength of one unreadable file).
    """
    state = read_json_state(path, log)
    if state is None:
        return None
    try:
        return float(state["ts"])
    except (KeyError, TypeError, ValueError) as exc:
        log(f"ignoring {label} at {path} with unusable 'ts': {exc!r}")
        return None


def stamp_clock(path: str, log: LogSink) -> bool:
    """Atomically stamp *path* with the current time and provenance; True iff it landed.

    The shared CLOCK-layer write primitive paired with read_clock_epoch.
    Python analogue of restart-all-orchestrators.sh's stamp_fleet_deploy_clock,
    emitting the identical schema: an integer epoch plus a human-readable UTC
    rendering that exists purely so an operator reading the file by hand does
    not have to decode a bare number, plus the two provenance keys below.

    SCHEMA {ts, iso, source, pytest_session} (task 4823; ts/iso predate it).
    The two provenance keys are ADDITIVE and inert to every reader — each
    extracts `ts` and nothing else — and what they are FOR is stated once, in
    df_pytest_isolation.py::deploy_clock_change_report. The one contract this
    writer must hold on its own: `pytest_session` is ALWAYS present, empty
    included, because empty is the positive statement "no pytest session was an
    ancestor of this write" whereas an omitted key is indistinguishable from a
    pre-4823 writer and fails the run.

    NO SANITISER HERE, unlike the bash sibling, and the asymmetry is deliberate
    rather than an oversight: atomic_write_json serialises through json.dumps,
    which escapes any value correctly. stamp_fleet_deploy_clock builds its body
    with printf, which cannot escape JSON at all, so it must strip the token
    first (falling back to a non-empty sentinel, since stripping to "" would
    forge the one value that forgives a change) or risk emitting a corrupt clock
    — and read_clock_epoch fails OPEN on a corrupt body, which would disarm the
    very cap the stamp arms.

    Fail-soft by inheritance from atomic_write_json: a makedirs/temp/rename
    error is logged and swallowed rather than raised, and reported back as
    False. Callers whose SAFETY depends on the stamp landing (rather than
    merely their convenience) must inspect that value and say so at their own
    call site — see
    scripts/orchestrator-watchdog.py::_stamp_fm_liveness_restart_clock.

    Every clock this serves gets the fields uniformly, whether or not it is
    one of the guarded paths: a schema that diverged between siblings is
    exactly the drift these mirrors exist to prevent.
    """
    now = time.time()
    return atomic_write_json(
        path,
        {
            "ts": int(now),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(now)),
            "source": CLOCK_SOURCE,
            "pytest_session": os.environ.get(PYTEST_SESSION_TOKEN_ENV, ""),
        },
        log,
    )


def min_interval_remaining_secs(
    secs: int, read_epoch: Callable[[], float | None], *, now: float | None = None
) -> float:
    """Return how many seconds *read_epoch*'s clock still holds a *secs* cap; 0.0 once it does not.

    The shared min-interval arithmetic, extracted so every cap shares one
    definition of "too soon" rather than a copy each that could drift. A cap
    that must say how long it still holds reads the clock once, here. *now*
    defaults to the wall clock; a caller that owns its tick time passes it.

    Takes the clock READER rather than a path, so each caller keeps its own
    test seam: the watchdog's caps each pass a named reader
    (scripts/orchestrator-watchdog.py::_read_last_fleet_deploy_epoch and its
    siblings) that their tests substitute, while a caller whose clock path is
    injected can pass a lambda over read_clock_epoch instead.

    Two fail directions, both toward LETTING the caller act:
      - *secs* <= 0 disables the cap outright, and the clock is not even read
        (a disabled cap must not depend on a readable file);
      - a None epoch (missing/corrupt/unreadable clock) counts as OUTSIDE the
        window. Failing the other way would let one unreadable file suppress
        every future restart indefinitely.
    A future-dated clock holds for longer than *secs*; nothing here clamps it.
    """
    if secs <= 0:
        return 0.0
    last = read_epoch()
    if last is None:
        return 0.0
    current = time.time() if now is None else now
    return max(0.0, secs - (current - last))


def within_min_interval(
    secs: int, read_epoch: Callable[[], float | None], *, now: float | None = None
) -> bool:
    """Return True iff the cap still holds; see min_interval_remaining_secs."""
    return min_interval_remaining_secs(secs, read_epoch, now=now) > 0
