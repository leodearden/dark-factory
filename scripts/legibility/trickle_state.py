"""scripts/legibility/trickle_state.py — the nightly trickle run-state
record: WHY a night produced nothing, written by the pipeline itself.

See plans/confusion-reduction-prd.md §6 decision 7. That decision bans
inferring pipeline health from the REPO's contents (git history, codebook
mtime) because a legitimately quiet night commits nothing, so an EXTERNAL
observer cannot tell "produced nothing because there was nothing" from
"produced nothing because it is broken". That information is not missing
from the world — only from the observer. It already exists INSIDE the
pipeline, as :class:`legibility.sampling.SampleResult`'s conservation
invariant. So this module has the pipeline RECORD its own reason in
machine-readable form; :mod:`check_trickle_progress` reads it. A quiet
night is recorded AS quiet and never alarms, so decision 7's stated
rationale is satisfied rather than circumvented.

Division of labour between the probes:

- ``check_trickle_liveness.sh``  — did the UNIT RUN? (systemd unit state)
- ``check_trickle_progress.py``  — did SIGNAL FLOW? (this state file)
- ``check_census_progress.py``   — did a FIRED census advance census-state?
  (this state file + census-state.json)

STDLIB-ONLY IMPORTS ARE A HARD CONSTRAINT, NOT A STYLE PREFERENCE.
``check_trickle_progress.py`` imports this module and, when bound as a
``before_done`` predicate, is EXEC'd directly by
``deterministic_runner._default_run_script``
(``asyncio.create_subprocess_exec(script, *args)``) — no ``uv run``
wrapper, no project venv, bare ``#!/usr/bin/env python3``. Meanwhile
``nightly.py`` and ``legibility.config`` both pull in PyYAML. A stray
third-party import HERE would break the predicate at DISPATCH time, not
at test time — i.e. it would fail in exactly the silent way this module
exists to prevent. ``state_docs`` and the ``shared.safe_io`` it imports
are under the same constraint. Same reason ``DEFAULT_MAX_BARREN_RUNS`` is a module
constant rather than a ``legibility.yaml`` field: reading that config
would drag pydantic + PyYAML onto the predicate path.

Keeping the classifier in ONE module (rather than inlining it in
``nightly.py``) also stops nightly's escalation and the probe's verdict
from drifting apart (INV-5, no lockstep duplication) — the exact drift
that let a suppressed night read like a quiet one for 14 nights
(2026-07-16..29).
"""
from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import os
import pwd
import tempfile
from collections.abc import Iterable, Mapping
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import NamedTuple

# Import state_docs by the same spelling this module was imported by, so a
# caller matching on Missing / Malformed / Ok sees the classes this module
# returns: one source file loaded as two modules has two copies of each.
if __package__:
    from . import state_docs
else:
    import state_docs

logger = logging.getLogger(__name__)

STATE_FILENAME = 'trickle-state.json'
"""Basename of the per-project run-state file."""

STATE_SCHEMA_VERSION = 1
"""Version of the recorded document's shape. :func:`load_state` reads any
OTHER value as malformed rather than guessing at unknown fields.

Fields added after the first writer are ADDITIVE and do not bump it:
``consecutive_failed_runs``, ``capped``, ``retry_backlog``,
``last_scheduled_target_date``, ``census_fired``, ``census_fire_streak``
and ``barren_escalated``. :meth:`TrickleState.from_json` reads an absent one
as its default, and an absent ``barren_escalated`` as ``None``
(unrecorded). Bumping for such a field would make every live state file
read malformed on the first post-deploy probe.

THE RULE FOR THE NEXT PERSON: bump only for a change that would make an
OLD reader MIS-READ a NEW document. Adding a field an old reader ignores
is not that; changing the meaning or type of an existing field is."""


# ---------------------------------------------------------------------------
# Outcome vocabulary and the classifier
# ---------------------------------------------------------------------------

class RunOutcome(StrEnum):
    """What one recorded night was, as :func:`classify_run` derives it."""

    PRODUCTIVE = 'productive'
    """Digests were built: ``selected_count > 0``."""

    QUIET = 'quiet'
    """Nothing reached the sampling or budget stage: a legitimately quiet or
    dormant night. Never an alarm."""

    BARREN = 'barren'
    """Real, distinct, non-duplicate signal reached the sampling or budget
    stage and nothing was digested: the absence that looks quiet from
    outside the pipeline."""

    FAILED = 'failed'
    """The run did not complete (``exit_code != 0``). Its counters are still
    recorded, so whether signal flowed in stays answerable."""

    DEFERRED = 'deferred'
    """The usage cap deferred the night's coding (``coder.is_cap_deferral``),
    so it observed neither whether signal flows nor whether the pipeline
    finishes."""


@dataclasses.dataclass(frozen=True)
class RunCounters:
    """One night's :class:`sampling.SampleResult` counters, each a
    non-negative int."""

    total_records: int
    zero_signal_dropped: int
    dedupe_collapsed: int
    below_sampling_cut: int
    budget_skipped: int
    selected_count: int

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            state_docs.as_count(field.name, getattr(self, field.name))

    @classmethod
    def from_json(cls, raw: object) -> RunCounters:
        doc = state_docs.json_object(raw, 'counters')
        return cls(**{
            field.name: state_docs.required(doc, field.name, state_docs.as_count)
            for field in dataclasses.fields(cls)
        })

    def to_json(self) -> dict[str, int]:
        return dataclasses.asdict(self)


def classify_run(
    *, exit_code: int, counters: RunCounters, capped: bool
) -> RunOutcome:
    """Classify one night; the first matching branch wins.

    1. ``exit_code != 0``                         -> FAILED, whatever else is
       true: selected digests prove signal arrived, not that the night
       finished.
    2. ``capped``                                 -> DEFERRED.
    3. ``selected_count > 0``                     -> PRODUCTIVE, including a
       night that also skipped records on budget (the budget working).
    4. ``budget_skipped + below_sampling_cut > 0`` -> BARREN: only real,
       distinct, non-duplicate signal leaves by either door.
    5. otherwise                                  -> QUIET.

    Branch 5 never hides a barren night. By SampleResult's conservation
    identity ``total_records == zero_signal_dropped + dedupe_collapsed +
    below_sampling_cut + budget_skipped + selected_count``, reaching it means
    every record left by the zero-signal or dedupe door, or none was
    enumerated: the genuinely quiet night PRD decision 7 protects.

    Every parameter is a required keyword, so a caller cannot silently
    default ``exit_code`` to 0 or ``capped`` to False."""
    if exit_code != 0:
        return RunOutcome.FAILED
    if capped:
        return RunOutcome.DEFERRED
    if counters.selected_count > 0:
        return RunOutcome.PRODUCTIVE
    if (counters.budget_skipped + counters.below_sampling_cut) > 0:
        return RunOutcome.BARREN
    return RunOutcome.QUIET


# ---------------------------------------------------------------------------
# project_state_dir / trickle_state_path — where the record lives, and why
# not in the repo
# ---------------------------------------------------------------------------

STATE_ROOT_ENV = 'DARK_FACTORY_LEGIBILITY_STATE_ROOT'
"""The ONE supported lever for relocating the legibility state root.

Deliberately a dedicated name rather than a general-purpose one. See
:func:`project_state_dir` for why that distinction is the whole fix."""


def project_state_dir(project_id: str) -> Path:
    """Return this host's legibility state directory for *project_id*,
    resolved from the ACCOUNT rather than from the calling process's
    environment.

    ``<passwd home>/.local/state/dark-factory/legibility/<project_id>/``,
    overridable only via :data:`STATE_ROOT_ENV`. It holds the trickle
    run-state file (:func:`trickle_state_path`) and the quarantine of
    refused legibility writes (``scripts/legibility/unlanded.py::
    quarantine_root``). Never raises.

    WHY THE PASSWD ANCHOR. The home in the passwd database is a property
    of the account that owns the pipeline, read from NSS, and is identical
    for the same uid in every process environment. Measured 2026-09-19 on
    this host: under ``env -i`` and under ``HOME=/tmp/otherhome``,
    ``pwd.getpwuid(os.getuid()).pw_dir`` is unmoved while ``Path.home()``
    follows ``HOME``. ``pwd`` is stdlib, so the module docstring's hard
    stdlib-only constraint for the bare-``python3`` predicate path
    survives.

    WHY THE DEDICATED OVERRIDE IS NOT THE SAME BUG. ``XDG_STATE_HOME`` and
    ``HOME`` are AMBIENT, GENERAL-PURPOSE variables that a login shell, a
    ``systemd --user`` manager, a container and CI each set differently
    for reasons having nothing to do with legibility — so writer and
    reader diverged with nobody having intended to redirect anything.
    ``DARK_FACTORY_LEGIBILITY_STATE_ROOT`` is never set incidentally; it
    is set only by someone who means THIS directory. The shipped systemd
    units deliberately do not set it (pinned by
    ``scripts/tests/test_install_trickle_health_timer.py``), so production
    always takes the anchored branch.

    WHAT THE DEGRADATION NOW MEANS. The old code fell through to
    ``tempfile.gettempdir()`` whenever ``Path.home()`` raised — which a
    stripped systemd environment with no ``HOME`` did reach. That wrote to
    a path the nightly never reads, while logging to a logger the
    bare-``python3`` predicate never configures, so the divergence was
    invisible. That case now resolves correctly. The tempdir fallback
    survives only for a genuinely absent passwd entry (an arbitrary-uid
    container), preserving the never-raise property this helper needs on
    the nightly run's unconditional path.

    RE-DERIVED, NOT REUSED — AND NO LONGER THE SAME SHAPE. This used to
    mirror ``orchestrator/src/orchestrator/mcp_lifecycle.py::
    managed_runtime_data_dirs`` (task 2439) verbatim, and that claim is now
    false: that function still resolves ``${XDG_STATE_HOME:-~/.local/state}``
    from its own environment. That is CORRECT there, because a single
    orchestrator process both creates and consumes those dirs, and WRONG
    here, because two independently-launched processes must agree. The
    rooting convention (``dark-factory/`` under a state root) is still
    shared; the resolution of the root is deliberately not. That function
    stays un-imported regardless: the ``orchestrator`` package is not
    importable under the bare-``python3`` predicate path this module must
    survive.

    ONE DELIBERATE DIVERGENCE from that convention: an extra
    ``legibility/`` segment between ``dark-factory/`` and ``<project_id>/``
    (mcp_lifecycle uses ``dark-factory/<project_id>/queue|reconciliation``).
    This keeps legibility state from colliding with the managed
    fused-memory runtime dirs for the same project id.
    """
    override = os.environ.get(STATE_ROOT_ENV)
    if override:
        base = Path(override)
    else:
        try:
            base = Path(pwd.getpwuid(os.getuid()).pw_dir) / '.local' / 'state'
        except (KeyError, OSError):
            logger.warning(
                'project_state_dir: no passwd entry for uid %s; falling '
                'back to the OS temp dir for project_id=%s — the recorded '
                'legibility state will not survive a reboot. Set %s to a '
                'durable path.',
                os.getuid(),
                project_id,
                STATE_ROOT_ENV,
            )
            base = Path(tempfile.gettempdir())

    return base / 'dark-factory' / 'legibility' / project_id


def trickle_state_path(project_id: str) -> Path:
    """Return the per-project run-state file path:
    ``<project_state_dir(project_id)>/trickle-state.json``.

    WHAT THIS FILE IS. Its identity is "this host's legibility state for
    this USER and this project" — not "this PROCESS's state dir". Two
    processes that must agree on ONE file cannot each resolve it from
    their own environment. The writer is
    ``legibility-trickle@<project>.service`` under the ``systemd --user``
    manager (user-record HOME, no shell rc). The reader is
    ``legibility-trickle-health@<project>.service``, or an
    orchestrator-EXEC'd ``before_done`` predicate inheriting whatever
    shell launched the orchestrator, or a dev shell. Nothing pinned them
    to agree, so before task 4514 they silently read and wrote different
    files — and a probe that cannot find the file reports ``missing`` and
    fails PERMANENTLY, which for a milestone binding is a born-at-L2
    ``milestone_check_failed`` for a pipeline running perfectly. How the
    directory is resolved so that they DO agree is
    :func:`project_state_dir`'s business.

    NOT UNDER ``docs/legibility/``, despite ``census-state.json`` living
    there and looking like the local precedent. That file is git-TRACKED
    and advances only when a census fires (rare, and committed then). A
    file rewritten EVERY night cannot live on a tracked path: it would
    either leave the machine-operated ``project_root`` checkout
    permanently dirty — poisoning the startup dirty-tree guard and
    warm-lane GC, the exact pollution class task 2439 fixed — or force a
    nightly commit, which would make "the repo has a commit today" a valid
    liveness signal and thereby CONTRADICT PRD decision 7 outright.
    Host-local state outside every checkout is what this actually is: a
    record of what the local timer did. It also needs no ``.gitignore``
    entry.
    """
    return project_state_dir(project_id) / STATE_FILENAME


# ---------------------------------------------------------------------------
# Retry backlog — the target dates still owed a run
# ---------------------------------------------------------------------------

RETRY_MAX_AGE_DAYS = 7
"""Age in days past which an owed target date is abandoned.

At :data:`RETRIES_PER_NIGHT` retries per finished night, oldest first,
every date of a k-night outage is retried at age k+1, so, when every
later night finishes, an outage of up to ``RETRY_MAX_AGE_DAYS - 1``
nights is fully recovered and a longer one loses only its oldest dates.
On the same condition a systemd outage of up to ``RETRY_MAX_AGE_DAYS``
missed elapses is fully recovered, since its catch-up pass covers the
last date itself. Each later night that does not finish retries nothing,
so it costs the oldest owed date. A module constant, not a
``legibility.yaml`` field, for the same bare-``python3`` reason as
:data:`DEFAULT_MAX_BARREN_RUNS`."""

RETRY_MAX_ATTEMPTS = 3
"""Unfinished passes of one date (its scheduled pass plus retries) after
which it is abandoned, so a date that fails deterministically cannot hold
the head of the oldest-first queue until it ages out. It counts unfinished
passes only, so a date no pass ran gets this many retries."""

RETRIES_PER_NIGHT = 1
"""Retry passes one invocation runs after its own scheduled pass finished.
It bounds the extra spend to one night's, and it is the rate the
:data:`RETRY_MAX_AGE_DAYS` recovery bound is derived from."""


class RetryFold(NamedTuple):
    """The backlog after folding one pass, and what that fold abandoned."""

    backlog: dict[date, int]
    abandoned: dict[date, int]


class NeverRun(NamedTuple):
    """The target dates a scheduled pass jumped over: the ``owed`` ones
    still inside the age window, oldest first, and how many older ones
    ``expired`` before any pass could run them."""

    owed: tuple[date, ...]
    expired: int


def finished(*, exit_code: int, capped: bool) -> bool:
    """Whether a pass completed: a clean exit the usage cap did not defer."""
    return exit_code == 0 and not capped


def owes_retry(*, exit_code: int, capped: bool, commit_made: bool) -> bool:
    """Whether a pass leaves its target date owed a run: it did not
    finish, and it landed nothing."""
    return not commit_made and not finished(exit_code=exit_code, capped=capped)


def parse_retry_backlog(raw: object) -> dict[date, int]:
    """Read a recorded ``retry_backlog`` ({ISO date: unfinished attempts}).

    Never raises. A non-mapping gives an empty backlog, and an entry whose
    key is not an ISO date or whose count is not a non-negative int is
    dropped: a damaged backlog under-reports, like a lost streak."""
    if not isinstance(raw, dict):
        return {}
    backlog: dict[date, int] = {}
    for key, attempts in raw.items():
        owed = _parse_iso_date(key)
        if owed is not None and _is_attempt_count(attempts):
            backlog[owed] = attempts
    return backlog


def _parse_iso_date(raw: object) -> date | None:
    if not isinstance(raw, str):
        return None
    try:
        return date.fromisoformat(raw)
    except ValueError:
        return None


def _is_attempt_count(raw: object) -> bool:
    return isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0


def _oldest_in_window(today: date) -> date:
    return today - timedelta(days=RETRY_MAX_AGE_DAYS)


def _within_window(owed: date, today: date) -> bool:
    return owed >= _oldest_in_window(today)


def never_run_dates(
    last_scheduled: date | None, *, target_date: date, today: date
) -> NeverRun:
    """The target dates no scheduled pass ran: strictly after the last
    scheduled pass's target and strictly before this one's. Only those
    inside the age window on *today* are listed; older ones would be
    abandoned on arrival, so they are counted, never listed."""
    if last_scheduled is None:
        return NeverRun(owed=(), expired=0)
    gap_start = last_scheduled + timedelta(days=1)
    owed_start = min(max(gap_start, _oldest_in_window(today)), target_date)
    return NeverRun(
        owed=tuple(
            owed_start + timedelta(days=n)
            for n in range((target_date - owed_start).days)
        ),
        expired=max(0, (owed_start - gap_start).days),
    )


def fold_retry_backlog(
    backlog: Mapping[date, int],
    *,
    target_date: date,
    owes_retry: bool,
    today: date,
    never_run: Iterable[date] = (),
) -> RetryFold:
    """Fold one pass of *target_date* into a copy of *backlog*.

    Each *never_run* date not already owed enters with 0 attempts. An
    unfinished pass adds its date or counts one more attempt; a finished
    pass removes it. Every entry then outside the age window, or at
    :data:`RETRY_MAX_ATTEMPTS`, moves to ``abandoned``."""
    folded = dict(backlog)
    for owed in never_run:
        folded.setdefault(owed, 0)
    if owes_retry:
        folded[target_date] = folded.get(target_date, 0) + 1
    else:
        folded.pop(target_date, None)
    kept = {
        owed: attempts
        for owed, attempts in folded.items()
        if _within_window(owed, today) and attempts < RETRY_MAX_ATTEMPTS
    }
    abandoned = {
        owed: attempts for owed, attempts in folded.items() if owed not in kept
    }
    return RetryFold(backlog=kept, abandoned=abandoned)


def select_due_retries(
    backlog: Mapping[date, int],
    *,
    today: date,
    limit: int = RETRIES_PER_NIGHT,
) -> tuple[date, ...]:
    """The owed dates to retry now: in-window only, oldest first, at most
    *limit*."""
    due = sorted(owed for owed in backlog if _within_window(owed, today))
    return tuple(due[:limit])


def _serialize_retry_backlog(backlog: Mapping[date, int]) -> dict[str, int]:
    return {owed.isoformat(): backlog[owed] for owed in sorted(backlog)}


# ---------------------------------------------------------------------------
# The census-fire streak — did the census each FIRE launched advance
# census-state?
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class CensusObservation:
    """One night's census-trigger fact, as :func:`record_run` receives it.

    ``last_census_at`` is the census-state value the trigger was evaluated
    against, read BEFORE any launch: the launch is synchronous, so a census
    that lands has already rewritten census-state by the time the night is
    recorded. ``None`` means census-state had no value (never censused)."""

    fired: bool
    last_census_at: str | None
    census_state_path: str


@dataclasses.dataclass(frozen=True)
class CensusFireStreak:
    """``fires`` consecutive recorded nights on which the trigger FIRED
    against one unchanged ``last_census_at`` at one ``census_state_path``,
    starting on the target_date ``since``."""

    fires: int
    last_census_at: str | None
    census_state_path: str
    since: str


def _parse_census_fire_streak(raw: object) -> CensusFireStreak | None:
    """The recorded ``census_fire_streak`` object, or ``None`` when absent,
    null or unusable: a damaged streak under-reports rather than making the
    whole document malformed. ``fires`` must be a real int of at least 1."""
    if not isinstance(raw, dict):
        return None
    fires = raw.get('fires')
    anchor = raw.get('last_census_at')
    path = raw.get('census_state_path')
    since = raw.get('since')
    if not isinstance(fires, int) or isinstance(fires, bool) or fires < 1:
        return None
    if anchor is not None and not isinstance(anchor, str):
        return None
    if not isinstance(path, str) or not isinstance(since, str):
        return None
    return CensusFireStreak(
        fires=fires, last_census_at=anchor, census_state_path=path, since=since,
    )


def read_last_census_at(path: Path) -> tuple[str, object]:
    """``('missing', None)``, ``('malformed', None)`` or ``('ok', raw
    last_census_at)`` for census-state.json at *path*: the ONE read shared
    by the nightly's pre-launch anchor and ``check_census_progress.py``'s
    live comparison, so the two sides of its exact-equality test agree.

    QUIET: ``census_trigger.load_census_state`` owns the malformed WARNING
    for this file. Timestamps are deliberately not validated: the trigger
    never FIREs on a malformed census-state, so no streak can be anchored
    on one."""
    if not path.exists():
        return 'missing', None
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return 'malformed', None
    if not isinstance(data, dict):
        return 'malformed', None
    return 'ok', data.get('last_census_at')


def _fold_census_fire_streak(
    prev: CensusFireStreak | None,
    census: CensusObservation | None,
    target_date: date,
) -> CensusFireStreak | None:
    if census is None:
        return prev
    if not census.fired:
        return None
    if (
        prev is not None
        and prev.last_census_at == census.last_census_at
        and prev.census_state_path == census.census_state_path
    ):
        return dataclasses.replace(prev, fires=prev.fires + 1)
    return CensusFireStreak(
        fires=1,
        last_census_at=census.last_census_at,
        census_state_path=census.census_state_path,
        since=target_date.isoformat(),
    )


# ---------------------------------------------------------------------------
# The recorded document
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class TrickleState:
    """One recorded night, as ``trickle-state.json`` holds it.

    :meth:`from_json` and :meth:`to_json` are the document's only codec.
    ``barren_escalated`` is ``None`` for a document written before the
    field existed: UNRECORDED, which a reader must not take for False."""

    project_id: str
    target_date: date
    recorded_at: datetime
    exit_code: int
    outcome: RunOutcome
    applied: int
    commit_made: bool
    budget_suppressed: bool
    capped: bool
    consecutive_barren_runs: int
    consecutive_failed_runs: int
    last_productive_at: datetime | None
    last_scheduled_target_date: date | None
    census_fired: bool | None
    census_fire_streak: CensusFireStreak | None
    counters: RunCounters
    retry_backlog: Mapping[date, int]
    barren_escalated: bool | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, 'retry_backlog', MappingProxyType(dict(self.retry_backlog))
        )

    @classmethod
    def from_json(cls, raw: object) -> TrickleState:
        """Strict about every field present; an absent additive field takes
        its default (see :data:`STATE_SCHEMA_VERSION`). The census streak
        and the retry backlog keep their under-reporting parses."""
        doc = state_docs.json_object(raw, 'trickle state')
        version = state_docs.required(doc, 'schema_version', state_docs.as_int)
        if version != STATE_SCHEMA_VERSION:
            raise ValueError(
                f'schema_version: unknown version {version!r} (this module '
                f'writes {STATE_SCHEMA_VERSION})'
            )
        required, defaulted = state_docs.required, state_docs.defaulted
        return cls(
            project_id=required(doc, 'project_id', state_docs.as_str),
            target_date=required(doc, 'target_date', state_docs.as_date),
            recorded_at=required(doc, 'recorded_at', state_docs.as_datetime),
            exit_code=required(doc, 'exit_code', state_docs.as_int),
            outcome=required(doc, 'outcome', state_docs.as_member(RunOutcome)),
            applied=required(doc, 'applied', state_docs.as_count),
            commit_made=required(doc, 'commit_made', state_docs.as_bool),
            budget_suppressed=required(doc, 'budget_suppressed', state_docs.as_bool),
            capped=defaulted(doc, 'capped', state_docs.as_bool, False),
            consecutive_barren_runs=required(
                doc, 'consecutive_barren_runs', state_docs.as_count
            ),
            consecutive_failed_runs=defaulted(
                doc, 'consecutive_failed_runs', state_docs.as_count, 0
            ),
            last_productive_at=defaulted(
                doc, 'last_productive_at', state_docs.as_datetime, None
            ),
            last_scheduled_target_date=defaulted(
                doc, 'last_scheduled_target_date', state_docs.as_date, None
            ),
            census_fired=defaulted(doc, 'census_fired', state_docs.as_bool, None),
            census_fire_streak=_parse_census_fire_streak(
                doc.get('census_fire_streak')
            ),
            counters=RunCounters.from_json(doc.get('counters')),
            retry_backlog=parse_retry_backlog(doc.get('retry_backlog')),
            barren_escalated=defaulted(
                doc, 'barren_escalated', state_docs.as_bool, None
            ),
        )

    def to_json(self) -> dict[str, object]:
        return {
            'schema_version': STATE_SCHEMA_VERSION,
            'project_id': self.project_id,
            'target_date': self.target_date.isoformat(),
            'recorded_at': self.recorded_at.isoformat(),
            'exit_code': self.exit_code,
            'outcome': self.outcome.value,
            'applied': self.applied,
            'commit_made': self.commit_made,
            'budget_suppressed': self.budget_suppressed,
            'capped': self.capped,
            'consecutive_barren_runs': self.consecutive_barren_runs,
            'consecutive_failed_runs': self.consecutive_failed_runs,
            'last_productive_at': _isoformat_or_none(self.last_productive_at),
            'last_scheduled_target_date': _isoformat_or_none(
                self.last_scheduled_target_date
            ),
            'census_fired': self.census_fired,
            'census_fire_streak': (
                None if self.census_fire_streak is None
                else dataclasses.asdict(self.census_fire_streak)
            ),
            'counters': self.counters.to_json(),
            'retry_backlog': _serialize_retry_backlog(self.retry_backlog),
            'barren_escalated': self.barren_escalated,
        }


def _isoformat_or_none(value: date | None) -> str | None:
    return None if value is None else value.isoformat()


def load_state(path: str | Path) -> state_docs.StateRead[TrickleState]:
    """Read a trickle run-state file: Missing, Malformed (with the reason,
    one WARNING) or Ok. Never raises."""
    return state_docs.read_state(Path(path), TrickleState.from_json)


def recorded_age_hours(state: TrickleState, now: datetime | None = None) -> float:
    """Hours since *state* was recorded, against *now* (the wall clock by
    default): the one freshness reading both probes share."""
    current = datetime.now(UTC) if now is None else now
    return (current - state.recorded_at).total_seconds() / 3600.0


def due_retries(state: TrickleState, *, today: date) -> tuple[date, ...]:
    """The owed dates *state* makes due now, per :func:`select_due_retries`."""
    return select_due_retries(state.retry_backlog, today=today)


# ---------------------------------------------------------------------------
# record_run — the writer
# ---------------------------------------------------------------------------

def record_run(
    project_id: str,
    *,
    target_date: date,
    recorded_at: datetime,
    exit_code: int,
    total_records: int,
    zero_signal_dropped: int,
    dedupe_collapsed: int,
    below_sampling_cut: int,
    budget_skipped: int,
    selected_count: int,
    applied: int = 0,
    commit_made: bool = False,
    budget_suppressed: bool = False,
    capped: bool = False,
    scheduled: bool = False,
    census: CensusObservation | None,
) -> dict:
    """Classify this run, fold it into the running streak, and write the
    state file atomically. Returns the document it wrote.

    The run is also folded into ``retry_backlog`` by
    :func:`fold_retry_backlog`, so a date whose pass did not finish stays
    owed a run.

    A *scheduled* pass advances ``last_scheduled_target_date`` to its
    target, never backwards, and owes, with 0 attempts, the
    :func:`never_run_dates` since the previous scheduled pass. A manual
    pass neither moves that cursor nor owes them, because it rewrites
    ``target_date`` (the backfill case).

    ``recorded_at`` is an INJECTED parameter, not a ``datetime.now()``
    inside this function, so the whole module stays clock-injectable and
    streak/freshness tests need no time faking — mirroring
    ``nightly.run_nightly``'s existing ``now=`` seam and
    ``census_trigger.evaluate_census_step``'s clock seam.

    Streak semantics: ``consecutive_barren_runs`` is the previous value
    plus one when this run is barren, and 0 otherwise. BOTH ``productive``
    and ``quiet`` reset it — a legitimately quiet night is not evidence of
    breakage, which is PRD decision 7's no-false-alarm guarantee expressed
    in the streak rather than only in the classifier.

    ``failed`` is the exception: it neither increments nor resets
    ``consecutive_barren_runs``, it CARRIES IT FORWARD. A crashed run is
    evidence about the RUN, not about whether signal is flowing.
    Resetting would let a permanently broken pipeline erase a real barren
    streak — the same silent-degradation shape one layer up; incrementing
    would attribute an absence the sampler never observed, since the run
    did not finish and its counters describe an unfinished night.

    ``failed`` gets its OWN counter rather than being folded into the
    barren one because the two have genuinely DIFFERENT remedies — the
    same reason ``SampleResult`` keeps ``budget_skipped`` and
    ``below_sampling_cut`` separate. Barren means "fix the budget or
    sampling config"; failed means "read ``journalctl --user -u
    legibility-trickle@<project>``".

    ``scripts/legibility/nightly.py::_escalate_barren_streak`` is
    UNAFFECTED by either. It returns early unless ``outcome ==
    OUTCOME_BARREN``, so a failed run cannot fire it; and it gates on
    EXACT equality with the threshold, which a carried-forward streak
    still passes through at most once, so the carry-forward cannot
    double-fire it either.

    ``last_productive_at`` is stamped with ``recorded_at`` on a productive
    run that :func:`finished` and carried forward UNCHANGED otherwise. It is
    the field an operator reads to answer "when did this pipeline last
    actually do anything", so a *capped* night, which built its digests and
    coded none of them, still reads ``productive`` but is not stamped.

    ``census_fire_streak`` counts consecutive FIREs against one unchanged
    anchor (:class:`CensusObservation`): a FIRE with the streak's anchor and
    path increments it, any other FIRE restarts it at 1, a NO-FIRE clears
    it, and ``census=None`` (the trigger was not evaluated: the night failed
    or was capped first) carries it forward, as ``failed`` carries the
    barren streak. ``census`` is REQUIRED with no default, for the reason
    :func:`classify_run` gives for ``exit_code``: a defaulted ``None`` would
    silently blind ``check_census_progress.py`` into a permanent OK.

    A ``missing`` or ``malformed`` predecessor simply starts the streak at
    this run (and drops ``last_productive_at``): losing streak history
    degrades to UNDER-reporting, never to a crash inside the nightly run.

    The write is atomic: the payload goes to a ``.tmp`` sibling in the
    SAME directory and is then ``os.replace``d onto the final path.
    Same-directory is required — ``os.replace`` is only atomic within one
    filesystem — and the tmp file is removed on any failure, so a crashed
    write can never leave a partial document that ``load_state`` would
    report as ``malformed``.
    """
    path = trickle_state_path(project_id)

    read = load_state(path)
    prev_status = 'ok' if isinstance(read, state_docs.Ok) else 'missing'
    prev = read.value.to_json() if isinstance(read, state_docs.Ok) else None
    prev_streak = 0
    prev_failed = 0
    last_productive_at = None
    if prev_status == 'ok' and prev is not None:
        raw_streak = prev.get('consecutive_barren_runs')
        if isinstance(raw_streak, int) and raw_streak >= 0:
            prev_streak = raw_streak
        raw_failed = prev.get('consecutive_failed_runs')
        if isinstance(raw_failed, int) and raw_failed >= 0:
            prev_failed = raw_failed
        raw_last = prev.get('last_productive_at')
        if isinstance(raw_last, str):
            last_productive_at = raw_last

    outcome = classify_run(
        exit_code=exit_code,
        counters=RunCounters(
            total_records=total_records,
            zero_signal_dropped=zero_signal_dropped,
            dedupe_collapsed=dedupe_collapsed,
            below_sampling_cut=below_sampling_cut,
            budget_skipped=budget_skipped,
            selected_count=selected_count,
        ),
        capped=capped,
    )

    recorded_at_iso = recorded_at.isoformat()
    if outcome == RunOutcome.PRODUCTIVE and finished(exit_code=exit_code, capped=capped):
        last_productive_at = recorded_at_iso

    cursor, never_run = _schedule_cursor(
        project_id,
        prev,
        target_date=target_date,
        today=recorded_at.date(),
        scheduled=scheduled,
    )

    census_streak = _fold_census_fire_streak(
        _parse_census_fire_streak(
            None if prev is None else prev.get('census_fire_streak')
        ),
        census,
        target_date,
    )

    doc = {
        'schema_version': STATE_SCHEMA_VERSION,
        'project_id': project_id,
        'target_date': target_date.isoformat(),
        'recorded_at': recorded_at_iso,
        'exit_code': exit_code,
        'outcome': outcome,
        'applied': applied,
        'commit_made': bool(commit_made),
        'budget_suppressed': bool(budget_suppressed),
        'capped': bool(capped),
        'consecutive_barren_runs': (
            prev_streak + 1 if outcome == RunOutcome.BARREN
            else prev_streak if outcome == RunOutcome.FAILED
            else 0
        ),
        'consecutive_failed_runs': (
            prev_failed + 1 if outcome == RunOutcome.FAILED else 0
        ),
        'last_productive_at': last_productive_at,
        'last_scheduled_target_date': cursor.isoformat() if cursor else None,
        'census_fired': None if census is None else census.fired,
        'census_fire_streak': (
            None if census_streak is None
            else dataclasses.asdict(census_streak)
        ),
        'counters': {
            'total_records': total_records,
            'zero_signal_dropped': zero_signal_dropped,
            'dedupe_collapsed': dedupe_collapsed,
            'below_sampling_cut': below_sampling_cut,
            'budget_skipped': budget_skipped,
            'selected_count': selected_count,
        },
        'retry_backlog': _next_retry_backlog(
            project_id,
            prev,
            target_date=target_date,
            recorded_at=recorded_at,
            exit_code=exit_code,
            capped=capped,
            commit_made=commit_made,
            never_run=never_run,
        ),
    }

    _write_state(path, doc)
    return doc


def _schedule_cursor(
    project_id: str,
    prev: dict | None,
    *,
    target_date: date,
    today: date,
    scheduled: bool,
) -> tuple[date | None, tuple[date, ...]]:
    """The schedule cursor to record, and the never-run dates this pass owes."""
    prev_cursor = (
        None if prev is None
        else _parse_iso_date(prev.get('last_scheduled_target_date'))
    )
    if not scheduled:
        return prev_cursor, ()
    if prev_cursor is None:
        return target_date, ()
    gap = never_run_dates(prev_cursor, target_date=target_date, today=today)
    if gap.expired:
        _warn_expired_never_run(project_id, prev_cursor, gap.expired)
    return max(prev_cursor, target_date), gap.owed


def _warn_expired_never_run(
    project_id: str, last_scheduled: date, expired: int
) -> None:
    first = last_scheduled + timedelta(days=1)
    logger.warning(
        'trickle retry backlog: abandoning %d never-run date(s) %s..%s for '
        'project=%s, already past the %d-day window — their sightings are '
        'permanently lost to the trickle',
        expired,
        first.isoformat(),
        (first + timedelta(days=expired - 1)).isoformat(),
        project_id,
        RETRY_MAX_AGE_DAYS,
    )


def record_retry(
    project_id: str,
    *,
    target_date: date,
    recorded_at: datetime,
    exit_code: int,
    capped: bool,
    commit_made: bool,
) -> dict | None:
    """Fold one RETRY pass of *target_date* into the recorded backlog.
    Returns the document it wrote, or ``None`` when it wrote nothing.

    A retry pass rewrites ONLY ``retry_backlog``, so the health record, and
    the ``recorded_at`` freshness both probes read, keep describing the
    scheduled night. With no readable predecessor there is no backlog to
    fold into: it logs one WARNING and writes nothing."""
    path = trickle_state_path(project_id)
    read = load_state(path)
    status = type(read).__name__
    prev = read.value.to_json() if isinstance(read, state_docs.Ok) else None
    if prev is None:
        logger.warning(
            'trickle state: cannot record retry of %s for project=%s: no '
            'readable trickle state (%s)',
            target_date.isoformat(),
            project_id,
            status,
        )
        return None

    doc = {
        **prev,
        'retry_backlog': _next_retry_backlog(
            project_id,
            prev,
            target_date=target_date,
            recorded_at=recorded_at,
            exit_code=exit_code,
            capped=capped,
            commit_made=commit_made,
        ),
    }
    _write_state(path, doc)
    return doc


def _next_retry_backlog(
    project_id: str,
    prev: dict | None,
    *,
    target_date: date,
    recorded_at: datetime,
    exit_code: int,
    capped: bool,
    commit_made: bool,
    never_run: tuple[date, ...] = (),
) -> dict[str, int]:
    today = recorded_at.date()
    fold = fold_retry_backlog(
        parse_retry_backlog(None if prev is None else prev.get('retry_backlog')),
        target_date=target_date,
        owes_retry=owes_retry(
            exit_code=exit_code, capped=capped, commit_made=commit_made
        ),
        today=today,
        never_run=never_run,
    )
    for owed, attempts in sorted(fold.abandoned.items()):
        logger.warning(
            'trickle retry backlog: abandoning %s for project=%s after %d '
            'unfinished attempt(s), age %d day(s) (limits: %d attempts, %d '
            'days) — its sightings are permanently lost to the trickle',
            owed.isoformat(),
            project_id,
            attempts,
            (today - owed).days,
            RETRY_MAX_ATTEMPTS,
            RETRY_MAX_AGE_DAYS,
        )
    return _serialize_retry_backlog(fold.backlog)


def _write_state(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + '.tmp')
    try:
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(doc, f, indent=2, sort_keys=True)
            f.write('\n')
        os.replace(tmp_path, path)
    except OSError:
        with contextlib.suppress(OSError):
            tmp_path.unlink()
        raise


DEFAULT_MAX_BARREN_RUNS = 3
"""Consecutive barren runs before ``nightly.py`` escalates and
``check_trickle_progress.py`` fails.

WHY 3. One barren night can be an ordinary bad day — an unusually large
session, a transient spike that ate the budget. THREE CONSECUTIVE means
PERSISTENT CONFIG STATE: the budget or the sampling cut is simply wrong
for this project's session sizes, and no night is going to fix itself.
That is the same reasoning ``nightly._report_sample_outcome`` already
records for why total suppression repeats every night, applied to a
window instead of a single run. The real 2026-07-16..29 incident would
have fired on night 3 of 14 rather than never.

A MODULE CONSTANT, NOT A ``legibility.yaml`` FIELD, deliberately: reading
that config would drag pydantic + PyYAML onto the bare-``python3``
predicate path this module must survive (see the module docstring). The
progress probe also takes the threshold as an argument, so a binding that
wants a different window passes one rather than editing config.
"""

DEFAULT_MAX_FAILED_RUNS = 2
"""Consecutive FAILED runs before ``check_trickle_progress.py`` fails.

WHY 2, AND WHY TIGHTER THAN ``DEFAULT_MAX_BARREN_RUNS = 3``. One failed
night is already owned TWICE OVER — by ``nightly.py``'s own fail-loud
escalation for that run, and by ``check_trickle_liveness.sh``'s ``Result
!= success`` gate — so firing at 1 would only duplicate them. TWO
consecutive is the PERSISTENT shape neither per-run signal can express,
and it is what the permanently-broken-coder scenario produces on night
two. The barren threshold is looser because one barren night can be an
ordinary bad day, whereas a night that did not finish is already an
anomaly on its own.

A MODULE CONSTANT, NOT A ``legibility.yaml`` FIELD, for the same reason
as its sibling above: reading that config would drag pydantic + PyYAML
onto the bare-``python3`` predicate path this module must survive.
"""

DEFAULT_MAX_UNADVANCED_CENSUS_FIRES = 3
"""Consecutive census-trigger FIREs against one unchanged ``last_census_at``
before ``check_census_progress.py`` fails.

WHY 3. One or two unadvanced fires can be a census deferred for usage
headroom, which ``census.py``'s ``_defer`` already escalates at info. A
census that DID advance blocks the trigger for the next ``floor_days``
(stock 5) nights, so three consecutive unadvanced fires are persistent.
The threshold bounds how many unadvanced paid censuses run BEFORE an
operator is alerted, not how many run: nothing here stops the trigger, so
every later FIRE still launches one until someone acts. A deferred census
COUNTS toward the streak: nothing the pipeline records tells it apart.

A MODULE CONSTANT, NOT A ``legibility.yaml`` FIELD, for the same reason
as its siblings above.
"""
