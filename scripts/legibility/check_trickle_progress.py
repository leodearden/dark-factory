#!/usr/bin/env python3
"""Predicate: is the legibility trickle pipeline for <project_id> actually
PRODUCING anything? Exit 0 = signal is flowing, non-zero = escalate.

Usage: check_trickle_progress.py <project_id> <max_barren_runs>
           [max_age_hours] [max_failed_runs]

  max_age_hours defaults to 72, matching PRD task kappa's liveness window.
  max_failed_runs defaults to trickle_state.DEFAULT_MAX_FAILED_RUNS.

SIBLING, NOT REPLACEMENT, of check_trickle_liveness.sh. The two answer
DIFFERENT questions and neither subsumes the other:

  check_trickle_liveness.sh  — did the UNIT RUN?    (systemd unit state)
  check_trickle_progress.py  — did SIGNAL FLOW?     (this state file)

The 2026-07-16..29 incident passed the liveness probe every single night
while digesting literally nothing: the timer fired, the unit exited 0, and
the byte budget discarded every candidate. Liveness cannot see that, and
is not meant to.

NEVER INVOKES GIT, SYSTEMD, OR ANY SUBPROCESS. It reads exactly one JSON
file. WHY THIS IS NOT THE PROBE PRD DECISION 7 BANS: decision 7 forbids
inferring pipeline health from the REPO's contents (git history, codebook
mtime), because a legitimately quiet night commits nothing and an external
observer cannot tell "nothing to do" from "broken" — such a probe
false-alarms on a healthy quiet timer. This probe reads what the PIPELINE
RECORDED ABOUT ITS OWN RUN, not what the repo happens to contain. A
genuinely quiet night is recorded AS ``quiet`` (proven false-alarm-free
from SampleResult's conservation invariant — see
``trickle_state.classify_run``) and PASSES, rather than being
inferred-absent from a missing commit. Decision 7's rationale is
satisfied, not circumvented.

Every failure verdict is DISTINCT: never-recorded vs unreadable vs
stale-recorder vs FAILED streak vs barren streak vs a barren streak
CARRIED FORWARD under a crashed or a cap-deferred run, and each names ITS
OWN remedy — a barren streak the specific door it left by, a failed streak
the journal, a streak carried under a crash the journal first and no door
at all, one carried under a deferral a completed run and no door. Each
verdict on a readable record, and the OK line, prints the recorded
``barren_escalated`` (True, False, or ``unrecorded`` before the field
existed), so an operator can read whether the nightly escalated. A probe that
cannot say WHICH absence it found is the trap this script exists to close,
and a verdict that reads the LAST night's door counters as though they
described an EARLIER night's streak is that same trap wearing the right
label.

A run can flow signal IN and still break downstream, which is why
``failed`` is an outcome of its own: ``selected_count > 0`` answers "did
signal flow", never "did the night finish". A ``deferred`` night (the usage
cap deferred its coding) is weather: it reads OK on its own and never moves
a streak.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# The deterministic runner EXECs a bound predicate directly
# (`deterministic_runner._default_run_script` ->
# `asyncio.create_subprocess_exec(script, *args)`) — no `uv run` wrapper,
# no project venv, no package context. So `trickle_state` must be
# importable as a bare top-level module from this file's own directory.
# DO NOT "clean this up" into a package-relative import: it would break the
# predicate at DISPATCH time, not at test time.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import state_docs
import trickle_state

USAGE = (
    'usage: check_trickle_progress.py <project_id> <max_barren_runs> '
    '[max_age_hours] [max_failed_runs]'
)

DEFAULT_MAX_AGE_HOURS = 72


@dataclass(frozen=True)
class _Thresholds:
    max_barren_runs: int
    max_age_hours: int
    max_failed_runs: int


def main(argv: list[str]) -> int:
    args = argv[1:]
    # Arity + numeric guard, mirroring check_trickle_liveness.sh's
    # `[ $# -ne 2 ]` shape.
    if len(args) not in (2, 3, 4):
        print(USAGE, file=sys.stderr)
        return 1

    project_id = args[0]
    try:
        limits = _Thresholds(
            max_barren_runs=int(args[1]),
            max_age_hours=int(args[2]) if len(args) >= 3 else DEFAULT_MAX_AGE_HOURS,
            max_failed_runs=(
                int(args[3]) if len(args) == 4
                else trickle_state.DEFAULT_MAX_FAILED_RUNS
            ),
        )
    except ValueError:
        print(
            f'{USAGE}\n'
            f'ERROR: max_barren_runs, max_age_hours and max_failed_runs '
            f'must be integers',
            file=sys.stderr,
        )
        return 1

    path = trickle_state.trickle_state_path(project_id)
    match trickle_state.load_state(path):
        case state_docs.Missing():
            return _fail(
                f'ERROR: legibility trickle for {project_id} has never recorded '
                f'a run (no state file at {path}). Either the nightly pipeline '
                f'has never run for this project, or it is running a build that '
                f'predates run-state recording.'
            )
        case state_docs.Malformed(reason=reason):
            return _fail(
                f'ERROR: legibility trickle state for {project_id} is '
                f'unreadable/corrupt at {path}: {reason}. The pipeline may be '
                f'healthy; the RECORD is not, so progress cannot be assessed.'
            )
        case state_docs.Ok(value=state):
            return _judge(project_id, path, state, limits)


def _judge(
    project_id: str, path: Path, state: trickle_state.TrickleState,
    limits: _Thresholds,
) -> int:
    """The verdict on a readable record; the first failing check wins.

    Freshness comes first because a recorder that stopped writing leaves a
    record that still reads healthy. The failed streak comes before the
    barren one because the barren streak is carried across failed runs, so
    during a failure window it is stale by construction."""
    age_hours = trickle_state.recorded_age_hours(state)
    if age_hours > limits.max_age_hours:
        return _fail(
            f'ERROR: legibility trickle for {project_id} last recorded a run '
            f'{age_hours:.0f}h ago, exceeding the {limits.max_age_hours}h '
            f'window (state file {path}). The pipeline has stopped recording '
            f'state — check the timer with check_trickle_liveness.sh. '
            f'{_escalation_fact(state)}.'
        )
    if state.consecutive_failed_runs >= limits.max_failed_runs:
        return _fail(_failed_streak_verdict(project_id, path, state, limits))
    if trickle_state.barren_streak_due(state, limits.max_barren_runs):
        return _fail(_barren_streak_verdict(project_id, path, state, limits))

    print(
        f'OK: legibility trickle for {project_id} last run was '
        f'{state.outcome} {age_hours:.0f}h ago, '
        f'consecutive_barren_runs={state.consecutive_barren_runs} '
        f'(threshold {limits.max_barren_runs}), '
        f'consecutive_failed_runs={state.consecutive_failed_runs} '
        f'(threshold {limits.max_failed_runs}), '
        f'last_productive_at={_isoformat(state.last_productive_at)}, '
        f'{_escalation_fact(state)}.'
    )
    return 0


def _failed_streak_verdict(
    project_id: str, path: Path, state: trickle_state.TrickleState,
    limits: _Thresholds,
) -> str:
    # WHERE the pipeline broke is conditional on the recorded counter: a
    # night that raised before selecting anything records `failed` with
    # selected_count=0, and then where signal stopped is unknown. Neither
    # wording names a barren-door config key, not even to disclaim it: this
    # verdict steers the operator AWAY from them.
    selected_count = state.counters.selected_count
    if selected_count > 0:
        flow = (
            f'Signal DID reach the digest stage '
            f'(selected_count={selected_count}) and the pipeline broke '
            f'DOWNSTREAM of it, so this is NOT a budget or sampling-cut '
            f'problem and the barren-streak remedies do not apply.'
        )
    else:
        flow = (
            'The run selected nothing (selected_count=0) AND did not '
            'complete, so the recorded counters describe an UNFINISHED '
            'night and where signal stopped is UNKNOWN — which is also '
            'why no barren-streak remedy can be read off this record.'
        )
    failed_streak = state.consecutive_failed_runs
    return (
        f'ERROR: legibility trickle for {project_id} has not COMPLETED '
        f'for {failed_streak} consecutive runs (threshold '
        f'{limits.max_failed_runs}); last recorded exit_code='
        f'{state.exit_code}. {flow} Remedy: read '
        f'journalctl --user -u legibility-trickle@{project_id} --since '
        f"'{failed_streak + 1} days ago'. "
        f'last_productive_at={_isoformat(state.last_productive_at)}. '
        f'{_escalation_fact(state)}. State file: {path}'
    )


def _barren_streak_verdict(
    project_id: str, path: Path, state: trickle_state.TrickleState,
    limits: _Thresholds,
) -> str:
    """The door counters belong to the LAST recorded night, so only a barren
    last night names a door. A streak carried under a deferred or failed
    night stays loud, with a verdict of its own and no door."""
    streak = state.consecutive_barren_runs
    counters = state.counters
    facts = (
        f'last_productive_at={_isoformat(state.last_productive_at)}. '
        f'{_escalation_fact(state)}. State file: {path}'
    )
    match state.outcome:
        case trickle_state.RunOutcome.BARREN:
            return (
                f'ERROR: legibility trickle for {project_id} has been '
                f'{state.outcome} for {streak} consecutive runs (threshold '
                f'{limits.max_barren_runs}): real signal reached the '
                f'sampling/budget stage and nothing was digested. Doors: '
                f'below_sampling_cut={counters.below_sampling_cut} '
                f'budget_skipped={counters.budget_skipped}. Remedy: '
                f'{trickle_state.barren_remedy(counters)}. {facts}'
            )
        case trickle_state.RunOutcome.DEFERRED:
            return (
                f'ERROR: legibility trickle for {project_id} is carrying a '
                f'barren streak of {streak} runs (threshold '
                f'{limits.max_barren_runs}) under a last run whose coding was '
                f'deferred by a usage cap: the streak is UNRESOLVED, and the '
                f'deferred night coded nothing, so it re-observed neither '
                f'sampling/budget door and no door-specific remedy can be read '
                f'off this record. The deferral needs no action and clears at '
                f'the cap reset; re-read this probe once a run has COMPLETED. '
                f'{facts}'
            )
        case _:
            failed_streak = state.consecutive_failed_runs
            return (
                f'ERROR: legibility trickle for {project_id} is carrying a '
                f'barren streak of {streak} runs forward (threshold '
                f'{limits.max_barren_runs}) under a last recorded outcome of '
                f'{state.outcome.value!r}: the barren streak is UNRESOLVED, '
                f'and the recorded doors (below_sampling_cut='
                f'{counters.below_sampling_cut} budget_skipped='
                f'{counters.budget_skipped}) describe that {state.outcome} '
                f'run rather than the barren ones, so no door-specific remedy '
                f'can be read off this record. Clear the {state.outcome} run '
                f'first — read journalctl --user -u '
                f'legibility-trickle@{project_id} --since '
                f"'{streak + failed_streak + 1} days ago' — then re-read "
                f'this probe once a run has COMPLETED and re-observed the '
                f'sampling/budget doors. consecutive_failed_runs='
                f'{failed_streak} (threshold {limits.max_failed_runs}). '
                f'{facts}'
            )


def _escalation_fact(state: trickle_state.TrickleState) -> str:
    """Whether the nightly escalated this night's barren streak, as
    recorded; ``unrecorded`` for a document written before the field."""
    recorded = 'unrecorded' if state.barren_escalated is None else state.barren_escalated
    return f'barren_escalated={recorded}'


def _isoformat(stamp: datetime | None) -> str | None:
    return None if stamp is None else stamp.isoformat()


def _fail(verdict: str) -> int:
    print(verdict, file=sys.stderr)
    return 1


if __name__ == '__main__':
    sys.exit(main(sys.argv))
