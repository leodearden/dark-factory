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
CARRIED FORWARD under a crashed run, and each names ITS OWN remedy — a
barren streak the specific door it left by, a failed streak the journal, a
carried-forward streak the journal first and no door at all. A probe that
cannot say WHICH absence it found is the trap this script exists to close,
and a verdict that reads the LAST night's door counters as though they
described an EARLIER night's streak is that same trap wearing the right
label.

THE VOCABULARY GAINED ``failed`` IN TASK 4514, and the reason is that a
run can flow signal IN and still break downstream. Before it, such a night
classified ``productive``: the 2026-08-18 reify run selected six digests,
exited 1, applied 0 and made no commit, and this probe reported "OK: last
run was productive 0h ago" — indefinitely, for as long as the coder stayed
broken. ``selected_count > 0`` answers "did signal flow", never "did the
night finish".
"""
from __future__ import annotations

import sys
from pathlib import Path

# The deterministic runner EXECs a bound predicate directly
# (`deterministic_runner._default_run_script` ->
# `asyncio.create_subprocess_exec(script, *args)`) — no `uv run` wrapper,
# no project venv, no package context. So `trickle_state` must be
# importable as a bare top-level module from this file's own directory.
# DO NOT "clean this up" into a package-relative import: it would break the
# predicate at DISPATCH time, not at test time.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import trickle_state  # noqa: E402  (must follow the sys.path shim above)

USAGE = (
    'usage: check_trickle_progress.py <project_id> <max_barren_runs> '
    '[max_age_hours] [max_failed_runs]'
)


def main(argv: list[str]) -> int:
    args = argv[1:]
    # Arity + numeric guard, mirroring check_trickle_liveness.sh's
    # `[ $# -ne 2 ]` shape.
    if len(args) not in (2, 3, 4):
        print(USAGE, file=sys.stderr)
        return 1

    project_id = args[0]
    try:
        max_barren_runs = int(args[1])
        max_age_hours = int(args[2]) if len(args) >= 3 else 72
        max_failed_runs = (
            int(args[3]) if len(args) == 4
            else trickle_state.DEFAULT_MAX_FAILED_RUNS
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
    status, doc = trickle_state.load_state(path)

    # 1. Never recorded a run at all. Distinct from a barren streak: the
    #    recorder itself has never fired for this project.
    if status == 'missing':
        print(
            f'ERROR: legibility trickle for {project_id} has never recorded '
            f'a run (no state file at {path}). Either the nightly pipeline '
            f'has never run for this project, or it is running a build that '
            f'predates run-state recording.',
            file=sys.stderr,
        )
        return 1

    # 2. Recorded, but the record cannot be read. Fail loud rather than
    #    guessing at a verdict from a document we do not understand.
    if status != 'ok' or doc is None:
        print(
            f'ERROR: legibility trickle state for {project_id} is '
            f'unreadable/corrupt at {path}. The pipeline may be healthy; the '
            f'RECORD is not, so progress cannot be assessed.',
            file=sys.stderr,
        )
        return 1

    # 3. Stale recorder: the pipeline stopped writing state at all. The
    #    recorded outcome alone would still read healthy here, which is
    #    exactly why freshness is checked before the streak.
    #
    #    The arithmetic (and the naive-stamp-is-UTC normalization) lives in
    #    `trickle_state.recorded_age_hours`, so this probe and
    #    `check_trickle_health.py`'s suppression rule cannot drift into two
    #    readings of "fresh". The VERDICTS stay here: `None` means
    #    freshness cannot be assessed, and what that is worth is each
    #    caller's decision, not the helper's.
    age_hours = trickle_state.recorded_age_hours(doc)

    if age_hours is None:
        print(
            f'ERROR: legibility trickle state for {project_id} has an '
            f'unparseable recorded_at {doc.get("recorded_at")!r} at {path}; '
            f'freshness cannot be assessed.',
            file=sys.stderr,
        )
        return 1

    if age_hours > max_age_hours:
        print(
            f'ERROR: legibility trickle for {project_id} last recorded a run '
            f'{age_hours:.0f}h ago, exceeding the {max_age_hours}h window '
            f'(state file {path}). The pipeline has stopped recording '
            f'state — check the timer with check_trickle_liveness.sh.',
            file=sys.stderr,
        )
        return 1

    streak = doc.get('consecutive_barren_runs')
    streak = streak if isinstance(streak, int) else 0
    outcome = doc.get('outcome')
    counters = doc.get('counters') or {}
    budget_skipped = counters.get('budget_skipped') or 0
    below_sampling_cut = counters.get('below_sampling_cut') or 0

    # 4. Failed streak: the run did not COMPLETE, for max_failed_runs
    #    consecutive runs.
    #
    #    CHECKED BEFORE THE BARREN STREAK, and not merely for tidiness. A
    #    run that did not complete tells you nothing reliable about whether
    #    the sampling/budget doors are the problem, and `record_run`
    #    deliberately CARRIES `consecutive_barren_runs` FORWARD across
    #    failed runs rather than advancing it — so a barren streak read
    #    during a failure window is stale by construction. Report the
    #    failure first.
    failed_streak = doc.get('consecutive_failed_runs')
    failed_streak = failed_streak if isinstance(failed_streak, int) else 0

    if failed_streak >= max_failed_runs:
        # WHERE the pipeline broke is CONDITIONAL on the recorded counter,
        # never asserted. `nightly.run_nightly` pre-seeds
        # `NightlyResult(exit_code=1, ...)` BEFORE the digest stage and
        # records that sentinel whenever a later stage raises, so a night
        # that selected nothing at all — a quiet night, a codebook/merge
        # crash, an extractor crash — records `failed` with
        # selected_count=0. Measured before this branch existed: two such
        # nights printed "Signal DID reach the digest stage
        # (selected_count=0)", a sentence contradicting itself inside its
        # own parenthesis, in the one verdict whose whole job is to steer
        # the remedy.
        selected_count = counters.get('selected_count') or 0
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
        # The two barren-door config keys are named NOWHERE in either
        # wording, not even to say they do not apply. This verdict's whole
        # job is to steer an operator AWAY from them, and the sibling
        # door-specific tests discriminate which remedy was prescribed by
        # the PRESENCE of the key — so a negated mention here would read as
        # a hit to anyone (or anything) grepping the verdict for it, which
        # is the conflation this file's distinct-verdict contract forbids.
        print(
            f'ERROR: legibility trickle for {project_id} has not COMPLETED '
            f'for {failed_streak} consecutive runs (threshold '
            f'{max_failed_runs}); last recorded exit_code='
            f'{doc.get("exit_code")}. {flow} Remedy: read '
            f'journalctl --user -u legibility-trickle@{project_id} --since '
            f"'{failed_streak + 1} days ago'. "
            f'last_productive_at={doc.get("last_productive_at")}. '
            f'State file: {path}',
            file=sys.stderr,
        )
        return 1

    # 5. Barren streak at or over threshold. TWO verdicts, chosen by the
    #    RECORDED OUTCOME rather than by the streak alone, because the
    #    door counters belong to the LAST recorded night and only describe
    #    the streak when that night was itself barren.
    #
    #    `record_run` CARRIES `consecutive_barren_runs` forward across a
    #    failed run, so the history (barren, barren, barren, failed)
    #    arrives here with streak=3 under outcome='failed' and
    #    consecutive_failed_runs=1 — below the failed threshold of 2, so
    #    branch 4 did not fire. Measured before this split existed, that
    #    history printed: "has been failed for 3 consecutive runs
    #    (threshold 3): real signal reached the sampling/budget stage and
    #    nothing was digested. Doors: below_sampling_cut=0
    #    budget_skipped=0. Remedy: inspect the recorded counters — no door
    #    counter is set, which should be impossible for a barren run."
    #    The failed outcome interpolated into barren prose, zero doors for
    #    a "barren" run, and a closing sentence telling the operator that
    #    the state the writer produces by design is impossible.
    if streak >= max_barren_runs:
        if outcome == trickle_state.OUTCOME_BARREN:
            # Door-SPECIFIC remedy. SampleResult's docstring is explicit
            # that the two doors have different fixes: raising the byte
            # budget recovers a budget_skipped record and does nothing
            # whatever for a below_sampling_cut one. Never conflate them.
            remedies = []
            if budget_skipped > 0:
                remedies.append(
                    'raise budgets.max_daily_digest_bytes (candidates '
                    'competed and were ALL discarded on the byte budget)'
                )
            if below_sampling_cut > 0:
                remedies.append(
                    'raise sampling.top_fraction / sampling.per_stratum_min '
                    '(real signal ranked below its stratum cut and never '
                    'reached the budget phase)'
                )
            remedy = '; '.join(remedies) or (
                'inspect the recorded counters — no door counter is set, '
                'which should be impossible for a barren run'
            )

            message = (
                f'ERROR: legibility trickle for {project_id} has been '
                f'{outcome} for {streak} consecutive runs (threshold '
                f'{max_barren_runs}): real signal reached the '
                f'sampling/budget stage and nothing was digested. Doors: '
                f'below_sampling_cut={below_sampling_cut} '
                f'budget_skipped={budget_skipped}. Remedy: {remedy}. '
                f'State file: {path}'
            )
        else:
            # A barren streak CARRIED FORWARD under a last run that was
            # not barren — in practice always `failed`, since `record_run`
            # RESETS the streak on productive and quiet. Still loud: three
            # unresolved barren nights do not stop being a finding because
            # the fourth night crashed, and going silent here would be the
            # permanent-suppression shape this probe exists to close. But
            # it gets its OWN verdict, naming no door, because the doors
            # in this record are the crashed night's and a crashed night
            # legitimately has none.
            message = (
                f'ERROR: legibility trickle for {project_id} is carrying a '
                f'barren streak of {streak} runs forward (threshold '
                f'{max_barren_runs}) under a last recorded outcome of '
                f'{outcome!r}: the barren streak is UNRESOLVED, and the '
                f'recorded doors (below_sampling_cut={below_sampling_cut} '
                f'budget_skipped={budget_skipped}) describe that '
                f'{outcome} run rather than the barren ones, so no '
                f'door-specific remedy can be read off this record. '
                f'Clear the {outcome} run first — read journalctl --user '
                f'-u legibility-trickle@{project_id} --since '
                f"'{streak + failed_streak + 1} days ago' — then re-read "
                f'this probe once a run has COMPLETED and re-observed the '
                f'sampling/budget doors. consecutive_failed_runs='
                f'{failed_streak} (threshold {max_failed_runs}). '
                f'last_productive_at={doc.get("last_productive_at")}. '
                f'State file: {path}'
            )

        print(message, file=sys.stderr)
        return 1

    # 6. Healthy. `outcome` is printed VERBATIM, which is what makes a
    #    sub-threshold failed night read honestly now that classify_run can
    #    say `failed` — rather than the "last run was productive 0h ago"
    #    this probe used to print for a crashed one.
    print(
        f'OK: legibility trickle for {project_id} last run was {outcome} '
        f'{age_hours:.0f}h ago, consecutive_barren_runs={streak} '
        f'(threshold {max_barren_runs}), '
        f'consecutive_failed_runs={failed_streak} '
        f'(threshold {max_failed_runs}), '
        f'last_productive_at={doc.get("last_productive_at")}.'
    )
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
