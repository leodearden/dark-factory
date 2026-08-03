#!/usr/bin/env python3
"""Predicate: is the legibility trickle pipeline for <project_id> actually
PRODUCING anything? Exit 0 = signal is flowing, non-zero = escalate.

Usage: check_trickle_progress.py <project_id> <max_barren_runs> [max_age_hours]

  max_age_hours defaults to 72, matching PRD task kappa's liveness window.

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
stale-recorder vs barren streak, and a barren streak names the specific
door and ITS OWN remedy. A probe that cannot say WHICH absence it found is
the trap this script exists to close.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
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
    '[max_age_hours]'
)


def main(argv: list[str]) -> int:
    args = argv[1:]
    # Arity + numeric guard, mirroring check_trickle_liveness.sh's
    # `[ $# -ne 2 ]` shape.
    if len(args) not in (2, 3):
        print(USAGE, file=sys.stderr)
        return 1

    project_id = args[0]
    try:
        max_barren_runs = int(args[1])
        max_age_hours = int(args[2]) if len(args) == 3 else 72
    except ValueError:
        print(
            f'{USAGE}\n'
            f'ERROR: max_barren_runs and max_age_hours must be integers',
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
    recorded_at = doc.get('recorded_at')
    try:
        stamp = datetime.fromisoformat(str(recorded_at))
    except (TypeError, ValueError):
        print(
            f'ERROR: legibility trickle state for {project_id} has an '
            f'unparseable recorded_at {recorded_at!r} at {path}; freshness '
            f'cannot be assessed.',
            file=sys.stderr,
        )
        return 1

    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    age_hours = (datetime.now(timezone.utc) - stamp).total_seconds() / 3600.0

    if age_hours > max_age_hours:
        print(
            f'ERROR: legibility trickle for {project_id} last recorded a run '
            f'{age_hours:.0f}h ago, exceeding the {max_age_hours}h window '
            f'(state file {path}). The pipeline has stopped recording '
            f'state — check the timer with check_trickle_liveness.sh.',
            file=sys.stderr,
        )
        return 1

    # 4. Barren streak: signal reached the sampling/budget stage and
    #    NOTHING was digested, for max_barren_runs consecutive runs.
    streak = doc.get('consecutive_barren_runs')
    streak = streak if isinstance(streak, int) else 0
    outcome = doc.get('outcome')
    counters = doc.get('counters') or {}
    budget_skipped = counters.get('budget_skipped') or 0
    below_sampling_cut = counters.get('below_sampling_cut') or 0

    if streak >= max_barren_runs:
        # Door-SPECIFIC remedy. SampleResult's docstring is explicit that
        # the two doors have different fixes: raising the byte budget
        # recovers a budget_skipped record and does nothing whatever for a
        # below_sampling_cut one. Never conflate them.
        remedies = []
        if budget_skipped > 0:
            remedies.append(
                'raise budgets.max_daily_digest_bytes (candidates competed '
                'and were ALL discarded on the byte budget)'
            )
        if below_sampling_cut > 0:
            remedies.append(
                'raise sampling.top_fraction / sampling.per_stratum_min '
                '(real signal ranked below its stratum cut and never '
                'reached the budget phase)'
            )
        remedy = '; '.join(remedies) or (
            'inspect the recorded counters — no door counter is set, which '
            'should be impossible for a barren run'
        )

        print(
            f'ERROR: legibility trickle for {project_id} has been {outcome} '
            f'for {streak} consecutive runs (threshold {max_barren_runs}): '
            f'real signal reached the sampling/budget stage and nothing was '
            f'digested. Doors: below_sampling_cut={below_sampling_cut} '
            f'budget_skipped={budget_skipped}. Remedy: {remedy}. '
            f'State file: {path}',
            file=sys.stderr,
        )
        return 1

    # 5. Healthy.
    print(
        f'OK: legibility trickle for {project_id} last run was {outcome} '
        f'{age_hours:.0f}h ago, consecutive_barren_runs={streak} '
        f'(threshold {max_barren_runs}), '
        f'last_productive_at={doc.get("last_productive_at")}.'
    )
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
