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

Division of labour between the two probes:

- ``check_trickle_liveness.sh``  — did the UNIT RUN? (systemd unit state)
- ``check_trickle_progress.py``  — did SIGNAL FLOW? (this state file)

STDLIB-ONLY IMPORTS ARE A HARD CONSTRAINT, NOT A STYLE PREFERENCE.
``check_trickle_progress.py`` imports this module and, when bound as a
``before_done`` predicate, is EXEC'd directly by
``deterministic_runner._default_run_script``
(``asyncio.create_subprocess_exec(script, *args)``) — no ``uv run``
wrapper, no project venv, bare ``#!/usr/bin/env python3``. Meanwhile
``nightly.py`` and ``legibility.config`` both pull in PyYAML. A stray
third-party import HERE would break the predicate at DISPATCH time, not
at test time — i.e. it would fail in exactly the silent way this module
exists to prevent. Same reason ``DEFAULT_MAX_BARREN_RUNS`` is a module
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
import json
import logging
import os
import pwd
import tempfile
from datetime import UTC, date, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

STATE_FILENAME = 'trickle-state.json'
"""Basename of the per-project run-state file."""

STATE_SCHEMA_VERSION = 1
"""Version of the recorded document's shape. :func:`load_state` treats any
OTHER value as ``malformed`` rather than guessing at unknown fields — a
reader that silently accepts a shape it does not understand is exactly the
silent-degradation mode this module exists to close.

DELIBERATELY NOT BUMPED for ``consecutive_failed_runs`` (task 4514). That
field is ADDITIVE and every reader fetches it via ``.get()`` with a 0
fallback, so no reader can MIS-READ a document that lacks it. Bumping
would make every live state file on every project read ``malformed`` on
the first post-deploy probe — one guaranteed false alarm per project,
plus a reset streak — which is precisely the degradation this module
exists to close.

THE RULE FOR THE NEXT PERSON: bump only for a change that would make an
OLD reader MIS-READ a NEW document. Adding a field an old reader ignores
is not that; changing the meaning or type of an existing field is."""


# ---------------------------------------------------------------------------
# Outcome vocabulary
# ---------------------------------------------------------------------------

OUTCOME_PRODUCTIVE = 'productive'
"""Digests were built: ``selected > 0``."""

OUTCOME_QUIET = 'quiet'
"""Nothing was digested and nothing reached the sampling/budget stage — a
legitimately quiet or dormant night. NEVER an alarm."""

OUTCOME_BARREN = 'barren'
"""Nothing was digested even though real, distinct, non-duplicate signal
DID reach the sampling/budget stage. This is the absence mode that looks
identical to a quiet night from outside the pipeline."""

OUTCOME_FAILED = 'failed'
"""The run did not COMPLETE (``exit_code != 0``), whatever the sampler
counters say.

Signal may well have flowed IN — the 2026-08-18 reify run selected six
digests — but the pipeline broke downstream, so those counters describe a
night whose work was never finished. The counters are still recorded, so
the "was signal flowing" question stays answerable.

THE VOCABULARY HOLE THIS CLOSES. Without a fourth outcome, a permanently
broken coder samples > 0, storms, exits 1, and records ``productive`` /
streak 0 / a fresh ``last_productive_at`` EVERY NIGHT FOREVER, while
``check_trickle_progress.py`` prints "OK: last run was productive 0h ago"
indefinitely. That is the 2026-07-16..29 silent-degradation shape (task
3270) entering through a different door."""


# ---------------------------------------------------------------------------
# classify_run — the three-valued absence classifier
# ---------------------------------------------------------------------------

def classify_run(
    *,
    exit_code: int,
    total_records: int,
    zero_signal_dropped: int,
    dedupe_collapsed: int,
    below_sampling_cut: int,
    budget_skipped: int,
    selected_count: int,
) -> str:
    """Classify one nightly trickle run as failed / productive / barren /
    quiet.

    DERIVED FROM, not tuned against, :class:`sampling.SampleResult`'s
    conservation identity::

        total_records == zero_signal_dropped + dedupe_collapsed
                         + below_sampling_cut + budget_skipped
                         + len(selected)

    Four branches:

    1. ``exit_code != 0``                                    -> failed.
       The run did not finish. See below for why this is read FIRST.
    2. ``selected_count > 0``                                -> productive.
       Digests were built. This deliberately INCLUDES a night that also
       skipped records on budget: a partially-truncated night is the byte
       budget working as designed, never an absence.
    3. ``(budget_skipped + below_sampling_cut) > 0``          -> barren.
       Both are doors that only records with real, distinct,
       non-duplicate signal can leave by, so reaching this branch proves
       genuine signal existed and NOTHING was digested. The two doors are
       kept separate in the recorded counters because they have DIFFERENT
       remedies (``budgets.max_daily_digest_bytes`` vs
       ``sampling.top_fraction``/``per_stratum_min`` — SampleResult's own
       docstring is explicit that conflating them is wrong), but for the
       PRESENCE question they are one signal: real signal in, nothing out.
    4. otherwise                                             -> quiet.

    WHY ``failed`` TAKES PRIORITY. ``selected_count > 0`` proves signal
    reached the digest stage; it does NOT prove the night FINISHED. When
    both are true the operator needs to know the run BROKE — the counters
    still answer "was signal flowing", and they are recorded either way,
    so nothing is hidden by reading the exit code first.

    ``exit_code`` IS A REQUIRED KEYWORD-ONLY PARAMETER WITH NO DEFAULT,
    deliberately. A defaulted ``0`` is exactly how a future caller would
    silently reintroduce the hole this branch closes, so the parameter is
    impossible to forget.

    ``exit_code`` is READ here and NEVER WRITTEN. Writing
    ``result.exit_code`` from an observability path is the
    permanent-false-alarm inversion ``scripts/legibility/nightly.py::
    _escalate_barren_streak`` refuses in writing.

    WHY BRANCH 4 IS PROVABLY SAFE — the no-false-alarm guarantee, and it
    is UNWEAKENED BY CONSTRUCTION by the new first branch, which is gated
    purely on ``exit_code != 0``: a night that exited 0 reaches the
    remaining three branches untouched. Reaching the ``else`` means
    ``selected_count == 0`` and both cut counters are 0, so by the
    identity ``total_records == zero_signal_dropped + dedupe_collapsed``:
    every enumerated record left by the zero-signal or dedupe door, or
    nothing was enumerated at all. That is EXACTLY the "genuinely quiet
    night" PRD decision 7 protects, so a quiet or dormant project can
    never be classified barren. This is a proof from the invariant, not a
    threshold someone picked — which is what lets a progress probe exist
    without re-opening decision 7's false-alarm objection.

    ``total_records`` and ``zero_signal_dropped`` are accepted (and
    RECORDED by :func:`record_run`) but deliberately NOT consulted by the
    branch logic — they are what an operator reads to understand the shape
    of a night, and the identity above is what makes the classification
    auditable after the fact. Do not "simplify" them out of the signature
    or out of the recorded state.
    """
    if exit_code != 0:
        return OUTCOME_FAILED
    if selected_count > 0:
        return OUTCOME_PRODUCTIVE
    if (budget_skipped + below_sampling_cut) > 0:
        return OUTCOME_BARREN
    return OUTCOME_QUIET


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
# load_state — three-valued reader (mirrors census_trigger.load_census_state)
# ---------------------------------------------------------------------------

def load_state(path: str | Path) -> tuple[str, dict | None]:
    """Read a trickle run-state file. Three-valued, never raises.

    - path does not exist -> ``("missing", None)``, NO warning logged. A
      project that has never recorded a run is a normal, expected state,
      not a degradation.
    - unreadable / invalid JSON / non-dict top level / unknown
      ``schema_version`` -> ``("malformed", None)`` + exactly ONE WARNING.
      An unknown schema version is malformed rather than best-effort
      parsed: a reader that silently accepts a shape it does not
      understand reintroduces exactly the silent degradation this module
      exists to close.
    - otherwise -> ``("ok", data)``.

    Shape AND vocabulary mirror the sibling reader in this same package,
    :func:`census_trigger.load_census_state` (census_trigger.py:239),
    VERBATIM — same tuple shape, same literal status strings
    (``'malformed'``, never ``'invalid'``), same fail-safe posture — so
    the two state readers in ``scripts/legibility/`` cannot drift into two
    vocabularies for one concept.
    """
    path = Path(path)
    if not path.exists():
        return 'missing', None

    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning('trickle state at %s is malformed: %s', path, exc)
        return 'malformed', None

    if not isinstance(data, dict):
        logger.warning(
            'trickle state at %s is malformed: expected a JSON object, got %s',
            path,
            type(data).__name__,
        )
        return 'malformed', None

    schema_version = data.get('schema_version')
    if schema_version != STATE_SCHEMA_VERSION:
        logger.warning(
            'trickle state at %s is malformed: unknown schema_version %r '
            '(this module writes %d)',
            path,
            schema_version,
            STATE_SCHEMA_VERSION,
        )
        return 'malformed', None

    return 'ok', data


# ---------------------------------------------------------------------------
# recorded_age_hours — the freshness reading BOTH probes share
# ---------------------------------------------------------------------------

def recorded_age_hours(doc) -> float | None:
    """Hours elapsed since *doc*'s ``recorded_at``, or ``None``.

    ``None`` means FRESHNESS CANNOT BE ASSESSED — deliberately not "old"
    and deliberately not "fresh". Each caller decides what that means,
    which is precisely what lets the two callers below disagree about it
    without either re-deriving the arithmetic:

    - ``check_trickle_progress.py`` reports its own distinct
      unparseable-``recorded_at`` verdict, separate from its stale one,
      because that file's contract is that every failure verdict names its
      OWN remedy;
    - ``check_trickle_health.py::_should_escalate`` treats it as
      post-worthy, matching the posture it already takes toward
      ``missing``/``malformed`` — a record whose freshness cannot be
      established must never be trusted to prove someone else already
      alarmed.

    A NAIVE ``recorded_at`` is read as UTC, which is what the progress
    probe's staleness branch did before this helper existed.

    Stdlib-only, like the rest of this module: it is on the
    bare-``python3`` predicate path (see the module docstring)."""
    if not isinstance(doc, dict):
        return None
    try:
        stamp = datetime.fromisoformat(str(doc.get('recorded_at')))
    except (TypeError, ValueError):
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=UTC)
    return (datetime.now(UTC) - stamp).total_seconds() / 3600.0


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
) -> dict:
    """Classify this run, fold it into the running streak, and write the
    state file atomically. Returns the document it wrote.

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
    run and carried forward UNCHANGED across barren and quiet runs. It is
    the field an operator reads to answer "when did this pipeline last
    actually do anything".

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

    prev_status, prev = load_state(path)
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
        total_records=total_records,
        zero_signal_dropped=zero_signal_dropped,
        dedupe_collapsed=dedupe_collapsed,
        below_sampling_cut=below_sampling_cut,
        budget_skipped=budget_skipped,
        selected_count=selected_count,
    )

    recorded_at_iso = recorded_at.isoformat()
    if outcome == OUTCOME_PRODUCTIVE:
        last_productive_at = recorded_at_iso

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
        'consecutive_barren_runs': (
            prev_streak + 1 if outcome == OUTCOME_BARREN
            else prev_streak if outcome == OUTCOME_FAILED
            else 0
        ),
        'consecutive_failed_runs': (
            prev_failed + 1 if outcome == OUTCOME_FAILED else 0
        ),
        'last_productive_at': last_productive_at,
        'counters': {
            'total_records': total_records,
            'zero_signal_dropped': zero_signal_dropped,
            'dedupe_collapsed': dedupe_collapsed,
            'below_sampling_cut': below_sampling_cut,
            'budget_skipped': budget_skipped,
            'selected_count': selected_count,
        },
    }

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

    return doc


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
