#!/usr/bin/env bash
# before_done predicate for task 3169 — the write_triage_enabled flip gate.
#
# WHY THIS EXISTS
# ---------------
# Task 3169 is a pure human gate: a person edits `write_triage.enabled` in
# fused-memory/config/config.yaml and restarts the service. Nothing in this repo
# can prevent that edit, and this script does not try to. What it does is make a
# PREMATURE flip machine-detected and permanently blocking on the dependent leaf
# (task 3131), in a way that resolving the escalation cannot wave through:
# DeterministicRunner.run re-runs this predicate on EVERY resume, before the
# resume-to-done branch, because "resolving the escalation alone is NOT proof the
# invariant now holds".
#
# The invariant: three review findings from task 3128's cycle-2 review must be on
# main BEFORE the judge goes live. They are numbered here by the RECOVERED
# PAYLOAD's numbering (task 4762's own enumeration), NOT by the cycle-2 artifact's
# issue numbers -- payload 1/2/4 map to artifact issues #3/#6/#8, so a reader who
# follows these numbers into reviews-cycle-2/ reads an entirely different set.
#
#   item 1  the judge path does not bind a verdict to a determinate candidate:
#           the judge is shown several candidates while the attach always targets
#           the band's top-1 -- so a verdict earned by candidate #3 is filed
#           against candidate #1, and x_contested is stamped on a canonical the
#           entry never contradicted. Harmless while the flag is off; it
#           ACTIVATES on the flip.
#
#           Checked by EXECUTING the ref's judge module, via
#           scripts/check_write_triage_attach_target.py. It used to be a grep of
#           that module's source for `candidate_id`, which asserted which
#           MECHANISM landed rather than whether the invariant holds: it failed a
#           correct fix that established the invariant another way, and it passed
#           prose that changed no behaviour at all. Task 4810 replaced it. EITHER
#           remedy now closes item 1 -- a verdict that names its own candidate
#           (option a, task 4798 item 7), or a prompt told which candidate the
#           attach will touch whose rendering actually depends on it (option b,
#           task 4762). Marking candidates[0] is NOT one of them; see the probe's
#           own report for the measured reason.
#   item 2  the confusion-column order is derived by iterating a frozenset, so the
#           committed accuracy artifact is PYTHONHASHSEED-dependent. Measured
#           2026-08-27: the committed .md and .json disagree on column order, so
#           the .md the flip operator reads is provably not the render of the
#           committed .json. Values agree -- what is broken is traceability.
#   item 4  report_path.with_suffix('.md') means `--report-path foo.md` writes the
#           JSON and then OVERWRITES it with the markdown, losing the JSON.
#
# Items 2 and 4 corrupt or churn the very artifact step 1 of the gate tells the
# operator to read. Item 1 was re-raised as `correctness` in task 3128's fifth and
# final review verdict.
#
# Provenance: recovered from escalation esc-markup-residue-1 after a curator
# combine dropped the constraint from task 4762 on 2026-08-26. Full detail lives
# in task 4762 (description + details) and in esc-3169-1's triage note.
#
# CONTRACT
#   exit 0  -- all three fixes are on main; the flip may proceed
#   exit 1  -- at least one is missing. DeterministicRunner files a born-at-L2
#              milestone_check_failed escalation carrying this script's stdout,
#              re-stamps gate_escalated_at, and blocks task 3169 again.
#
# Assertions are made against the `main` REF, not the working tree, so a fix that
# exists only in someone's checkout does not satisfy the gate.
#
# To HOLD rather than flip, follow task 3169's own step 4: leave the gate's
# escalation OPEN. Do NOT resolve with action='abandon' -- that cancels 3169, and
# a cancelled dependency SATISFIES the scheduler's dependency check, which would
# silently unblock leaf epsilon (3131).

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REF="${WRITE_TRIAGE_GATE_REF:-main}"

JUDGE='fused-memory/src/fused_memory/server/write_triage_judge.py'
EVAL='fused-memory/scripts/eval_write_triage_judge.py'
CONF='fused-memory/config/config.yaml'

# The item-1 probe, and the interpreter that runs it. The env seam mirrors
# scripts/check_sandbox_soak.sh's CHECK_SANDBOX_SOAK_PY: it is what lets the
# hermetic tests point the probe at their own interpreter instead of resolving
# the fused-memory virtualenv.
PROBE="$REPO/scripts/check_write_triage_attach_target.py"

# The probe's own PASS line, matched literally (grep -F). Both of its PASS
# branches -- option (a) and option (b) -- emit this prefix, and no FAIL or
# UNVERIFIABLE path does. Pinned by the hermetic tests in
# scripts/tests/test_check_write_triage_flip_preconditions.py so the two
# cannot drift apart silently.
PROBE_PASS_MARKER='PASS  the judge path binds a verdict to a determinate candidate'

# The probe's marker for a pass that rests partly on a PARAMETER NAME rather
# than on behaviour alone (it accepts a target-named parameter that echoes its
# argument, because a real option (b) naming the target in a header is
# structurally identical to free text that merely interpolates it). Such a pass
# is real but unconfirmed, and a WARN buried in the probe's report is not a
# mitigation: on a PASS this gate exits 0, DeterministicRunner files no
# escalation, and this stdout is forwarded nowhere. So it gets its own machine
# -detectable channel, reported on item 1's own line AND in the report's tail.
# Pinned from both ends by the hermetic tests, like PROBE_PASS_MARKER.
PROBE_PENDING_MARKER='PASS-NEEDS-CONFIRMATION'
if [ -n "${CHECK_WRITE_TRIAGE_ATTACH_TARGET_PY:-}" ]; then
  PROBE_PY="$CHECK_WRITE_TRIAGE_ATTACH_TARGET_PY"
elif [ -x "$REPO/.venv/bin/python3" ]; then
  PROBE_PY="$REPO/.venv/bin/python3"
else
  PROBE_PY="uv run --frozen --project $REPO/fused-memory python"
fi

# Bounded well inside the before_done predicate's own 120s budget. A host
# without coreutils' `timeout` runs the probe unbounded rather than failing
# every run on a missing binary.
if command -v timeout >/dev/null 2>&1; then
  PROBE_TIMEOUT='timeout 90'
else
  PROBE_TIMEOUT=''
fi

fail=0
report=''

#: Set when item 1 passed only with the probe's name-assisted forgiveness. Not
#: a failure -- refusing it would re-block task 3169 against a valid
#: header-marking fix, the false-FAIL class this gate was rewritten to remove
#: -- but not a silent pass either.
item1_pending=0

#: The NUMBERS of the items that failed, space-separated, in check order. Read
#: only by the compact summary line at the very end of the report — see the
#: comment there for why the report's tail is the only part that reliably
#: reaches an operator.
failed_items=''

note() { report="${report}$1"$'\n'; }

# Set the exit status AND record which item earned it. Always called from this
# shell, never from inside a `$(...)` — see the note above read_ref_file.
record_fail() { fail=1; failed_items="${failed_items}${1} "; }

# item 1 extracts the ref's package tree to a temp dir. `git archive` is
# read-only and touches no .git state, unlike `git worktree add` -- which
# matters in this repo, where refs are shared across every worktree.
PROBE_TMP=''
cleanup() {
  if [ -n "${PROBE_TMP:-}" ]; then
    rm -rf "$PROBE_TMP"
  fi
}
trap cleanup EXIT

# Fail closed if the ref or a file is unreadable — an unverifiable invariant is
# not a satisfied one.
#
# NOTE the shape here is load-bearing: this sets REF_CONTENT as a side effect and
# returns a status, rather than PRINTING the content for a caller to capture with
# `$(...)`. A command substitution runs in a SUBSHELL, so a `fail=1` assigned
# inside one is discarded when it exits — an unreadable ref then skipped its whole
# check block and the script exited 0, i.e. the gate PASSED on unverifiable input.
# Caught 2026-08-27 by a `WRITE_TRIAGE_GATE_REF=no-such-ref` negative control;
# keep that control whenever this file is edited.
REF_CONTENT=''
read_ref_file() {
  if ! REF_CONTENT="$(git -C "$REPO" show "$REF:$1" 2>/dev/null)"; then
    REF_CONTENT=''
    return 1
  fi
  return 0
}

note "write_triage flip preconditions — checked against ref '$REF' in $REPO"
note ""

# --- item 1: the judge path must bind a verdict to a determinate candidate ----
#
# EVERY unverifiable outcome here calls record_fail: a missing probe, a temp dir that
# cannot be made, a failed archive, an interpreter that will not run, a probe
# crash or a timeout. An unverifiable invariant is not a satisfied one, and
# note that each record_fail below runs in THIS shell and never inside a
# `$(...)`, for the reason recorded above read_ref_file.
if [ ! -f "$PROBE" ]; then
  note "FAIL  item 1  UNVERIFIABLE: probe missing at $PROBE. Failing closed."
  record_fail 1
else
  PROBE_TMP="$(mktemp -d 2>/dev/null)"
  if [ -z "$PROBE_TMP" ] || [ ! -d "$PROBE_TMP" ]; then
    note "FAIL  item 1  UNVERIFIABLE: cannot create a temp dir to extract '$REF'. Failing closed."
    record_fail 1
  elif ! git -C "$REPO" archive "$REF" fused-memory/src 2>/dev/null \
       | tar -x -C "$PROBE_TMP" 2>/dev/null; then
    note "FAIL  item 1  UNVERIFIABLE: cannot extract fused-memory/src from ref '$REF'."
    note "              Failing closed."
    record_fail 1
  else
    # shellcheck disable=SC2086  # PROBE_TIMEOUT and PROBE_PY are command word lists.
    probe_out="$($PROBE_TIMEOUT $PROBE_PY "$PROBE" \
      --src-root "$PROBE_TMP/fused-memory/src" 2>&1)"
    probe_rc=$?
    # BELT AND BRACES: rc 0 alone is not a PASS. The probe EXECUTES the ref's
    # own judge module, so a SystemExit out of that code (a lazily-imported
    # dependency's import guard calling sys.exit()) used to terminate it with
    # THAT code, printing nothing. Measured before the fix: a judge whose only
    # statement was `raise SystemExit(0)` produced rc 0 and an EMPTY report,
    # and this branch declared PASS and the gate authorised the flip. The probe
    # now fails closed on BaseException, and this second lock means the gate
    # never again depends on the probe alone getting that right: a report that
    # does not CLAIM a pass is not one.
    if [ "$probe_rc" -eq 0 ] && printf '%s' "$probe_out" | grep -qF "$PROBE_PASS_MARKER"; then
      if printf '%s' "$probe_out" | grep -qF "$PROBE_PENDING_MARKER"; then
        item1_pending=1
        note "PASS  item 1  (NEEDS CONFIRMATION) the judge path binds a verdict to a"
        note "              determinate candidate -- but the probe accepted it partly on a"
        note "              PARAMETER NAME, not on behaviour alone. See its WARN below."
      else
        note "PASS  item 1  the judge path binds a verdict to a determinate candidate"
      fi
    elif [ "$probe_rc" -eq 0 ]; then
      note "FAIL  item 1  UNVERIFIABLE: the probe exited 0 without reporting a PASS."
      note "              Its report claims no verdict, so nothing was asserted about"
      note "              the invariant. Failing closed."
      record_fail 1
    elif [ "$probe_rc" -eq 1 ]; then
      # DELIBERATELY TERSE. The harm, BOTH accepted remedies and the
      # candidates[0] warning are all stated by the probe's own report,
      # printed directly below this and carrying the MEASURED slate. Stating
      # them here too cost ~1.1 KB of the 2000-char window that is all
      # _default_run_script forwards to the operator -- budget items 2 and 4
      # have to share. Say it once, in the copy that measured it.
      note "FAIL  item 1  the judge path does NOT bind a verdict to a determinate candidate"
      note "              Subject: $JUDGE at ref '$REF'."
      note "              The harm, BOTH accepted remedies and what was measured are in"
      note "              the probe's own report below."
      record_fail 1
    else
      note "FAIL  item 1  UNVERIFIABLE: the probe could not be run (exit $probe_rc)."
      note "              Interpreter: $PROBE_PY. Failing closed."
      record_fail 1
    fi
    # The probe's own report, indented under the verdict. It carries the
    # measured slate and, on an unverifiable outcome, its own UNVERIFIABLE line.
    note "$(printf '%s\n' "$probe_out" | sed 's/^/              /')"
  fi
fi

# --- item 2: the committed accuracy artifact must be reproducible -------------
if read_ref_file "$EVAL"; then
  eval_src="$REF_CONTENT"
  if printf '%s' "$eval_src" | grep -q 'dict\.fromkeys(TRIAGE_OUTCOMES\|list(TRIAGE_OUTCOMES)'; then
    note "FAIL  item 2  $EVAL still iterates the TRIAGE_OUTCOMES frozenset directly"
    note "              Column/key order is PYTHONHASHSEED-dependent, so the committed"
    note "              JSON and markdown churn between identical runs. Measured"
    note "              2026-08-27: the committed .md header order and the committed"
    note "              .json confusion order DISAGREE, so the report the flip operator"
    note "              reads is not the render of the committed JSON. Values agree —"
    note "              it is traceability that is broken, and step 1 of this gate"
    note "              rests on that artifact."
    note "              Fix: a module-level EVAL_OUTCOMES = tuple(sorted(TRIAGE_OUTCOMES))"
    note "              mirroring the existing EVAL_CLASSES tuple, used in both places."
    note "              Precedent for pinning it (task 4012): assert"
    note "              MD.read_text() == render_markdown(json.loads(JSON.read_text()))."
    record_fail 2
  else
    note "PASS  item 2  $EVAL no longer iterates the frozenset directly"
  fi

  # --- item 4: --report-path must not destroy its own JSON --------------------
  if printf '%s' "$eval_src" | grep -q "report_path\.with_suffix('\.md')"; then
    note "FAIL  item 4  $EVAL still derives the markdown sibling via with_suffix('.md')"
    note "              '--report-path foo.md' writes the JSON and then OVERWRITES it"
    note "              with the markdown, losing the JSON silently; 'foo.tar.gz' writes"
    note "              to foo.tar.md; a suffix-less path silently gains .md."
    note "              Fix: refuse a --report-path not ending in .json, or compose the"
    note "              sibling as parent/(stem + '.md') and assert it differs from"
    note "              report_path."
    note "              NOTE the guard_committed_report guard added post-cycle-2 does"
    note "              NOT cover this — it addresses dry-run/--limit publishing and"
    note "              returns early for any non-committed path."
    record_fail 4
  else
    note "PASS  item 4  $EVAL no longer uses with_suffix('.md') for the sibling"
  fi
else
  note "FAIL  items 2+4  UNVERIFIABLE: cannot read $EVAL at ref '$REF'. Failing closed."
  record_fail '2 4'
fi

# --- premature-flip detection -------------------------------------------------
# Reported whether or not it changes the verdict: if the flag is already true on
# main while any item is missing, the flip happened before its preconditions and
# the damage is live, not hypothetical.
if read_ref_file "$CONF"; then
  conf_src="$REF_CONTENT"
  wt_enabled="$(printf '%s' "$conf_src" \
    | awk '/^write_triage:/{f=1;next} f && /^[a-zA-Z_]/{f=0} f && /^[[:space:]]*enabled:[[:space:]]*/{print $2; exit}')"
  note ""
  note "write_triage.enabled on '$REF' = ${wt_enabled:-<unreadable>}"
  if [ "${wt_enabled:-}" = "true" ] && [ "$fail" -ne 0 ]; then
    note ""
    note "*** PREMATURE FLIP DETECTED ***"
    note "The flag is ALREADY true while at least one precondition above is unmet."
    note "This is the state this gate exists to prevent. The judge is live and"
    note "mis-attaching, and/or the accuracy artifact is unreliable. Consider"
    note "setting write_triage.enabled back to false and restarting fused-memory"
    note "before continuing."
  fi
fi

note ""
if [ "$fail" -eq 0 ]; then
  note "RESULT: all preconditions satisfied — the flip may proceed."
else
  note "RESULT: preconditions NOT satisfied. Items 2 and 4 are task 4762's (priority"
  note "        high); see its description and details for the verbatim findings."
  note "        Item 1 is closed by EITHER attach-target remedy -- option (a) is task"
  note "        4798 item 7, option (b) is task 4762 -- so whichever lands first"
  note "        satisfies it. See its report above for what was measured."
fi

# LAST, deliberately. DeterministicRunner._default_run_script returns only the
# TRAILING 2000 characters of this script's stdout, and _run_predicate feeds
# exactly that into the milestone_check_failed escalation's detail. The all-FAIL
# report is several times that and item 1 is emitted FIRST, so item 1's guidance
# -- the corrected spec an implementer is meant to read -- is precisely what
# gets truncated away. Anything that must reach the operator has to sit at the
# tail. The detailed guidance stays where it is: read in full, the report is
# still ordered for a human.
if [ -n "$failed_items" ]; then
  note "FAILING ITEMS: ${failed_items% }"
else
  note "FAILING ITEMS: none"
fi
# Same reasoning as the summary above, for the one PASS that is not a clean
# one. It sits AFTER the failing-items line because on the run where it matters
# there are no failing items, and this is then the last thing the report says.
if [ "$item1_pending" -ne 0 ]; then
  note "ITEM 1 NEEDS CONFIRMATION: it passed on a target-NAMED parameter that ECHOES"
  note "        its argument into the prompt rather than matching it against the"
  note "        candidates. That is what a real header-marking fix looks like too, so"
  note "        it is accepted -- but confirm BY EYE that the judge prompt tells the"
  note "        model which candidate the attach will touch before flipping the flag."
fi

printf '%s' "$report"
exit "$fail"
