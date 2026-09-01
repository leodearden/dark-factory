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
#   item 1  the judge's verdict carries no candidate id, while the attach always
#           targets the band's top-1 -- so a verdict earned by candidate #3 is
#           filed against candidate #1, and x_contested is stamped on a canonical
#           the entry never contradicted. Harmless while the flag is off; it
#           ACTIVATES on the flip.
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

fail=0
report=''

note() { report="${report}$1"$'\n'; }

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

# --- item 1: the judge verdict must be able to name its candidate -------------
if read_ref_file "$JUDGE"; then
  judge_src="$REF_CONTENT"
  if printf '%s' "$judge_src" | grep -q 'candidate_id'; then
    note "PASS  item 1  candidate_id present in $JUDGE"
  else
    note "FAIL  item 1  candidate_id ABSENT from $JUDGE"
    note "              The judge is shown up to judge_candidate_count candidates but"
    note "              returns a bare verdict string, while the attach targets the"
    note "              band's top-1. Once the flag is on, a verdict reasoned about"
    note "              candidate #3 lands on candidate #1 and stamps x_contested on a"
    note "              canonical the entry never contradicted."
    note "              Fix (a) is the one that matches declared intent:"
    note "              build_judge_prompt's own docstring says ids are rendered"
    note "              'because the model must be able to say which candidate it"
    note "              means'. Add candidate_id to the judge's JSON contract,"
    note "              validate it against the slate in parse_judge_verdict, and"
    note "              thread it through BandDecision."
    fail=1
  fi
else
  note "FAIL  item 1  UNVERIFIABLE: cannot read $JUDGE at ref '$REF'. Failing closed."
  fail=1
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
    fail=1
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
    fail=1
  else
    note "PASS  item 4  $EVAL no longer uses with_suffix('.md') for the sibling"
  fi
else
  note "FAIL  items 2+4  UNVERIFIABLE: cannot read $EVAL at ref '$REF'. Failing closed."
  fail=1
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
  note "RESULT: preconditions NOT satisfied. Task 4762 (priority high) owns these"
  note "        fixes; see its description and details for the verbatim findings."
fi

printf '%s' "$report"
exit "$fail"
