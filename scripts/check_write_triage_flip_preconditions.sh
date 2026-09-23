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
# main BEFORE the judge goes live, plus item 5 below, which closes a gap in the
# first of them rather than being a fourth finding. They are numbered here by the RECOVERED
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
#   item 5  item 1's CONSUMPTION half, and not one of the three recovered
#           findings: it closes a gap in item 1 itself (task 4949). Item 1 stops
#           at the judge path -- it proves a verdict CAN be bound to a
#           determinate candidate, and never executes the attach. So a change
#           that only widened the parse contract opened item 1 while the write
#           still landed on the band's top-1, which is the harm item 1
#           describes, still live.
#
#           Checked by EXECUTING the ref's triage_write with an injected fake
#           judge AND by reading the ref's judge module, via
#           scripts/check_write_triage_attach_consumption.py. TWO branches
#           satisfy it, and the verdict below names the one that did -- they
#           rest on different evidence:
#             - judge-side designation swap (option a): the judge names its
#               candidate back and the attach tracks it across two different
#               designations. MEASURED, by running the write twice.
#             - judge-module attach target (option b, task 4762): judge_write
#               feeds build_judge_prompt the decision.canonical_id it already
#               holds, and triage_write is unchanged. Consumption holds BY
#               CONSTRUCTION -- announced target and attach target are the same
#               expression -- not by a measured swap.
#
#           WHAT IT DOES NOT ASSERT. The probe stops at
#           BandDecision.canonical_id -- the value tools.py::add_memory consumes
#           verbatim as `attached_to`. It does not execute that stamp, so a
#           later change to add_memory's own target selection would still pass
#           here. The probe says so on its own PASS report; confirm it
#           separately before flipping.
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
#   exit 0  -- every item below holds on main; the flip may proceed
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
TRIAGE='fused-memory/src/fused_memory/server/write_triage.py'
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
#
# COMMAND WORD LISTS, not strings. The env seam is documented to accept a
# multi-word command, which is why the old spelling was interpolated unquoted
# with SC2086 disabled -- and that split every OTHER word too, so a $REPO
# containing a space silently tore the resolved interpreter path into pieces
# and the gate reported UNVERIFIABLE on a judge that passes. An array splits
# exactly where the author put a boundary and nowhere else.
if [ -n "${CHECK_WRITE_TRIAGE_ATTACH_TARGET_PY:-}" ]; then
  # The ONE place word splitting is still wanted: the seam may carry
  # `uv run --frozen --project <dir> python`. A bare interpreter path (what the
  # hermetic tests pass) comes through as a single word unchanged.
  read -r -a PROBE_PY_CMD <<< "$CHECK_WRITE_TRIAGE_ATTACH_TARGET_PY"
elif [ -x "$REPO/.venv/bin/python3" ]; then
  PROBE_PY_CMD=("$REPO/.venv/bin/python3")
else
  PROBE_PY_CMD=(uv run --frozen --project "$REPO/fused-memory" python)
fi

# Item 5's probe: the CONSUMPTION half. Its env seam mirrors item 1's exactly,
# including the ARRAY (see the command-word-list note above) -- a $REPO
# containing a space must not tear the resolved interpreter path apart.
PROBE5="$REPO/scripts/check_write_triage_attach_consumption.py"

# Item 5's PASS line, matched literally (grep -F). BOTH of its PASS branches
# emit this prefix (see the item 5 block above), and no FAIL or
# UNVERIFIABLE path does. Pinned by the hermetic tests in
# scripts/tests/test_check_write_triage_flip_preconditions.py so the two cannot
# drift apart silently.
PROBE5_PASS_MARKER='PASS  the judge-bound candidate is CONSUMED by the attach'

# The probe's one machine-readable line naming WHICH of those two branches
# held, quoted into the verdict below. `PASS item 5` alone cannot tell an
# operator whether a swap was MEASURED or whether option (b) held BY
# CONSTRUCTION, and those authorise the production flag flip on different
# evidence. Pinned from both ends by the hermetic tests, like the marker above.
PROBE5_BRANCH_MARKER='ITEM5-BRANCH  '
# How much of the branch name is quoted back. The probe keeps that line ASCII
# for this bound: a cut inside a multi-byte character would emit a broken one.
PROBE5_BRANCH_CHARS=120

if [ -n "${CHECK_WRITE_TRIAGE_ATTACH_CONSUMPTION_PY:-}" ]; then
  read -r -a PROBE5_PY_CMD <<< "$CHECK_WRITE_TRIAGE_ATTACH_CONSUMPTION_PY"
elif [ -x "$REPO/.venv/bin/python3" ]; then
  PROBE5_PY_CMD=("$REPO/.venv/bin/python3")
else
  PROBE5_PY_CMD=(uv run --frozen --project "$REPO/fused-memory" python)
fi

#: Wall-clock bound for EACH probe, and the only place either one is tuned.
#
# THE ARITHMETIC, which a retune has to redo rather than nudge. This gate runs
# as a before_done DeterministicCheck whose descriptor pins `timeout_secs: 120`
# (fused-memory/tests/server/test_write_triage_flip_gate_invariants.py::FLIP_GATE_DELIVERED_CHECK).
# TWO probes now share that budget -- item 1's and item 5's -- so the worst
# case is 2 * PROBE_TIMEOUT_SECS, plus a few seconds of archive/tar and items
# 2 and 4. At 45 that is 90s + ~5s against 120s. This was 90 when item 1 was
# the only probe; leaving it there gave a MEASURED 181s, which overruns.
#
# And overrunning does not FAIL the check, it ERRORS it: per
# docs/task-authoring.md 3.3 an ERRORED check is a fail-safe wait with NO
# streak bump and NO escalation, so the dependent (task 3169) waits silently
# and indefinitely instead of being told which item is unmet -- strictly worse
# than the clean FAIL everything below exists to produce. So the SUM is the
# constraint, not the individual bound: do not raise this without re-checking
# 2 * PROBE_TIMEOUT_SECS against the descriptor, and do not widen the
# descriptor instead, which buys a longer silent hold with a bounded
# predicate. Measured typical for the whole gate: 10.3s.
#
# BOTH probes below build their `timeout` from this ONE value (SPOT). Bounded
# apart they could be retuned apart, and nothing in a passing run would show
# it, which is why
# scripts/tests/test_check_write_triage_flip_preconditions.py pins this
# spelling AND measures the sum with both probe subjects hanging.
PROBE_TIMEOUT_SECS=45
# A host without coreutils' `timeout` runs the probes unbounded rather than
# failing every run on a missing binary.
if command -v timeout >/dev/null 2>&1; then
  PROBE_TIMEOUT_CMD=(timeout "$PROBE_TIMEOUT_SECS")
else
  PROBE_TIMEOUT_CMD=()
fi

#: How much of the probe's STDERR is quoted back, and only where it is the
#: only diagnostic there is -- see the invocation below.
PROBE_STDERR_LINES=10

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

# Did item $1 record a failure? Derived from failed_items -- the SAME string
# the authoritative `FAILING ITEMS:` line below is built from -- so the
# RESULT block's ownership prose can never disagree with it again. A `case`
# glob, deliberately: no pipe (see item 1's SIGPIPE note above) and no
# `$(...)` subshell (see the note above read_ref_file). Tolerates the
# multi-word `record_fail '2 4'` on the unreadable-EVAL branch below: the
# comparison is " $failed_items" against "*\" $1 \"*", and failed_items
# already carries a TRAILING space per entry, so " 2 4 " matches both
# *" 2 "* and *" 4 "*.
item_failed() { case " $failed_items" in *" $1 "*) return 0 ;; esac; return 1; }

# Removed on exit however the script leaves. Filled by the single extraction
# below, which both probe items read.
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

# --- the ref's source tree, extracted ONCE for both probe items ---------------
#
# `git archive` is read-only and touches no .git state, unlike `git worktree
# add` -- which matters in this repo, where refs are shared across every
# worktree. write_triage.py and write_triage_judge.py are siblings in one tree,
# so items 1 and 5 read the same extraction: a second archive of the same
# pathspec buys nothing and costs wall clock the 120s delivered-check budget
# cannot spare (see PROBE_TIMEOUT_CMD above).
#
# EXTRACTED is read by both items, and on failure BOTH fail closed -- each at
# its own check site below, so `FAILING ITEMS` stays in ascending order and each
# item's verdict sits with its own report. EXTRACT_ERROR carries the one
# diagnosis both sites quote, so it is stated in one place. Every record_fail
# runs in THIS shell -- never inside a `$(...)`, whose assignment a subshell
# discards; that is how an unreadable ref once skipped a whole check block and
# the gate exited 0 on unverifiable input.
EXTRACTED=0
EXTRACT_ERROR=''
PROBE_TMP="$(mktemp -d 2>/dev/null)"
if [ -z "$PROBE_TMP" ] || [ ! -d "$PROBE_TMP" ]; then
  EXTRACT_ERROR="cannot create a temp dir to extract '$REF'"
elif ! git -C "$REPO" archive "$REF" fused-memory/src 2>/dev/null \
     | tar -x -C "$PROBE_TMP" 2>/dev/null; then
  EXTRACT_ERROR="cannot extract fused-memory/src from ref '$REF'"
else
  EXTRACTED=1
fi
# BEST-EFFORT, and deliberately non-fatal. The real write_triage.py imports
# shared.storm_counter, a different workspace member that no fused-memory/src
# pathspec reaches. Extracting it too means item 5 measures the REF's copy
# rather than whatever is installed. But a repo whose layout carries no
# shared/src -- every hermetic fixture repo here, and any project laid out
# differently -- must still get ordinary verdicts: a second archive that could
# fail an item would be a new way for the gate to report UNVERIFIABLE against a
# tree that is perfectly readable. So a failure here only drops the flag.
PROBE5_EXTRA_ARGS=()
if [ "$EXTRACTED" -eq 1 ] \
   && git -C "$REPO" archive "$REF" shared/src 2>/dev/null \
      | tar -x -C "$PROBE_TMP" 2>/dev/null; then
  PROBE5_EXTRA_ARGS=(--extra-path "$PROBE_TMP/shared/src")
fi

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
elif [ "$EXTRACTED" -eq 1 ]; then
  {
    # STDERR IS CAPTURED SEPARATELY, not folded in with 2>&1. The probe imports
    # the REF's tree, so anything that tree (or a transitive dependency) writes
    # at import time lands here -- measured on this checkout: a multi-line
    # PydanticDeprecatedSince20 warning from graphiti_core, above the probe's
    # own first line. Indenting that into the report spends part of the 2000
    # characters _default_run_script forwards to an operator on chatter from a
    # tree this gate does not control. It is quoted back only on the branches
    # where the gate could not read a verdict from stdout, and bounded even
    # there. The probe carries its OWN warnings on stdout for this reason.
    probe_err="$PROBE_TMP/probe.stderr"
    probe_out="$(${PROBE_TIMEOUT_CMD[@]+"${PROBE_TIMEOUT_CMD[@]}"} \
      "${PROBE_PY_CMD[@]}" "$PROBE" \
      --src-root "$PROBE_TMP/fused-memory/src" 2>"$probe_err")"
    probe_rc=$?
    # Set on the outcomes where stdout carried no verdict, so stderr is the
    # only evidence of what went wrong.
    show_probe_stderr=0
    # BELT AND BRACES: rc 0 alone is not a PASS. The probe EXECUTES the ref's
    # own judge module, so a SystemExit out of that code (a lazily-imported
    # dependency's import guard calling sys.exit()) used to terminate it with
    # THAT code, printing nothing. Measured before the fix: a judge whose only
    # statement was `raise SystemExit(0)` produced rc 0 and an EMPTY report,
    # and this branch declared PASS and the gate authorised the flip. The probe
    # now fails closed on BaseException, and this second lock means the gate
    # never again depends on the probe alone getting that right: a report that
    # does not CLAIM a pass is not one.
    # HERE-STRING, NOT A PIPE, and this is load-bearing under the `set -uo
    # pipefail` at the top of this script. `printf ... | grep -q` races: grep
    # exits the instant it matches, so printf can be killed by SIGPIPE and
    # exit 141, and pipefail then makes the whole pipeline non-zero WITH THE
    # MARKER PRESENT. Measured
    # on this host: 16 spurious failures in 3000 iterations of this exact
    # pipeline (~0.5%/call; a reviewer measured ~1.6% on a busier run), and the
    # hermetic suite below failed 3 of 8 full runs from it alone. Both harms are
    # silent-ish and opposite: a flake here reports item 1 UNVERIFIABLE against a
    # judge that demonstrably passes (re-blocking 3169 on a correct fix -- the
    # exact false-FAIL class this gate was rewritten to remove), and a flake on
    # the PENDING check drops the name-assisted-pass channel entirely, printing
    # an ordinary clean `PASS item 1` and silently withdrawing the eyeball
    # confirmation on a run that authorises a production flip. A here-string has
    # no writer process to signal, so there is no race to lose: 0 failures in
    # 3000 iterations of both spellings. Do not "tidy" these back into pipes.
    if [ "$probe_rc" -eq 0 ] && grep -qF "$PROBE_PASS_MARKER" <<<"$probe_out"; then
      if grep -qF "$PROBE_PENDING_MARKER" <<<"$probe_out"; then
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
      show_probe_stderr=1
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
      note "              Interpreter: ${PROBE_PY_CMD[*]}. Failing closed."
      record_fail 1
      show_probe_stderr=1
    fi
    # The probe's own report, indented under the verdict. It carries the
    # measured slate and, on an unverifiable outcome, its own UNVERIFIABLE line.
    note "$(printf '%s\n' "$probe_out" | sed 's/^/              /')"
    if [ "$show_probe_stderr" -ne 0 ] && [ -s "$probe_err" ]; then
      note "              --- probe stderr, last $PROBE_STDERR_LINES lines ---"
      note "$(tail -n "$PROBE_STDERR_LINES" "$probe_err" 2>/dev/null \
        | sed 's/^/              /')"
    fi
  }
else
  note "FAIL  item 1  UNVERIFIABLE: $EXTRACT_ERROR. Failing closed."
  record_fail 1
fi

# --- item 2: the committed accuracy artifact must be reproducible -------------
if read_ref_file "$EVAL"; then
  eval_src="$REF_CONTENT"
  # Here-string, not a pipe -- same pipefail/SIGPIPE race as item 1 above, and
  # here it fails the OTHER way: a spurious non-zero takes the else branch and
  # prints `PASS item 2` for an eval script that still iterates the frozenset.
  # Measured: a 960 KB $eval_src with the pattern first gives PIPESTATUS=(141 0)
  # piped -- grep matched, printf SIGPIPE-killed -- 200/200 spurious PASS piped
  # vs 0/200 here-string. Pinned by
  # scripts/tests/test_check_write_triage_flip_preconditions.py::TestItemsTwoAndFourReadingIsNotRaceProne
  # so the two cannot drift apart silently. Do not "tidy" this back into a pipe.
  if grep -q 'dict\.fromkeys(TRIAGE_OUTCOMES\|list(TRIAGE_OUTCOMES)' <<<"$eval_src"; then
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
  # Here-string, not a pipe -- see item 2's comment above for the measurement.
  # A flake here spuriously PASSES too. Pinned by the same
  # scripts/tests/test_check_write_triage_flip_preconditions.py::TestItemsTwoAndFourReadingIsNotRaceProne.
  if grep -q "report_path\.with_suffix('\.md')" <<<"$eval_src"; then
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

# --- item 5: the bound candidate must be CONSUMED by the attach ---------------
#
# Placed after items 2 and 4 so `FAILING ITEMS` reads in ascending order; it is
# item 1's other half and nothing here depends on the order.
#
# Same verdict ladder and the same fail-closed discipline as item 1: every
# unverifiable outcome calls record_fail 5, and each call runs in THIS shell,
# never inside a `$(...)` -- see the note above read_ref_file for why that
# distinction once made the gate PASS on unverifiable input.
if [ ! -f "$PROBE5" ]; then
  note "FAIL  item 5  UNVERIFIABLE: probe missing at $PROBE5. Failing closed."
  record_fail 5
elif [ "$EXTRACTED" -eq 1 ]; then
  {
    # Stderr captured separately, for item 1's reason: the probe EXECUTES the
    # ref's own triage module, so that tree's import-time chatter lands here and
    # would otherwise spend the 2000-char operator window on noise this gate
    # does not control. The probe carries its own WARNs on stdout, and last.
    probe5_err="$PROBE_TMP/probe5.stderr"
    probe5_out="$(${PROBE_TIMEOUT_CMD[@]+"${PROBE_TIMEOUT_CMD[@]}"} \
      "${PROBE5_PY_CMD[@]}" "$PROBE5" \
      --src-root "$PROBE_TMP/fused-memory/src" \
      ${PROBE5_EXTRA_ARGS[@]+"${PROBE5_EXTRA_ARGS[@]}"} 2>"$probe5_err")"
    probe5_rc=$?
    show_probe5_stderr=0
    # BELT AND BRACES, and a HERE-STRING rather than a pipe: both for item 1's
    # measured reasons. rc 0 alone is not a PASS because the probe executes the
    # ref's own code and a SystemExit out of it exits 0 having printed nothing;
    # and `printf | grep -q` races on SIGPIPE under `set -o pipefail`, which
    # reported ~0.5% spurious failures per call. Do not "tidy" this into a pipe.
    if [ "$probe5_rc" -eq 0 ] && grep -qF "$PROBE5_PASS_MARKER" <<<"$probe5_out"; then
      note "PASS  item 5  the judge-bound candidate is consumed by the attach"
      # HERE-STRING and `grep -m1`, never a pipe and never `| head -1`: the
      # measured SIGPIPE race under `set -o pipefail` documented on item 1,
      # and a second process in the pipeline is a second thing to lose it to.
      # An absent or reformatted line leaves this empty and the verdict above
      # stands exactly as it reads -- a drift in the probe's report format may
      # not turn a PASS into a failure.
      probe5_branch="$(grep -m1 -F "$PROBE5_BRANCH_MARKER" <<<"$probe5_out")"
      probe5_branch="${probe5_branch#*"$PROBE5_BRANCH_MARKER"}"
      if [ -n "$probe5_branch" ]; then
        note "              via ${probe5_branch:0:$PROBE5_BRANCH_CHARS}"
      fi
    elif [ "$probe5_rc" -eq 0 ]; then
      note "FAIL  item 5  UNVERIFIABLE: the probe exited 0 without reporting a PASS."
      note "              Its report claims no verdict, so nothing was asserted about"
      note "              the invariant. Failing closed."
      record_fail 5
      show_probe5_stderr=1
    elif [ "$probe5_rc" -eq 1 ]; then
      # Terse, for item 1's reason: the harm, both accepted remedies and the
      # measured slate are all in the probe's own report printed directly below.
      note "FAIL  item 5  the judge-bound candidate is NOT consumed by the attach"
      note "              Subject: $TRIAGE at ref '$REF'."
      note "              What was measured, and every branch the probe evaluated, are"
      note "              in its own report below."
      record_fail 5
    else
      note "FAIL  item 5  UNVERIFIABLE: the probe could not be run (exit $probe5_rc)."
      note "              Interpreter: ${PROBE5_PY_CMD[*]}. Failing closed."
      record_fail 5
      show_probe5_stderr=1
    fi
    note "$(printf '%s\n' "$probe5_out" | sed 's/^/              /')"
    if [ "$show_probe5_stderr" -ne 0 ] && [ -s "$probe5_err" ]; then
      note "              --- probe stderr, last $PROBE_STDERR_LINES lines ---"
      note "$(tail -n "$PROBE_STDERR_LINES" "$probe5_err" 2>/dev/null \
        | sed 's/^/              /')"
    fi
  }
else
  note "FAIL  item 5  UNVERIFIABLE: $EXTRACT_ERROR. Failing closed."
  record_fail 5
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
  # BOTH ownership clauses below are gated on their OWN item(s)' pass/fail
  # state via item_failed(), not just the items-2/4 one. This whole arm runs
  # on ANY failure, so an ungated clause names an item the report declared
  # PASS a few lines above, contradicting the authoritative `FAILING ITEMS:`
  # line -- measured both directions: items 2/4 blamed while passing on
  # judge='flat'/eval_src='fixed', and item 1 blamed while passing on
  # judge='by_id'/eval_src='failing'. Pinned by
  # scripts/tests/test_check_write_triage_flip_preconditions.py::TestResultBlockBlamesOnlyFailingItems.
  note "RESULT: preconditions NOT satisfied."
  clause_printed=''
  triage_subject=''
  if item_failed 2 && item_failed 4; then triage_subject='Items 2 and 4 are'
  elif item_failed 2; then triage_subject='Item 2 is'
  elif item_failed 4; then triage_subject='Item 4 is'
  fi
  if [ -n "$triage_subject" ]; then
    note "        $triage_subject task 4762's (priority high); see its description"
    note "        and details for the verbatim findings."
    clause_printed=1
  fi
  if item_failed 1; then
    note "        Item 1 is closed by EITHER attach-target remedy -- option (a) is task"
    note "        4798 item 7, option (b) is task 4762 -- so whichever lands first"
    note "        satisfies it. See its report above for what was measured."
    clause_printed=1
  fi
  # Gated on its OWN item, like the two clauses above and for the same measured
  # reason: this arm runs on ANY failure, so an ungated clause names an item the
  # report declared PASS a few lines up, contradicting the authoritative
  # FAILING ITEMS line.
  if item_failed 5; then
    note "        Item 5 is item 1's CONSUMPTION half: a verdict can be bound to a"
    note "        determinate candidate and still not be the id the write attaches to,"
    note "        so closing item 1 does not close this. Either branch listed in"
    note "        this script's item 5 block satisfies it."
    clause_printed=1
  fi
  # Fallback so this block can never go guidance-free: unreachable today
  # (every record_fail call site passes 1, 2, 4, or '2 4', so one of the two
  # clauses above always fires when fail -ne 0), but a future item added
  # without a matching clause here would otherwise degrade silently to a
  # bare RESULT line with no ownership guidance.
  if [ -z "$clause_printed" ]; then
    note "        See the FAILING ITEMS line below and each item's report above."
  fi
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
