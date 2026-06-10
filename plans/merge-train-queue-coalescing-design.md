# Follow-up spec: retroactive merge-queue train coalescing (A′ β-former gap)

**Status:** design input for `/prd`. Extends the A′ coupling-tolerant train former
(`merge_train_former_enabled`, df 1704–1708). Motivated by a live 2026-06-10 observation.

## Problem (with live evidence)

The A′ β-former batches **only at a task's merge-decision point**
(`workflow.py:_maybe_form_train` @ `:895`, called from `:1055`), and its candidate pool is
"other tasks currently `in-progress` with a resolvable branch" (`_train_candidates` @ `:856`).
So a train forms only when ≥2 stackable tasks reach the merge point **at the same time**.

In reify's actual regime the single verifier throttles completions to ~1 every ~25 min, so
tasks reach the merge point **minutes apart**, submit **solo**, and **pile up in the merge
queue** — which the former cannot touch. Empirically (2026-06-10, ~55 min after enabling A′):
**0 trains formed**, while the queue sat at 4–6 singles behind one ~30-min verify.

**Concrete proof the gap is the *only* obstacle:** at observation time the queue held
`4450` (verifying) + **`4442`, `4455`, `cargo-run-prebuilt-fix` (queued)**. All three queued
branches are **pairwise file-disjoint** (4442 → compiler/eval `solver_elastic`; 4455 →
`docs/scripts/infra`; cargo-run-prebuilt-fix → `reify-eval` tests), i.e. a **clean,
ideal 3-train** — one full-scope verify would land all three instead of three ~90-min
verifies. They are not a train **solely** because each passed the decision point at a
different time before any later one could batch it, and the former has **no mechanism to
coalesce already-queued singles**.

## Goal / user-observable signal (G2)

When ≥2 mutually-stackable **waiting** (not-yet-verifying) single `MergeRequest`s sit in the
merge queue, the worker **coalesces** them into one `GroupMergeRequest` (train): the would-be
N verifies collapse to 1, queue depth drops, and a `train_formed` (or new `train_coalesced`)
event names the absorbed `request_id`s. Given the live case above, `{4442,
cargo-run-prebuilt-fix}` (± 4455, see gating) become one train.

## Design — a coalescing pass over the waiting queue

Add a coalescing step in `SpeculativeMergeWorker` that runs when ≥2 waiting requests exist
(reuse the existing snapshot of `self._queue._queue`; `snapshot()` already reads it):

1. **Candidate set** = requests in the queue that are **waiting only** — never the one in
   `_verify_item`/in-flight, never a `GroupMergeRequest`, never one whose waiter already
   detached/cancelled.
2. **Select** a mutually line-stackable subset (reuse `_select_train_members` @ `:386` +
   line-range fetch) capped at `merge_train_max_members` (3), ordered by any dependency.
3. **Stack** the branches (`stack_train_branches`); conflicts → eject to solo (unchanged).
4. **Remove** the survivors from the queue (drain + re-enqueue the non-selected, or move to a
   side structure), assign `metadata.train`, and enqueue one `GroupMergeRequest` with the
   members + `status_check`/`mark_member_done` callbacks (same as the β path).
5. **Resolve the absorbed solo requests' Futures** so their submitting workflows learn the
   merge was absorbed (see seam below). On train landing, `mark_member_done` flips all members.

The existing β decision-point former **stays** — it catches genuine bursts; this adds the
retroactive path for the steady-trickle backlog that is reify's actual regime.

## Load-bearing seam (G4) — the workflow `superseded` consumer

Each absorbed solo request has a **live waiter** (the submitting workflow blocked in
`_submit_to_merge_queue`). The clean resolution is `MergeOutcome('superseded',
superseded_by=<train request_id>)` → the workflow's merge consumer treats it as "absorbed
into train T; my task will be flipped done when T lands." **This consumer does not exist
yet** — it is the **same missing piece** that keeps `AUTO_CHAIN_GENERATIONS_ENABLED=False`
(`merge_queue.py:140`: "the workflow.py 'superseded' consumer handler"). So building it
**unblocks both** retroactive coalescing *and* γ2 generation auto-chaining — call this out in
the PRD as a shared substrate, not duplicate work. (Same-branch coalescing's
`InFlightMergeRegistry.attach`/`detach` fan-out does **not** apply — train members are flipped
by callback, not by future fan-out, so `superseded` is the right model.)

## Merge-ready confidence gating (the 4455 lesson)

The live case shows the all-or-nothing trap concretely: **`4455` carries a known
verify-blocking history** (the OCCT `verify.sh` scope bug that blocked it before). Batching a
known-risky member means its failure **derails the whole train** (4442 + cargo-run-prebuilt
get rejected with it). So the coalescer must gate on **merge-ready confidence**: exclude a
candidate with a recent verify failure / `dry_run_proposals` block / flakiness flag, and let
it merge solo. For the live case: train `{4442, cargo-run-prebuilt-fix}`, leave `4455` solo.
This is new vs the β former (which gates only on stackability).

## Hard constraints (correctness — unchanged from A′)

- The train runs **one full-scope `verify.sh --scope all`** on the merged tip before advance →
  no unverified code on `main`. Correctness is identical to single merges.
- Never coalesce the in-flight/verifying request, nor a request with a detached/cancelled
  waiter. Idempotent; no double-forming (skip requests already carrying `metadata.train`).
- A coalesced train that fails verify → members re-park / re-dispatch solo (the existing train
  derail + solo-merge fallback). Cost ≈ 1 wasted verify; GO-N3 (`s(3)=0.962`) says +EV.

## Economics / gate

No new economic gate — the s(N) GO-N3 decision (esc-4455-16) already cleared the thrash
profile for N≤3, and the correctness gate is unchanged. This is purely a **trigger/coverage**
extension of an already-greenlit lever.

## Open decisions for /prd

- **Where the pass runs:** inside the merge worker (has the live queue; natural at the
  pre-dequeue point) vs a scheduler sweep. Worker-side is simpler (no cross-process queue
  view) but must mutate the `asyncio.Queue` carefully.
- **Trigger cadence / debounce:** run on each enqueue when depth ≥2? periodic? Avoid
  thrashing the stack repeatedly as new requests arrive.
- **Confidence signal source:** what marks a candidate "risky" (last merge_attempt outcome,
  `metadata.dry_run_proposals`, a flakiness counter).
- **Interaction with speculative pipeline:** a coalesced train bypasses the speculative
  look-ahead (`_do_train_merge` already does); confirm the depth-1/K cap bookkeeping is
  consistent when a coalesce removes queued items.

## Signal / boundary tests

- 3 waiting **stackable** singles + the verifier busy → one train of 3 forms, emits the event,
  the 3 solo waiters resolve `superseded`, and on landing all 3 flip done.
- 3 waiting where 2 overlap on a line range → a train of the 2 stackable + the 3rd stays solo.
- A waiting single with a recent verify failure → **excluded** from the train, merges solo.
- The in-flight/verifying request is **never** absorbed; a request whose waiter detached
  mid-pass is skipped without resolving its (already-cancelled) future.
