# Merge-chain visibility: a live "Verifying now" view, chain rounds recorded by the merge lane, and a "Chain rounds" history panel

**Status:** draft, revision 2 (folds in PRD critic pass 1; table at the end). Authored 2026-11-18 from the ruled
design note `plans/merge-chain-dashboard-design-2026-11-17/DESIGN.md` (revision 3, untracked) and Leo's rulings of
2026-11-18. Not yet decomposed.
**Code anchors** verified against main `8874271539` (2026-11-18; `d74194d263` plus task 10723), and task 8528's
in-progress branch `.worktrees/8528` at `1a28529270`. Main moves fast: cite by symbol; re-locate at implementation.
**Evidence:** the design note's `seats/` (producer, measure, dashboard, tasks, questions, lead notes, three critic
passes, `prd-critic-pass1.md`), re-checked here. Every count names its window and the commit or time it was read at.

Words used throughout:

- **Round.** One deep-chain verify: the deep policy chose depth k ≥ 2 for a dispatch, reserved k − 1 waiting
  requests (*links*) behind the dispatching *owner*, built the cumulative *tip* and verified it. Its *members* are
  the owner and the links. After 8528 a round's basis is fresh (`pass_estimate`, `no_estimate`) or `bisect`.
- **Search.** After 8528, a red fresh round makes its members *suspects*; the lane verifies halves until none
  remain. A search ends when the suspect list empties or is replaced.
- **Hold.** A `ChainReservation` (`merge_lane/chain_reservation.py`). Expiry ends a hold while its round continues.

## 1. Goal

An operator on the dashboard's Merge Queue section sees, per project:

1. **now:** every merge verify in flight, and for a chain its members, host, phase, age, and any member whose
   request is already decided;
2. **recently:** how each round ended (landed whole, a prefix, nothing, voided, cancelled, aborted) and why, and,
   after 8528, which suspect a search found red.

The merge lane records rounds as a unit (Leo, Q2), so (2) is read from records, not inferred.

## 2. Consumer and user-observable surface (G1)

| Mechanism | Consumer |
|---|---|
| α's `verifying_now` Datum, projected from the one `get_merge_queue` reply | "Verifying now" (new `static/redux/merge_chains.jsx`, mounted from `tabs.jsx::MergeTab`); "Currently queued"'s "in flight ↑" |
| β's round, search and rejected-record events | ε's `chain_rounds` read; β's audit script (§10) |
| β's `chain_id` / `origin_chain_id` on `merge_verify`, `chain_id` on `verdict_voided` | 8547's verify list (A-8547) |
| γ's `restart_drain.verifies_in_flight[]` additions, `entries[].chain`, `entries[].unchained_mark`, `chain_policy` | "Verifying now" (round view), "Currently queued" columns, the project head's controller line |
| δ's bisect split in `summarize_chain_telemetry`'s module | 10621's Deep chains card, the digest, `scripts/merge_lane_throughput.py` |
| ε's `runs_read/merge_lane.py::chain_rounds`, `CHAIN_OUTCOME_FAMILY` | "Chain rounds" panel |

Recent deep verifies with members are 8547's list (A-8547); this PRD adds no second list.

## 3. Background

**The design note** settles the facts table (F1–F11), the rendering and the round model (its §3–§5). This PRD turns
it into leaves and changes it where code moved (decisions marked *Departs*).

**Measured evidence.**

- df `runs.db`, 2026-11-10T16:37Z → 11-17T16:37Z (read at `f7ebb6b0c3`): 138 deep verifies, 86 failed (62.3%);
  singles 237/304 passed; 153 landings carry `landed_via_chain`; reify 22 deep, 0 failed.
- Outcome cannot be joined from events: of 54 deep verifies with member ids, 30 were unattributable; 10 of 234
  `verdict_voided` rows join their verify by sha; a cancelled chain verify records no members (160 cancels,
  ~41.6 h).
- Same window plus to 11-18 (critic, ~16:10Z): 38 pairs of deep verifies overlapped in time; in 13 both were red.
- 2026-11-17 07:00–16:00Z: 11 deep verifies, all red; 8826, cancelled 13:42Z, stayed chained.
- Since 10617 deployed (2026-11-17T20:49Z → 11-18T15:49Z, one run, recount at `d74194d263`): 20 deep verifies,
  5 red; 42 landings via chain.
- Tasks 10721 (four of five rebased heads lost their landing in 6-chains), 10723 (5715 chained twice after voided
  red verdicts; fixed `328a238fbb`), 6267 (six cancelled requests resurrected by the 20:49Z restart).

The dashboard saw none of it: the probe keeps six keys per entry plus `metrics`
(`dashboard/src/dashboard/data/merge_queue.py::_normalize_entry`, `::_probe_live_one`) and reads no `merge_verify`.

**What changed since the note** (main `8874271539`, 8528 branch `1a28529270`):

1. 10617 landed: `chain_snapshot` skips decided requests and tasks in `_next_dispatch_unchained`. B1 is done.
   10723 landed: the mark now lasts until the task's verdict is applied.
2. A decided link is still reachable: `cancel_request` refuses only finalize-window states and the link being
   landed, so a reserved link can be cancelled mid-verify; `_land_chain_prefix` then stops on it.
3. `_next_dispatch_unchained` is not on the snapshot.
4. On 8528's branch a red tip sets suspects (`chain_policy.py::on_tip_result`); a red *fresh* round replaces the
   suspects wholesale; a lone red bisect verify empties them; `_set_chain_policy` is the one writer;
   `seed_chain_policy` refills only the pass window, so a restart ends a search. The branch adds two context
   managers that end a held reservation.
5. 8528's branch edits `test_merge_queue_deep_reservation.py`, which asserts a `chain_reservations[]` row by exact
   dict equality, and rewrites the deep test scenes (`depth_table` replaces `chain_cap` as the depth source).
6. `_land_rebased_or_park_gate` can finish the owner through `_finalize_landed_by_successor`, which runs no walk.

## 4. Sketch of approach

- **α (before 8528), dashboard only.** Project `restart_drain.verifies_in_flight[]` (task 5371) and
  `chain_reservations[]` (task 7720) from the existing reply into one server-side Datum; render "Verifying now".
- **β (after 8528, 10624, 10721).** A fail-safe `ChainRounds` ledger mints, records and settles rounds and searches;
  wire records in a new `shared/src/shared/merge_chain.py`; an audit script.
- **γ (after β).** The snapshot exposes the round view and policy state; the dashboard switches to it.
- **δ (after 8528, 10621).** Telemetry counts bisect rounds apart. **ε (after θ, 8547, γ).** "Chain rounds".
- **Companion corrections** to 8547, 8329, 5245, θ 9852, γ2 9802 and 8532, each order-independent (§11).

## 5. Resolved design decisions

**Leo's rulings, 2026-11-18**

- **D1 (Q1 a).** Retire the speculative hit-rate figures inside 10621. Applied (10621 `updatedAt` 15:40Z).
- **D2 (Q2 a).** The lane records rounds as a unit: an id, a started record, at most one settled record.
- **D3 (Q3 a).** File the pre-8528 live view (α).
- **D4 (Q4).** Moot: 10617 fixed it; follow-ups sequenced in §7.

**Round model**

- **D5.** A round is k ≥ 2. 8528's lone-suspect and `awaiting_bisect` verifies are not rounds; the lone-suspect
  verify's `merge_verify` carries `origin_chain_id`.
- **D6.** The ledger is `merge_lane/chain_rounds.py::ChainRounds` (heuristic 6, one purpose; heuristic 9, deep
  module). `chain_reservation.py` keeps the hold; the ledger, not `ChainPolicyState`, maps suspects to searches.
- **D7.** A search's id is the `chain_id` of the red fresh round that opened it; its bisect rounds and lone-suspect
  verifies carry it as `origin_chain_id`.
- **D8 (*Departs*).** Settles are keyed by `chain_id`, minted before the build and carried on `ChainResult` and the
  in-flight entry, never by owner request id (one request can own a fresh then a bisect round). Every exit has a
  named end (§8.1 table). `_end_chain_reservation` takes `chain_id` and an end, and settles as the fallback, so
  callers added later (8528's context managers) are covered; specific sites settle first and the latch keeps the
  first. A fallback settle is recorded as `aborted`/`unattributed`, which the audit counts and the H-rows keep at 0.
- **D9 (*Departs*).** Outcomes add `head_not_landed` (tip green, owner not on main) and a typed `fault` for it and for
  `aborted`. The landed-by-successor exit is `landed_prefix`, `landed 1`, `stop head_landed_by_successor`.
- **D10.** `ChainStop` has eight arms: the six exits of `_land_chain_prefix`'s five `break`s (the halt and CAS
  cases share one, split by `_HALT_ADVANCE_RESULTS`), plus `head_landed_by_successor` and `walk_interrupted` (the
  owner advanced, then the walk was cut off, e.g. cancelled after `stop()`). They are log lines today
  (INV-11). `lease_contended` is defensive: the walk's `advance_main` path takes no verify lease today.
- **D11 (*Departs*).** `started` carries `tip_budget_deadline_ts` (build deadline + merge-verify command budget) and
  `lifetime_deadline_ts` = tip budget + `MAX_CAS_RETRIES` × (1 + `MAX_POST_MERGE_VERIFY_TIMEOUTS` +
  `MAX_POST_MERGE_VERIFY_ENOSPC_RETRIES`) × budget. The lifetime is advisory: a full gate re-verify leaves the round
  (task 8525, `_run_gate_history_leg`), so only a parked history leg keeps one open. Panels claim no liveness.
- **D12.** β emits `merge_chain_search_settled` per search; ε reads only round and search records, never
  `merge_verify`. No "back to lane" figure.
- **D13 (*Departs*).** A red fresh round while a search is open supersedes it: `track_suspects` settles the open
  search `superseded` (naming the new round) before opening the next. At most one open search.
- **D14.** `basis` is typed `ChainDepthBasis`; `awaiting_bisect` is rejected at construction.
- **D15 (*Departs*).** No R2: `merge_verify`'s members are one verify's built set; `started`'s are a round's planned
  set. Two facts, one home each (INV-9). `merge_verify_cancelled` gains no chain fields.
- **D16 (new, blocker 1).** The ledger is optional observability and fails safe: no `ChainRounds` method raises into
  the lane (the house rule of `_build_deep_chain`'s `except Exception` arm). A record that fails construction or
  emission becomes a `merge_chain_record_rejected` event naming record type, `chain_id`, field and value (INV-2),
  and a WARNING log. That event is the single home of the fact: ε and the audit count it; no in-memory counter
  (`EventStore.emit` never raises, so a store failure is not a ledger branch). Lease, permit and chain-lane
  releases never depend on the ledger.

**Live view**

- **D17 (*Departs*).** α touches no orchestrator file and no file in 8528's plan, and adds no orchestrator test (8528
  rewrites the deep scenes). Request ids, per-member `decided` and the round id come in γ from the ledger.
- **D18.** α's decided-link flag is keyed by task id within one reply: a reserved link task with no `entries[]` row
  reads "⚠ no live request (cancelled while chained)", or "landing" when its owner's kind is `finalizing`. Its
  limit (a duplicate live request for the task hides it) is in the title. γ's `decided` replaces it. It does not
  show 6267's case: a resurrected request is live and listed (*Departs* from the note's §9).
- **D19 (*Departs*).** γ serves the mark as `entries[].unchained_mark`, rendered "marked: verifies alone". The
  search and depth state go on the controller line ("search ch-… open: only the first suspect chains"; "depth-1
  row: no chaining"), so no per-entry label claims a dispatch decision the policy makes later.
- **D20.** The projector is a pure function in `data/merge_chains.py`, called on the one probe reply (one-datum
  PRD, sketch item 1 and decision 11): a server-side Datum with `LIVE_QUEUE_FRESHNESS_BOUND_SECONDS`; a failed
  probe is `unknown` with its reason (no persisted chain history exists). γ2's `project` callable calls it (A-9802).
- **D21.** "Is X verifying" has one owner, `restart_drain.verifies_in_flight[]`. "Currently queued" shows
  "in flight ↑" from it, by task id under α and request id under γ.
- **D22.** `chain_reservations[]` stays on the wire for the lane's tests only; no dashboard reader uses it after γ.
- **D23.** The live wire is pinned by one committed fixture, `shared/tests/fixtures/merge_queue_snapshot_chain.json`.
  α captures it once from df's real `get_merge_queue` reply (the 2026-11-17 capture
  `plans/merge-chain-dashboard-design-2026-11-17/seats/producer-live-snapshot-df.json` may serve; it holds two
  verifies in flight, so a fresh capture taken while a chain is reserved is preferred) and the dashboard tests
  check its type skeleton against the keys the projector reads; γ adds the orchestrator test that regenerates it
  from a live lane scene and fails on skeleton drift.
- **D24 (*Departs*, critic 7).** α is the live panel only. "Recent deep verifies" is dropped: 8547's list carries
  members (A-8547). This removes α's runs.db decode, its `chain_telemetry` edit and θ's extra port.
- **D25.** P2 and P3 of the note are one leaf (γ).

**History view**

- **D26.** `chain_rounds` is a `RunsQuery` in θ's `runs_read/merge_lane.py`; no raw SQL outside the read model
  (S22 ω). The only joins are round ↔ settle and round ↔ search, on minted ids. A store with no record of either
  type reads "no chain-round record in this runs.db (β not deployed)", not "no rounds" (INV-13).
- **D27.** Badges come from `CHAIN_OUTCOME_FAMILY: Mapping[ChainOutcome, OutcomeFamily]` in
  `data/outcome_colors.py`, beside θ's `OUTCOME_FAMILY`, with a totality test.
- **D28 (*Departs*, critic 6).** δ owns the bisect split; 8532 gets the identical rule as text and no edge, so its
  7-day clock is not moved. δ's agreement test pins `scripts/merge_lane_throughput.py` to the shared function.
- **D29.** `worker.py` (21,805 lines at `8874271539`) gains calls, not logic: about 25 sites in β, 4 in γ. The
  merge-lane ratchet bounds cognitive complexity per function; β and γ plan an authorised-raise entry under
  `orchestrator/tests/merge_lane_ratchet_raises/` if a function moves.

## 6. Pre-conditions (G3)

| Assumed capability | Evidence or prerequisite |
|---|---|
| `restart_drain.verifies_in_flight[]` | `worker.py::SpeculativeMergeWorker._verifies_in_flight`, `merge_lane/types.py::VerifyInFlight.to_wire` |
| `chain_reservations[]`, `entries[].verify_base` | `ChainReservations.snapshot`; `snapshot`'s entry builder (task 5447) |
| `summarize_chain_telemetry` in shared, the Deep chains card | task 10621 (pending) → δ |
| `ChainDepthBasis`, `choose_depth`, `_set_chain_policy`, `_note_tip_result`, suspects | 8528 (in progress) → β, γ, δ |
| Walk faults through public seams | `orchestrator/tests/_merge_lane_fakes.py::make_lane` takes any `git_ops`; `test_merge_speculation.py::_AdvanceFailingGitOps` overrides `advance_main` |
| Abort causes | `_run_inflight_verify`'s `VerifyCancelCause.ABANDONED`, `OPERATOR_HALT`, `NO_PROGRESS`; `_tear_down_inflight_entry` (`CASCADE`, `GATE_REANCHOR`) |
| Event reads for the audit | `event_store.py::EventStore.fetch_events_by_type_all_runs` |
| `shared.runs_read`, `runs_read/merge_lane.py`, `OUTCOME_FAMILY` | S22 β 9800, θ 9852 → ε |
| jsdom harness, wire fixtures | `dashboard/tests/js/dom_harness.mjs`, `wire_fixtures.mjs` |

## 7. Cross-PRD relationship and seam owners (G4)

Task store (read-only, 2026-11-18): 9766 → 9140 → 8528; 9800 → 9766; 9802 → 9162 → 9157 → 9140; 9852 → 9800,
9802, 9714, 8563; 8547 → 9852; 8329, 5245, 9133, 4767, 6267, 8532 → 8528. 10619, 10620, 10621, 8222, 10624, 10721
do not reach 8528. 10723 is done; 8528 is in progress; 10624 and 10721 are pending.

| Other | Direction | Seam | Owner | Status |
|---|---|---|---|---|
| S22 (`plans/review-all-s22-dashboard-readers-and-aggregation-prd.md`) | consumes | `runs_read`, θ's module and `OUTCOME_FAMILY`, γ2's probe helper, ω's guard | S22; ε adds its constructor and row; A-9852, A-9802 (either order) | queued |
| 8528 | consumes | `ChainDepthBasis`, `choose_depth`, `_set_chain_policy`, `_note_tip_result`, the reservation context managers | 8528; β instruments its merged code | in progress |
| 10624, 10721 | same sites | red-tip exit, `finally` of `_run_inflight_verify`, `_deep_chain_placement` | those tasks; β waits | pending |
| 6267 | none here | merge journal | 6267 | pending |
| 8547 | produces for | `chain_id`, `origin_chain_id`, members on deep rows | 8547 (A-8547) | behind θ |
| 8329 | shared emit site | `verdict_voided` gains `verify_elapsed_ms` (8329) and `chain_id` (β), additive | each its key | pending |
| 5245 | same derivation | "what is verifying" from `_verifies_in_flight`; γ owns the `VerifyInFlight` additions | 5245 (A-5245), either order | pending |
| 10621 | consumes | the shared telemetry module, the card | 10621 | pending |
| 8532 | same rule | bisect split stated as text | δ owns the code; 8532 the readout (A-8532) | pending |
| 9133, 4767 | none | probe, `merge_verify.depth`; nothing here reads them | those tasks | pending |
| 8932 | future | a hedge may give a round several verifies; panels group by `chain_id` | 8932 | pending |
| 8359, 10588 | same file | `types.py` split; γ edits `VerifyInFlight` | file locks; second lander rebases | pending |

No reciprocal ownership remains. No out-of-batch task gains an edge on this batch: same-file serialisation with
S22, 8547 and 5245 relies on the orchestrator's `metadata.files` locks, as it does for 8359 and 10588.

## 8. Contract (B + H)

B + H because four packages are touched, the producer seam is the merge lane, and two consumers (8547, ε) read β.

### 8.1 Round records and the ledger

`shared/src/shared/merge_chain.py`: frozen dataclasses with `to_event_data()` / `from_event_data()`, shared by
producer and every decode (heuristic 11; INV-5); stdlib and local types only (heuristic 13).

```
ChainMember(request_id: str, task_id: str)
ChainRoundStarted(chain_id, origin_chain_id: str | None, basis: ChainDepthBasis,
                  owner: ChainMember, members: tuple[ChainMember, ...],     # planned, land order, owner first
                  tip_budget_deadline_ts: float, lifetime_deadline_ts: float)
ChainRoundSettled(chain_id, outcome: ChainOutcome, built: int | None, truncated_reason: str | None,
                  landed: int, stop: ChainStop | None, cancel_cause: str | None,  # a VerifyCancelCause value
                  fault: ChainFault | None, finals: tuple[ChainFinal, ...])
ChainFinal(request_id: str, final: MemberFinal)
ChainSearchSettled(origin_chain_id, rounds: tuple[str, ...], solo_verdicts: tuple[SoloVerdict, ...],
                   ended_by: SearchEnd, superseded_by: str | None, wall_ms: int)
SoloVerdict(task_id: str, request_id: str, passed: bool)
ChainRecordRejected(record_type: str, chain_id: str | None, field: str, value: str, error: str)
```

Enums: `ChainOutcome` ∈ `build_declined`, `landed_whole`, `landed_prefix`, `tip_red`, `head_not_landed`,
`voided_at_adoption`, `cancelled`, `aborted`, `shutdown`. `ChainStop` ∈ `link_taken_elsewhere`, `link_abandoned`,
`cas_not_advanced`, `halt_fault`, `lease_contended`, `link_raised`, `head_landed_by_successor`, `walk_interrupted`.
`ChainFault` ∈ `verify_raised`, `verify_skipped`, `runner_unavailable`, `verify_lease_contended`, `rebased_at_gate`,
`gate_red`, `cas_exhausted`, `finalize_raised`, `unattributed`. `MemberFinal` ∈ `landed`, `restored`, `cancelled`, `not_built`. `SearchEnd` ∈
`suspects_cleared`, `suspects_left_lane`, `chains_switched_off`, `superseded`, `shutdown`.

Events (in `EventType` wherever it lives at dispatch; S22 α moves members verbatim): `merge_chain_round_started`,
`merge_chain_round_settled`, `merge_chain_search_settled`, `merge_chain_record_rejected`. `merge_verify` gains
`chain_id` and `origin_chain_id` through `merge_verify_facts.py::MergeVerifyDispatchFacts`; `verdict_voided` gains
`chain_id` and `killer.chain_id` (site `chain_released`).

**Invariants** (checked at construction, raising `ValueError` naming field and value, which the ledger turns into
a rejected record; checked again by the audit):
- I1. At most one settle per `chain_id` and one search settle per `origin_chain_id` (the ledger's latch).
- I2. `members[0] == owner`; distinct request ids; `len(members) >= 2`; `basis != awaiting_bisect`.
- I3. `built` and `landed` count members including the owner (`built` = `merge_verify.chain_items`).
  `finals` names every planned member once; `landed == count(final == landed)`.
- I4. `build_declined` ⇒ `built null`, every final `not_built` (the owner verifies alone outside the round).
  `landed_whole` ⇒ `landed == built`. `landed_prefix` ⇒ `1 <= landed < built`, `stop` set. `tip_red`,
  `voided_at_adoption`, `head_not_landed`, `cancelled`, `aborted`, `shutdown` ⇒ `landed 0` (a round whose owner
  advanced never takes these ends).
  `cancel_cause` iff `cancelled`; `fault` iff `aborted` or `head_not_landed`; `stop` iff `landed_prefix`;
  `superseded_by` iff `superseded`.
- I5. `from_event_data` on an unknown enum raises `ChainRecordError(field, value)`; ε counts and shows such rows.

**Ledger** `ChainRounds(sink, clock)`, holding `(request_id, task_id)` pairs, never request objects
(heuristic 5): `mint(...) -> chain_id`; `note_member_landed(chain_id, request_id)`; `settle(chain_id, end) -> bool`;
`round_of(request_id)`; `origin_of(task_id)`; `track_suspects(suspects, cause, red_round)`;
`note_solo_verdict(...)`; `settle_open_for_shutdown(finalizing)`. `note_member_landed` is called for the owner in
`_advance_candidate`'s advanced arm and for each link the walk lands. **None raises** (D16). Bounds
(INV-7): open rounds ≤ verifies in flight + builds; one open search.

**End per exit** (β re-derives every row from 8528's merged code; a row it cannot place is a stop for the leaf:
escalate):

| Exit | Site | End |
|---|---|---|
| Build returns None / raises | `_deep_chain_placement`'s two releases | `build_declined` |
| Tip red (a code verdict, 8528's `_is_code_verdict`) | `_run_inflight_verify` chain arm, before requeue | `tip_red`; `track_suspects` |
| Tip verify skipped (low disk) | same arm, `_is_code_verdict` false | `aborted`, `verify_skipped` |
| Owner abandoned mid-verify | abort trigger 1 | `cancelled`, `abandoned` |
| Operator halt | abort trigger 2 | `cancelled`, `operator_halt` |
| No-progress abort | abort trigger 3 | `cancelled`, `no_progress` |
| Tip verify raised | chain-arm `except Exception` | `aborted`, `verify_raised` |
| Runner unavailable | `except RunnerUnavailable` | `aborted`, `runner_unavailable` |
| Contended verify lease (defer) | the lease-contended defer arm | `aborted`, `verify_lease_contended` |
| Cascade / gate re-anchor | `_tear_down_inflight_entry`, before its teardown await | `cancelled`, its cause |
| Tip green, walk ends | end of `_land_chain_prefix` (every `break`) | `landed_whole` / `landed_prefix` + stop |
| Voided at adoption | `_void_and_remerge` | `voided_at_adoption` |
| Landed by successor | `_land_rebased_or_park_gate` → `_finalize_landed_by_successor` | `landed_prefix`, `head_landed_by_successor` |
| Gate re-verify becomes a dispatch | `_finalize_inflight` (task 8525 path) | `head_not_landed`, `rebased_at_gate` |
| Gate history leg red / CAS exhausted | `_finalize_inflight` arms | `head_not_landed`, `gate_red` / `cas_exhausted` |
| Owner abandoned at finalize | `_finalize_inflight` abandoned check | `cancelled`, `abandoned` |
| Exception in finalize, owner not advanced | `_finalize_inflight`'s `finally` fallback | `aborted`, `finalize_raised` |
| Walk cut off after the owner advanced (raise, cancel) | `_finalize_inflight`'s `finally` fallback | `landed_prefix` (or `landed_whole`), `stop walk_interrupted`, `landed` from `note_member_landed` |
| Held-reservation context managers (8528) | `_end_chain_reservation` | none: hold end; round already settled |
| Expiry | `_expire_chain_reservations` | none: hold end only |
| Shutdown | `stop()` | rounds whose owner has not advanced: `shutdown`. A finalizing round is never settled by `stop()` (its walk outlives the 5 s `_shutdown_timeout`); it settles at the walk's end or in `_finalize_inflight`'s `finally`. A process that dies first leaves it unsettled ("orchestrator restarted since") |

Searches: a lone red bisect verify or a passing round empties the suspects → `suspects_cleared`; pruning →
`suspects_left_lane`; `_end_bisect` → `chains_switched_off`; a red fresh round → `superseded`; `stop()` →
`shutdown`. `track_suspects` takes the cause from `_set_chain_policy`'s caller.

### 8.2 Snapshot round view (γ)

Additive keys on `SpeculativeMergeWorker.snapshot()`:
- `restart_drain.verifies_in_flight[]` rows gain `request_id` and `chain: {chain_id, basis, members:
  [{request_id, task_id, decided}], phase: building | verifying, build_deadline_ts, tip, reservation_seconds_left}
  | null`; `host` is the dispatching owner's leased host once held.
- `entries[]` gain `chain: {chain_id, position} | null` and `unchained_mark: bool`.
- Top level gains `chain_policy: {enabled, depth_table, pass_estimate, row, suspects, search_id}`.
- `chain_reservations[]` is unchanged and tests-only (D22).

### 8.3 Dashboard Datums

- `verifying_now` (α; γ reshapes): per project in the `/merge-queue` payload; verifies (task, request id after γ,
  kind, host, start, deadline, base main sha or null, chain or null) and any reservation with no matching verify.
  Unknowns: the probe error; "orchestrator predates task 5371" when `restart_drain` is absent.
- `chain_rounds` (ε): `chain_rounds(since, until, limit=30)` → rounds newest first (started, settled or null,
  search), the window total, undecodable and rejected counts, and whether any chain record exists.

Rendering: no state branching in JSX; unknown is "—" with reason; an unsettled round reads "no settle record ·
‹elapsed›", "(past tip-verify budget)" after the tip deadline, "past its lifetime" after the lifetime deadline,
and "orchestrator restarted since" when a later `run_id` exists.

## 9. Boundary-test sketch

Producer rows drive a real lane through public seams only: `make_lane(git_ops, …)` with a real `GitOps` over a
temporary repository, a real `EventStore`, `FakeVerifier` scripts, `FakeClock`, `lane.cancel_request`,
`lane.stop()`, halt calls, and `GitOps` subclasses overriding public methods (`advance_main`,
`is_ancestor_checked`) as `_AdvanceFailingGitOps` does. No test patches a `worker` name. Records are read back
through `from_event_data` (β) and `chain_rounds` (ε). Rows use four members (owner + 3 links) unless stated.

**History seam (β writes, ε reads)**

| # | Scenario | Then |
|---|---|---|
| H1 | All pass | `landed_whole`, `built 4`, `landed 4`; `merge_verify.chain_id` matches |
| H2 | First candidate conflicts (zero links) | `build_declined`, every final `not_built` |
| H3 | Tip red | `tip_red`; a search opens; the next bisect round's `origin_chain_id` is H3's id |
| H4 | H3, then a lone suspect red | `search_settled` `suspects_cleared` naming it; its `merge_verify.origin_chain_id` set |
| H5 | `advance_main` not-advanced for link 2 | `landed_prefix`, `landed 2` (owner + link 1), `stop cas_not_advanced`; links 2, 3 `restored` |
| H6 | Link 2's advance returns a halt result | `stop halt_fault`; the lane halts |
| H7 | `advance_main` raises `MergeVerifyLeaseContended` for link 2 | `stop lease_contended` |
| H8 | Link 2's advance raises | `stop link_raised` |
| H9 | `cancel_request` on link 2 mid-verify | `stop link_abandoned`; link 2 `cancelled` |
| H10 | Reservation expires; merger takes link 2 | settles later, `stop link_taken_elsewhere` |
| H11 | Main moves under the owner mid-verify | `voided_at_adoption`; `verdict_voided.chain_id` |
| H12 | Owner's advance fails permanently | `head_not_landed`, `cas_exhausted` |
| H13 | Cascade | `cancelled`, `cascade` |
| H14 | Tip verify raises | `aborted`, `verify_raised` |
| H15 | `stop()` mid-verify, mid-search | `shutdown`; search `shutdown` |
| H15b | `stop()` during a walk whose links outlast `_shutdown_timeout` | the records match what landed: `landed` counts the owner and every landed link; `landed_whole`, or `landed_prefix` with `walk_interrupted` if the walk is cut off; never `shutdown` |
| H16 | Owner parks on a declared history leg past the tip budget | no settle until it resolves; audit passes |
| H17 | Operator halt / no-progress abort / owner abandoned, mid-tip-verify | `cancelled` with each cause |
| H18 | Landed by successor (`is_ancestor_checked` true) | `landed_prefix`, `landed 1`, `stop head_landed_by_successor` |
| H19 | Two overlapping fresh rounds, both red | first search `superseded` naming the second; one open search |
| H20 | An inconsistent end passed to `ChainRounds.settle` (ledger unit test, and a lane scene); a ledger built on a sink that raises (unit test) | `settle` returns without raising; one `merge_chain_record_rejected` row and a WARNING; in the scene the lane resolves the request with host slot free, speculation permit returned and chain lane back in the pool |
| H21 | Any of H1–H19 | no `chain_id` settles twice; no `unattributed` fault; the audit script exits 0 |
| H22 | ε's contract row over H1–H19's store | each round read with outcome, k of n, stop; H4 under H3 |
| H23 | Store with deep `merge_verify` rows and no chain records | ε: "β not deployed"; audit refuses |
| H24 | A row with `outcome: "teleported"` | counted undecodable and shown; the endpoint does not raise |

**Live seam**

| # | Scenario | Then |
|---|---|---|
| L1 | γ regenerates the fixture from a lane mid-round | rewrites on request; fails on skeleton drift |
| L2 | α projects the committed fixture | each verify, its links in order, base sha by task id |
| L3 | α flag: a link's entry removed; owner `finalizing` | "no live request"; "landing" |
| L4 | α unknowns | probe failure → unknown with error; no `restart_drain` → "predates task 5371" |
| L5 | γ round view | round id, basis, `building` with deadline, a `decided` member "cancelled, still chained" |
| L6 | γ queue columns | "round ch-…, 2"; "marked: verifies alone" |
| L7 | γ controller | on/off, p̂, row, open search; `enabled false` → "chaining off" |
| L8 | One label unreachable | `/merge-queue` 200; that label unknown, the other fresh |

UI: jsdom through `dom_harness.mjs` with a `wire_fixtures.mjs` entry per state (busy, idle "nothing verifying",
unknown; ε also "no rounds in this window" and "β not deployed"). No source-text tests.

## 10. Decomposition plan

Five leaves; same-file pairs carry real edges (overlay). Sizes are expected changed lines.

- **α — Merge tab: "Verifying now" from the live reply** (dashboard).
  Files: new `dashboard/src/dashboard/data/merge_chains.py`; `data/merge_queue.py` (call it from `_probe_live_one`);
  `data/redux_api.py`; `static/redux/merge_queue.js`; new `static/redux/merge_chains.jsx`; `static/redux/tabs.jsx`
  (mount; "in flight ↑"); `static/redux/index.html` (script tag, cache buster); new
  `shared/tests/fixtures/merge_queue_snapshot_chain.json`; new `dashboard/tests/test_merge_chains_data.py`;
  `dashboard/tests/js/merge_queue.test.mjs`, `jsx_render.test.mjs`, `wire_fixtures.mjs`. About 12 files.
  Decisions D17, D18, D20–D23. No file in 8528's plan, no orchestrator test. Call the projector from
  `_probe_live_one`, or from `probe_escalation_urls`'s `project` callable if task 9802 has landed.
  Signal: on df, while a verify runs, "Verifying now" lists it with kind, host or "host —, dispatching", age and
  deadline; a chain owner shows its links in order, "building" until its tip exists; L2–L4, L8 and UI states pass.
  Prerequisites: 10621, 8222 (same files). Size ~700–900.
- **β — Merge lane records chain rounds and culprit searches** (orchestrator, shared, scripts).
  Files: new `merge_lane/chain_rounds.py`; new `shared/src/shared/merge_chain.py` (records and
  `audit_round_records`); `merge_lane/worker.py` (§8.1 table); `merge_lane/types.py`
  (`ChainResult.chain_id`); `event_store.py` (or `shared/run_events.py`); `merge_verify_facts.py`;
  `merge_lane/dead_base_ledger.py` (`killer.chain_id`); new `scripts/check_chain_round_records.py`; new
  `orchestrator/tests/test_merge_chain_rounds.py`; new `shared/tests/test_merge_chain.py`; ratchet raise if needed.
  Decisions D5–D16, D29.
  Signal: `scripts/check_chain_round_records.py --runs-db <path> --since <ts>` refuses a path that does not exist
  before it constructs `EventStore` (whose constructor would create an empty store), then reads through `EventStore`
  (no raw SQL) and reports per window: rounds started and settled, rejected records, and violations. Violations are
  a double settle; an `unattributed` fault; a `started` with no settle whose owner request has a later
  `merge_verify` or `merge_finalized` row in the same run (owner progress); and a deep `merge_verify` without a
  `chain_id`. It exits non-zero on any violation; refuses ("deep verifies but no round records: β not deployed or
  not recording") when deep verifies exist and round records do not; and prints "no deep verifies in window:
  chains at depth 1 or off" when there are none. Lifetime is for panel labels only. CI runs it over H1–H19's store
  (H21, H23). On df, run once a week after deploy.
  Prerequisites: 8528, 10624, 10721 (10723 done). Size ~1,300–1,600, the top of the band (24 lane-scene rows, ~25
  `worker.py` sites). Tripwire: if the plan saturates the architect's 180-turn cap, split the ledger and records
  (with H20–H21) from the `worker.py` instrumentation; do not retry.
- **γ — Snapshot round view; round, queue and controller lines** (orchestrator, dashboard).
  Files: `merge_lane/types.py` (`VerifyInFlight`); `merge_lane/worker.py` (`_verifies_in_flight`, `snapshot`);
  `merge_lane/chain_rounds.py`; `merge_lane/chain_policy.py` (`ChainPolicyState.wire`); new
  `orchestrator/tests/test_merge_queue_snapshot_chain_fixture.py`; the fixture JSON; `data/merge_chains.py`;
  `static/redux/merge_queue.js`, `merge_chains.jsx`, `tabs.jsx`, `index.html`; dashboard tests. About 14 files.
  Decisions D19, D21–D23, D25. Replaces α's task-id flag. If 5245 has landed, extend its derivation from
  `_verifies_in_flight`; do not add a second one.
  Signal: on df, "Verifying now" shows round id, basis, members and phase, "building, build deadline …" with the
  owner's host, "cancelled, still chained" on a decided member; "Currently queued" shows round and position and
  "marked: verifies alone"; the project head shows the controller line; L1, L5–L8 pass.
  Prerequisites: β, α; 8528. Size ~1,000–1,300.
- **δ — Chain telemetry counts bisect rounds apart** (shared, orchestrator, scripts, dashboard).
  Files: the module holding `summarize_chain_telemetry` after 10621; `orchestrator/src/orchestrator/digest.py`;
  `scripts/merge_lane_throughput.py`; `scripts/tests/test_chain_telemetry_agreement.py`; the Deep chains card's
  reader in `static/redux/merge_queue.js` and its JSX; tests. Rule: a deep verify (`chain_items >= 2`) is a bisect
  round when `chain_depth_reason.basis == "bisect"`, otherwise fresh (absent on pre-8528 rows counts fresh); the
  deep-fail rate is over fresh rounds; card and digest add "bisect rounds: N (M red)".
  Signal: the digest, the card and `merge_lane_throughput.py` print the bisect count; the agreement test covers a
  bisect row. Prerequisites: α (same `merge_queue.js`); 8528, 10621. Size ~350–500.
- **ε — "Chain rounds" history panel; history-seam integration gate** (shared, orchestrator tests, dashboard).
  Files: `shared/src/shared/runs_read/merge_lane.py` (`chain_rounds`); `orchestrator/tests/test_runs_read_contract.py`
  (row written by a lane scene through a real `EventStore`); `data/merge_chains.py`; `data/redux_api.py`;
  `data/outcome_colors.py`; `static/redux/merge_queue.js`, `merge_chains.jsx`, `index.html`;
  `dashboard/tests/test_outcome_colors.py`; dashboard tests. Decisions D11–D13, D26, D27.
  Signal: on df, "Chain rounds" lists the window's rounds newest first with outcome badge, k of n, wall and stop,
  searches grouped under their opening round with the red suspect named, "newest N of M", following the WINDOW
  chip; a pre-β store reads "β not deployed"; H22–H24 and UI states pass.
  Prerequisites: β, γ (same files); θ 9852, 8547. Size ~1,000–1,300.

**No reverse edges.** Out-of-batch tasks sharing files with these leaves (θ 9852, γ2 9802, 8547 with α; 5245 with
γ) are serialised by the orchestrator's `metadata.files` locks, and their amendments read correctly in either
order (§11). Edges from them onto this batch would put 348 open tasks (mode=ro, 2026-11-18) behind α, β and γ.

```
8222 · 10619 → 10620 → 10621 → α → δ ← 8528
                                 α → γ ← β ← 8528, 10624, 10721
9852 → 8547 → ε ← γ
```

**Companion corrections** (`update_task` before `commit_planning`; texts in §11): A-8547, A-8329, A-5245, A-9852,
A-9802, A-8532.

**G7 walk (advisory now; blocking at decompose).** INV-1: typed, constructor-checked records; outcome totality.
INV-2: settle carries outcome, stop, cause, fault; rejected records name field and value. INV-5: one wire module.
INV-6: no liveness claim. INV-7: ledger bounds. INV-9: D15; the tip verdict on `merge_verify.passed` and on the
settle comes from one code path (named reconciliation). INV-10: behaviour tests and an executable audit. INV-11: the
ledger fails safe and loudly (D16); undecodable records counted. INV-12: ratchet raises carry this batch's task.
INV-13: "β not deployed"; the audit refuses an unrecorded window. INV-3, 4, 8, 14: no hit. No waivers.

## 11. Companion corrections (self-contained texts)

**A-8547** (append to details; no file added: the list already lives in this task's files):
> AMENDED 2026-11-18 (plans/merge-chain-visibility-prd.md). Show each deep row's members (`chain_task_ids`, head
> first); rows before 2026-11-15T11:19Z read "members unknown (before 9908)", never an empty list. When present,
> show `chain_id` and `origin_chain_id` (the PRD's leaf β adds them; a lone bisect verify carries only
> `origin_chain_id`) and `verdict_voided.verify_elapsed_ms` (task 8329). This list is the dashboard's one view of
> recent deep verifies; no other panel lists them.

**A-8329** (append):
> AMENDED 2026-11-18 (plans/merge-chain-visibility-prd.md). Whichever option is chosen, `verdict_voided` gains
> `verify_elapsed_ms: int | null`: the voided verdict's own verify time at point=adoption, null at point=dispatch.
> In df runs.db, 2026-11-10T16:37Z to 11-17T16:37Z, only 10 of 234 voids joined their verify by sha. The PRD's leaf
> β adds `chain_id` to the same payload at `worker.py::SpeculativeMergeWorker._emit_verdict_voided`; both additive.

**A-5245** (append; no dependency added):
> AMENDED 2026-11-18 (plans/merge-chain-visibility-prd.md). The finalized plan (2026-10-03) predates task 5371;
> re-plan. Do not add a merger-held verify window. Derive `verify_in_progress`, `occupancy` and `hosts[].slot_state`
> for held items from `worker.py::SpeculativeMergeWorker._verifies_in_flight` (task 5371), which already covers
> MERGING and DISPATCHING items and trains (`_train_verify_host`), so "what is verifying" has one derivation
> (heuristic 11). Do not add `request_id`, a `chain` field or a dispatching owner's host to `VerifyInFlight`: the
> PRD's leaf γ adds them. If γ has landed, read them.

**A-9852 (θ)** (append; no dependency added):
> AMENDED 2026-11-18 (plans/merge-chain-visibility-prd.md). If task 10621 has landed, port its single `chains`
> statement (window `merge_verify` and `merge_finalized` rows) to one constructor in `runs_read/merge_lane.py`,
> decoded by `summarize_chain_telemetry`, with a contract-test row, and do not port `speculative_stats` (retired by
> 10621). If it has not, do nothing for it: 10621 writes its read as a RunsQuery itself.

**A-9802 (γ2)** (append; no dependency added):
> AMENDED 2026-11-18 (plans/merge-chain-visibility-prd.md). If `dashboard/src/dashboard/data/merge_chains.py`
> exists, the merge-queue projector calls its live projection on the same probe reply (one probe, two
> projections). If it does not, the PRD's leaf α will call it from the `project` callable.

**A-8532** (append; no dependency added):
> AMENDED 2026-11-18 (plans/merge-chain-visibility-prd.md). Count a deep verify (`chain_items >= 2`) as a
> culprit-search round when `chain_depth_reason.basis == "bisect"`, otherwise as fresh (rows without the key count
> fresh); report the deep-fail rate over fresh rounds and bisect rounds separately. The PRD's leaf δ puts the same
> rule in the shared chain-telemetry module and `scripts/merge_lane_throughput.py`; if δ has landed, use it.

## 12. Out of scope

Chaining-behaviour fixes (10624, 10721, 6267, 8329's options); `merge_verify.depth` and the probe (4767, 9133); an
all-project chain figure and the speculative figures (10621); retiring the legacy member keys (D15); a hedge's own
rendering (8932); lane-wide questions routed elsewhere (10620, 8547, 9908).

## 13. Open questions (tactical)

1. **Budget sources at the mint.** `verify.py::merge_verify_command_budget_secs` and
   `merge_deep.chain_build_timeout_secs`. *Default:* compute both at the mint. Decide in β.
2. **Landed count.** Via `note_member_landed`, for the owner in `_advance_candidate`'s advanced arm and for each
   link in the walk. *Default:* as stated. Decide placement in β.
3. **Payload budget.** α and γ add at most hosts + held items per project; ε 30 rounds. *Default:* measure and
   append per the `payload_budgets.json` README. Decide per leaf.

## 14. Gate walk (pre-answered)

- **G1.** §2: every mechanism has a consumer.
- **G2.** §10: each leaf names a df-observable signal; β's is an executable audit that cannot pass vacuously.
- **G3.** §6; new substrate is built upstream of its readers; the GitOps fault seam and abort causes are verified.
- **G4.** §7, six order-independent companion corrections, no reverse edges; no reciprocal ownership.
- **G5.** B + H: §8 contract (with the end-per-exit table), §9 rows; γ gates the live seam, ε the history seam.
- **G6.** No numeric bound. "One settle per id" is the latch; the open-round clause is scoped to a run that did not
  restart and past an advisory lifetime; α's flag claims only what one reply shows (D18).
- **G7.** §10 walk; no waivers.
- **META.** Yes: queued unattended, the leaves build one model (rounds as recorded units keyed by a minted id, one
  owner per fact, one probe and one query per reading), the ledger cannot harm the lane, every exit has a named end,
  amendments read correctly in either order, no leaf touches 8528's files before 8528 lands, and every interim copy has a
  retirer (α's flag → γ).

## Revision 2 (PRD critic pass 1)

| Finding | Disposition |
|---|---|
| B1 settle raises in cleanup `finally` | Accepted, verified (`_finalize_inflight`, `_run_inflight_verify`, `_tear_down_inflight_entry`; `_land_rebased_or_park_gate` → `_finalize_landed_by_successor`). D16 fail-safe ledger and rejected records; H20; `stop head_landed_by_successor`, H18 |
| B2 amendment ordering | Accepted: edges 9852 → α, 9802 → α, 8547 → α, 5245 → γ; cycle-checked mode=ro |
| M3 missing exits | Accepted, verified the three abort causes; §8.1 end-per-exit table, `ChainFault`; H17 |
| M4 shutdown mislabels walks | Accepted, verified `stop()`'s finalizing-head comment; table row, H15b |
| M5 suspects overwritten | Accepted, verified `on_tip_result`; D13 `superseded`; H19 |
| M6 8532 clock | Accepted: edge dropped; A-8532 carries the rule as text (D28) |
| M7 α two sources | Accepted: α is the live panel only; recent rows via A-8547; A-9852 trimmed |
| M8 α orchestrator test | Accepted: regenerator moved to γ; α uses a hand-authored fixture with a skeleton check |
| M9 vacuous β signal | Accepted: `scripts/check_chain_round_records.py`, refuses unrecorded windows, CI over H fixtures |
| 10 lifetime | Accepted: formula and advisory status in D11; H16 declares a history leg |
| 11 owner counts | Accepted: I3, I4; H5 rewritten with four members |
| 12 five breaks | Accepted: D10 |
| 13 H7 seam | Accepted, verified the walk takes no verify lease; H7 overrides `advance_main`; arm marked defensive |
| 14 basis type | Accepted: D14, I2 |
| 15 stale status | Accepted: §3, §7, DAG refreshed (10723 done, 21,805 lines) |
| 16 module path | Accepted: cited by `summarize_chain_telemetry` |
| 17 settle key | Accepted: D8, settle by `chain_id` on `ChainResult` |
| 18 alone-next label, double wire | Accepted with variation: the mark is served as `unchained_mark` and search/depth state goes on the controller line (D19); `chain_reservations[]` tests-only (D22) |
| 19 A-8547 files | Moot under M7; stated "no file added" |

## Revision 3 (PRD critic pass 2)

| Finding | Disposition |
|---|---|
| N1 reverse edges block 348 open tasks | Accepted, verified mode=ro (9802: 348, 9852: 346, 5245: 277, 8547: 0 open descendants). All four edges dropped (8547 → α had no remaining reason); A-9852, A-9802, A-5245, α and γ made order-independent; §7 and §10 rely on `metadata.files` locks |
| N2 `stop()` and the walk | Accepted, verified `_shutdown_timeout = 5.0`. `stop()` never settles a finalizing round; fallback after the owner advanced is `landed_prefix`/`landed_whole` with `walk_interrupted`; `note_member_landed` covers the owner; H15b rewritten |
| N3 vacuous audit | Accepted, verified `EventStore.__init__` creates the store. Missing path refused first; owner-progress clause replaces lifetime; `unattributed` is a violation |
| N4 counter | Accepted, verified `EventStore.emit` never raises. `MergeMetrics` counter dropped; the rejected event is the one home; H20 rescoped |
| m1 | Accepted, verified 8528's `_is_code_verdict`; skipped tip is `aborted`/`verify_skipped` |
| m2 | Accepted: "a stop for the leaf: escalate" |
| m3 | Accepted: α's fixture captured once from df's real reply (D23) |
| m4 | Accepted: β names the 180-turn architect-cap tripwire and its split line |
