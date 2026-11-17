# PRD: An LLM adjudicator between the auto-consolidation predicate and any human

**Status:** active — authored 2026-11-17. Amends `plans/memory-auto-consolidation-prd.md`
(the "parent PRD"): §13 lists every parent decision this document replaces.
**Project:** dark_factory. **Approach:** B+H (contracts + boundary-test sketch; G5 hit: it
rewrites a load-bearing decision of a B+H parent and touches the Mem0 write path).
**Code anchors** verified against main `a473533aee` (2026-11-17). Main moves fast —
cite-by-symbol; re-locate at implementation time. `fm/` = `fused-memory/src/fused_memory/`.
The S19 module moves (tasks 9767, 9770, 9772) may relocate some files named here.
**Committed with this PRD:** `fused-memory/calibration/consolidation_adjudicator_eval_set.jsonl`
(+ `.md`), the labelled set §6 E7 scores against.
**Evidence trail** (untracked, main checkout only; nothing a task needs is left there):
`plans/detector-options-10244-2026-11-17/` — `prd-scout/`, `predicate-offline/`,
`adjudicator-pilot/`, `adjudicator-eval/`, `live/`.

## 1. Goal (G1 consumer + user-observable surface)

Leo's ruling of 2026-11-17: "I want to build towards designs where human attention is less
necessary (ultimately not necessary at all)." His principle of 2026-11-12: per decision area,
one clean partition between deterministic work done by code and judgement applied by an LLM
to the ambiguous residue, with a to-human escape hatch.

The parent PRD's partition is: Stage 1 (LLM) proposes a cluster; a pure predicate rules; a
PASS writes; every FAIL files a human curator gate. Two things are wrong with it:

- The predicate's FAIL codes mix true invariants, decidable facts whose resolution is a
  judgement, and text patterns standing in for judgement (a correction-banner regex, a
  slug-similarity threshold). All of them end at a human. The ambiguous residue skips the
  LLM tier.
- Nothing checks a PASS. That the notes teach one lesson is decided once, by the proposer,
  and written on its word.

This PRD inserts one narrow LLM call, the **adjudicator**, into the existing executor
post-step, makes the service prove that adjudicator on a labelled set before it may write,
and removes the human curator gate from the automatic path.

**Consumer:** the executor post-step
`fm/reconciliation/consolidation_auto_executor.py::run_auto_consolidation` (calls the
adjudicator, acts on its verdict); the report script and the review tasks (read its
records); Leo, only when a scripted check trips or a review finds something significant.

**What an operator observes when this lands:**

- For a staged project, every proposal the executor rules on carries either a code decision
  or an adjudication (decision, kept members, reason code, identity, cost) on its ledger
  row. `scripts/report_auto_consolidations.py` lists them.
- The ledger holds one calibration record per adjudicator identity, pass or fail with the
  observed figures. No write happens under an identity that has not passed.
- No `Consolidation gate: … (auto-consolidation refused …)` task is ever filed.
- A consolidated topic has a short index canonical pinned first in `search`, written with
  no human in the loop; `--revert` undoes it.
- One week and one month after switch-on an agent reads the rejected and the executed
  proposals and writes a dated review; after that a monthly scripted check stays silent
  unless rejection patterns cross thresholds the reviews set.

## 2. Background (measured 2026-11-17)

- **The executor has never run.** `fused-memory/config/config.yaml` leaves the
  `consolidation_auto:` block commented out, so `enabled_projects` is `[]` and
  `consolidation_auto_executor.py::_run` returns at its first check. Parent §11 step 2 (a
  human reads one dry-run cycle) is a step no task owns; task 5242, which carries its report
  script, has been pending since 2026-09-09 and was never dispatched.
- **Stage 1 already proposes.** The ledger holds 8 `consolidation_proposal` rows for
  dark_factory (7 topics), written 2026-11-11 to 11-14, all `proposed`. They expire
  2026-11-18 to 11-21, so none survives to switch-on.
- **The predicate over those 7, offline:** 5 PASS, 2 FAIL. Neither FAIL is a disagreement
  between notes. One is `member_carries_correction_banner` ×2 plus `member_not_found` (the
  banners sit on records repaired in place and the claim states the corrected fact; one
  member was deleted after the proposal). The other is `mixed_category`. Each would have
  filed a human gate, even in dry-run.
- **A FAIL files a gate even when `enabled` is false**
  (`consolidation_auto_executor.py::_gate_hazard` reads no switch); a dry-run moves a
  proposal out of `proposed`; a dry-run consumes the execution budget.
- **Rejected rows vanish after 7 days** (they keep `expires_at`; Stage 2's `ledger.gc`
  deletes them), which also resets the refusal streak. Stage 1 is shown nothing about
  earlier proposals, so nothing stops an identical re-proposal each cycle.
- **A dispatched task agent has no API keys.**
  `shared/src/shared/secret_env.py::agent_subprocess_env` strips every `*_API_KEY`. A
  measurement that calls a model therefore cannot be an ordinary task; the fused-memory
  service, which already calls OpenAI for the write-triage judge, can.
- **Pilot of the adjudicator** (63 labelled cases; prompt in §14, written once and not
  revised; gpt-6.1-sol at low effort). Figures under the keep rule of K4 and with the 7
  mixed-category cases removed, as K1 would refuse them in code:

  | Class | Result |
  |---|---|
  | the 6 real Stage 1 proposals expected to fold | 6 of 6 (including the two bannered records) |
  | folds kept, curator-folded and rater-approved groups | 30 of 33 = 0.91 (95% 0.76–0.97) |
  | a foreign record kept in a contaminated group | 0 of 10 (1 of 14 before the keep rule) |
  | proceeded on a group two blind raters said must not merge | 3 of 13 = 0.23 (95% 0.08–0.50); 5 of 13 before the keep rule |
  | invalid replies | 0 of 63 (strict JSON schema, generous timeout and retries; see K3) |
  | replacement claims that pass the existing claim rules | 33 of 33 |

  The keep rule was found after seeing these results, on these cases. Cost per call
  $0.0073; median latency 5.4 s, p95 10.9 s with six in flight. gpt-6-luna agreed with
  gpt-6.1-sol on 61 of 63 decisions, but its false folds keep whole groups, so the rule
  does not help it.

## 3. Sketch of approach

Flow per full cycle, for each ranked proposal (changes in bold):

1. Shape re-check and facts read, as today.
2. **Memo.** An adjudicator decision already made for the same members is reused with no
   call. A code refusal is always re-checked, but a repeat files nothing.
3. Predicate. **One table (K1) says who decides each reason code: code (no-op, repair,
   retry, refuse) or the adjudicator.**
4. **Code repairs** what is decidable (a deleted member, a member that belongs to another
   topic, a mis-declared category) and re-runs the predicate.
5. **Every proposal that would write, and every judgeable FAIL, gets one adjudicator call
   and one structured verdict (K3).**
6. **Code applies the verdict under fixed rules (K4), re-reads the facts, re-runs the
   predicate with the waivers it derived, and only then executes.**
7. **A closure failure after the writes reverts that execution.** No gate.
8. Every outcome is recorded on the ledger row with what was shown, and kept 90 days.

Before step 1, once per adjudicator identity: **the service scores the adjudicator on the
committed eval set and records pass or fail (K5). No pass, no calls and no writes.**

| # | Mechanism | Consumer |
|---|---|---|
| N1 | Reason-code class table and predicate waivers (`consolidation_auto.py`) | N4 |
| N2 | Adjudicator module: prompt, schema, validation, identity, keep rule, call (`fm/reconciliation/consolidation_adjudicator.py`) | N4, N5 |
| N3 | Ledger: states `dropped` / `unresolved`, adjudication record, members key, retention | N4 (writer); N7 |
| N4 | Executor flow: memo, repair, adjudicate, apply, revert; gate filing removed | `MemoryConsolidator.run` (existing call site) |
| N5 | Self-calibration (`fm/reconciliation/consolidation_adjudicator_calibration.py`) over N6 | N4 (may it call and write?); φ |
| N6 | Eval set `fused-memory/calibration/consolidation_adjudicator_eval_set.jsonl` | N5 |
| N7 | `scripts/report_auto_consolidations.py` (task 5242, amended): listing, `--revert`, `--json` digest | Leo; ρ1, ρ3; N8 |
| N8 | `scripts/check_auto_consolidation.py` (`code-live`, `live-clean`, `rejections`) + `fused-memory/config/consolidation-review.yaml` | gates Γ and φ; chain ρ2 |

## 4. Contracts (H)

### K1 — One partition table

`consolidation_auto.py` gains the single home of who decides each `AutoReasonCode` (a
mapping from code to class; a test pins that every enum member is classified):

| Class | Codes | What happens |
|---|---|---|
| **no-op** | `already_gated`, `already_consolidated`, `incumbent_canonical_stripped` (a disclosure) | unchanged |
| **repair** | `member_not_found`, `member_different_topic`, `member_already_canonical`, `proposal_category_mismatch` | the executor drops the offending member (for the category: uses the one category the members share), records the repair, re-runs the predicate |
| **retry** | `member_unreadable`, `canonical_count_unavailable`, and any failed read K3 or K4 adds | a read did not answer. Row stays `proposed`; the attempt is counted against the members (K4, "Bounded retry") |
| **refuse** | `member_id_repeated`, `no_retained_members`, `mixed_category`, `canonical_category_mismatch`, `multiple_canonicals`, `canonical_count_contradicted`; and, until the eval set carries cases for them, `member_carries_correction_metadata`, `canonical_carries_correction`, `slug_near_collision` | row `refused`. The checks are always re-run; a repeat of the same codes for the same members files nothing (K4, "Memo"). `multiple_canonicals` and `canonical_count_contradicted` say the store is wrong, not the proposal: the facts are re-read once, and if the code fires again one `blocking` record is filed under finding `store_inconsistent` |
| **adjudicator** | `member_carries_correction_banner`; and every PASS / PASS_TAG_ONLY | K3 |

- **The keep rule** (one pure function, home N2, used by the executor and the calibration):
  `kept ≥ member_min and 3 × kept ≥ 2 × proposed`, where `proposed` counts the members the
  proposal named, less a stripped incumbent. It is applied after repair (fail → `refused`,
  code `too_few_members_after_repair`) and again to the adjudicator's kept list (fail →
  `dropped`, code `too_few_kept`). It is integer arithmetic on purpose: keep 2 of 3 passes,
  keep 2 of 4 does not.
- **Size.** A proposal with more than `adjudicator_member_max` (8) members after repair is
  `refused` with `too_many_members_to_adjudicate`: the eval set measures 2 to 8.
- **A routine refusal reaches nobody.** A first refusal files the existing info record,
  folded per topic. A repeat for the same members files nothing and does not count toward
  `refusal_streak`. Cross-category consolidation and moving a record between topics are out
  of scope (§8), which is why those codes are decided by code and never sent to the model.
- **Only a measured judgement is delegated.** The eval set has 38 banner cases and none
  for correction metadata, a corrected incumbent or a near-colliding slug, so those three
  stay with code. Moving one to the adjudicator class is a table edit plus eval cases, and
  a new identity.

### K2 — Predicate waivers

`evaluate_auto_predicate` gains one keyword argument,
`waived: frozenset[tuple[AutoReasonCode, str]]` (code, subject member id). A waived hazard
is skipped. The function
raises if a waiver names a code outside the adjudicator class.

The model never writes a waiver. The executor derives them: on a `proceed`, every hazard
that was presented and whose subject is a kept member. Being
shown a hazard and keeping its subject is the judgement; an extra bookkeeping field for the
model to fill in would add a way to fail and no information. The predicate stays pure and
an import leaf.

### K3 — The adjudicator call

One home: `fm/reconciliation/consolidation_adjudicator.py`. It imports no executor code.

**Input** (rendered by one function from a typed request; the instructions are a module
constant; §14 is the starting text):

- the topic slug, the claim, the effective category;
- each member after repair: id, category, text cut at `adjudicator_field_chars`;
- each member the repair excluded because it belongs elsewhere: id, the topic it carries,
  the first paragraph of that topic's canonical. This is what lets the adjudicator say the
  proposal duplicates an existing topic;
- for PASS_TAG_ONLY: the incumbent canonical's text;
- the judgeable hazards as `{code, subject_id}`.

The incumbent and the excluded members' canonicals are new reads (`get_memories_by_metadata` returns no
content, so each is followed by `get_memory_by_id`). A failed read is the retry class.

**Verdict** (strict JSON schema; parsed into a frozen dataclass; unknown keys rejected):

```
decision:        "proceed" | "drop" | "unresolved"
keep_member_ids: [id, ...]          # subset of the shown member ids; [] unless proceed
claim:           str | null         # replacement claim, or null to keep the proposal's
reason_code:     "same_lesson" | "not_same_lesson" | "conflicting_claims"
                 | "stale_or_superseded" | "same_referent_as_existing_topic"
                 | "insufficient_information"
reason:          str                # one sentence
```

**A reply is a verdict only if** it parses, every kept id was shown, and a replacement
claim passes the claim rules (a callable the executor passes in; the claim arm of
`fm/server/consolidation.py::proposal_shape_codes`). Anything else is a typed **failure**.
Failures are of two kinds and the type says which: **transport** (timeout, connection, 429,
5xx) and **content** (everything else). A failure is never recorded as a judgement.

**The prompt states**, beyond §14: each hazard is rendered beside its subject; keeping a
member that carries a listed hazard means the adjudicator has read it and judges the fold
right all the same; a tag-only proceed attaches the members to an incumbent note that will
not be amended; a proposal whose lesson an existing topic already
teaches is a `drop` with `same_referent_as_existing_topic`.

**Identity:** one function returns
`sha256(model, reasoning_effort, endpoint host, instructions, input template, output
schema, adjudicator_field_chars, adjudicator_member_max, member_min, the keep rule's
constants)`.
It is stamped on every adjudication, on the minted canonical
(`x_auto_consolidated.adjudicator_identity`) and on the calibration record. Anything that
can change a verdict or what code does with it changes the identity. The provider's
`resolved_model` is recorded with every reply; it is not in the identity, and K5 says what
happens when it moves.

**Transport:** the OpenAI client path the write-triage judge uses. β turns
`fm/server/write_triage_judge.py::_call_openai_responses` into a shared helper that takes
the system prompt, the user content, the response format, `max_output_tokens` and the
timeout as arguments at call time (`fused-memory/scripts/write_triage_judge_wording.py`
swaps `JUDGE_SYSTEM_PROMPT` at run time, so the judge must keep passing its prompt per
call). The adjudicator uses a strict `json_schema` format, `max_output_tokens` 4000 and timeout
`adjudicator_timeout_seconds`; one attempt in the executor, up to three on a transport
failure in calibration. Default model `gpt-6.1-sol` at `low`. A
different model family from the Stage 1 proposer is deliberate: the second opinion should
not share the proposer's blind spots.

### K4 — What the executor does

**Members key.** `members_key = sha256(sorted (member id, content sha256))` over the members
the proposal names as read this cycle (a missing member hashes as missing). It ignores the
topic and the claim, which the proposer words freshly each time.

**Memo.** Once per cycle the executor loads the project's unexpired proposal rows and
indexes the **original** decisions (rows without `reused_from`) made in the last `memo_days`
(30). A row copied from a memo never extends one.

- *Adjudicator decisions.* Before a call, the executor computes
  `context_key = sha256(identity, members_key, excluded member ids, incumbent content
  hash)`. An original adjudication with the same key is reused with `reused_from`, and no
  call is made. What is reused is the verdict, not the row's state: a reused `drop` or
  `unresolved` ends the proposal as before, and a reused `proceed` (from a `dry_run` row)
  continues through steps 5 to 8 below under the switches as they are now. So a cluster
  approved while writes were off executes once they are back on, without a second call.
- *Code refusals.* The predicate and the rules always run; they are free and their inputs
  (categories, canonicals, config) can change while the members do not. When the same codes
  fire for the same `members_key` as in an original refusal, the row is `refused` with
  `reused_from` and no queue record is filed. When they no longer fire, the proposal
  simply proceeds.
- *Reverts.* An original `reverted` row refuses the same `members_key` with
  `closure_failed_previously` until its `memo_days` run out; then one more attempt is
  allowed.

**Order for a proposal:**

1. Predicate; repairs; keep rule; size rule; refuse class. (No model call so far.)
2. If the proposal would write and `max_auto_per_cycle` is already spent, leave the row
   `proposed` and untouched. No call is made for a write that cannot happen this cycle.
3. Adjudicator memo, else one adjudicator call. Failure → bounded retry (below).
4. `drop` → row `dropped`. `unresolved` → row `unresolved`. Nothing written, nothing filed.
5. `proceed` → keep rule on the kept list (fail → `dropped`, `too_few_kept`).
6. Re-read the cluster facts and re-run the predicate over the kept members with the
   derived waivers (INV-3: up to a minute has passed). A failed re-read is the retry class.
   A FAIL → row `refused` with the codes; NOOP → row `addressed` as today.
7. Execute (`apply_auto_verdict`) with the kept members, the effective claim and the
   effective category. The system record, the closure block and the outcome carry the
   **kept** members. For PASS_TAG_ONLY a replacement claim is recorded as a suggestion and
   not applied (parent D14).
8. Closure check fails → `revert_auto_consolidation` for that row; row `reverted` with the
   closure codes; one info record (`closure_failed_reverted`). The memo then refuses the
   same members, so execute-revert cannot loop. If the revert itself fails → one `blocking`
   record (`closure_failed_revert_failed`).

**Bounded retry.** A `proposed` row cannot carry an outcome, and each cycle's re-proposal is
a new row, so attempts are counted on their own ledger row, kind
`consolidation_adjudication_attempt`, keyed by project and `members_key`, expiring after 7
days. A retry-class code or an adjudicator failure increments it and leaves the proposal
`proposed`. At 3 the proposal becomes `unresolved` with code `unreadable` or
`adjudicator_failed`, and while the counter row lives a re-proposal of the same members
gets the same outcome with no call.

**Outage.** After its calls, the executor reads the project's last three Stage 1
`cycle_summary` rows (`flag_type='memory_consolidator'`; Stage 2 writes summaries too and
they carry no auto-consolidation stats), counting the current cycle. If each attempted at
least one adjudication (`…_adjudications + …_adjudicator_failures ≥ 1`, calibration calls
excluded) and none succeeded, it files one `blocking` record under finding
`adjudicator_unreachable`. The counter is the ledger, not process memory.

**Budgets.** `max_adjudications_per_cycle` (default 6) is the ranked slice and the cap on
calls; `max_auto_per_cycle` (3) caps executions; the backlog threshold is
`max_adjudications_per_cycle × backlog_multiplier`. `max_gate_filings_per_cycle` is retired.
Calls are awaited one at a time: worst case 6 × `adjudicator_timeout_seconds` added to
Stage 1's wall time, none of it on the loop thread (INV-8).

**Switches.**

| project staged | calibration for the running identity | `enabled` | Behaviour |
|---|---|---|---|
| no | – | – | skipped, as today; no calibration, no calls |
| yes | none, or `running` and fresh | – | start or wait for K5; this cycle is skipped |
| yes | `inconclusive` | – | skipped; K5 tries again next cycle, up to its cap |
| yes | `fail_first_run` | – | skipped; K5's second run happens next cycle |
| yes | `fail` | – | skipped, `auto_consolidation_skipped_unavailable='adjudicator_calibration_failed'`; rows stay `proposed` |
| yes | `pass` | false | dry-run: adjudicate and record (`dry_run` rows, memoised like any other decision), write nothing, file nothing, no execution budget used |
| yes | `pass` | true | live |

### K5 — Self-calibration

`fm/reconciliation/consolidation_adjudicator_calibration.py::run_calibration` scores the
running adjudicator over the eval set through the production call path, six calls in
flight. Each case goes through K1's case-decidable rules first (a `mixed_category` case is
not adjudicated and is left out of every denominator), then the call, then the keep rule.

- **The eval set is pinned in code.** Its sha256 is a constant beside the bounds and the
  file is resolved from the package, not from config. A missing or different file is not
  scored: the executor is skipped and one `blocking` record is filed
  (`adjudicator_eval_set_invalid`).
- **Three outcomes.** A transport failure is retried, up to three attempts per case. A case
  that still has no reply is *unanswered*. More than 5 unanswered cases, or more than 15
  minutes of wall time, make the run `inconclusive`: no verdict is recorded and the next
  staged cycle runs it again. Otherwise the run is scored against E7. A content failure
  is a wrong answer and an invalid reply.
- **One second run.** A run inside the bounds is a `pass`. A first failing run is
  recorded as `fail_first_run` and followed, next cycle, by a second full run; the verdict
  (`pass` or `fail`) is then taken on the two runs pooled (same rates, doubled counts). A
  bad adjudicator fails twice as surely; a good one is not turned away by one unlucky
  draw. There is no third run.
- **Record:** one ledger row, kind `consolidation_adjudicator_calibration`, keyed by
  identity, global to the service (not per project). It is written as `running` with an
  attempt number before the first call, using the ledger's compare-and-set so two projects
  cannot calibrate at once, and rewritten at the end with the outcome, every figure of E7
  with numerator and denominator, the `resolved_model`, the endpoint host, cost and time. A
  `running` row older than two hours is a crashed run. After three runs that ended
  `inconclusive` or crashed, one `blocking` record is filed
  (`adjudicator_calibration_stalled`) and no run starts for 24 hours; then the count starts
  again. A provider outage therefore delays the adjudicator and never retires it. `pass`
  and `fail` rows never expire.
- **Drift.** If a reply's `resolved_model` differs from the one on the `pass` record, the
  record is marked stale and the next cycle re-calibrates. A changed endpoint host is a new
  identity by construction.
- **When:** the first time a staged project's executor runs under an identity with no
  record. Unstaged projects never trigger it. About 180 calls and $1.30 per run.
- **Fail:** one `blocking` record under finding `adjudicator_calibration_failed` naming the
  bounds that failed, and the executor is skipped for every project until the identity
  changes. This is the escape hatch: someone with model access revises the prompt on the
  dev split and ships a new identity.
- **Bounds:** module constants beside the scoring function (one home, changed by code
  review, not by reload). §6 E7 records the filing values.
- **Test hygiene:** the test split is scored only here. Every identity ever scored leaves a
  row, so repeated attempts are visible. The dev split is for whoever revises the prompt.

### K6 — Records

- **Ledger row** (`fm/reconciliation/consolidation_proposal.py`): `ProposalState` gains
  `dropped` and `unresolved`. `ProposalOutcome` gains, each read with a default so existing
  rows still decode: `members_key`, `kept_ids`, `repaired` (`{code: [ids]}`),
  `effective_claim`, `effective_category`, `reused_from`, and `adjudication`
  (`{decision, reason_code, reason, waivers, claim_suggestion, identity, context_key,
  resolved_model, input_tokens, output_tokens, cost_usd, latency_s,
  shown: [{id, content_sha256, text}]}`), where `waivers` are the ones the executor derived
  and `text` is the first 1,500 characters of what the adjudicator saw. The row is the
  review's whole input: a reviewer needs no store access.
- **Retention.** A non-executed outcome gets `expires_at = acted_at +
  outcome_retention_days` (default 90). Executed rows keep no expiry, as today.
  `refusal_streak` skips rows that carry `reused_from`.
- **Queue records** stay under the two registered categories
  (`escalation/src/escalation/categories.py::EscalationCategory`), told apart by `finding`:
  `store_inconsistent`, `adjudicator_unreachable`, `adjudicator_calibration_failed`,
  `adjudicator_calibration_stalled`, `adjudicator_eval_set_invalid`,
  `closure_failed_reverted`, `closure_failed_revert_failed`, and the existing
  `refusal_streak`. `dropped` and `unresolved` file nothing: the row is their home.
- **Stats** (`auto_consolidation_initial_stats`, the one home): add
  `auto_consolidation_adjudicator_identity`, `…_adjudications`, `…_memo_hits`,
  `…_adjudicator_failures`, `…_dropped`, `…_unresolved`, `…_repairs` (`{code: n}`),
  `…_reverted`, `…_adjudication_cost_usd`, `…_calibration` (the outcome when a run ended
  this cycle); remove `auto_consolidation_gates_filed` and `…_unfiled`. Calibration calls
  are not counted in `…_adjudications`. The identity key is published on every Stage 1 cycle, staged or not: it is how
  Γ sees that the new code is the code running.

### K7 — Config (`fm/config/schema.py::ConsolidationAutoConfig`)

Removed: `max_gate_filings_per_cycle`. Added, all hot-reloadable
(`fm/config/reload.py::RELOADABLE_FIELDS`; `tests/test_config_reload.py` pins the leaf
count, which goes from 14 to 21, and the yaml comment "All fourteen leaves" follows):
`max_adjudications_per_cycle` 6; `adjudicator_model` `'gpt-6.1-sol'`;
`adjudicator_reasoning_effort` `'low'`; `adjudicator_timeout_seconds` 60;
`adjudicator_field_chars` 4000; `adjudicator_member_max` 8; `outcome_retention_days` 90;
`memo_days` 30. All eight land in one task (β). The root
settings ignore unknown keys, so a yaml naming these under older code is silently ignored;
Γ exists because of that.

## 5. Boundary-test sketch (the integration gate's signal — task η)

Executor side, with a scripted adjudicator double and the real ledger; one e2e test per row
in `fused-memory/tests/test_auto_consolidation_e2e.py`. Parent rows B4 and B8 are replaced.

| # | Scenario | Preconditions | Postconditions |
|---|---|---|---|
| A1 | Clean pass | PASS proposal; verdict `proceed`, all kept | canonical minted carrying the identity, members tagged, row `addressed` with adjudication; zero tasks |
| A2 | Bannered members kept | 2 members with a banner; `proceed` keeping both | executes; the row's derived waivers name both |
| A3 | Bannered member excluded | 3 members, 1 with a banner; `proceed` keeping the other 2 | executes over 2; no waiver recorded; the bannered member is untagged |
| A4 | Subset rescue refused | 6 members; verdict keeps 3 | row `dropped`, `too_few_kept`; nothing written |
| A5 | Keep 2 of 3 | 3 members; verdict keeps 2 | executes over 2; system record and closure name 2 |
| A6 | Deleted member repaired | 3 proposed, 1 missing | repair recorded; adjudicator shown 2; executes |
| A7 | Too few after repair | 4 proposed, 2 missing | row `refused`, `too_few_members_after_repair`; no call |
| A8 | Mixed category, twice, then fixed | members in 2 categories; re-proposed next cycle; then one member re-categorised and proposed again | first: `refused`, one info record; second: `refused` with `reused_from`, nothing filed, streak unchanged; third: adjudicated |
| A9 | Store inconsistent | topic with 2 canonicals, still 2 on re-read | row `refused`; one `blocking` record `store_inconsistent`; zero tasks |
| A10 | Unresolved, then re-proposed under another topic and claim | verdict `unresolved`; same members next cycle | second row `unresolved` with `reused_from`; adjudicator not called |
| A11 | Other-topic member | 3 proposed, 1 stamped with another topic | adjudicator is shown 2 members and the excluded one with its topic; `drop` `same_referent_as_existing_topic` → row `dropped`; nothing minted |
| A12 | Adjudicator down | the call raises for the same members in three cycles | attempt row counts 1, 2, 3; proposals `proposed`, `proposed`, then `unresolved` `adjudicator_failed`; a fourth proposal of the same members gets that outcome with no call; one `blocking` record `adjudicator_unreachable` |
| A13 | Calibration passes | staged, no record, double scores inside the bounds | row `running`, then `pass`; the next cycle runs A1 |
| A14 | Calibration fails twice | double keeps every foreign record | first run `fail`, second run next cycle, pooled `fail` naming the bound; one `blocking` record; executor skipped; no third run |
| A15 | Calibration inconclusive | double times out on 10 cases | no verdict; re-run next cycle; after three such runs one `blocking` record `adjudicator_calibration_stalled`; a run starts again 24 hours later |
| A16 | Dry-run | `enabled=false`, calibration passed, hazard input | adjudication recorded; row `dry_run`; zero writes, zero tasks; execution budget untouched |
| A17 | Closure failure | writes land, closure check fails; same members re-proposed | execution reverted, row `reverted`, info record; second row `refused` `closure_failed_previously`; zero tasks |
| A18 | Execution cap | 5 passing proposals, `max_auto_per_cycle` 3 | 3 calls, 3 executions; 2 rows untouched and `proposed` |
| A19 | Stale facts | a kept member is deleted between the call and the execute | the re-read sees it; row `refused`; nothing written |
| A20 | Old rows decode | a row written before this change | `outcome_from_record` returns it; the report script lists it |
| A21 | Eval set tampered | the eval file's sha differs from the constant | no scoring; executor skipped; one `blocking` record `adjudicator_eval_set_invalid` |

Adjudicator side (unit, no network): a malformed reply, an id outside the shown set, an
unknown key and an invalid replacement claim each return a content failure; a timeout and a
429 each return a transport failure; the identity changes when any of its inputs changes;
the keep rule's truth table for 2 to 8 members. Predicate side: a waiver outside the
adjudicator class raises.

## 6. Resolved design decisions

- **E1 Three tiers, one table.** Code decides, repairs and refuses what is decidable from
  typed facts; the adjudicator judges the residue and every write; a human is reached only
  by a `blocking` queue record, a failed scripted check, or a review that finds something.
  Replaces parent D1's "everything after emission is code".
- **E2 The adjudicator judges every proposal that would write** (Leo, 2026-11-17).
- **E3 No human curator gate on the automatic path** (Leo, 2026-11-17). A cluster that
  cannot be resolved is left unconsolidated and recorded. `_gate_hazard`, `_gate_unclosed`
  and `_file_hazard_gate` are deleted from the executor with their tests. The gate
  builder's auto-refusal arm stays: `scripts/migrate_open_consolidation_gates.py` reads it.
  Replaces parent D8 and D9. The `already_gated` rung stays: a human sitting can still own
  a topic.
- **E4 A separate call in the executor post-step.** Not Stage 2 (its job is task-to-memory
  sync, and a ruling buried in a long agent turn cannot be scored), and not Stage 1
  re-proposing (it would judge its own proposal).
- **E5 The verdict may only shrink the member list and replace the claim.** Keeping the
  subject of a presented hazard is how it accepts that hazard. It cannot add members,
  change a category or move a record between topics.
- **E6 Code keeps the last word.** Code derives the waivers, re-reads the facts and re-runs
  the predicate after the verdict, and the keep rule turns a `proceed` into a `drop` without
  asking again.
- **E7 The service proves its adjudicator before it may write** (Leo, 2026-11-17: a scripted
  gate replaces the human read). K5 passes on all of:

  | Bound | Split and population | Definition | Basis (pilot, dev split) |
  |---|---|---|---|
  | contaminated false-keep ≤ 6 of 45 | test, 45 contaminated | the foreign record is in the kept list after the keep rule | 0 of 10; 1 of 14 before the rule. A true rate of 7% passes 96 times in 100 |
  | fold rate ≥ 32 of 45 | test, 45 curator-consolidated groups | `proceed`, and the kept list passes the keep rule against the case's members | 30 of 33 overall; 13 of 15 on the curator class, the test split's kind |
  | must-not-merge proceeds ≤ 15 of 30 | test, 30 three-topic triples | decision `proceed` after the keep rule | a sanity floor, not a measured bound: an adjudicator that merges half of three unrelated topics is broken |
  | must-not-merge proceeds ≤ 5 of 13 | dev, the rater-judged class | same | 3 of 13 with the rule, 5 of 13 without it |
  | real-proposal folds ≥ 5 of 6 | dev | as fold rate | 6 of 6 |
  | invalid replies ≤ 5% | every adjudicated case | content failures, plus cases left unanswered | 0 of 63 malformed; 33 of 33 replacement claims pass the claim rules |

  Always-proceed fails the first and third bounds; always-drop fails the second. An
  adjudicator at the pilot's rates passes all six about 87 times in 100 on one run, and more
  often with the pooled second run. The two dev-split bounds are regression guards on cases
  the keep rule was fitted to, and the record labels them so.

  What the set cannot show: it has no tag-only case (an incumbent shown) and no case with
  an excluded member shown, so those two inputs are extrapolations from the measured
  judgement; ρ1 and ρ3 read every tag-only execution and every execution that carried a
  waiver. The test split's must-not-merge cases are an easy construction
  (three different curated topics). The hard class, near neighbours two raters refused, is
  the 13 dev cases only. 38 of the 45 contaminated test cases reuse a fold case's group with
  a different member sample, so the two kinds are not independent.
- **E8 Stated limit.** On a group that should not be merged at all the adjudicator still
  proceeds about a quarter of the time: 3 of 13 (95% 8–50%), on 13 cases, after a rule fitted
  to them. That is a per-ruling rate. The memo makes a drop final for those members for
  `memo_days`, so a cluster is not re-drawn until it changes or the memo lapses; a proposer
  that varies the member set gets a fresh draw each time. Stage 1's real proposals were good
  7 times in 7, which is seven cases. This is accepted for Stage 1 as the proposer because a
  write is reversible, a minted canonical carries the identity that approved it, and ρ1 and
  ρ3 read executed rows, not only rejected ones. Any other proposer (§8) must be scored on
  its own proposals first.
- **E9 Calibration lives in the service, authorised by a ledger record.** A dispatched agent
  has no API key and cannot write the live ledger; the service has both. No yaml pin and no
  committed report: the record is the one home of "this identity may write".
- **E10 A closure failure reverts**, and the memo stops the loop. Writes are reversible by
  construction (parent D7).
- **E11 A rejection holds for its members until they change or 30 days pass** (the memo),
  and its row is kept 90 days. A code refusal is re-checked every time; only its queue
  record is suppressed on a repeat.
- **E12 Rejections and executions are reviewed** (Leo, 2026-11-17): by an agent after a week
  and after a month, then by a monthly scripted alarm whose thresholds those reviews set.
- **E13 Per-call cost is recorded** on the adjudication. None of the three existing judges
  persists it.

Code-quality heuristics (`docs/code-quality.md`) these rest on: **SPOT** (one class table,
one prompt, one identity function, one keep rule, one stats home, one OpenAI helper, one
set of bounds); **structured data instead of meaningful strings** (verdict, waivers,
findings, the calibration record); **clear invariants, redundantly enforced** (E6);
**well-defined purpose for each entity** (E4); **prefer stateless interactions** (the
adjudicator is a function of its request; the memo is keyed on the members);
**files make internal sense in isolation** (the adjudicator imports no executor code; the
predicate stays an import leaf).

## 7. Pre-conditions / substrate (G3 — verified 2026-11-17 against main `a473533aee`)

| Assumed | Evidence |
|---|---|
| Executor call site, awaited, full cycles only | `fm/reconciliation/stages/memory_consolidator.py::MemoryConsolidator._run_auto_consolidation` |
| Outcome is a payload dict; new fields need defaults | `consolidation_proposal.py::ProposalOutcome.from_payload` (`_str_tuple` raises on a missing key) |
| A subset of members and a replaced claim execute | `consolidation_auto_executor.py::apply_auto_verdict` takes `claim` and `verdict.retain_ids`; `fm/services/consolidation_ops.py::apply_retain_arm` validates neither |
| Reversal exists and is tested | `consolidation_auto_executor.py::revert_auto_consolidation` (no production caller today) |
| Benign-refusal channel and streak | `consolidation_auto_executor.py::_refuse`, `consolidation_proposal.py::refusal_streak` |
| Queue categories are a closed enum; this PRD adds none | `escalation/src/escalation/categories.py::EscalationCategory` |
| The service can call OpenAI | `fm/server/write_triage_judge.py::_call_openai_responses`, live with `write_triage.enabled: true` |
| Every full cycle writes its stats to the ledger | `fm/reconciliation/summary_pool.py::write_cycle_summary` |
| Unknown config keys are ignored, not rejected | `fm/config/schema.py` root settings (`extra='ignore'`) |
| Predicate-task shape; `recheck_secs` needs a `dated` milestone | `docs/task-authoring.md` §5–§6 |
| Recurring chain carrier | `docs/task-authoring.md` §6.1 |
| A task agent reads main-checkout stores read-only | `CLAUDE.md` "Forensic reads of tasks.db"; `scripts/tasks_db_schema.py` resolves the live store from a worktree |
| Eval set | committed with this PRD (N6) |

**Not on main, built here:** every mechanism in §3 except N6 and N7's task (5242,
existing, amended).

## 8. Out of scope

- A second proposer (duplicate-detector groups fed to the executor). The live measurement
  of 2026-11-17 supports it; E8 applies. A separate PRD.
- Telling Stage 1 which members were rejected. The memo makes a repeat free; the reviews
  report how often it happens.
- Cross-category consolidation, re-categorising a record, moving a record between topics,
  retargeting a near-colliding slug, amending an incumbent canonical.
- Adjudicating correction metadata, a corrected incumbent or a near-colliding slug. Each
  needs eval cases first (K1).
- The backlog brake. Unchanged; it must be revisited before a second proposer.
- Staging reify or any other project. ρ1 recommends and files it.
- The process finding that task 10243's agent took `OPENAI_API_KEY` from the main
  checkout's `.env`. Reported to Leo; not fixed here.

## 9. Cross-PRD / seam ownership (G4)

| Other | Direction | Seam | Owner | Status |
|---|---|---|---|---|
| `plans/memory-auto-consolidation-prd.md` | amends | executor, predicate, proposal ledger, config block, §11 rollout | this PRD | parent gets a dated pointer (§13) |
| same, task 5242 (ζ) | consumes | `scripts/report_auto_consolidations.py`, skills, `OPERATIONS.md` | task 5242, amended at decompose to this PRD's ζ | pending, never dispatched; task 10419 depends on it |
| `plans/write-triage-link-healing-prd.md` | precedent only | in-service calibration before writes | that PRD | no shared code |
| write-triage judge (tasks 5277, 9946, 10508) | shares | the parameterised OpenAI helper | β extracts; those tasks rebase onto it | queued |
| review-all S16 ε2b (task 9700) | consumes later | role-based model routing for fused-memory judges | task 9700 | when it changes the model name the identity changes and the service re-calibrates by itself |
| review-all S19 (tasks 9767, 9770, 9772) | file moves | `recon_ledger`, `consolidation_gate`, `server/consolidation` | those tasks | serialise by dependency at decompose |
| tasks 5881, 7499 | same file, and semantics | topic-guard seeding, and what a revert leaves behind | those tasks | 7499 becomes load-bearing once the executor reverts by itself: 5881 must not land without it |
| tasks 7151, 8149 | same files | executor and predicate | those tasks | serialise by dependency at decompose |
| `plans/live-task-facts-at-recall-prd.md` κ (task 10087) | same file | `stages/memory_consolidator.py` | task 10087 | this PRD does not edit that file |

## 10. Decomposition plan (signals are the G2 gate; Greek labels local to this PRD)

- **α — Predicate and ledger shape.** K1's table and classes, K2; K6's
  states, outcome fields, `members_key`, retention, the streak rule. Files:
  `consolidation_auto.py`, `consolidation_proposal.py`, `recon_ledger.py`, their tests.
  *Unlocks:* γ.
- **β — Adjudicator, calibration, helper, config.** K3, K5, all of K7, the eval-set loader.
  Files: the two new modules, `write_triage_judge.py` (helper only), `config/schema.py`,
  `config/reload.py`, `tests/test_config_reload.py`, the yaml comment block, their tests.
  Deps α (it uses K1's table and the two new ledger kinds). *Unlocks:* γ.
- **γ — Executor flow.** K4, K6's records and stats; deletes the three gate functions.
  Files: `consolidation_auto_executor.py`, its tests. Deps α, β. *Unlocks:* η.
- **η — Integration gate.** §5's twenty-one rows as e2e tests through the real ledger, the
  real executor and the real calibration function with a scripted model double. Deps γ.
  *Signal (leaf):* the e2e file passes, and the capability manifest maps A1–A21 to test
  names.
- **ζ — Review and check surfaces (task 5242, amended).** `report_auto_consolidations.py`
  (adjudication columns, calibration records, `--revert`, `--json` with the rows' `shown`
  text; resolves the main-checkout ledger from a worktree);
  `scripts/check_auto_consolidation.py` with `code-live`, `live-clean` and `rejections`;
  `fused-memory/config/consolidation-review.yaml` (the alarm thresholds, one home); the
  watcher playbook rows for the new findings; `OPERATIONS.md` (the switches table, the kill
  switch, how to read a rejection, what to do on `adjudicator_calibration_failed`); the
  parent PRD's pointers. Deps η. *Signal (leaf):* against a ledger written by η's run the
  report lists an execution, a dropped row, an unresolved row and a calibration record, and
  each check prints "no producer has written here" and exits 75 on an empty ledger
  (INV-13).
- **Γ — Gate: the adjudicator code is the code running** (deterministic, predicate
  `check_auto_consolidation.py code-live`, `script_root: project`; deps ζ; `dated`
  milestone set at decompose; `recheck_secs` 86400). Exit 0 when dark_factory's newest
  Stage 1 `cycle_summary` row (`flag_type='memory_consolidator'`) carries
  `auto_consolidation_adjudicator_identity`; 75 until then. `delivered_checks` re-runs it.
  *Unlocks:* ε.
- **ε — Switch on dark_factory** (normal, `complexity: simple`). Uncomment
  `consolidation_auto:` in `fused-memory/config/config.yaml` with
  `enabled_projects: ['dark_factory']` and `enabled: true`. Deps Γ. It takes effect at the
  next fused-memory restart (the fleet redeploy restarts that tier) or `reload_config`; ε
  calls neither, φ watches for it. *Unlocks:* φ.
- **φ — Gate: calibrated, live and clean** (deterministic, predicate
  `check_auto_consolidation.py live-clean`; deps ε; `dated` milestone; `recheck_secs`
  86400). Exit 1, each with its reason in the note, if: the running identity's calibration
  is `fail` (not `fail_first_run`, `running` or `inconclusive`, which are 75); a `closure_failed_revert_failed`, `adjudicator_calibration_stalled` or
  `adjudicator_eval_set_invalid` record is open; or the newest Stage 1 summary still shows
  the project skipped more than 72 hours after the yaml change landed (the config was not
  applied). Exit 75 until a `pass` record and at least one adjudication exist; else 0.
  *Signal (leaf):* the gate closes on rows the live service wrote.
- **ρ1 — First review** (normal agent task; deps φ; `milestone: {mode: delayed, after_secs:
  604800}`). From the report's JSON, which carries what the adjudicator was shown, read
  every `refused`, `dropped`, `unresolved` and `reverted` row and every executed row since
  switch-on. Judge: was each rejection right;
  was each fold right; is a class of rejection a missing capability (cross-category, a
  topic that should have been retargeted); is Stage 1 re-proposing rejected members. Append
  a dated entry to `docs/auto-consolidation-reviews.md`, write the findings to
  fused-memory, set `consolidation-review.yaml`'s thresholds with the reasoning, and file a
  task or an `escalate_info` for anything significant, including whether to stage reify.
  *Signal (leaf):* the dated entry and the thresholds change land on main.
- **ρ3 — Second review** (as ρ1; deps ρ1; `milestone: {mode: delayed, after_secs:
  2592000}`). Its entry also says whether further agent reviews are warranted and files the
  next one if so.
- **ρ2 — Monthly alarm** (recurring chain `consolidation-rejection-review`,
  `interval_secs` 2592000; deterministic, predicate `check_auto_consolidation.py
  rejections`; seed link `dated` 45 days after decompose, `recheck_secs` 604800; deps φ).
  Exit 75 only while no `pass` calibration exists. Exit 0 when fewer than 20 proposals were
  ruled in the last 30 days (too few to say anything) or every threshold in
  `consolidation-review.yaml` holds; else exit 1, which reaches Leo as an L2 escalation
  carrying the digest. Filing values, to be replaced by ρ1: rejected share ≥ 0.60 of ruled
  proposals; one reason code ≥ 25 original rejections; one `members_key` rejected afresh in
  3 or more memo windows; any open `blocking` record. They are alarm thresholds set from
  one data point (2 of 7 rejected offline, before the adjudicator existed), which is why ρ1
  owns them.

Decompose-time actions, not tasks: amend task 5242 to ζ's scope and add its dependency on
η; add the parent's dated pointer (§13); wire §9's file-overlap dependencies; set the three
`dated` milestones.

## 11. Rollout

Replaces parent §11. No step waits on a person.

1. α, β, γ, η, ζ land. Defaults leave every project unstaged, so nothing calls a model.
2. Γ sees the new code running. ε stages dark_factory with writes on.
3. On the first staged cycle the service calibrates. Pass: it is live from the next cycle.
   Fail, after the pooled second run: it files one `blocking` record and stays off; φ
   fails and Leo hears. Inconclusive: it tries again, three times at most.
4. φ closes. ρ1 after a week, ρ3 after a month, ρ2 monthly.
5. Kill switch: `enabled: false` stops writes within one cycle; removing the project from
   `enabled_projects` stops the calls; `--revert <ledger_key>` undoes one execution.

## 12. Open questions (tactical, implementation-time)

1. **Where the shared OpenAI helper lives.** Beside the judge or in a small `fm/llm/`
   module. Decide in β; task 9700 may move it again.
2. **How the ledger addresses a row that belongs to no project** (the calibration record).
   Suggested: a reserved project id. Decide in β with `recon_ledger`'s owner tests.
3. **`adjudicator_timeout_seconds`.** 60 is about six times the pilot's p95. The
   calibration's invalid-reply bound will say if it is too tight.
4. **Whether the first paragraph of a canonical is enough context for an excluded member's
   topic.** Decide in γ against A11.

## 13. Parent decisions this document replaces

| Parent | Replaced by |
|---|---|
| D1 "everything after emission is code" | E1, E4 |
| C2 predicate signature | K2 (adds `waived`, derived by the executor); the reason codes are unchanged, K1 classifies them |
| C4 "hazard FAIL files a gate exactly as when enabled"; dry-run semantics | K4's switches table; E3 |
| C6 hazard gate shape; D8; D9 | E3 (the executor files no gate) |
| D10 benign refusals to the reconciliation queue | kept; widened to the refuse class; a repeat is a silent memo hit |
| B4, B8 | A2, A3, A16 |
| §11 rollout, including the human-read cycle | §11 here; E7; E9 |
| `max_gate_filings_per_cycle` (D6's shared block) | `max_adjudications_per_cycle` (K7); task 4916, the block's other consumer, was cancelled 2026-11-17 |

## 14. Appendix: the piloted instructions (dated snapshot, 2026-11-17)

The text the pilot's figures were measured on. β starts from it and must change it to match
K3 (the banner hazard rendered beside its member and what keeping it means, the excluded
members, the incumbent for tag-only, the `same_referent_as_existing_topic` code; rule 3 goes, since code refuses mixed categories).
Any change is a new identity, which K5 scores before it may write. Once β lands, the module
constant is the home and this appendix is history.

```text
You review one proposal to consolidate notes in a knowledge base that software agents write and later search. The proposal names a set of member notes and a one-sentence claim meant to state the lesson they share.

What proceed does: a short index note carrying the claim is written and pinned first in search results for this topic, and each kept member is tagged with the topic. Nothing is deleted. A wrong proceed pins a misleading claim above notes it misrepresents and ties unrelated notes together; a wrong drop only leaves the notes as they were.

Rules:
1. Notes belong together when they teach the same lesson about the same subject: a restatement, a rediscovery, the same lesson with different supporting detail, or a note and its later correction when the claim states the corrected fact. They do NOT belong together when they merely share an area, a tool, a file, a task or a mechanism but teach different things.
2. Exclude members that do not belong rather than dropping the whole proposal: if at least two members belong together, answer proceed and list exactly those members in keep_member_ids.
3. A consolidated note gets ONE category. All kept members must share one category. If the members that belong together span categories and no single-category subset of at least two of them belongs together, answer drop with reason_code cross_category.
4. When the notes make incompatible claims and the texts do not show which is current, answer unresolved with reason_code conflicting_claims. Never guess which one is current. A note that explicitly corrects or supersedes an earlier one shows which is current; that is not a conflict.
5. If no two members teach the same lesson, answer drop with reason_code not_same_lesson. If the shared lesson is no longer true because the notes are stale or superseded, answer drop with reason_code stale_or_superseded. If the texts are too thin to judge, answer unresolved with reason_code insufficient_information.
6. The claim must be right for the kept members: one sentence, at most 200 characters, one assertion, stating the lesson they share and nothing they do not support. If the given claim is wrong for the kept members (says something they do not say, is broader or narrower than their shared lesson, or states a fact they correct), put a replacement in claim; otherwise claim is null. Give a claim only when the decision is proceed.
```

Input template of the pilot: the proposed claim, the member count, each member as
`### Member {id}` / `category: {category}` / text, then "Facts computed from the records"
with one line per hazard, or "none".
