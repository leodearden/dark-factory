# Decision stream: a supervised executor tier, one sitting over one docket, and no resident watchers

**Status:** active PRD, authored 2026-11-04 (`/prd`, fable session team-df-3597171, T3 thread
"team:df watcher-design") from `plans/portfolio-review-2026-11-01/followups/watcher-design.md`,
whose §6a records every ruling Leo gave on 2026-11-04. All design questions are ruled; none is
re-opened here. **B + H**: five packages, eleven mechanisms, and the escalation authority seam.
**Code anchors** cite symbols; verified against main `526cd1e3d5` (2026-11-04). Main moves fast —
re-locate at implementation time.
**Evidence:** `followups/watcher-design.md` §1 and §8 (figures with rules), seats
`followups/watcher-design-seats/W1–W6.md`, critic `C1.md`; `seats/X1.md`, `seats/Q6.md`,
`followups/preparer.md`.

## 1. Goal

The human tier of the escalation ladder becomes a decision stream: Leo reads one prepared docket
and rules by number; everything that is mechanical given a ruling already made is executed by a
headless rotation the orchestrator supervises, under a server-enforced identity and policy table,
with every autonomous act recorded and sampled for audit; nothing on the human side is resident.
The per-project interactive L2 watcher sessions (median 75 h, 777k tokens, 74% empty wakes, dead
leases, no attribution; watcher-design.md §1.2) are retired, and the serious classes they used to
relay within an hour reach a sitting within minutes through a criteria-triggered launch.

### What an operator observes when this lands

- `scripts/orchestrator-watchdog.py --report` and `runs.db` show, per project, two supervised
  rotations: `escalation-watcher-auto` (levels 0,1) and `escalation-executor` (levels 0,1,2), each
  with `rotation_started` / `rotation_exited` events carrying role, outcome and cost.
- A `provenance_unattributed` L2 whose task's deliverable is carried by a named sibling commit is
  closed by the executor: the task flips to done `found_on_main`, the record carries
  `policy_class=landing_provenance`, `resolved_by=orchestrator-escalation-executor`, and a
  DecisionRecord with the quoted evidence exists; the next return brief lists it under
  "Autonomous closes" and, one in ten, as `audit N`.
- A record outside the policy table, or in a class whose switch is `shadow` or `off`, is left
  pending with a `stamp_triage` recommendation; the next docket renders it as a numbered item.
- An executor `resolve_issue` on a `milestone_gate`, on a record filed by `orchestrator-watchdog`,
  or on a class whose switch is not `live`, returns `{'code': 'level_forbidden'}` and changes
  nothing.
- A red main confirmed by two signals is recovered by the executor with one `recover_main` CAS
  move; the DecisionRecord quotes both signals and the target; the class renders at 100% in the
  audit block until twenty have happened.
- A B3 low-risk proposal is applied by the executor through `unblock-low-risk` in a worktree and
  lands through the merge queue; a handed-off merge survives the rotation boundary in
  `data/orchestrator/executor-inflight.json` and is polled by the next rotation.
- Leo types `/sitting` in a T3 thread: it seeds from `ledger-nightly.json`, re-verifies stale
  preparations, takes `N=OPTION` answers and `audit N=ok|wrong` answers, and applies each ruling
  with one `apply` verb that refuses a record no longer pending. A `wrong` flips that class to
  `shadow`.
- When a trigger class or the accumulation score fires and Leo has typed in T3 within the last
  P minutes, a `/sitting` thread opens in T3 with the docket loaded (a gnome-terminal window when
  T3 is down). Nothing notifies him. When he is absent, nothing launches: the executor already
  handles the mechanical half within a minute.
- The veto-streak companions and info-only strand sweeps no longer reach L2; the recon queue's
  self-clearing findings are closed by the executor and the rest render in the docket.
- No `watcher-<project>` lease is claimed by anything; `recon-watch/run.sh` is gone; the
  `escalation-watcher` skill is the `sitting` skill.

### Consumers (G1)

| mechanism | consumer |
|---|---|
| policy table + executor identity in `authority.py` (α) | `server.py::resolve_issue` (enforces it), the executor rotation (reads it), the sitting's `apply` verb (reads class state for audit flips), the return brief §2/§6 |
| shadow-vocabulary lockstep (α: `FIRST_TRANCHE_CLASSES`, `REVERSIBLE_ACTIONS`, `_HUMAN_RESOLVERS`) | `shadow_ruling.agreement_report`, rendered by `return_brief.py::_agreement_lines`; the η flip gate reads it |
| role-parameterised rotation supervisor + rotation events + inflight record (β) | the executor rotation (hosted), the L1 rotation (unchanged host), `runs.db` forensic reads, OPERATIONS.md, the ι observation |
| executor skill (γ) | the β supervisor loads it as the executor rotation's system prompt |
| `apply` verb, audit draw and outcome, docket blocks (δ1) | the sitting skill (δ2); the nightly `return_brief.py` render; Leo reading the page |
| sitting skill (δ2) | Leo, in the T3 thread the trigger (ε) or he opens |
| trigger evaluator + timer (ε) | the sitting (launched); the docket's "trigger state" line (readers prove the producer) |
| Q6 detector rules in the L1 skill (ζ) | the L1 rotation; the executor (fewer records); X1's derivative count |
| flip gate (η) and observation (ι) | Leo (η is the one human gate); ι is the INV-13 proof the producers wrote in production |
| retirement (θ) | every future session that would have launched a resident watcher; OPERATIONS.md readers |

## 2. Background

- The L1 rotation (sonnet, 4 h / 50 items, $532 in 32 days) is the only supervised watcher; it
  closes benign info records and wraps the rest one-to-one into L2s (92% single-member, 165 s
  after the L1). The interactive L2 session closes 647 of those 771 and 805 of all 932 closed
  L2s, unmanaged: nothing launches, restarts or reaps it; four leases are held by dead pids; no
  L2 watcher has run since 11-01. watcher-design.md §1.1–1.2, W1, W2.
- Server-enforced authority today: `authority.py::ROLE_LEVEL_ALLOWLIST` caps one identity
  (`orchestrator-escalation-watcher-auto`) at levels {0,1}; `authority.py::l2_auto_close_class`
  (task 2630) lets that identity `close_only` three classes on evidence regexes;
  `authority.py::PROMOTE_ALLOWED` gates minting. Any other identity, or none, is unrestricted.
  W6 §3, W4 §3.
- Leo's self-execute grants (items 11–14, 10-29 to 10-31) already let the interactive session
  close classes 1b/2/3; they live in `skills/escalation-watcher/SKILL.md`, with no server gate, no
  DecisionRecord and no audit. The shadow framework has never been stamped (n = 0); the audit
  sample draws positionally over DecisionRecords none of which is a policy close. W3 §1, §4.
- The sitting tooling (`scripts/sitting/prepare_sitting.py` verbs) is cross-project but has no
  verb that applies a ruling and no queue→port map; the 10-25 coordinator applied 15 records
  through an untracked port client and stamped 18 invented sitting ids. W4 §1–2.
- Rulings this PRD implements (watcher-design.md §6a): executor + on-demand and
  criteria-triggered sitting; gnome-terminal fallback; the moves list; executor runs
  `recover_main` and may find the SHA; executor posture widened under four conditions; skeleton
  reversibility condition waived for the done-flip; ten random audits a week with class flips;
  recon queue folded in; host alarm home = dark-factory's queue (8270); Q6's three detector
  answers.
- Related PRDs: `plans/escalation-connection-capability-guard-prd.md` (the authority seam this
  PRD extends), `plans/escalation-l2-tiering.md` (the ladder; its race-by-construction argument
  is why class-keyed disjointness is server-enforced here), `plans/startup-hardening-prd.md`
  (8270's host alarm, a deny role here), `plans/fleet-cockpit-prd.md` (DecisionRecord),
  `plans/escalation-lifecycle-dashboard-prd.md` (`resolution_class`), the preparer follow-up's
  6214/8253.

## 3. Resolved design decisions

1. **The ladder stays; the roles change.** L0/L1/L2 and the server level gate are unchanged. Two
   headless rotations per project share one supervisor; the human tier is a sitting, never a
   resident session.
2. **One policy table, one home.** `escalation/src/escalation/authority.py` holds the table:
   rows keyed on `(category | agent_role pattern | root_cause pattern | "applies a named ruling")`
   → allowed actions, evidence regexes (presence-only, as 2630), reversible flag, and a switch
   `off | shadow | live` persisted in `data/escalations/policy-table-state.json` (one file per
   queue, written by the sitting's audit flip and by the storm escape, read by the server on
   every executor call). The executor skill carries procedures per class, and a contract test
   holds the skill's class list equal to the table (no-lockstep-duplication, the same device as
   `tests/scripts/test_shadow_ruling_doc_contract.py`).
3. **Two new identities, both server-known.** `orchestrator-escalation-executor`: levels
   {0,1,2}, every action class-gated, deny categories `L2_AUTO_CLOSE_DENY_CATEGORIES` plus deny
   roles `orchestrator-watchdog`, `dashboard-watchdog`, `orchestrator-deterministic`; `promote`
   allowed. `interactive:coordinator`: full levels, denied only on a record carrying a fresh
   executor claim; a member of `classify._HUMAN_RESOLVERS` so sitting closes count in the human
   tier. Header-less connections keep full authority (the esc-2087-2 guarantee is untouched).
4. **The class claim is a record field.** The executor writes `policy_class` and `claimed_at`
   through a new server tool `claim_policy_class` before acting; the claim expires with the
   triage-ack freshness window (6 h). This is W3 §5's missing class key, the coordinator deny's
   input, and how class 3 (a disposition, not a category) becomes selectable.
5. **Executor tool posture is the interactive posture, bounded** (Leo, Q4): Edit/Write/git
   allowed, but file-writing classes act only in a task worktree and land through the merge
   queue; the one direct-to-main write (sidecar/docs commit) goes through `git commit --only`;
   a file-writing class is skipped for the rotation when `project_root` has uncommitted changes;
   in-flight work is journaled in `data/orchestrator/executor-inflight.json` before the act and
   read at rotation start; file-writing classes roll out project by project (dark-factory
   first) and are audited at 100% until n ≥ 20; the executor rotation runs opus.
6. **Red-main recovery is an executor class** (Leo, Q3): red confirmed by two independent
   signals (merge-lane gate failing on the main tip AND the main sweep or deterministic re-run
   agreeing); target = newest main tip whose landing verify was green in `runs.db`; one
   `recover_main` CAS move per episode; loop guard (red again inside the cooldown → leave for
   the sitting); `project_root` clean; the SHA may be found by a read-only sub-agent; 100% audit
   until n ≥ 20.
7. **Adoption route** (Leo, Q2/Q5): classes already granted to the interactive session move to
   the executor as `live` after a shadow period that only measures (1b, 2's `update_task` half,
   3; items 1, 3, 4, 5, 11, 12; the 2630 classes; items 6 carve-out and 7 via the "applies a named
   ruling" row). New delegations (class 5, class 1a's false-positive arm, detector-close
   confirmation, dead-letter replay, quarantine re-probe, recon self-clears, B3, red main) start
   `shadow` and flip `live` per class at n ≥ 10 and ≥ 95% agreement, by Leo at the η gate; task
   3346 is a dependency of η for the new classes only. The skeleton's reversibility condition is
   waived for the 1b/3 done-flip (`REVERSIBLE_ACTIONS` gains `done_flip` so the codec accepts the
   stamp; the comparable rate is the trailing `close_only`). Amended 2026-11-09 (Leo, sitting
   item 522, esc-8475-2): item 11 does not move. Class 1a's TRUE arm (`completion_claim_true`)
   closes the gate record `close_only`, cancelling the router-spawned task stays
   interactive-only, and reversibility is not waived for `abandon`.
8. **Audit** (Leo, Q6): population = executor closes (DecisionRecords whose `closing_evidence`
   names a `policy_class`); draw = random, ten per week, seeded per render; rendered in the return
   brief §6 with an answer slot; `audit N=ok|wrong:note` is a sitting answer; `wrong` sets the
   class to `shadow` (below n = 20 any `wrong`; at n ≥ 20 above 5%) and records
   `audit_outcome` on the DecisionRecord. A class also auto-flips to `shadow` when its closes in
   one rotation exceed `executor_class_storm_cap` (storm escape).
9. **Sitting** (Leo, Q1/Q7): on demand (`/sitting`) and criteria-triggered (I6). The evaluator is
   deterministic and timer-run; presence = a T3 user message within P minutes
   (`~/.t3/userdata/statev2.sqlite` `orchestration_v2_projection_messages`, role `user`); present
   → launch a `/sitting` thread via the T3 lane, else the gnome-terminal `wm` lane; absent →
   nothing. One launch at a time under a `coordinator` lease, per-trigger-set cooldown, daily
   cap. No notification of any kind.
10. **Recon queue** (Leo, Q8): a second executor rotation hosted by dark-factory's supervisor,
    pointed at `data/reconciliation/escalations` and port 8103, with recon rows in the table; the
    `recon-escalation-watcher` skill's self-clearing dispositions become rows; the hand-run loop
    retires.
11. **Host alarm** (Leo, Q10): 8270 files into dark-factory's queue; this PRD's riders are the
    deny role in the table and a "Host" block at the top of the docket.
12. **Q6 detectors** (Leo, Q9): the L1 rotation never promotes a `__recovery_veto_streak__` L1
    whose listed pins include a pending L2 or a member of one; promotes a `__stale_l0_strand_sweep__`
    L1 only when a stranded L0 had a workflow waiting; tier-10 recover-and-route moves to the L1
    skill. 7716 is a dependency. A 14-day clean check is a dated gate, after which the
    `streak_escalation_enabled` arm is re-enabled in df and reify.
13. **Retirement** (phase 4): after η, the `escalation-watcher` skill is replaced by the `sitting`
    skill; `recon-watch/run.sh` is deleted; OPERATIONS.md and ARCHITECTURE.md §6 describe the
    new roles; nothing claims `watcher-<project>` leases.

## 4. Pre-conditions for activating

- Task 7716 (veto-streak cluster-L2 case) for ζ; task 3346 for η's new-class flips only;
  6214 and 8253 (preparer failover and trial scope) are upstream of the docket's usefulness, not
  of the executor, and are not hard dependencies of any leaf here.
- No novel substrate beyond what α–ε build: every symbol in §6 exists on main (G3 verified
  2026-11-04: `authority.py::ROLE_LEVEL_ALLOWLIST`, `::l2_auto_close_class`,
  `shadow_ruling.py::FIRST_TRANCHE_CLASSES`, `::REVERSIBLE_ACTIONS`, `classify.py::_HUMAN_RESOLVERS`,
  `harness.py::_start_watcher_supervisor`, `::_run_watcher_rotation`, `::_WATCHER_ALLOWED_TOOLS`,
  `::_WATCHER_ESCALATION_HEADERS`, `session_registry.py::DecisionRecord`,
  `::close_decision_with_evidence`, `::build_lease_name`, `prepare_sitting.py::_cmd_resolve_answers`,
  `payloads.py::ApplyPayload`, `inventory.py::queue_project_roots`,
  `dashboard/src/dashboard/config.py::_discover_escalation_urls`, `spawn_t3_lane.py::_start_argv`,
  `spawn-claude.sh` lanes `t3|tmux|wm`, `b3_gate.py` verbs `check`/`record-launch`,
  `recover_main.py`, `escalation.submit`, the fused-memory recon `create_server` on 8103, the T3
  messages table).

## 5. Contract (B + H)

### 5.1 Policy table (`escalation/src/escalation/authority.py`)

```python
@dataclass(frozen=True)
class PolicyRow:
    policy_class: str                  # slug, also the shadow-vocabulary slug
    match: PolicyMatch                 # category | agent_role regex | root_cause regex | ruling_citation
    actions: frozenset[str]            # subset of {'close_only','resume','done_flip','update_task','recover_main','b3_apply', ...}
    evidence: tuple[re.Pattern, ...]   # ALL must match `resolution` (presence-only, as 2630)
    reversible: bool                   # False when any act is outside the reversible set: done_flip (waived), recover_main, b3_apply, replay_dead_letters, reprobe_quarantine (amended 2026-11-09)
    file_writing: bool                 # worktree-only rule + 100% audit until n>=20
    audit_full_until: int              # 20 for file_writing and recover_main; 0 otherwise

POLICY_TABLE: tuple[PolicyRow, ...]
EXECUTOR_IDENTITY = 'orchestrator-escalation-executor'   # levels {0,1,2}
COORDINATOR_IDENTITY = 'interactive:coordinator'          # full levels; denied on a fresh claim
EXECUTOR_DENY_ROLES = L2_AUTO_CLOSE_DENY_ROLES | {'orchestrator-watchdog', 'dashboard-watchdog'}

def executor_policy_class(record, action, resolution, state) -> str | None
    # None => level_forbidden. Order: deny categories/roles -> claim present and fresh -> row match
    # -> row.actions contains action -> state[row.policy_class] == 'live' -> every evidence regex present.
def coordinator_denied(record, now) -> bool   # True iff record.policy_class claimed within CLAIM_TTL
```

Invariants: the table is the only enumeration of classes (the executor skill's class list and
`shadow_ruling.FIRST_TRANCHE_CLASSES` are held equal to it by contract tests); state transitions
are `off → shadow → live → shadow` only, written by the sitting (`audit wrong`), the storm escape,
or `reload_config` for `off`; a row with `file_writing=True` is refused when the queue's
`project_root` is dirty (server reads a `clean` flag the executor passes, and the executor must
pass the truth; the audit catches a lie).

### 5.2 Server (`escalation/src/escalation/server.py`)

- `resolve_issue`: for `EXECUTOR_IDENTITY`, `executor_policy_class(...)` must return a class, else
  `level_forbidden`; `resolved_by` is stamped with the identity; `resolution_class` is required.
  For `COORDINATOR_IDENTITY`, `coordinator_denied(...)` → `{'code': 'claimed_by_executor',
  'policy_class': ..., 'claimed_at': ...}`.
- New tool `claim_policy_class(escalation_id, policy_class)`: executor-only, writes
  `policy_class`, `claimed_at`, `claimed_by`; refused if a fresh claim exists or the row's state
  is `off`. Compact drains return the claim fields.
- `stamp_triage` unchanged. `promote_to_l2`: `PROMOTE_ALLOWED` gains the executor identity.

### 5.3 Supervisor (`orchestrator/src/orchestrator/harness.py`)

```python
@dataclass(frozen=True)
class RotationSpec:
    role: str                 # 'escalation-watcher-auto' | 'escalation-executor' | 'escalation-executor-recon'
    skill: str                # skill name loaded by load_skill_system_prompt
    identity: str             # X-Escalation-Identity
    levels: str               # X-Escalation-Levels
    queue_dir: Path; escalation_url: str
    model: str; effort: str; budget_usd: float; max_turns: int
    rotation_hours: float; rotation_items: int
    allowed_tools: list[str]; disallowed_tools: list[str]
    precheck: Callable[[], bool]      # L1: actionable L1 exists; executor: claimable L2 exists
    cost_role: str                    # 'escalation-executor' (does not match '%watcher%')
```

One supervisor loop per spec (`_rotation_supervisor_loop(spec)`), same guards and backoff as
today; `record_event('rotation_started' | 'rotation_exited', role, outcome, cost_usd,
duration_ms)`; the executor's precheck is "a pending L2 with no fresh claim, no fresh triage
ack, no declared pin, outside deny roles/categories, in a row whose state is not `off`". The
inflight record `data/orchestrator/executor-inflight.json` is `{task_id, escalation_id,
policy_class, request_id?, worktree?, started_at, rotation_id}` entries; a rotation reads it
first and either resumes (B3 `handed_off` poll, red-main re-check) or marks the entry orphaned
with a stamp on the record. Config: `executor_enabled`, `executor_model` (opus),
`executor_effort`, `executor_rotation_hours`, `executor_rotation_items`,
`executor_rotation_budget_usd`, `executor_max_turns`, `executor_daily_cost_ceiling_usd` (its own
ceiling; a breach pauses the executor supervisor, never the scheduler),
`executor_class_storm_cap`, `executor_extra_queues` (df: the recon queue dir + 8103).

### 5.4 Sitting tooling (`scripts/sitting/`)

- `apply --ledger L --answer N=OPTION[:note] ...`: resolves numbers to `(queue_dir, esc_id)`,
  joins queue→URL through `_discover_escalation_urls ∘ queue_project_roots` (moved into
  `scripts/sitting/ports.py`), connects with `COORDINATOR_IDENTITY`, **refuses** a record whose
  status is not pending (exit 4, item listed), then stamps markers, `resolve_issue`, `x_ruling`,
  `close-decision`, two memory writes; one JSON row per item with the outcome of each step.
- `audit --ledger L --answer N=ok|wrong[:note]`: writes `audit_outcome` on the DecisionRecord;
  on `wrong` writes `shadow` into the queue's policy-table state for that class.
- `return_brief.py`: §6 draws ten random executor closes from the week (seed = render date),
  numbered into the ledger as `audit N`; renders `no executor close in window` as a distinct
  state; adds a "Host" block (records whose `agent_role` is a host filer) above the projects, an
  "unanswered since" block, a "trigger state" line (last evaluator run, last launch, reason), and
  the roots statement.
- `sitting_trigger.py` (ε): score = Σ over pending L2s of `severity_weight × (1 + blocked_dependents)`;
  classes = {red main, guard/ceiling-paused scheduler, merge-halt owner, verify-host quarantine
  older than X, host alarm (`agent_role` host filer), recon overflow}; fires when any class is
  present, or score ≥ N, or `Docket page: recommended`, or an item is unanswered ≥ D days;
  presence = a T3 user message within P minutes; launch via `spawn-claude.sh` with
  `CLAUDE_SPAWN_BACKEND=t3` when `t3code.service` is active else `wm`; `coordinator` lease;
  state file with cooldown and daily cap; writes its own last-run line for the docket. Timer
  `sitting-trigger.timer` every 5 min. Defaults (tactical, config-tunable): N = 20, D = 7, P = 30,
  X = 2 h, cooldown 2 h per trigger set, cap 4/day.

### 5.5 Boundary-test sketch

| # | scenario | preconditions | postconditions |
|---|---|---|---|
| 1 | executor closes a live-class record | row `landing_provenance` live; record matches; evidence present; claim fresh | status dismissed; `resolved_by=orchestrator-escalation-executor`; `policy_class` set; DecisionRecord with evidence; `resolution_class` set |
| 2 | executor acts on a shadow-class record | row state shadow | `level_forbidden`; record pending; `x_shadow_ruling` stamp accepted by the codec (class in `FIRST_TRANCHE_CLASSES`, action in `REVERSIBLE_ACTIONS`) |
| 3 | executor acts on a deny-role record | `agent_role=orchestrator-watchdog` | `level_forbidden`; no claim possible |
| 4 | coordinator applies to a freshly claimed record | claim 10 min old | `claimed_by_executor` error; `apply` reports the item, exit 4 |
| 5 | coordinator applies to a record the executor closed a minute ago | status dismissed | `apply` refuses before resolving; nothing written; item listed |
| 6 | coordinator applies after the claim expired | claim 7 h old, record pending | ruling applied; markers stamped; `resolved_by=interactive:coordinator`; counted in the human tier |
| 7 | header-less client resolves anything | no headers | unchanged behaviour (full authority) |
| 8 | audit `wrong` on class C | DecisionRecord for C, state live | `audit_outcome=wrong`; state file C=shadow; next executor call on C → `level_forbidden` |
| 9 | storm escape | 31 closes of class C in one rotation, cap 30 | state C=shadow; a `rotation_exited` event names the escape |
| 10 | file-writing class with a dirty project_root | uncommitted file in main checkout | executor skips the class this rotation; L1 note stamped; no write |
| 11 | B3 handed off across a rotation boundary | inflight entry with request_id; rotation exits | next rotation polls `merge_status`, completes, clears the entry |
| 12 | red main, one signal only | gate red, sweep green | no recovery; stamp "awaiting second signal" |
| 13 | red main, two signals | both red; green landing tip known | one `recover_main` CAS; DecisionRecord quotes both; main at target; re-land via queue |
| 14 | red again inside cooldown | recovery 30 min ago | no second move; left pending for the sitting |
| 15 | trigger fires, Leo present | red-main record; T3 user message 5 min ago; T3 up | one `/sitting` thread launched; lease held; cooldown stamped |
| 16 | trigger fires, Leo absent | same, last T3 user message 3 h ago | nothing launched; trigger-state line updated |
| 17 | trigger fires, T3 down | T3 service inactive; present | gnome-terminal `wm` lane launch |
| 18 | second trigger inside cooldown | lease held or cooldown fresh | no launch |
| 19 | veto-streak L1 with a pending-L2 pin | L1 lists esc-X-2 pending | L1 rotation does not promote; release edge closes it |
| 20 | strand sweep, no workflow waiting | all stranded L0s info | handled at L1 (recover-and-route), no L2 |
| 21 | recon self-clear | `recon_stale_run` resolved by a later run | recon executor closes it under its row |
| 22 | rotation events | one executor rotation completes | `rotation_started` and `rotation_exited` rows with role, outcome, cost |
| 23 | no executor close in a week | fresh install | §6 renders "no executor close in window", not an empty sample |

## 6. Decomposition plan

Labels are Greek; ids are assigned at decompose. Sizing per the overlay: large leaves, same-file
serialization by real dependencies. All leaves `task_kind=normal` unless stated.

| label | title | modules | observable signal | prereqs |
|---|---|---|---|---|
| α | Policy table, executor and coordinator identities, class claim, shadow-vocabulary lockstep | `escalation/src/escalation/authority.py`, `server.py`, `models.py`, `shadow_ruling.py`, `classify.py`, `docs/escalation-standing-policy.md`, `tests/scripts/test_shadow_ruling_doc_contract.py`, escalation tests | boundary rows 1–8 as server tests against a fixture queue: executor `resolve_issue` on a live row succeeds with identity stamped, on shadow/deny → `level_forbidden`; `claim_policy_class` round-trips through the compact drain; coordinator on a fresh claim → `claimed_by_executor`; header-less unchanged; `x_shadow_ruling` for `landing_provenance` with `done_flip` parses | — |
| β | Role-parameterised rotation supervisor, executor rotation, rotation events, inflight record, executor config | `orchestrator/src/orchestrator/harness.py`, `config.py`, `defaults.yaml`, `dark-factory-orchestrator.yaml` (executor_extra_queues for recon), orchestrator tests, OPERATIONS.md §8 | with `executor_enabled: true` on a dev config, `runs.db` gains `rotation_started`/`rotation_exited` rows for both roles; the executor's MCP config carries `X-Escalation-Identity: orchestrator-escalation-executor` and levels `0,1,2`; the precheck launches no rotation when every pending L2 is claimed, pinned or denied; row 11 (inflight resume) and row 22 as tests; cost rows carry `role=escalation-executor` and the `%watcher%` ceiling ignores them | α |
| γ | Executor skill with the per-class procedures, red-main class, B3 class, worktree-only rule, inflight protocol, recon rows | `skills/escalation-executor/SKILL.md`, `skills/_shared/*` fragments moved out of `skills/escalation-watcher/SKILL.md`, `tests/skills/test_executor_skill_policy_contract.py` | the contract test holds the skill's class list equal to `POLICY_TABLE`; a dry rotation on a fixture queue (rows 1, 2, 10, 12, 13, 14, 21) produces the expected calls (recorded through the fake server used by the capability-guard tests) | α, β |
| δ1 | Sitting tooling: `apply`, `audit`, ports module, random audit draw, docket blocks (Host, unanswered-since, trigger state, roots) | `scripts/sitting/apply.py` (or verbs in `prepare_sitting.py`), `ports.py`, `payloads.py`, `return_brief.py`, `brief.py`, `fleet_state.py`, `session_registry.py` (audit_outcome), tests | rows 4, 5, 6, 8, 23 as tests against two fixture queues on two ports; `return_brief.py` on a fixture with 12 executor closes renders ten `audit N` items and the Host block; `apply` exit 4 lists a non-pending item | α |
| δ2 | Sitting skill | `skills/sitting/SKILL.md` (extracted from `skills/escalation-watcher/SKILL.md`: seed, docket, always-ask, H1–H6, apply order via `apply`/`audit`, spawn discuss/carry threads sequentially), `skills/spawn/SKILL.md` handoff section | a sitting on a fixture docket applies two answers and one audit through the verbs with zero direct `resolve_issue` calls (transcript check in the skill's own test harness, as `skills/_shared` fragments are tested today); the `escalation-watcher` skill's apply section becomes a pointer | δ1 |
| ε | Criteria-triggered sitting: evaluator, presence probe, lanes, lease, cooldown, timer | `scripts/sitting/sitting_trigger.py`, `scripts/sitting-trigger.{service,timer}`, `skills/spawn/spawn-claude.sh` (wm fallback order when `CLAUDE_SPAWN_BACKEND=auto` and no tty), tests, OPERATIONS.md §12 | rows 15–18 as tests with a fake T3 store and a fake spawn; live: `systemctl --user list-timers` shows `sitting-trigger.timer`; the docket's trigger-state line names the last evaluator run | δ2 |
| ζ | Q6 detector rules in the L1 skill, tier-10 move, recon-queue precheck note | `skills/escalation-watcher-auto/SKILL.md`, `tests/skills/*` | rows 19, 20 as skill tests on fixture L1s (no promote for a pending-L2-pinned veto L1; strand with no waiting workflow handled at L1) | 7716 |
| η1 | Flip gate for the moved classes (decision 7's adopted list): shadow agreement ≥ 95% at n ≥ 10 per class → Leo flips `live` | human gate (`task_kind=deterministic`, `always_escalates=True`, `execution_class=decision`, delayed milestone 7 d after its deps so shadow stamps accrue); the evidence is `shadow_ruling.agreement_report` over every queue, rendered by `return_brief.py::_agreement_lines` | the gate's L2 lists per class n, agreement, and the proposed state; Leo answers per class; δ1's `policy-state` verb writes the state files | α, β, γ, δ1 |
| η2 | Flip gate for the new delegations (decision 7's shadow-start list) | as η1 | as η1, for the new classes | η1, 3346 |
| θ1 | Retire residents: `escalation-watcher` skill becomes a pointer to `sitting` + executor; delete `recon-watch/run.sh` and `scripts/recon-watch.sh`; ARCHITECTURE.md §6, OPERATIONS.md, CLAUDE.md watcher mentions; `lease-reap` of dead `watcher-*` leases | skills, docs (leases reaped by the `session_registry.py lease-reap` CLI; `~/.claude/fleet/` is in the agent write-set) | `git grep -l "escalation-watcher" skills docs OPERATIONS.md ARCHITECTURE.md` lists only the pointer and history; `ls ~/.claude/fleet/leases` has no `watcher-*` file after reap | η1, δ2, ε |
| θ2 | The 14-day Q6 check: no veto-streak or strand-sweep L2 that ζ's rules should have prevented, in df and reify; if clean, `streak_escalation_enabled: true` in df's yaml | `dark-factory-orchestrator.yaml`, PRD §10 record | a dated record in §10 with the count per project and the rule; df's flag flipped only on a clean verdict | ζ, β (same yaml) |
| λ | reify half of θ2: flip reify's `streak_escalation_enabled` when θ2's record says reify is clean | reify `dark-factory-orchestrator.yaml` (a reify task, external dep on θ2) | reify's yaml reads `true` on reify main, or the task closes citing θ2's not-clean verdict | θ2 (cross-project) |
| κ | Deploy gate: install and enable `sitting-trigger.timer` on the host | host only (`~/.config/systemd/user/` is outside the agent write-set; `task_kind=deterministic`, `always_escalates=True`, `execution_class=operational`) | `systemctl --user list-timers` lists `sitting-trigger.timer`; the docket's trigger-state line names a run within 5 min | ε |
| ι | Observation, 7 days after η1 and θ1: producers wrote in production | `task_kind=normal`, delayed milestone 7 d, read-only measurement appended to §10 (the startup-hardening η shape; no predicate script exists to bind a deterministic gate to) | §10 records: `runs.db` executor `rotation_exited` rows with ≥ 1 close in every active project; ≥ 1 DecisionRecord with `policy_class` and zero executor closes lacking one; the return brief rendered ≥ 1 `audit N`; no `watcher-*` lease claimed; the trigger evaluator ran within the last hour; a filed follow-up for every failure | η1, θ1, κ |

**Decompose amendments (2026-11-04, `/prd` decompose session prd-df-3475198).** No design
decision is re-opened; these make the plan executable as filed.

- η split into η1/η2 so that 3346 gates only the new-class flips (decision 7). Classes below
  n = 10 when η1 or η2 fires stay `shadow` and are flipped later at any sitting with the same
  verb once the brief's agreement block shows the bar.
- The `shadow → live` write had no producer in §5 (the listed writers are the audit flip, the
  storm escape and `reload_config`); δ1 owns it as a `policy-state` verb that refuses any
  transition outside `off → shadow → live → shadow`.
- θ split into θ1 (retirement, after η1) and θ2 (the dated Q6 check, after ζ), because they
  hang off different chains; λ carries the reify yaml, which no df worktree can write.
- κ added: ε's live timer signal needs a host install no agent can perform.
- γ, δ2 and ζ: no harness runs an LLM over a skill in tests, so their "dry rotation" and
  "transcript" signals are rewritten as INV-10 guards (`guards-exercise-behaviour`): marker-
  delimited spans in each skill mirrored against their live artifact (γ: `POLICY_TABLE`; δ2:
  δ1's CLI; ζ: the sentinel constants), and γ's example call blocks run against the fixture fake
  server — never a regex over prose. The behavioural proof is η's shadow agreement, θ2's count
  and ι. New skill tests live in `tests/scripts/` (`tests/skills/` is not under any
  `test_command`).
- Same-file serialization: δ2 after γ (`skills/escalation-watcher/SKILL.md`), ε after β
  (OPERATIONS.md), θ1 after ε, θ2 after β (`dark-factory-orchestrator.yaml`).

Companion edits inside the leaves above (no separate tasks): `docs/escalation-standing-policy.md`
gains the table's classes and the waiver (α); the preparer tasks 6214/8253 are referenced, not
duplicated; 8270's rider lines are already in its details.

G7 walk (advisory, author mode): `contracts-machine-checked` — the table is code with tests, the
skill-to-table equality is a test (α, γ); `structured-facts-at-failure` — rotation events (β);
`corroborate-before-acting` — two-signal red, claim-before-act, `apply` re-reads status (γ, δ1);
`storm-escape-required` — `executor_class_storm_cap` auto-shadow and the executor's own ceiling
(α, β); `no-lockstep-duplication` — one table, contract tests (α, γ); `status-matches-liveness` —
inflight entries orphaned by the next rotation (β); `holds-owned-and-bounded` — claims expire
with the 6 h window (α); `one-fact-one-home` — class state in one file per queue, audit outcome
on the DecisionRecord only (α, δ1); `guards-exercise-behaviour` — server tests against a fixture
queue, not prose greps (α); `no-silent-fail-soft` — the evaluator writes its last-run line and
the docket renders staleness (ε); `exceptions-owned-or-ratified` — the reversibility waiver is
recorded with Leo's 2026-11-04 ruling (α); `readers-prove-their-producer` — §6 renders "no
executor close in window" and ι is the production proof (δ1, ι). No waiver needed.

## 7. Cross-PRD relationship

| other PRD | direction | seam mechanism | owner | status |
|---|---|---|---|---|
| `plans/escalation-connection-capability-guard-prd.md` | extends | `authority.py` identities and gates | this PRD (α) | queued |
| `plans/escalation-l2-tiering.md` | consumes | the ladder; the L1 rotation and its skill (ζ edits) | this PRD (ζ) | queued |
| `plans/startup-hardening-prd.md` (8270 δ) | consumes | host-alarm records in dark-factory's queue → deny role + Host block | this PRD (α, δ1); 8270 unchanged | queued |
| `plans/fleet-cockpit-prd.md` | extends | `DecisionRecord.audit_outcome`; `coordinator` lease name | this PRD (δ1) | queued |
| `plans/escalation-lifecycle-dashboard-prd.md` | consumes | `resolution_class` required on executor closes | this PRD (γ) | wired |
| preparer follow-up tasks 6214, 8253 | consumes | nightly preparation quality; trial scope rendered as the executor shadow baseline | 8253 (amend its trial line) | upstream, not blocking |
| `plans/task-escalation-state-graph-prd.md` (veto streak) | consumes | 7716 | 7716 | dependency of ζ |
| `plans/plan-deviation-recording-prd.md` | none | class 4 already off L2 | — | wired |

## 8. Out of scope

- Reducing the ~60 real weekly decisions (R5 gate inventory, hand carries).
- The sibling-commit coverage predicate for classes 1b/3 (the executor decides what the
  interactive session decides today; a stronger predicate is a later task).
- A host-level escalation sink (ruled out, Q10).
- Executor rotations for projects whose orchestrator is not supervised by this harness.
- Changing the nightly preparer beyond 6214/8253.

## 9. Open questions (tactical)

1. **Trigger defaults** (N = 20, D = 7 d, P = 30 min, X = 2 h, cooldown 2 h, cap 4/day) and the
   severity weights. Suggested: config in `scripts/sitting-trigger.service` env; decide in ε.
2. **Executor model split.** One rotation at opus, or sonnet with an opus sub-agent for 1b/3 and
   red main. Suggested: opus for the rotation; decide in β after the first cost week.
3. **Where the recon executor's rows live** (same table with a `queue_kind=recon` match, or a
   second table). Suggested: same table; decide in α.
4. **`apply` as a verb in `prepare_sitting.py` or a sibling script.** Suggested: sibling
   `apply.py` sharing `payloads.py`; decide in δ1.
5. **Whether `claim_policy_class` is a new tool or a `stamp_triage` extension.** Suggested: new
   tool, since `stamp_triage` replaces the note and must stay ungated; decide in α.

## 10. Observation record

θ2 (the 14-day Q6 check) and ι (the production observation) append dated records here.
