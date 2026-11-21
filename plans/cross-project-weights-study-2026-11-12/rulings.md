# rulings.md — what is already filed, ruled or rejected on dividing CPU, memory, verify capacity and the agent allowance between projects

Seat: RULINGS AND OWNERS. Read-only. Written 2026-11-12. Code read: Dark Factory `main` at `7848173a7d` (the running DF service started 2026-11-10T23:48Z and may lag main). Task stores read through `file:...?mode=ro`:
`/home/leo/src/dark-factory/.taskmaster/tasks/tasks.db` (9467 tasks), `/home/leo/src/reify/.taskmaster/tasks/tasks.db` (9954), `/home/leo/src/solar-challenge/.taskmaster/tasks/tasks.db` (1431), `/home/leo/src/solar-challenge-platform/.taskmaster/tasks/tasks.db` (1923). All four paths exist and are non-empty. Task `id` is only unique within a store.

Quotes are verbatim. A quote is followed by its source. Anything labelled MY READING is mine, not a recorded position.

## 0. Headline for the lead (nine points)

1. Nothing in any store, plan or memory assigns the VALUE of a cross-project CPU weight, a per-project share of the agent allowance, or a cross-project urgency order. The nearest owner (task 3394) builds the slice topology only and says so. The follow-up that would set values is named in two places as "3394 (or its follow-up)" and was never filed (verified: no task in the DF store mentions "cross-project split" or "43/43/14" except 3394 and 5205 themselves).
2. The rejected option is narrow and precisely worded: a HOST-GLOBAL CROSS-PROJECT VERIFY SEMAPHORE (a shared slots directory that holds verifies). Static kernel weights hold nothing, and sit outside its text. The grounds still apply to weights as a test: a project that is deprioritised must not starve "almost entirely" (section rulings R1).
3. `plans/load-throttle-harmonisation-prd.md` has no section 3.3. The text the brief points at is section 2 item 3 (consequence (a)), section 3 D5 and D12, and section 9 (last bullet). Quoted below.
4. Three live constraints on any weight design that are NOT in the brief's list: (i) Leo's 2026-09-07 directive that merge verify is never throttled (quoted inside tasks 5203 and 5139); (ii) task 3394's amendment 4(b): `df.slice` is a sibling of `app.slice`, so ALL factory work shares ONE weight against desktop and interactive scopes, a "policy point this task must STATE"; (iii) `C-no-load-derived-count`: counts and `-n` never derived from live load.
5. The allowance has NO per-project, per-urgency or reserve mechanism today: `shared/src/shared/usage_gate.py::UsageGate.before_invoke(scope, *, park, max_wait_secs)` takes no project, priority or urgency; accounts are walked in list order (`try_lease` loop `for acct in self._accounts`). Each of the nine orchestrators runs its own gate over the same file (X7: "each gate independently observes the same physical cap").
6. Leo's recorded stance on the allowance (2026-11-02): individual seat caps are negligible unless the whole pool is all-capped; the question that matters is how much weekly allowance goes unused. Measured by seat X7: pool all-capped 0.83 h in 30 days; unused allowance 14.3% / 17.9% / 6.9% of the allowance in cycles W41/W42/W43, all on the two seats at the back of the fill-first order. Leo's 2026-11-12 ruling forbids proposing "cap persistence, a cap authority" again without new evidence (quoted below). A pacing design that needs shared cap state touches that wording.
7. The three resume tasks 8791/8792/8793 do exactly one thing: call `resume_scheduler` over the target project's escalation MCP (:8100, :8106, :8107) once their dependencies are done. They cannot fire until ζ 8528 is done, and 8528 sits behind ε 8527, which sits behind δ 8526 (in-progress) and 7720 (BLOCKED). 8791 also needs 7721 and 7824. 8792 and 8793 become eligible in the same instant; 8791 may come later. Reify's 2,336 non-spine tasks are held by a SEPARATE dependency on reify task 9739 (Ω), so resuming reify's scheduler does not release them.
8. Existing project-level levers that already divide something between projects: per-project `max_concurrent_tasks` (df 18/24, reify 16, scp 14; ruled 2026-11-02, revisited 2026-11-12), per-project `verify_admission_order`, per-project daily cost ceiling (`orch_daily_cost_ceiling_usd`, task 8262 pending, ruled effectively off), per-project scheduler pause. None is a weight.
9. No document gives a ranking of the projects by importance. The closest statements are quoted in section "relative importance".

## 1. owners (table)

Status is read live from the stores at the end of this seat's run (see section 7 for the timestamp). "Owns" means owns part of the topic; most rows own something adjacent, and the "leaves out" column says what.

| id | project | status | what it is | owns / leaves out |
|---|---|---|---|---|
| 3394 | dark-factory | section 7 | Per-project cgroup slice topology: one parent slice per project holding daemon, agents and verify scopes | OWNS the topology and the slice-naming contract for both repos (`df-<project_id>.slice`). LEAVES OUT choosing weights, MemoryHigh values and quotas, and interactive sessions. |
| 5220 | dark-factory | section 7 | Own-arm calibration gate: recommend `own_cpu_some_avg10` (or leave off) from 14+ days of `df-*.slice` samples | Waits on 3394. Calibrates a dispatch threshold, not a cross-project weight. |
| 5205 | dark-factory | section 7 | Per-role cgroup `CPUWeight` at the verify-scope spawn: merge 100 / task 33 / background 10 | OWNS the per-ROLE weights, identical in every project. After 3394 its arithmetic "becomes intra-slice". |
| 5206 / 5208 | dark-factory | section 7 | Enable `verify_use_cgroup_scope` for dark_factory; after-gate stall re-measure | Per-role, per-project; not cross-project. |
| 7112 | dark-factory | section 7 | Investigate why 5205's weights did not restore reify's local merge-gate latency (hypothesis: scopes compete as siblings of ~8 service cgroups at weight 100) | Evidence bearing on whether weights are a strong lever at all. Pending, low. |
| 3590 / 5216 / 3592 | dark-factory | section 7 | psi_admission v2 (runqueue-ratio arm and own-slice arm into the dispatch gate), PsiSample v2, load sampler with per-slice pressure | The measurement and per-project dispatch-hold half of the load-throttle PRD. Own-slice arm ships off. |
| 5139 | dark-factory | section 7 | `_admission_slot` role gate: merge/mainprobe bypass all concurrency capping (facet 1 only) | OWNS whether mainprobe is capped. Constraint recorded: merge is never throttled behind task admission. Per-project (slots dir is per project). |
| 5628 | dark-factory | section 7 | Ordered task-verify admission gate with recorded wait (arrival order, role classes) | Per project. 5139's prerequisite. |
| 8261 | dark-factory | section 7 | `verify_admission_order: priority_aged` | OWNS priority ordering of the task-verify slot WITHIN one project (K per project). |
| 5052 | dark-factory | section 7 | Cross-project verify-host arbitration (RunnerBusy + host-global admission) | The task the rejection was enforced by. |
| 5529 (prereq 5089, which also waits on 9132) | dark-factory | section 7 | Per-test timeout markers kill xdist workers under CPU starvation | OWNS the consequence of starvation on tests; does not divide CPU. |
| 8791 / 8792 / 8793 | dark-factory | section 7 | Resume reify / solar-challenge / solar-challenge-platform schedulers (deterministic deploy) | OWN the end of the 2026-11-08 whole-project pause. |
| 8275 | dark-factory | section 7 | Resume reify's scheduler after the alpha-spine PRD (earlier pause) | Precedent, same script. |
| 1767 / 1782 | dark-factory | section 7 | Agent CPU de-prioritisation (nice/CPUWeight on agent spawn) and wiring of reify's cgroup governance into the agent-launch path | Agent placement exists only as reify's `cpu_governance`, which has never run (reify 5908). |
| 8827 | reify | section 7 | Experiment (needs Leo's go): `verify_cgroup_cpu_weight_merge` 100 to 300 on reify | Leo, 2026-10-13: "not now". A per-project per-role weight experiment. |
| 5908 | reify | section 7 | `cpu_governance` has never run; config fix staged behind slice topology | Depends on reify 9739. Agent subtree weights task 100 / merge 300. |
| 4632 | reify | section 7 | `cpu-governed-exec.sh` / `lib_cgroup.sh`: cgroup-v2 `cpu.weight` placement by role, invariants C-G1..C-G3 | Source of "proportional, NEVER a quota" and "all governed scopes share ONE parent slice". |
| 7335 / 7336 | reify | section 7 | `@own-slice` sentinel for reify's verify gates and jobserver balancer; redeploy jobserver under `df-reify.slice` | Per-project own-pressure; depends on 3394 and 9739. |
| 7533 | reify | section 7 | Milestone: reassess reify `max_concurrent_tasks` 48 to 24 once 5205 live 10 days | Superseded by the 2026-11-02 concurrency cut (reify 16). |
| 605 / 905 | solar-challenge / solar-challenge-platform | section 7 | `verify_use_cgroup_scope: true` (uniform merge/task/background weights fleet-wide) | The solar pair already run verifies in weighted scopes. |
| 1271 / 1702 | solar-challenge / solar-challenge-platform | section 7 | Adopt remote task-leg verify once `dark_factory:6673` has passed | Laptop sharing between projects. |
| 6673 | dark-factory | section 7 | Remote task verify ε (operator gate): DF rollout of remote task legs, 3-day parity window | Gate for the laptop's task-leg share. |
| 6106 / 8384 / 5415 | dark-factory | section 7 | Laptop verify-scope memory limits as a config field (`MemoryHigh`/`MemoryMax`/`MemorySwapMax`); measurement of `memory.peak` | OWN per-scope memory limits on the laptop. No per-project memory share anywhere. |
| 9109 | dark-factory | section 7 | Retire reify's laptop scope drop-in once reify runs DF code with 6106 | Blocked on 8791 (the reify pause). |
| 8480 | dark-factory | section 7 | Reserve a concurrency slot (or bypass the cap) for main-red heals | A per-project slot reserve, not cross-project. |
| 8262 / 1323 | dark-factory | section 7 | Daily cost ceilings: package default `orch_daily_cost_ceiling_usd` 200 to 10000 (Leo 2026-11-02, effectively off); watcher 150 | The only per-project dollar budget knob. Not a division of the allowance. |
| 4741 / 4945 | dark-factory | section 7 | Evals share the fleet pool; max-a interactive-only | RULING on pool membership (quoted below). |
| 2859 / 2860 / 5407 | dark-factory | section 7 | Model-scoped (per account and model) caps; stale `claude-fable-5` literal in `scoped_cap_models` | Cap scoping by model, not by project. |
| 6628 | dark-factory | section 7 | INV-11 on "one account pool from `config/usage-accounts.yaml`" in `UsageCapConfig._load_accounts_file` | Pool-loading invariant. |
| 6214 / 8252 / 8253 | dark-factory | section 7 | Preparer failover delivered (6214 done); cap persistence rejected (8252 cancelled 2026-11-12); 8253 reduced to return-brief trial windows | See ruling R8. |
| 9315 / 6998 / 6729 / 6994 | dark-factory | section 7 | Trickle DEFERRED when no pool account has headroom; cap-episode observability; non-pool backends leasing a Claude account | Observability around the shared pool. |
| 2293 / 2302 | dark-factory | section 7 | Fleet Cockpit scoring: `cockpit/src/cockpit/priority.py::score` with per-project `project_weights` | A cross-project urgency score, for the HUMAN decision queue only. `project_weights: {}` in `priorities.default.yaml`; `~/.claude/fleet/priorities.yaml` does not exist. |

Not owners (searched, found nothing): no task in any of the four stores owns cross-project weighting of CPU, memory or the agent allowance, a forecast of the cap, a reserve for an urgent task, an arbitration between projects' schedulers, or a MemoryHigh/memory share per project. Solar-challenge and solar-challenge-platform stores contain only the adoption tasks above (605/905/606/906/1271/1702/1727); every other hit for "slice" in those stores is unrelated code ("vertical slice", "billing slice") and every "pacing" hit is simulator pacing.

How the search was done (so it can be re-run): `/tmp/team-weights-1112/rulings/search.py`, `search2.py`, `search3.py`. Terms: cross-project, per-project, CPUWeight, cpu.weight, slice, cgroup, usage cap, usage allowance, account pool, pacing, reserve, quota, budget, cap forecast, fair share, arbitration, portfolio, MemoryHigh, memory pressure. Population: all statuses including cancelled and deferred. Title hits were read in full (DF 99, reify 55, sc 8, scp 12 rows; files `titles_*.txt`); description-level phrase hits were filtered on narrower phrases (`desc_*.txt`) because bare "slice", "reserve", "budget" and "cap" match thousands of unrelated tasks (911 DF tasks match the broad term list). Details fields were not searched for the narrow phrases, so a task whose only mention is in `details` could be missed.

Documents (plans/docs/OPERATIONS.md), relevant section only:

| document | section that matters |
|---|---|
| `plans/load-throttle-harmonisation-prd.md` | section 2 item 3 (consequences (a), (b)), section 3 D2/D5/D6/D7/D9/D12, section 9 last bullets, section 10 Q3, Q5 |
| `plans/cpu-load-robust-verify-prd.md` | section 6 "Out of scope (RED-TIER...)" |
| `plans/remote-task-leg-verify-prd.md` | header "Rulings this PRD executes", decision 16 (hosts symmetric), "Nothing here is a cross-project gate", section on out of scope |
| `plans/verify-oversubscription-control-prd.md` | section 3 hard constraints (C-merge-priority, C-no-load-derived-count), section 5 |
| `plans/merge-lane-throughput-prd.md` | "RULED 2026-09-03 — task C is cancelled" |
| `plans/verify-merge-queue-policy-study-2026-09-18/design-options.md` | section 4.1 "Hard rejections" R1 to R6 |
| `plans/occt-throttle-layer-2-3.md` | "Background — why Layer 2/3" (the reify incident) |
| `plans/orchestrator-suite-slowdown-2026-09-19/G-priority-audit.md` | headline, "Cross-project alignment (question 5b)" |
| `docs/prds/dispatch-admission-load-cap.md` | DA-D3 (per-orchestrator floor), out of scope list, section 8 "Recovery jitter" |
| `plans/portfolio-review-2026-11-01/portfolio-review.md` | section 1.1, 1.2, 2.1 (Leo 11-02 note), 3.1, 3.2 |
| `plans/portfolio-review-2026-11-01/seats/X6-cpu-share.md` | since-restart CPU share table; X7-unused-capacity.md (allowance) |
| `plans/agent-capacity-study-2026-09-09/C1-critique.md` | section 7 table ("Raising max_concurrent_tasks, a second verify slot, cross-project semaphore, verify-ahead: Do not") |
| `config/usage-accounts.yaml` (comment lines only read) | header comments: account A interactive-only, evals share the pool, max-b/max-h stay listed |
| `docs/task-authoring.md` section 5 | deterministic tasks, `before_done` |

## 2. rulings

Each entry: verbatim quote, source, date, what it forbids, and MY READING of inside or outside for the three candidate mechanisms: (a) static kernel CPU weight per project, (b) per-project share of the agent allowance, (c) one cross-project urgency order used by each project's scheduler.

### R1. Host-global cross-project verify semaphore: tried and rejected

Quote 1 (`plans/cpu-load-robust-verify-prd.md`, section 6 "Out of scope (RED-TIER — human decision on `host-verify-cpu-oversubscription-df`)", PRD authored 2026-07-18):
> Host-global cross-project verify admission semaphore (shared slots dir) — **TRIED-AND-REJECTED, not merely deferred**: under a single fair global semaphore, Reify's very long (30min+ cargo) verifies starved dark_factory almost entirely. The correct shape is per-project admission (current design) + per-project off-hot-path serialized non-merge-blocking lanes, NOT a shared global gate. Do not revisit this as an option.

Quote 2 (`plans/remote-task-leg-verify-prd.md`, header, Leo 2026-10-09, verify-rate study MEMO section 4): "never build host-wide verify admission across projects (RED-TIER, `plans/cpu-load-robust-verify-prd.md` §6)". Same file, section 1: "Nothing here is a cross-project gate." Same file, out-of-scope list: "Any host-wide or cross-project verify admission (RED-TIER), any load-derived slot or `-n` count, and an in-process second task slot."

Quote 3 (`plans/merge-lane-throughput-prd.md`, "RULED 2026-09-03 — task C is cancelled"): "both rule out a **host-global cross-project verify admission semaphore** — the mechanism task C proposes — as *'TRIED-AND-REJECTED, not merely deferred … Do not revisit this as an option'*, after reify's 30-min+ cargo verifies starved dark_factory almost entirely." Task 5052 (cancelled 2026-09-03) enforced it; same section: "Building C's arbitration would additionally have **created** the host-global semaphore the RED-TIER ruling above forbids, inside a system whose per-project design is what that ruling calls correct."

Quote 4 (`plans/verify-merge-queue-policy-study-2026-09-18/design-options.md`, section 4.1, a study seat's table restating the above, not Leo's text): "R1 | **A host-global cross-project verify semaphore is TRIED-AND-REJECTED, 'do not revisit this as an option.'** ... *Every A-option must stay **per-project**.*" and "R6 | Fail-open is correct for task-verify admission and **wrong** for cross-project arbitration (INV-11 `no-silent-fail-soft`)."

Also a matching statement in `plans/integration-test-lane-prd.md` (section 11, line 156): "*Host-global cross-project verify semaphore* — **tried-and-rejected** (reify's 30-min+ verifies starved dark-factory almost entirely)."

Date and status: the cockpit decision `host-verify-cpu-oversubscription-df` is named as the home of the RED-TIER levers; I did not read it (not in the files read). The wording was authored 2026-07-18, re-affirmed by Leo 2026-09-03 (cancel of 5052, "Leo's question ... has no good answer, because we can") and 2026-10-09.

What it forbids: a shared slot or queue ACROSS projects that holds a verify until another project's turn. Also, by R6, any cross-project arbitration that fails open without an event.

Inside or outside, MY READING:
- (a) static kernel weight: OUTSIDE the text. A weight holds nothing, admits nothing, queues nothing, is work-conserving (reify task 4632 contract C-G1: "proportional, NEVER a quota: set cpu.weight by role; NEVER set cpu.max ... A lone scope under the slice gets 100% of the box"). The rejection's GROUNDS still apply as a test: a bimodal service-time mix (30-min cargo versus short pytest) starved the short jobs under a size-blind fair discipline. A weight does not remove that risk, it only changes the CPU share while both run; the measured weakness of 5205's weights (task 7112: reify merge-gate p50 52.5 / 77.7 / 97.8 min across the 48-era, 24-era and post-5205) is the nearest evidence on how strong the lever is.
- (b) per-project share of allowance: OUTSIDE (the rule is about verify admission). But the same GROUNDS (size-blind fairness over a bimodal mix) are relevant if the share is enforced by a shared first-come queue across projects.
- (c) one cross-project urgency order: OUTSIDE if each project's scheduler orders its own work using a shared key vocabulary and no state is shared that can hold another project. INSIDE (forbidden shape) if the order is enforced by a single host-wide queue or slot that holds a project's verify or merge behind another project's. Design-options R1 states the working rule: every option stays per-project.

### R2. Slot counts and `-n` are never derived from live load (the reify incident)

Quote (`plans/verify-oversubscription-control-prd.md`, section 3 "Hard constraints (locked this session)", dated 2026-07-09):
> **C-no-load-derived-count:** N and `-n` are fixed config, never derived from instantaneous loadavg (reify positive-feedback-collapse lesson).

Quote (`plans/remote-task-leg-verify-prd.md`, header, Leo 2026-10-09): "a static leg-identity class is allowed, a load-derived count is not (`plans/verify-oversubscription-control-prd.md` §3 C-no-load-derived-count; the 09-11 synthesis ruling R3); a BINARY pressure hold in the runner's health probe is the one allowed load-shaped lever".

The incident (`plans/occt-throttle-layer-2-3.md`, "Background — why Layer 2/3"): "A `/deb` session on 2026-05-28 traced reify task 4000's L2 escalation (`esc-4000-39`) to a positive-feedback collapse in `reify/scripts/cargo-test-occt-gated.sh`. The wrapper's load-based formula `N = clamp(nproc - load_1m_int, 1, MAX_CAP)` collapsed N to 1 under sibling-worktree load because it measured run-queue length, which is itself fed by the work being throttled. Every concurrent wrapper then contended for slot-1 alone even when slots 2..MAX_CAP were idle." Research note (`plans/oversubscription-reify-to-df-research-2026-07-09.md` line 92): "Load-average-derived worker counts → positive-feedback collapse to N=1 (the throttle measures run-queue length, which is fed by the work it throttles). ... Use a fixed cap / token pool, not `nproc − load`."

Also R4 in `design-options.md`: "`verify_admission_task_slots` stays 1 (3 slots produced an 80 % timeout wave 2026-09-03, reverted 09-08; '2 slots is not a middle ground either')". Live state is 2 (memory 2026-11-12 load monitor stopped: "don't raise above 2 just because task legs are waiting"). Leo, 2026-11-02 (portfolio review section 3.2, also memory): "the host is at 100% CPU and verify is the CPU-bound phase, so more task-verify slots buy nothing until CPU capacity grows; modelled and experimented, taken as given."

What it forbids: slot or `-n` counts, and by extension any weight or share, computed from a live load signal. Allowed: static config, static identity classes, and a binary hold.

Inside or outside, MY READING:
- (a) static weight: INSIDE what is allowed (static config). A weight that is auto-adjusted from live PSI or runqueue is OUTSIDE (forbidden) for the same reason: the signal is fed by the work it throttles. The existing PSI dispatch hold (DA-D3, binary, with a per-orchestrator in-flight floor so it cannot wedge) is the accepted exception shape.
- (b) allowance share: a share that moves with live cap state would be the same shape. A static share, or an ordering fixed by config, is allowed. A forecast of the cap from `account_events` is derived from measured cap history rather than instantaneous load, and I found no ruling against it; but the loop risk (pacing changes consumption which changes the forecast) is the same positive-feedback class and needs the argument written down.
- (c) urgency order: a static key (task tier, PRD priority, project weight) is inside; an order that reads live queue depth or load is outside (design-options R3: "The policy key must be static config or task/leg identity — never live load, never live queue depth").

### R3. Task 3394: interactive sessions are privileged deliberately; weights are excluded; two PRD-assigned consequences

Quote (task 3394 `description`, "OPERATOR RULING (Leo, 2026-08-03) — INTERACTIVE WORK IS PRIVILEGED DELIBERATELY."):
> Measured: 30 of 57 `claude` processes (53%) run in interactive terminal cgroups (konsole / gnome-terminal / tmux scopes), NOT under any `orchestrator-*.service`. A per-project slice therefore covers only the orchestrator-dispatched half of agent work, and human-launched agents keep competing at the session's default weight — i.e. they are effectively favoured under contention.
>
> That is INTENDED, not a coverage bug. Do NOT extend this task to sweep interactive processes into project slices, and do NOT file a follow-up to "fix" it. If a future change would incidentally capture terminal-launched work, raise it as a decision rather than implementing it.

Quote (3394, "DELIBERATELY EXCLUDED FROM THIS TASK"): "choosing the actual weights, MemoryHigh values or quotas. This task delivers the TOPOLOGY — a place to put a policy — and nothing more. Picking values needs demand data (per-cgroup runnable-thread counts plus aggregate runqueue wait over 24h) that does not exist yet, and a premature value is worse than none. Note for whoever sets values later: measured `memory.current` is dominated by reclaimable page cache (reify 25.3 GiB current vs 3.6 GiB anon), so any memory limit must be sized against `anon`, never `current`."

Quote (3394 amendment 2026-09-08, point 4):
> (a) Once slices exist, SLICE weights rather than scope weights set the CROSS-PROJECT split, so 5205's 43/43/14 arithmetic becomes intra-slice. Re-deriving the cross-project values is this task's or an explicit follow-up's — say which in the commit rather than leaving it implied.
> (b) `df.slice` is a sibling of `app.slice` under `user@1000.service` (systemd derives the parent from the name), so ALL factory work shares one weight against desktop and interactive scopes. That is consistent with the 2026-08-03 operator ruling recorded above — interactive work is privileged deliberately — but it is a POLICY POINT this task must STATE rather than inherit silently.

Quote (3394 amendment point 1): "SET CPUWeight=100 ON THE SLICE AT CREATION. ... the PRESENCE of the property is what vivifies cpu.weight for the subtree ... a scope created under df-prd-probe.slice had NO cpu.weight file at all until `systemctl --user set-property ... CPUWeight=` was applied."

Quote (3394 DESC, measured 2026-07-30): "A policy applied to `orchestrator-<project>.service` controls UNDER HALF its target: 53% of reify's CPU was measured inside the sibling `df-verify-reify-*.scope`, 47% inside the service ... Capping the service alone just pushes work to the uncapped escape path."

Quote (3394, point 5 of the amendment): items 2 (agent placement) and 4 (reaping `reify-govtest*.slice`) "may be split into their own tasks".

Same position stated in `plans/load-throttle-harmonisation-prd.md`:
- section 2 item 3: "Two consequences 3394 owns: (a) slice weights, not scope weights, then set the **cross-project** split, so 5205's 43/43/14 arithmetic becomes intra-slice and 3394 (or its follow-up) re-derives the cross-project values; (b) `df.slice` is a sibling of `app.slice` under `user@1000.service` ..., so all factory work shares one weight against desktop and interactive scopes — consistent with Leo's 2026-08-03 ruling on 3394 that interactive work is privileged deliberately, but a policy point 3394 must state rather than inherit."
- section 3 D5: "Interactive agents stay outside project slices (Leo, 2026-08-03, on 3394)."
- section 9 Out of scope: "Slice `CPUWeight` *values* beyond the vivifying default (3394's exclusion) and the re-derivation of 5205's cross-project split after slices exist (3394 or its follow-up)." and "Interactive-terminal agents (3394's ruling); 3394's items 2 and 4."
- section 3 D6: "Re-key reify's throttles; do not retire them yet." D12: "3394 is the prerequisite for D2/ρ2 only; 5205/5206/5208 unchanged."
- section 10 Q3: "Balancer reservoir retirement once 3394 + 5205 exist and ε2 reports own-slice pressure rarely above 50 while tokens are held."

The brief cited "sections 3.3 and 9": there is no section 3.3 in that file (headings are 0 to 10, section 3 is "Resolved design decisions" D1 to D12). The cross-project ownership text is the section 2 item 3 quote above and section 9.

What it forbids: sweeping interactive processes into project slices; filing a follow-up to "fix" that; setting weights inside 3394; silently inheriting the df.slice-versus-app.slice weight.

Inside or outside, MY READING:
- (a) static weight per project: INSIDE 3394's ownership in principle (the slice weights are exactly what "set the cross-project split") but the value-setting is the unfiled "follow-up". Constraint carried: the demand data 3394 says is missing (per-cgroup runnable threads and runqueue wait over 24 h) is what `sampler`/3592 now records; ε2 (5220) waits on 3394.
- Interactive stays outside: any weight design must leave T3/Konsole sessions at the default and must state the factory-versus-interactive weight (df.slice versus app.slice) as a policy point. Task 5205 reasoned the same way: "a 300 merge scope would also put interactive shells (weight-100 siblings under app.slice) at a quarter share under contention."
- (b) and (c): not touched by R3.

### R4. Merge verify is never throttled (Leo, 2026-09-07)

Quote (task 5203 description): "Leo's 2026-09-07 directive is that the merge verify (the serial landing bottleneck) is never throttled".
Quote (task 5139 details, "CONSTRAINT FOR FACET 1 (recorded by the L2 watcher 2026-09-11 ...)"): "whatever Facet 1 decides about bounding concurrency, it must not put the merge role behind task-role admission. If anything is capped, mainprobe is the candidate; merge is not, unless Leo rules otherwise."
Also `plans/verify-oversubscription-control-prd.md` section 3: "C-merge-priority / never-block: merge-role pytests never acquire the counting semaphore and get the highest verify nice; task can never starve merge (anti-livelock)." And `plans/remote-task-leg-verify-prd.md` decision 16 (Leo, 2026-10-31): "merge → one `HostAllocator` slot, any host, ungated. Task → one `AdmissionGate` turn, any host. Neither role reads the other's state on any host."

What it forbids: capping or queueing the merge role; merge reading task state.

MY READING:
- (a) a project-level weight applies to merge scopes inside the project's slice. Whether a low project weight counts as "throttling" merge verify is a question for Leo; the text of the directive is about admission and `-n` pins. I found no ruling either way. Flag as a decision, not an answer.
- (c) an urgency order at the verify slot concerns the TASK slot only (8261's `NOT IN SCOPE`: "the merge-over-task arm ... is owned by task 5139"); a cross-project order that reached merge would run into this directive and R1.

### R5. Task 5139 and 8261: priority at verify admission is within-project, and what they leave out

Task 5139 (pending, high; deps: 5628 done; depended on by 9129): facet 1 only, whether and how merge/mainprobe verify concurrency should be bounded. Facet 2 was split out as 5424 (2026-09-11, Leo: "Split 5139 and set the urgent bit high merge priority"). Amendment 2026-09-18: the probe's role should be threaded from the caller. It builds nothing cross-project: the slots dir is per project (task 2501; 5052's correction 1: "OrchestratorConfig._default_verify_admission_slots_dir now derives /tmp/df-verify-slots-<uid>-<sha256(project_root)[:12]>, deliberately PER-PROJECT so co-tenant projects stop contending").

Task 8261 (pending, high; no dependencies): quote of title: "verify_admission_order: add 'priority_aged' — rank admission by the task's effective priority tier ... first-come within a tier, with an aging credit of K hours per tier so low tiers cannot starve; reverses FIFO PRD decision 7 ('no priority key') per Leo 2026-11-02". Builds: new `AdmissionOrder` literal `priority_aged`, ordering key `(class, arrival + tier_rank x K, seq)`, tier read through a lookup callable at each pick (not stamped at arrival), new hot-reloadable knob `verify_admission_priority_age_secs` (K), tier recorded on the `verify_admission_plan` event, wait-alarm decision, PRD delta note. "Suggested K per project: about 2 h on dark_factory, about 4 h on reify and solar-challenge-platform (size K to the project's p90 wait)." Leaves out (its text): "NOT IN SCOPE: the merge-over-task arm (merge and mainprobe verify capping and classing) is owned by task 5139 ...; the re-entry-after-green credit (Q9 O6) ...; flipping solar-challenge-platform and solar-challenge from the code default 'unordered' to 'arrival' is a zero-code reload_config yaml edit outside this task." Evidence it carries (Q9 open-loop replay, since 2026-10-02, relative only): "priority-weighted mean wait down 17-30% vs arrival (df -26%, reify -17%, scp -25%)", "Strict priority without aging (K to infinity) is explicitly NOT wanted."

Ruling behind it (Leo 2026-11-02, memory `project_portfolio_review_rulings_2026_11_02`): "Aged effective-priority ordering filed as a task (seat W-2), reversing FIFO PRD decision 7." and "`verify_admission_order: arrival` committed + reloaded on scp, sc, scoping-intake, stream-touch, pump-web-ui (package default is `unordered`, config.py:4303)".

MY READING of inside or outside:
- (c): 8261 is exactly the within-project half of an urgency order and is a hard prerequisite of any design that wants urgency to bind at the verify slot. It cannot order across projects by construction (one gate per project slots dir). Do not re-propose it; cite it.
- (a): independent of it.

### R6. Task 5529 and 5089: why hosts cannot simply run full, and the cost of starvation

Quote (5529 DESC, rescoped 2026-09-29 by Leo via esc-5671-3): "Task-role verify runs at `nice -n 15 ionice -c2 -n7` ... The host runqueue bursts to 15-37 runnable per core for 1-3 min every 30-60 min. A test needing about 0.1 s of CPU then stretches past its marker, and pytest-timeout's thread method `os._exit()`s the whole worker." "Field asymmetry, 2026-09-25..28: 0 crashes in 76 merge-role legs vs 14 in about 138 task-role legs."
5529 builds: signal-mode on every explicit per-test marker via one conftest hook, a guard test, and a measurement on an unloaded host; "DEPENDS ON 5089" (5089 depends on 6864 done and 9132 pending). "NOT IN SCOPE: widening any marker ...; the task-tier nice policy; evidence persistence for the global thread kill (5527); the merge-phase VerifyInfraError escape (6039)."
MY READING: a low cross-project weight is a CPU-starvation lever, so it feeds exactly this failure for the low-weight project's TASK-role legs until 5529 and 5089 land. Treat 5529/5089 as a prerequisite or a stated risk of any weight design.

### R7. Per-project concurrency, ordering and pause levers already ruled (Leo, 2026-11-02, applied; changed 2026-11-12)

Quote (memory `project_portfolio_review_rulings_2026_11_02`):
> **Concurrency cut** (answer 1, seat Q3): df 24→18, reify 24→16 (+spare_warm_lanes 8→16 so the pool stays 32), scp add 14; restart-tier, rides the next redeploy; tripwire waiter-present share <0.90 or landings below the W44 rate ⇒ revert; re-raise when 7977 remote legs reach a project.
> **Reify PAUSED** (item 13, "Do it"): `halt_scheduler` on http://127.0.0.1:8100/mcp ... It is to resume only through the capstone of the reify-alpha spine PRD ... The spine must be short.
> **R1** agreed with stream-touch EXEMPT from parking (cheap, and Leo wants it ready if his touchstream breaks).

2026-11-12 (memory `project_three_gates_scheduling_study_2026_11_12`): "`max_concurrent_tasks` 18 → 24: branch `task/slots-24-revert` ... merge request `mr-477890c0` queued 15:15Z. Check it landed" and "the 18-slot cap has met its own revert tripwire" and "No owner exists for: merge queue ordered by urgency, PRD identity/priority/completion bias, agent-allowance pacing, cross-project weights, recording eligibility history." The same entry says of 3394: "(per-project CPU groups; the weights themselves have no owner)".

Rejected options on concurrency: `plans/verify-merge-queue-policy-study-2026-09-18/design-options.md` R5: "Raising `max_concurrent_tasks` is rejected." (source `plans/agent-capacity-study-2026-09-09/C1-critique.md` section 7) — that note predates the 11-02 cut and the 11-12 revert request, so treat as historical.

What it constrains: per-project concurrency is already the de facto share of AGENT slots between projects (df 18-24, reify 16, scp 14 on a pool of "32"). A weight design must say how it composes with it. Pause is the existing whole-project lever the weights are meant to replace.
MY READING: (b) can be built partly on this lever (a per-project concurrency count is a static cap, allowed by R2) without a new mechanism; it divides agent concurrency, not allowance consumed (a lane's token burn varies by role and model).

### R8. The agent allowance: what has been ruled

Quote (Leo, 2026-10-06, memory `project_max_b_h_stay_listed_ruling_2026_10_06`, verbatim): "I do not want the max-b nor max-h accounts disabled. The code fix sounds good, though. They're switched off on the anthropic side by my deliberate choice, and I want the simplest possible way to re-enable them when I choose to. The trigger for that will be when we get bottlenecked by the token limits on the accounts that are currently active."
Consequence recorded: "Never file or perform work that retires, comments out or reorders max-b/max-h as 'dead'." and "An earlier signal than an actual stall is 'only one active account still serving'." File header in `config/usage-accounts.yaml`: "Do not comment them out."

Quote (Leo, 2026-08-30, task 4945 description, verbatim): "max-h has been fully saved. max-a is reserved for interactive use. Evals have no unique account and should be fully robust to cap events (they share with the fleet)." Header in `config/usage-accounts.yaml`: "Account A is reserved for INTERACTIVE use only (not here) — Leo's own sessions exhaust its weekly cap most weeks, so it cannot double as a reserve."

Quote (Leo, 2026-11-02, memory `feedback_self_improvement_is_high_value_no_slots_at_full_cpu_caps_negligible`, item 3): "Individual seat caps are negligible unless the pool is all-capped. Failover absorbs a capped seat; only a simultaneous cap of all seven Max seats counts as lost capacity. The question to answer instead is how much weekly allowance goes unused." In the portfolio review section 3.2: "if the pool is never all-capped, treat individual caps as negligible".

Quote (Leo, 2026-11-12, memory `project_cap_ledger_8253_rejected_2026_11_12`, verbatim form in the memory): "Leo ruled 2026-11-12: the persisted cap file for one-shot runs (task 8253's 'gate cap ledger') is not built. Task 8253 is now only the return-brief trial windows (low, depends on 8563); task 8252 is cancelled (failover delivered by 6214, persistence rejected)." Applying rule: "Do not propose cap persistence, a cap authority, or shortening the 5 s failover pause again without new evidence that a run was lost while a seat was open." And: "Leo's ranking: the return brief is the less valuable nightly job and may be replaced, so do not harden its preparer. The confusion census matters more; it may run late or as one larger run, as long as mid-term coverage holds." Grounds: "after task 6214 ... a one-shot run loses 18–49 s at $0 rediscovering capped seats; the file could never save a night. Nights are lost only when all seven seats are capped (3 of 31 at the preparer's start; 8.4% of hours, rising from 3.0%)."

Quote (Leo, 2026-11-12, memory `feedback_deterministic_vs_llm_single_partition`, design principle): "for each decision area we should have a single (therefore uniform by construction) clean partition between deterministic work done by deterministic code and judgement applied to ambiguous or infeasibly complex situations by LLMs (with a to-human escape hatch)."

Quote (Leo, 2026-11-02 / 11-04, memory `project_model_routing_followup_2026_11_02`, rulings): "stance accepted (strongest model at PRD/architect; cheap-first only behind strong gates)", "(4) fable rung only with evidence opus fails". Round 2: "option 1 (sonnet-first architect) WITHDRAWN; strongest model at PRD/architect, cheap-first only behind strong fast gates". Leo's answer in round 2: "intelligence pays most upstream (process > PRD > architect)".

Facts about the mechanism (code read, main `7848173a7d`): `shared/src/shared/usage_gate.py::UsageGate.before_invoke(self, scope=None, *, park=True, max_wait_secs=None)` and `::try_lease(*, scope=None, exclude=None)` select by walking `self._accounts` in list order, skipping capped, probing and auth-failed accounts. There is no project, priority or urgency argument. `scope` is a model scope (fable), not a project scope. Pool membership per project: DF config `usage_cap.accounts_file: "${USAGE_ACCOUNTS_FILE:/home/leo/src/dark-factory/config/usage-accounts.yaml}"`; reify config the same key with default `/home/leo/src/reify/config/usage-accounts.yaml` (a symlink to DF's per COMMON.md); solar-challenge and solar-challenge-platform configs carry no `usage_cap` block, so they take the package default `accounts_file: "${DARK_FACTORY_ROOT:/home/leo/src/dark-factory}/config/usage-accounts.yaml"` (`orchestrator/src/orchestrator/defaults.yaml`), which resolves to DF's file unless `DARK_FACTORY_ROOT` is set differently in their unit environments (not checked).

Evidence on the allowance (portfolio review seat X7, window 2026-09-20 to 2026-11-02): "The pool was all-capped for **0.83 h in 30 days**: 10 episodes, the longest 11 min ... Unused allowance, in ledger-visible list-USD, was **14.3% / 17.9% / 6.9%** of the pool allowance for the seat cycles starting in ISO W41 / W42 / W43 ... All of it sits on the two seats at the back of the fill-first order (max-c, max-h). It is structural: demand did not reach those seats. It is not cap-shaped." Spend share (portfolio review 3.1, 2026-10-02..11-01, list USD): dark-factory 30%, solar pair 38%, reify 16%, scoping-intake 11%, stream-touch 5%.

What it forbids / constrains:
- do not disable, comment out or reorder max-b/max-h as dead; their return is the operator's lever when allowance binds;
- no private reserve account for evals (shared pool), max-a stays interactive-only: a pacing design cannot assume a reserve seat;
- no persisted cap file / cap authority without new evidence of a run lost while a seat was open;
- design partition: deterministic code for deterministic decisions, LLM for the ambiguous residue; any pacing rule must name who decides each row;
- pacing by downshifting model tier collides with the 2026-11-04 routing stance (strongest at PRD/architect).

MY READING:
- (b) per-project share of the allowance: no ruling forbids it by name; what exists is (1) the objective Leo set (maximise use of allowance; seat caps only matter when all seven are capped), (2) the prohibition on a persisted cap authority, (3) the "no reserve" structure. A static per-project ordering of accounts (each project's accounts file listing the same accounts in a different order, since the gate walks list order) needs no new mechanism, but `config/usage-accounts.yaml` is shared by symlink and by default; changing order for one project means a per-project accounts file, which `6628` (INV-11 on "one account pool") and the file header ("single source of truth") argue against. A dynamic share needs cross-process state: this is the point where the 2026-11-12 wording ("cap authority") applies and new evidence would be needed. The new evidence Leo asked for is "a run was lost while a seat was open"; the 8.4% all-capped hours figure in that memory is evidence about the all-capped state, which Leo had said does count.
- (a) and (c): untouched by R8, except that (c) as an allowance order needs the same cross-process state.
- The model-tier constraint means "pace by cheaper model" is OUTSIDE what Leo has accepted; "pace by order or deferral of low-urgency work when a cap is forecast" is not addressed by any ruling.

### R9. Memory

Findings: no task or ruling divides memory between projects. Existing memory controls are per verify scope on the LAPTOP only (tasks 6106, 8384, 5415, 9109): `verify_cgroup_memory` config field with integer-GiB `MemoryHigh` / `MemoryMax` / `MemorySwapMax` (OPERATIONS.md line ~1447). Leo ruling 2026-10-09 (`plans/verify-rate-improvement-2026-10-09/MEMO.md` section 4): "Laptop scopes switched 2026-10-09 from MemoryMax-kill to MemoryHigh throttle + swap (DF 24G/40G/32G, reify 32G/48G/32G ...) ... by ruling." Task 3394 has a note that memory limits must be sized against `anon`, never `current`, and excludes MemoryHigh values. Dispatch gating on memory is the DA gate: DA-D1 ranks memory above io and CPU for reporting and "memory `full` hard trip" (`docs/prds/dispatch-admission-load-cap.md`); it is per orchestrator and reads the host file, so each project holds independently.

MY READING: (a) does nothing for memory (cpu.weight only); any memory split is new work, and its only precedent is the laptop `verify_cgroup_memory` field. No ruling forbids it.

### R10. Resume of the paused projects — see section 3

### Relative importance of the projects: what is stated

No document gives a numeric or ordinal ranking of the projects. The nearest statements:
- Portfolio review section 1.1 (2026-11-01): "Income of GBP 42k/yr is the priority." "The mission programme: reify plus dark-factory as the existence proof; reify into a 1–3-human alpha." "The Solar Challenge board is deadlocked and Leo's posture there is 'step back and stay in touch'." Strategy names three outcomes: "income, the reify alpha, a client-visible demo".
- Leo, 2026-11-02 (portfolio review section 2.1): "improvements to the factory's throughput, efficiency, reliability, required supervision or output quality (or work that enables those) have a compounding benefit on future improvements and a multiplicative benefit across every hosted project." i.e. the factory's self-improvement is HIGH value. (Memory form: "Self-improvement is high value.")
- Section 1.2 table gives each project's "role in the portfolio": dark-factory "the factory; its own largest workload", reify "mission existence proof; pre-release alpha", solar-challenge-platform "client ops platform ... client deadlocked", solar-challenge "client simulator library ... client deadlocked", stream-touch "onboarded late Oct; hardware gates", scoping-intake "Stage-1 instrument of the block offer; prototype", pump-web-ui "idle with backlog", know-live "parked venture", autopilot-video "code-complete, awaiting first footage".
- Leo's 2026-11-08 act itself (pause reify, solar-challenge and solar-challenge-platform to give Dark Factory the CPU, "to free host CPU for its verifies") is a recorded preference for Dark Factory's merge-lane work over the three, for the duration of the push.
- Seat X6 (2026-11-02, since-restart window 2026-11-01 07:30 to 11-02 09:30 UTC): CPU share of hosted work dark_factory 47.0%, reify 37.3%, solar_challenge 7.2%, solar_challenge_platform 4.1%, scoping_intake 4.0%; against spend share 30 / 16 / 38 (solar pair) / 11. "CPU follows verify volume ... not LLM tokens."
Anything beyond this (a weight ratio between projects) is a decision for Leo and is not recorded.

## 3. resume-mechanics

Code read: `/home/leo/src/dark-factory/scripts/resume_project_scheduler.py` (main `7848173a7d`), `docs/task-authoring.md` section 5, `orchestrator/src/orchestrator/scheduler.py` (deterministic handling).

What the script does when a resume task completes:
- `resume_project_scheduler.py --url <project escalation MCP> --reason <text>` opens a streamable-HTTP MCP session (initialize, initialized, `tools/call`), and calls the target project's `resume_scheduler` tool with `{"reason": reason}`.
- Exit 0 only if the reply has `resumed` true, or `resumed` false with `was_paused` false (already running). Any JSON-RPC error, `isError`, unparseable payload, a payload with an `error` key, or a transport failure exits 1; the runner then escalates rather than stamping done. Timeout per request 30 s; the task's `before_done.timeout_secs` is 120.
- Effect in the target: its runs.db gets a `scheduler_resumed` event with the reason; its next `scheduler_tick` can dispatch. It does NOT remove anything else that holds work.

The three tasks (all `task_kind: deterministic`, `before_done.kind: deploy`, `files: []`, priority high, pending since 2026-11-08T16:16Z):
- 8791 reify: `--url http://127.0.0.1:8100/mcp`. Dependencies: 8528, 8531, 7721, 7824. (7721 and 7824 were added by Leo on 2026-11-10: "reify runs the spec warm-lane pool and chain_cap 6, so reify must not resume at volume on head chaining until the lane-leak fix (7721) and the off-loop chain build (7824) have landed.") Depended on by 9109.
- 8792 solar-challenge: `--url http://127.0.0.1:8106/mcp`. Dependencies: 8528, 8531.
- 8793 solar-challenge-platform: `--url http://127.0.0.1:8107/mcp`. Dependencies: 8528, 8531.
- All three carry the reason text "merge-lane policy capstone ..." "tasks 8528 and 8531 landed; the 2026-11-08 operator pause is lifted, resume dispatch".

Dependency state at read time: 8531 done. 8528 pending, depends on 8527 (pending), which depends on 8526 (in-progress) and 7720 (BLOCKED). 7721 pending (depends on 8527). 7824 pending (depends on 7720 blocked and 7721). 8525 done. So the earliest possible firing for 8792 and 8793 is when 8528 becomes done; for 8791 it is the later of 8528, 7721 and 7824. 8792 and 8793 have identical dependency sets, so they become eligible in the same tick and both resume together (the eligibility is per scheduler tick and the DF scheduler runs one task at a time through `DeterministicRunner`; I did not establish whether the two run in the same tick or successive ticks).

Gating conditions beyond dependencies:
- The DF scheduler must itself be running (it is: its last scheduler event is a resume on 2026-10-26).
- Deterministic tasks are exempt from the PSI dispatch hold: `scheduler.py` line ~7510: "`if ctx.psi_hold and not self.is_deterministic(task):` ... deterministic candidates are exempt (DA-D4)". Module locks: the tasks declare no files, so I infer no lock conflict (not read in code).
- Born-at-L2 escalation only if the script fails (`always_escalates` is not set; field-combo preset "present / false: run action; escalate only on failure; else done").
- No time condition, no load condition, no check that the target project's work is safe to start. Nothing in the task checks the host's CPU or memory when it fires.

What the target does when released:
- Live pause state (read from each project's `data/orchestrator/runs.db` `events`, read-only): reify latest `scheduler_paused` 2026-11-08T16:04:41Z, no `scheduler_resumed` since; solar-challenge `scheduler_paused` 2026-11-08T16:04:45Z (latest of the four scheduler event types); solar-challenge-platform `scheduler_paused` 2026-11-08T16:04:48Z (latest). The reason text on reify: "Operator pause (Leo, 2026-11-08): dark-factory merge-lane policy work (PRD plans/merge-lane-policy-prd.md, tasks 8524-8528, 8531) is being landed as fast as possible; this project's scheduler is paused to free host CPU for its verifies. In-flight tasks finish and merge normally. A deterministic capstone in the dark-factory store resumes this scheduler when 8528 and 8531 are done."
- Pause did not halt merge queues or the offline lane: memory `project_merge_lane_policy_prd_2026_11_05`: "Schedulers halted (operator) on reify 8100 / sc 8106 / scp 8107 ...; merge queues NOT halted (transient + kills in-flight verify)." and "reify's offline deep lane (`warm-lanes/worktrees/_offline-deep`, timer-driven) is outside the scheduler."
- Reify has a second, independent hold: reify task 9739 (Ω, "alpha spine complete (human gate; the parking holder every non-spine pending task depends on)") has 2,336 dependents in the reify store (the store has 2,529 pending tasks; 194 of them do not depend on 9739). Resuming reify's scheduler releases only the spine tasks (9727-9740) and those 194, until 9739 is done. (Pending-count figures read from the reify store at write time.)
- Simultaneous resume of solar-challenge and solar-challenge-platform, and later reify, is the "N siblings release in lockstep" case the dispatch-admission PRD recorded as an open tactical question (`docs/prds/dispatch-admission-load-cap.md` section 8): "Recovery jitter: whether to add a small per-orchestrator jitter to the re-evaluation so N siblings do not release in lockstep on PSI recovery ... add jitter only if a measured thundering-herd appears." Held-back DF tasks are a separate matter (139 `worker.py` tasks plus ~47 transitively are parked behind 8528 by dependency edges and "self-release when 8528 is done", memory `project_chain_parking_behind_8528_2026_11_10`). So at 8528's completion DF's parked tasks and two projects' schedulers release together.

Replacement: if weights replace the pause, 8791/8792/8793 are the objects to retire or re-point; they are deterministic deploys whose `before_done` is this script, and 9109 (a laptop drop-in gate) depends on 8791. Tasks 8791 and 8275 are the pattern; the tasks live in the DF store because "a paused scheduler dispatches nothing, deterministic tasks included".

## 4. MY READING of the three mechanisms, as a whole (labelled reasoning, not a ruling)

(a) Static kernel CPU weight per project (slice `CPUWeight`): outside R1 (holds nothing, work-conserving), inside R2's allowed class (static config), explicitly the structure owned by 3394 and explicitly left without an owner for the values. Open points that rulings leave to Leo, not answered anywhere: how the factory-wide `df.slice` weight against interactive and desktop (3394 amendment 4(b)); whether a low weight on a project's merge scopes is "throttling" merge verify (R4); who sets the values and on what data (3394 says "demand data ... that does not exist yet" and the sampler now records some). Known limits on what it can do: nothing for memory; strongly limited when work runs inside service cgroups and when the slice covers only the dispatched half of agent work (3394 measured 53% / 47% scope/service for reify; 7112's hypothesis); starves tests under severe CPU shortage unless 5529/5089 land.

(b) Per-project share of the agent allowance: not ruled either way. It is the least-owned of the three: no task, no ruling, no code input (the gate has no project argument). It sits closest to the 2026-11-12 "cap authority" wording only if the share needs persisted or cross-process cap state. It is compatible with Leo's stated objective only if framed as using more of the allowance (the unused 7 to 18% on the back-of-order seats), because he said individual caps do not matter unless all seven are capped. It must respect max-b/max-h staying listed, no reserve seat, the evals sharing the pool, and the 11-04 model stance. A per-project concurrency count (R7) is already the existing coarse lever.

(c) One cross-project urgency order used by each project's scheduler: no ruling blocks a shared key VOCABULARY applied by each scheduler to its own queue. The within-project half is already owned (8261 priority_aged at the verify slot; `set_task_priority_override` and the priority-inheritance walk at dispatch; `cockpit/src/cockpit/priority.py::score` for the human queue). What would be OUTSIDE (R1) is any enforcement that holds one project's work behind another project's through shared state, and any use of live load or queue depth as the order's key (R2). Two practical facts bear on it: priority tiers are per-project vocabularies (reify's "high" is not DF's "high"; the 2026-11-12 memory notes "priority lives only at the scheduler" and verify admission and merge queue are first-come), and cross-project dependencies read statuses only (`get_external_statuses`), not priorities.

Combination: (a) is the only one of the three that can arbitrate physical CPU between projects without a new queue; (b) and (c) need a source of cross-project information that does not exist, and R1/R6 say that information path must not hold work and must not fail silently.

## 5. unknowns

1. Whether the cockpit decision `host-verify-cpu-oversubscription-df` carries more text than the PRD quotes (not read).
2. Whether "merge verify is never throttled" (2026-09-07) is meant to cover a project-level CPU weight (R4). No recorded ruling; needs Leo.
3. Whether the follow-up for the slice weight VALUES was ever filed under another phrasing. I searched title and description for "cross-project split", "43/43/14" and "re-derive ... weights"; only 3394 and 5205 matched. A task whose wording I did not guess could exist.
4. Details fields were not searched for the narrow phrases (only title and description), so an owner mentioned only in details could be missed.
5. Whether solar-challenge and solar-challenge-platform really resolve the accounts file to DF's: depends on `DARK_FACTORY_ROOT` in their service environments (not read). COMMON.md says their checkouts have no `config/usage-accounts.yaml`; their yaml has no `usage_cap` block, so they take the package default.
6. Whether 8792 and 8793 would be dispatched in the same tick (the scheduler code path for picking two deterministic candidates was not read to that depth).
7. Whether a persisted or cross-process cap state is "a cap authority" in Leo's 2026-11-12 sense when its purpose is pacing rather than failover (the ruling's grounds concern one-shot failover; the wording is broader).
8. The live CPUWeight of every unit and scope (not read: out of this seat's brief).
9. The statuses of tasks 7720's cause of blockage and 8526's progress were read as store states only.

## 7. live status of every task named (re-read from the stores at 2026-11-12T15:54Z)

| project | id | status | priority | store updated_at |
|---|---|---|---|---|
| dark-factory | 3394 | pending | high | 2026-09-08T18:10 |
| dark-factory | 5139 | pending | high | 2026-11-11T12:55 |
| dark-factory | 5529 | pending | high | 2026-09-29T06:16 |
| dark-factory | 5089 | pending | high | 2026-10-25T03:24 |
| dark-factory | 8261 | pending | high | 2026-11-02T17:59 |
| dark-factory | 8791 | pending | high | 2026-11-10T22:12 |
| dark-factory | 8792 | pending | high | 2026-11-08T16:16 |
| dark-factory | 8793 | pending | high | 2026-11-08T16:16 |
| dark-factory | 5205 | done | medium | 2026-10-04T17:58 |
| dark-factory | 5206 | done | medium | 2026-10-09T11:24 |
| dark-factory | 5208 | done | medium | 2026-10-12T12:51 |
| dark-factory | 5220 | pending | medium | 2026-09-08T18:14 |
| dark-factory | 7112 | pending | low | 2026-10-18T08:32 |
| dark-factory | 3590 | done | high | 2026-10-20T00:54 |
| dark-factory | 5216 | done | high | 2026-09-13T14:16 |
| dark-factory | 3592 | done | high | 2026-09-19T16:16 |
| dark-factory | 5628 | done | high | 2026-10-13T14:59 |
| dark-factory | 5052 | cancelled | high | 2026-09-03T20:27 |
| dark-factory | 8275 | done | high | 2026-11-02T20:59 |
| dark-factory | 1767 | done | high | 2026-06-15T14:00 |
| dark-factory | 1782 | done | high | 2026-06-17T13:27 |
| dark-factory | 6673 | blocked | critical | 2026-11-11T11:58 |
| dark-factory | 6106 | done | medium | 2026-11-10T19:19 |
| dark-factory | 8384 | done | medium | 2026-11-10T19:23 |
| dark-factory | 5415 | done | high | 2026-10-10T04:37 |
| dark-factory | 9109 | pending | low | 2026-11-11T12:51 |
| dark-factory | 8480 | pending | high | 2026-11-08T17:36 |
| dark-factory | 8262 | pending | high | 2026-11-02T22:13 |
| dark-factory | 1323 | done | high | 2026-05-15T12:08 |
| dark-factory | 4741 | done | high | 2026-08-30T13:15 |
| dark-factory | 4945 | done | high | 2026-09-07T09:32 |
| dark-factory | 2859 | done | medium | 2026-07-21T04:25 |
| dark-factory | 2860 | done | low | 2026-07-21T03:22 |
| dark-factory | 5407 | pending | low | 2026-10-04T21:02 |
| dark-factory | 6628 | pending | medium | 2026-10-10T02:58 |
| dark-factory | 6214 | done | high | 2026-11-10T02:00 |
| dark-factory | 8252 | cancelled | high | 2026-11-12T14:49 |
| dark-factory | 8253 | pending | low | 2026-11-12T14:50 |
| dark-factory | 9315 | pending | low | 2026-11-11T13:15 |
| dark-factory | 6998 | pending | low | 2026-10-15T08:39 |
| dark-factory | 6729 | pending | low | 2026-10-10T21:54 |
| dark-factory | 6994 | pending | low | 2026-10-15T06:52 |
| dark-factory | 2293 | done | high | 2026-07-09T23:21 |
| dark-factory | 2302 | done | medium | 2026-07-14T10:33 |
| dark-factory | 8528 | pending | high | 2026-11-10T20:55 |
| dark-factory | 8531 | done | medium | 2026-11-09T08:55 |
| dark-factory | 8527 | pending | high | 2026-11-10T22:11 |
| dark-factory | 8526 | in-progress | high | 2026-11-12T15:33 |
| dark-factory | 7720 | blocked | high | 2026-11-12T15:12 |
| dark-factory | 7721 | pending | high | 2026-11-10T22:11 |
| dark-factory | 7824 | pending | high | 2026-11-10T22:12 |
| dark-factory | 8525 | done | high | 2026-11-12T06:52 |
| dark-factory | 9132 | pending | medium | 2026-11-10T23:05 |
| reify | 8827 | pending | low | 2026-10-13T15:12 |
| reify | 5908 | pending | medium | 2026-10-13T15:09 |
| reify | 4632 | done | high | 2026-06-17T11:09 |
| reify | 7335 | pending | high | 2026-10-15T13:56 |
| reify | 7336 | pending | medium | 2026-09-08T18:14 |
| reify | 7533 | done | medium | 2026-10-18T08:20 |
| reify | 9739 | pending | high | 2026-11-02T20:43 |
| solar-challenge | 605 | done | medium | 2026-10-09T18:08 |
| solar-challenge | 1271 | pending | medium | 2026-11-03T07:28 |
| solar-challenge-platform | 905 | done | medium | 2026-10-09T19:31 |
| solar-challenge-platform | 1702 | pending | medium | 2026-11-03T07:27 |
