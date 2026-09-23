# D6 milestone +1d — is the claude-fable-5-1 admission actually dispatching?

**Verdict: yes.** The admission is live, resolving, running, completing and
contained. None of the four escalation triggers this check was armed with
fired. One finding outside those triggers is recorded in § Residual risk and
filed separately; it does not block closing this milestone.

| | |
|---|---|
| Apply commit | `526e0eba99bfc66904427426a5b1beb54afeb50d` |
| Apply message | config: admit claude-fable-5-1 for the merger and the steward's second attempt on dark-factory (ruling D6, 2026-09-10; applied 2026-09-12) |
| Apply time | 2026-09-12T07:43:16+01:00 = **2026-09-12T06:43:16Z** — the anchor every window below is measured from |
| Measured at | 2026-09-13T13:59Z (~31.3 h after apply) |
| Store | `/home/leo/src/dark-factory/data/orchestrator/runs.db`, 181 870 592 bytes, mtime 2026-09-13 14:50:45 +0100 |
| Command | `python scripts/audit_model_admission.py --model claude-fable-5-1 --expect-roles merger,steward --since 2026-09-12T06:43:16+00:00 --ceiling 150 --window 24h --format markdown` |

The body in § Measurements is **generated** by that command and pasted
verbatim — no number in it was transcribed by hand. `scripts/audit_model_admission.py`
is strictly read-only (every connection is a `mode=ro` SQLite URI), so it is
safe against the live store while the fleet is merging, and task 5441 can
re-run it at +14d with a wider `--window` and diff against this table.

One caveat on reading the table, because it is the cell most likely to be
misread: **`over flat ceiling` is not a failure column.** `timeouts.merger: 600`
is enforced flatly only until a transcript proves liveness; from turn 1 the
bound becomes `max(timeouts.working_idle_secs, 600) = 1800 s` as an *idle*
bound, itself capped by `invocation_timeout: 7200 s`. A merger that keeps
producing turns runs past 600 s by design. `timed out` is the producer's own
kill verdict and is the column that means what "wall clock" sounds like.

## Measurements

### 1. Routing decisions for `claude-fable-5-1` since 2026-09-12T06:43:16+00:00

| timestamp | task | role | source_layer | rule_id | tier |
|---|---|---|---|---|---|
| 2026-09-12T08:52:56.079465+00:00 | 4377 | merger | config | - | 1 |
| 2026-09-13T02:15:16.452029+00:00 | 4377 | steward | policy_rule | steward-retry-fable | 1 |
| 2026-09-13T07:53:52.827788+00:00 | 4211 | steward | policy_rule | steward-retry-fable | 1 |

Rejections naming a model, any role, since 2026-09-12T06:43:16+00:00:

_none_

Unparseable payloads skipped: 0

### 2. Invocations on `claude-fable-5-1` and how they ended, since 2026-09-12T06:43:16+00:00

| task | project | role | account | cost $ | turns | ok | timed out | model @end | duration ms | over flat ceiling | merge |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4377 | dark_factory | merger | max-b | 6.08 | 45 | True | False | claude-fable-5-1 | 1149252 | True | done (d411f107c3676e4ce8d322634bf1cce8bdf1110e) |
| 4377 | dark_factory | steward | max-b | 2.14 | 13 | True | False | - | 527497 | None | - |
| 4211 | dark_factory | steward | max-b | 1.72 | 9 | True | False | - | 207867 | None | - |

### 3. Dispatches at retry tier >= 1 since 2026-09-12T06:43:16+00:00

| timestamp | task | role | tier | rule_id |
|---|---|---|---|---|
| 2026-09-12T08:52:56.079465+00:00 | 4377 | merger | 1 | - |
| 2026-09-13T02:15:16.452029+00:00 | 4377 | steward | 1 | steward-retry-fable |
| 2026-09-13T07:53:52.827788+00:00 | 4211 | steward | 1 | steward-retry-fable |

### 4. Scoped cap posture for `claude-fable-5-1` since 2026-09-12T06:43:16+00:00

_none_

Account-level (unscoped) cap hits in the same period: 9

Service restarts since then (only an orchestrator restart reloads a restart-tier leaf):

| timestamp | service | reason |
|---|---|---|
| 2026-09-12T07:32:20.828171+00:00 | fused-memory | post_merge_fused_memory_code_change |
| 2026-09-12T11:46:28.878952+00:00 | fused-memory | post_merge_fused_memory_code_change |
| 2026-09-12T23:45:05.875131+00:00 | dashboard | post_merge_dashboard_code_change |
| 2026-09-13T12:40:11.505210+00:00 | fused-memory | post_merge_fused_memory_code_change |

### 5. Spend on `claude-fable-5-1` over [2026-09-12T13:56:43.215668+00:00, 2026-09-13T13:56:43.215668+00:00)

| invocations | total $ | ceiling $ | headroom $ | at/over ceiling |
|---|---|---|---|---|
| 2 | 3.86 | 150.00 | 146.14 | False |

### 6. Roles observed on `claude-fable-5-1` since 2026-09-12T06:43:16+00:00

Admitted roles: merger, steward

| role | invocations | total $ | admitted |
|---|---|---|---|
| merger | 1 | 6.08 | True |
| steward | 2 | 3.86 | True |

Roles outside the admitted set: none


## Verdict per check

**1 — Is the resolver choosing it?** Yes, and unanimously. Every one of the
three dispatches above resolved to `claude-fable-5-1`. The merger came from the
`config` layer (`models.merger`, an absolute model string, `rule_id: null`);
both steward dispatches came from the `policy_rule` layer. There were **no
rejections**, and that is a stronger negative than it looks: across the whole
store all 13 588 `routing_decision` events carry `rejected: []`, on every model
and every role, so nothing was refused for allowlist, ceiling or capacity
reasons anywhere. Merger coverage is total — exactly one merger routing
decision occurred in the window and it selected Fable; no merger dispatch
resolved to anything else after the apply.

**2 — Is it running and completing?** Yes. Three invocations, all
`success: true`, all `timed_out: false`, none capped. The merger run drove 45
turns and resolved its merge to sha `d411f107c3676e4ce8d322634bf1cce8bdf1110e`.
No `merge_finalized` state of `blocked` appears as a final outcome for any task
a Fable merger touched, so the "was a blocked merge actually a post-merge
verification failure rather than the merger failing?" question does not arise
on this data — there is no blocked terminal state to attribute. (The audit
resolves this class of ambiguity structurally regardless: it reads the *last*
`merge_finalized` per task, not the first, so an intermediate blocked
generation that is later retried to done cannot be misreported as a merger
failure.)

On duration: the one Fable merger ran 1 149 252 ms (19.2 min) and so shows
`over flat ceiling: True`. This is normal for the role, not a Fable trait — the
three most recent `opus` merger runs measured 641 797 / 881 547 / 1 103 341 ms
(10.7 / 14.7 / 18.4 min). Fable sits in the same band, and `timed_out` is
`false`. **No Fable merger died at a wall clock.**

**3 — Did the steward retry rule match?** Yes, twice, and the rule is named in
the evidence rather than inferred: `2026-09-13T02:15:16Z` (task 4377) and
`2026-09-13T07:53:52Z` (task 4211), both `source_layer: policy_rule`,
`rule_id: steward-retry-fable`, `routing_tier: 1`. The "absence is not failure"
branch this check was written to handle turned out not to be needed.

**4 — Scoped cap posture.** Zero cap hits scoped to `claude-fable-5-1`, so the
scope has not been exercised. **But the restart-tier leaf is not yet live**, and
the audit's restart table is what shows it: all four service restarts since the
apply were `fused-memory` (×3) and `dashboard` (×1). The last *orchestrator*
restart was **2026-09-04T17:32:40Z — eight days before the apply**.
`usage_cap.scoped_cap_models` is restart-tier (red), so it has not been loaded;
the nine account-level cap hits counted in the window are unrelated to Fable but
demonstrate that the unscoped path is the one currently in service. See
§ Residual risk. (This does not undermine checks 1–3: `models.*` and `routing.*`
are green-tier and hot-reloadable, which is precisely why dispatch works while
the red-tier leaf does not.)

**5 — Cost against the ceiling.** Nowhere near it. $3.86 over the trailing 24 h
and $9.93 across all three invocations since the apply — that is the *unrounded*
sum, $9.932856; § 2 rounds each row to the cent, so adding its three rendered
cells (6.08 + 2.14 + 1.72) gives $9.94. Either way, against a $150.00/day
`per_model_daily_ceiling_usd` that is 6.6 % of one day's ceiling consumed in
31 hours. No ceiling rejection fired, and none could have.

**6 — Role containment held.** Observed roles are exactly `{merger, steward}`;
`unexpected_roles` is empty. This is the property `dark-factory-orchestrator.yaml`'s
**"DELIBERATE DEVIATION from P4-06's 'ladder top = fable'"** comment — the one
guarding `routing.ladder` — deliberately engineered for: the retry ladder was
left unchanged and absolute model strings used instead, so a `+1` retry-tier-up
cannot route an implementer, debugger or architect to Fable.
Note both steward dispatches ran at `routing_tier: 1`, which is the rule
working as specified (second attempt), not a ladder leak.

## The two things the table cannot say on its own

**No lineage alias.** `invocations.model` reads the literal string
`claude-fable-5-1` for all three rows — not a task-4826 lineage alias, not a
resolved-to-underlying-model rewrite. For the merger row this is corroborated
by a second, independent witness: its `invocation_end` payload carries its own
`"model": "claude-fable-5-1"`. That corroboration is **unavailable for the two
steward rows**, and the audit prints `-` rather than inventing it — the steward's
`invocation_end` payload is a different shape entirely
(`escalation_id`, `category`, `retry_count`) and carries no `model` key at all.
So: one row doubly witnessed, two rows singly witnessed, zero rows showing an
alias.

**The `opus` merger invocation is a pre-apply straggler, not a check-1 finding.**
Task 4635, `project_id: dark_factory`, $2.3155, completed
**2026-09-12T00:14:16Z — 6 h 29 m before the apply time**. The two other recent
opus merger runs (tasks 4095, 4259) are from 2026-09-10. No merger invocation
after 2026-09-12T06:43:16Z ran on anything but Fable.

## Residual risk (outside this check's trigger set)

`usage_cap.scoped_cap_models: [claude-fable-5, claude-fable-5-1]` is configured
but **inert**, because no orchestrator restart has occurred since the apply
(last: 2026-09-04T17:32:40Z). Until one does, a Fable cap hit takes the
account-level path and would mark the whole account CAPPED rather than capping
only the Fable scope — the exact failure mode check 4 exists to detect, in its
pre-restart form. Filed as an escalation against this task; no config was
edited, per the task's own prohibition. This is a latent exposure, not a
realised one: zero Fable cap hits have occurred.

## Conditional: not triggered

| Trigger | Observed | Fires? |
|---|---|---|
| Any rejection naming `claude-fable-5-1` | zero rejections store-wide, any model, any role | no |
| A $0.00-cost Fable invocation | $6.08, $2.14, $1.72 — none zero | no |
| Every Fable merger at/over the 600 s wall clock | 1 merger run, 1149 s, `timed_out: false`, merge resolved; 600 s is a flat pre-liveness ceiling, not the binding wall clock | no |
| A role outside `{merger, steward}` | `unexpected_roles: []` | no |

Closing with the numbers. Task 5441 (+14d, fires 2026-09-26) re-runs the same
command with a wider `--window` and evaluates the D6 kill criteria against it.
