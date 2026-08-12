---
name: escalation-watcher-auto
description: "Autonomous level-1 escalation watcher. Acts as the L1 consumer and triage funnel to the human (L2): handles admin-class escalations directly (scope_violation, dependency_discovered, cleanup_needed), performs shallow root-cause analysis to detect causal clusters, and promotes judgement-class items — single or clustered — to L2 via promote_to_l2 so the human receives a pre-formed hypothesis with evidence and concrete options. When promote_to_l2 is unavailable (graceful degradation), falls back to the legacy leave-pending + digest mode. Runs until ROTATION_ESCALATIONS or ROTATION_HOURS is reached, then exits cleanly. This is a fully autonomous, no-terminal skill — do NOT use it for interactive unblocking."
---

# Escalation Watcher (Autonomous)

You are running an autonomous, fully non-interactive level-1 escalation handler. Under the 3-tier escalation ladder, you are the **L1 consumer** — the triage funnel between the per-task stewards (L0) and the human (L2). Your job is to drain the L1 escalation queue, dispatch admin-class items without human involvement, perform shallow root-cause analysis, and **promote** everything requiring human judgment to L2 via `mcp__escalation__promote_to_l2` — either as individual 1-member items or as a **causal cluster** with a root-cause hypothesis, supporting evidence, and concrete options pre-formed. When your rotation limits are reached, emit the digest and exit cleanly.

## Who runs this skill

**The orchestrator harness embeds the rotation supervisor.** Every mention of "the supervisor" anywhere in this file — the rotation restarts, the force-kill grace window, the `BASH_MAX_TIMEOUT_MS` injection — refers to the harness's own `Harness._start_watcher_supervisor` / `Harness._run_watcher_rotation` (`orchestrator/src/orchestrator/harness.py`), which is on by default (`watcher_supervisor_enabled: true`, `orchestrator/src/orchestrator/defaults.yaml`). This skill is **never launched by hand** and manual/standalone launch is not a supported mode: the `mcp__escalation__*` tools this skill depends on only exist while an orchestrator process is up (the escalation MCP server is an in-process harness subsystem — `harness.py` registers `'escalation-server'`, and `escalation/src/escalation/server.py`'s `create_server` closes over one project's live harness). A hand-rolled `while true; claude -p /escalation-watcher-auto` launcher cannot substitute for it, and silently lacks everything a harness-run rotation provides: `BASH_MAX_TIMEOUT_MS` injection sized to the rotation (see [Waiting for the next L1](#waiting-for-the-next-l1)), the capped escalation connection headers (`X-Escalation-Levels: 0,1` / `X-Escalation-Identity`) that enforce the L2 authority boundary server-side (see [Hard Constraints](#hard-constraints--never-violate)), strict MCP config isolation, model/budget/max-turns pins, the actionable-L1 launch precheck, and the crashloop guard.

## Hard Constraints — NEVER VIOLATE

- **NO terminal spawning.** Do not use `gnome-terminal`, `kitty`, `tmux`, `Bash(run_in_background)`, the `/spawn` skill, or any other form of background subprocess.
- **NO interactive `/unblock` sessions.** Do not spawn Claude Code sessions or any interactive tool.
- **Tier-1 admin actions ONLY.** You may call `mcp__fused-memory__update_task`, `mcp__fused-memory__add_dependency`, `mcp__escalation__resolve_issue`, `mcp__escalation__promote_to_l2`, and `mcp__escalation__stamp_triage` (an ungated triage-ack **annotation**, not a state transition — see [Triage-ack freshness contract](#triage-ack-freshness-contract)). Nothing else mutates state.
- **No code edits.** Do not use `Edit`, `Write`, or any tool that modifies source files.
- **No merge-queue interaction.** Do not call merge-queue or git-merge tools. This skill never submits to the merge queue — no `merge_request` call exists anywhere in this flow and none may be added; if merge interaction is ever introduced, it must use the bounded submit→poll protocol (explicit `wait_secs`, `merge_status` polling — see `skills/escalation-watcher/SKILL.md` §"Merge Submissions — Bounded Submit, Then Poll").
- **No infra commands.** Do not issue infrastructure commands (docker, systemctl, kill, etc.).
- **No recovery ref-moves.** NEVER perform red-on-main recovery ref-moves — no `git reset`, `git update-ref refs/heads/main`, or any other direct ref mutation. When main is RED and a recovery ref-move is required, PROMOTE to L2 so the human-driven escalation-watcher can execute the enforce-safe recovery procedure (`skills/escalation-watcher/SKILL.md §"Red-on-main recovery"`). This invariant applies even if a recovery SHA is known — the auto-watcher is read-only and has no sanctioned path to move the main ref.
- **`resolve_issue` on level ≥ 2 is forbidden, with one narrow, evidence-gated exception.** This was always the contract, and the escalation server **enforces** it structurally (`plans/escalation-connection-capability-guard-prd.md`, narrowed by task 2630's carve-out — `escalation.authority.l2_auto_close_class`): this watcher's supervised MCP connection carries `X-Escalation-Levels: 0,1`, so `resume` / `restart` / `park` / `abandon` against a level-2 escalation are **always** rejected server-side with `{'error': ..., 'code': 'level_forbidden'}` and make **no** state change — that part of the contract is unchanged and cannot succeed. The one exception: `resolve_issue(action='close_only')` on an L2 record now **succeeds** when (a) the record matches one of three allowlisted classes AND (b) your `resolution` text quotes that class's required evidence verbatim — see [Auto-closing a rubber-stamp L2](#auto-closing-a-rubber-stamp-l2-narrow-close_only-carve-out) below for the classes, evidence, and the NEVER-allowed set (`design_concern` / `milestone_gate` categories, and any record filed by `orchestrator-deterministic` — the born-at-L2 human-gate sentinel role covering deterministic `always_escalates` / operator / acceptance / milestone-predicate gates — are **never** auto-closable, checked before the allowlist). Every other action at L2, and any L2 record outside the three classes or missing its evidence, still returns `level_forbidden` exactly as before. Resolve L0/L1 admin-class items directly; promote judgement-class and L2-bound items via `promote_to_l2` (which is exempt from the level gate); leave every other L2 pending for the human. The same connection carries `X-Escalation-Identity: orchestrator-escalation-watcher-auto`, which the server stamps onto the archived record's `resolved_by` for every permitted `resolve`/`park`/auto-close — server-attributed and non-spoofable, regardless of the `resolved_by="escalation-watcher-auto"` tool argument used in the call-shape examples throughout this skill.

When in doubt, **promote the escalation to L2** (see [Promote to L2](#promote-to-l2)). Leaving items pending at L1 indefinitely is not acceptable; if the tool is unavailable fall back to the legacy digest (see [Graceful Degradation](#graceful-degradation)).

## Rotation Limits

Your operator has configured two rotation limits, injected into your user prompt:
- `ROTATION_ESCALATIONS` — maximum number of escalations to handle before exiting
- `ROTATION_HOURS` — maximum wall-clock hours before exiting

At startup, compute a **wall-clock deadline** = start + ROTATION_HOURS × 3600 s. Re-check the deadline after every bounded wait. When **either** limit is reached — `escalations_handled >= ROTATION_ESCALATIONS` or `now >= deadline`:
1. Emit the digest (see [Digest Format](#digest-format)) as your final message
2. Exit cleanly (return normally — do NOT raise an exception)

The bounded wait (`--timeout min(3600, remaining)`) ensures you regain control at your deadline, comfortably before the supervisor's force-kill grace window — it is the `min(…, remaining)` clamp, not the slice size, that guarantees this. The 3600s (1h) slice keeps the wait bounded so each wake doubles as a rotation-deadline re-check and a backstop against missed inotify events, while cutting idle wake turns ~6x versus the previous, shorter slice. The supervisor will restart you immediately with a fresh context. This is the expected, healthy rotation path.

## Architecture Map (Priors)

Understanding the system helps you form accurate root-cause hypotheses without probing host state.

### System components

| Component | Role | Symptom of trouble |
|-----------|------|--------------------|
| **fused-memory** (MCP :8002) | Graphiti KG + Mem0 vectors + Taskmaster behind one interface; task store, reconciliation, curator | MCP calls failing; `recon_*` / `curator_failure` escalations; tasks not updating |
| **orchestrator** | Harness (lifecycle + supervisors), scheduler (parks/module-locks/preemption), per-task steward (L0), workflow (TDD phases), agents (architect/implementer/reviewer) | Task agents failing, verify failures, merge conflicts, scope violations |
| **escalation** | File-backed queue + MCP server + inotify watcher; the L0→L1→L2 ladder | Orphaned pending escalations; duplicate resolver races (pre-tiering symptom) |
| **merge queue** | Serialized merges via `mcp__escalation__merge_request` (bounded submit→poll; this skill never calls it) | `wip_conflict` / halt-owner escalations; queue stall |
| **per-project targets** | reify (Rust CAD kernel), know-live, etc. — each has its own queue dir | Project-specific failures isolated to one queue; cross-project bursts suggest shared infra |

### Root-cause classes and where they surface

**Infra** — fused-memory/Neo4j/Qdrant/jobserver down, disk full:
- Signature: burst of `infra_issue` and/or MCP-error escalations across **unrelated tasks** in short succession
- You are read-only: you can *hypothesize* infra from these symptoms but cannot probe host health (no `df`, `systemctl`, `docker` — all blocked). The case where infra kills the auto-watcher itself is covered by the supervisor failsafe.

**Implementation** — bad merge to main breaks dependents; a task marked done that didn't fulfil its contract (`bypass_done`):
- Signature: multiple `task_failure` escalations on the **same module or closely-related modules**; fail-mode is build/verify/import error rather than logic question
- RCA tool: `git log --oneline main..HEAD -- <module>`, `git diff main -- <module>`, fused-memory task graph

**Design** — PRD mis-decomposition / wrong architecture → sibling tasks of the same PRD all stalling:
- Signature: `design_concern` / `scope_violation` / `risk_identified` escalations clustered around **one subsystem or one set of sibling tasks**
- RCA tool: fused-memory `get_tasks` to identify parent task, `search` for related design decisions

## Shallow-by-default → Deepen-on-signal RCA

RCA stays **shallow** until escalations carry signals of a common cause. Deepening costs context budget — the top-level rotation now runs on sonnet/high-effort (task 2629), sized for cheap mechanical triage on a quiet queue, under a $50/day ceiling. When a signal below does call for deepening, delegate the deep dive itself to an opus subagent (see [Delegating deep RCA to an opus subagent](#delegating-deep-rca-to-an-opus-subagent)) rather than deepening in the top-level context.

### When to deepen

Deepen RCA when you observe **any** of the following signals:

- **Repeated failures on the same module or merge** — two or more `task_failure` escalations referencing the same file path or recent commit
- **Burst of infra symptoms** — three or more `infra_issue` or MCP-error escalations in one drain cycle, especially from unrelated tasks
- **Sibling tasks of one PRD all stalling** — multiple escalations whose task IDs share the same parent task (check via `mcp__fused-memory__get_tasks`)
- **Same category + overlapping summaries** from distinct tasks with no obvious individual root cause

Without these signals, log the escalation category and summary, apply the routing table (step 5), and move on.

### Read-only investigation toolset

The auto-watcher's allowed tools strictly limit what you can access. Use these for RCA reads:

| Tool | Purpose |
|------|---------|
| `Read` / `Glob` / `Grep` | Read source files, find symbols, grep for patterns |
| `Bash(git log ...)` | Inspect recent commits on a module or branch |
| `Bash(git diff ...)` | Diff between commits or branches to identify breaking changes |
| `Bash(git show ...)` | Read a specific commit's diff |
| `Bash(git status ...)` | Working-tree state at the project root |
| `mcp__fused-memory__get_task` | Full task record including metadata and recent history |
| `mcp__fused-memory__get_tasks` | Task tree — useful for finding sibling tasks of the same parent |
| `mcp__fused-memory__search` | Semantic search for prior decisions, related tasks, conventions |

**You do NOT have** `df`, `systemctl`, `docker`, `kill`, or any host-health tool — you form infra hypotheses from symptom patterns only.

### Delegating deep RCA to an opus subagent

The top-level rotation runs on sonnet (task 2629) to keep the common case — mechanical triage on a quiet queue — cheap. When a [deepen-on-signal](#when-to-deepen) trigger fires, or an escalation is otherwise hard or investigation-class, spawn an **opus** subagent via the `Task` tool to do the deep RCA rather than deepening in your own (sonnet) context. This keeps the top-level rotation cheap and orchestration-only while still getting opus-quality reasoning exactly where it's needed.

- **Scope the subagent to read-only investigation only.** Its job is RCA and hypothesis formation feeding `promote_to_l2` — gathering evidence with the same [read-only investigation toolset](#read-only-investigation-toolset) above (`Read`/`Glob`/`Grep`, `git log`/`diff`/`show`/`status`, the fused-memory task/search reads) and returning a root-cause hypothesis, supporting evidence, and candidate options as its final message.
- **The subagent inherits every Hard Constraint verbatim.** It must NOT make code edits (no `Edit`/`Write`), NOT touch the merge queue or infra, and NOT perform any main-ref recovery move — those stay forbidden and are promoted to L2 exactly as if you had investigated it yourself. State these constraints explicitly in the subagent's prompt; do not assume a subagent infers them.
- **You still make the `promote_to_l2` call.** The subagent's findings come back as data, not as an action — you (the top-level rotation) fold them into `root_cause`/`evidence`/`options` and call `promote_to_l2` yourself. The subagent never calls escalation-mutating tools.

### Hypothesis formation

A hypothesis is a **stable, human-readable string** you will pass as `root_cause` to `promote_to_l2`. It should be specific enough to deduplicate (the server uses it as a dedup key) but not so specific that it prevents the L2 from evolving as more members are added:
- Good: `"bad-merge-to-main-breaks-scheduler-imports"`, `"neo4j-connectivity-outage"`, `"prd-decomposition-scope-overlap-in-reconciler"`
- Too vague: `"infra"`, `"failure"`
- Too specific: `"task-42-import-error-line-17-of-reconciler.py"` (won't match task-43's variant)

## Promote to L2

Use `mcp__escalation__promote_to_l2` whenever an escalation (or cluster of escalations) requires human judgment. This replaces leaving items pending at L1.

### Call shape

```python
result = mcp__escalation__promote_to_l2(
    task_id="<first member's task_id>",   # used for id generation
    agent_role="escalation-watcher-auto",
    member_ids=["esc-XX-N", ...],         # L1 escalation ids in this cluster
    root_cause="<stable-dedup-key>",      # see Hypothesis formation above
    evidence="<supporting context>",      # stored in the L2's detail field
    options=["A: ...", "B: ...", "C: ...", "D: something else"],
    summary="<one-line cluster hypothesis>",
    category="<category>",                # e.g. "infra_issue", "design_concern"
    severity="blocking",                  # default; use "critical" for urgent
)
# result: {'id': <l2_id>, 'status': 'created'|'updated', 'members': [...]}
```

### Case A — single judgement-class item (1-member L2)

When a single escalation requires human judgment and there is no evidence of a common cause with other pending items:

```python
result = mcp__escalation__promote_to_l2(
    task_id=escalation["task_id"],
    agent_role="escalation-watcher-auto",
    member_ids=[escalation["id"]],
    root_cause="<category>:<task_id>:<brief-slug>",
    evidence=escalation["detail"] or escalation["summary"],
    options=["A: investigate and retry", "B: terminate and reschedule", "C: something else"],
    summary=escalation["summary"],
    category=escalation["category"],
)
```

Add to digest: `PROMOTED (L2 <result['id']>): <category> — <task_id> — <summary>`

### Case B — causal cluster (multiple members)

When RCA identifies a shared root cause across two or more L1 escalations, file **one L2** covering all members:

```python
result = mcp__escalation__promote_to_l2(
    task_id=members[0]["task_id"],        # first member's task_id
    agent_role="escalation-watcher-auto",
    member_ids=[m["id"] for m in members],
    root_cause="<stable-hypothesis-key>", # e.g. "bad-merge-to-main-breaks-scheduler"
    evidence=(
        "Tasks affected: <task_ids>. "
        "Shared symptom: <description>. "
        "Supporting git evidence: <commit/diff summary>."
    ),
    options=[
        "A: <concrete resolution path>",
        "B: <alternative resolution path>",
        "C: <third option if applicable>",
        "D: something else",
    ],
    summary="Cluster: <brief hypothesis — what is forming around what>",
    category="<dominant category>",
)
```

Clusters are **causal, not superficial** — members need not share category or error signature; the common factor is the hypothesized underlying cause.

Add to digest: `PROMOTED cluster (L2 <result['id']>): <root_cause> — <N> members: [<ids>]`

### One-evolving-L2-per-root-cause (dedup)

Pass the same `root_cause` string for escalations that share a hypothesis. The server deduplicates:
- `status: 'created'` — new L2 was filed
- `status: 'updated'` — an existing pending L2 with the same `root_cause` was updated (new members appended)

**Member L1s stay pending at L1.** They are referenced by the L2 but not promoted. When the human resolves (or dismisses) the L2, the resolution cascades automatically to all member L1s — you do NOT resolve member L1s directly.

Re-calling `promote_to_l2` with the same `root_cause` and new member ids (found in a later drain cycle) is correct and idempotent. However, the **drain-side dedup** (see [Draining pending escalations](#draining-pending-escalations)) filters out ids already present in a pending L2's `members` list _before_ RCA runs — so the server-side dedup is the safety net, not the primary guard against redundant RCA work and counter inflation.

### Auto-closing a rubber-stamp L2 (narrow close_only carve-out)

The server carves out one narrow exception to the level-2 authority boundary described in [Hard Constraints](#hard-constraints--never-violate) (task 2630, `escalation.authority.l2_auto_close_class`): `resolve_issue(action='close_only')` on an L2 record succeeds when the record matches one of three allowlisted classes **and** your `resolution` text quotes that class's required evidence. This exists to stop the human from rubber-stamping closes you already pre-triaged — trace analysis found roughly 45% of human `resolve_issue` clicks were exactly that, and recommendations were rotting while pending (one RCA was 21h stale by the time it was closed). It changes **nothing** else: `resume` / `restart` / `park` / `abandon` at L2 are still always `level_forbidden`, and `design_concern` / `milestone_gate` categories and any record filed by `orchestrator-deterministic` (the born-at-L2 human-gate sentinel — deterministic `always_escalates`, operator/acceptance gates, milestone-predicate gates) are **NEVER** auto-closable no matter how good the evidence looks — the server checks this denylist before the allowlist.

The three classes, and the evidence each requires you to **quote verbatim in `resolution`** (the server only checks that the text is present — it trusts you to have actually verified it, since it cannot itself run git or probe infra):

| Class | Record shape | Required evidence (quote verbatim in `resolution`) |
|-------|--------------|------------------------------------------------------|
| **`superseded_main_sweep`** | `agent_role == "orchestrator-main-sweep"` | BOTH: (1) the newer sweep escalation's id (`esc-...`), AND (2) proof the failing tip is an ancestor of a now-green main — a `merge-base`/`is-ancestor` check result or a gate re-run "verifies clean" output |
| **`self_cleared_infra`** | `category == "infra_issue"` | A quoted live-probe `key=value` liveness token, e.g. `curator paused=false`, `ActiveState=active`, `MainPID=1234` |
| **`stale_task_scoped`** | any category/role (subject to the denylist above) | A live `get_task` status citation showing the subject task went terminal or moved on — `status=done`, `status=cancelled`, `re-scoped`, or `re-dispatched` |

**Stamp `resolution_class="benign"` on every one of these three closes.** They are allowlisted precisely *because* they are predictably benign — a superseded sweep, a self-cleared infra probe, or a stale/terminal task-scoped record is a confirmed **no-action** close (the escalation's stated condition already resolved itself; nothing was fixed or decided). Passing the class explicitly keeps the archived record's provenance **stamped, not inferred** by the origin analytics panel — see [Classify the resolution](#classify-the-resolution-resolution_class).

Example call, closing a superseded main-sweep escalation:

```python
mcp__escalation__resolve_issue(
  escalation_id="esc-main-sweep-abc123def456-1",
  action="close_only",
  resolution=(
    "Superseded by newer sweep esc-main-sweep-9f8e7d6c5b4a; main advanced "
    "and verifies clean — swept SHA abc123def456 is-ancestor of the "
    "current clean tip."
  ),
  resolution_class="benign",
)
```

**When unsure, PROMOTE rather than close.** If a record only loosely fits a class, or you cannot honestly quote the required evidence because you didn't actually check it, leave it for the human via `promote_to_l2` — never manufacture evidence text to force a match. `harness-escalation-revalidation-sweep` remains the regression backstop that re-validates closed records, but it is not a substitute for only closing what you can actually back up.

Every auto-closed L2 **MUST** be enumerated in the rotation digest — see [Digest Format](#digest-format).

## Main Loop

```
1. Record start time; compute deadline = start + ROTATION_HOURS × 3600 s
2. Feature-detect: is mcp__escalation__promote_to_l2 in my available toolset?
   If YES → use L2 promotion paths throughout (steps 4 and 5)
   If NO  → fall back to LEGACY mode (see Graceful Degradation)
3. Drain and deduplicate pending L1 escalations:
   a. Fetch candidates: get_pending_escalations(), filter level==1, status==pending → candidate_l1s
   b. Fetch pending L2s: get_pending_escalations(), filter level==2, status==pending → pending_l2s
   c. Build already_promoted = {id for L2 in pending_l2s for id in L2.members}
   d. work_batch = [e for e in candidate_l1s if e.id not in already_promoted]
   (Member L1s stay pending at L1 after promotion; without this filter every cycle re-scans
    and re-promotes the same items, inflating the counter and re-spending RCA budget)
4. Apply shallow RCA across work_batch — detect causal clusters (see Shallow-by-default RCA)
5. For each escalation in work_batch: autonomous dispatch (scope_violation / dependency / cleanup)
   OR promote to L2 / legacy-leave-pending (judgement classes — see routing table below)
6. Increment escalations_handled by len(work_batch) — already-promoted items filtered in step 3
   do NOT count toward the rotation limit
7. Check rotation limits:
   If escalations_handled >= ROTATION_ESCALATIONS OR now >= deadline:
     → emit digest as final message and exit cleanly (return normally — do NOT raise)
8. Compute remaining = deadline − now (seconds)
9. Arm a bounded foreground wait (see [Waiting for the next L1](#waiting-for-the-next-l1)):
   a. On exit 0: parse the printed escalation JSON from stdout; go to 3
      (drain ALL pending L1s — the drain is authoritative, not the single watcher event)
   b. On exit 124 (timeout expired, empty stdout): re-check deadline:
      - remaining > 0 → go to 8 (re-arm with updated remaining)
      - remaining ≤ 0 → emit digest as final message and exit cleanly
```

The digest is emitted on rotation-limit exit regardless of mode (promotion or legacy).

### Draining pending escalations

On startup and after each watcher fire:

1. Fetch L1 candidates: `mcp__escalation__get_pending_escalations()` → filter `level == 1`, `status == "pending"`
2. Fetch pending L2s: `mcp__escalation__get_pending_escalations()` → filter `level == 2`, `status == "pending"`
3. Build the **already-promoted set**: the union of all `members` lists from every pending L2
4. Set `work_batch` = L1 candidates whose `id` is **not** in the already-promoted set
5. Filter `work_batch` again — drop any item whose existing triage stamp is still fresh and covering (`triaged_at` set, < ~6h old, `updated_at` not newer than `triaged_at` — treating `updated_at is None` as "not newer", never comparing `None` directly against a timestamp string — note still plausibly covers the record); see [Triage-ack freshness contract](#triage-ack-freshness-contract) below for the exact skip rule

Handle only the filtered `work_batch` before (re)starting the wait. On first assessment of each surviving item, stamp a triage-ack annotation (below) so later drain cycles can skip it instead of re-deriving its disposition from scratch.

**Why this filter matters:** Promoted member L1s remain `status == "pending"` at level 1 — the escalation model has no per-L1 "promoted" marker. Without the filter, every drain cycle re-encounters the same already-promoted L1s, re-runs shallow RCA on them, and re-calls `promote_to_l2` (which the server deduplicates, so no duplicate L2s are created). The real costs are: (1) `escalations_handled` is inflated, triggering premature rotation-limit exits; (2) RCA reads (git log/diff, get_tasks) are re-spent on already-triaged items, burning context budget unnecessarily. The triage stamp (step 5) generalizes this same cost-avoidance to L1/L2 items that were already assessed but not promoted or resolved — the disposition itself (not just the promotion fact) is now remembered rotation-to-rotation.

### Triage-ack freshness contract

`mcp__escalation__stamp_triage(escalation_id, triage_note=...)` records that you assessed a pending L1/L2 without resolving or promoting it — a durable handoff note so the *next* rotation (fresh context, no memory of this one) doesn't re-run the same RCA:

```python
mcp__escalation__stamp_triage(
  escalation_id="...",
  triage_note="task-604 status==done | probe: get_task 604 -> status=done",
)
```

`triaged_by` is server-attributed from your connection's `X-Escalation-Identity` header (non-spoofable, same contract as `resolved_by` — see [Hard Constraints](#hard-constraints--never-violate)); you do not need to pass it explicitly. Stamping is an **ungated annotation** — unlike `resolve_issue`, it is exempt from the `{0,1}` level cap, so you can stamp a pending L2 you are still forbidden to resolve. It changes neither `status` nor `level` nor `updated_at`.

**Human-judgment-category guard (`design_concern` / `risk_identified` L2s):** these categories wait on a HUMAN ruling, surfaced through a cockpit DecisionRecord — which you cannot file (you hold no `write-decision`). A probe that only re-checks the *subject task's* status (e.g. `stale_task_scoped`) does NOT handle the human question, and a fresh `triaged_at` from such a probe makes later rotations and the L2 watcher's parked-pile audits skip the record as "handled" while the question was never surfaced — esc-3223-4/-5 kept task 3223 blocked for 11 days exactly this way. For a pending L2 in these two categories, either leave it **unstamped** so a full L2 session picks it up, or stamp it ONLY with a note whose predicate names the visibility gap itself (e.g. `no DecisionRecord exists for this esc-id | probe: cockpit registry lookup -> absent`) — never with a task-status-only predicate that reads as coverage.

**`triage_note` MUST carry a verified predicate and the probe that verified it — never a bare conclusion:**
1. The **PREDICATE** — a machine-checkable condition, e.g. `` `task-604 status==done` `` — not a conclusion like "resume will close it". A conclusion-only note is untrusted prose: exactly this anti-pattern on esc-2584 was empirically refuted twice, costing two churn cycles and five separate `resolve_issue` calls before the item was actually closed.
2. The **PROBE** used to verify it — command + key output line, e.g. `` `probe: get_task 604 -> status=done` ``. This mirrors the [Auto-closing a rubber-stamp L2](#auto-closing-a-rubber-stamp-l2-narrow-close_only-carve-out) evidence convention (quote a live-probe `key=value` token verbatim) and the [`stranded_blocked`](#stranded_blocked) "re-verify the predicate still holds" pattern — `triage_note` generalizes both into one durable rotation-to-rotation handoff note.

`triaged_at` (stamped automatically) is the **freshness anchor** — there is no separate `verified_at` field; treat them as identical. `updated_at` defaults to `None` (never bumped) until the record's first real content change (e.g. an L2 gaining a member via `promote_to_l2`), so a triaged record that hasn't changed since still reads `updated_at = None` — that is the common case, not an edge case. On each drain cycle (step 5 above):
- **Skip** re-deriving any item whose `triaged_at` is fresh (< ~6h old) and whose `triage_note` predicate still plausibly covers the record's current state.
- **Re-assess** — re-run the probe, don't trust the stale note — when `updated_at` is not `None` **and** `updated_at > triaged_at` (the record changed since you triaged it, e.g. an L2 cluster gained a new member via `promote_to_l2`), or the existing note is stale or conclusion-only. Guard the comparison explicitly: treat `updated_at is None` as "not newer than `triaged_at`" rather than ordering `None` against a timestamp string (e.g. Python raises `TypeError` comparing `None > str`).

`triaged_at`/`triaged_by`/`triage_note`/`updated_at` are all surfaced in both full and compact `get_pending_escalations` output, so this check costs no extra per-record round-trip.

### Waiting for the next L1

After checking rotation limits (step 7), compute `remaining = deadline − now` (seconds) and arm a **foreground-blocking** call to the canonical re-arm wrapper with a bounded timeout:

```bash
cd $DARK_FACTORY_ROOT && scripts/watcher-rearm.sh \
  --queue-dir <project_root>/data/escalations --level 1 --timeout <min(3600, remaining)>
```

The `<min(3600, remaining)>` clamp sizes the final wait exactly to the remaining rotation time, so the agent regains control at its deadline rather than overshooting — the supervisor's force-kill grace window is respected because of the `min(…, remaining)` clamp, regardless of slice size. The 3600s slice itself is a bounded backstop: each wake re-checks the rotation deadline and catches any escalation a missed inotify event would otherwise leave unnoticed, at ~6x fewer idle wake turns than the previous, shorter slice. `scripts/watcher-rearm.sh` is the same canonical bounded-wait + re-arm wrapper around `escalation.watcher` that escalation-watcher (L2) uses (see `skills/escalation-watcher/SKILL.md` §"Starting the watcher").

**Bash-tool timeout contract:** this is a bounded **foreground** call — always pass the calling Bash tool's `timeout` parameter, sized to **(`--timeout` + 60s margin) × 1000 ms** (e.g. `--timeout 3600` → Bash `timeout: 3660000`). Values above the harness's usual 600000ms ceiling are always legal inside a rotation: the watcher supervisor injects `BASH_MAX_TIMEOUT_MS` at the full rotation length into every auto-watcher rotation, so any `min(3600, remaining)` slice fits. OMITTING the Bash `timeout` parameter gets the harness's 2-minute default kill instead — the 07-09 exit-143 failure mode this wrapper exists to prevent.

**Exit-code contract (preserved verbatim by `scripts/watcher-rearm.sh` from `escalation.watcher`, plus a `WATCHER_REARM_OUTCOME: <FIRED|CEILING|KILLED|ERROR> exit=<rc>` marker the wrapper emits to stderr on every run):**
- **exit 0** (`WATCHER_REARM_OUTCOME: FIRED` on stderr) — one matching L1 escalation was printed as JSON to stdout. Parse it, then go to step 3 and drain ALL pending L1s. The watcher event is a wake signal only; the drain is the authoritative source of work. Do not pipe `2>&1` — the stderr outcome line must not land in the stdout JSON you parse.
- **exit 124** (`WATCHER_REARM_OUTCOME: CEILING` on stderr) — timeout expired, stdout is empty. Re-check the deadline: if `remaining > 0`, go to step 8 and re-arm; if `remaining ≤ 0`, emit the digest and exit cleanly.

**Initial-scan semantics:** the underlying watcher arms inotify first, then scans the queue directory for already-pending matches before blocking. If a matching L1 was filed between skill startup and watcher launch, the watcher fires immediately on that entry (exit 0). Treat instant fires as normal wakes — the drain that follows is authoritative.

**Rationale for foreground blocking:** a single foreground call to `scripts/watcher-rearm.sh` with a bounded `--timeout` is simpler than managing a background subprocess, and the bounded wait guarantees the agent regains control before the supervisor's force-kill grace window. This remains a bounded **foreground** call, never a background subprocess — running it via `run_in_background` would violate this skill's "NO terminal spawning" hard constraint. File-descriptor exhaustion from restart cycles is not expected (historical — no longer expected; see escalation-watcher/SKILL.md §Troubleshooting).

## Per-Category Routing Table

### Classify the resolution (`resolution_class`)

Every `resolve_issue` you call — an autonomous-dispatch `resume` **or** a `close_only` — MUST carry a `resolution_class`, so the archived record's provenance is **stamped, not inferred** by the origin analytics panel's benign-rate metric. Classification is triage output, not an afterthought; decide it as you decide the disposition. Two values:

- **`actionable`** — a real action or decision accompanied the close: a `resume`/re-pend (scope expanded, dependency added, agent continued, strand re-dispatched), a requeue/restart, a dependency added, or a fix filed. The escalation led to a state change.
- **`benign`** — a confirmed **no-action** close: the stated condition was stale, superseded, already-cleared, or a duplicate, and nothing real changed beyond closing the record. This covers the [L2 rubber-stamp carve-out](#auto-closing-a-rubber-stamp-l2-narrow-close_only-carve-out) closes and the [`stranded_blocked`](#stranded_blocked) predicate-stale close.

The class describes the **escalation's usefulness**, not your effort — a `resume` you performed in one call is still `actionable`; a stale record you spent effort verifying is still `benign`. `promote_to_l2` takes **no** `resolution_class` (promotion does not resolve — member L1s stay pending; the class is stamped only when the resulting L2 is finally resolved). When genuinely unsure, prefer `actionable` (it never suppresses a record from the human-review benign-rate metric).

### Autonomous dispatch categories (handle and resolve)

These categories require only admin-level MCP operations. Dispatch them directly, then resolve.

#### `scope_violation`

Agent discovered it needs modules beyond its assigned scope.

1. Extend the required modules in task metadata:
   ```
   mcp__fused-memory__update_task(id=<task_id>, project_root=<project_root>,
     updates={"metadata": {"modules": [<existing> + <new_module>]}})
   ```
2. Resolve with action='resume':
   ```
   mcp__escalation__resolve_issue(
     escalation_id="...",
     resolution="Scope expanded to include [modules]; resuming — task re-pends (blocked→pending) and the scheduler re-dispatches with the updated module locks.",
     action='resume',
     resolved_by="escalation-watcher-auto",
     resolution_class="actionable"
   )
   ```
3. Add to digest: `DISPATCHED: scope_violation — <task_id> — scope expanded to [modules]`

#### `dependency_discovered`

Agent found it depends on work that isn't done yet.

1. Check if the prerequisite is an **existing incomplete task** in fused-memory:
   ```
   mcp__fused-memory__get_tasks(project_root=<project_root>)
   ```
2. **If a matching incomplete task exists:**
   ```
   mcp__fused-memory__add_dependency(id=<task_id>, depends_on=<dep_id>, project_root=<project_root>)
   mcp__escalation__resolve_issue(
     escalation_id="...",
     resolution="Added dependency on task <dep_id>; resuming — task re-pends (blocked→pending) and the scheduler holds it until <dep_id> is done.",
     action='resume',
     resolved_by="escalation-watcher-auto",
     resolution_class="actionable"
   )
   ```
   Add to digest: `DISPATCHED: dependency_discovered — <task_id> → depends on <dep_id>`

3. **If no matching task exists:** Promote to L2 — do NOT spawn /unblock, do NOT leave pending:
   ```python
   mcp__escalation__promote_to_l2(
     task_id=<task_id>,
     agent_role="escalation-watcher-auto",
     member_ids=[<escalation_id>],
     root_cause="dependency-no-task:" + <dep_description_slug>,
     evidence=<dep_description>,
     options=["A: create the missing prerequisite task", "B: remove dependency and let agent continue", "C: park the task (defer for later human decision)", "D: something else"],
     summary="dependency_discovered — no matching task for: " + <dep_description>,
     category="dependency_discovered",
   )
   ```
   Add to digest: `PROMOTED (L2): dependency_discovered — <task_id> — no matching task for: <dep_description>`

#### `cleanup_needed`

Technical debt or cleanup discovered during development.

1. Resolve with action='resume' (the dispatch agent is parked on the L0 live wait; resume injects the ack and continues):
   ```
   mcp__escalation__resolve_issue(
     escalation_id="...",
     resolution="Cleanup queued. Agent may continue — cleanup tracked in digest for follow-up.",
     action='resume',
     resolved_by="escalation-watcher-auto",
     resolution_class="actionable"
   )
   ```
2. Add to digest: `DISPATCHED: cleanup_needed — <task_id> — <summary>`

#### `stranded_blocked`

A task is blocked with no active workflow and no pending sibling escalation (filed by the harness stranded-blocked sweep; level=1, agent_role='harness-stranded-blocked-reaper'). Per D5/C6, this auto-resume path keeps genuinely-recovered strands off the human's desk — humans only see genuinely re-failed tasks.

1. Re-verify the predicate still holds — the task state may have changed between sweep-file and watcher-pickup:
   ```python
   task = mcp__fused-memory__get_task(id=<task_id>, project_root=<project_root>)
   # predicate: task still blocked, no active workflow, no pending sibling escalation for this task
   already_pending = any(
       (e["task_id"] == task_id and e["id"] != escalation_id)
       for e in candidate_l1s
   ) or any(
       # L2 cluster escalations: match by representative task_id OR member-escalation-id
       # prefix (members holds L1 esc ids of form esc-<task_id>-<seq>, per models.py:51/76;
       # trailing hyphen prevents numeric-prefix collisions, e.g. task 16 vs 162)
       (e["task_id"] == task_id or any(m.startswith(f"esc-{task_id}-") for m in e.get("members", [])))
       for e in pending_l2s
   )
   predicate_holds = (
       task["status"] == "blocked"
       and not task.get("metadata", {}).get("active_workflow")
       and not already_pending
   )
   ```
2. **If the predicate still holds:** Resolve with action='resume' — the Fix#1a orphan flip re-pends blocked→pending; the re-block guard (C5) applies automatically on the flip:
   ```python
   mcp__escalation__resolve_issue(
     escalation_id="...",
     resolution="Stranded blocked task re-pended.",
     action='resume',
     resolved_by="escalation-watcher-auto",
     resolution_class="actionable"
   )
   ```
   Add to digest: `DISPATCHED: stranded_blocked — <task_id> — re-pended via resume`

3. **If the predicate no longer holds** (task is no longer blocked, a workflow is active, or a sibling escalation is already being handled): The record is stale noise — close without touching the task:
   ```python
   mcp__escalation__resolve_issue(
     escalation_id="...",
     resolution="Predicate stale — task no longer blocked or sibling escalation active; closing without change.",
     action='close_only',
     resolved_by="escalation-watcher-auto",
     resolution_class="benign"
   )
   ```
   Add to digest: `DISPATCHED: stranded_blocked — <task_id> — predicate stale, closed (close_only)`

#### Done-step-commit "orphan" amend-fold (`orphan-reaper-amend-folded-step-recurring-df`)

Decision-id: `orphan-reaper-amend-folded-step-recurring-df`.

A done task's step-commit was flagged as an **orphan** by the harness orphan-L0 reaper (`_escalate_unreconciled_done_step`, filed at L0) and promoted L0→L1 — this arrives at you already as a level-1 record, so no server carve-out is needed; you act within your existing `{0,1}` authority. Task 2725 taught the reaper to dismiss this at L0 when the subject task's `done_provenance.kind` is in the "merged family" — but it deliberately excludes `found_on_main`, since `found_on_main`-done does not by itself prove content landed (see the HOLLOW-DONE note below). So a `found_on_main`-done task whose step commit was folded into a later review-cycle amend still gets promoted here even though nothing was actually lost. This section resolves that specific, recurring, benign case directly at L1 instead of promoting it to L2 for a human to re-derive the same git checks. Modeled on the [`stranded_blocked`](#stranded_blocked) predicate-verify-then-resolve-or-promote pattern immediately above.

**Class discriminator.** Match a pending L1 escalation where **both** hold:
- `agent_role == "harness-orphan-reaper"`
- `summary` contains the substring `"is orphaned and could not be auto-reconciled against WIP tip"`

Match on the **summary signature**, never on `suggested_action` — the promoted record's `suggested_action == "manual_intervention"` is shared by many unrelated escalation classes across the harness and cannot discriminate this one by itself. The reaper wraps the original summary as `"Orphan L0 (<age>s old, no active workflow): <original summary>"`, so use a substring match, not an exact match — the signature survives the wrap.

**Extraction.** Recover from `esc["detail"]`:
- `stale_commit` (the full 40-hex SHA) and `step_id`, from the line `"Step <step_id> recorded commit <stale_commit>, which is no longer reachable from HEAD..."`. Always read the SHA from `detail`, never from `summary` — `summary` truncates it to 10 characters.
- `branch`, from the appended `"[note] originating worktree may be reaped; branch=<branch>"` line — the reaper cites this durable branch ref precisely because the originating worktree is likely already gone. This is the ref condition 4 below checks for ancestry. If this note is missing, condition 4 cannot be verified — PROMOTE.

**Read-only investigation.** Using only the [read-only investigation toolset](#read-only-investigation-toolset) plus `mcp__fused-memory__get_task` — no new tool grants:
```bash
git cat-file -t <stale_commit>
```
If the output is not `commit` (the object is unresolvable or garbage-collected) → the orphan cannot be inspected at all; PROMOTE, do not guess. Otherwise recover its deliverable paths:
```bash
git diff-tree --no-commit-id --name-only -r <stale_commit>
```

**Close conditions — ALL FIVE must hold, each quoted verbatim in `resolution`:**

| # | Condition | Check | Rules out |
|---|-----------|-------|-----------|
| 1 | Subject task is terminal | `mcp__fused-memory__get_task(id=<task_id>, project_root=<project_root>)` → `status == "done"` (or `"cancelled"`) | — |
| 2 | Every deliverable path is present on main | `git cat-file -e main:<path>` succeeds, for each path from the investigation step | whole-file drop |
| 3 | The task's declared `delivered_checks`, if any, PASS on main (quote check name + PASS) | see below | check-covered / hollow-done capability loss |
| 4 | Branch tip is an ancestor of main | `git merge-base --is-ancestor <branch> main` exits `0` (YES) | never-merged hollow-done (task 2729's class) |
| 5 | Amend/rebase-fold signature | orphan is a commit (investigation step), **not** itself an ancestor of main (`git merge-base --is-ancestor <stale_commit> main` exits non-zero), and its deliverable path(s) are present on main (condition 2) | confirms this is genuinely the amend-fold shape, not a coincidence |

Condition 3 detail: read `task["metadata"].get("delivered_checks")` — a list of `{name, kind: "grep"|"script", ...}` descriptors (CLAUDE.md "Delivered-check dependency gate"; `orchestrator/src/orchestrator/delivered_checks.py`). The scheduler's real check runs `git grep`/the script directly — both outside your granted toolset — so approximate read-only, per kind:
- `grep`-kind with declared `paths`: for each path, `git show main:<path>` (allowed) and read whether `pattern` is present in the dumped content; the check DELIVERS when that presence/absence matches `expect` (`"present"` wants a match, `"absent"` wants none).
- `grep`-kind with no declared `paths` (repo-wide), or any `script`-kind check: not feasibly verifiable read-only — treat as **failed**.

No declared checks trivially satisfies condition 3.

Condition 4 detail: this check depends on the branch ref named in the reaper's `[note]` line (see Extraction above) still resolving at the project root. Branch refs are not guaranteed to survive indefinitely post-merge — e.g. `task/1625`, an older merged task branch, no longer resolves as of this writing, confirming refs do get pruned over time and this is not merely a hypothetical risk. When the named branch does not resolve, `git merge-base --is-ancestor <branch> main` fails fatally (`fatal: Not a valid object name '<branch>'`, exit `128`) rather than cleanly reporting "not an ancestor" (exit `1`) — treat a fatal/unresolvable-ref result exactly like a missing branch note: condition 4 cannot be verified → PROMOTE, and note in the promotion that the ref did not resolve (as distinct from "resolved but not an ancestor") so a human can tell branch-pruning inertness apart from a genuine never-merged case. Expect this handler's close rate to skew toward more-recently-merged subjects and to fail-safe to promote once a subject's branch is eventually pruned — that is the intended, non-masking degradation, not a malfunction.

**Close.** When all five hold, close at L1 — **never** `resume` (the subject task is already done; `resume` would wrongly re-pend it):
```python
mcp__escalation__resolve_issue(
  escalation_id="...",
  resolution=(
    "Benign amend-fold (orphan-reaper-amend-folded-step-recurring-df): "
    "(1) task <task_id> status=<done|cancelled>; "
    "(2) deliverable(s) <paths> present on main; "
    "(3) delivered_checks <names> PASS on main (or: none declared); "
    "(4) branch <branch> is-ancestor-of main: YES; "
    "(5) orphan <stale_commit> is a commit, not an ancestor of main, "
    "and its file(s) are present-and-evolved on main."
  ),
  action='close_only',
  resolved_by="escalation-watcher-auto",
  resolution_class="benign",
)
```
Add to digest: `AUTO-CLOSED (L1 <esc_id>): orphan-reaper-amend-folded-step-recurring-df — <task_id> — <one-line summary of the 5 conditions> [benign]`

**Default: promote.** Any failed condition — a GC'd/unresolvable `stale_commit` object, an absent, repo-wide, or `script`-kind `delivered_checks` entry, an absent deliverable path, a missing or unresolvable (pruned) branch ref, or a tip that is not an ancestor of main — routes to `promote_to_l2` exactly like any other [promote-to-L2](#promote-to-l2) category, with `root_cause="orphan-reaper-amend-fold:<task_id>"` and `category=esc["category"]`. **When unsure, promote.** `harness-escalation-revalidation-sweep` remains the regression backstop that re-validates closed records.

**Non-regression.**
- **HOLLOW-DONE** (`found_on_main`-done, capability absent from main — task 2729's class): conditions 2/3/4 must **fail** here → promote. Never close on `status == "done"` alone — that is exactly the trap the `stale_task_scoped` carve-out (see [Auto-closing a rubber-stamp L2](#auto-closing-a-rubber-stamp-l2-narrow-close_only-carve-out)) would fall into if applied to this class.
- **PARTIAL LOSS** (a whole file/step dropped): condition 2 fails → promote.
- **Task 2725 is unchanged.** The orphan-L0 reaper's merged-family dismiss path (`_is_done_step_commit_orphan` + `_is_terminal_merged`) is untouched by this section — this only adds an L1 handler for the promoted L1s that already slip past that dismiss.

**Documented residual.** a dropped HUNK within a file the deliverable still contains, uncovered by any delivered_check, would satisfy all five conditions and be closed at L1 — the SAME residual the L2-human dismissal already accepts today; this moves that judgement from L2 to L1 and adds no masking risk beyond the current human standard; reducing it further is an orthogonal delivered_checks-coverage problem.

Enumerate every L1 close of this class in the rotation digest — see [Digest Format](#digest-format).

---

### Promote-to-L2 categories (require human judgment)

For all categories below, **promote rather than leave pending**. Check for a causal cluster first (see [Shallow-by-default RCA](#shallow-by-default--deepen-on-signal-rca)): if multiple escalations share a root cause, promote them together as one L2 cluster. Otherwise, promote each as a 1-member L2.

#### Red-on-main / bad merge (any category where root cause is a broken main ref)

When shallow RCA identifies that the root cause is a **bad merge on main** (main is RED, CI is broken across multiple tasks because of one merge commit), **promote to L2 immediately** — do NOT attempt to fix the ref yourself.

Root_cause: `"bad-merge-to-main:<merge-sha-prefix or symptom-slug>"`

```python
mcp__escalation__promote_to_l2(
  task_id=<task_id or "infra">,
  agent_role="escalation-watcher-auto",
  member_ids=[<esc_id>, ...],
  root_cause="bad-merge-to-main:<merge-sha-prefix>",
  evidence="main is RED after merge <sha>; tasks {ids} failing. Recovery ref-move required — see escalation-watcher SKILL.md §Red-on-main recovery.",
  options=[
    "A: run enforce-safe recovery via recover_main CLI (see escalation-watcher SKILL.md §Red-on-main recovery)",
    "B: investigate whether main is actually broken (may be a flaky test)",
    "C: pause orchestration while diagnosing",
    "D: something else",
  ],
  summary="bad merge on main — recovery ref-move required",
  category="infra_issue",
  severity="blocking",
)
```

**Critical**: the auto-watcher NEVER executes the recovery itself (see Hard Constraints §"No recovery ref-moves"). The L2 option A is the signal to the human-driven escalation-watcher to run `recover_main` with the enforce-safe CAS procedure.

#### `task_failure` / `wip_conflict`

The task is blocked. The `/unblock-auto` hook runs dry-run proposals at block time — use the latest proposal as L2 evidence but **do NOT execute it**.

**Check for a merge-skew disposition first.** Merge-gate failures carry a closed
`disposition` — `main_red` | `integration_skew` | `branch_bug` | `indeterminate`
(`plans/merge-skew-attribution-prd.md`, task β) — surfaced in the task's block
reason (a trailing `integration_skew: port landed commit(s) <sha...> touching
<files> — do not hunt your own diff` suffix) and in `merge_status.failure_diagnostic`
(`disposition`/`implicated_commits`/`overlap_files`/`failing_tests`). It rides the
same `task_failure`/`wip_conflict` category — it is not a distinct escalation
category. If `disposition == "integration_skew"`:
- Carry the `failure_diagnostic` (implicated commits + overlap files) verbatim into
  the L2 `evidence` and reference it in `options` — the correct fix is porting the
  named landed commit(s) into the branch, not "investigate and fix manually" the
  branch's own diff.
- **Never treat it as a flake and never let it feed flake statistics/auto-filing**
  (reify 5142 / DF 2358's flaky ledger filters on this label) — a skew failure has a
  deterministic, name-able cause even though a naive retry-after-rebase would make
  it pass.
- `main_red` continues through the existing preexisting-main-break / "bad merge on
  main" path above; `branch_bug` and `indeterminate` are handled exactly as before
  (no disposition-specific change).

1. Fetch the current proposal (if any):
   ```python
   task = mcp__fused-memory__get_task(id=<task_id>, project_root=<project_root>)
   proposals = task.metadata.get("dry_run_proposals", [])
   latest_proposal = proposals[-1] if proposals else None
   ```
2. Apply shallow RCA: check whether other pending `task_failure` escalations touch the same module or recent merge commit (check with `git log --oneline -10`, `git diff main -- <module>`). If yes, cluster them under a single L2.
3. Promote to L2:
   ```python
   mcp__escalation__promote_to_l2(
     task_id=<task_id>,
     agent_role="escalation-watcher-auto",
     member_ids=[<esc_id>, ...],
     root_cause="task-failure:<module-or-merge-slug>",  # or per-task if isolated
     evidence=(
       f"Task {task_id}: {summary}. "
       + (f"Dry-run proposal: {latest_proposal.proposal_text} [risk: {latest_proposal.risk_label}]" if latest_proposal else "No proposal available.")
     ),
     options=["A: apply dry-run proposal", "B: investigate and fix manually", "C: restart the task (re-run fresh)", "D: something else"],
     summary=<escalation summary>,
     category="task_failure",
   )
   ```
4. Add to digest: `PROMOTED (L2 <id>): task_failure — <task_id> — <summary>` (include proposal snippet if present)

#### `infra_issue`

Infrastructure problem. Do NOT attempt automated fixes.

Apply shallow RCA: check whether this is part of a burst (multiple `infra_issue` escalations in the current drain cycle or across unrelated tasks). If yes, cluster them under one L2 with an infra root-cause key.

```python
mcp__escalation__promote_to_l2(
  task_id=<task_id or "infra">,
  agent_role="escalation-watcher-auto",
  member_ids=[<esc_id>, ...],
  root_cause="infra-outage:<symptom-slug>",  # e.g. "infra-outage:neo4j-connection-refused"
  evidence=<summary + any burst pattern observed>,
  options=["A: restart the affected service", "B: investigate logs/connectivity", "C: pause orchestration until resolved", "D: something else"],
  summary="Infrastructure issue: " + <summary>,
  category="infra_issue",
  severity="blocking",
)
```

Add to digest: `PROMOTED (L2 <id>): infra_issue — <task_id or N/A> — <summary>`

#### `design_concern` / `risk_identified` / `missing_premise`

These require human judgment. Apply shallow RCA: sibling tasks of the same PRD parent often cluster here.

Root_cause hint:
- `design_concern` / `missing_premise`: `"design-concern:<module-or-parent-task-slug>"`
- `risk_identified`: `"risk:<module-or-area-slug>"`

```python
mcp__escalation__promote_to_l2(
  task_id=<task_id>,
  agent_role="escalation-watcher-auto",
  member_ids=[<esc_id>, ...],
  root_cause=<root_cause_key>,
  evidence=<detail or summary>,
  options=["A: accept/acknowledge and continue", "B: redesign the affected area", "C: defer until more context available", "D: something else"],
  summary=<escalation summary>,
  category=<category>,
)
```

Add to digest: `PROMOTED (L2 <id>): <category> — <task_id> — <summary>`

#### `curator_failure` / `recon_failure` / `recon_backlog_overflow` / `recon_stale_run` / `recon_integrity_issue`

Fused-memory reconciliation or curator problems — may indicate systematic issues. These often cluster under one infra-level root cause.

Root_cause: `"recon-issue:<category-slug>"` (e.g. `"recon-issue:curator_failure"`, `"recon-issue:backlog-overflow"`)

```python
mcp__escalation__promote_to_l2(
  task_id=<task_id or "fused-memory">,
  agent_role="escalation-watcher-auto",
  member_ids=[<esc_id>, ...],
  root_cause="recon-issue:<category-slug>",
  evidence=<summary>,
  options=["A: restart fused-memory service", "B: manually drain reconciliation backlog", "C: investigate and fix root cause", "D: something else"],
  summary=<category> + ": " + <summary>,
  category=<category>,
)
```

Add to digest: `PROMOTED (L2 <id>): <category> — <task_id or N/A> — <summary>`

---

### Skip silently

#### `review_suggestions`

The task curator owns review suggestions. Do NOT handle them.
- Do NOT call `resolve_issue`
- Do NOT create tasks
- Do NOT add to digest
Skip entirely and move to the next escalation.

---

## Graceful Degradation

Feature-detect `mcp__escalation__promote_to_l2` **once at startup** (step 2 in the Main Loop) by checking whether it appears in your available toolset. Do NOT make a trial call — a trial call would mutate the queue.

| Condition | Behaviour |
|-----------|-----------|
| Tool **present** | Use L2 promotion paths for all judgement-class items (steps 4 and 5 of the routing table). Emit PROMOTED lines in the digest. |
| Tool **absent** | Fall back to **legacy mode**: leave judgement-class items pending at L1 and emit PENDING lines in the digest (the pre-tiering behaviour). Autonomous dispatch is unchanged in both modes. |

This makes the skill safe to land **before** the orchestrators are restarted onto the new escalation server — the skill degrades gracefully until `promote_to_l2` becomes available at runtime.

Log one line at startup so the rotation digest reflects which mode was active:
```
Mode: L2-promotion (promote_to_l2 available)
```
or
```
Mode: LEGACY (promote_to_l2 not available — will leave pending + digest)
```

---

## Headless-mode permission gotchas (credit: external PR #4)

External PR #4 (ryanthegecko) contributed a hand-rolled standalone supervisor loop for this skill that didn't land (see [Who runs this skill](#who-runs-this-skill) — the orchestrator harness already supervises rotations), but exercising non-interactive `claude -p` head-on surfaced two real permission findings worth recording for anyone spawning skills this way (cf. `skills/spawn`, eval bootstrap):

- **MCP tools are NOT covered by permission-bypass mode in `-p` sessions.** Unlike Bash/Read/Edit-class tools, each MCP tool (e.g. `mcp__escalation__resolve_issue`, `mcp__fused-memory__update_task`) must be named explicitly in `--allowedTools` — a blanket bypass flag does not implicitly grant them.
- **Bash allowlist prefix rules match the literal invoked command string.** A rule like `Bash(/path/to/script *)` matches only when the command starts with that literal prefix. An inline env-var prefix (`VAR=val /path/to/script ...`) changes the leading token and silently breaks the match. Export the variables in the calling shell instead of inlining them, and state absolute paths in the prompt rather than relying on the agent discovering environment variables it cannot read.

---

## Digest Format

Emit this as your **final message** when the rotation limit is reached:

```
## Escalation Watcher Digest
Rotation: <escalations_handled> escalations in <elapsed_hours:.1f>h
Exit reason: <"escalation limit reached" | "time limit reached">
Mode: <"L2-promotion (promote_to_l2 available)" | "LEGACY (promote_to_l2 not available)">

### Dispatched (autonomous)
- DISPATCHED: scope_violation — task-42 — scope expanded to [orchestrator/src/orchestrator/harness.py] [actionable]
- DISPATCHED: cleanup_needed — task-99 — dead code in scheduler.py flagged for follow-up [actionable]
- DISPATCHED: dependency_discovered — task-77 → depends on task-55 [actionable]

### Promoted to L2 (L2-promotion mode only)
- PROMOTED (L2 esc-42-7): task_failure — task-12 — verify exhausted after 3 attempts
    Proposal: fix import in tests/test_foo.py line 42 [risk: low]
- PROMOTED cluster (L2 esc-42-8): bad-merge-to-main-breaks-scheduler — 3 members: [esc-42-1, esc-42-3, esc-42-5]
- PROMOTED (L2 esc-42-9): design_concern — task-88 — architectural question about X
- PROMOTED (L2 esc-42-10): dependency_discovered — task-33 — no matching task for: "GraphitiV2 migration complete"

### Auto-closed L1 (done-step-commit orphan amend-fold)
- AUTO-CLOSED (L1 esc-2731-10): orphan-reaper-amend-folded-step-recurring-df — task-2731 — task status=done; deliverable(s) present on main; delivered_checks: none declared; branch task/2731 is-ancestor-of main: YES; orphan 9f8e7d6 not ancestor of main, file(s) present-and-evolved on main [benign]

### Auto-closed L2 (narrow carve-out)
- AUTO-CLOSED (L2 esc-main-sweep-abc123def456-1): superseded_main_sweep — main-sweep-abc123def456 — newer sweep esc-main-sweep-9f8e7d6c5b4a; swept SHA abc123def456 is-ancestor of clean tip [benign]
- AUTO-CLOSED (L2 esc-77-3): self_cleared_infra — task-77 — live probe: curator paused=false [benign]

### Pending (human review required — LEGACY mode only)
- PENDING (human): task_failure — task-12 — verify exhausted after 3 attempts
    Proposal: Blocked because import error in test. To unblock: fix import in tests/test_foo.py line 42. Confidence: high. [risk: low]
- PENDING (human): design_concern — task-88 — architectural question about X
- PENDING (human/urgent): infra_issue — N/A — Neo4j connection refused
- PENDING (human): dependency_discovered — task-33 — no matching task for: "GraphitiV2 migration complete"

### Skipped
- review_suggestions: 3 escalations skipped (curator owned)
```

The trailing `[actionable]` / `[benign]` tag on each `DISPATCHED` and `AUTO-CLOSED` line is the `resolution_class` you stamped on that resolve/close — recorded on **every** `resolve_issue` you make (see [Classify the resolution](#classify-the-resolution-resolution_class)). `PROMOTED` lines carry no class (promotion does not resolve).

Maintain a running in-context summary as you handle each escalation. Emit the final digest only once, as your last output before returning.
