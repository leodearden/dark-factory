# AFK A7: Bound and consume the recon escalation queue (RE-SCOPED 2026-05-26)

## DIRECTION 3 UPDATE (2026-05-27): dedup abandoned for recon_integrity_issue; pile dismissed; watcher built

A7a/A7b/A7c all landed (tasks 1483/1484/1485 done) and the service restarted onto
them — but a live dry-run proved the **content-fingerprint dedup is ineffective on
real data**. The records have no stable identity: the LLM emits a different
`finding.category` (task_memory_mismatch / other / systemic_pattern / …) and a
polluted `affected_ids` (volatile memory UUIDs, co-flagged tasks, inconsistent
`452` / `task-452` / `task_452` encodings) every cycle.

- A7c backfill dry-run: 6,048 → **5,703** survivors (only 345 folded).
- Even a normalized task-id + summary-class key: **~3,670** survivors; **1,460**
  findings carry no extractable task-id at all.

Root cause is upstream of dedup: escalating **non-actionable** info findings to a
human-facing queue is a category error (the only human action is accept-as-known,
which = not filing). Operator chose **Direction 3**:

1. **Bulk-dismiss the pile** — `fused-memory/scripts/dismiss_recon_integrity_noise.py`
   (dry-run default, idempotent, archives not deletes). Applied 2026-05-27:
   **6,048 → 90 pending** (5,958 recon_integrity_issue archived under
   `data/reconciliation/escalations/archive/2026-05-27/`; 90 kept = 21 blocking +
   recon_stale_run 53, infra_issue 14, risk_identified 9, recon_failure 5,
   dependency_discovered 4, cleanup_needed 4, recon_backlog_overflow 1).
2. **Watcher built now** — `skills/recon-escalation-watcher/` (sibling of
   escalation-watcher; sole closer; action set verify-fixed / accept-as-known /
   file-a-real-task / fix-directly-via-fused-memory). Launcher: `recon-watch/`
   (`mcp.json` → escalation 8103 + fused-memory 8002; `run.sh`). Registered via
   `~/.claude/commands/recon-escalation-watcher.md` symlink. Invoke with
   `./recon-watch/run.sh` (or `/recon-escalation-watcher` in a session whose
   `escalation` MCP points at 8103).
3. **Go-forward fix = follow-up task** — stop the harness escalating non-actionable
   info findings into 8103 (route to a log; escalate only the genuinely-needs-human
   subset, e.g. past a recurrence threshold). Filed 2026-05-27 as **task 1512**.
   Until it lands the watcher holds the line against a steady trickle.

The dedup machinery (A7a/A7b) and A7c backfill script remain in the tree but are
**not the chosen cleanup path** for recon_integrity_issue. Everything below is the
2026-05-26 re-scope, preserved for context.

## Status

**Re-scoped, awaiting implementation approval.** The original premise ("~2,347 actionable
findings pile up without closure") was paused 2026-05-15 pending investigation. The 2026-05-26
investigation (below) found the pile is real but lives somewhere the original probe never
looked, and is made of different stuff than assumed. Direction agreed with operator: **deterministic
dedup + one-shot backfill + make the queue consumable by an escalation-watcher**. A8 is **cancelled**
(closure is in-process; no HTTP transport needed).

---

## Investigation findings (2026-05-26, recomputed against live data)

### 1. Actionable findings are NOT escalated — by design

`harness.py:_maybe_remediate` (~L1092) partitions each cycle's Stage-3 `items_flagged`:
- **non-actionable** → escalated immediately (`_escalate('recon_integrity_issue', …)`, info severity)
- **actionable** → handed to `_run_remediation_pass` (~L1149), an in-process LLM S1→S2→S3 pass that
  *attempts the fix* (real memory/task writes). Whatever its own S3 still flags afterward is escalated
  as `"Unresolved after remediation: …"`.

The Stage-3 system prompt is explicit: *"Inconsistencies found here will be addressed in the next
cycle's Stage 1 and Stage 2."* So actionable findings flow through remediation **and** forward-feed
into the next cycle (`_get_prior_s3_findings`, ~L1072) — they never enter any escalation queue.
**There is therefore no actionable-finding closure problem.** Remediation does real work; it is just
ineffective against a recurring intractable set (missing `PERPLEXITY_API_KEY`, manual `tasks.json`
edits, `done_provenance` backfills) that needs a human.

### 2. The real pile is in a SEPARATE queue the 2026-05-15 probe never checked

Recon runs its **own** escalation MCP server + filesystem queue, distinct from the orchestrator's:
- recon queue: `data/reconciliation/escalations/`, port **8103** (`config.escalation_queue_dir`,
  `escalation_port`; server started in `harness.py:_start_escalation_server` ~L575)
- orchestrator queues: `data/escalations/` (dark_factory, **8102**) and
  `/home/leo/src/reify/data/escalations/` (reify, **8100**)

The original probe counted the orchestrator queue and correctly found **zero** recon escalations
there — that gap is intended. The recon queue tells the real story:

| recon queue (`data/reconciliation/escalations/`) | count |
|---|---|
| total files (root) | 5,823 |
| **pending** | **5,822** |
| ever resolved | 1 |
| archived | 7 |
| `recon_integrity_issue` (info) | 5,735 |
| ↳ "Unresolved after remediation" | 3,881 |
| ↳ "Non-actionable integrity finding" | 1,851 |
| `recon_stale_run` (info) | 51 |
| genuinely `blocking` & pending (recon_failure, backlog_overflow) | 20 |
| filed in April / May 2026 | 3,250 / 2,573 (still growing) |

The pile is dominated by **recurring duplicates** of a few dozen intractable findings (tasks 452,
361, 605/606, 1155, 3655, 1188 …) re-fired every cycle — e.g. one finding for task 452 appears 24×.

### 3. Nobody consumes or closes port 8103

`ss` confirms the server is up (fused-memory pid), but the orchestrator, escalation-watcher, and
dashboard all point at 8100/8102. Nothing reads or resolves 8103. Recon files via `queue.submit()`
**directly**, bypassing even the existing infra_issue submit-time dedup
(`escalation/dedupe.py` + `server._submit_or_dedupe`). The result is a **write-only, un-deduped,
unbounded queue**.

### Answers to the three paused questions
1. *Why do findings sit without escalations (in the orch queue)?* — Actionable findings are routed to
   in-process remediation + forward-feed, never escalated. The escalations that ARE filed go to the
   separate 8103 queue, not the orch queue the probe inspected.
2. *Is the gap the bug, or intended?* — The orch-queue gap is **intended**. The **bug** is the 8103
   queue: un-deduped, un-consumed, unbounded growth.
3. *What does `_run_remediation_pass` do?* — Runs a focused LLM S1→S2→S3 pass that attempts the fix
   and writes memory/tasks; it resolves some findings but is ineffective on the recurring intractable
   set, whose residue becomes the "Unresolved after remediation" escalations. It closes nothing.

---

## Re-scoped problem statement

Reconciliation files thousands of info-severity `recon_integrity_issue` escalations into its own
queue (8103), with **no deduplication, no consumer, and no closure**. The same intractable findings
re-fire every cycle, so the queue grows ~100/day (5,822 pending today) and buries the ~20 genuinely
blocking operational escalations under >5,700 info duplicates. The fix is not "wire closure for
actionable findings" (they aren't escalated) — it is to **bound the queue deterministically and make
it a watchable signal**.

## Chosen approach (operator-agreed)

1. **Deterministic dedup at the recon submit path.** Collapse repeat filings of the same finding into
   one escalation with a recurrence count, using a *content fingerprint* — NOT an LLM, NOT the
   existing `summary_dedupe_key` (whose first-3-token key is the shared boilerplate prefix and would
   collapse everything into one). Fingerprint = `(escalation.category, finding.category,
   tuple(sorted(finding.affected_ids)))`, with a normalized-description hash as a tiebreak when
   `affected_ids` is empty.
2. **One-shot backfill** of the existing 5,822: dedup-collapse to the surviving canonical records,
   dismiss the rest with a resolution note. Net survivors expected in the low hundreds.
3. **Make 8103 consumable** by an escalation-watcher session (see "Operating the recon watcher"
   below) with a recon-tailored playbook (resolve = verified-fixed / accept-as-known /
   file-a-real-task / fix-directly-via-fused-memory), distinct from the orchestrator's task-resume
   flow. **The watcher is the SOLE closer** (operator decision 2026-05-26).

**Rejected:** per-cycle in-process auto-closure (former "A7d"). Operator chose the watcher as sole
closer — recon only files (deduped); a human/agent decides every resolution. Recon never resolves its
own escalations. (Trade-off accepted: the queue grows whenever no watcher is running; dedup keeps that
growth to one-per-distinct-finding.)

A8 (HTTP client) is **not used**: recon holds its own `EscalationQueue` in-process and resolves with
a direct `queue.resolve(...)`; a watcher reaches 8103 as an ordinary MCP client. See `afk-A8-*.md`.

## Pre-split into single-package tasks (architect budget ~$12/task)

- **A7a — escalation package** (pure, testable, no fused-memory): add a content-fingerprint dedup key
  function + make dedup window/categories configurable (recon_integrity_issue, effectively unbounded
  window) so `find_dedupe_parent`/`attach_dedupe_child` can fold recon findings. Inject the key fn
  rather than hard-coding `summary_dedupe_key`.
- **A7b — fused-memory** (depends on A7a): route recon's `_escalate`/`queue.submit()` through the
  dedup path and stamp the finding fingerprint onto the `Escalation` so the key fn can read it.
- **A7c — fused-memory** (depends on A7a; standalone script): one-shot backfill collapsing/dismissing
  the 5,822 existing pending escalations.

**Submitted now: A7a–A7c only** (operator decision 2026-05-26). A7d (auto-closure) is **rejected**
(watcher is sole closer). A7e is **deferred** (land after A7c, when the queue is small):

- **A7e — ops/skill** (DEFERRED; no architect): recon-escalation-watcher enablement — MCP config
  recipe for 8103 + a recon-tailored watcher skill. It is a **sibling** of `escalation-watcher`, not a
  reuse: same MCP poll-loop scaffolding, but a different subject (integrity findings vs task-pipeline
  blockers), different resolution semantics (`resolve_issue` just marks a finding handled — there's no
  agent/worktree/resume to drive; synthetic `recon-<runid>` task_id), and a different action set
  (verify-fixed / accept-as-known / file-a-real-task / fix-directly via fused-memory `update_edge`,
  `delete_memory`, `merge_entities`). It never touches the merge queue or worktrees, has no
  steward/L0–L1 tiering, and uses the dedup recurrence count as its priority signal.

## Operating the recon watcher (answer to "how do I watch 8103 without disturbing 8102")

The escalation-watcher skill binds to whatever `.mcp.json` names `escalation`
(`mcp__escalation__{get_pending_escalations,resolve_issue,get_escalation}`). The 8103 server exposes
the identical tool set. Run a **separate Claude session** whose `escalation` server points at 8103 —
MCP connections are per-process, so the existing 8102 session is untouched:

- **Recommended (zero blast radius):** a dedicated working dir (e.g. `recon-watch/`) containing only
  a `.mcp.json` with `{"mcpServers": {"escalation": {"type": "http", "url":
  "http://127.0.0.1:8103/mcp"}}}`; run `claude` from there.
- **Alternative:** `claude --strict-mcp-config --mcp-config <file>` pointing `escalation` at 8103
  (verify flag names with `claude --help`).
- **Do NOT** add a second `escalation-recon` server to the repo `.mcp.json` and expect the stock
  skill to use it — the skill hard-codes `mcp__escalation__*`.

**Caveat:** do not point a fresh watcher at 8103 before A7c lands — it would try to drain 5,822
escalations. Backfill first; then a deduped queue is watchable.

---

## Original out-of-scope notes (preserved for reference)

The original A7 design (closure wiring + backfill keyed on a forward `escalation_id` link) operated
on the wrong problem (actionable-finding closure, which does not exist). The forward-linkage and
queue-driven-revalidation ideas are superseded by the dedup-first approach above; the
revalidation idea survives as the optional A7d.
