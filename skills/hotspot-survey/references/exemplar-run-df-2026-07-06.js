// EXEMPLAR — verbatim workflow script from the proven dark-factory run (2026-07-06,
// session d1235f32-4830-4c28-8877-8ee89e87ae0e, run wf_195b1a6d-573: 28 agents,
// 2.54M subagent tokens, 64 min, 75 findings -> plans/bug-hotspot-survey-2026-07-06.md).
// Do NOT run as-is: cluster list, known-context paragraphs, and churn stats are
// DF-specific and pinned to that date. Adapt via references/orchestration.md, which
// also folds in fixes for the failure modes this run exposed (degenerate miner
// output, mining_summaries noise, verdict-count conflation).
export const meta = {
  name: 'bug-hotspot-survey',
  description: 'Mine git/task history for bug hotspots, deep-review each, verify findings, synthesize cross-system improvements',
  phases: [
    { title: 'Mine', detail: 'fix-task + git-history + plans mining', model: 'sonnet' },
    { title: 'Review', detail: 'one deep architectural reviewer per hotspot cluster' },
    { title: 'Verify', detail: 'skeptic checks each finding against the code', model: 'sonnet' },
    { title: 'Synthesize', detail: 'cross-system defect→patch chains and priorities' },
  ],
}

const ROOT = '/home/leo/src/dark-factory'

const SUBSYSTEM_KEYS = [
  'merge-queue', 'workflow', 'harness', 'git-worktrees', 'scheduler', 'verify',
  'fm-task-layer', 'fm-recon', 'fm-memory', 'shared-infra', 'escalation', 'dashboard', 'other',
]

const MINING_SCHEMA = {
  type: 'object',
  properties: {
    themes: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          subsystem: { type: 'string', enum: SUBSYSTEM_KEYS },
          theme: { type: 'string', description: 'short name for the recurring bug class / fix pattern' },
          evidence: { type: 'string', description: '2-5 sentences: concrete examples (commit subjects, task ids/titles, dates) showing recurrence' },
          count_estimate: { type: 'integer', description: 'rough number of distinct fixes/tasks in this theme' },
        },
        required: ['subsystem', 'theme', 'evidence', 'count_estimate'],
      },
    },
    summary: { type: 'string', description: 'overall picture in <=10 sentences' },
  },
  required: ['themes', 'summary'],
}

const REVIEW_SCHEMA = {
  type: 'object',
  properties: {
    hotspot: { type: 'string' },
    architecture_notes: { type: 'string', description: 'how the subsystem is actually structured today, key state, key seams; <=12 sentences' },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          title: { type: 'string' },
          kind: { type: 'string', enum: ['duplication', 'missing-invariant', 'mismatched-abstraction', 'redundant-abstraction', 'cross-system-patching', 'god-module', 'state-machine-gap', 'other'] },
          files: { type: 'array', items: { type: 'string' } },
          problem: { type: 'string', description: 'the defect-generating structure, with concrete file:line evidence' },
          proposal: { type: 'string', description: 'concrete systemic improvement: named module/class/invariant, what moves, what gets enforced where' },
          bug_history_link: { type: 'string', description: 'which historical bug class this structure produced (from fix mining or git log), or "speculative" if none' },
          impact: { type: 'string', enum: ['high', 'medium', 'low'] },
          effort: { type: 'string', enum: ['high', 'medium', 'low'] },
        },
        required: ['title', 'kind', 'files', 'problem', 'proposal', 'bug_history_link', 'impact', 'effort'],
      },
    },
    cross_system_notes: { type: 'string', description: 'evidence that defects in ANOTHER subsystem drove ad-hoc patching here (or vice versa); name the subsystem; empty string if none' },
  },
  required: ['hotspot', 'architecture_notes', 'findings', 'cross_system_notes'],
}

const SKEPTIC_SCHEMA = {
  type: 'object',
  properties: {
    verdicts: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          title: { type: 'string', description: 'must match the finding title verbatim' },
          verdict: { type: 'string', enum: ['confirmed', 'weakened', 'refuted'] },
          notes: { type: 'string', description: 'what you checked in the code and what you found; cite file:line' },
        },
        required: ['title', 'verdict', 'notes'],
      },
    },
  },
  required: ['verdicts'],
}

const CROSS_SCHEMA = {
  type: 'object',
  properties: {
    chains: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          name: { type: 'string' },
          systems: { type: 'array', items: { type: 'string' } },
          description: { type: 'string', description: 'the defect→ad-hoc-patch chain across systems, with evidence' },
          proposal: { type: 'string', description: 'the fundamental fix that removes the need for downstream patching' },
        },
        required: ['name', 'systems', 'description', 'proposal'],
      },
    },
    top_priorities: { type: 'array', items: { type: 'string' }, description: 'ranked: the 5-8 systemic improvements with best payoff, one sentence each' },
    contradictions: { type: 'string', description: 'places where two cluster reviews disagree or propose conflicting changes; empty if none' },
  },
  required: ['chains', 'top_priorities', 'contradictions'],
}

// ---------- Phase 1: Mine ----------
phase('Mine')
log('Mining fix history from 3 sources: tasks.json (1117 tasks), git log (~5.2k fix commits since Jan), plans/ postmortems')

const COMMON_MINING_HEADER = `You are a data-mining agent for a bug-hotspot survey of the dark-factory repo at ${ROOT}.
This is a "software factory": an autonomous TDD orchestrator (orchestrator/), a unified memory+task MCP server (fused-memory/), an escalation server (escalation/), a dashboard (dashboard/), and shared libs (shared/). Most commits are made by autonomous agents.
Your job: identify RECURRING bug classes / fix patterns and tag each with the subsystem it belongs to.
Subsystem keys (use exactly these): merge-queue (orchestrator merge_queue.py + merge_* satellites, suffix_graph, overlap_footprint, recover_main), workflow (workflow.py, steward.py, agents/, review_checkpoint, dry_run_unblock, artifacts), harness (harness.py, service_restart, mcp_lifecycle, event_store, run_store — the orchestrator supervisor/watcher), git-worktrees (git_ops.py, warm_lane_pool, worktree_identity, offline_lane, cargo_scope), scheduler (scheduler.py, deterministic_runner, task_status, overrides, substrate_gate, park_eviction), verify (verify.py, verify_runner, verify_cancel, b3_gate), fm-task-layer (fused-memory server/tools.py, middleware/task_interceptor.py, task_curator, backends/sqlite_task_backend), fm-recon (fused-memory reconciliation/*: harness, stages/, flag_dedup, targeted, prompts), fm-memory (services/memory_service, backends/graphiti_client, mem0), shared-infra (shared/usage_gate.py, shared/cli_invoke.py), escalation (escalation/server.py, queue.py), dashboard, other.
Do NOT modify any files. Return findings via the structured output schema. Themes must be RECURRING (>=2 occurrences) — one-off fixes are noise. Prefer fewer, sharper themes (10-25 total) with concrete evidence over exhaustive lists.`

const miningResults = await parallel([
  () => agent(`${COMMON_MINING_HEADER}

SOURCE: the task database at ${ROOT}/.taskmaster/tasks/tasks.json — JSON with data.master.tasks (~1117 tasks, fields: id, title, description, details, status, dependencies, priority, metadata).
Method: write a python3 script (run via Bash; put temp files in /tmp/claude-1000 if needed) to extract tasks whose title/description/details match fix-flavored patterns (fix, bug, regression, guard, race, leak, stale, orphan, crash, wedge, starv, deadlock, retry, fault, FP, false.positive, escalat). For each, capture id, title, status, and any metadata.files / file paths mentioned in the text. Then READ the matched task titles+descriptions yourself (print them) and cluster into recurring themes per subsystem. Note which themes have LIVE (pending/in-progress/blocked/deferred) fix tasks vs historical (done/cancelled) — mention live task ids in evidence. Pay special attention to families of tasks that patch the SAME area repeatedly over time.`,
    { label: 'mine:tasks', phase: 'Mine', model: 'sonnet', effort: 'medium', schema: MINING_SCHEMA }),

  () => agent(`${COMMON_MINING_HEADER}

SOURCE: git history of ${ROOT} (main branch, since 2026-01-01; ~45k commits, ~5.2k fix-flavored).
Method (Bash + git):
1. Per-subsystem fix-commit subject dumps, e.g.: git log --since=2026-03-01 --oneline -i --grep='fix' --grep='bug' --grep='amend' --grep='regression' -- orchestrator/src/orchestrator/merge_queue.py orchestrator/src/orchestrator/merge_*.py | head -150 — repeat for each subsystem's source files.
2. Look specifically at "amend:" commits (post-merge patch-ups) and "Fix red-main" commits (broke main) — these mark the weakest code: git log --since=2026-01-01 --oneline --grep='amend:' --grep='red-main' with --name-only to attribute to files.
3. Read the SUBJECTS (and git show --stat a sample of ~20 interesting ones) to cluster recurring fix themes: what keeps breaking, in which file, in what way (races, None-handling, stale state, lock/ordering, path handling, subprocess handling, metadata-shape drift...).
4. Also compute which files have the highest ratio of fix-commits to total commits in the last 6 weeks — recency matters.
Cluster into themes per subsystem with commit-subject examples as evidence.`,
    { label: 'mine:git', phase: 'Mine', model: 'sonnet', effort: 'medium', schema: MINING_SCHEMA }),

  () => agent(`${COMMON_MINING_HEADER}

SOURCE: design docs and postmortems: ${ROOT}/plans/*.md (PRDs, many written to fix bug classes), ${ROOT}/CHANGELOG.md, ${ROOT}/DESIGN.md, ${ROOT}/docs/ and ${ROOT}/fused-memory/docs/.
Method: ls plans/ (there may be many; prioritize filenames mentioning fix/bug/guard/hotfix/invariant/race/leak/staleness/recovery and the 30 most recent by mtime). Read/skim them plus CHANGELOG.md. Each PRD written to fix a bug class is direct evidence of a historical hotspot — extract WHAT kept breaking and WHERE (subsystem), and whether the PRD notes root cause in a DIFFERENT subsystem than the symptom (cross-system patching — flag these explicitly in evidence).
Known live context you should weigh (from operator memory): warm-lane orphaned-registration reuse fault (fix tasks 2097-2100); deterministic self-restart detached-cwd bug (task 2105); ORCH_UNIT self-kill deploys (2064); speculation-slot census false positives (2096); recon false-premise cleanup batch (2083/2092/2093); stranded_blocked resume leaves in-progress; B3 gate aborts with no dry-run proposal; conftest forces full-suite verify.`,
    { label: 'mine:plans', phase: 'Mine', model: 'sonnet', effort: 'medium', schema: MINING_SCHEMA }),
])

const mined = miningResults.filter(Boolean)
const allThemes = mined.flatMap(m => m.themes || [])
log(`Mining done: ${allThemes.length} recurring fix themes across ${mined.length}/3 sources`)

function briefFor(key) {
  const t = allThemes.filter(x => x.subsystem === key)
  if (!t.length) return '(no mined themes tagged to this subsystem — rely on git log yourself)'
  return t.map(x => `- [${x.count_estimate}x] ${x.theme}: ${x.evidence}`).join('\n')
}

// ---------- Phase 2+3: Review each cluster, then skeptic-verify (pipelined) ----------
const CLUSTERS = [
  {
    key: 'merge-queue', model: null,
    files: 'orchestrator/src/orchestrator/merge_queue.py (9.4k lines — LARGEST file in repo, still #1 in fix churn AFTER a recent 17-task refactor batch that split out satellites), merge_gates.py, merge_types.py, merge_shadow.py, merge_liveness.py, merge_speculation_controller.py, merge_queue_store.py, merge_request_ledger.py, merge_drift.py, suffix_graph.py, overlap_footprint.py, rebase_cost_readout.py, recover_main.py',
    context: 'Recent refactor batch (df 1985-2002) did module split + invariants + sum-type merge_types. Yet merge_queue.py remains 9.4k lines with 180 changes in the last 3 weeks. Known bug classes: speculation-slot accounting false positives (_finalizing_head invisible to _inflight_speculative_count during head verify — detector counts drift from reality); branch-retention guards on acquire-fault ref-delete; _safe_get_task silent-fallthrough; merge-verify red-main incidents. Ask: did the refactor extract the RIGHT seams? Is speculative-merge state (in-flight, finalizing, head, lanes) a single coherent state machine or scattered counters/flags that keep desynchronizing?',
  },
  {
    key: 'workflow', model: null,
    files: 'orchestrator/src/orchestrator/workflow.py (8.8k lines), steward.py, agents/ (roles.py etc.), review_checkpoint.py, dry_run_unblock.py, artifacts.py',
    context: '59% of workflow.py commits are fixes — highest fix-ratio of any big file. Known: steward single-attempt interruptions leave WIP mid-plan (healthy WIP misdiagnosed as failure); architect refuses non-TDD operational tasks (routing meta-fix task 2085); dry_run_proposals only produced at agent-block time so merge-verify failures never get one; SIMPLE_TASK fast path vs full architect path routing. Ask: is the per-task workflow a real state machine with enforced transitions, or ad-hoc status strings + metadata stamps? Where does agent-lifecycle state live and who else pokes it?',
  },
  {
    key: 'harness', model: null,
    files: 'orchestrator/src/orchestrator/harness.py (9k lines), service_restart.py, mcp_lifecycle.py, event_store.py, run_store.py',
    context: 'Supervisor/watcher layer. Known bug classes: auto-watcher L1→L2 overstep (needed server-side level_forbidden enforcement); recon-watchdog kill-storm guard false-fires on every restart (counts orphan recoveries as kills, task 2039); zombie-lock crash-loop after SIGKILL (PID reuse keeps orchestrator.lock alive); dirty-tree start-guard crash-loops every 60s; detached systemd-run self-restart ignores before_done.cwd → exit 127 (task 2105, sibling bug at service_restart.py:138); ORCH_UNIT unset → in-cgroup restart self-kill (2064). Ask: process-lifecycle + restart logic is spread across harness.py / service_restart.py / deterministic_runner — should there be ONE process-supervision abstraction with an explicit contract (own-unit detection, cwd, verify-by-fresh-PID)? Are watchdog guards built on counters that conflate distinct event kinds?',
  },
  {
    key: 'git-worktrees', model: null,
    files: 'orchestrator/src/orchestrator/git_ops.py (6.8k lines), warm_lane_pool.py, worktree_identity.py, offline_lane.py, cargo_scope.py',
    context: 'Known bug classes: warm-lane orphaned-registration reuse fault — orphan-reaper ran `git worktree prune` and wiped ALL lane registrations, then reuse fast-path + startup restore_assignment re-poisoned lanes on every requeue (fix tasks 2097-2100; 2099 = prune-guard, CRITICAL); acquire-fault has TWO distinct root causes that were repeatedly mis-triaged as one (missing CoW base vs stale-checkout collision "already used by worktree at _lane-K"); deleting a lane dir killed an unrelated task plan.json (quarantine-don\'t-delete lesson). Ask: is warm-lane lifecycle (seed → registered → assigned → in-use → released → pruned) an enforced state machine with a single owner? Who is allowed to run `git worktree prune` and is that enforced? Is lane registration state durable vs derived, and can derived state silently diverge?',
  },
  {
    key: 'scheduler', model: null,
    files: 'orchestrator/src/orchestrator/scheduler.py (4.7k lines), deterministic_runner.py, task_status.py, overrides.py, substrate_gate.py, park_eviction_requests.py',
    context: 'Known: stranded_blocked resume flips task blocked→in-progress which the scheduler never re-dispatches (dead state); external-dep gate had a long bug arc (tasks 1854/1855/1799 + a retired workaround convention); dispatch gates spread across local deps, external deps, complexity routing, hard-blocker veto, force_full_path. Ask: is "dispatchable" computed in one place with one predicate, or re-derived in several? Is the task-status lifecycle (pending/in-progress/blocked/done/cancelled/deferred + metadata guards like reblock_guard) a closed set of legal transitions enforced anywhere? Which status transitions are dead ends?',
  },
  {
    key: 'verify', model: null,
    files: 'orchestrator/src/orchestrator/verify.py (3.6k lines), verify_runner.py, verify_cancel.py, b3_gate.py',
    context: 'Known: conftest.py (or any non-collectable test-data file) change forces full-suite verify (has_conftest heuristic); scoped vs unscoped pyright rules (Protocol/TypedDict → unscoped); B3 gate hard-aborts post-merge red-main class regardless of risk; merge-verify failures produce NO dry-run proposal so trivial type/lint fixes always escalate to human. Ask: is verify-scope derivation (which tests/type-checks to run for a diff) principled or an accretion of special cases? Is there duplicated scope logic between verify.py, merge gates, and offline_lane? Where would a "verify plan" abstraction (input: diff; output: declarative plan) simplify?',
  },
  {
    key: 'fm-task-layer', model: null,
    files: 'fused-memory/src/fused_memory/server/tools.py (4k lines — MCP tool surface), middleware/task_interceptor.py (4.1k lines), middleware/task_curator.py, backends/sqlite_task_backend.py, server/main.py',
    context: 'All task ops route through fused-memory so TaskInterceptor can emit reconciliation events. Known: submit_task validation accretion (task_kind/before_done/always_escalates combos, external deps routed into metadata.external_deps when ":" present); operational live-data tasks mis-routed to TDD pipeline = recurring class (meta-fix task 2085); interceptor at 4.1k lines with 92 fix commits. Ask: metadata is a schemaless dict carrying load-bearing contracts (external_deps, complexity, force_full_path, before_done, reblock_guard, dry_run_proposals, done_provenance, stamps) consumed by ANOTHER process (orchestrator) — where does shape drift bite, and would a typed metadata schema (shared lib, versioned) + validation at the write boundary be the systemic fix? Is interceptor logic duplicating tool-layer validation?',
  },
  {
    key: 'fm-recon', model: null,
    files: 'fused-memory/src/fused_memory/reconciliation/harness.py (2.9k lines), stages/task_knowledge_sync.py (4.2k lines — top fused-memory fix file), flag_dedup.py, targeted.py, stages/memory_consolidator.py, prompts/stage1.py, prompts/stage2.py',
    context: 'LLM-driven reconciliation pipeline. Known bug classes: an entire cleanup batch (2083/2092/2093) was filed on a FALSE model of stage1_flag_marker behavior — the system files fix-tasks against its own misunderstanding of itself; stale-run detector reports recovery-math ages not death ages (chronic mis-triage); dead-letter replay tasks go moot because the durable queue self-re-drives; recon files tasks that are unactionable on the TDD pipeline. Ask: the recon system\'s SELF-MODEL (what stage1/stage2 actually do, what markers mean) is implicit in prompts + code — can invariants be made machine-checkable so recon can\'t file false-premise work? Is task-filing from recon gated on any actionability contract?',
  },
  {
    key: 'shared-infra', model: null,
    files: 'shared/src/shared/usage_gate.py (1.4k lines, 68% fix ratio — HIGHEST in repo), shared/src/shared/cli_invoke.py (1.6k lines), and CHECK: orchestrator/src/orchestrator/usage_gate.py also exists — is it a duplicate/fork of shared/usage_gate.py?',
    context: 'usage_gate rotates a 6-account pool on 429 weekly caps; cli_invoke wraps Claude CLI subprocess invocation (known: zero-output hangs on Anthropic 529 bursts, heredoc flattening, background-task reaping). These are load-bearing for EVERY agent invocation. Ask: why does usage_gate need constant fixing — is account-rotation state shared across processes without coordination? Is cli_invoke conflating transport-level retry, rate-limit failover, and process supervision in one layer? Compare orchestrator/usage_gate.py vs shared/usage_gate.py line by line for fork-drift.',
  },
  {
    key: 'fm-memory', model: 'sonnet',
    files: 'fused-memory/src/fused_memory/services/memory_service.py (2.9k lines), backends/graphiti_client.py (1.9k lines), server/main.py (memory routes)',
    context: 'Known: duplicate Graphiti nodes requiring manual FalkorDB merges; stale-but-valid edge summaries needing update_edge remediation; refresh/rebuild entity-summary machinery grew its own fix history (test_rebuild_entity_summaries 47 fix commits). Ask: dedup/summary-refresh invariants — what guarantees node identity at write time, and why do duplicates recur? Is there duplicated routing logic between memory_service and server tool layer?',
  },
  {
    key: 'escalation', model: 'sonnet',
    files: 'escalation/src/escalation/server.py (1.9k lines), escalation/src/escalation/queue.py, plus how orchestrator + watchers consume it',
    context: '3-tier ladder: L0 per-task steward → L1 auto-watcher → L2 human. Known: auto-watcher overstepped L1→L2 until server-side level_forbidden enforcement; resolve_issue action semantics are a trap (resume vs restart vs close_only; resume on stranded_blocked leaves in-progress; resolution TEXT only reaches the agent on the L0 live-workflow path — silently dropped otherwise); born-at-L2 memberless records don\'t re-pend; stale already-fixed L2s pile up because the autonomous watcher cannot close them. Ask: is the escalation record lifecycle (levels, actions, who-may-close, what-each-action-does-per-state) an enforced state machine? Would an action×state legality matrix, enforced server-side with loud errors, kill this whole bug class?',
  },
  {
    key: 'dashboard', model: 'sonnet',
    files: 'dashboard/src/dashboard/app.py (1.35k lines), dashboard/src/dashboard/data/ (merge_queue.py etc.)',
    context: '46% fix ratio. Suspicion: dashboard data layer scrapes ANOTHER system\'s internal state (orchestrator merge-queue internals, task metadata shapes) and breaks whenever those internals shift — classic cross-system coupling. Verify against git history: are dashboard fixes correlated in time with merge_queue/scheduler changes? If so the systemic fix is a versioned read API / published schema instead of reaching into internals.',
  },
]

phase('Review')
log(`Fanning out ${CLUSTERS.length} deep-review agents (full model for 9 core clusters, sonnet for 3 peripheral); each pipelines straight into a skeptic verifier`)

function reviewPrompt(c) {
  return `You are a senior architect doing a deep code-quality and architecture review of ONE bug hotspot in the dark-factory repo at ${ROOT}. READ-ONLY: do not modify, create, or delete any files; do not run any state-changing commands (no MCP writes, no git writes).

REPO CONTEXT: a "software factory" — an autonomous TDD orchestrator (orchestrator/) dispatches LLM agents to implement tasks in git worktrees, merges via a speculative merge queue, verifies with pytest/pyright; fused-memory/ is the MCP server unifying memory (Graphiti+Mem0) and task management with LLM-driven reconciliation; escalation/ is the human-in-the-loop ladder. Nearly all code was written by the factory itself via TDD. Fix-commit density is the survey's hotspot signal.

YOUR HOTSPOT: ${c.key}
CORE FILES: ${c.files}
KNOWN CONTEXT (operator memory + prior incidents — treat as leads to verify, not gospel): ${c.context}

MINED FIX THEMES for this subsystem (from task-db + git-log + postmortem mining):
${briefFor(c.key)}

METHOD (be strategic — some files are thousands of lines):
1. Map structure first: grep for 'def |class ' listings, read the module docstring and top-level state.
2. Read the fix history: git log --since=2026-04-01 --oneline -i --grep='fix' --grep='amend' --grep='red-main' -- <core files> | head -100, and git show --stat a sample of the most thematic ones. The GOAL is root-cause structure: what property of the code made each recurring bug class possible?
3. Deep-read the sections implicated by the fix themes. Follow cross-module seams (who else reads/writes this state?).
4. For every candidate finding, verify against actual code with file:line evidence.

WHAT TO LOOK FOR (in priority order):
a. Implicit/unenforced invariants: state machines maintained as scattered flags/counters/status strings with no legal-transition enforcement; dict-shaped contracts (task metadata, config, JSON stamps) consumed across process boundaries with no schema; counters that must equal derived reality but can drift; ordering/locking assumptions not encoded anywhere.
b. Cross-system patching: code here that exists to compensate for a defect or missing guarantee in a DIFFERENT subsystem (retries, sleeps, re-checks, "guard" wrappers, defensive re-reads). Name the upstream system and the missing guarantee.
c. Duplicated logic: same decision computed in 2+ places that can disagree (dispatch eligibility, verify scoping, path derivation, status parsing, retry policy).
d. Mismatched abstractions: an abstraction whose shape forces every caller to work around it; or a refactor that extracted the wrong seam (satellite modules that still reach back into the god-module's privates).
e. Redundant abstractions: layers that only forward, or two mechanisms for the same job kept alive in parallel.
f. God-module decomposition: only propose splits along REAL seams (single-writer state, one lock domain, one lifecycle) — not line-count cosmetics.

PROPOSALS must be concrete and systemic: name the new module/class/function or the invariant + where it's enforced (type system / runtime assert / single-writer / schema validation / state-machine table), what code moves or dies, and which historical bug class it would have prevented. 3-8 findings, quality over quantity. Rank by (bugs prevented) × (feasibility).

Return via the structured output schema. In bug_history_link, tie each finding to concrete history (commit subjects, task ids) or mark "speculative".`
}

function skepticPrompt(c, review) {
  const findings = (review.findings || []).slice(0, 8)
  return `You are an adversarial verifier for a code-review finding set on the dark-factory repo at ${ROOT}. READ-ONLY — do not modify anything.
A reviewer examined the "${c.key}" hotspot (files: ${c.files}) and produced the findings below. Your job: try to REFUTE each one against the actual code. A finding is:
- "refuted" if the claimed structure doesn't exist as described (wrong file, already fixed, duplication isn't real, invariant IS enforced somewhere the reviewer missed — check for existing guards/asserts/tests before confirming).
- "weakened" if the observation is real but materially overstated, or the proposal conflicts with an existing mechanism / recent refactor already underway.
- "confirmed" only if you verified the load-bearing claims at the cited (or actual) locations.
Check the code, not the prose. For duplication claims, open BOTH sites. For missing-invariant claims, grep for existing enforcement (asserts, validators, tests, guards) before agreeing it's missing. Cite file:line in notes. Return one verdict per finding, title matched verbatim.

FINDINGS (JSON):
${JSON.stringify(findings, null, 1)}`
}

const clusterResults = await pipeline(
  CLUSTERS,
  (c) => {
    const opts = { label: `review:${c.key}`, phase: 'Review', effort: 'high', schema: REVIEW_SCHEMA }
    if (c.model) opts.model = c.model
    return agent(reviewPrompt(c), opts)
  },
  (review, c) => {
    if (!review || !(review.findings || []).length) return review ? { review, verdicts: [] } : null
    return agent(skepticPrompt(c, review), { label: `verify:${c.key}`, phase: 'Verify', model: 'sonnet', effort: 'medium', schema: SKEPTIC_SCHEMA })
      .then(v => ({ review, verdicts: (v && v.verdicts) || [] }))
  }
)

const clusters = clusterResults.filter(Boolean)
// merge verdicts into findings
for (const cr of clusters) {
  const byTitle = {}
  for (const v of cr.verdicts) byTitle[v.title] = v
  for (const f of cr.review.findings || []) {
    const v = byTitle[f.title]
    f.verdict = v ? v.verdict : 'unverified'
    f.verdict_notes = v ? v.notes : ''
  }
}
const totalFindings = clusters.reduce((n, cr) => n + (cr.review.findings || []).length, 0)
const surviving = clusters.reduce((n, cr) => n + (cr.review.findings || []).filter(f => f.verdict !== 'refuted').length, 0)
log(`Review+verify done: ${clusters.length}/${CLUSTERS.length} clusters, ${surviving}/${totalFindings} findings survived skeptic pass`)

// ---------- Phase 4: Cross-system synthesis ----------
phase('Synthesize')

const digest = clusters.map(cr => ({
  hotspot: cr.review.hotspot,
  architecture_notes: cr.review.architecture_notes,
  cross_system_notes: cr.review.cross_system_notes,
  findings: (cr.review.findings || []).filter(f => f.verdict !== 'refuted').map(f => ({
    title: f.title, kind: f.kind, files: f.files, problem: f.problem, proposal: f.proposal,
    bug_history_link: f.bug_history_link, impact: f.impact, effort: f.effort, verdict: f.verdict,
  })),
}))

const cross = await agent(`You are the synthesis lead for a bug-hotspot architecture survey of the dark-factory repo at ${ROOT}. READ-ONLY — verify in code where needed, modify nothing.
Below: per-hotspot verified findings from 12 parallel reviewers, plus mined recurring fix themes. Your jobs:
1. CROSS-SYSTEM CHAINS: find defect→ad-hoc-patch chains SPANNING subsystems that no single reviewer could see whole (e.g. a missing guarantee in fused-memory's task metadata contract producing defensive patching in scheduler + merge-queue + dashboard; or escalation-action semantics producing workarounds in watcher skills and harness). Use each reviewer's cross_system_notes as leads; verify the chain is real by spot-checking code on both ends. Merge duplicate findings that describe the same root cause from different sides — say which findings merge.
2. TOP PRIORITIES: rank the 5-8 systemic improvements with the best (bug classes killed) × (feasibility) across the whole repo. Where two proposals overlap or conflict, resolve into one coherent recommendation.
3. CONTRADICTIONS: flag reviewer proposals that conflict with each other or with mechanisms that already exist.

MINED FIX THEMES (all subsystems):
${JSON.stringify(allThemes, null, 1)}

PER-HOTSPOT VERIFIED FINDINGS:
${JSON.stringify(digest, null, 1)}`,
  { label: 'synthesize:cross-system', phase: 'Synthesize', effort: 'high', schema: CROSS_SCHEMA })

return {
  mining_summaries: mined.map(m => m.summary),
  themes: allThemes,
  clusters: digest,
  refuted: clusters.flatMap(cr => (cr.review.findings || []).filter(f => f.verdict === 'refuted').map(f => ({ hotspot: cr.review.hotspot, title: f.title, why: f.verdict_notes }))),
  cross_system: cross,
}