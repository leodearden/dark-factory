# Orchestration — workflow template, schemas, prompts, failure modes

This is the generalized form of the proven dark-factory run (`exemplar-run-df-2026-07-06.js`, 28 agents, 64 min, 2.54M subagent tokens, 0 workflow-level errors). Adapt the placeholders (`<ROOT>`, subsystem keys, cluster list, corpora descriptions) from Phase 0 scouting + the project overlay, then run via the Workflow tool in the background.

## Meta and constants

```js
export const meta = {
  name: 'bug-hotspot-survey',
  description: 'Mine git/task history for bug hotspots, deep-review each, verify findings, synthesize cross-system improvements',
  phases: [
    { title: 'Mine', detail: 'fix-task + git-history + postmortem mining', model: 'sonnet' },
    { title: 'Review', detail: 'one deep architectural reviewer per hotspot cluster' },
    { title: 'Verify', detail: 'skeptic checks each finding against the code', model: 'sonnet' },
    { title: 'Synthesize', detail: 'cross-system defect→patch chains and priorities' },
  ],
}

const ROOT = '<absolute repo root>'
// Fixed vocabulary — every phase tags with EXACTLY these keys; this is what joins
// mining → review → synthesis. One key per hand-picked cluster, plus 'other'.
const SUBSYSTEM_KEYS = [ /* cluster keys from Phase 0 */, 'other' ]
```

Each `CLUSTERS` entry (hand-authored in Phase 0 — do not delegate this to agents):

```js
{
  key: 'git-worktrees',
  model: null,          // null → session default (deep review); 'sonnet' for peripheral clusters
  files: 'orchestrator/src/orchestrator/git_ops.py (6.8k lines), warm_lane_pool.py, ...',  // with line counts + churn stats
  context: 'Known bug classes: <operator memory + incident history, 3-6 sentences>. ' +
           'Ask: <2-4 pointed structural questions, e.g. "is the lane lifecycle an enforced ' +
           'state machine with a single owner? Who is allowed to run `git worktree prune`?">',
}
```

## Structured-output schemas

`MINING_SCHEMA` — shared by all three miners:

```js
const MINING_SCHEMA = {
  type: 'object',
  properties: {
    themes: { type: 'array', items: { type: 'object', properties: {
      subsystem: { type: 'string', enum: SUBSYSTEM_KEYS },
      theme: { type: 'string', description: 'short name for the recurring bug class / fix pattern' },
      evidence: { type: 'string', description: '2-5 sentences: concrete examples (commit subjects, task ids/titles, dates) showing recurrence' },
      count_estimate: { type: 'integer', description: 'rough number of distinct fixes/tasks in this theme' },
    }, required: ['subsystem', 'theme', 'evidence', 'count_estimate'] } },
    summary: { type: 'string', description: 'overall picture in <=10 sentences' },
  },
  required: ['themes', 'summary'],
}
```

`REVIEW_SCHEMA` — per-cluster reviewer. Two additions over the proven run, both bought by report-layer gaps: `anchors` (file:line was previously buried in prose) and the `latent-bug` kind (previously extracted by hand at report time); plus `churn_exoneration` (from the reify run — prevents filing against healthy churn):

```js
const REVIEW_SCHEMA = {
  type: 'object',
  properties: {
    hotspot: { type: 'string' },
    architecture_notes: { type: 'string', description: 'how the subsystem is actually structured today, key state, key seams; <=12 sentences' },
    churn_exoneration: { type: 'string', description: "if this cluster's raw churn stats are misleading (young feature under disciplined TDD, mechanical renames), say so and why; empty string if the churn is genuinely bug-driven" },
    findings: { type: 'array', items: { type: 'object', properties: {
      title: { type: 'string' },
      kind: { type: 'string', enum: ['latent-bug', 'duplication', 'missing-invariant', 'mismatched-abstraction', 'redundant-abstraction', 'cross-system-patching', 'god-module', 'state-machine-gap', 'other'] },
      files: { type: 'array', items: { type: 'string' } },
      anchors: { type: 'array', items: { type: 'string' }, description: 'file:line anchors for the load-bearing claims' },
      problem: { type: 'string', description: 'the defect-generating structure, with concrete file:line evidence' },
      proposal: { type: 'string', description: 'concrete systemic improvement: named module/class/invariant, what moves, what gets enforced where, what existing code becomes deletable' },
      bug_history_link: { type: 'string', description: 'which historical bug class this structure produced (commit subjects, task ids), or "speculative" if none' },
      impact: { type: 'string', enum: ['high', 'medium', 'low'] },
      effort: { type: 'string', enum: ['high', 'medium', 'low'] },
    }, required: ['title', 'kind', 'files', 'anchors', 'problem', 'proposal', 'bug_history_link', 'impact', 'effort'] } },
    cross_system_notes: { type: 'string', description: 'evidence that defects in ANOTHER subsystem drove ad-hoc patching here (or vice versa); name the subsystem; empty string if none' },
  },
  required: ['hotspot', 'architecture_notes', 'churn_exoneration', 'findings', 'cross_system_notes'],
}
```

`SKEPTIC_SCHEMA` — one verdict per finding, title-matched verbatim:

```js
const SKEPTIC_SCHEMA = {
  type: 'object',
  properties: { verdicts: { type: 'array', items: { type: 'object', properties: {
    title: { type: 'string', description: 'must match the finding title verbatim' },
    verdict: { type: 'string', enum: ['confirmed', 'weakened', 'refuted'] },
    notes: { type: 'string', description: 'what you checked in the code and what you found; cite file:line' },
  }, required: ['title', 'verdict', 'notes'] } } },
  required: ['verdicts'],
}
```

`CROSS_SCHEMA` — the synthesizer:

```js
const CROSS_SCHEMA = {
  type: 'object',
  properties: {
    chains: { type: 'array', items: { type: 'object', properties: {
      name: { type: 'string' },
      systems: { type: 'array', items: { type: 'string' } },
      description: { type: 'string', description: 'the defect→ad-hoc-patch chain across systems, with evidence' },
      proposal: { type: 'string', description: 'the fundamental fix that removes the need for downstream patching' },
    }, required: ['name', 'systems', 'description', 'proposal'] } },
    top_priorities: { type: 'array', items: { type: 'string' }, description: 'ranked: the 5-8 systemic improvements with best payoff, one sentence each' },
    contradictions: { type: 'string', description: 'places where two cluster reviews disagree or propose conflicting changes; empty if none' },
  },
  required: ['chains', 'top_priorities', 'contradictions'],
}
```

## Prompt templates

### Common mining header (prepended to all three miners)

```
You are a data-mining agent for a bug-hotspot survey of the <project> repo at ${ROOT}.
<2-3 sentence repo overview: what the components are, who writes the commits.>
Your job: identify RECURRING bug classes / fix patterns and tag each with the subsystem it belongs to.
Subsystem keys (use exactly these): <each key WITH its concrete files enumerated, e.g.
git-worktrees (git_ops.py, warm_lane_pool.py, worktree_identity.py, ...)>.
Do NOT modify any files. Return findings via the structured output schema.
Themes must be RECURRING (>=2 occurrences) — one-off fixes are noise. Prefer fewer, sharper
themes (10-25 total) with concrete evidence over exhaustive lists.
IMPORTANT: every theme goes into the `themes` array as its own entry — a summary-only response
will be rejected by the schema. Never emit placeholder or test content to satisfy the schema:
if a source is genuinely empty, return an empty themes array and explain in `summary`.
```

(The final IMPORTANT paragraph is new — see §Failure modes, degenerate-output incident.)

### mine:tasks — fix-task history

Mine the tracker's storage directly (a file read beats N MCP round-trips); the overlay names the source.

```
SOURCE: the task database at ${ROOT}/<tracker path> — <shape probed in Phase 0, e.g. "JSON with
data.master.tasks (~1117 tasks, fields: id, title, description, details, status, dependencies,
priority, metadata)">.
Method: write a python3 script (run via Bash; temp files under the scratchpad) to extract tasks
whose title/description/details match fix-flavored patterns (fix, bug, regression, guard, race,
leak, stale, orphan, crash, wedge, starv, deadlock, retry, fault, FP, false.positive, escalat).
For each, capture id, title, status, and any file paths mentioned. Then READ the matched task
titles+descriptions yourself (print them) and cluster into recurring themes per subsystem.
Note which themes have LIVE (pending/in-progress/blocked/deferred) fix tasks vs historical
(done/cancelled) — mention live task ids in evidence. Pay special attention to families of tasks
that patch the SAME area repeatedly over time.
```

### mine:git — git history

```
SOURCE: git history of ${ROOT} (main branch, since <SINCE>; ~<N> commits, ~<M> fix-flavored).
Method (Bash + git):
1. Per-subsystem fix-commit subject dumps, e.g.:
   git log --since=<recent> --oneline -i --grep='fix' --grep='bug' --grep='amend' --grep='regression' -- <subsystem files> | head -150
   — repeat for each subsystem's source files.
2. Look specifically at "amend:" commits (post-merge patch-ups) and "red-main" fix commits (broke
   main) — these mark the weakest code:
   git log --since=<SINCE> --oneline --grep='amend:' --grep='red-main' --name-only
   (Adapt the marker vocabulary to the project's commit conventions — overlay.)
3. Read the SUBJECTS (and `git show --stat` a sample of ~20 interesting ones) to cluster recurring
   fix themes: what keeps breaking, in which file, in what way (races, None-handling, stale state,
   lock/ordering, path handling, subprocess handling, metadata-shape drift...).
4. Compute which files have the highest ratio of fix-commits to total commits in the last 6 weeks —
   recency matters.
Cluster into themes per subsystem with commit-subject examples as evidence.
```

### mine:plans — postmortems/design docs, with operator memory injected

```
SOURCE: design docs and postmortems: ${ROOT}/<plans dir>/*.md (PRDs, many written to fix bug
classes), CHANGELOG.md, DESIGN.md, <docs dirs>.
Method: ls the dir (prioritize filenames mentioning fix/bug/guard/hotfix/invariant/race/leak/
staleness/recovery and the 30 most recent by mtime). Read/skim them. Each PRD written to fix a bug
class is direct evidence of a historical hotspot — extract WHAT kept breaking and WHERE (subsystem),
and whether the doc notes root cause in a DIFFERENT subsystem than the symptom (cross-system
patching — flag these explicitly in evidence).
Known live context you should weigh (from operator memory): <bulleted incident list from Phase 0
memory search — task ids, one-line descriptions>.
```

### Per-cluster review prompt (`reviewPrompt(c)`)

```
You are a senior architect doing a deep code-quality and architecture review of ONE bug hotspot in
the <project> repo at ${ROOT}. READ-ONLY: do not modify, create, or delete any files; do not run
any state-changing commands (no MCP writes, no git writes).

REPO CONTEXT: <3-sentence repo overview>. Fix-commit density is the survey's hotspot signal.

YOUR HOTSPOT: ${c.key}
CORE FILES: ${c.files}
KNOWN CONTEXT (operator memory + prior incidents — treat as leads to verify, not gospel): ${c.context}

MINED FIX THEMES for this subsystem (from task-db + git-log + postmortem mining):
${briefFor(c.key)}

METHOD (be strategic — some files are thousands of lines):
1. Map structure first: grep for 'def |class ' listings, read the module docstring and top-level state.
2. Read the fix history: git log --since=<recent> --oneline -i --grep='fix' --grep='amend'
   --grep='red-main' -- <core files> | head -100, and git show --stat a sample of the most thematic
   ones. The GOAL is root-cause structure: what property of the code made each recurring bug class
   possible?
3. Deep-read the sections implicated by the fix themes. Follow cross-module seams (who else
   reads/writes this state?).
4. For every candidate finding, verify against actual code with file:line evidence.

WHAT TO LOOK FOR (in priority order):
a. Implicit/unenforced invariants: state machines maintained as scattered flags/counters/status
   strings with no legal-transition enforcement; dict-shaped contracts consumed across process
   boundaries with no schema; counters that must equal derived reality but can drift;
   ordering/locking assumptions not encoded anywhere.
b. Cross-system patching: code here that exists to compensate for a defect or missing guarantee in
   a DIFFERENT subsystem (retries, sleeps, re-checks, "guard" wrappers, defensive re-reads). Name
   the upstream system and the missing guarantee.
c. Duplicated logic: same decision computed in 2+ places that can disagree.
d. Mismatched abstractions: an abstraction whose shape forces every caller to work around it; or a
   refactor that extracted the wrong seam (satellite modules still reaching into the god-module's
   privates).
e. Redundant abstractions: layers that only forward, or two mechanisms for the same job kept alive
   in parallel.
f. God-module decomposition: only propose splits along REAL seams (single-writer state, one lock
   domain, one lifecycle) — not line-count cosmetics.
Additionally: any concrete standalone bug you find en route (wrong guard, missing timeout, unhandled
None) is a finding of kind 'latent-bug' — small, immediately fileable, file:line anchored.

PROPOSALS must be concrete and systemic: name the new module/class/function or the invariant + where
it's enforced (type system / runtime assert / single-writer / schema validation / state-machine
table), what code moves or DIES (name the compensations that become deletable), and which historical
bug class it would have prevented. 3-8 findings, quality over quantity. Rank by (bugs prevented) ×
(feasibility).

If this cluster's raw churn stats turn out to be misleading (healthy TDD feature churn, not
bug-driven), say so in churn_exoneration — a correct exoneration is as valuable as a finding.

Return via the structured output schema. In bug_history_link, tie each finding to concrete history
(commit subjects, task ids) or mark "speculative".
```

### Skeptic prompt (`skepticPrompt(c, review)` — findings capped at 8 per skeptic)

```
You are an adversarial verifier for a code-review finding set on the <project> repo at ${ROOT}.
READ-ONLY — do not modify anything.
A reviewer examined the "${c.key}" hotspot (files: ${c.files}) and produced the findings below.
Your job: try to REFUTE each one against the actual code. A finding is:
- "refuted" if the claimed structure doesn't exist as described (wrong file, already fixed,
  duplication isn't real, invariant IS enforced somewhere the reviewer missed — check for existing
  guards/asserts/tests before confirming).
- "weakened" if the observation is real but materially overstated, or the proposal conflicts with an
  existing mechanism / recent refactor already underway.
- "confirmed" only if you verified the load-bearing claims at the cited (or actual) locations.
Check the code, not the prose. For duplication claims, open BOTH sites. For missing-invariant
claims, grep for existing enforcement (asserts, validators, tests, guards) before agreeing it's
missing. Cite file:line in notes. Return one verdict per finding, title matched verbatim.

FINDINGS (JSON):
${JSON.stringify(findings, null, 1)}
```

### Synthesis prompt

```
You are the synthesis lead for a bug-hotspot architecture survey of the <project> repo at ${ROOT}.
READ-ONLY — verify in code where needed, modify nothing.
Below: per-hotspot verified findings from ${K} parallel reviewers, plus mined recurring fix themes.
Your jobs:
1. CROSS-SYSTEM CHAINS: find defect→ad-hoc-patch chains SPANNING subsystems that no single reviewer
   could see whole. Use each reviewer's cross_system_notes as leads; verify the chain is real by
   spot-checking code on both ends. Merge duplicate findings that describe the same root cause from
   different sides — say which findings merge.
2. TOP PRIORITIES: rank the 5-8 systemic improvements with the best (bug classes killed) ×
   (feasibility) across the whole repo. Where two proposals overlap or conflict, resolve into one
   coherent recommendation.
3. CONTRADICTIONS: flag reviewer proposals that conflict with each other or with mechanisms that
   already exist.

MINED FIX THEMES (all subsystems):
${JSON.stringify(allThemes, null, 1)}

PER-HOTSPOT VERIFIED FINDINGS:
${JSON.stringify(digest, null, 1)}
```

## Execution skeleton

```js
phase('Mine')
const miningResults = await parallel([
  () => agent(MINE_TASKS_PROMPT, { label: 'mine:tasks', phase: 'Mine', model: 'sonnet', effort: 'medium', schema: MINING_SCHEMA }),
  () => agent(MINE_GIT_PROMPT,   { label: 'mine:git',   phase: 'Mine', model: 'sonnet', effort: 'medium', schema: MINING_SCHEMA }),
  () => agent(MINE_PLANS_PROMPT, { label: 'mine:plans', phase: 'Mine', model: 'sonnet', effort: 'medium', schema: MINING_SCHEMA }),
])

// SEMANTIC validation — schema validation is not enough (see Failure modes).
function degenerate(m) {
  if (!m || !Array.isArray(m.themes)) return true
  const real = m.themes.filter(t => t.theme && t.evidence && t.evidence.length > 40 &&
                                    !/^test( |$)/i.test(t.theme) && !/^test( |$)/i.test(t.evidence))
  return real.length < 3   // a healthy lane yields >=3 real themes on any mature repo
}
const mined = []
for (let i = 0; i < miningResults.length; i++) {
  let m = miningResults[i]
  if (degenerate(m)) {
    log(`miner ${i} degenerate — re-running once`)
    m = await agent(MINER_PROMPTS[i] + '\n\nNOTE: a previous attempt produced empty/placeholder output. Every theme MUST be a real, evidenced entry in the themes array.',
                    { label: `mine:retry:${i}`, phase: 'Mine', model: 'sonnet', effort: 'medium', schema: MINING_SCHEMA })
  }
  if (!degenerate(m)) mined.push(m)
  else log(`miner ${i} lost — proceeding without that lane (record in report method line)`)
}
const allThemes = mined.flatMap(m => m.themes || [])

function briefFor(key) {
  const t = allThemes.filter(x => x.subsystem === key)
  if (!t.length) return '(no mined themes tagged to this subsystem — rely on git log yourself)'
  return t.map(x => `- [${x.count_estimate}x] ${x.theme}: ${x.evidence}`).join('\n')
}

phase('Review')
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

// Merge verdicts into findings by verbatim title; unmatched → verdict 'unverified'.
// Digest for synthesis: strip refuted, trim to schema fields, keep verdict + verdict_notes.
const clusters = clusterResults.filter(Boolean)
for (const cr of clusters) {
  const byTitle = new Map(cr.verdicts.map(v => [v.title, v]))
  for (const f of (cr.review.findings || [])) {
    const v = byTitle.get(f.title)
    f.verdict = v ? v.verdict : 'unverified'
    f.verdict_notes = v ? v.notes : ''
  }
}
const digest = clusters.map(cr => ({ ...cr.review,
  findings: (cr.review.findings || []).filter(f => f.verdict !== 'refuted') }))

phase('Synthesize')
const cross = await agent(SYNTHESIS_PROMPT, { label: 'synthesize:cross-system', phase: 'Synthesize', effort: 'high', schema: CROSS_SCHEMA })

return {
  themes: allThemes,
  clusters: digest,
  refuted: clusters.flatMap(cr => (cr.review.findings || [])
    .filter(f => f.verdict === 'refuted')
    .map(f => ({ hotspot: cr.review.hotspot, title: f.title, why: f.verdict_notes }))),
  cross_system: cross,
}
```

(vs the exemplar: `mining_summaries` dropped from the return — prose blobs superseded by `themes`; semantic-validation loop added; `verdict_notes` kept through the digest.)

## Model / effort allocation (proven numbers)

| Stage | Model | Effort | Observed per-agent |
|---|---|---|---|
| Mine ×3 | sonnet | medium | 59–142k tok, 5–22 min |
| Review, core clusters ×9 | session default | high | 102–138k tok, 9–19 min |
| Review, peripheral ×3 | sonnet | high | 97–113k tok, 12–28 min |
| Verify ×12 | sonnet | medium | 56–91k tok, 4–13 min |
| Synthesize ×1 | session default | high | ~141k tok, ~5 min |

"Peripheral" is a Phase 0 judgment call: clusters with lower fix density, smaller blast radius, or less architectural entanglement.

## Failure modes (each observed in a real run — bake the mitigation in)

1. **Degenerate structured output passes schema validation.** The DF run's mine:plans agent did ~99 PRDs of real work, then put everything in `summary` with no `themes`; after two schema rejections it emitted `{"summary":"test","themes":[{"theme":"test theme","evidence":"test evidence",...}]}` — which validated and silently poisoned two downstream prompts. All real plans-lane themes were lost and the coordinator never noticed. Mitigations: the IMPORTANT paragraph in the mining header, the `degenerate()` check + single re-run, and never treating `filter(Boolean)` as validation.
2. **The workflow result is too big to read raw** (~270KB). Digest via Python against the full output file; print an index + high-impact detail only; a digest can itself overflow into a tool-result file — read that in chunks.
3. **Verdict-count conflation.** "N/N survived" counted `!refuted`; the true split was 72 confirmed / 3 weakened / 0 refuted. Count the three classes separately, and mark weakened findings visibly in the report (the DF report lost this distinction at the md layer).
4. **Skeptic join fragility.** Verdicts join back to findings by *verbatim title*; unmatched titles fall back to `unverified` rather than being dropped. Keep the per-skeptic cap (8 findings) aligned with the reviewer's "3–8 findings" instruction so nothing is silently truncated.
5. **Zero refutations is a yellow flag.** 12 skeptics / 0 refutations in the DF run. The skeptic prompt already demands an existing-enforcement grep per missing-invariant claim; if refutations come back zero again, consider blinding the skeptic to the reviewer's `proposal` field (verify the `problem` claim only) to reduce anchoring.
6. **Post-survey filing races** (second turn, for the hand-off): concurrent /prd sessions in one checkout sweep each other's staged files — `git commit --only <path>`; verify batches with `get_task`, not search/grep.
