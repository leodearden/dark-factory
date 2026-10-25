# Orchestration — workflow template, schemas, prompts, failure modes

This is the generalized form of the proven dark-factory run (`exemplar-run-df-2026-07-06.js`, 28 agents, 64 min, 2.54M subagent tokens, 0 workflow-level errors), extended for refresh mode and for `docs/quality-findings-contract.md` (cited as "contract §n"). Adapt the placeholders from Phase 0 and the project overlay, then run via the Workflow tool in the background. One template serves both modes: in full mode `PRIOR` is empty and every cluster has `review: true`.

## Meta and constants

```js
export const meta = {
  name: 'bug-hotspot-survey',
  description: 'Mine fix history for hotspots, re-verify prior findings, deep-review changed clusters, verify, synthesize',
  phases: [
    { title: 'Mine', detail: 'fix-task + git-history + postmortem mining, and prior-finding re-verification', model: 'sonnet' },
    { title: 'Review', detail: 'one deep architectural reviewer per selected cluster' },
    { title: 'Verify', detail: 'skeptic checks each new finding against the code', model: 'sonnet' },
    { title: 'Synthesize', detail: 'cross-system defect→patch chains and priorities' },
  ],
}

const ROOT = '<absolute repo root>'
const AS_OF = '<as_of_sha: the main commit pinned in Phase 0; every agent reads this tree>'
const SINCE = '<prior run as_of_sha in refresh mode; the window start (a commit) in full mode>'
// Fixed vocabulary: every phase tags with EXACTLY these keys; it joins mining → review → synthesis.
const SUBSYSTEM_KEYS = [ /* cluster keys from Phase 0 */, 'other' ]
// Contract §10: how the quality definition reaches every judging prompt, never hand-restated.
// Project carries docs/code-quality.md → 'Read docs/code-quality.md in full: it is the quality definition ("the definition" below).'
// Project without it (reify today) → 'The quality definition ("the definition" below):\n' + the block printed by
//   <dark-factory venv>/bin/python -c 'from orchestrator.agents.code_quality import guidance; print(guidance())'
// (orchestrator/src/orchestrator/agents/code_quality.py::guidance), pasted verbatim.
const QUALITY_DEF = '<one of the two forms above>'
// The kind-tags block of references/report-format.md, pasted verbatim when adapting. Never retyped.
const KIND_TAGS = { /* ... */ }
// Refresh only: prior findings whose disposition is open/filed/accepted, plus other instruments'
// open findings in these areas (contract §9). Built by the Phase 0 digest; `proposal` already removed.
// `ref` is the finding's key, or its `positional:` ref for a pre-contract baseline whose provisional keys collide.
const PRIOR = [ /* {ref, key, area, sub_area, anchor, tags, title, statement, verdict, disposition, source_run_id} */ ]
```

Each `CLUSTERS` entry (hand-authored in Phase 0; never delegated to agents):

```js
{
  key: 'git-worktrees',
  area: 'orchestrator',  // plain review/briefing.yaml area key, or 'repo'; the cluster key is the findings' sub_area (report-format.md §Area)
  model: null,           // null → session default; 'sonnet' for peripheral clusters
  review: true,          // refresh: true only for a material change or a new cluster (SKILL.md §Refresh mode)
  files: 'orchestrator/src/orchestrator/git_ops.py, warm_lane_pool.py, ...',  // with churn stats and fix ratio
  context: 'Known bug classes: <memory + incident history + other instruments\' open findings, 3-6 sentences>. ' +
           'Ask: <2-4 pointed structural questions>',
}
```

## Structured-output schemas

`MINING_SCHEMA`, shared by the three miners:

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

`REVIEW_SCHEMA`, per cluster reviewer. Field names follow contract §1 (`statement`, `anchor`, `tags`, `severity`); `kind` stays as the survey's lens:

```js
const KINDS = Object.keys(KIND_TAGS.primary_tag)
const TAG_PATTERN = '^(h([1-9]|1[0-4])|comments|tests|inv-[0-9]+|kind:[a-z-]+)$'
const REVIEW_SCHEMA = {
  type: 'object',
  properties: {
    hotspot: { type: 'string' },
    architecture_notes: { type: 'string', description: 'how the subsystem is structured today, key state, key seams; <=12 sentences' },
    churn_exoneration: { type: 'string', description: "if this cluster's churn is misleading (young feature under TDD, mechanical renames), say so and why; empty string otherwise" },
    findings: { type: 'array', items: { type: 'object', properties: {
      title: { type: 'string' },
      kind: { type: 'string', enum: KINDS },
      tags: { type: 'array', minItems: 2, items: { type: 'string', pattern: TAG_PATTERN },
              description: 'tags[0] = the heuristic or stance this rests on (fixed per kind where KIND_TAGS fixes it); include kind:<kind>' },
      anchor: { type: 'string', description: 'path::symbol of the enclosing top-level or class-level definition; no line numbers' },
      anchors: { type: 'array', items: { type: 'string' }, description: 'supporting path::symbol locations' },
      statement: { type: 'string', description: 'the claim and the measured facts behind it, cited as path::symbol; no proposal here' },
      proposal: { type: 'string', description: 'named module/class/invariant, what moves, where it is enforced, what becomes deletable' },
      bug_history_link: { type: 'string', description: 'commit subjects / task ids this structure produced, or "speculative"' },
      severity: { type: 'string', enum: ['high', 'medium', 'low'], description: 'contract §4' },
      effort: { type: 'string', enum: ['high', 'medium', 'low'], description: 'cost of the fix; never a severity input' },
    }, required: ['title', 'kind', 'tags', 'anchor', 'anchors', 'statement', 'proposal', 'bug_history_link', 'severity', 'effort'] } },
    prior_changes: { type: 'array', items: { type: 'object', properties: {
      ref: { type: 'string' }, field: { type: 'string', enum: ['severity', 'verdict'] },
      value: { type: 'string' }, why: { type: 'string' },
    }, required: ['ref', 'field', 'value', 'why'] }, description: 'refresh: a known finding your evidence changes (contract §9); empty otherwise' },
    cross_system_notes: { type: 'string', description: 'defects in ANOTHER subsystem that drove patching here (or vice versa); empty if none' },
  },
  required: ['hotspot', 'architecture_notes', 'churn_exoneration', 'findings', 'prior_changes', 'cross_system_notes'],
}
```

`SKEPTIC_SCHEMA`, one verdict per new finding, joined by display id (the title join of the proven run was fragile):

```js
const SKEPTIC_SCHEMA = {
  type: 'object',
  properties: { verdicts: { type: 'array', items: { type: 'object', properties: {
    id: { type: 'string', description: 'the finding id, verbatim' },
    verdict: { type: 'string', enum: ['confirmed', 'weakened', 'refuted'] },
    anchor: { type: 'string', description: 'the path::symbol you verified the claim at (correct the reviewer if needed)' },
    primary_tag: { type: 'string', pattern: TAG_PATTERN, description: 'the heuristic the verified claim rests on' },
    notes: { type: 'string', description: 'what you checked and found, cited as path::symbol' },
  }, required: ['id', 'verdict', 'anchor', 'primary_tag', 'notes'] } } },
  required: ['verdicts'],
}
```

`REVERIFY_SCHEMA`, refresh only, one outcome per prior finding:

```js
const REVERIFY_SCHEMA = {
  type: 'object',
  properties: { outcomes: { type: 'array', items: { type: 'object', properties: {
    ref: { type: 'string', description: 'the prior finding ref, verbatim' },
    outcome: { type: 'string', enum: ['confirmed', 'weakened', 'refuted', 'fixed'] },
    anchor: { type: 'string', description: 'path::symbol where the structure lives at AS_OF (may differ from the prior anchor); empty if fixed' },
    primary_tag: { type: 'string', description: 'only when the prior finding has no heuristic tag: the heuristic its statement rests on; else empty' },
    fixed_by: { type: 'string', description: 'outcome fixed: the commit that removed the structure, if found; else empty' },
    notes: { type: 'string' },
  }, required: ['ref', 'outcome', 'anchor', 'primary_tag', 'fixed_by', 'notes'] } } },
  required: ['outcomes'],
}
```

`CROSS_SCHEMA`, the synthesizer; members are cited by ref (earlier findings) or display id (new findings), and the digest resolves both to keys:

```js
const CROSS_SCHEMA = {
  type: 'object',
  properties: {
    chains: { type: 'array', items: { type: 'object', properties: {
      name: { type: 'string' },
      members: { type: 'array', items: { type: 'string' }, description: 'finding keys or ids in the chain' },
      description: { type: 'string', description: 'the defect→ad-hoc-patch chain across areas, with evidence' },
      proposal: { type: 'string', description: 'the fundamental fix that removes the downstream patching' },
    }, required: ['name', 'members', 'description', 'proposal'] } },
    top_priorities: { type: 'array', items: { type: 'string' }, description: 'ranked remedies (5-8), one sentence each, citing member keys/ids' },
    contradictions: { type: 'string', description: 'where two reviews disagree or conflict with existing mechanisms; empty if none' },
  },
  required: ['chains', 'top_priorities', 'contradictions'],
}
```

## Prompt templates

### Common mining header (prepended to all three miners)

```
You are a data-mining agent for a bug-hotspot survey of the <project> repo at ${ROOT}.
<2-3 sentence repo overview: what the components are, who writes the commits.>
Your window is the commit range ${SINCE}..${AS_OF} (refresh) or <window start>..${AS_OF} (full); read
nothing outside it.
Your job: identify RECURRING bug classes / fix patterns and tag each with the subsystem it belongs to.
Subsystem keys (use exactly these): <each key WITH its concrete files enumerated>.
A FIX COMMIT is a non-merge commit whose SUBJECT starts with fix:, bugfix: or hotfix: (optional
(scope) and !). `amend:` commits are review amendments made before merge, not fixes: never count them
as fixes. <overlay additions, e.g. a broke-main marker>.
Do NOT modify any files. Return findings via the structured output schema.
Themes must be RECURRING (>=2 occurrences); one-off fixes are noise. Prefer 10-25 sharp themes with
concrete evidence over exhaustive lists.
IMPORTANT: every theme goes into the `themes` array as its own entry; a summary-only response is
rejected by the schema. Never emit placeholder or test content to satisfy the schema: if a source is
genuinely empty, return an empty themes array and explain in `summary`.
```

### mine:tasks — fix-task history

Mine the tracker's storage directly (a file read beats N MCP round-trips); the overlay names the source and its schema probe.

```
SOURCE: the task store at <absolute path of the live store — it lives in the MAIN checkout,
not a worktree>. If it is SQLite, open it READ-ONLY by that absolute path so you never contend
with the live orchestrator, percent-encoding the path into the URI:
sqlite3.connect(pathlib.Path('<absolute path of the live store>').as_uri() + '?mode=ro', uri=True).
Shape: <paste the probe's actual output from Phase 0 step 3 (the overlay names the shape command,
e.g. `python3 scripts/tasks_db_schema.py`); never a column list remembered or copied from this
template or an older doc>. <refresh: only tasks with id > the prior run's method.extra.max_task_id.>
Method: write a python3 script (temp files under the scratchpad) to extract tasks whose
title/description/details match fix-flavored patterns (fix, bug, regression, guard, race, leak, stale,
orphan, crash, wedge, starv, deadlock, retry, fault, false.positive, escalat) or whose origin marks a
fix (<overlay: fix-origin metadata values>). Capture id, title, status, and file paths (from file-level
metadata where present, else mentioned in text). READ the matched titles+descriptions yourself and
cluster into recurring themes per subsystem. Note which themes have LIVE (pending/in-progress/blocked/
deferred) fix tasks vs historical (done/cancelled), with live task ids in evidence. Watch for families
of tasks patching the SAME area repeatedly.
```

### mine:git — git history

```
SOURCE: git history of ${ROOT}, range <window>..${AS_OF}.
Method (Bash + git):
1. Per-subsystem fix-commit subject dumps:
   git log --no-merges --format='%h %ad %s' --date=short <window>..${AS_OF} -- <subsystem files> \
     | grep -E '^[0-9a-f]+ [0-9-]+ (fix|bugfix|hotfix)(\([^)]*\))?!?:' | head -150
   repeated for each subsystem's source files.
2. <overlay markers for commits that broke main, if the project has them> mark the weakest code.
   `amend:` volume per file is review friction: report it in evidence as context, never as a fix count.
3. Read the SUBJECTS (and `git show --stat` ~20 interesting ones) to cluster recurring fix themes:
   what keeps breaking, in which file, in what way (races, None-handling, stale state, lock/ordering,
   path handling, subprocess handling, metadata-shape drift...).
4. Per file, the ratio of fix commits to all non-merge commits in the last 6 weeks of the window.
Cluster into themes per subsystem with commit-subject examples as evidence.
```

### mine:plans — postmortems/design docs, with operator memory injected

```
SOURCE: design docs and postmortems: ${ROOT}/<plans dir>/*.md, CHANGELOG.md, DESIGN.md, <docs dirs>.
Refresh: only docs added or changed in ${SINCE}..${AS_OF} (`git diff --name-only ${SINCE} ${AS_OF} -- <dirs>`).
Skip the other quality instruments' reports (/review, /review-all, census, earlier hotspot surveys):
the coordinator has already digested them into cluster context.
Method: prioritize filenames mentioning fix/bug/guard/hotfix/invariant/race/leak/staleness/recovery and
the 30 most recent. Each PRD written to fix a bug class is direct evidence of a hotspot: extract WHAT
kept breaking and WHERE, and whether the root cause sat in a DIFFERENT subsystem than the symptom
(cross-system patching; flag these explicitly in evidence).
Known live context to weigh (from operator memory): <bulleted incident list from Phase 0>.
```

### Per-cluster review prompt (`reviewPrompt(c)`)

```
You are a senior architect doing a deep code-quality and architecture review of ONE bug hotspot in the
<project> repo at ${ROOT}, at commit ${AS_OF}. READ-ONLY: do not modify, create, or delete any files; no
MCP writes, no git writes.

FIRST: ${QUALITY_DEF}
Quality is the expected cost and risk of the next change. Judge against the definition's fourteen
heuristics and two stances, and name the heuristic every finding rests on. Read docs/quality-findings-contract.md §1 and §4 for the finding fields and severity.

REPO CONTEXT: <3-sentence overview>. Fix-commit density chose this cluster as a place to look; it is not
a measure of how bad the code is, and nothing you report is ranked by it.

YOUR HOTSPOT: ${c.key} (area ${c.area}, sub_area ${c.key})
CORE FILES: ${c.files}
KNOWN CONTEXT (memory, incidents, other instruments' open findings; leads to verify, not gospel): ${c.context}

MINED FIX THEMES for this subsystem:
${briefFor(c.key)}

ALREADY-RECORDED FINDINGS for this cluster (refresh; re-verified this run, by ref):
${knownFor(c.key)}
Do not re-report these. If your evidence changes one's severity or verdict, put it in prior_changes with
the reason. Then look for NEW structure, starting with the code changed in this window:
git diff --stat ${SINCE} ${AS_OF} -- <core files>

METHOD (some files are thousands of lines; be strategic):
1. Map structure first: grep 'def |class ' listings, module docstrings, top-level state.
2. Read the fix history: the window's fix commits on the core files (subject prefix fix:/bugfix:/hotfix:),
   `git show --stat` a sample. The GOAL is root-cause structure: what property of the code made each
   recurring bug class possible?
3. Deep-read the implicated sections; follow cross-module seams (who else reads/writes this state?).
4. Verify every candidate finding in the code, cited as path::symbol.

WHAT TO LOOK FOR (priority order; the heuristic is named so you tag correctly):
a. Unenforced invariants (heuristic 10): state machines kept as scattered flags/counters/status strings
   with no legal-transition table; dict-shaped contracts crossing process boundaries with no schema;
   counters that must equal derived reality but can drift; ordering/locking assumptions encoded nowhere.
b. Cross-system patching (heuristic 7, often 11): code here that compensates for a missing guarantee in
   a DIFFERENT subsystem (retries, sleeps, re-checks, guard wrappers, defensive re-reads). Name the
   upstream system and the missing guarantee.
c. Duplicated decisions (heuristic 11): the same decision computed in 2+ places that can disagree.
d. Mismatched abstractions (heuristic 9, 13): a shape every caller works around, or an extraction along
   the wrong seam (satellites reaching into the parent's privates).
e. Redundant abstractions (heuristic 9): forwarding-only layers, or two mechanisms for one job.
f. Meaningful strings (heuristic 12): routing on string prefixes, status strings that are enums,
   ad-hoc parsers of internal values.
g. God modules (heuristic 6): a module whose purpose needs "and". Propose a split only after heuristic
   14's measurement protocol in the definition, recorded in the statement; never for line count.
h. Everything else the definition holds, by reference: heuristics 1, 2, 3, 4, 5, 8, the Comments stance and
   the Tests stance. Report these as kind 'other' with that heuristic first.
Any concrete standalone bug found en route (wrong guard, missing timeout, unhandled None) is kind
'latent-bug': one anchor, immediately fixable, tagged with the heuristic its absence violates.

TAGS: tags[0] is fixed per kind by this table (null = your choice of heuristic or stance):
${JSON.stringify(KIND_TAGS.primary_tag)}
Add the secondary heuristics the finding also rests on, then 'kind:<kind>'.

The measures docs/code-quality.md lists under Do not steer by may appear as context only, never as the
finding or its severity.

PROPOSALS must be concrete and systemic: name the new module/class/function or the invariant and where
it is enforced (type / runtime check at construction and use / single writer / schema / transition
table), what code DIES (name the compensations that become deletable), and which historical bug class
it would have prevented. 3-8 findings, quality over quantity. Order by severity, then feasibility.

If this cluster's churn turns out to be misleading (healthy TDD feature churn, not bug-driven), say so
in churn_exoneration; a correct exoneration is as valuable as a finding.
Return via the structured output schema. In bug_history_link, tie each finding to concrete history
(commit subjects, task ids) or write "speculative".
```

### Skeptic prompt (`skepticPrompt(c, findings)`, at most 8 findings, `proposal` removed)

```
You are an adversarial verifier for findings about the <project> repo at ${ROOT}, commit ${AS_OF}.
READ-ONLY. First: ${QUALITY_DEF} It defines the heuristics these findings cite.
A reviewer examined the "${c.key}" hotspot (files: ${c.files}). Below are its claims; you are shown
the claim only, not what the reviewer proposes to do about it. Try to REFUTE each against the code:
- "refuted": the claimed structure does not exist as described (wrong file, already fixed, the
  duplication is not real, the invariant IS enforced somewhere the reviewer missed; check for existing
  guards/asserts/validators/tests before confirming).
- "weakened": real but materially overstated.
- "confirmed": only if you verified the load-bearing claims at the cited (or actual) locations.
Check the code, not the prose. For duplication, open BOTH sites. For missing-invariant claims, grep for
existing enforcement first. Return the path::symbol you verified at as `anchor`, and as `primary_tag`
the heuristic the verified claim actually rests on. One verdict per finding, id verbatim.

FINDINGS (JSON):
${JSON.stringify(findings, null, 1)}
```

### Re-verification prompt (`reverifyPrompt(batch)`, refresh only, at most 8 prior findings)

```
You are re-verifying recorded findings about the <project> repo at ${ROOT}. They were confirmed at
commit ${SINCE}; the code is now at ${AS_OF}. READ-ONLY. First: ${QUALITY_DEF} It defines the
heuristics they cite. You see each finding's claim only.
For each, decide at ${AS_OF}:
- "confirmed": the structure is still there as described.
- "weakened": still there but materially smaller or partly mitigated.
- "fixed": it existed at ${SINCE} (check `git show ${SINCE}:<path>`) and is gone at ${AS_OF}. Find the
  commit that removed it if you can (`git log --format='%H %s' ${SINCE}..${AS_OF} -S'<symbol>' -- <path>`)
  and give it as fixed_by.
- "refuted": it was never true as described, even at ${SINCE}.
Code moves: if the structure now lives under another path or symbol, give that as `anchor`. Never mark a
finding fixed because its file moved. When a finding has no heuristic tag, name the one its statement
rests on as primary_tag. Cite path::symbol in notes. One outcome per finding, ref verbatim.

FINDINGS (JSON):
${JSON.stringify(batch, null, 1)}
```

### Synthesis prompt

```
You are the synthesis lead for a bug-hotspot survey of the <project> repo at ${ROOT}, commit ${AS_OF}.
READ-ONLY; verify in code where needed. First: ${QUALITY_DEF} Reason in its heuristics.
Below: verified findings from this run's reviewers, the re-verified still-open findings from earlier
runs, and the mined fix themes. Your jobs:
1. CROSS-SYSTEM CHAINS: defect→ad-hoc-patch chains SPANNING areas that no single reviewer saw whole.
   Use cross_system_notes as leads; spot-check both ends in code. Merge findings that describe one root
   cause from different sides, and say which merge.
2. TOP PRIORITIES: rank 5-8 remedies by payoff × feasibility, where payoff is the findings (by severity)
   and compensations a remedy removes. Historical fix counts may describe reach; they never rank, and
   "the hotspot will cool" is not a payoff. Resolve overlapping or conflicting proposals into one.
3. CONTRADICTIONS: proposals that conflict with each other or with mechanisms that already exist.
Cite members by the ref (earlier findings) or id (this run's, cluster.n) exactly as given.

MINED FIX THEMES: ${JSON.stringify(allThemes, null, 1)}
STILL-OPEN EARLIER FINDINGS: ${JSON.stringify(carried, null, 1)}
THIS RUN'S VERIFIED FINDINGS: ${JSON.stringify(digest, null, 1)}
```

## Execution skeleton

```js
const MINER_PROMPTS = [MINE_TASKS_PROMPT, MINE_GIT_PROMPT, MINE_PLANS_PROMPT]
const MINER_LABELS = ['mine:tasks', 'mine:git', 'mine:plans']
const byCluster = (xs) => SUBSYSTEM_KEYS.map(k => xs.filter(x => x.sub_area === k)).filter(g => g.length)
const chunk = (xs, n) => Array.from({ length: Math.ceil(xs.length / n) }, (_, i) => xs.slice(i * n, i * n + n))
const reverifyBatches = byCluster(PRIOR).flatMap(g => chunk(g, 8))

phase('Mine')  // miners and prior-finding re-verification are independent: one parallel stage
const stage1 = await parallel([
  ...MINER_PROMPTS.map((p, i) => () => agent(p, { label: MINER_LABELS[i], phase: 'Mine', model: 'sonnet', effort: 'medium', schema: MINING_SCHEMA })),
  ...reverifyBatches.map((b, i) => () => agent(reverifyPrompt(b), { label: `reverify:${b[0].sub_area}:${i}`, phase: 'Mine', model: 'sonnet', effort: 'medium', schema: REVERIFY_SCHEMA })),
])
const reverified = stage1.slice(MINER_PROMPTS.length).flatMap(r => (r && r.outcomes) || [])

// SEMANTIC validation of miners; schema validation is not enough (Failure modes 1).
function degenerate(m) {
  if (!m || !Array.isArray(m.themes)) return true
  const real = m.themes.filter(t => t.theme && t.evidence && t.evidence.length > 40 &&
                                    !/^test( |$)/i.test(t.theme) && !/^test( |$)/i.test(t.evidence))
  return real.length < 3
}
const mined = []
for (let i = 0; i < MINER_PROMPTS.length; i++) {
  let m = stage1[i]
  if (degenerate(m)) {
    log(`miner ${i} degenerate — re-running once`)
    m = await agent(MINER_PROMPTS[i] + '\n\nNOTE: a previous attempt produced empty/placeholder output. Every theme MUST be a real, evidenced entry in the themes array.',
                    { label: `mine:retry:${i}`, phase: 'Mine', model: 'sonnet', effort: 'medium', schema: MINING_SCHEMA })
  }
  if (!degenerate(m)) mined.push(m)
  else log(`miner ${i} lost — proceeding without that lane (record in method header)`)
}
const allThemes = mined.flatMap(m => m.themes || [])
const outcomeOf = new Map(reverified.map(o => [o.ref, o]))
const carried = PRIOR.filter(p => { const o = outcomeOf.get(p.ref); return o && ['confirmed', 'weakened'].includes(o.outcome) })

function briefFor(key) {
  const t = allThemes.filter(x => x.subsystem === key)
  if (!t.length) return '(no mined themes tagged to this subsystem — read git log yourself)'
  return t.map(x => `- [${x.count_estimate}x] ${x.theme}: ${x.evidence}`).join('\n')
}
function knownFor(key) {
  const k = carried.filter(p => p.sub_area === key)
  if (!k.length) return '(none)'
  return k.map(p => `- ${p.ref} [${p.tags[0] || 'untagged'}] ${p.title} @ ${(outcomeOf.get(p.ref).anchor || p.anchor)}: ${p.statement}`).join('\n')
}

phase('Review')
const clusterResults = await pipeline(
  CLUSTERS.filter(c => c.review),
  (c) => {
    const opts = { label: `review:${c.key}`, phase: 'Review', effort: 'high', schema: REVIEW_SCHEMA }
    if (c.model) opts.model = c.model
    return agent(reviewPrompt(c), opts)
  },
  (review, c) => {
    if (!review) return null
    const findings = (review.findings || []).map((f, n) => Object.assign(f, { id: `${c.key}.${n + 1}` }))
    if (!findings.length) return { cluster: c.key, review, verdicts: [] }
    const blinded = findings.slice(0, 8).map(({ proposal, ...claim }) => claim)
    return agent(skepticPrompt(c, blinded), { label: `verify:${c.key}`, phase: 'Verify', model: 'sonnet', effort: 'medium', schema: SKEPTIC_SCHEMA })
      .then(v => ({ cluster: c.key, review, verdicts: (v && v.verdicts) || [] }))
  }
)

// Join verdicts by id; an unmatched id stays 'unverified' (never dropped). Strip refuted before synthesis.
const clusters = clusterResults.filter(Boolean)
for (const cr of clusters) {
  const byId = new Map(cr.verdicts.map(v => [v.id, v]))
  for (const f of (cr.review.findings || [])) {
    const v = byId.get(f.id)
    Object.assign(f, v ? { verdict: v.verdict, verdict_notes: v.notes, verified_anchor: v.anchor, verified_primary_tag: v.primary_tag }
                       : { verdict: 'unverified', verdict_notes: '' })
  }
}
const digest = clusters.map(cr => ({ cluster: cr.cluster, ...cr.review,
  findings: (cr.review.findings || []).filter(f => f.verdict !== 'refuted') }))

phase('Synthesize')
const cross = await agent(SYNTHESIS_PROMPT, { label: 'synthesize:cross-system', phase: 'Synthesize', effort: 'high', schema: CROSS_SCHEMA })

return { themes: allThemes, lanes_run: mined.length, reverified, clusters, cross_system: cross }
```

The workflow returns raw material only; keys, dispositions and the artefact shape are made by the coordinator's digest (`SKILL.md` §Phase 5), which imports `finding_key` from `scripts/findings_artefact.py` rather than re-implementing contract §2.

## Model / effort allocation

| Stage | Model | Effort | Observed per agent (07-06 run) |
|---|---|---|---|
| Mine ×3 | sonnet | medium | 59–142k tok, 5–22 min (full window; a refresh window is smaller) |
| Re-verify, one per ≤8 prior findings | sonnet | medium | not yet observed; budget as a skeptic |
| Review, core clusters | session default | high | 102–138k tok, 9–19 min |
| Review, peripheral | sonnet | high | 97–113k tok, 12–28 min |
| Verify, one per review | sonnet | medium | 56–91k tok, 4–13 min |
| Synthesize ×1 | session default | high | ~141k tok, ~5 min |

"Peripheral" is a Phase 0 judgment call: lower fix density, smaller blast radius, less entanglement.

## Failure modes (each observed in a real run or measurement; the mitigation is baked in)

1. **Degenerate structured output passes schema validation.** The DF run's mine:plans agent put everything in `summary`, then after two schema rejections emitted `{"themes":[{"theme":"test theme","evidence":"test evidence",...}]}`, which validated and silently poisoned two downstream prompts. Mitigations: the IMPORTANT paragraph, `degenerate()` + one re-run, never treating `filter(Boolean)` as validation.
2. **The workflow result is too big to read raw** (~270KB). Digest via Python against the full output file; a digest can itself overflow into a tool-result file, so read that in chunks.
3. **Verdict-count conflation.** "N/N survived" counted `!refuted`; the true split was 72 confirmed / 3 weakened / 0 refuted. Count the classes separately; the conformance check now fails a header whose counts disagree with the findings.
4. **Skeptic join fragility.** The proven run joined verdicts by verbatim title; joins are now by display id, and an unmatched id stays `unverified`. Keep the per-skeptic cap (8) aligned with the reviewer's "3–8 findings".
5. **Zero refutations is a yellow flag.** 12 skeptics, 0 refutations of 75 in the DF run. The skeptic and re-verifier are now blind to `proposal`. If refutations are still zero, say so in the method header's prose rather than presenting it as confidence.
6. **Post-survey filing races** (the hand-off turn): concurrent /prd sessions in one checkout sweep each other's staged files, so `git commit --only <path>`; verify batches with `get_task`, not search.
7. **The fix signal drifts with commit conventions.** The proven run's grep (`-i fix|bug|amend|regression`, full message, every commit) matched 11% of DF commits at the survey and 46% in the window after it, while the subject-prefix ratio over source paths moved from 13% to 9.5%: the earlier denominator held ~30k task-bookkeeping auto-commits that later stopped, and `amend:` (pre-merge review amendments) grew from 9.5% to 15.3% of source commits. Hence the subject-prefix definition in `SKILL.md` §Phase 0 and the raw ratio reported beside it.
8. **File-grain anchors collide.** Minting the 2026-07-06 findings at file grain put 37 of 75 findings under 17 shared keys (keys over the plain area). Anchors must name the enclosing symbol, which is why both the skeptic and the re-verifier return one.
