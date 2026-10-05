# Orchestration — workflow template, args, schemas, prompts, failure modes

The Workflow-tool script for phases 3–4 of `/review-all`, in the house pattern of `skills/hotspot-survey/references/orchestration.md`. Phases 0–2 and 5–6 run in the lead's session (SKILL.md); the script receives their results as `args` and returns one JSON value the lead digests. Field vocabularies (area, anchor, tags, severity, evidence_source, verdict) are those of `docs/quality-findings-contract.md` §1–§4 and are not redefined here.

## Design notes

- Keys are minted in the lead's Phase 5 digest by importing the one key implementation — `shared/src/shared/finding_key.py` once it lands, until then `skills/hotspot-survey/scripts/findings_artefact.py` (`finding_key`, `normalise_anchor`) — never by a second implementation; the digest must agree with that module's test fixture. Inside the script the join identity is the contract §2 `canonical` string, whose normalisation mirrors that module's, so no hashing is needed. Same identity, two representations.
- A prior finding re-observed through `prior_canonical` is recorded under the prior's canonical; when the new anchor differs, the new finding carries `supersedes_canonical` (contract §2 anchor-moved rule) and the digest fills `supersedes` with the prior key. Every prior open finding assigned to a slice comes back with a verdict or is listed under `observation.unchecked`.
- A seat reviews a *slice* (an allocation of files); a finding carries the briefing *area* of its anchor. Slices never appear in findings.
- Line counts set a seat's read budget and nothing else; ranking inside the budget uses the §What to measure instruments. The `Do not steer by` list is enforced twice: in every prompt and by the critic.
- The skeptic sees statements and measurements, never proposals (hotspot failure mode 5, now the default rather than the fallback).
- Cross-area seats run behind a barrier because they need every area seat's canonical set to dedup against — one of the three justified barrier cases in the Workflow reference.
- The critic is a second reader of the join, not a third reviewer of the code. Its error classes are the ones `skills/team/SKILL.md` §4 names, plus the quality doc's two measurement traps (one term of the complexity pair; a do-not-steer measure as severity input).
- `routing` arrives in `args` so the fable provenance rule (`skills/team/SKILL.md` §Fable) is decided by the lead, outside the script.
- The quality definition reaches every judging prompt one of the contract §10's two ways: `Read docs/code-quality.md` when the tree carries it, else the lead embeds the `code_quality.py::guidance` render as `args.quality_guidance` and the `QUALITY` clause carries it; the prompts never restate it.

## `args` shape

Written by the lead to `<scratch>/args.json` after Phase 1 and passed as a JSON value (never a string).

```js
// JSON Schema for args
const ARGS_SCHEMA = {
  type: 'object',
  required: ['run_id', 'as_of_sha', 'since', 'project', 'root', 'scratch', 'areas', 'slices', 'cross', 'metrics_summary', 'import_graph_path', 'prior', 'routing', 'ceilings'],
  properties: {
    run_id: { type: 'string', description: 'review-all-<project_id>-<YYYYMMDD>[-<n>]' },
    as_of_sha: { type: 'string' },
    since: { type: 'string', description: "previous run's as_of_sha or 'none'" },
    project: { type: 'object', required: ['id', 'main_root', 'plans_dir'], properties: {
      id: { type: 'string' }, main_root: { type: 'string' }, plans_dir: { type: 'string' } } },
    root: { type: 'string', description: 'detached read-only worktree at as_of_sha; every seat reads here' },
    scratch: { type: 'string', description: '<session scratchpad>/review-all/<run_id>' },
    areas: { type: 'array', items: { type: 'string' }, description: "briefing subprojects keys + 'repo'" },
    slices: { type: 'array', items: { type: 'object',
      required: ['seat', 'area', 'read_fully', 'index_only', 'tests_reaching_internals', 'size_alarms', 'prior_open', 'leads'],
      properties: {
        seat: { type: 'string', description: '<area>/<slice-slug>, or <area> when the area is one slice' },
        area: { type: 'string' },
        read_fully: { type: 'array', items: { type: 'object', required: ['path', 'lines', 'prose_share', 'cognitive_max', 'cognitive_total', 'fan_in', 'fan_out', 'reach_back_imports', 'reexports', 'patch_targets'],
          properties: { path: { type: 'string' }, lines: { type: 'integer' }, prose_share: { type: 'number' }, cognitive_max: { type: 'integer' }, cognitive_total: { type: 'integer' }, fan_in: { type: 'integer' }, fan_out: { type: 'integer' }, reach_back_imports: { type: 'integer' }, reexports: { type: 'integer' }, patch_targets: { type: 'integer' } } } },
        index_only: { type: 'array', items: { type: 'string' } },
        tests_reaching_internals: { type: 'array', items: { type: 'object', required: ['test_path', 'targets'], properties: { test_path: { type: 'string' }, targets: { type: 'array', items: { type: 'string' } } } } },
        size_alarms: { type: 'array', items: { type: 'string' }, description: 'files over ceilings.alarm_lines; heuristic-14 protocol is mandatory for each' },
        prior_open: { type: 'array', items: { type: 'object', required: ['canonical', 'key', 'statement', 'severity', 'source_run'],
          properties: { canonical: { type: 'string' }, key: { type: 'string' }, statement: { type: 'string' }, severity: { type: 'string' }, source_run: { type: 'string' },
            sub_area: { type: 'string', description: 'contract §1 sub_area; a legacy <area>/<sub> hotspot area is split in Phase 0 into the plain area (in the key) and this field, and the finding is assigned to the slice holding the sub-area files' } } },
          description: 'every prior open finding of this area assigned to this slice; each must come back in prior_verdicts — the script lists any that do not' },
        leads: { type: 'array', items: { type: 'object', required: ['source', 'text'], properties: { source: { type: 'string', enum: ['hotspot', 'review', 'codebook', 'memory'] }, text: { type: 'string' } } } },
      } } },
    cross: { type: 'array', items: { type: 'object', required: ['lens', 'tag', 'title', 'question'],
      properties: { lens: { type: 'string' }, tag: { type: 'string' }, title: { type: 'string' }, question: { type: 'string' } } } },
    metrics_summary: { type: 'string', description: 'whole-repo Phase 1 snapshot rendered as ≤4k chars of tables' },
    quality_guidance: { type: ['string', 'null'], description: "null when <root>/docs/code-quality.md exists (prompts say Read it); otherwise the render of orchestrator/src/orchestrator/agents/code_quality.py::guidance, embedded verbatim (contract §10)" },
    import_graph_path: { type: 'string', description: 'JSON file under scratch: {edges:[[from,to]], reach_back:[...], deferred:[...], cycles:[[...]]}' },
    prior: { type: 'object', required: ['open', 'standing'], properties: {
      open: { type: 'array', items: { type: 'object', required: ['canonical', 'key', 'area', 'statement', 'severity', 'source_run', 'disposition'] } },
      standing: { type: 'array', items: { type: 'object', required: ['canonical', 'key', 'area', 'statement', 'disposition'] } } } },
    routing: { type: 'object', required: ['area', 'skeptic', 'cross', 'synthesis', 'critic'],
      additionalProperties: { type: 'object', required: ['model', 'effort'], properties: { model: { type: 'string' }, effort: { type: 'string' } } } },
    ceilings: { type: 'object', required: ['soft_lines', 'alarm_lines', 'max_findings_per_seat'],
      properties: { soft_lines: { type: 'integer' }, alarm_lines: { type: 'integer' }, max_findings_per_seat: { type: 'integer' } } },
  },
}
```

Default `cross` lenses (an overlay may add, never remove):

| lens | tag | question |
|---|---|---|
| `spot` | `h11` | Which facts, definitions and policies live in more than one module with no derivation edge between the copies, so they can disagree? Map onto INV-5/INV-9 where one applies. |
| `layering` | `h9` | Which modules import upward or sideways in cycles, which are shallow (interface ≈ implementation), and where is a stratum missing? Reach-back and deferred imports in the import graph are the leads. |
| `dimensions` | `h3` | Which independent axes of variability are handled by inline flag conjunctions, mode strings or kwargs threaded through several modules instead of one mechanism per axis? |
| `stateless` | `h7` | Which cross-module interactions read another module's attributes, depend on the order of prior calls, or are reached by tests patching into another module's privates? |

## The script

```js
export const meta = {
  name: 'review-all',
  description: 'Whole-project quality review against docs/code-quality.md: area seats, blinded skeptics, cross-area lenses, synthesis, critic',
  phases: [
    { title: 'Area review', detail: 'one seat per area slice; all fourteen heuristics and both stances', model: 'opus' },
    { title: 'Verify', detail: 'one skeptic per seat, blinded to proposals', model: 'sonnet' },
    { title: 'Cross-area', detail: 'h11 SPOT, h9 layering, h3 orthogonal dimensions, h7 stateless interactions', model: 'opus' },
    { title: 'Synthesize', detail: 'join prior and new findings by canonical; area x heuristic ranked view; candidate streams' },
    { title: 'Critique', detail: 'a critic reads the assembled synthesis for join errors' },
  ],
}

const A = args
const ROOT = A.root
const MAX = A.ceilings.max_findings_per_seat
const TAG_PATTERN = '^(h([1-9]|1[0-4])|comments|tests|inv-[0-9]+|kind:[a-z-]+)$'

const MEASUREMENT = { type: 'object', required: ['name', 'value', 'rule'], properties: {
  name: { type: 'string' },
  value: { type: 'string', description: 'the number or list, as text' },
  rule: { type: 'string', description: 'what was counted, what was excluded, over what population' },
} }

const FINDING = { type: 'object',
  required: ['display_id', 'area', 'anchor', 'tags', 'severity', 'evidence_source', 'statement', 'measurements', 'next_change', 'class'],
  properties: {
    display_id: { type: 'string', description: '<seat>.<n>' },
    area: { type: 'string', enum: A.areas },
    anchor: { type: 'string', description: 'path/to/file.py::symbol | path/to/file.py | slug:<kebab>; repo-relative, no line numbers' },
    tags: { type: 'array', minItems: 1, items: { type: 'string', pattern: TAG_PATTERN }, description: 'first entry is the primary heuristic' },
    severity: { type: 'string', enum: ['high', 'medium', 'low'] },
    evidence_source: { type: 'array', minItems: 1, items: { type: 'string', enum: ['present-tree', 'fix-history', 'agent-transcripts', 'metrics'] } },
    statement: { type: 'string', description: 'the claim with its measured facts; no proposal here' },
    measurements: { type: 'array', minItems: 1, items: MEASUREMENT },
    next_change: { type: 'string', description: 'the change an agent would plausibly make here, what it must read or know beyond the file, and what catches a wrong version' },
    proposal: { type: 'string', description: 'what would remove the cost; omit when you have none' },
    class: { type: 'string', enum: ['mechanical', 'structural'] },
    prior_canonical: { type: 'string', description: 'the canonical of the prior open finding this adds evidence to; the pipeline records that prior as re-observed, and when your anchor differs from it, records a supersedes link from this finding to it' },
  } }

const PRIOR_VERDICT = { type: 'object', required: ['canonical', 'status', 'checked'], properties: {
  canonical: { type: 'string' },
  status: { type: 'string', enum: ['still_present', 'gone', 'not_checked'] },
  checked: { type: 'string', description: 'what you read or ran; for gone, the commit or replacing code' },
} }

const H14 = { type: 'object', required: ['path', 'prose_share', 'prose_home_ok', 'partition', 'stays_whole'], properties: {
  path: { type: 'string' },
  prose_share: { type: 'number' },
  prose_home_ok: { type: 'boolean', description: 'false when prose belongs in docs/, a PRD, a test or a commit message instead' },
  partition: { type: 'array', items: { type: 'object', required: ['symbols', 'symptom'], properties: {
    symbols: { type: 'array', items: { type: 'string' } },
    symptom: { type: 'string', description: "'none', or the heuristic-13 symptom naming the two modules and the specific import that would close the cycle or reach back" },
  } } },
  stays_whole: { type: 'boolean' },
} }

const AREA_SCHEMA = { type: 'object',
  required: ['seat', 'area', 'read_coverage', 'area_health', 'prior_verdicts', 'findings', 'h14_measurements', 'seams_for_cross_area'],
  properties: {
    seat: { type: 'string' },
    area: { type: 'string', enum: A.areas },
    read_coverage: { type: 'object', required: ['files_read_fully', 'files_indexed', 'lines_read'], properties: {
      files_read_fully: { type: 'array', items: { type: 'string' } },
      files_indexed: { type: 'array', items: { type: 'string' } },
      lines_read: { type: 'integer' } } },
    area_health: { type: 'string', description: 'how the slice is structured today and where the next-change cost sits; >=200 chars' },
    prior_verdicts: { type: 'array', items: PRIOR_VERDICT },
    findings: { type: 'array', maxItems: MAX, items: FINDING },
    h14_measurements: { type: 'array', items: H14 },
    seams_for_cross_area: { type: 'string', description: 'duplicated policies, layering doubts, flag axes, attribute reaches you suspect but could not anchor inside this slice' },
  } }

const SKEPTIC_SCHEMA = { type: 'object', required: ['verdicts'], properties: {
  verdicts: { type: 'array', items: { type: 'object', required: ['display_id', 'verdict', 'checked', 'notes'], properties: {
    display_id: { type: 'string', description: 'the finding display_id, or the prior canonical for a gone verdict' },
    verdict: { type: 'string', enum: ['confirmed', 'weakened', 'refuted'] },
    checked: { type: 'string', description: 'the files opened and the commands run' },
    notes: { type: 'string', description: 'what was found; for weakened/refuted, which fact fails' },
  } } } } }

const CROSS_SCHEMA = { type: 'object', required: ['lens', 'findings', 're_observations', 'method'], properties: {
  lens: { type: 'string' },
  findings: { type: 'array', maxItems: MAX, items: FINDING },
  re_observations: { type: 'array', items: { type: 'object', required: ['canonical', 'added_evidence'], properties: {
    canonical: { type: 'string' }, added_evidence: { type: 'string' } } } },
  method: { type: 'string', description: 'what was read and computed, with counts' },
} }

const SYNTHESIS_SCHEMA = { type: 'object',
  required: ['matrix', 'streams', 'invariant_candidates', 'contradictions', 'exonerations', 'method'],
  properties: {
    matrix: { type: 'array', items: { type: 'object', required: ['area', 'tag', 'ranked'], properties: {
      area: { type: 'string' }, tag: { type: 'string' },
      ranked: { type: 'array', items: { type: 'string' }, description: 'canonicals, best payoff first' } } } },
    streams: { type: 'array', items: { type: 'object',
      required: ['name', 'canonicals', 'remedy', 'deletable', 'size', 'risk', 'mode', 'seam_candidates', 'open_choices'],
      properties: {
        name: { type: 'string' },
        canonicals: { type: 'array', items: { type: 'string' } },
        remedy: { type: 'string', description: 'the named mechanism: module, type, chokepoint, invariant and where enforced' },
        deletable: { type: 'string', description: 'existing code the remedy makes deletable' },
        size: { type: 'string', enum: ['S', 'M', 'L'] },
        risk: { type: 'string', enum: ['low', 'medium', 'high'] },
        mode: { type: 'string', enum: ['agent', 'spawn'], description: 'agent: mechanical /prd by an agent team; spawn: design-heavy interactive /prd session' },
        seam_candidates: { type: 'array', items: { type: 'string' }, description: 'shared artefacts that need one owner (G4)' },
        open_choices: { type: 'array', items: { type: 'object', required: ['question', 'options'], properties: {
          question: { type: 'string' },
          options: { type: 'array', minItems: 2, items: { type: 'object', required: ['option', 'tradeoff', 'long_term'], properties: {
            option: { type: 'string' }, tradeoff: { type: 'string' }, long_term: { type: 'string', description: 'what it costs the next agent change a year on' } } } } } } } } } },
    invariant_candidates: { type: 'array', items: { type: 'object', required: ['statement', 'heuristic', 'checkable_form', 'fixture_sketch'], properties: {
      statement: { type: 'string' }, heuristic: { type: 'string' }, checkable_form: { type: 'string' }, fixture_sketch: { type: 'string' } } } },
    contradictions: { type: 'array', items: { type: 'object', required: ['between', 'conflict', 'resolution_or_question'], properties: {
      between: { type: 'array', items: { type: 'string' } }, conflict: { type: 'string' }, resolution_or_question: { type: 'string' } } } },
    exonerations: { type: 'string', description: 'areas or files the metrics flag that the reading shows healthy, and why; empty string if none' },
    method: { type: 'string', description: 'the ranking rule used, stated once' },
  } }

const CRITIC_SCHEMA = { type: 'object', required: ['join_errors', 'verdict', 'notes'], properties: {
  join_errors: { type: 'array', items: { type: 'object', required: ['kind', 'where', 'detail', 'fix'], properties: {
    kind: { type: 'string', enum: ['inconsistent-rule', 'one-term-of-sum', 'evidence-contradicts-conclusion', 'do-not-steer-leak', 'duplicate-mechanism', 'unverified-fixed', 'severity-unsupported', 'area-unmapped', 'weakened-hidden'] },
    where: { type: 'string', description: 'canonical, stream name or matrix cell' },
    detail: { type: 'string' }, fix: { type: 'string' } } } },
  verdict: { type: 'string', enum: ['accept', 'revise'] },
  notes: { type: 'string' },
} }

const QUALITY = A.quality_guidance
  ? `The quality definition this review judges by follows, rendered from the normative docs/code-quality.md (this project carries no copy). Judge by its Definition:\n\n${A.quality_guidance}\n\n`
  : `Read ${ROOT}/docs/code-quality.md first and judge by its Definition: `
const QUALITY_REF = A.quality_guidance ? 'the quality definition above' : `${ROOT}/docs/code-quality.md`

// ANCHOR_RE, LINE_PIN_RE and normAnchor mirror skills/hotspot-survey/scripts/findings_artefact.py
// (ANCHOR_RE, LINE_PIN_RE, normalise_anchor) — the interim key reference of contract §2; the
// lead's Python digest imports that module (or shared/finding_key.py once landed) to mint keys.
const ANCHOR_RE = /^(slug:[a-z0-9]+(-[a-z0-9]+)*|[\w.-][\w./-]*(::[A-Za-z_][\w.]*)?)$/
const LINE_PIN_RE = /(\.[A-Za-z0-9]{1,8}):\d+(?:[-,]\d+)*/g
function normAnchor(a) {
  let s = String(a || '').trim().replace(/\\/g, '/')
  while (s.startsWith('./')) s = s.slice(2)
  return s.replace(LINE_PIN_RE, '$1')
}
function canonical(f) {
  return `${f.area}|${normAnchor(f.anchor)}|${(f.tags || [])[0] || ''}`.toLowerCase()
}
// Mechanical guard only. The terms are the measures docs/code-quality.md §"Do not steer by" names;
// that section, not this regex, is the rule.
const STEER_GUARD = /\b(line count|lines? of code|LOC|average complexity|mean complexity|coverage|test count|test-to-code)\b/i

function findingProblems(f, seat) {
  const p = []
  if (!ANCHOR_RE.test(normAnchor(f.anchor))) p.push('anchor shape')
  if ((f.statement || '').length < 120) p.push('statement too short')
  if (/^test\b/i.test(f.statement || '') || /^test\b/i.test(f.next_change || '')) p.push('placeholder')
  if (!(f.measurements || []).length) p.push('no measurement')
  if ((f.next_change || '').length < 60) p.push('no next-change scenario')
  if ((f.tags || [])[0] === 'h14') {
    const m = (seat.h14_measurements || []).find(x => normAnchor(x.path) === normAnchor(f.anchor).split('::')[0])
    if (!m) p.push('h14 without measurement protocol')
  }
  if (f.severity === 'high' && STEER_GUARD.test(f.statement || '') && (f.measurements || []).every(m => STEER_GUARD.test(m.name || ''))) p.push('severity rests on a do-not-steer measure')
  return p
}

const dropped = []
function cleanSeat(r, label) {
  if (!r || !Array.isArray(r.findings)) return null
  const kept = []
  for (const f of r.findings) {
    const problems = findingProblems(f, r)
    if (problems.length) { dropped.push({ seat: label, display_id: f.display_id, anchor: f.anchor, problems }); log(`${label}: dropped ${f.display_id} (${problems.join('; ')})`) }
    else { f.anchor = normAnchor(f.anchor); f.canonical = canonical(f); kept.push(f) }
  }
  return { ...r, findings: kept }
}
function degenerateSeat(r, raw) {
  if (!r) return true
  if ((r.area_health || '').length < 200) return true
  const rawCount = ((raw && raw.findings) || []).length
  return rawCount > 0 && r.findings.length === 0
}

function prettyFiles(list) {
  return list.map(f => `- ${f.path} (${f.lines} lines, prose ${Math.round(f.prose_share * 100)}%, cognitive max ${f.cognitive_max} / total ${f.cognitive_total}, fan-in ${f.fan_in}, fan-out ${f.fan_out}, reach-back imports ${f.reach_back_imports}, re-exports ${f.reexports}, test patch targets ${f.patch_targets})`).join('\n')
}

function areaPrompt(s) {
  return `You are one seat of a whole-project quality review of ${A.project.id}. The tree is ${ROOT}, a detached checkout of ${A.as_of_sha}. READ-ONLY: modify nothing, run no state-changing command, make no MCP write. Intermediates go under ${A.scratch}/seats/${s.seat}/.

${QUALITY}quality is the expected cost and risk of the next change, made by an agent from a partial view. The fourteen heuristics and the two stances are lenses you name on each finding (tags), not a checklist to walk; one finding per heuristic is the failure mode this prompt exists to prevent.

YOUR SLICE: ${s.seat} (findings carry area "${s.area}"). The reading list was ranked by the instruments in that doc's "What to measure" table; line counts set only how much fits in your budget.

READ FULLY, in this order:
${prettyFiles(s.read_fully)}

INDEX ONLY (module docstring + grep -n 'def \\|class ' listing; open a symbol only when a fully-read file leads you there):
${s.index_only.map(p => `- ${p}`).join('\n') || '- none'}

TESTS THAT REACH THIS SLICE'S INTERNALS (Tests stance; read the patch and attribute sites, not whole test files):
${s.tests_reaching_internals.map(t => `- ${t.test_path} -> ${t.targets.join(', ')}`).join('\n') || '- none measured'}

SIZE ALARMS (heuristic 14 protocol is mandatory for each; fill h14_measurements — prose share and whether the prose is in its right home, a named partition of top-level symbols, and per candidate the heuristic-13 symptom naming the two modules and the specific import, or none):
${s.size_alarms.map(p => `- ${p}`).join('\n') || '- none'}

PRIOR OPEN FINDINGS AT THIS AREA (re-verify each at this tree; report still_present or gone in prior_verdicts — a gone needs the commit or the replacing code; set prior_canonical on a new finding only when it adds evidence to one of these rather than restating it):
${s.prior_open.map(p => `- ${p.canonical}${p.sub_area ? ` (sub-area ${p.sub_area})` : ''} [${p.severity}, ${p.source_run}] ${p.statement}`).join('\n') || '- none'}

LEADS (fix-history themes, codebook confusions, operator memory — things to verify, never findings in themselves):
${s.leads.map(l => `- (${l.source}) ${l.text}`).join('\n') || '- none'}

METHOD
1. Map: symbol index, module docstrings, top-level state, and this slice's rows in the import graph at ${A.import_graph_path}.
2. Read fully in rank order. Follow a seam out of the slice only far enough to anchor a claim.
3. For every candidate, write next_change first: the change an agent would plausibly make here, what it must read or know beyond the file to make it correctly, and what catches a wrong version before it lands. If that cannot be written concretely, it is not a finding.
4. Measure: at least one measurement per finding, each with the rule it was computed under (what was counted, excluded, over what population). Re-run a metric you rely on rather than quoting the sheet.
5. Anchor at path::symbol (the enclosing top-level or class-level definition) at this tree; no line numbers.
6. Severity per docs/quality-findings-contract.md §4; class mechanical only when one anchor, no design choice, and fixable from the statement — any split or reduce-this-number proposal is structural.

NOT A FINDING: a style nit; a heuristic that could apply; anything whose severity rests on raw line count, average complexity, coverage or test count (the doc's "Do not steer by" — context only); a restatement of a prior open finding; a proposal without a statement; a cost you did not read the code for.

Emit at most ${MAX} findings ranked by next-change cost removed. Fewer is correct when the slice is healthy: say what you read and why it is healthy in area_health. Put the cross-module suspicions you could not anchor inside the slice in seams_for_cross_area. Record read coverage honestly. Never emit placeholder content to satisfy the schema.`
}

function skepticPrompt(label, findings, gone) {
  return `You are an adversarial verifier for a whole-project quality review of ${A.project.id} at ${ROOT} (detached checkout of ${A.as_of_sha}). READ-ONLY.

${A.quality_guidance ? QUALITY : ''}A reviewer judged "${label}" against ${QUALITY_REF} and produced the findings below as statements with measurements. You are deliberately not shown their proposals or severities. Try to REFUTE each.
- refuted: the anchor does not exist at this tree; or a measured fact is wrong by enough to remove the claim; or the cost is already removed — a derivation, guard, test or mechanism exists that the statement implies is missing (grep for it before agreeing it is missing); or the claim rests on a "Do not steer by" measure of the quality doc.
- weakened: the structure is real but a measurement is materially off, an existing mechanism covers most of the cost, or the next_change scenario does not support the claim's weight.
- confirmed: only after opening the anchor and reproducing at least one measurement yourself (say what you ran).
Check the code, not the prose. For duplication claims open every named site. One verdict per display_id.

${gone.length ? `The reviewer also marked these prior findings GONE. Confirm each by locating the change (git log -S, the replacing code); an unlocatable gone is refuted, meaning the prior finding stands. Use the canonical as display_id.\n${JSON.stringify(gone, null, 1)}\n` : ''}
FINDINGS:
${JSON.stringify(findings, null, 1)}`
}

function blind(findings) {
  return findings.map(f => ({ display_id: f.display_id, area: f.area, anchor: f.anchor, tags: f.tags, evidence_source: f.evidence_source, statement: f.statement, measurements: f.measurements, next_change: f.next_change }))
}

async function reviewSeat(s) {
  const opts = { label: `review:${s.seat}`, phase: 'Area review', model: A.routing.area.model, effort: A.routing.area.effort, schema: AREA_SCHEMA }
  let raw = await agent(areaPrompt(s), opts)
  let r = cleanSeat(raw, s.seat)
  if (degenerateSeat(r, raw)) {
    log(`${s.seat}: degenerate seat output — re-running once`)
    raw = await agent(areaPrompt(s) + '\n\nNOTE: a previous attempt returned empty, placeholder or unanchored output. Every finding needs a real anchor at this tree, a measurement with its rule, and a next-change scenario; a healthy slice needs a substantive area_health.', { ...opts, label: `review:retry:${s.seat}` })
    r = cleanSeat(raw, s.seat)
  }
  if (degenerateSeat(r, raw)) { log(`${s.seat}: seat lost — recorded in dropped`); dropped.push({ seat: s.seat, problems: ['seat degenerate twice'] }); return null }
  return r
}

async function verifySeat(r, label) {
  if (!r) return null
  const gone = (r.prior_verdicts || []).filter(v => v.status === 'gone')
  if (!r.findings.length && !gone.length) return { ...r, verdicts: [] }
  const v = await agent(skepticPrompt(label, blind(r.findings), gone), { label: `verify:${label}`, phase: 'Verify', model: A.routing.skeptic.model, effort: A.routing.skeptic.effort, schema: SKEPTIC_SCHEMA })
  return { ...r, verdicts: (v && v.verdicts) || [] }
}

function applyVerdicts(r) {
  const by = new Map(r.verdicts.map(v => [v.display_id, v]))
  for (const f of r.findings) {
    const v = by.get(f.display_id)
    f.verdict = v ? v.verdict : 'unverified'
    f.verdict_notes = v ? `${v.checked} — ${v.notes}` : 'no verdict matched this display_id'
  }
  for (const p of r.prior_verdicts || []) {
    if (p.status !== 'gone') continue
    const v = by.get(p.canonical)
    p.gone_verdict = v ? v.verdict : 'unverified'
    p.gone_notes = v ? v.notes : ''
  }
  return r
}

const RETRY_NOTE = '\n\nNOTE: a previous attempt returned empty, placeholder or unanchored output. Every entry must be real and evidenced; never emit placeholder content to satisfy the schema.'
async function seatWithRetry(label, call, isDegenerate) {
  let r = await call('')
  if (isDegenerate(r)) { log(`${label}: degenerate output — re-running once`); r = await call(RETRY_NOTE) }
  if (isDegenerate(r)) { log(`${label}: lost after two attempts`); dropped.push({ seat: label, problems: ['degenerate twice'] }); return null }
  return r
}

phase('Area review')
const seats = (await pipeline(
  A.slices,
  (s) => reviewSeat(s),
  (r, s) => verifySeat(r, s.seat),
)).filter(Boolean).map(applyVerdicts)

const unchecked = []
for (const s of A.slices) {
  const r = seats.find(x => x.seat === s.seat)
  const answered = new Set(((r && r.prior_verdicts) || []).map(p => p.canonical))
  for (const p of s.prior_open) if (!answered.has(p.canonical)) {
    unchecked.push({ seat: s.seat, canonical: p.canonical, key: p.key, sub_area: p.sub_area || null, why: r ? 'seat returned no verdict' : 'seat lost' })
  }
}
if (unchecked.length) log(`${unchecked.length} prior open findings came back without a verdict — listed under observation.unchecked, never silently not_checked`)

const known = new Map()
for (const r of seats) for (const f of r.findings) {
  if (f.verdict === 'refuted') continue
  if (known.has(f.canonical)) { known.get(f.canonical).also_seen_by = [...(known.get(f.canonical).also_seen_by || []), f.display_id]; log(`duplicate canonical across seats: ${f.canonical} (${f.display_id})`) }
  else known.set(f.canonical, f)
}

function digestFindings(fs) {
  return fs.filter(f => f.verdict !== 'refuted').map(f => ({ display_id: f.display_id, canonical: f.canonical, area: f.area, anchor: f.anchor, tags: f.tags, severity: f.severity, verdict: f.verdict, class: f.class, statement: f.statement.slice(0, 600), measurements: f.measurements.slice(0, 3) }))
}

function crossPrompt(l) {
  return `You are the ${l.title} seat (heuristic ${l.tag}) of a whole-project quality review of ${A.project.id} at ${ROOT} (detached checkout of ${A.as_of_sha}). READ-ONLY; intermediates under ${A.scratch}/cross/${l.lens}/.

${QUALITY}your lens is the one heuristic no single-area reader can judge whole. QUESTION: ${l.question}

INPUTS
- Whole-repo metrics snapshot:\n${A.metrics_summary}
- Import graph: ${A.import_graph_path} (edges, reach-back imports, deferred imports, cycles). Read it with a small script, not by eye.
- Per-area verified findings (statements only) and each seat's seams_for_cross_area:\n${JSON.stringify(seats.map(r => ({ seat: r.seat, seams: r.seams_for_cross_area, findings: digestFindings(r.findings) })), null, 1)}
- Existing canonical ids (area|anchor|primary_tag): ${JSON.stringify([...known.keys()])}

RULES
- A finding here must compare or span at least two modules. Anchor it at the one path::symbol where the cost would be removed (the copy that should derive, the layer that should hide the one below, the call that should carry its inputs); name the other sites in the statement; area is the anchor's area, repo only for a slug: anchor.
- A cost already in the canonical set is a re_observation with added_evidence, not a finding — the pipeline drops the duplicate anyway.
- Measure: every finding carries a measurement with its rule (sites counted, graph query run). Write next_change for each. Severity and class per docs/quality-findings-contract.md §4 and §6.
- Primary tag is ${l.tag}; add inv-<n> when a design invariant in docs/legibility/design-invariants.md encodes the same rule.
- Not a finding: anything the Do-not-steer list excludes; a restatement of a per-area finding; an unmeasured suspicion.
At most ${MAX} findings. State what you read and computed in method.`
}

phase('Cross-area')
let crossLenses = A.cross
if (budget.total && budget.remaining() < 800_000) { log(`budget: ${Math.round(budget.remaining() / 1000)}k left — cross-area lenses skipped, record under Dropped and degraded`); crossLenses = [] }
function degenerateCross(r) {
  if (!r || !Array.isArray(r.findings)) return true
  if ((r.method || '').length < 100) return true
  return r.findings.length > 0 && r.findings.every(f => findingProblems(f, r).length > 0)
}
const crossSeats = (await pipeline(
  crossLenses,
  (l) => seatWithRetry(`cross:${l.lens}`,
    (note) => agent(crossPrompt(l) + note, { label: `cross:${l.lens}${note ? ':retry' : ''}`, phase: 'Cross-area', model: A.routing.cross.model, effort: A.routing.cross.effort, schema: CROSS_SCHEMA }),
    degenerateCross),
  async (r, l) => {
    if (!r) return null
    const c = cleanSeat(r, `cross:${l.lens}`)
    const fresh = []
    for (const f of c.findings) {
      if (known.has(f.canonical)) { r.re_observations.push({ canonical: f.canonical, added_evidence: f.statement }); log(`cross:${l.lens}: ${f.display_id} re-observes ${f.canonical}`) }
      else fresh.push(f)
    }
    const v = fresh.length ? await agent(skepticPrompt(`cross:${l.lens}`, blind(fresh), []), { label: `verify:cross:${l.lens}`, phase: 'Verify', model: A.routing.skeptic.model, effort: A.routing.skeptic.effort, schema: SKEPTIC_SCHEMA }) : { verdicts: [] }
    return applyVerdicts({ ...r, seat: `cross:${l.lens}`, findings: fresh, prior_verdicts: [], verdicts: (v && v.verdicts) || [] })
  },
)).filter(Boolean)
for (const r of crossSeats) for (const f of r.findings) if (f.verdict !== 'refuted' && !known.has(f.canonical)) known.set(f.canonical, f)

const all = [...seats, ...crossSeats]
const counts = { confirmed: 0, weakened: 0, refuted: 0, unverified: 0 }
for (const r of all) for (const f of r.findings) counts[f.verdict] = (counts[f.verdict] || 0) + 1
if (counts.refuted === 0 && (counts.confirmed + counts.weakened) > 10) log('YELLOW FLAG: zero refutations across the run — lead re-verifies a sample inline before trusting the verdict counts')

const priorOpen = new Map(A.prior.open.map(p => [p.canonical, p]))
const observation = { new: [], re_observed: [], standing: A.prior.standing.map(p => p.canonical), fixed: [], unchecked }
const addOnce = (list, c) => { if (!list.includes(c)) list.push(c) }
for (const r of seats) for (const p of r.prior_verdicts || []) {
  if (p.status === 'gone' && p.gone_verdict === 'confirmed') addOnce(observation.fixed, p.canonical)
  else if (p.status === 'still_present' || (p.status === 'gone' && p.gone_verdict !== 'confirmed')) addOnce(observation.re_observed, p.canonical)
}
for (const [c, f] of known) {
  if (f.prior_canonical) {
    addOnce(observation.re_observed, f.prior_canonical)
    if (f.prior_canonical !== c) { f.supersedes_canonical = f.prior_canonical; addOnce(observation.new, c) }
  } else if (priorOpen.has(c)) addOnce(observation.re_observed, c)
  else if (!observation.standing.includes(c)) addOnce(observation.new, c)
}
for (const r of crossSeats) for (const o of r.re_observations || []) addOnce(observation.re_observed, o.canonical)

const digest = {
  run_id: A.run_id, as_of_sha: A.as_of_sha, since: A.since,
  seats: all.map(r => ({ seat: r.seat, area: r.area, area_health: r.area_health, read_coverage: r.read_coverage, h14_measurements: r.h14_measurements, prior_verdicts: r.prior_verdicts, findings: digestFindings(r.findings).map(f => ({ ...f, proposal: (known.get(f.canonical) || {}).proposal, next_change: (known.get(f.canonical) || {}).next_change, supersedes_canonical: (known.get(f.canonical) || {}).supersedes_canonical })) })),
  prior_open: A.prior.open, standing: A.prior.standing, observation, verification: counts, dropped,
}

function synthesisPrompt(critique) {
  return `You are the synthesis seat of a whole-project quality review of ${A.project.id} at ${ROOT} (detached checkout of ${A.as_of_sha}). READ-ONLY; verify in code where a join depends on it.

${QUALITY}then read docs/quality-findings-contract.md §1–§4, §6 (in the dark-factory checkout if this project carries no copy). Below: verified findings from ${seats.length} area seats and ${crossSeats.length} cross-area seats, the prior open findings from earlier instrument runs, standing (accepted) findings, and the observation classes already computed (new / re_observed / standing / fixed).

JOBS
1. MATRIX: one ranked view per area × primary tag that has findings, canonicals best payoff first. Payoff = next-change cost removed × breadth (sites, areas) × verdict weight (confirmed 1, weakened 0.5). Standing findings are listed, never ranked. State the rule once in method and apply it everywhere.
2. STREAMS: group findings that one named remedy removes together. Each stream names the mechanism, what becomes deletable, size and risk, the shared artefacts needing one owner, and mode (agent for mechanical, spawn for design-heavy). Mechanical findings that need no design stay out of streams — the lead files them. Where two viable designs exist, write open_choices with options, trade-offs and the long-term consequence; do not pick unless the analysis converges.
3. INVARIANT CANDIDATES: only rules with a checkable form and a fixture sketch (docs/code-quality.md §Relationship to the design invariants); map to an existing INV when one already encodes it instead of proposing a new one.
4. CONTRADICTIONS: seats proposing conflicting remedies, or a remedy that duplicates a mechanism that already exists; resolve or pose the question.
5. EXONERATIONS: what the metrics flag that the reading shows healthy.
Mark weakened findings visibly wherever they appear. Never rank by, or justify a severity with, a measure from the doc's Do-not-steer list.
${critique ? `\nA critic read your previous synthesis and found these join errors; fix each and say in method how:\n${JSON.stringify(critique, null, 1)}\n` : ''}
DIGEST:
${JSON.stringify(digest, null, 1)}`
}

phase('Synthesize')
const degenerateSynthesis = (s) => !s || (s.method || '').length < 50 || (known.size > 0 && !(s.matrix || []).length)
const degenerateCritic = (c) => !c || !['accept', 'revise'].includes(c.verdict) || (c.verdict === 'revise' && !(c.join_errors || []).length)
const synthOpts = (label) => ({ label, phase: 'Synthesize', model: A.routing.synthesis.model, effort: A.routing.synthesis.effort, schema: SYNTHESIS_SCHEMA })
let synthesis = await seatWithRetry('synthesize', (note) => agent(synthesisPrompt(null) + note, synthOpts(note ? 'synthesize:retry' : 'synthesize')), degenerateSynthesis)

function criticPrompt() {
  return `You are the critic of an assembled synthesis for a whole-project quality review of ${A.project.id} at ${ROOT}. READ-ONLY; open code where a check needs it.

You are not a third reviewer of the code. You read the JOIN: the places where seats' results were combined. ${QUALITY}attend to its "What to measure" and "Do not steer by" sections. Check, and report each hit as a join_error with a fix:
- inconsistent-rule: two figures compared under different rules (one seat counted prose lines, another did not; one measured per function, another per module) — compare the rules figure by figure, not seat by seat.
- one-term-of-sum: a conclusion about complexity or size that reads one term of a pair (module total without the per-function maximum, or the reverse; a split that lowers file size while total complexity rises).
- evidence-contradicts-conclusion: a ranking, stream or exoneration that a seat's own measurements or a skeptic's notes contradict.
- do-not-steer-leak: a line count, average complexity, coverage or test count used as ranking input or severity support.
- duplicate-mechanism: a stream remedy that re-creates a mechanism the tree already has (grep for it).
- unverified-fixed: a canonical classed fixed without a confirmed gone verdict.
- severity-unsupported: a high whose next_change scenario does not support it.
- area-unmapped: a finding whose area is not the area of its anchor path.
- weakened-hidden: a weakened verdict that vanished from the matrix or a stream.
Verdict: accept, or revise when any error changes a ranking, a stream or a disposition.

SYNTHESIS:
${JSON.stringify(synthesis, null, 1)}

DIGEST (the inputs the synthesis was built from):
${JSON.stringify(digest, null, 1)}`
}

phase('Critique')
let critic = null
let revised = false
if (!synthesis) {
  log('synthesis lost — critic skipped; the lead synthesises from the digest by hand and the report says so')
} else {
  critic = await seatWithRetry('critic',
    (note) => agent(criticPrompt() + note, { label: note ? 'critic:retry' : 'critic', phase: 'Critique', model: A.routing.critic.model, effort: A.routing.critic.effort, schema: CRITIC_SCHEMA }),
    degenerateCritic)
  if (critic && critic.verdict === 'revise') {
    log(`critic: ${critic.join_errors.length} join errors — one revision of the synthesis`)
    const revisedSynthesis = await seatWithRetry('synthesize:revised', (note) => agent(synthesisPrompt(critic.join_errors) + note, synthOpts(note ? 'synthesize:revised:retry' : 'synthesize:revised')), degenerateSynthesis)
    if (revisedSynthesis) { synthesis = revisedSynthesis; revised = true } else log('revision lost — the unrevised synthesis is returned with the critic errors attached')
  }
}

return {
  run_id: A.run_id, as_of_sha: A.as_of_sha, since: A.since,
  seats: all.map(r => ({ ...r, verdicts: undefined })),
  refuted: all.flatMap(r => r.findings.filter(f => f.verdict === 'refuted').map(f => ({ seat: r.seat, display_id: f.display_id, canonical: f.canonical, anchor: f.anchor, why: f.verdict_notes }))),
  verification: counts, observation, dropped,
  synthesis, critic, synthesis_revised: revised,
}
```

## Model / effort routing

Set in `args.routing` by the lead, explicit on every `agent()`; the session model is never inherited.

| Seat | Model | Effort | Why this tier |
|---|---|---|---|
| area review (per slice) | opus | high | judgment-dense: architectural review against a definition, not a checklist |
| skeptic (per seat) | sonnet | high | bounded verification of concrete claims that rewards thinking; the cheapest seat that reproduces a measurement |
| cross-area lens (×4) | opus | high | cross-input synthesis over a graph plus N reports |
| synthesis | fable (standing in the DF overlay; else per the provenance rule) | high | synthesis of conflicting complex reports — the team skill's named fable case; fallback opus / xhigh |
| critic | fable (standing in the DF overlay; else per the provenance rule) | high | adversarial review of that synthesis — the other named case; fallback opus / xhigh |

Provenance rule (`skills/team/SKILL.md` §Fable): a fable lead sets the two fable seats directly; an opus lead sets them under the project overlay's standing ruling (`.claude/skills/review-all/project.md`, as dark-factory's grants since 2026-10-05), `--fable`, or a statement in the conversation, otherwise runs the fallback and says in the report that fable was wanted.

Expected per-seat spend, extrapolated from the hotspot run's measured 100–140k for a deep reviewer: area seats 150–250k (they read more), skeptics 60–100k, cross seats 150–250k, synthesis 200–300k, critic 100–200k.

## Failure modes

1. **Degenerate structured output passes the schema** (hotspot failure mode 1). `cleanSeat` + `degenerateSeat` check every finding's anchor shape, statement length, measurement presence, next-change scenario and placeholder text; cross seats (`degenerateCross`), the synthesis (`degenerateSynthesis`) and the critic (`degenerateCritic`) have their own checks through `seatWithRetry`. Every seat is re-run once and then dropped with a `dropped` record the report prints; a lost synthesis skips the critic rather than handing it `null`.
2. **Fourteen-times-N trivial findings.** Prevented by the prompt's next-change filter and the `maxItems` cap; detected by the critic's `severity-unsupported` class. If a seat returns `MAX` findings all `low`, the lead reads its `area_health` before trusting any of them.
3. **A split proposed without the heuristic-14 protocol.** Dropped mechanically (`h14 without measurement protocol`); the report lists it so the seat's omission is visible.
4. **Skeptic anchoring.** The skeptic never sees proposals or severities (`blind`). Zero refutations with more than ten surviving findings logs a YELLOW FLAG; the lead samples five confirmed findings and re-verifies inline, and the report says so under `verification`.
5. **Verdict join by display_id.** Unmatched ids fall to `unverified` and are counted separately; never conflate "survived" with "confirmed".
6. **`fixed` without evidence.** A prior finding is `fixed` only when the seat said `gone` and the skeptic confirmed the locating evidence; the critic's `unverified-fixed` is the backstop.
7. **The result is too large to read raw** (expect 300–600 KB). The lead writes it to `<scratch>/workflow-result.json` and digests with a script (SKILL.md Phase 5).
8. **Main moves during a three-hour run.** Every seat reads the detached worktree at `as_of_sha`, never the main checkout; the lead removes the worktree after the report is committed. No stash anywhere (CLAUDE.md §Working in the main checkout).
9. **Budget directive.** `budget` (`total`, `spent()`, `remaining()`) is a documented Workflow script global (`workflow-authoring` reference §Script body hooks). Under a `+Nk` directive, the guard before `phase('Cross-area')` skips the lenses when fewer than ~800k tokens remain (four opus seats plus skeptics) and logs it; synthesis and critic still run on what exists, and the report lists the lenses under Dropped and degraded. Without a directive `budget.total` is null and the guard is inert.
10. **A prior finding silently unchecked.** Legacy hotspot areas `<area>/<sub>` are split in Phase 0 into the plain area and `sub_area` and assigned to the slice holding those files; after the area pipeline the script lists every assigned prior that came back without a verdict under `observation.unchecked`, so a seat that skipped its re-verification duty is visible in the report rather than read as "still open, nothing changed".
