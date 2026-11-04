# Orchestration — workflow template, args, schemas, prompts, failure modes

The Workflow-tool script for phases 3–4 of `/review-all`, in the house pattern of `skills/hotspot-survey/references/orchestration.md`. Phases 0–2 and 5–6 run in the lead's session (SKILL.md); the script receives their results as `args` and returns one JSON value the lead digests. Field vocabularies (area, anchor, tags, severity, evidence_source, verdict) are those of `docs/quality-findings-contract.md` §1–§4 and are not redefined here.

## Design notes

- Keys are minted in the lead's Phase 5 digest by importing the one key implementation — `shared/src/shared/finding_key.py` once it lands, until then `skills/hotspot-survey/scripts/findings_artefact.py` (`finding_key`, `normalise_anchor`) — never by a second implementation; the digest must agree with that module's test fixture. Inside the script the join identity is the contract §2 `canonical` string, whose normalisation mirrors that module's, so no hashing is needed. Same identity, two representations.
- The large static inputs live on disk, not in `args`: one brief file per seat (§The slice brief) and one JSON file holding every prior open finding in full. `args` carries identities, paths and counts only, so the lead can emit it as a tool-call value (failure mode 11).
- The synthesis and the critic refer to findings by short ids the script assigns (`f<n>` for this run's findings, `p<n>` for priors, one id per canonical), and the script maps them back to canonicals before returning. Canonicals are long and the synthesis abbreviates them when asked to repeat them (failure mode 13); an id has nothing to abbreviate.
- A prior finding re-observed through `prior_canonical` is recorded under the prior's canonical; when the new anchor differs, the new finding carries `supersedes_canonical` (contract §2 anchor-moved rule) and the digest fills `supersedes` with the prior key. Every prior open finding assigned to a slice comes back with a verdict or is listed under `observation.unchecked`.
- A seat reviews a *slice* (an allocation of files); a finding carries the briefing *area* of its anchor. Slices never appear in findings. The script asserts a seat's identity itself (`r.seat = s.seat`, `r.area = s.area`) rather than trusting the seat to echo it (failure mode 12).
- Line counts set a seat's read budget and nothing else; ranking inside the budget uses the §What to measure instruments. The `Do not steer by` list is enforced twice: in every prompt and by the critic.
- The skeptic sees statements and measurements, never proposals (hotspot failure mode 5, now the default rather than the fallback), and records one falsification attempt per measurement; a measurement it did not reproduce cannot support `confirmed` (failure mode 4).
- Cross-area seats run behind a barrier because they need every area seat's canonical set to dedup against — one of the three justified barrier cases in the Workflow reference.
- The critic is a second reader of the join, not a third reviewer of the code. Its error classes are the ones `skills/team/SKILL.md` §4 names, plus the quality doc's two measurement traps (one term of the complexity pair; a do-not-steer measure as severity input). A `severity-unsupported` item carries the severity it argues for, and the revised synthesis says whether it accepted each item, so the lead's digest can apply the accepted corrections to the record (failure mode 14).
- `routing` arrives in `args` so the fable provenance rule (`skills/team/SKILL.md` §Fable) is decided by the lead, outside the script.
- The quality definition reaches every judging prompt one of the contract §10's two ways: `Read docs/code-quality.md` when the tree carries it, else the lead embeds the `code_quality.py::guidance` render as `args.quality_guidance` and the `QUALITY` clause carries it; the prompts never restate it.

## `args` shape

Written by the lead to `<scratch>/args.json` after Phase 1 (SKILL.md Phase 1 §Slices describes the generator) and passed as a JSON value (never a string). Measured on the first dark-factory run: 60 KB in this shape against 820 KB with every slice's read list, test patch targets, prior statements and leads inline.

```js
// JSON Schema for args
const ARGS_SCHEMA = {
  type: 'object',
  required: ['run_id', 'as_of_sha', 'since', 'project', 'root', 'scratch', 'areas', 'slices', 'cross', 'metrics_summary', 'import_graph_path', 'snapshot_path', 'prior', 'routing', 'ceilings'],
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
      required: ['seat', 'area', 'brief', 'read_fully', 'index_only_count', 'tests_reaching_internals_count', 'size_alarms', 'prior_open', 'leads_count'],
      properties: {
        seat: { type: 'string', description: '<area>/<slice-slug>, or <area> when the area is one slice' },
        area: { type: 'string' },
        brief: { type: 'string', description: 'absolute path of the seat brief file (§The slice brief); the seat reads it first, in full' },
        read_fully: { type: 'array', description: 'the brief holds each entry with its snapshot numbers; here only the identity, in rank order, so the prompt can name the top entries', items: { type: 'object', required: ['path', 'lines'], properties: { path: { type: 'string' }, lines: { type: 'integer' } } } },
        index_only_count: { type: 'integer' },
        tests_reaching_internals_count: { type: 'integer' },
        size_alarms: { type: 'array', items: { type: 'string' }, description: 'files over ceilings.alarm_lines; heuristic-14 protocol is mandatory for each' },
        prior_open: { type: 'array', items: { type: 'object', required: ['canonical'],
          properties: { canonical: { type: 'string' }, key: { type: 'string' },
            sub_area: { type: 'string', description: 'contract §1 sub_area; a legacy <area>/<sub> hotspot area is split in Phase 0 into the plain area (in the key) and this field, and the finding is assigned to the slice holding the sub-area files' } } },
          description: 'every prior open finding of this area assigned to this slice, identity only (statements are in the brief and in prior.full_path); each must come back in prior_verdicts — the script lists any that do not' },
        leads_count: { type: 'integer' },
      } } },
    cross: { type: 'array', items: { type: 'object', required: ['lens', 'tag', 'title', 'question'],
      properties: { lens: { type: 'string' }, tag: { type: 'string' }, title: { type: 'string' }, question: { type: 'string' } } } },
    metrics_summary: { type: 'string', description: '`scripts/quality_metrics_snapshot.py --summary` plus its `--diff` sections, ≤4k chars' },
    language_note: { type: ['string', 'null'], description: "the overlay's Language note slot verbatim, or null/absent for a Python project: how to read the quality doc's Python-flavoured heuristic-13 and Tests-stance symptoms (reach-back and deferred imports, re-export shims, satellites over private state, patching privates) in the project's language, and what the complexity instrument measures; the area prompt inserts it after YOUR SLICE" },
    quality_guidance: { type: ['string', 'null'], description: "null when <root>/docs/code-quality.md exists (prompts say Read it); otherwise the render of orchestrator/src/orchestrator/agents/code_quality.py::guidance, embedded verbatim (contract §10)" },
    import_graph_path: { type: 'string', description: "JSON file under scratch: the snapshot's import_graph section unchanged — {edges:[[from,to]], reach_back:[{from,to,names,line}], deferred:[{from,line,imports,closes_cycle}], cycles:[[module,...]], hidden_cycles:[[module,...]], typing_cycles:[[module,...]]}, nodes are module import names (a file record's `module`); cycles are over import-time edges, hidden_cycles add function-local imports, typing_cycles add `if TYPE_CHECKING:` ones, and closes_cycle marks a deferred import lying on a hidden cycle (plans/quality-metrics-snapshot-prd.md decision 10)" },
    snapshot_path: { type: 'string', description: "the Phase 1 snapshot JSON under scratch; a seat with a file larger than its remaining budget reads that file's rows of the `functions` table ({'<path>::<Class>::<func>': cognitive, ...}) to rank what it reads (failure mode 15)" },
    prior: { type: 'object', required: ['standing', 'full_path'], properties: {
      full_path: { type: 'string', description: 'JSON file under scratch holding every prior open finding in full — contract §1 fields plus canonical, source_run, sub_area, disposition — the synthesis reads it; the script derives the open set from slices[].prior_open and never carries a second copy' },
      standing: { type: 'array', items: { type: 'object', required: ['canonical', 'key', 'area', 'statement', 'disposition'] } } } },
    routing: { type: 'object', required: ['area', 'skeptic', 'cross', 'synthesis', 'critic'],
      additionalProperties: { type: 'object', required: ['model', 'effort'], properties: { model: { type: 'string' }, effort: { type: 'string' } } } },
    ceilings: { type: 'object', required: ['soft_lines', 'alarm_lines', 'max_findings_per_seat'], description: "soft_lines and alarm_lines come from the snapshot's params (h14_soft_ceiling_lines, h14_alarm_lines)",
      properties: { soft_lines: { type: 'integer' }, alarm_lines: { type: 'integer' }, max_findings_per_seat: { type: 'integer' } } },
  },
}
```

### The slice brief

One markdown file per seat at `<scratch>/seats/<seat>/brief.md`, written by the lead's generator (SKILL.md Phase 1 §Slices) from the full slice record, in this order, each section headed as named so the prompt can cite it:

1. **READ FULLY, in this order** — one line per file with its snapshot numbers: lines, prose share, cognitive max / total, fan-in, fan-out, reach-back imports, cycle-closing deferred imports, re-exports, test patch targets. The numbers come from the snapshot's file record for the path: `prose_share = prose_ratio` (0 where null), `fan_in = fan_in_src`, `reexports = len(reexport_names)`, `patch_targets` = the count of distinct `private_patch_targets` across tests records that continue this record's module, `cycle_closing_imports` from the import graph's `deferred` entries with `closes_cycle` for this module; the rest are the record's own fields. An entry whose area is not the slice's — a one-small-file area read inside a neighbouring slice (SKILL.md Phase 1 §Slices) — carries `area: <area>` on its line, and the seat anchors findings there in that area.
2. **INDEX ONLY** — the remaining paths.
3. **TESTS THAT REACH THIS SLICE'S INTERNALS** — `test_path -> targets`.
4. **SIZE ALARMS** — paths over `ceilings.alarm_lines`.
5. **PRIOR OPEN FINDINGS AT THIS AREA** — per prior: the exact `canonical`, key, severity, source run, sub-area, and the full statement (never truncated below what the seat needs to re-verify it).
6. **LEADS** — `(source) text`.

`<scratch>/prior_open.json` holds the same priors (every slice's, plus the unassigned ones) in full for the synthesis. `args.prior.full_path` names it.

Default `cross` lenses (an overlay may add, never remove):

| lens | tag | question |
|---|---|---|
| `spot` | `h11` | Which facts, definitions and policies live in more than one module with no derivation edge between the copies, so they can disagree? Map onto INV-5/INV-9 where one applies. |
| `layering` | `h9` | Which modules import upward or sideways in cycles, which are shallow (interface ≈ implementation), and where is a stratum missing? Reach-back imports and the deferred imports marked `closes_cycle` are the leads; `hidden_cycles` and `typing_cycles` name the cycles they close. |
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
    { title: 'Calibrate', detail: 'when the area skeptics refuted under 5 %: ten confirmed findings (every high first) re-verified by fresh skeptics', model: 'sonnet' },
    { title: 'Cross-area', detail: 'h11 SPOT, h9 layering, h3 orthogonal dimensions, h7 stateless interactions', model: 'opus' },
    { title: 'Reverify high stratum', detail: 'when the calibration sample disputes >= 30 % or >= 2 highs: every remaining confirmed high re-verified by a fresh skeptic', model: 'sonnet' },
    { title: 'Synthesize', detail: 'join prior and new findings by short id; area x heuristic ranked view; candidate streams' },
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
      lines_read: { type: 'integer' },
      partial: { type: 'array', description: 'files larger than the remaining budget, read by the symbol-index + ranked-functions + module-state protocol instead of in full', items: { type: 'object', required: ['path', 'lines_read_verbatim', 'how'], properties: {
        path: { type: 'string' }, lines_read_verbatim: { type: 'integer' }, how: { type: 'string', description: 'which functions (by cognitive max rank) and which module-level state were read, and what was only indexed' } } } } } },
    area_health: { type: 'string', description: 'how the slice is structured today and where the next-change cost sits; >=200 chars' },
    prior_verdicts: { type: 'array', items: PRIOR_VERDICT },
    findings: { type: 'array', maxItems: MAX, items: FINDING },
    h14_measurements: { type: 'array', items: H14 },
    seams_for_cross_area: { type: 'string', description: 'duplicated policies, layering doubts, flag axes, attribute reaches you suspect but could not anchor inside this slice' },
  } }

const FALSIFICATION = { type: 'object', required: ['measurement', 'ran', 'would_refute', 'found', 'reproduced'], properties: {
  measurement: { type: 'string', description: "the measurement's name, or 'gone' for a prior marked gone" },
  ran: { type: 'string', description: 'the command run, or the file and symbol opened' },
  would_refute: { type: 'string', description: 'the result that would have refuted this measurement' },
  found: { type: 'string', description: 'what you actually found, with your own figure' },
  reproduced: { type: 'boolean', description: "true only when your own figure agrees with the reviewer's under the stated rule" },
} }

const SKEPTIC_SCHEMA = { type: 'object', required: ['verdicts'], properties: {
  verdicts: { type: 'array', items: { type: 'object', required: ['display_id', 'verdict', 'checked', 'notes', 'falsification'], properties: {
    display_id: { type: 'string', description: 'the finding display_id, or the prior canonical for a gone verdict' },
    verdict: { type: 'string', enum: ['confirmed', 'weakened', 'refuted'] },
    checked: { type: 'string', description: 'the files opened and the commands run' },
    notes: { type: 'string', description: 'what was found; for weakened/refuted, which fact fails' },
    falsification: { type: 'array', minItems: 1, items: FALSIFICATION, description: 'one entry per measurement of the finding' },
  } } } } }

const CROSS_SCHEMA = { type: 'object', required: ['lens', 'findings', 're_observations', 'method'], properties: {
  lens: { type: 'string' },
  findings: { type: 'array', maxItems: MAX, items: FINDING },
  re_observations: { type: 'array', items: { type: 'object', required: ['canonical', 'added_evidence'], properties: {
    canonical: { type: 'string' }, added_evidence: { type: 'string' } } } },
  method: { type: 'string', description: 'what was read and computed, with counts' },
} }

const ID_DESC = 'short ids from the digest (f<n> for this run, p<n> for priors), bare — never canonicals, paths, abbreviations or appended flags'
const SYNTHESIS_SCHEMA = { type: 'object',
  required: ['matrix', 'streams', 'invariant_candidates', 'contradictions', 'exonerations', 'method'],
  properties: {
    matrix: { type: 'array', items: { type: 'object', required: ['area', 'tag', 'ranked'], properties: {
      area: { type: 'string' }, tag: { type: 'string' },
      ranked: { type: 'array', items: { type: 'string' }, description: `${ID_DESC}; best payoff first` } } } },
    streams: { type: 'array', items: { type: 'object',
      required: ['name', 'canonicals', 'remedy', 'deletable', 'size', 'risk', 'mode', 'seam_candidates', 'open_choices'],
      properties: {
        name: { type: 'string' },
        canonicals: { type: 'array', items: { type: 'string' }, description: `${ID_DESC}; the script maps them back to canonicals` },
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
      between: { type: 'array', items: { type: 'string' }, description: ID_DESC }, conflict: { type: 'string' }, resolution_or_question: { type: 'string' } } } },
    exonerations: { type: 'string', description: 'areas or files the metrics flag that the reading shows healthy, and why, naming findings by short id; empty string if none' },
    method: { type: 'string', description: 'the ranking rule used, stated once' },
    critic_responses: { type: 'array', description: 'present only in a revision: one entry per critic join_error, in order', items: { type: 'object', required: ['join_error', 'accepted', 'how'], properties: {
      join_error: { type: 'integer', description: '0-based index into the critic join_errors you were given' },
      accepted: { type: 'boolean' },
      how: { type: 'string', description: 'what changed, or why the item was declined' } } } },
  } }

const CRITIC_SCHEMA = { type: 'object', required: ['join_errors', 'verdict', 'notes'], properties: {
  join_errors: { type: 'array', items: { type: 'object', required: ['kind', 'where', 'detail', 'fix'], properties: {
    kind: { type: 'string', enum: ['inconsistent-rule', 'one-term-of-sum', 'evidence-contradicts-conclusion', 'do-not-steer-leak', 'duplicate-mechanism', 'unverified-fixed', 'severity-unsupported', 'area-unmapped', 'weakened-hidden'] },
    where: { type: 'string', description: 'a short id (f<n>/p<n>), a stream name or a matrix cell' },
    detail: { type: 'string' }, fix: { type: 'string' },
    severity_to: { type: 'string', enum: ['high', 'medium', 'low'], description: 'severity-unsupported only: the severity the next_change scenario supports' } } } },
  verdict: { type: 'string', enum: ['accept', 'revise'] },
  notes: { type: 'string' },
} }

const QUALITY = A.quality_guidance
  ? `The quality definition this review judges by follows, rendered from the normative docs/code-quality.md (this project carries no copy). Judge by its Definition:\n\n${A.quality_guidance}\n\n`
  : `Read ${ROOT}/docs/code-quality.md first and judge by its Definition: `
const QUALITY_REF = A.quality_guidance ? 'the quality definition above' : `${ROOT}/docs/code-quality.md`
// The dangling-tree rule (failure mode 8), one home for every prompt that reads ROOT.
const TREE_EXPORT = `${A.scratch}/tree-export`
const treeGuard = (noteField) => `IF THE TREE DOES NOT RESOLVE (${ROOT} missing, or a dangling path): do not read the main checkout or any other tree, whatever state it is in. Read the export of ${A.as_of_sha} at ${TREE_EXPORT} in place of ${ROOT} wherever this prompt or your work reads the tree, and say so in ${noteField}. When ${TREE_EXPORT} does not exist yet, build it with this one command; it extracts into a fresh directory and renames that into place, so a concurrent seat never reads a half-written tree, and a failed rename means another seat built it first — use theirs:
d=$(mktemp -d -p ${A.scratch}) && git -C ${A.project.main_root} archive ${A.as_of_sha} | tar -x -C "$d" && mv -T "$d" ${TREE_EXPORT} || rm -rf "$d"`

// ANCHOR_RE, LINE_PIN_RE, SYMBOL_PATH_RE and normAnchor mirror skills/hotspot-survey/scripts/findings_artefact.py
// (ANCHOR_RE, LINE_PIN_RE, SYMBOL_PATH_RE, normalise_anchor) — the interim key reference of contract §2; the
// lead's Python digest imports that module (or shared/finding_key.py once landed) to mint keys.
const ANCHOR_RE = /^(slug:[a-z0-9]+(-[a-z0-9]+)*|[\w.-][\w./-]*(::[A-Za-z_][\w.]*)?)$/
const LINE_PIN_RE = /(\.[A-Za-z0-9]{1,8}):\d+(?:[-,]\d+)*/g
// A symbol path after the file (Rust `Type::method`) is a class-level definition, written `Type.method` (failure mode 17).
const SYMBOL_PATH_RE = /::([A-Za-z_][\w.]*(?:::[A-Za-z_][\w.]*)+)$/
function unpinAnchor(a) {
  let s = String(a || '').trim().replace(/\\/g, '/')
  while (s.startsWith('./')) s = s.slice(2)
  return s.replace(LINE_PIN_RE, '$1')
}
const dotSymbolPath = (s) => s.replace(SYMBOL_PATH_RE, (_, path) => '::' + path.split('::').join('.'))
const normAnchor = (a) => dotSymbolPath(unpinAnchor(a))
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
    else {
      const unpinned = unpinAnchor(f.anchor)
      if (dotSymbolPath(unpinned) !== unpinned) log(`${label}: ${f.display_id} anchor ${unpinned} normalised to ${dotSymbolPath(unpinned)} (failure mode 17)`)
      f.anchor = normAnchor(f.anchor)
      f.canonical = canonical(f)
      kept.push(f)
    }
  }
  return { ...r, findings: kept }
}
function degenerateSeat(r, raw) {
  if (!r) return true
  if ((r.area_health || '').length < 200) return true
  const rawCount = ((raw && raw.findings) || []).length
  return rawCount > 0 && r.findings.length === 0
}

const plannedLines = (s) => s.read_fully.reduce((n, f) => n + f.lines, 0)

function topReads(s) {
  return s.read_fully.slice(0, 6).map(f => `${f.path} (${f.lines} lines)`).join(', ') + (s.read_fully.length > 6 ? `, … (${s.read_fully.length} files)` : '')
}

function areaPrompt(s) {
  return `You are one seat of a whole-project quality review of ${A.project.id}. The tree is ${ROOT}, a detached checkout of ${A.as_of_sha}. READ-ONLY: modify nothing, run no state-changing command, make no MCP write. Intermediates go under ${A.scratch}/seats/${s.seat}/.

${treeGuard('area_health')}

${QUALITY}quality is the expected cost and risk of the next change, made by an agent from a partial view. The fourteen heuristics and the two stances are lenses you name on each finding (tags), not a checklist to walk; one finding per heuristic is the failure mode this prompt exists to prevent.

YOUR SLICE: ${s.seat} (findings carry area "${s.area}" unless a READ FULLY entry names another area). ${A.language_note ? `LANGUAGE NOTE: ${A.language_note}\n\n` : ''}Your slice brief is the file ${s.brief} — Read it first, in full. It holds, in this order: READ FULLY (ranked by the instruments in that doc's "What to measure" table, each entry with its snapshot numbers; line counts set only how much fits in your budget — top entries: ${topReads(s)}), INDEX ONLY (${s.index_only_count} files), TESTS THAT REACH THIS SLICE'S INTERNALS (${s.tests_reaching_internals_count} test files), SIZE ALARMS (${s.size_alarms.length ? s.size_alarms.join(', ') : 'none'} — heuristic 14 protocol mandatory for each; fill h14_measurements with prose share and whether the prose is in its right home, a named partition of top-level symbols, and per candidate the heuristic-13 symptom naming the two modules and the specific import, or none), PRIOR OPEN FINDINGS AT THIS AREA (${s.prior_open.length}, with full statements; re-verify each at this tree and report every one in prior_verdicts using its exact canonical string — still_present or gone, a gone needs the commit or the replacing code; set prior_canonical on a new finding only when it adds evidence to one rather than restating it), and LEADS (${s.leads_count}; things to verify, never findings in themselves).

YOUR READ BUDGET is ${plannedLines(s)} lines: the READ FULLY list, in rank order. Read until it is spent — the list is what the ranking chose, and a seat that stops early leaves chosen files unread (on the first reify run 12 of 30 seats read under half their plan). A file you cannot finish goes through the protocol below, never left at its first lines. Report read_coverage.lines_read as the lines you actually read verbatim; a short honest figure is useful, an inflated one is not.

A FILE LARGER THAN YOUR REMAINING BUDGET: when one READ FULLY entry exceeds the lines you can still read in full, do not read it from the top until the budget runs out. Read its symbol index (grep -n 'def \\|class '), its module-level state (assignments and constants outside any def), and its functions in descending cognitive max — the rows of the snapshot's functions table at ${A.snapshot_path} whose key starts with that path (a {"<path>::<Class>::<func>": cognitive} map; filter with a small script) — until the budget is spent; the rest stays indexed. Record it under read_coverage.partial (path, lines read verbatim, which functions and state you read), never under files_read_fully.

METHOD
1. Map: symbol index, module docstrings, top-level state, and this slice's rows in the import graph at ${A.import_graph_path}.
2. Read fully in rank order. Follow a seam out of the slice only far enough to anchor a claim.
3. For every candidate, write next_change first: the change an agent would plausibly make here, what it must read or know beyond the file to make it correctly, and what catches a wrong version before it lands. If that cannot be written concretely, it is not a finding.
4. Measure: at least one measurement per finding, each with the rule it was computed under (what was counted, excluded, over what population). Re-run a metric you rely on rather than quoting the sheet.
5. Anchor at path::symbol (the enclosing top-level or class-level definition) at this tree; no line numbers.
6. Severity per docs/quality-findings-contract.md §4, and before writing high, answer from the next_change scenario: is the wrong version caught, and by what (a test, type check, gate, invariant, or a reader who must open the other site)? A wrong version that a named check catches before it lands is medium; high needs a wrong version that lands uncaught. Class mechanical only when one anchor, no design choice, and fixable from the statement — any split or reduce-this-number proposal is structural.

NOT A FINDING: a style nit; a heuristic that could apply; anything whose severity rests on raw line count, average complexity, coverage or test count (the doc's "Do not steer by" — context only); a restatement of a prior open finding; a proposal without a statement; a cost you did not read the code for.

Emit at most ${MAX} findings ranked by next-change cost removed. Fewer is correct when the slice is healthy: say what you read and why it is healthy in area_health. Put the cross-module suspicions you could not anchor inside the slice in seams_for_cross_area. Record read coverage honestly. Never emit placeholder content to satisfy the schema.`
}

function skepticPrompt(label, findings, gone) {
  return `You are an adversarial verifier for a whole-project quality review of ${A.project.id} at ${ROOT} (detached checkout of ${A.as_of_sha}). READ-ONLY.

${treeGuard("each verdict's checked")}

${A.quality_guidance ? QUALITY : ''}A reviewer judged "${label}" against ${QUALITY_REF} and produced the findings below as statements with measurements. You are deliberately not shown their proposals or severities. Try to REFUTE each.

FOR EVERY FINDING, FOR EACH OF ITS MEASUREMENTS, make one explicit falsification attempt and record it in falsification: what you ran (the command, or the file and symbol you opened), what result would have refuted the measurement, what you found with your own figure, and reproduced = true only when your figure agrees with the reviewer's under the stated rule. A measurement you did not reproduce cannot support confirmed: the verdict is then weakened by default (the pipeline enforces this), or refuted when your figure removes the claim.
- refuted: the anchor does not exist at this tree; or a measured fact is wrong by enough to remove the claim; or the cost is already removed — a derivation, guard, test or mechanism exists that the statement implies is missing (grep for it before agreeing it is missing); or the claim rests on a "Do not steer by" measure of the quality doc.
- weakened: the structure is real but a measurement is materially off or not reproduced, an existing mechanism covers most of the cost, or the next_change scenario does not support the claim's weight.
- confirmed: only when every measurement reproduced and no falsification attempt succeeded.
Check the code, not the prose. For duplication claims open every named site. One verdict per display_id.
Calibration: in the first dark-factory run and its two sibling instruments at the same tree, 536 findings drew zero refutations. A verifier that never refutes is not measuring; a refutation that holds is the most valuable output you can return.

${gone.length ? `The reviewer also marked these prior findings GONE. Confirm each by locating the change (git log -S, the replacing code); an unlocatable gone is refuted, meaning the prior finding stands. Use the canonical as display_id and 'gone' as the falsification measurement name.\n${JSON.stringify(gone, null, 1)}\n` : ''}
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
  if (r.seat !== s.seat || r.area !== s.area) log(`${s.seat}: seat echoed identity "${r.seat}"/"${r.area}" — overwritten with the slice's`)
  r.seat = s.seat
  r.area = s.area
  r.read_coverage = { ...r.read_coverage, planned_lines: plannedLines(s) }
  return r
}

async function verifySeat(r, label) {
  if (!r) return null
  const gone = (r.prior_verdicts || []).filter(v => v.status === 'gone')
  if (!r.findings.length && !gone.length) return { ...r, verdicts: [] }
  const v = await agent(skepticPrompt(label, blind(r.findings), gone), { label: `verify:${label}`, phase: 'Verify', model: A.routing.skeptic.model, effort: A.routing.skeptic.effort, schema: SKEPTIC_SCHEMA })
  return { ...r, verdicts: (v && v.verdicts) || [] }
}

const calibration = { downgraded_unreproduced: 0, high_stratum_reverified: false }
// A confirmed verdict with an unreproduced measurement is weakened (failure mode 4). Pure, so the Calibrate
// phase applies the same rule without counting its sample into the run's figures.
function falsificationGate(v) {
  const missing = (v.falsification || []).filter(x => x && x.reproduced === false)
  if (v.verdict !== 'confirmed' || !missing.length) return v
  return { ...v, verdict: 'weakened', notes: `[pipeline: confirmed → weakened; not reproduced: ${missing.map(x => x.measurement).join(', ')}] ${v.notes}` }
}
function enforceFalsification(v) {
  const g = falsificationGate(v)
  if (g !== v) {
    calibration.downgraded_unreproduced += 1
    log(`verdict ${v.display_id}: confirmed with unreproduced measurement(s) — downgraded to weakened`)
  }
  return g
}
function countVerdicts(rs) {
  const c = { confirmed: 0, weakened: 0, refuted: 0, unverified: 0 }
  for (const r of rs) for (const f of r.findings) c[f.verdict] = (c[f.verdict] || 0) + 1
  return c
}

function applyVerdicts(r) {
  const by = new Map(r.verdicts.map(v => [v.display_id, enforceFalsification(v)]))
  for (const f of r.findings) {
    const v = by.get(f.display_id)
    f.verdict = v ? v.verdict : 'unverified'
    f.verdict_notes = v ? `${v.checked} — ${v.notes}` : 'no verdict matched this display_id'
    f.falsification = v ? v.falsification : []
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
    unchecked.push({ seat: s.seat, canonical: p.canonical, key: p.key || null, sub_area: p.sub_area || null, why: r ? 'seat returned no verdict' : 'seat lost' })
  }
}
if (unchecked.length) log(`${unchecked.length} prior open findings came back without a verdict — listed under observation.unchecked, never silently not_checked`)

const known = new Map()
for (const r of seats) for (const f of r.findings) {
  if (f.verdict === 'refuted') continue
  if (known.has(f.canonical)) { known.get(f.canonical).also_seen_by = [...(known.get(f.canonical).also_seen_by || []), f.display_id]; log(`duplicate canonical across seats: ${f.canonical} (${f.display_id})`) }
  else known.set(f.canonical, f)
}

const notRefuted = (f) => f.verdict !== 'refuted'
const digestFinding = (f) => ({ display_id: f.display_id, canonical: f.canonical, area: f.area, anchor: f.anchor, tags: f.tags, severity: f.severity, verdict: f.verdict, verdict_notes: f.verdict_notes, class: f.class, statement: f.statement.slice(0, 600), measurements: f.measurements.slice(0, 3) })
function digestFindings(fs) {
  return fs.filter(notRefuted).map(digestFinding)
}

function crossPrompt(l) {
  return `You are the ${l.title} seat (heuristic ${l.tag}) of a whole-project quality review of ${A.project.id} at ${ROOT} (detached checkout of ${A.as_of_sha}). READ-ONLY; intermediates under ${A.scratch}/cross/${l.lens}/.

${treeGuard('method')}

${QUALITY}your lens is the one heuristic no single-area reader can judge whole. QUESTION: ${l.question}

INPUTS
- Whole-repo metrics snapshot:\n${A.metrics_summary}
- Import graph: ${A.import_graph_path} (edges, reach-back imports, deferred imports with closes_cycle, cycles, hidden_cycles, typing_cycles). Read it with a small script, not by eye.
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

// Calibrate (failure modes 4, 19): when the area skeptics refuted under 5 % of what they verified, one fresh
// skeptic per sampled confirmed finding re-verifies it, so the lead reads ten outcomes rather than re-reading
// code. Started here, it runs alongside the cross lenses with no barrier and is awaited before the digest, so
// the synthesis and critic see its disputes; its agents name their phase in opts rather than calling phase(),
// which the concurrent cross arm sets.
const CALIBRATION_SAMPLE = 10
// Evenly spaced picks over a canonical-sorted list: deterministic (the script cannot use Math.random) and
// spread across areas, where the first k entries would all come from the alphabetically first area.
function spreadPick(list, k) {
  if (list.length <= k) return list
  return Array.from({ length: k }, (_, i) => list[Math.floor(i * list.length / k)])
}
async function freshVerdict(f, seat, phaseTitle, label) {
  const v = await agent(skepticPrompt(seat, blind([f]), []), { label, phase: phaseTitle, model: A.routing.skeptic.model, effort: A.routing.skeptic.effort, schema: SKEPTIC_SCHEMA })
  const vs = (v && v.verdicts) || []
  const got = vs.find(x => x.display_id === f.display_id) || vs[0]
  return got ? falsificationGate(got) : null
}
// A fresh skeptic that does not confirm records a dispute beside the finding; the finding's own verdict stays,
// because one skeptic does not silently overwrite another — the deliberation rules.
function recordOutcome(f, g) {
  const outcome = g ? g.verdict : 'lost'
  if (outcome === 'weakened' || outcome === 'refuted') f.calibration_disputes_verdict = { to: outcome, note: g.notes }
  return { canonical: f.canonical, display_id: f.display_id, severity: f.severity, outcome, note: g ? g.notes : 'skeptic lost', falsification: g ? g.falsification : [] }
}
function tally(outcomes) {
  const n = (o) => outcomes.filter(x => x.outcome === o).length
  return { confirmed: n('confirmed'), weakened: n('weakened'), refuted: n('refuted'), lost: n('lost') }
}
const isDisputed = (o) => o.outcome === 'weakened' || o.outcome === 'refuted'
async function calibrate() {
  const c = countVerdicts(seats)
  // Rate rule: sample whenever the area skeptics refuted under 5 % of the findings they verified — a rate near
  // zero is the soft-skeptic signal, not only zero itself — and more than ten findings survived.
  if (c.refuted / (c.confirmed + c.weakened + c.refuted) >= 0.05 || (c.confirmed + c.weakened) <= 10) return null
  const byCanonical = (a, b) => (a.f.canonical < b.f.canonical ? -1 : a.f.canonical > b.f.canonical ? 1 : 0)
  const confirmed = seats.flatMap(r => r.findings.filter(f => f.verdict === 'confirmed').map(f => ({ f, seat: r.seat }))).sort(byCanonical)
  const highs = spreadPick(confirmed.filter(x => x.f.severity === 'high'), CALIBRATION_SAMPLE)
  const rest = spreadPick(confirmed.filter(x => x.f.severity !== 'high'), CALIBRATION_SAMPLE - highs.length)
  const sample = [...highs, ...rest]
  log(`calibrate: the area skeptics refuted ${c.refuted} of ${c.confirmed + c.weakened + c.refuted} — ${sample.length} confirmed findings (${highs.length} high) go to fresh skeptics`)
  const results = await parallel(sample.map(({ f, seat }) => () => freshVerdict(f, seat, 'Calibrate', `calibrate:${f.display_id}`)))
  const outcomes = sample.map(({ f }, i) => recordOutcome(f, results[i]))
  const disputed = outcomes.filter(isDisputed)
  // Drift signal: >= 30 % of the sample, or >= 2 sampled highs, disputed re-verifies the whole high stratum.
  const drift = disputed.length >= 0.3 * sample.length || disputed.filter(o => o.severity === 'high').length >= 2
  const sampled = { sampled: sample.length, strata: { high: highs.length, random: rest.length }, ...tally(outcomes), drift, outcomes }
  log(`calibrate: ${sampled.confirmed} confirmed, ${sampled.weakened} weakened, ${sampled.refuted} refuted, ${sampled.lost} lost of ${sample.length}${drift ? ' — drift: the high stratum is re-verified' : ''}`)
  return sampled
}
// Every confirmed high the sample did not cover gets a fresh skeptic; outcomes are recorded as Calibrate's are.
async function reverifyHighStratum(sample, rs) {
  const covered = new Set(sample.outcomes.map(o => o.canonical))
  const stratum = []
  for (const r of rs) for (const f of r.findings) {
    if (f.verdict !== 'confirmed' || f.severity !== 'high' || covered.has(f.canonical)) continue
    covered.add(f.canonical)
    stratum.push({ f, seat: r.seat })
  }
  log(`reverify: ${stratum.length} remaining confirmed highs go to fresh skeptics`)
  const results = await parallel(stratum.map(({ f, seat }) => () => freshVerdict(f, seat, 'Reverify high stratum', `reverify:${f.display_id}`)))
  const outcomes = stratum.map(({ f }, i) => recordOutcome(f, results[i]))
  calibration.high_stratum_reverified = true
  calibration.high_stratum = { reverified: stratum.length, ...tally(outcomes), outcomes }
  log(`reverify: ${calibration.high_stratum.confirmed} confirmed, ${calibration.high_stratum.weakened} weakened, ${calibration.high_stratum.refuted} refuted, ${calibration.high_stratum.lost} lost of ${stratum.length}`)
}
const calibrationRun = calibrate()

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
    return applyVerdicts({ ...r, seat: `cross:${l.lens}`, area: null, findings: fresh, prior_verdicts: [], verdicts: (v && v.verdicts) || [] })
  },
)).filter(Boolean)
for (const r of crossSeats) for (const f of r.findings) if (f.verdict !== 'refuted' && !known.has(f.canonical)) known.set(f.canonical, f)

const all = [...seats, ...crossSeats]
const yellowFlagSample = await calibrationRun
if (yellowFlagSample && yellowFlagSample.drift) {
  phase('Reverify high stratum')
  await reverifyHighStratum(yellowFlagSample, all)
}
const counts = countVerdicts(all)
const verified = counts.confirmed + counts.weakened + counts.refuted
calibration.refutation_rate = verified ? Math.round(1000 * counts.refuted / verified) / 1000 : null
if (counts.refuted === 0 && (counts.confirmed + counts.weakened) > 10) log('YELLOW FLAG: zero refutations across the run — the lead reads yellow_flag_sample (the Calibrate phase) before trusting the verdict counts, and compares refutation_rate with prior runs')

// Short ids: one per canonical. Priors first (p<n>), then this run's findings (f<n>); a finding whose
// canonical is also a prior keeps the prior's id, so the synthesis never sees two ids for one canonical.
const PRIOR_OPEN_ALL = A.slices.flatMap(s => s.prior_open)
const priorOpen = new Map(PRIOR_OPEN_ALL.map(p => [p.canonical, p]))
const idOf = new Map()
let pn = 0
for (const p of PRIOR_OPEN_ALL) if (!idOf.has(p.canonical)) idOf.set(p.canonical, `p${++pn}`)
let fn = 0
for (const c of known.keys()) if (!idOf.has(c)) idOf.set(c, `f${++fn}`)
const canonicalOf = new Map([...idOf].map(([c, id]) => [id, c]))
const ID_RE = /^[fp]\d+$/
function resolveIds(list, where, unresolved) {
  return (list || []).map(x => {
    const s = String(x).trim()
    const c = canonicalOf.get(s)
    if (c) return c
    unresolved.push({ where, id: s, why: ID_RE.test(s) ? 'unknown id' : 'not a short id' })
    return s
  })
}
function mapIds(syn) {
  if (!syn) return syn
  const unresolved = []
  for (const cell of syn.matrix || []) cell.ranked = resolveIds(cell.ranked, `matrix ${cell.area}×${cell.tag}`, unresolved)
  for (const st of syn.streams || []) st.canonicals = resolveIds(st.canonicals, `stream ${st.name}`, unresolved)
  for (const c of syn.contradictions || []) c.between = resolveIds(c.between, 'contradiction', unresolved)
  if (unresolved.length) log(`synthesis: ${unresolved.length} entries were not bare short ids — kept verbatim and listed under unresolved_ids`)
  return { ...syn, unresolved_ids: unresolved }
}

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

const refuted = all.flatMap(r => r.findings.filter(f => f.verdict === 'refuted').map(f => ({ seat: r.seat, display_id: f.display_id, canonical: f.canonical, anchor: f.anchor, why: f.verdict_notes })))

const digest = {
  run_id: A.run_id, as_of_sha: A.as_of_sha, since: A.since,
  ids_rule: 'every finding and prior carries a short id (f<n> this run, p<n> prior; one id per canonical); the synthesis and critic refer to findings by these ids only',
  seats: all.map(r => ({ seat: r.seat, area: r.area, area_health: r.area_health, read_coverage: r.read_coverage, h14_measurements: r.h14_measurements, prior_verdicts: r.prior_verdicts, findings: r.findings.filter(notRefuted).map(f => ({ id: idOf.get(f.canonical), ...digestFinding(f), calibration_disputes_verdict: f.calibration_disputes_verdict, proposal: (known.get(f.canonical) || {}).proposal, next_change: (known.get(f.canonical) || {}).next_change, supersedes_canonical: (known.get(f.canonical) || {}).supersedes_canonical })) })),
  prior_ids: PRIOR_OPEN_ALL.filter((p, i, a) => a.findIndex(q => q.canonical === p.canonical) === i).map(p => ({ id: idOf.get(p.canonical), canonical: p.canonical, key: p.key || null, sub_area: p.sub_area || null })),
  prior_open_file: A.prior.full_path, prior_open_count: priorOpen.size, standing: A.prior.standing, observation, verification: counts, calibration, dropped, refuted,
}

function synthesisPrompt(critique, mechanical) {
  return `You are the synthesis seat of a whole-project quality review of ${A.project.id} at ${ROOT} (detached checkout of ${A.as_of_sha}). READ-ONLY; verify in code where a join depends on it.

${treeGuard('method')}

${QUALITY}then read docs/quality-findings-contract.md §1–§4, §6 (in the dark-factory checkout if this project carries no copy). Below: verified findings from ${seats.length} area seats and ${crossSeats.length} cross-area seats, standing (accepted) findings, and the observation classes already computed (new / re_observed / standing / fixed). The ${priorOpen.size} prior open findings from earlier instrument runs are in the JSON file ${A.prior.full_path} — Read it: their statements, severities, classes and keys are inputs to the matrix and streams exactly like this run's findings. digest.prior_ids maps each prior's short id to its canonical in that file. Each finding carries its skeptic's verdict_notes (and calibration_disputes_verdict where a fresh skeptic disputed it); digest.refuted lists the refuted findings with why — none is ranked.

IDS: every finding and prior has a short id (f<n>, p<n>) in the digest. In matrix.ranked, streams.canonicals and contradictions.between write bare ids and nothing else — no canonicals, no paths, no abbreviation scheme of your own, no area prefixes, no appended flags such as [w] or [p]; the pipeline maps ids back to canonicals, and anything that is not a bare id is reported as unresolved. Name findings by id in exonerations and method too.

JOBS
1. MATRIX: one ranked view per area × primary tag that has findings, ids best payoff first. Payoff = next-change cost removed × breadth (sites, areas) × verdict weight (confirmed 1, weakened 0.5; a prior re-checked this run takes this run's verdict, an unchecked prior keeps its instrument's). Standing findings are listed, never ranked. State the rule once in method and apply it everywhere.
2. STREAMS: group findings that one named remedy removes together. Each stream names the mechanism, what becomes deletable, size and risk, the shared artefacts needing one owner, and mode (agent for mechanical, spawn for design-heavy). Mechanical findings that need no design stay out of streams — the lead files them. Where two viable designs exist, write open_choices with options, trade-offs and the long-term consequence; do not pick unless the analysis converges.
3. INVARIANT CANDIDATES: only rules with a checkable form and a fixture sketch (docs/code-quality.md §Relationship to the design invariants); map to an existing INV when one already encodes it instead of proposing a new one.
4. CONTRADICTIONS: seats proposing conflicting remedies, or a remedy that duplicates a mechanism that already exists; resolve or pose the question. A proposed helper, module or check the tree already has is a duplicate-mechanism contradiction: grep the tree for it before a stream carries the remedy, and when it exists, name it and re-aim the stream at using it.
5. EXONERATIONS: what the metrics flag that the reading shows healthy. An exoneration resting on a test-share ratio or a per-function max alone is not an exoneration: each is one term of a pair, so name the reading of the code that shows the flagged cost is absent (the guard, the derivation, the test that fails on the wrong version), or leave the flag standing.
A weakened finding keeps its weight through its id; the report marks it from the record. Never rank by, or justify a severity with, a measure from the doc's Do-not-steer list.
${critique ? `\nA critic read your previous synthesis and found these join errors. Fix each you accept, say in method how, and fill critic_responses with one entry per item in order (accepted true/false and how); a declined item needs its reason:\n${JSON.stringify(critique, null, 1)}\n` : ''}${mechanical && mechanical.length ? `\nThe pipeline's mechanical join checks found these in your previous synthesis; fix each and say how in method:\n${JSON.stringify(mechanical, null, 1)}\n` : ''}
DIGEST:
${JSON.stringify(digest, null, 1)}`
}

phase('Synthesize')
const degenerateSynthesis = (s) => !s || (s.method || '').length < 50 || (known.size > 0 && !(s.matrix || []).length)
const degenerateCritic = (c) => !c || !['accept', 'revise'].includes(c.verdict) || (c.verdict === 'revise' && !(c.join_errors || []).length)
const synthOpts = (label) => ({ label, phase: 'Synthesize', model: A.routing.synthesis.model, effort: A.routing.synthesis.effort, schema: SYNTHESIS_SCHEMA })
let synthesis = await seatWithRetry('synthesize', (note) => agent(synthesisPrompt(null, null) + note, synthOpts(note ? 'synthesize:retry' : 'synthesize')), degenerateSynthesis)

// Set-membership joins checked by the script, not the critic (failure mode 21): a key in two streams, a key
// twice in one matrix cell, a high structural finding in no stream (mechanical highs are filed by the lead, never streamed). Runs on short ids, before mapIds.
function mechanicalJoinErrors(syn) {
  if (!syn) return []
  const errors = []
  const streamsOf = new Map()
  for (const st of syn.streams || []) for (const id of new Set((st.canonicals || []).map(x => String(x).trim()))) streamsOf.set(id, [...(streamsOf.get(id) || []), st.name])
  for (const [id, names] of streamsOf) if (names.length > 1) errors.push({ check: 'key-in-two-streams', id, canonical: canonicalOf.get(id) || null, where: names })
  for (const cell of syn.matrix || []) {
    const ids = (cell.ranked || []).map(x => String(x).trim())
    for (const id of new Set(ids.filter((x, i) => ids.indexOf(x) !== i))) errors.push({ check: 'duplicate-in-cell', id, canonical: canonicalOf.get(id) || null, where: [`${cell.area}×${cell.tag}`] })
  }
  for (const [c, f] of known) if (f.severity === 'high' && f.class === 'structural' && !streamsOf.has(idOf.get(c))) errors.push({ check: 'high-in-no-stream', id: idOf.get(c), canonical: c, where: [] })
  if (errors.length) log(`synthesis: ${errors.length} mechanical join errors — ${errors.map(e => `${e.check} ${e.id}`).join(', ')}`)
  return errors
}
const firstJoinErrors = mechanicalJoinErrors(synthesis)

function criticPrompt(mechanical) {
  return `You are the critic of an assembled synthesis for a whole-project quality review of ${A.project.id} at ${ROOT}. READ-ONLY; open code where a check needs it.

${treeGuard('notes')}

You are not a third reviewer of the code. You read the JOIN: the places where seats' results were combined. ${QUALITY}attend to its "What to measure" and "Do not steer by" sections. Findings are named by short id (f<n>, p<n>) in both documents below; use those ids in where. Check, and report each hit as a join_error with a fix:
- inconsistent-rule: two figures compared under different rules (one seat counted prose lines, another did not; one measured per function, another per module) — compare the rules figure by figure, not seat by seat.
- one-term-of-sum: a conclusion about complexity or size that reads one term of a pair (module total without the per-function maximum, or the reverse; a split that lowers file size while total complexity rises).
- evidence-contradicts-conclusion: a ranking, stream or exoneration that a seat's own measurements or a skeptic's notes contradict. The skeptic's verdict_notes are in the digest on every finding, with calibration_disputes_verdict where a fresh skeptic disputed it, and digest.refuted lists the refuted ones: check every weakened or disputed finding's rank and stream against its notes.
- do-not-steer-leak: a line count, average complexity, coverage or test count used as ranking input or severity support.
- duplicate-mechanism: a stream remedy that re-creates a mechanism the tree already has (grep for it).
- unverified-fixed: a canonical classed fixed without a confirmed gone verdict.
- severity-unsupported: a high whose next_change scenario does not support it — set severity_to to the severity the scenario does support, so the correction can be applied to the record.
- area-unmapped: a finding whose area is not the area of its anchor path.
- weakened-hidden: a weakened verdict that vanished from the matrix or a stream.
The pipeline has already checked three set-membership joins — a key in two streams, a key twice in one matrix cell, a high structural finding in no stream — and its hits are listed below; do not re-report them. Spend the reading on evidence-contradicts-conclusion and the exonerations: across three runs 17 of 53 critic errors were in exonerations and 8 were duplicate-mechanism.
Verdict: accept, or revise when any error changes a ranking, a stream or a disposition.

MECHANICAL JOIN ERRORS (already found):
${JSON.stringify(mechanical, null, 1)}

SYNTHESIS:
${JSON.stringify(synthesis, null, 1)}

DIGEST (the inputs the synthesis was built from):
${JSON.stringify(digest, null, 1)}`
}

const RECHECK_SCHEMA = { type: 'object', required: ['items'], properties: {
  items: { type: 'array', items: { type: 'object', required: ['join_error', 'resolved', 'checked'], properties: {
    join_error: { type: 'integer', description: '0-based index into the critic join_errors' },
    resolved: { type: 'boolean', description: 'true when an accepted item\'s entry now says what its fix asked, or a declined item\'s reason holds in the code it cites; false for an item the responses do not answer' },
    checked: { type: 'string', description: 'the synthesis entry you read, and for a declined item the code or file you opened' } } } } } }

function recheckPrompt(errors, responses, revisedSynthesis) {
  return `You are a re-check seat for a whole-project quality review of ${A.project.id} at ${ROOT}. READ-ONLY.

${treeGuard("each item's checked")}

A critic listed join errors in a synthesis; the synthesis was revised, and the revision's responses say which items it accepted. Judge an accepted item from the revised synthesis: the named entry (short id, stream or matrix cell) must now say what the item's fix asked. For a declined item, open the code its reason cites and decide whether the reason holds. An item the responses do not answer is not resolved. This is a cheap re-check of the critic's listed items only — do not look for new errors, and do not re-judge the findings. One entry per item, in order.

CRITIC JOIN ERRORS:
${JSON.stringify(errors, null, 1)}

THE REVISION'S RESPONSES:
${JSON.stringify(responses, null, 1)}

REVISED SYNTHESIS:
${JSON.stringify(revisedSynthesis, null, 1)}`
}

phase('Critique')
let critic = null
let criticRecheck = null
let revised = false
if (!synthesis) {
  log('synthesis lost — critic skipped; the lead synthesises from the digest by hand and the report says so')
} else {
  critic = await seatWithRetry('critic',
    (note) => agent(criticPrompt(firstJoinErrors) + note, { label: note ? 'critic:retry' : 'critic', phase: 'Critique', model: A.routing.critic.model, effort: A.routing.critic.effort, schema: CRITIC_SCHEMA }),
    degenerateCritic)
  if (critic && critic.verdict === 'revise') {
    log(`critic: ${critic.join_errors.length} join errors — one revision of the synthesis`)
    const revisedSynthesis = await seatWithRetry('synthesize:revised', (note) => agent(synthesisPrompt(critic.join_errors, firstJoinErrors) + note, synthOpts(note ? 'synthesize:revised:retry' : 'synthesize:revised')), degenerateSynthesis)
    if (revisedSynthesis) {
      synthesis = revisedSynthesis
      revised = true
      const route = A.routing.recheck || { model: 'sonnet', effort: 'medium' }
      criticRecheck = await agent(recheckPrompt(critic.join_errors, synthesis.critic_responses || [], synthesis), { label: 'critic:recheck', phase: 'Critique', model: route.model, effort: route.effort, schema: RECHECK_SCHEMA })
      if (!criticRecheck) log('critic re-check lost — the revision is returned unchecked')
    } else log('revision lost — the unrevised synthesis is returned with the critic errors attached')
  }
}

// Severity corrections the critic argued for, with the revised synthesis's acceptance; the lead's
// digest applies the accepted ones to the findings record (report-format.md §Artefact 2).
const severityCorrections = ((critic && critic.join_errors) || []).map((e, i) => ({ e, i }))
  .filter(({ e }) => e.kind === 'severity-unsupported' && e.severity_to)
  .map(({ e, i }) => {
    const c = canonicalOf.get(String(e.where).trim())
    const f = c ? known.get(c) : null
    const resp = ((revised && synthesis && synthesis.critic_responses) || []).find(r => r.join_error === i)
    return { canonical: c || null, where: e.where, from: f ? f.severity : null, to: e.severity_to, reason: e.detail, accepted: resp ? resp.accepted : null, how: resp ? resp.how : null }
  })
if (severityCorrections.some(s => !s.canonical)) log(`critic: ${severityCorrections.filter(s => !s.canonical).length} severity correction(s) name no resolvable id — listed with canonical null`)

const joinErrorsMechanical = revised ? mechanicalJoinErrors(synthesis) : firstJoinErrors

return {
  run_id: A.run_id, as_of_sha: A.as_of_sha, since: A.since,
  seats: all.map(r => ({ ...r, verdicts: undefined })),
  refuted,
  verification: counts, calibration, yellow_flag_sample: yellowFlagSample, observation, dropped,
  short_ids: [...idOf].map(([c, id]) => ({ id, canonical: c })),
  synthesis: synthesis && { ...mapIds(synthesis), join_errors_mechanical: joinErrorsMechanical }, critic, synthesis_revised: revised, critic_recheck: criticRecheck, severity_corrections: severityCorrections,
}
```

## Model / effort routing

Set in `args.routing` by the lead, explicit on every `agent()`; the session model is never inherited.

| Seat | Model | Effort | Why this tier |
|---|---|---|---|
| area review (per slice) | opus | high | judgment-dense: architectural review against a definition, not a checklist |
| skeptic (per seat) | sonnet | high | bounded verification of concrete claims that rewards thinking; the cheapest seat that reproduces a measurement |
| calibration skeptic (×10, only when the area skeptics refuted under 5 %) | sonnet | high | the same skeptic prompt and falsification protocol on one finding, fresh context; ~60–100k each |
| high-stratum skeptic (one per remaining confirmed high, only when the sample drifts) | sonnet | high | the same seat as the calibration skeptic, over the rest of the high stratum |
| cross-area lens (×4) | opus | high | cross-input synthesis over a graph plus N reports |
| synthesis | fable (standing in the DF overlay; else per the provenance rule) | high | synthesis of conflicting complex reports — the team skill's named fable case; fallback opus / xhigh |
| critic | fable (standing in the DF overlay; else per the provenance rule) | high | adversarial review of that synthesis — the other named case; fallback opus / xhigh |
| critic re-check (one, only after a revision; `args.routing.recheck` overrides) | sonnet | medium | bounded look at whether each listed item is gone from the revised synthesis |

Provenance rule (`skills/team/SKILL.md` §Fable): a fable lead sets the two fable seats directly; an opus lead sets them under the project overlay's standing ruling (`.claude/skills/review-all/project.md`, as dark-factory's grants since 2026-10-05), `--fable`, or a statement in the conversation, otherwise runs the fallback and says in the report that fable was wanted.

Expected per-seat spend, extrapolated from the hotspot run's measured 100–140k for a deep reviewer: area seats 150–250k (they read more), skeptics 60–100k, cross seats 150–250k, synthesis 200–300k, critic 100–200k. Measured on the first dark-factory run: 14.2M subagent tokens over 50 seats for phases 3–4 (20 area, 23 skeptic, 4 cross, 3 fable); SKILL.md §Cost gate carries the whole-run calibration.

## Failure modes

1. **Degenerate structured output passes the schema** (hotspot failure mode 1). `cleanSeat` + `degenerateSeat` check every finding's anchor shape, statement length, measurement presence, next-change scenario and placeholder text; cross seats (`degenerateCross`), the synthesis (`degenerateSynthesis`) and the critic (`degenerateCritic`) have their own checks through `seatWithRetry`. Every seat is re-run once and then dropped with a `dropped` record the report prints; a lost synthesis skips the critic rather than handing it `null`.
2. **Fourteen-times-N trivial findings.** Prevented by the prompt's next-change filter and the `maxItems` cap; detected by the critic's `severity-unsupported` class. If a seat returns `MAX` findings all `low`, the lead reads its `area_health` before trusting any of them.
3. **A split proposed without the heuristic-14 protocol.** Dropped mechanically (`h14 without measurement protocol`); the report lists it so the seat's omission is visible.
4. **Skeptic anchoring and a skeptic that never refutes.** The skeptic never sees proposals or severities (`blind`). Measured on the first dark-factory run: 0 refutations across 203 findings, and 0 across the hotspot refresh (174) and the `/review` run (159) at the same tree — 536 findings, 0 refuted; the five-finding sample all reproduced, which cannot tell "all real" from "skeptic too soft". So the skeptic now records one falsification attempt per measurement (what it ran, what would have refuted it, what it found, whether it reproduced), and `enforceFalsification` downgrades a `confirmed` verdict with any unreproduced measurement to `weakened` (counted in `calibration.downgraded_unreproduced`). A refutation rate under 5 % from the area skeptics (refuted / (confirmed + weakened + refuted)), with more than ten surviving findings, starts the `Calibrate` phase: a stratified sample of ten confirmed findings — every confirmed `high` first, then the rest, each stratum taken as evenly spaced picks over its canonical-sorted list because the script cannot use `Math.random` — goes to one fresh skeptic each (`A.routing.skeptic`, the same `skepticPrompt` and falsification protocol, `falsificationGate` applied without counting into `calibration`), concurrently with the cross lenses, and the result's `yellow_flag_sample` carries the per-finding outcomes (`lost` for a skeptic that returned nothing). The lead's context is the scarce resource, so the re-verification is a seat's job, not the lead's (SKILL.md §Lead context budget): the lead reads the sample's outcomes, the digest copies it to `extra.yellow_flag_sample`, and the lead compares `calibration.refutation_rate` with the previous runs' `verification.refutation_rate`. Those prior figures, `refutation_rate_prior_runs` in the record (`report-format.md` §Artefact 1), are READ by the digest from each previous record's `method.verification.refutation_rate`, never typed (solar_challenge_platform cited reify as 0.0; reify's record says 0.003). What the sample's outcomes feed is failure mode 19. The full-run YELLOW FLAG log still fires on zero refutations across every seat.
5. **Verdict join by display_id.** Unmatched ids fall to `unverified` and are counted separately; never conflate "survived" with "confirmed".
6. **`fixed` without evidence.** A prior finding is `fixed` only when the seat said `gone` and the skeptic confirmed the locating evidence; the critic's `unverified-fixed` is the backstop.
7. **The result is too large to read raw** (expect 300–600 KB). The lead writes it to `<scratch>/workflow-result.json` and digests with a script (SKILL.md Phase 5).
8. **Main moves during a three-hour run.** Every seat reads the detached worktree at `as_of_sha`, never the main checkout; the lead removes the worktree after the report is committed. No stash anywhere (CLAUDE.md §Working in the main checkout). If that worktree is missing when a seat starts, the seat must not fall back to whatever tree it can find: every prompt that reads `ROOT` (area, skeptic, cross, synthesis, critic, re-check) carries `treeGuard`, which points it at one shared `git archive` export of `as_of_sha` at `<scratch>/tree-export`. The first seat to need it builds it in a fresh directory and renames it into place, so seats running at the same time never read a half-written tree and the export is built once, not once per slice.
9. **Budget directive.** `budget` (`total`, `spent()`, `remaining()`) is a documented Workflow script global (`workflow-authoring` reference §Script body hooks). Under a `+Nk` directive, the guard before `phase('Cross-area')` skips the lenses when fewer than ~800k tokens remain (four opus seats plus skeptics) and logs it; synthesis and critic still run on what exists, and the report lists the lenses under Dropped and degraded. Without a directive `budget.total` is null and the guard is inert.
10. **A prior finding silently unchecked.** Legacy hotspot areas `<area>/<sub>` are split in Phase 0 into the plain area and `sub_area` and assigned to the slice holding those files; after the area pipeline the script lists every assigned prior that came back without a verdict under `observation.unchecked`, so a seat that skipped its re-verification duty is visible in the report rather than read as "still open, nothing changed".
11. **`args` too large to pass inline.** With every slice's read list, test patch targets, prior statements and leads embedded, the first dark-factory run's `args` was 820 KB, which the lead would have had to emit as a single tool-call value. The run moved the static inputs to one brief file per seat and one `prior_open.json` (§The slice brief) and passed a 60 KB `args` of identities, paths and counts; that is now the shape. The script derives the open prior set from `slices[].prior_open` and never carries a second copy. Even 60 KB is a large tool-call value; the zero-paste alternative, used on the first reify run, is to write `const A = <args JSON>` into the script file in place of `const A = args` and launch with `scriptPath` only (`args` unused) — the journal's replay key is each call's prompt and opts, which this leaves unchanged.
12. **Seat name drift.** Two of 19 area seats echoed `seat` without the area prefix (`harness-agents`, `task-layer-curator`); the post-pipeline join `seats.find(x => x.seat === s.seat)` missed them and 41 priors they had in fact re-verified were reported as `unchecked`. `reviewSeat` now overwrites `seat` and `area` from the slice after the call (logging a mismatch), and the cross arm sets its own `seat`; a seat's echo is never a join key.
13. **The synthesis invents an abbreviation scheme.** Asked to repeat 480 canonicals in the matrix, the fable synthesis shortened them its own way — path tokens (`O:=` for the orchestrator package), omitted `<area>|` prefixes, trailing `[w,p,m]` flags, `alias:filed-by-anchor` notes — and the lead's digest had to reverse-engineer it. The script now assigns short ids (`p<n>` priors, `f<n>` findings, one per canonical), the prompts forbid anything but bare ids in `ranked`, `canonicals` and `between`, and `mapIds` maps them back before returning, listing anything unresolvable under `synthesis.unresolved_ids`; `short_ids` in the result lets the lead's digest read the ids the critic prose and `method`/`exonerations` still use.
14. **An accepted critic correction never reaches the record.** The critic's `severity-unsupported` on `fk-3505aba0481d` (`evals/runner.py::run_architect_eval`, high → medium) was accepted in the revised synthesis, but the findings record kept the seat's `high` and the lead had to rule at decompose. The critic now states `severity_to`, the revision answers each item in `critic_responses`, and the script returns `severity_corrections` with the acceptance; the lead's digest applies the accepted ones and flags the rest (`report-format.md` §Artefact 2).
15. **A file larger than the read budget.** `merge_lane/worker.py` was 22,081 lines against a 15k budget; the seat read 9,970 verbatim and the rest as a code-only rendition, and nothing in `read_coverage` said so. The area prompt now prescribes the protocol (symbol index, module-level state, functions in descending cognitive max from the snapshot's `functions` table) and `read_coverage.partial` records it, so the report can show what was not read.
16. **A usage cap mid-run.** Measured on the first reify run (`review-all-reify-20261010`): the account's weekly usage limit fired at 11.2M subagent tokens, 37 of 70 agents failed with "usage limit", and the workflow kept dispatching until every remaining call had failed — `agent()` returns `null` on a terminal API error, so each failure was logged as a degenerate seat and its retry burned another call. Such a run must be **resumed, not restarted**: `Workflow({scriptPath, resumeFromRunId})` once the limit resets. But a resume replays only the longest unchanged prefix of `agent()` calls *in journal order*, and under `pipeline()` (and the Calibrate phase running beside the cross lenses) the completion order differs between runs, so every call after the first divergence runs live again: on reify 5 already-completed area seats and 6 completed skeptics re-ran, about 1.2M tokens. Two rules follow. (a) Never stop a run to patch prompts for seats that have not run yet — an edited prompt moves the divergence point earlier still. Let it finish, then re-run only the changed tail as a standalone workflow over the completed results: a script that rebuilds the synthesis digest from `workflow-result.json` (seat identity from the slice list, short ids, `prior_ids`, `calibration`), embeds it with the current `orchestration.md`'s Calibrate (with its high-stratum step), Synthesize and Critique code lifted verbatim, and runs Calibrate → Synthesize → Critique (the reify run's `build_synth_input.py` + `gen_synth_workflow.py` pair, kept with its intermediates, is the shape to lift; its paths are that run's). (b) At the cost gate, check the account's remaining weekly headroom against the roster's estimate, Phase 2 included (SKILL.md §Cost gate); a run that will not fit waits for the reset rather than being cut by the cap mid-pipeline.
17. **An anchor-shape drop is silent to the seat.** `cleanSeat` drops a finding whose anchor fails `ANCHOR_RE` and never re-asks, so neither its skeptic nor the synthesis sees it. Measured on the first reify run: 15 of 347 findings dropped for `anchor shape`, 3 of them `high` — Rust method anchors `path.rs::Type::method` (`engine_eval.rs::Engine::eval_cached`, `::Engine::evaluate_let_bindings`, `reify-kernel-occt/src/lib.rs::OcctKernel::execute`), which the grammar's single `::symbol` rejects; the lead recovered them from the journal by hand. `normAnchor` now maps a symbol path after the file onto the class-level form (`::Type::method` → `::Type.method`, in parity with `findings_artefact.py::normalise_anchor`) before the shape check, and `cleanSeat` logs each normalisation; other shape failures are still dropped and listed under `dropped`, so read that list before trusting a seat's count.
18. **Seats read about half their plan.** Measured on the first reify run: 30 area seats handed 427k lines (15k each) reported 234k read; 12 read under half, typically stopping 4–8k lines into a slice of two 13–15k-line files, and none of the 9 listed files with a per-function CCN max ≥ 60 was reported read in full. The area prompt now states the budget in lines, tells the seat to read until it is spent and to take a file it cannot finish through the large-file protocol (failure mode 15), and `reviewSeat` records `read_coverage.planned_lines` beside the seat's own `lines_read`, so the report's read-coverage table shows the shortfall per seat. The lead sizes read lists at ~8–12k lines and rotates them across runs (SKILL.md Phase 1 §Read shortfall, §Rotation).
19. **Calibration outcomes went nowhere.** Measured on reify: 5 of 10 sampled highs came back `weakened` (solar_challenge_platform: 2 of 10), and every one stayed `confirmed` in the record, the matrix and the streams — `calibrate()` returned its outcomes and nothing consumed them; it also ran only at zero refutations. Now it runs under a 5 % refutation rate (failure mode 4); `recordOutcome` attaches `calibration_disputes_verdict: {to, note}` to each sampled finding a fresh skeptic weakens (or refutes) and leaves the finding's own verdict standing — one skeptic does not silently overwrite another, the deliberation rules (`report-format.md` §Artefact 2); and a sample with ≥ 30 % disputed or ≥ 2 disputed highs is the drift signal: the `Reverify high stratum` step gives every remaining confirmed `high` a fresh skeptic (`A.routing.skeptic`), records outcomes the same way under `calibration.high_stratum`, and sets `calibration.high_stratum_reverified = true`. The sample is awaited before the digest, so the synthesis and the critic see the disputes.
20. **Skeptic notes never reached the join.** `digestFindings` carried the verdict but not the skeptic's note, so a weakened finding reached the synthesis uncorrected and the critic could not check it (reify's critic worked "vs fNN skeptic notes" by hand; solar_challenge's: "skeptic notes are absent"). The digest now carries each finding's `verdict_notes` and `calibration_disputes_verdict`, and `digest.refuted` the refuted list (the one the result returns), and the critic prompt tells it to check weakened and disputed findings against them.
21. **Critics re-found set-membership errors by hand.** All three critics reported the same structural join errors, and the lesser items never reached the revision. `mechanicalJoinErrors` now runs after the synthesis and again after a revision — new: a key in two streams, a key twice in one matrix cell, a high structural finding in no stream (mechanical highs are filed by the lead, not streamed; priors carry no severity in the script and are not checked) — logs them, feeds the first set to the critic (as already found) and to the revision, and returns the final set as `synthesis.join_errors_mechanical`. Kept unchanged: the critic's `weakened-hidden` class and the pipeline's duplicate-canonical-across-seats log. The critic's budget goes to evidence-contradicts-conclusion and exonerations (17 of 53 critic errors across three runs were in exonerations; 8 were duplicate-mechanism).
22. **A revision nobody checked.** The revised synthesis answered the critic's items in `critic_responses`, and the lead's digest took those answers on trust. One sonnet/medium seat now makes a cheap re-check of exactly the critic's listed items (`recheckPrompt`): an accepted item against the revised synthesis, a declined item's reason against the code it cites. The result returns as `critic_recheck`; an item with `resolved: false` is the lead's to rule on at the digest. The same revision carries the two checks the critic kept finding by hand — a proposed helper the tree already has, and an exoneration resting on a test-share ratio or a per-function max alone (synthesis jobs 4–5) — and the area seat asks of each high whether the wrong version is caught, and by what (every one of the four critic downgrades was high → medium on that basis).
