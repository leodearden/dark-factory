# Capability manifest — plans/design-invariants-gate-prd.md

Bindings audited 2026-07-13 against main @ 7517a50cd9 (PRD commit). Evidence
verified by direct read/grep in the authoring session.

## Leaf ε — fixtures + rehearsal (the batch's sole G2 leaf)

| Capability asserted | Evidence | Verdict |
|---|---|---|
| `docs/legibility/design-invariants.md` with the five stable slugs | producer: task α, upstream of ε; extent = five entries each carrying slug/rule/question/evidence/pattern + census-seam + fixture-pointer paragraphs (PRD §The-five-invariants specifies the full content) | PASS producer-upstream |
| gates.md §G7 text to rehearse against | producer: task β, upstream of ε; extent = SKILL.md gate row + §G7 section + both application-order insertions + decompose-mode step (PRD §G7-gate-spec) | PASS producer-upstream |
| phase-2 invariant-audit step to rehearse against | producer: task γ, upstream of ε; extent = Step 5.5 + `invariant_findings` schema + SKILL.md bullet (PRD §review-consumption-spec) | PASS producer-upstream |
| Rejection assertion: "G7 / phase-2 flag each seeded violation with the correct slug" | rejection-check bound **in-task**: ε authors the violating fixtures (the X), walks the as-landed β/γ text (the substrate check), and records the observed flag per fixture in the rehearsal verdict table; a miss is a wording defect fixed within ε's prose scope | PASS rehearsal-in-task |
| `docs/legibility/` directory on main | `docs/legibility/confusion-codebook.yaml` committed 0691d13263 | PASS wired |

DAG direction: α, β, γ all upstream of ε — no inversion.

## Intermediates (verification evidence, not leaf bindings)

- **α**: consumers β/γ/ε named in-batch. Substrate: `docs/legibility/`
  exists (above); `CLAUDE.md:356` `## Reference` list exists for the pointer
  line; survey §3 source committed
  (`plans/agent-legibility-survey-2026-07-13.md`, 0691d13263). The invariant
  content is fully specified in the PRD table — no external capability
  assumed.
- **β**: consumer ε named in-batch (plus every future decompose session).
  Substrate anchors on main: `skills/prd/SKILL.md:59` Gates table (G6 row
  :70); `skills/prd/references/gates.md:135` §G6 / `:162` §Capability
  Manifest (G7 inserts between) / `:205`+`:217` application orders;
  `skills/prd/references/decompose-mode.md:38` Step 2.5 / `:44` Step 3
  template (`:72` metadata block for the `g7_waivers` line).
  `~/.claude/skills/prd` → repo symlink verified 2026-07-13 (readlink) — the
  edit serves interactive and project use with no second copy.
- **γ**: consumer ε named in-batch (plus every future /review session).
  Substrate anchors: `skills/review/references/phase2-architecture.md:163`
  Step 5 (Step 5.5 inserts after) / `:196` Step 6 / `:240` Step 8 report
  schema; `skills/review/SKILL.md:109` Phase-2 step list.

## Cross-batch citations (not deps)

Tasks 2563 (routing lint, pending) and 2558 (structured evidence + streak
gate, pending) are cited in the doc as point-enforcement precedents of
INV-1/INV-2/INV-4 — prose citations only; no dependency edges (the doc is
correct whether or not they have landed).

No grammar fixtures (no DSL surface), no numeric floors (no quantitative
bounds anywhere in the batch), no field-population checks (no runtime result
values). No FAIL bindings — batch clear to queue.
