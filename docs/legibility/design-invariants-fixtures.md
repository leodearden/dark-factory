# Design-invariants fixtures

Calibration fixtures and a rehearsal verdict table for the two consumers of
`docs/legibility/design-invariants.md`: `/prd` decompose's G7 gate
(`skills/prd/references/gates.md`) and `/review` phase 2's Step 5.5
design-invariants audit (`skills/review/references/phase2-architecture.md`).

**Normative source.** `docs/legibility/design-invariants.md` is the single
normative copy of the five invariant slugs, rules, and checkable design
questions — this doc does not restate them (per INV-5
`no-lockstep-duplication`). When in doubt about a rule or a checkable
question, Read the normative doc, not this one.

**Two fixture shapes.** Each invariant below carries exactly two seeded
violations — both expressions of the SAME underlying violation, so the two
gate consumers stay calibrated against one shared meaning per slug:

- **PRD-leaf-shaped** — a realistic 2-4-line decomposition-plan row: the
  shape `/prd` decompose's G7 walk (section "G7 — Design invariants pass",
  `skills/prd/references/gates.md`) sees when it walks a batch.
- **Code-snippet-shaped** — a short illustrative snippet or described
  module shape: the shape `/review` phase 2's Step 5.5 design-invariants
  audit (`skills/review/references/phase2-architecture.md`) sees when it
  audits modules in scope. File paths in these snippets are illustrative —
  chosen to NOT collide with any real file in this repo — not pointers
  into the actual codebase.

**Expected-verdict formats.**

- PRD-leaf-shaped fixtures are annotated with the expected G7 disposition —
  `flag: <slug>` — plus the redesign that clears it, so the fixture also
  demonstrates the fix, not just the failure.
- Code-snippet-shaped fixtures are annotated with the expected `/review`
  Step 5.5 finding: an `invariant_findings` entry
  `{"invariant": <slug>, "file": ..., "line": ..., "issue": ..., "severity": ...}`
  (schema per phase2-architecture.md Step 8), with `severity` drawn from
  `{high, warning, info}`.

**Rehearsal verdict-table legend.** The table at the end of this doc walks
the AS-LANDED G7 text against every PRD-leaf-shaped fixture, and the
AS-LANDED Step 5.5 text against every code-snippet-shaped fixture, then
records the verdict each yields against the expected slug. Columns:

| Column | Meaning |
|---|---|
| Fixture ID | `<INV-n>-<PRD\|CODE>` — identifies the fixture block and shape below |
| Shape | `PRD` (walked against G7) or `CODE` (walked against Step 5.5) |
| Invariant | The numeric alias + slug being targeted |
| Expected slug | The exact slug string the gate/review text should emit |
| Verdict | The disposition the as-landed gate/review text actually yields when walked against the fixture |
| Match | `Y` if the verdict's slug equals the expected slug, else `N` |

Acceptance: every fixture flags with the correct slug — all 10 rows `Y`.

## INV-1 `contracts-machine-checked`

## INV-2 `structured-facts-at-failure`

## INV-3 `corroborate-before-acting`

## INV-4 `storm-escape-required`

## INV-5 `no-lockstep-duplication`

## Rehearsal verdict table
