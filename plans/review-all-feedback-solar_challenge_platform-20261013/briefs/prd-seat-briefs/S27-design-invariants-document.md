# /prd seat brief — S27 design-invariants-document (agent mode)

Program: `/home/leo/src/solar-challenge-platform/plans/review-all-program-solar_challenge_platform-2026-10-14.md` — READ IT FIRST, in full (section "Design invariants promoted (ruling R19)" is your scope). Run `review-all-solar_challenge_platform-20261013`, record `/home/leo/src/solar-challenge-platform/plans/review-all-solar_challenge_platform-20261013.json` (its `synthesis.invariant_candidates` holds the ten candidates with statement, heuristic, checkable_form and fixture_sketch). Pinned tree `40a65566270dc43caf55e8f82232d61b3b558b0d`; main has moved.

## Ruling (do not relitigate)

R19: each of the ten invariant candidates lands as a checkable test inside its owning stream AND this leaf creates `docs/legibility/design-invariants.md` as the project's single invariants list in checkable form (the project had none). The program doc carries one pointer per invariant and owning stream, never a second list (INV-9 one-fact-one-home).

## Scope

One small PRD (`docs/prds/review-all-design-invariants-document.md` + `.capability-manifest.yaml`) whose leaves: (1) create `docs/legibility/design-invariants.md` with the ten invariants as INV-1..INV-10 (id, statement, heuristic, checkable form, the owning stream S<n> and the test that will enforce it, status `proposed` until that test lands), with the same shape dark-factory's `docs/legibility/design-invariants.md` uses (read it at `/home/leo/src/dark-factory/docs/legibility/design-invariants.md` for the shape; do not copy its content); (2) land the one gate this leaf itself owns: the repo grep gate "a rationale pointer in a docstring names a tracked path" (candidate 10) as a pytest under tests/; (3) point `review/briefing.yaml` conventions at the new document (one line) and CLAUDE.md's project section at it (one line). Consumer (G1): /prd G7 and /review phase 2 read the document; the orchestrator's reviewer prompt cites it. Leaf signal (G2): the document exists with ten entries and the grep gate is green on main (today's tree has `.task/plan.json` pointers that fail it — list them as the fixture's seed and route their removal to the streams that own those files: S13 consent/gate.py+store.py, S19/S5 ingestion, S26 timeseries, S1 billing/statement.py; do not fix them here).

## Finding keys

None directly; this leaf serves every stream's invariant. Stamp `metadata.x_finding_run = review-all-solar_challenge_platform-20261013`, `source = review-all`, and `x_finding_key = []` on its tasks.

## What you return (author pass; no commits, no filing, no edits to the program doc)

{prd_path, manifest_path, open_questions, seam_dependencies_on_other_streams, cross_reference_lines, leaves_estimate}
