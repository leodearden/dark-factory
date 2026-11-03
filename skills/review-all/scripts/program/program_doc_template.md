# /review-all remediation program — ${project_id} — ${run_id}

Run `${run_id}`, pinned at `${as_of_sha}`; run record `${record_path}`. Rendered by `program_doc.py` from the run home; never hand-edited.

## Streams

${streams}

## Routed keys

${routed}

## Seam ownership (G4 — authoritative)

${seams}

## Invariants

${invariants}

## Resolved decisions (do not relitigate)

${decisions}

## Shared conventions (every session)

- Every task the program files carries `metadata.x_finding_key` (the keys it closes), `metadata.x_finding_run = "${run_id}"` and `metadata.source = "review-all"`. The rest of the filing mechanics, the held-leaf convention and the one-editor rule are `skills/review-all/references/report-format.md` §Artefact 3's.
- Freshness: every anchor was verified at `${as_of_sha}`. Before building a task on one, run `git diff ${as_of_sha}..main -- <path>` and re-read the symbol.
${conventions_extra}

## Triage briefing

${triage}

## FILED — program status

${filed}

## Held leaves

${held}

## Ledger

${ledger}
