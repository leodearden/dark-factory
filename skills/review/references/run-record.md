# Run record schema

One run writes one committed report (`docs/quality-findings-contract.md`
§5): `review/reports/<run_id>.json`, plus its rendering
`review/reports/<run_id>.md`. The JSON is the record; the `.md` is
generated from it in the same run and adds nothing of its own. A rerun
never overwrites an earlier report; `--phase triage` completes the report
of the run it triages and keeps that run's `run_id` and `as_of_sha`.

## Method header

The JSON's `method` object and the `.md`'s `## Method` section carry the
same keys (§5): exactly `run_id`, `as_of_sha`, `since`, `evidence`,
`verification`, `cost` and `inputs_consumed`, with every /review-only key
under `extra`.

```json
"method": {
  "run_id": "review-<project_id>-20261004",
  "as_of_sha": "<40-hex main commit the pinned tree was built from>",
  "since": "<40-hex> | none",
  "evidence": {
    "changed_files": 41,
    "phase2_files": 63,
    "carried_findings": 12,
    "briefing": "review/briefing.yaml (last_updated 2026-10-04)"
  },
  "verification": {"confirmed": 0, "weakened": 0, "refuted": 0, "unverified": 0},
  "cost": {"agents": 9, "subagent_tokens": 812000, "wall_clock_min": 74},
  "inputs_consumed": ["hotspot-survey-<project_id>-20260930", "review-all-<project_id>-20261003", "codebook@<as_of_sha>"],
  "extra": {
    "project_id": "<project_id>",
    "mode": "full | since",
    "full_reason": "--full | no prior report | since not an ancestor | yardstick changed | null",
    "scope": "full | <area> | focused:<path>,<path>",
    "phases_run": [1, 2, 3],
    "findings_only": false,
    "launched_from": "esc-6301-1 | null",
    "briefing_defects": ["subprojects: workspace member cockpit has no entry", "known_gaps[escalation][1]: no accepted_by"],
    "trigger_chain": {"form": "final | interim | none", "task_ids": [6301], "superseded": [6188], "reason": null}
  }
}
```

`subagent_tokens` is `null` when the runtime does not report it.
`verification` counts every finding in the report, carried ones included.

## Phase 1 block

Phase 1 output is operational, not findings (contract §1: counts and
caveats are report prose). A red main is operational breakage, never a
quality-finding severity (§4).

```json
"phase1": {
  "commands": [
    {"name": "test", "cmd": "<as run>", "cwd": "<pinned tree>", "exit": 1,
     "passed": 7389, "failed": 2, "skipped": 4}
  ],
  "failures": [
    {"test": "tests/test_x.py::test_y", "member": "orchestrator",
     "error": "<first relevant line>", "classification": "new | known-flake | pre-existing",
     "owner_task": 6120}
  ],
  "lint": {"exit": 0, "issues": []},
  "typecheck": {"exit": 0, "errors": []},
  "smoke": [
    {"name": "<briefing what_working_means line>", "cmd": "<constructed>", "passed": false,
     "diagnosis": "Connection refused — service not running"}
  ]
}
```

## Findings

Every Phase 2 step, the project audit and the carried re-verification
write into one list with one schema. The fields and their rules are
contract §1; the key is minted per §2.

```json
"findings": [
  {
    "key": "fk-3f9a0c41d2e7",
    "display_id": "R3",
    "area": "orchestrator",
    "anchor": "orchestrator/src/orchestrator/harness.py::Harness",
    "tags": ["h7", "inv-9", "kind:deep-read"],
    "severity": "high",
    "evidence_source": ["present-tree"],
    "statement": "Harness reads the merge worker's private state directly at 4 call sites ...",
    "proposal": "Expose a snapshot accessor on the merge worker ...",
    "class": "mechanical | structural",
    "verdict": "confirmed",
    "disposition": "open | filed:<id> | accepted:<id> | refuted | fixed:<sha>",
    "dedup_step": "1a | 1b | 1c | 2 | 3 | not-routed",
    "supersedes": [],
    "sub_area": null,
    "first_seen": "review-<project_id>-20260930",
    "last_seen": ["review-<project_id>-20261004"],
    "change_note": null
  }
]
```

Rules /review adds on top of the contract:

- **`tags[0]` is the lens** — `h<n>`, `comments`, `tests` or `inv-<n>` —
  because the key hashes it (§2). `kind:<step>` tags follow, from
  `kind:audit`, `kind:stub`, `kind:critical-path`, `kind:deep-read`,
  `kind:cross-module`, `kind:invariant`, `kind:dead-code`, `kind:coverage`,
  `kind:convention`. A wrong-behaviour bug that no heuristic honestly
  explains takes `kind:defect` as `tags[0]`; two such bugs at one anchor
  become one finding whose statement lists both.
- **`area`** is a plain `subprojects` key of the briefing or `repo` (§3), never `<area>/<sub>`; it is what the key hashes. A finer grain goes in the optional `sub_area`, which is never part of the key.
- **`supersedes`** is a list of keys, usually empty; a moved anchor appends the old key (§2). With no
  briefing, use the workspace member name and list the gap in
  `briefing_defects`.
- **`class`** is contract §6. A proposal to split a file or reduce a number
  is `structural`.
- **`dedup_step`** records which contract §8 step decided the disposition;
  `not-routed` for structural findings and for runs that skipped Phase 3.
- **`last_seen`** is a list: each run that re-observes the key appends its
  `run_id` (§9). `first_seen` is copied from the earliest report, of any
  instrument, that carried the key.
- **`change_note`** says which field a re-observation changed (verdict,
  severity, anchor) and why; `null` when nothing changed.
- `refuted` and `fixed:<sha>` findings stay in this report and are not
  carried to the next run.

## Severity from the old vocabulary

Reports before 2026-10 used `blocking | high | warning | info`. Map
`blocking`/`high` → `high`, `warning` → `medium`, `info` → `low`; emit only
the three contract values (§4). Task priority equals severity.

## Project audit block

When Phase 2 Step 1 ran the project's `/audit`, its raw result is kept for
the audit trail, and each audit finding is also converted into a
`findings` entry tagged `kind:audit` whose disposition is the task or
escalation the audit already created:

```json
"f_infra": {
  "audit_skill_present": true,
  "since_iso": "<committer date of since>",
  "escalated": [{"pattern": "P5", "escalation_id": "esc-3520-2", "key": "fk-…"}],
  "filed_task_ids": [3681, 3682],
  "logged_only": 3
}
```

## Markdown rendering

`review/reports/<run_id>.md` is rendered from the JSON and never edited by
hand. Under the H1, its `## Method` section's first element is a fenced
`yaml` block holding the `method` object (the seven keys, then `extra:`),
so a script reads it with one YAML load. Then the interactive summary from
SKILL.md, then one table of findings: `display_id | key | area | anchor |
tags | severity | class | verdict | disposition`.
