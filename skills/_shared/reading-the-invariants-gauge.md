# Reading the Invariants Gauge — Canonical Reference

This document is the single statement of the watchers' once-per-cycle read of the claimant-invariant
gauge, `GET /api/v2/dashboard/invariants`. Two skills point here:

- [`skills/escalation-watcher/SKILL.md`](../escalation-watcher/SKILL.md) — Main Loop step 7
- [`skills/recon-escalation-watcher/SKILL.md`](../recon-escalation-watcher/SKILL.md) — Main Loop step 8

Each skill owns two things itself: which roots it covers, and where its findings are handled.

---

## The read

The dashboard serves the gauge at `http://127.0.0.1:8080` (OPERATIONS.md §"Dashboard"). The body
is about 23 KB (measured 2026-10-12), so never print it whole. Read this digest instead:

```bash
out=$(mktemp /tmp/invariants.XXXXXX.json)
if curl -sS -m 30 --fail -o "$out" http://127.0.0.1:8080/api/v2/dashboard/invariants; then
python3 - "$out" <ROOT-OR-NOTHING> <<'PY'
import json, sys
body = json.load(open(sys.argv[1]))
only = sys.argv[2] if len(sys.argv) > 2 else None
print('served_at', body['served_at'])
for r in body['INVARIANTS']:
    if only and r['root'] != only:
        continue
    where = f"{r['project']} ({r['root']})"
    if r['status'] == 'blind':
        print(f"FINDING blind {where}: {r['error']}")
        continue
    unledgered = [v for v in r['terminal_tier']['violations'] if not v['ledgered']]
    if unledgered:
        ids = ' '.join(f"{v['tag']}#{v['task_id']}" for v in unledgered)
        print(f"FINDING terminal {where} unledgered={len(unledgered)}: {ids}")
PY
else
  echo "GAUGE UNREADABLE: curl rc=$?"
fi
rm -f "$out"
```

Replace `<ROOT-OR-NOTHING>` with the watched `<project_root>` to read only that root's record, or
delete it to read every root. Each `FINDING` line is one finding. A `served_at` line with no
`FINDING` line under it is a clean reading.

The record shapes are `dashboard/src/dashboard/data/root_invariants.py::RootInvariants.to_wire`
and `dashboard/src/dashboard/data/claimant_invariants.py::TerminalTier.to_wire`. The contract is
E-1 in `docs/prds/claimant-invariant-detection.md`.

## A failed read is not a clean read

`GAUGE UNREADABLE` means curl exited non-zero (with `--fail`, an HTTP error is rc 22), so there is
no reading this cycle. A Python traceback in place of `served_at` is the same. Either is a finding
in its own right: a dashboard outage, under the skill's Priority 1 (system stability). Never carry
the previous reading forward as current. (Detection PRD D-2: blind ≠ clean.)

## What is a finding

1. A root whose `status` is `blind`. It makes no count claim, and its `error` says why.
2. A root whose `terminal_tier.unledgered_count` is above 0. The digest lists each unledgered row
   as `tag#task_id`.

Not findings: ledgered violations, which are sanctioned C4-E2 mints (detection PRD D-5), and the
`hygiene_tier`, which is reported but never alarmable (`docs/prds/claimant-invariant-enforcement.md`
D3; detection PRD "Out of scope").

## Triage only what changed

The gauge re-reports standing residue every cycle. On 2026-10-12 that was 77 and 43 unledgered
rows on two roots, plus two blind roots. Each finding has an identity:

- blind: (root)
- terminal: (root, `tag#task_id`), one per unledgered row

Triage only identities absent from the previous reading. Keep the identities already triaged, and
their dispositions, in top-level context. Carry undisposed ones across sessions the way the skill
carries its other work. The first reading of a session is all new, so check the handover
(escalation-watcher) and existing tasks before triaging it.

## Evidence and the deeper dive

- Blind: the `error` string, and whether `<root>/.taskmaster/tasks/tasks.db` exists and is
  non-empty.
- Terminal: the listed rows.

Then run the census (detection PRD Open question 2):

```bash
cd $DARK_FACTORY_ROOT && uv run --project fused-memory python scripts/clear_leaked_claimants.py --project-root <root> --json
```

Without `--apply` it is a read-only dry run that never dials the MCP server. `--apply` writes
claimants. That is a disposition the skill decides, never part of this read.

## What a nonzero unledgered tier can mandate

`docs/prds/claimant-invariant-detection.md` §"D-4" makes any post-ζ observation of a nonzero
unledgered terminal tier the trigger that mandates filing ε.2's work. §"D-7" pre-designs that
record's shape. Read both there before filing.

## Filing a task for a finding

Both watchers read this endpoint, so first run `mcp__fused-memory__search_tasks` for an existing
task naming that root and finding. Then file with the Case B keys of
[`ticket-failure-handling.md`](ticket-failure-handling.md), derived here so that both watchers
mint the same R4 key:

```python
import hashlib

escalation_id = f"invariants-{kind}-{project}"   # project: the finding record's `project`
suggestion_hash = hashlib.sha256(f"{root} {kind} {identity}".encode()).hexdigest()[:16]
```

| `kind` | `identity` |
|---|---|
| `blind` | the record's `error` string |
| `terminal` | the space-joined, sorted `tag#task_id` list the filing covers |
