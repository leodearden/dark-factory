# Reading the Invariants Gauge — Canonical Reference

This document is the single statement of the watchers' once-per-cycle read of the invariants gauge,
`GET /api/v2/dashboard/invariants`: its claimant tiers and its recurrence chains. Two skills point
here:

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
import json, os, sys
body = json.load(open(sys.argv[1]))
only = os.path.realpath(sys.argv[2]) if len(sys.argv) > 2 else None
print('served_at', body['served_at'])
matched = 0
for r in body['INVARIANTS']:
    if only and os.path.realpath(r['root']) != only:
        continue
    matched += 1
    where = f"{r['project']} ({r['root']})"
    if r['status'] == 'blind':
        print(f"FINDING blind {where}: {r['error']}")
        continue
    unledgered = [v for v in r['terminal_tier']['violations'] if not v['ledgered']]
    if unledgered:
        ids = ' '.join(f"{v['tag']}#{v['task_id']}" for v in unledgered)
        print(f"FINDING terminal {where} unledgered={len(unledgered)}: {ids}")
    for c in r['chains']:
        link = f"{c['tag']}/{c['key']} link #{c['link_id']} {c['link_status']} at={c['at']}"
        if c['state'] in ('overdue', 'broken'):
            print(f"FINDING chain-{c['state']} {where} {link}")
        elif c['state'] == 'paused':
            print(f"paused {where} {link} pinned_by={c['pending_escalation_ids']}")
if only and not matched:
    roster = ' '.join(r['root'] for r in body['INVARIANTS'])
    print(f"FINDING absent {only}: no record in roster (roots: {roster})")
PY
else
  echo "GAUGE UNREADABLE: curl rc=$?"
fi
rm -f "$out"
```

Replace `<ROOT-OR-NOTHING>` with the watched `<project_root>` to read only that root's record, or
delete it to read every root. The digest compares both roots after `os.path.realpath`, so a
trailing slash or a symlinked spelling of the watched root still matches, and a root that matches
no record prints `FINDING absent`. Each `FINDING` line is one finding, and so is a `paused` line
reading `pinned_by=[]` (see "What is a finding"). A `served_at` line with neither under it is a
clean reading.

The record shapes are `dashboard/src/dashboard/data/root_invariants.py::RootInvariants.to_wire`,
`dashboard/src/dashboard/data/claimant_invariants.py::TerminalTier.to_wire` and
`dashboard/src/dashboard/data/recurrence_gauge.py::ChainEntry.to_wire`. The contracts are E-1 in
`docs/prds/claimant-invariant-detection.md` and C-4 in `docs/prds/recurring-deterministic-tasks.md`.

## A failed read is not a clean read

`GAUGE UNREADABLE` means curl exited non-zero (with `--fail`, an HTTP error is rc 22), so there is
no reading this cycle. A Python traceback in place of `served_at` is the same. Either is a finding
in its own right: a dashboard outage, under the skill's Priority 1 (system stability). Never carry
the previous reading forward as current. (Detection PRD D-2: blind ≠ clean.)

## What is a finding

1. A root whose `status` is `blind`. It makes no count claim, and its `error` says why.
2. A root whose `terminal_tier.unledgered_count` is above 0. The digest lists each unledgered row
   as `tag#task_id`.
3. A recurrence chain whose `state` is `overdue` or `broken`, the alarm-shaped states
   (`docs/prds/recurring-deterministic-tasks.md` C-4; state function
   `dashboard/src/dashboard/data/recurrence_gauge.py::chain_state`).
4. A root filter that matched no record (`FINDING absent`). The gauge said nothing about that
   root, so this is not a clean reading (detection PRD D-2: blind ≠ clean). The printed `roots:`
   list is the dashboard's roster, `dashboard/src/dashboard/project_dbs.py::project_roots`:
   `DASHBOARD_PROJECT_ROOT` first, then `DASHBOARD_KNOWN_PROJECT_ROOTS`
   (`dashboard/src/dashboard/config.py::DashboardConfig.from_env`). Either the watched project is
   missing from that roster, or the root argument names a path that does not resolve to the
   dashboard's spelling of it.

Not findings: ledgered violations, which are sanctioned C4-E2 mints (detection PRD D-5), and the
`hygiene_tier`, which is reported but never alarmable (`docs/prds/claimant-invariant-enforcement.md`
D3; detection PRD "Out of scope"). Nor are chains in state `scheduled` or `ended`.

A `paused` chain is not this rule's. Its hold is owned by its pinning escalation, listed in the
entry's `pending_escalation_ids` (a `milestone_check_failed` L2 on the orchestrator queue): inspect
it there. Two edge cases:

- `null` (the digest prints `pinned_by=None`) means the pinning lookup is unknown: the
  orchestrator queue was not fully read, or the link is outside the default tag, which that queue
  knows nothing of. Re-read it next cycle. If it stays `null`, read the link's escalations with
  `mcp__escalation__get_task_escalations`.
- `[]` means no pending escalation pins the hold. Nothing owns it, so treat that one entry as a
  finding.

## Triage only what changed

The gauge re-reports standing residue every cycle. On 2026-10-12 that was 77 and 43 unledgered
rows on two roots, plus two blind roots. Each finding has an identity:

- blind: (root)
- terminal: (root, `tag#task_id`), one per unledgered row
- chain: (root, `state`, `tag`, `key`, `link_id`). The state is part of it, because one link can
  read `overdue` and then, run but left without a successor, `broken`.
- absent: (root), the normalised root the digest printed

Triage only identities absent from the previous reading. Keep the identities already triaged, and
their dispositions, in top-level context. Carry undisposed ones across sessions the way the skill
carries its other work. The first reading of a session is all new, so check the handover
(escalation-watcher) and existing tasks before triaging it.

## Evidence and the deeper dive

- Blind: the `error` string, and whether `<root>/.taskmaster/tasks/tasks.db` exists and is
  non-empty.
- Terminal: the listed rows.
- Absent: the printed `roots:` list, compared against `realpath <root>` of the watched root. A
  root missing from the list is a gap in the dashboard's roster, and the knob for it is
  `DASHBOARD_KNOWN_PROJECT_ROOTS`. A root present under another spelling means the watcher's
  argument needs fixing. There is no record behind an absent finding, so the census below does
  not apply to it.

For a blind or terminal finding, run the census next (detection PRD Open question 2):

```bash
cd $DARK_FACTORY_ROOT && uv run --project fused-memory python scripts/clear_leaked_claimants.py --project-root <root> --json
```

Without `--apply` it is a read-only dry run that never dials the MCP server. `--apply` writes
claimants. That is a disposition the skill decides, never part of this read.

## Evidence for a chain finding

Name the chain key (`tag`/`key`) and the latest link (`link_id`, `link_status`, `at`), and read
the link with `get_task(link_id)`.

- **Overdue:** the link was not dispatched within its grace window after `at` (grace:
  `dashboard/src/dashboard/data/recurrence_gauge.py::overdue_grace`). An unreadable `at` also
  reads overdue (recurring PRD, Open question 1).
- **Broken:** the current link is `done` with no successor. The re-mint sweep repairs that at
  fused-memory startup and hourly
  (`fused-memory/src/fused_memory/middleware/recurrence_mint.py::run_remint_sweep_loop`), so a
  chain still broken means a mint attempt failed. Look for the attempt in the fused-memory
  journal:

```bash
journalctl --user -u fused-memory --since -65min --no-pager -o cat \
  --grep 'predecessor=<link_id> key=|recurrence_remint_sweep_failed'
```

Give it a Bash `timeout` of at least 180000. It measured 45s and 65s on 2026-10-12/13, because
the unit logs about 2k lines a minute, and a window of hours outran a 120s timeout. The 65-minute
window always spans at least one attempt: the done-write mint or a sweep. Exit 1 means no matching
line, which is an answer, not a failure. These lines carry no journald priority, so `-p err` finds
nothing.

- A `recurrence_mint_failed: predecessor=<link_id> ... reason=` line names the reason.
- No line at all is itself evidence that no attempt ran. Was fused-memory restarted, or is the
  sweep dead?

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

# project: the record's `project`; for `absent`, which has no record, os.path.basename(root)
escalation_id = f"invariants-{kind}-{project}"
suggestion_hash = hashlib.sha256(f"{root} {kind} {identity}".encode()).hexdigest()[:16]
```

`kind` is the word after `FINDING` in the digest line, or `chain-paused` for an unowned hold:

| `kind` | `identity` |
|---|---|
| `blind` | the record's `error` string |
| `terminal` | the space-joined, sorted `tag#task_id` list the filing covers |
| `chain-overdue`, `chain-broken`, `chain-paused` | `tag`, `key` and `link_id`, space-joined |
| `absent` | the normalised root the digest printed |
