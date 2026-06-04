# PRD-3 cross-PRD seam log (append-only)

Discovered during PRD-3 authoring (escalation re-pend state machine & merge gating).
Protocol per `plans/escalation-flow-2026-06-04-prd-briefs.md`: append-only; siblings glob
`plans/escalation-flow-gaps-prd*.md` before finalizing.

## 2026-06-04 — authoring session

1. **`escalation/src/escalation/models.py` — PRD-3 claims an additive
   `Escalation.resolution_action` field.** models.py is unassigned in the register. The field
   rides through queue.py's full-record JSON read/write untouched (no queue.py edit). PRD-1:
   your queue/sweep/archive work must treat unknown Escalation fields as pass-through (current
   serialization already does).

2. **`server.py` regions beyond the register's assignment.** Register gives PRD-3 only the
   CATEGORIES list + `resolve_issue` docstring. PRD-3 additionally claims, under its "re-pend
   mechanism / categories" mandate: (a) the `resolve_issue` handler body (new `action` enum
   param, `terminate` hard-error), (b) the submit-path chokepoint (~server.py:120-160) for the
   agent-critical severity downgrade. PRD-1 wires `escalation.sweep` into server **startup** —
   different region, prose-coherence only; no collision expected.

3. **`resolve_issue(terminate=...)` → `resolve_issue(action=...)` migration touches snippets
   in sections owned by other PRDs.** The `terminate` param will be REMOVED (hard,
   self-explaining error). Snippets exist in: escalation-watcher-auto Main Loop (PRD-1),
   escalation-watcher B3 subsection + AFK shift 2 (PRD-2), and out-of-register skills
   (`skills/unblock/SKILL.md`, `skills/recon-escalation-watcher/SKILL.md`). Ask: PRD-1/2 use
   `action=` in any NEW snippets you write (semantics: `resume` = answer & continue/re-pend;
   `restart` = kill + pending; `park` = kill + deferred; `abandon` = kill + cancelled;
   `close_only` = record only). PRD-3 ships a companion sweep task migrating remaining legacy
   snippets after PRD-1/2 land; until then the server error message names the new values.

4. **In-process `queue.resolve()` callers (steward.py:717, harness internals) carry no
   action.** PRD-3 defines the legacy mapping for `resolution_action=None` records:
   `dismiss=True → close_only`, `dismiss=False → resume` — never destructive, so steward
   requires no edit. Noted for PRD-1 awareness (steward consumes L0s).

5. **The escalation server package also serves the recon queue (port 8103).** The action enum
   lands there too; recon findings are not orchestrator-task-attached, so only the record
   disposition matters (`resume`/`close_only`), but `skills/recon-escalation-watcher/SKILL.md`
   snippets need the same migration (covered by the companion sweep task in item 3).

6. **Ack of sibling entries (read 2026-06-04, post-authoring glob).**
   - PRD-1 entry 2 (server.py startup-sweep block near `create_server`): consistent with our
     entry 2 — disjoint regions, prose-coherence only. PRD-3's server.py edits stay inside the
     resolve_issue handler + submit chokepoint + CATEGORIES.
   - PRD-1 entry 3 (watcher instant-fire at launch): one-line note folded into PRD-3 task η
     (L2 SKILL.md rewrite).
   - PRD-2 entry 2 (B3 applicability hardcoded in PRD-3-owned `task_failure`/`review_issues`
     handlers, SKILL.md:378/:386): adopted — task η rewrites those handlers to pointer form
     ("if the low-risk auto-unblock gate applies — see 'Low-risk auto-unblock gate (B3)' — try
     it first"), never restating posture rules.
   - PRD-1 entries 1/4, PRD-2 entries 1/3: no PRD-3 surface intersection (we make no agent
     invocations needing env_overrides; we don't touch queue-root file handling).
