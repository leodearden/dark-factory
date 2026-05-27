# Task 1503 — L1 = human audit findings

Origin: see [plans/escalation-l2-tiering.md](escalation-l2-tiering.md) (E1 ladder).

Under the new escalation ladder (E1, merged):
- **L0** agent → steward
- **L1** steward/workflow → escalation-watcher-auto (NOT a human)
- **L2** auto-watcher → human (direct; bypasses auto-watcher)

Authoritative contract: `escalation/src/escalation/models.py:3-9` and `:45`.

---

## Files updated in task 1503

(doc/comment-only changes — no behavior change)

| File | Lines | Change |
|------|-------|--------|
| `orchestrator/src/orchestrator/agents/roles.py` | L215, L783 | Replace "escalate to a human curator" and "(steward→human)" with accurate L1-consumer language |
| `orchestrator/src/orchestrator/agents/briefing.py` | L322 | Replace "will then escalate to a human" |
| `orchestrator/src/orchestrator/artifacts.py` | L314, L345 | Replace "only a human" / "only a human/curator" docstrings |
| `orchestrator/src/orchestrator/workflow.py` | L72, L4122–4125, L4843, L5494 | Replace stale "(human intervention)", "level-1 (human)", "(steward→human)", "(human-only)" |
| `orchestrator/src/orchestrator/harness.py` | L2896–2898 | Replace "(steward→human) escalations are intentionally preserved" |

---

## Flagged behavioral L1-assumptions

(NOT changed in task 1503 — flagged for the L2-handler package (E2/O1/S1))

These call sites contain **logic** that assumes a human directly reads L1.
Under the new ladder the auto-watcher reads L1 first and is expected to
promote-to-L2 (born-at-L2 by severity; or by `suggested_action='manual_intervention'`).

### 1. `workflow.py:_handle_wip_conflict`
- **What it does:** Files `level=1`, `category=wip_conflict`,
  `suggested_action=manual_intervention`.
- **Why it's an L1=human assumption:** `manual_intervention` implies a human
  will act directly. Auto-watcher should detect `wip_conflict` (or
  `manual_intervention`) and promote to L2.
- **Suggested L2-handler hook:** auto-watcher promotes on
  `category='wip_conflict'` or `suggested_action='manual_intervention'`.

### 2. `workflow.py:_handle_wip_recovery`
- **What it does:** Files `level=1`, `category=wip_conflict`.
- **Why it's an L1=human assumption:** Same pattern as `_handle_wip_conflict`.
- **Suggested L2-handler hook:** Same as above.

### 3. `workflow.py:_handle_wip_recovery_no_advance`
- **What it does:** Files `level=1`, `category=wip_conflict`.
- **Why it's an L1=human assumption:** Same pattern.
- **Suggested L2-handler hook:** Same as above.

### 4. `workflow.py:_handle_unmerged_state`
- **What it does:** Files `level=1`, `category=unmerged_state`,
  `suggested_action=manual_intervention` (only a human can safely resolve
  unmerged git state).
- **Why it's an L1=human assumption:** `unmerged_state` requires human git
  intervention; auto-watcher cannot act on it.
- **Suggested L2-handler hook:** auto-watcher promotes on
  `category='unmerged_state'`.

### 5. `workflow.py:_handle_terminal_exit_on_block`
- **What it does:** Files `level=1`, `category=bypass_done`; reopens the task
  row and signals a forensics-class bypass.
- **Why it's an L1=human assumption:** Bypass forensics are inherently a
  human-review concern.
- **Suggested L2-handler hook:** auto-watcher promotes on
  `category='bypass_done'`.

### 6. `workflow.py:_ensure_l1_escalation_for_blocked`
- **What it does:** Fallback "someone should know" L1 for any unresolved
  BLOCKED task (`category=task_failure`).
- **Why it's an L1=human assumption:** Original intent was "human should
  investigate". Auto-watcher is now the immediate consumer; promotion
  criterion TBD by S1 (e.g. promote after N hours without resolution).

### 7. `scheduler.py:trigger_retry_cap_exhausted`
- **What it does:** Files `level=1`, `category=retry_cap_exhausted`.
- **Why it's an L1=human assumption:** Retry exhaustion is a signal that
  automated strategies are depleted; original intent was human triage.
  Comment at `scheduler.py:2753` ("human-resolution starts from zero") is
  stale-but-incidental; flagged for the same package.
- **Suggested L2-handler hook:** auto-watcher promotes on
  `category='retry_cap_exhausted'`.

### 8. `workflow.py:_mark_blocked(escalate_to_human=...)`
- **What it does:** The parameter name `escalate_to_human=True` means "skip
  steward, file L1 directly". Under the new ladder L1 is consumed by the
  auto-watcher (not a human directly).
- **Why it's an L1=human assumption:** The parameter name is now misleading.
- **Deferred action:** Rename to `escalate_past_steward` or `skip_steward`.
  This is a multi-callsite refactor flagged for a follow-up ticket — see
  "Deferred / out-of-scope" below.

---

## Deferred / out-of-scope

Items acknowledged but not changed in task 1503:

- **`workflow.py` log strings `'escalating to human'`** (lines 2136, 2246, 2316):
  Operator-facing telemetry; "human" is colloquially still accurate (auto-watcher
  promotes to a human on these paths). Not a contract violation; can be cleaned
  up in a separate pass.
- **`escalation/SKILL.md` edits:** Owned by S1/S2 tasks in the L2-tiering batch.
- **Test-file docstrings** (e.g. `test_workflow_e2e.py:4546`): They describe
  the workflow under test; updating in lock-step with workflow.py is fine but
  not required for acceptance.
- **Renaming `_mark_blocked(escalate_to_human=...)`:** Multi-callsite refactor.
  Risk of merge conflicts with concurrent L2-tiering tasks. Flagged as item 8
  above; should be a dedicated ticket.

---

## Cross-references

- `escalation/src/escalation/models.py:3-9` — authoritative consumer-per-level
  contract (single source of truth for what L0/L1/L2 mean).
- `escalation/src/escalation/models.py:45` — `Escalation.level` inline comment.
- [plans/escalation-l2-tiering.md](escalation-l2-tiering.md) — design doc + task
  batch (O1 supervisor failsafe; S1 auto-watcher RCA + promote_to_l2). The
  flagged items in section "Flagged behavioral L1-assumptions" above are the
  priority inputs for E2/O1/S1.
