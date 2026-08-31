# Capability manifest — recon-codebase-verifier-fix-prd

Binds each task's signal capabilities to evidence (G3+G6 mechanized).
Evidence verified against main `ea876cb624` (2026-08-24); branch evidence
against `task/3241` tip `0c8459f273`. Cite-by-symbol elsewhere; the grep
patterns below were each executed at authoring time and confirmed
**non-vacuous** (every `expect: absent` pattern matches today and every
`expect: present` pattern is absent today, so no gate is born unable to
fire — the esc-4545-2 hazard).

Machine-readable twin: `recon-codebase-verifier-fix-prd.capability-manifest.yaml`.

## α — thread per-task codebase root into verify + AgentLoop cwd

- `caller-scope-available` → **PASS** (wired on main):
  `TargetedReconciliation.reconcile_task` builds
  `ProjectScope(ProjectId(project_id), ProjectRoot(project_root))` after
  `require_project_root`; every `_on_task_done` call site holds `scope`.
  Pre-existing substrate — evidence binding only, no delivered_check (a
  regression gate on unowned substrate is the false-FAIL shape the
  merge-worktree batch deliberately avoided).
- `refusal-token-contract` → **PASS** (producer: α, upstream of δ/ε).
  Contract-fixed literal `codebase_root_unresolved` (PRD §Contract) —
  delivered_check `expect: present`.
- `verifier-drops-global-root` → **PASS** (producer: α).
  `CodebaseVerifier` stops reading `explore_codebase_root` (PRD D3) —
  delivered_check `expect: absent` scoped to `verify.py` (the key remains
  legitimate in `agent_loop.py`'s fallback and the task-1989 sites).
- `audited-agent-failed-path` → **PASS** (wired on main): task 4343's
  `verify|codebase` outcome-row contract in `_on_task_done` writes
  `failure_token` into the audit detail; the refusal rides it unchanged.
  Evidence binding only.

## β — verdict-specific templates + prompt hardening + contradicted → L1

- `old-shared-template-retired` → **PASS** (producer: β). The
  `Completed task '{title}'` f-string template is replaced by
  verdict-specific wording (PRD D7.1) — delivered_check `expect: absent`.
- `contradicted-files-l1-escalation` → **PASS** (producer: β). Existing
  category literal `risk_identified` (deliberately not a new category —
  the Escalation model's comment makes the next addition an
  enum-promotion) — delivered_check `expect: present`.
- `escalation-substrate` → **PASS** (wired on main):
  `TargetedReconciliation._sweep_escalate_l1` files L1 `Escalation` into
  `EscalationQueue(Path(project_root) / 'data/escalations')` behind the
  `_HAS_ESCALATION` guard — the production precedent β mirrors. Evidence
  binding only.
- `no-status-mutation-on-verdict` → **PASS** by construction (G6 branch 4
  inverse): β adds no `set_task_status`/reopen call; the escalation is an
  alert (INV-3, esc-3105-3 pin). `kind: manual` — judged by β's tests.

## γ — correct the `schema_salvaged` docstring in cli_invoke.py

- `false-claim-retired` → **PASS** (producer: γ). The single-line fragment
  ``commonly ``error_max_turns`` paired`` matches today (`AgentResult`
  docstring) and must not survive γ — delivered_check `expect: absent`.
  (Pattern anchored to one physical line; the sentence wraps, so the full
  phrase would be a vacuous never-matching check.)

## δ — task 3241 (adopted): land branch task/3241

- `per-invocation-cap-raised` → **PASS** (producer: δ; branch evidence).
  `_AGENT_CLI_MAX_TURNS = 10` defined and passed as
  `max_turns=_AGENT_CLI_MAX_TURNS` (branch `agent_loop.py`, symbols
  `_AGENT_CLI_MAX_TURNS` / `_call_claude_cli`) — delivered_check
  `expect: present`.
- `unit-cap-call-site-gone` → **PASS** (producer: δ). Call-site form
  `max_turns=1,` (trailing comma — the branch legitimately keeps
  `max_turns=1 ->` in its measurement-table comment, so the bare string
  would false-FAIL) — delivered_check `expect: absent`.
- `regression-test-pins-floor` → **PASS** (branch evidence):
  `test_agent_loop.py` asserts `call_args.kwargs['max_turns'] >= 3`
  (invariant, not `== 10`); reverting the constant to 1 was verified to
  fail it. `kind: manual` — test property.
- `branch-merges-clean` → **PASS** (measured 2026-08-24):
  `git merge-tree --write-tree main task/3241` rc=0; merge-result suite
  461 passed. Re-check at landing (main moves). `kind: manual`.

## ε — soak gate: census shows the verifier live and correctly scoped

- `census-instrument-exists` → **PASS** (wired on main): task 4343's
  one-outcome-row-per-invocation contract makes the census exact by
  construction. Evidence binding only.
- `census-moved` → **PASS** (premise achievable: probe `mt=10 → 3/3` on
  CLI 2.1.241 plus 8 pre-regression production verdict memories prove the
  end-to-end chain). Operational ruling against production DB —
  `kind: manual` (quoted query in the task; ≥1 non-`error` outcome row
  AND ≥1 non-dark_factory invocation correctly scoped or refused; no
  success-*rate* assertion — the failure is stochastic and task 4344's
  unreproduced residual stays out of the predicate).
