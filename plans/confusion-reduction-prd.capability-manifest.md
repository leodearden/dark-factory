# Capability manifest — plans/confusion-reduction-prd.md

Per-leaf capability→evidence bindings (G3+G6 mechanized). Evidence verified against main 2026-07-13 (PRD commit 3001e94d7a). Task ids stamped at commit_planning.

## Shared substrate (asserted by every leaf)

| Capability | Evidence | Verdict |
|---|---|---|
| Codebook v1 exists on main | `docs/legibility/confusion-codebook.yaml`, commit 0691d13263 | PASS |
| Transcript-dir encoding known + wired | `grep:orchestrator/src/orchestrator/session_registry.py:451` (`transcript_path_for_cwd`, `/` and `.` → `-`), used at :1398 (production enrichment path) | PASS (wired) |
| Encodings enumerable by prefix | `ls ~/.claude/projects \| grep -c '^-home-leo-src-dark-factory'` → 57; reify/warm-lane → 275 | PASS |
| Headless LLM calls with model override | `claude` CLI 2.1.207 at `/home/leo/.local/bin/claude`; `-p --model` documented modes | PASS |
| pytest convention for scripts | `scripts/tests/` (8 existing test modules + conftest) | PASS |

## ε — nightly trickle end-to-end (integration-gate leaf)

| Capability | Evidence | Verdict |
|---|---|---|
| Digest extraction | `producer:task-α` (upstream dep) | PASS |
| Inventory/scoring/sampling + project config | `producer:task-β` (upstream dep) | PASS |
| Codebook v2 schema + sole-writer merger | `producer:task-γ` (upstream dep) | PASS |
| Haiku coding → coding records | `producer:task-δ` (upstream dep) | PASS |
| Census-trigger evaluation | `producer:task-ζ` (upstream dep) | PASS |
| Docs-only commit to machine-operated main | `git commit --only` precedent (standing feedback; exercised this session, commit 3001e94d7a) | PASS |
| systemd user service+timer precedent | `scripts/orchestrator-watchdog.timer` + `.service`; per-project `orchestrator-*.service` templates | PASS |
| Loud escalation channel | `escalation/src/escalation/server.py:382-458` (`escalate_blocker`/`escalate_info`) per project port | PASS |

## ε′ — deploy dark_factory timer (deterministic)

| Capability | Evidence | Verdict |
|---|---|---|
| `before_done.script` exists+executable at filing | `scripts/legibility/install-trickle-timer.sh` mode 100755, commit 3001e94d7a (fail-loud stub; real impl = `producer:task-ε`, upstream — stub cannot run early because deps gate dispatch) | PASS |
| DeterministicRunner deploy path (blocking + verify, cross-unit) | CLAUDE.md "Deterministic task kind" (production; precedent task 2456) | PASS |
| `systemctl --user` on host | `orchestrator-watchdog.timer` runs under user systemd today | PASS |

## η — census runner (leaf)

| Capability | Evidence | Verdict |
|---|---|---|
| Digests / sampling / merger / trigger state | `producer:task-α,β,γ,ζ` (all upstream) | PASS |
| Coding-against-codebook module (reused at Sonnet tier) | `producer:task-δ` (upstream) | PASS |
| Curator-path task filing | `mcp__fused-memory__submit_task` (production; survey addendum filed 17 tasks through it 2026-07-13) | PASS |
| Done-count delta for trigger corroboration | `mcp__fused-memory__get_statuses` (production tool) | PASS |
| Saturation premise (≥90% dup rate reachable) | Survey appendix: 12/16 clusters matched existing taxonomy; head saturating (336 incidents, 2026-07-13) — G6 basis | PASS |
| Fable synthesis tier available | this host runs Fable sessions (current session); `--model` override | PASS |
| Headroom preflight | designed as probe-call (no usage-API substrate assumed — PRD decision 5) | PASS (no assumption) |

## θ — reify enablement (reify-project leaf)

| Capability | Evidence | Verdict |
|---|---|---|
| Reify repo + registration | `scripts/orchestrator-reify.service` ExecStart references `/home/leo/src/reify/orchestrator.yaml` (live unit) | PASS |
| Config/codebook schema to instantiate | `producer:dark_factory:ε` external dep (transitively γ) — upstream | PASS |
| Reify transcript encodings exist | 275 dirs matching reify/warm-lane prefixes in `~/.claude/projects` | PASS |

## θ′ — deploy reify timer (deterministic leaf)

| Capability | Evidence | Verdict |
|---|---|---|
| Install script (generic, project arg) | stub committed (3001e94d7a); real impl `producer:task-ε` via local dep ε′→ε — upstream | PASS |
| Reify config present at run time | external dep `reify:θ` — upstream (DAG-direction correct) | PASS |

## κ — trickle liveness milestone (deterministic predicate leaf)

| Capability | Evidence | Verdict |
|---|---|---|
| Delayed-milestone predicate machinery | CLAUDE.md "Milestone tasks" + `shared/src/shared/task_metadata.py` (`Milestone` model, production; exemplar = `scripts/check_merge_flakiness.sh` pattern) | PASS |
| Predicate script exists+executable at filing | `scripts/legibility/check_trickle_liveness.sh` mode 100755, commit 3001e94d7a (fail-loud stub; real impl `producer:task-ε` via ε′→ε — upstream) | PASS |
| Timer-ran-recently observable without commits | `systemctl --user show <unit> -p ExecMainExitTimestamp` etc. (PRD decision 7 — probes unit state, not git, so quiet nights don't false-fail) | PASS |
| Rejection mechanism (escalates on dead timer) | DeterministicRunner predicate exit≠0 → born-at-L2 `milestone_check_failed` (production contract, CLAUDE.md) — rejection-mechanism-backed | PASS |

## Intermediates (substrate-bearing bindings only)

- **α**: transcript JSONL format — real session files under `~/.claude/projects/-home-leo-src-dark-factory/` (57 dirs); scorer signal classes proven by the 2026-07-13 survey (584 sessions scored). PASS.
- **β**: fused-memory reachability irrelevant (zero-LLM, filesystem-only); config file is its own deliverable. PASS.
- **γ**: codebook v1 YAML on main (0691d13263); migration in place, never-delete invariant enforced by the merger it ships. PASS.
- **δ**: claude CLI headless (above); strict-JSON contract is its own deliverable; fail-loud path to escalation server :382-458. PASS.
- **ζ**: `get_statuses` (production); `census-state.json` is its own deliverable. PASS.

**No FAIL bindings. Batch clear to queue.**
