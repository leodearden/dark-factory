# hotspot-survey overlay — dark-factory

- **Memory identity**: `project_id="dark_factory"`, `agent_id="claude-interactive"` (write-tagging convention in CLAUDE.md).
- **Task tracker**: SQLite at `.taskmaster/tasks/tasks.db` (~4,700 tasks). Mine it directly with
  python3/sqlite3 — not via MCP round-trips. Open read-only so you never contend with the live
  orchestrator: `sqlite3.connect('file:.taskmaster/tasks/tasks.db?mode=ro', uri=True)`.
  `tasks` columns: `tag, id, title, description, details, test_strategy, status, priority,
  metadata, updated_at, claimant_run_id, heartbeat_at, candidate_key` (`metadata` is a JSON
  string). There is **no `dependencies` column** — dependencies are their own table keyed
  `(tag, task_id, depends_on)`; dependents of N = `select task_id from dependencies where
  depends_on=N`. ⚠️ `.taskmaster/tasks.db` at the top level is a 0-byte decoy — use the
  `tasks/` one. The former `tasks/tasks.json` was retired at the SQLite cutover (2026-05-06),
  sat 4 months stale (frozen at max id 1152 while the db kept growing), and was DELETED
  2026-08-25 — if one reappears, do not mine it.
- **Output**: `plans/bug-hotspot-survey-<date>.md` + `-full-findings.json`, committed. Use `git commit --only <paths>` — direct-to-main commits race the live merge queue (ref lock → re-add + retry); pre-commit (3× pyright) can exceed the Bash 2-min ceiling → `setsid git commit ... &` + poll.
- **Fix-commit vocabulary**: generic set plus `--grep='amend:'` (post-merge patch-ups) and `--grep='red-main'` (broke main) — these mark the weakest code.
- **History window**: `--since=2026-01-01` (the autonomous-factory era; most commits are agent-authored TDD).
- **Subsystem vocabulary seed** (from the 2026-07-06 run; re-derive sizes/churn fresh in Phase 0):
  merge-queue (merge_queue.py + merge_* satellites), workflow (workflow.py, workflow_types.py), harness (harness.py, invoke/steward), git-worktrees (git_ops.py, warm_lane_pool.py, worktree_identity.py, offline_lane.py, cargo_scope.py), scheduler, verify, fm-task-layer (fused-memory task backend + curator), fm-recon (reconciliation), fm-memory (graphiti/mem0 clients), shared-infra (shared/ incl. usage_gate.py), escalation, dashboard.
- **Doc corpora**: `plans/*.md` (PRDs/postmortems), `CHANGELOG.md`, `DESIGN.md`, `docs/`, `fused-memory/docs/`.
- **Deterministic audit fold-in**: none (no /audit CLI in this repo).
- **Known-context sources**: fused-memory `search` + the auto-memory index (`~/.claude/projects/-home-leo-src-dark-factory/memory/MEMORY.md`) — incident/fix-batch entries seed cluster `context` paragraphs.
- **Hand-off**: PRDs land in `plans/`; program doc `plans/bug-hotspot-remediation-program-<date>.md`. Release gates for deferred batches = deterministic pure-gate tasks (`task_kind='deterministic'`, `always_escalates=true` — see CLAUDE.md), NOT bare no-op tasks. Filing: `planning_mode=True` + bulk `commit_planning`; verify batches with `get_task`, never search/grep.
- **Prior run (exemplars)**: `plans/bug-hotspot-survey-2026-07-06.md`, `-full-findings.json`, `plans/bug-hotspot-remediation-program-2026-07-06.md`; workflow script mirrored at `skills/hotspot-survey/references/exemplar-run-df-2026-07-06.js`. A re-survey after the W1–W11 waves merge should compare against this baseline to measure whether the hotspots cooled.
