# Capability manifest — agent-transcript-archival-prd

Per-leaf capability→evidence bindings (G3+G6 mechanized). Substrate verified on `main`
`d19b3645df`, 2026-07-17. Line refs drift; symbols are canonical. No numeric-accuracy floor is
asserted by any leaf (the ~2 MB gz/task sizing is a disk-headroom fact, not a correctness bound).
`delivered_check`s are informational for this batch — the `commit_planning` stamper (its own
producer batch) is not yet live on `main`, so no task here carries `metadata.delivered_checks`;
the sidecar is hand-stamped at decompose.

## α — Archiver primitive + producer hook (must-have core)

| Capability asserted by the signal | Evidence | Verdict |
|---|---|---|
| Orchestrator knows the just-used session id at the `_invoke` finally | grep:`workflow.py:8348-8352` `session_id_val = str(uuid.uuid4())` → `self._last_invoke_session_id` | PASS wired |
| Orchestrator owns the config dir (transcript root) with a `.path` | grep:`config_dir.py:34-37` `claude-config-{task_id}`; `.path` property `config_dir.py:70-73`; `projects/` kept per-task `config_dir.py:23` | PASS wired |
| The `_invoke` `finally` is the producer hook point (currently `clear_agent_session()`) | grep:`workflow.py:8421-8423` `finally: … clear_agent_session()` | PASS wired |
| A completed role leaves a readable transcript jsonl (+ subagents) under `projects/` | filesystem ground truth: `.worktrees/2359/.task/claude-config-2359/projects/*/<sid>.jsonl` (8 files incl. `subagents/agent-*.jsonl`) | PASS (artifact exists) |
| Durable store `data/orchestrator/` exists to write `agent-transcripts/` under | filesystem: `data/orchestrator/` holds `runs.db`/`scheduler_state.json` (git-ignored data dir) | PASS wired |
| `OrchestratorConfig` is the home for the new green-tier `transcript_archive.*` block | grep:`config.py` `OrchestratorConfig` + existing green-tier leaf tunables (git.offline_lane_*, review.*) | PASS wired |
| Archive is credential-safe — only `projects/**/*.jsonl`, never `.credentials.json` (rejection) | built+bound by α; boundary test **E4** asserts `.credentials.json` absent anywhere under the archive | PASS (bound as E4) |
| Archive failure is soft **and** loud — structured fact + failure counter (INV-2/INV-4) | built by α; boundary test **E7** asserts task completes + a structured archival-failure fact is logged/counted | PASS (bound as E7) |

## β — Teardown backstop at the `cleanup_worktree` chokepoint

| Capability asserted by the signal | Evidence | Verdict |
|---|---|---|
| The single teardown chokepoint exists to hook | grep:`git_ops.py:8738` `async def cleanup_worktree(self, worktree, branch)` (the 12 harness reconcile/crash paths delegate here) | PASS wired |
| The archiver helper to call | producer:task-α (upstream of β) — `archive_task_transcripts` | PASS producer upstream |
| Idempotent — an already-archived transcript is not re-copied (negative assertion) | α's helper skips on current size/mtime; boundary test **E3** authors one archived + one un-archived and asserts the archived one is byte-unchanged | PASS (bound as E3) |
| DAG-direction (anti-inversion) | α upstream of β; no owner depends on its consumer | PASS producer upstream |

## γ — Legibility multi-root gz-aware enumerate + turn the root ON

| Capability asserted by the signal | Evidence | Verdict |
|---|---|---|
| `enumerate_sessions`/`iter_project_dirs` are already root-parameterized to generalize to a list | grep:`inventory.py:43-64` `iter_project_dirs(projects_root, cwd_prefixes)`; `inventory.py:188-238` `enumerate_sessions(projects_root, cwd_prefixes, date)` | PASS wired |
| The cwd filter already admits worktree cwds (so fleet transcripts pass) | grep:`inventory.py:67-77` `is_member` admits `.worktrees`/`.claude-worktrees` descendants of a prefix; `legibility.yaml:13-14` `cwd_prefixes` | PASS wired |
| Config loader reads `legibility.yaml` (home for `agent_transcript_roots`) | grep:`scripts/legibility/config.py` `load_config`; `docs/legibility/legibility.yaml` (existing per-project config) | PASS wired |
| Archived transcripts to enumerate | producer:task-α (upstream of γ) — the archive dir + gz layout | PASS producer upstream |
| The root is shipped **ON** (`agent_transcript_roots` set in `legibility.yaml`, not empty) | built by γ; delivered_check greps the set knob present in `legibility.yaml` | PASS (delivered by γ) |
| DAG-direction | α upstream of γ | PASS producer upstream |

## δ — Retention GC sweep

| Capability asserted by the signal | Evidence | Verdict |
|---|---|---|
| The archive layout to prune | producer:task-α (upstream of δ) — `data/orchestrator/agent-transcripts/<id>/…` | PASS producer upstream |
| Config home for `transcript_archive.retention_*` caps | grep:`config.py` `OrchestratorConfig` green-tier leaf tunables (same home as α's block) | PASS wired |
| GC is loud — logs each dropped dir + a summary count (INV-4) | built by δ; boundary test **E8** asserts over-cap dirs pruned + logged, default caps → no-op | PASS (bound as E8) |
| DAG-direction | α upstream of δ | PASS producer upstream |

## ε — End-to-end boundary gate (B+H integration)

| Capability asserted by the signal | Evidence | Verdict |
|---|---|---|
| All structural capabilities (archiver, backstop, mining root, GC) | producers: α, β, γ, δ — all upstream of ε | PASS producer upstream |
| The boundary matrix (E1–E8) is producible from the integrated path | the task IS the check suite; its signal is the CI-green boundary table (§Appendix B) | PASS (integration gate) |
| No numeric accuracy/throughput floor asserted in any leaf | sizing is disk-headroom, not a bound; GC caps are safety valves, not accuracy floors | PASS (floor branch n/a) |

No FAIL bindings. Batch clear to queue.
