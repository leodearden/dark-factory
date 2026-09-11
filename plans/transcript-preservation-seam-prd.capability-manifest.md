# Capability manifest — transcript-preservation-seam-prd

Binds each leaf's user-observable signal to substrate evidence, mechanizing
G3 + G6 once here instead of once per task at dispatch.
PRD: `plans/transcript-preservation-seam-prd.md` (committed `d015581f99`).
Built 2026-08-04 during `/prd decompose`. Machine-readable twin:
`plans/transcript-preservation-seam-prd.capability-manifest.yaml`.

Greek labels: α = leaf 1, β = leaf 2, γ = leaf 3, δ = leaf 4.

---

## Gate findings resolved during this decompose

Three bindings did **not** clear on the PRD's own text and were resolved by
widening / narrowing leaf scope. All three are carried into the filed task
descriptions, not just recorded here.

**F1 — α: the `.gz` consumer set in PRD §4 is INCOMPLETE (`producer-extent-short`).**
§4 enumerates four production consumers. A grep over all non-worktree `*.py`
finds a fifth production reader the PRD never names —
`fused-memory/scripts/memory_eval_transcript_corpus.py` (`:280-281` strips
`.jsonl.gz`; `:697` globs `*/**/*.jsonl.gz`) — plus eight test modules that
build or assert `.gz` fixtures (`shared/tests/test_transcript_archive.py`,
`scripts/tests/test_gc_agent_transcripts.py`,
`scripts/tests/test_legibility_{digest,inventory,nightly}.py`,
`fused-memory/tests/test_memory_eval_transcript_corpus.py`,
`orchestrator/tests/test_transcript_archive_{producer_hook,backstop}.py`) and
a stale docstring in `orchestrator/src/orchestrator/git_ops.py:11775`.
**Resolution:** α's declared scope widened to the measured set. Leaving the
fifth reader behind would have shipped a silently-empty memory-eval corpus.

**F2 — α: "delete the `inventory.py:101-160` error-normalization block" is
WRONG as written (`producer-extent-short`, inverted).**
`as_unreadable_file_error` normalizes four shapes, and the module's own
docstring (`inventory.py:136-139`) states that the `UnicodeDecodeError` shape
"is the only shape reachable on a PLAIN `.jsonl` path as well as a `.gz` one
(both are opened under strict `encoding='utf-8'`)". Deleting the block wholesale
removes a degrade path that is still load-bearing post-gzip, and one undecodable
byte would then abort a whole-archive walk. **Resolution:** α deletes only the
gzip-reachable shapes (`EOFError`, `zlib.error`, and the `gzip.BadGzipFile`
prose) and **retains** the `UnicodeDecodeError` normalization plus
`digest.load_transcript`'s shared use of it.

**F3 — β: the startup sweeper lands in `harness.py`, which collides with live
task 3256 (`DAG-direction` hazard).**
PRD §4 states leaves 1 and 2 do not touch 3256's files. Verified otherwise: the
boot-time surviving-worktree scan the sweeper must ride is
`Harness._recover_crashed_tasks` (`harness.py:3112`), immediately adjacent to
`_adopt_recovered_session` (`:2980-3057`); the alternative registration point
(`background_service.LifecycleService`) is also registered from `harness.run()`
(`harness.py:2048`). 3256 is `in-progress` with a live claimant
(`run-3248b9e51fe1/3256-04d3f475`, heartbeat 2026-08-04T12:32Z) editing
`harness.py`. **Resolution:** β additionally `depends_on` 3256. Cost is ~zero —
β is already gated behind α — and it frees the implementer to site the sweeper
where it belongs instead of contorting around a file lock.

---

## α — leaf 1: drop gzip; one plain, greppable corpus

*Signal:* `find data/orchestrator/agent-transcripts -name '*.gz' | wc -l` → **0**;
`rg <string> data/orchestrator/agent-transcripts` matches transcript content
directly with no `zcat`; neither `inventory.py` nor `digest.py` imports `gzip`.

| Capability | Evidence | Verdict |
|---|---|---|
| Single writer owns the archive format | `grep:shared/src/shared/transcript_archive.py:37` `import gzip`, `:101` `.gz` suffix — one writer, wired from `workflow.py:11824` and `git_ops.py:11788` | PASS |
| Readers already accept plain `.jsonl` | `grep:scripts/legibility/inventory.py:409-421` — one `rglob('*.jsonl*')` filtered by suffix, a strict superset; `grep:scripts/legibility/digest.py:86` branches on `.gz` | PASS |
| Complete `.gz` consumer set | Re-enumerated 2026-08-04 by grep over all non-worktree `*.py`. PRD §4's four + `fused-memory/scripts/memory_eval_transcript_corpus.py` + 8 test modules + a `git_ops.py:11775` docstring. See **F1** | PASS *(after widening)* |
| `UnicodeDecodeError` normalization must survive | `grep:scripts/legibility/inventory.py:136-139` — reachable on plain `.jsonl`. See **F2** | PASS *(after narrowing)* |
| Migration corroborates before deleting source (INV-3) | Producer: α. Per-file gunzip → verify readable → unlink; abort loud on any failure | PASS |
| Migration is idempotent + re-runnable, and ordered before the reader-branch deletion (INV-7) | Producer: α. A half-completed migration leaves a mixed corpus; deleting reader branches first would make the residue silently unreadable | PASS |
| Disk headroom (numeric floor) | MEASURED 2026-08-04: archive 462 MB / 4,369 `.gz` / 600 task dirs; free 2.1 TB of 6.8 TB. One-off delta +1.3 GB = **0.06 %** of free. `floor: 1.3 GB ≪ 2.1 TB` | PASS |

## β — leaf 2: transcript preservation as a precondition of config-dir deletion

*Signal:* after a SIGTERM restart with sessions in flight, every session preserved
**by that restart** has its transcript at
`data/orchestrator/agent-transcripts/<task_id>/<enc>/<sid>.jsonl`; the §2 coverage
query over the post-fix cohort reads 100 % where today's cohort reads 12.5 %.

| Capability | Evidence | Verdict |
|---|---|---|
| Same-filesystem `os.rename` archive-ward | `stat -c '%d'` → `.worktrees` **66312**, `data/orchestrator` **66312**, one `/dev/nvme2n1p5`. VERIFIED 2026-08-04 | PASS |
| Rename is synchronous ⇒ uncancellable at SIGTERM | The cancellable construct is `await asyncio.to_thread(archive_task_transcripts, …)` (`workflow.py:11823`, `git_ops.py:11787`), whose `except asyncio.CancelledError: raise` (`workflow.py:11830-11841`) is exactly what drops the transcript. Removing compression (α) removes the reason to offload | PASS *(producer: α, upstream)* |
| Deletion sites enumerated | `_cleanup_config_dir` (`workflow.py:7723-7733`, reached from the `cleanup_config_dir` teardown step at `:3130`) and the `cleanup_worktree` backstop (`git_ops.py:11777-11812`). A **third** site — `dry_run_unblock.py:450` — is owned by **task 3271** and is deliberately out of β's scope; 3271 is named as a downstream consumer of β's shared helper (INV-5) | PASS *(seam declared)* |
| Archival-failure escalation exists to extend (INV-4) | `grep:shared/src/shared/transcript_archive.py:44` module-level per-file failure counter already present | PASS |
| Startup sweeper hook site | `Harness._recover_crashed_tasks` (`harness.py:3112`) walks surviving worktrees at boot. Collides with 3256 — see **F3** | PASS *(dep on 3256 wired)* |
| Credential container still dies fast (D1) | Producer: β. A permanently-failing archive must **not** convert into an unbounded hold on a `.credentials.json`-bearing dir (INV-7). Required shape: delete every non-transcript member unconditionally; the hold is scoped to the un-archivable `.jsonl` alone | PASS *(constraint carried into the task)* |
| Coverage claim is achievable (numeric) | 100 % is asserted over the **post-fix preserved cohort only** — the historical 12.5 % cohort is not retroactively recoverable. Stated explicitly so the RED signal is turnable-green | PASS *(scoped)* |
| `EXDEV` fallback | PRD §9 Q4. Not reachable today (one device), but `os.rename` must fall back to copy+unlink rather than assume | PASS |

## γ — leaf 3: deterministic config-dir resolution + ambiguity signal

*Signal:* a recovered worktree whose `.task/` holds no `claude-config-<task_id>`
emits a structured `session_config_dir_ambiguous` event carrying
`{expected, found[], session_id, task_id}` and stashes **nothing**, where today
it silently stashes a non-matching dir and yields a guaranteed `no_transcript`.

| Capability | Evidence | Verdict |
|---|---|---|
| `CONFIG_DIR_PREFIX` importable, sole name source (INV-5) | `grep:shared/src/shared/config_dir.py:34` constant; `:232` `base / f'{CONFIG_DIR_PREFIX}{task_id}'` is the only mkdir site in the tree | PASS |
| Config-dir name derives from `task_id`, not branch | `harness.py:3024-3026`'s "the dir name embeds the branch" comment is **factually wrong** — the full branch is `task/<id>`, not one path component. Every production creator passes a `task_id` stem | PASS *(prose defect is γ's target)* |
| New `EventType` member addable | `grep:orchestrator/src/orchestrator/event_store.py:229-237` — `session_resume` / `_fallback` / `_capped` family already present | PASS |
| Separately-deduped L1 precedent | `has_open_l1(<sentinel>)` at `harness.py:6076`; the `capped` carve-out at `harness.py:7653-7661` + `config.py:869` ("does NOT feed the fallback-storm streak") is the exact shape to mirror (D7) | PASS |
| Storm-streak exclusion is reachable | `_session_resume_fallback_streak += 1` sits in the `else` branch at `harness.py:7668-7670`; a new branch above it excludes cleanly | PASS |
| Ambiguity L1 must not collide on the cockpit decision id | Distinct sentinel required; a shared sentinel lets a reap close the wrong queue's gate (task 3528) | PASS *(constraint carried)* |
| 3464 premise (rejection-shape) | MEASURED on disk 2026-08-04: `.worktrees/3464/.task/` holds **exactly one** `claude-config-*` — `claude-config-3464-unblock` — so `len(config_dirs) == 1` and the `>1` warning at `harness.py:3032` never fires. `.worktrees/2971/.task/claude-config-df_task_13` is a real foreign-task sibling | PASS |
| Signal survives the live worktree disappearing | `.worktrees/3464` belongs to a `blocked` task and may be reaped before γ dispatches. The signal must therefore also be reproducible from a fixture mirroring that shape, with the live dir as corroboration while it exists | PASS *(scoped)* |
| Fixtures encode the wrong contract and will break | `grep:orchestrator/tests/test_crash_recovery.py:144` and `orchestrator/tests/test_session_resume_integration_gate.py:166` both build `base / f'claude-config-{session_id}'`. VERIFIED — that breakage is the signal, fix the fixtures not the resolver | PASS |

## δ — leaf 4: retention sizing so age is the only binding policy

*Signal:* the GC sweep prunes on **age**, never on dir count, at the measured
peak rate; when the count cap *does* bind it says so loudly instead of silently
truncating the 90-day window.

| Capability | Evidence | Verdict |
|---|---|---|
| `max_task_dirs` is a live, hot-reloadable config leaf | `grep:orchestrator/src/orchestrator/config.py:1025-1031` default `5000`; `:4920-4923` — `transcript_archive.retention` reloads as one atomic submodel leaf | PASS |
| Three lock-step sites must move together (INV-5) | `config.py:1026` default, `scripts/gc_agent_transcripts.py:107` `DEFAULT_MAX_TASK_DIRS`, and the pinning assert at `scripts/tests/test_gc_agent_transcripts.py:410` (`== RetentionConfig().max_task_dirs == 5000`). Already test-pinned — the house pattern; δ updates all three coherently | PASS |
| Count-pruning is oldest-first and silent today | `grep:scripts/gc_agent_transcripts.py:183` `is_count = max_task_dirs > 0 and rank >= max_task_dirs` — no distinct signal when the count cap rather than age is what pruned | PASS |
| Derived-bound test is **non-vacuous** (INV-1) | A test asserting `cap ≥ 90 × rate × safety` with a **hardcoded** rate is arithmetic on constants and can never fail — the `integration_skew` I7-gate shape. Required: derive the rate from real archive dir mtimes (skip when the archive is absent), and make a binding count-cap loud at runtime | PASS *(constraint carried)* |
| Peak-rate premise (numeric) | MEASURED 2026-08-04: **600** task dirs over 17 days = mean 35/day → 3,150 @ 90d; recent 7-day 46.6/day → 4,194 (84 % of the 5,000 cap); peak 71/day → 6,390, binding at ~day 70. Live count re-verified today: `ls data/orchestrator/agent-transcripts | wc -l` = 600 | PASS |
| 50,000 is defensible, not a guess | 50,000 / (90 × 71) ≈ **7.8×** headroom over the measured peak; ~15× the recent rate. Ruled by Leo 2026-08-04 | PASS |

---

## Amendment (not a leaf) — task 3578

Not a manifest entry: 3578 is an existing `pending` task, amended in place per
D9. Its archive-coverage assumption is bound by β's coverage capability above
(12.5 % measured for its own target population on 2026-08-04), and its cwd HARD
GATE is vacuous for dark-factory — no `warm_lane_pool` key in
`dark-factory-orchestrator.yaml` (VERIFIED 2026-08-04), worktrees are
task-id-named and retained, so the encoded-cwd component is identical across
restarts (D10).
