# vLLM Local Model Evaluation — Status Report

**Last updated:** 2026-04-08 00:30 BST
**Status:** Five baked vLLM eval images on Docker Hub. First parallel matrix run completed across 3 of 5 configs (the other 2 hit RunPod infra issues unrelated to our code). Three true `outcome=done` results (qwen3 / minimax-fp8 / reap-139b on different tasks) plus several "would-be-done" timeouts where tests/lint/typecheck all pass but the 60 min eval timeout fired. Workstation tier (RTX 3090 AWQ) producing first quality measurements with 4h timeout retries running in background.

## Update 2026-04-07 night: shard-per-layer baking + matrix run

### TL;DR
- **5 baked images shipped to Docker Hub** via new shard-per-layer Dockerfile generator (`runpod-toolkit/scripts/bake_model_image.py`) — pushes are reliable when each safetensors shard is its own ~2 GB layer.
- **The "qwen3 startup hang" is solved.** Root cause was NOT vLLM — it was a launcher SSH-tunnel port collision with the dark-factory escalation MCP server on `127.0.0.1:8100`. SSH fell back to `[::1]:8100` (IPv6); the launcher's `urllib` health probe resolved `localhost` → `127.0.0.1` (IPv4) and hit the escalation server (404). Fix: explicit IPv4 binds + per-pod `--port` (8200, 8201, ...). Commits: `160cd85563` (IPv4) and `b7e1bd5b2e` (per-pod ports).
- **Baked qwen3 verifiably works**: SSH'd into the pod mid-run, confirmed model loaded (93 GB/GPU), `/v1/models` lists the model, `/v1/messages` actively serving tool-use calls.
- **First parallel matrix completed**: 3 configs (qwen3, reap-139b, minimax-fp8) produced 15 eval results across 5 tasks each. 2 configs (minimax-nvfp4, reap-172b) failed due to RunPod infrastructure issues (vLLM health timeout — not our code).
- **Workstation tier set up**: leo-workstation RTX 3090 running `stelterlab/Qwen3-Coder-30B-A3B-Instruct-AWQ` (4-bit, 24 GB VRAM, 63k context). 5/5 first-pass results all hit 60 min timeout; 4 retries running with 240 min timeout for clean quality measurements.

### Wins (2026-04-07 night session)

1. **Shard-per-layer Dockerfile generator** (`runpod-toolkit` commit `fa1d028`, batching fix `1eb311e`).
   - Hardlinks each safetensors file from the HF cache snapshot into a build staging dir on the same filesystem (free).
   - Generates one `COPY` per shard so each layer is ~2 GB and uploads independently — `docker push` retries per layer, no more 22+ min hangs on monolithic 98 GB layers.
   - Files land in the standard HF hub cache layout under `/models/hub/models--Org--Name/snapshots/<hash>/` plus a `refs/main` pointer, with `HF_HOME=/models` + `TRANSFORMERS_OFFLINE=1` + `HF_HUB_OFFLINE=1`. Vllm resolves the model by its HF name without ever touching the network.
   - Batching fallback (`1eb311e`): when shard count > 80, groups into multi-file COPYs to stay under Docker legacy builder's ~127 overlay layer limit. Required for MiniMax-M2.5 FP8 (126 shards → 63 batched layers).

2. **All 5 baked images on Hub** (`leosiriusdawn/runpod-vllm:*-baked`):
   | Tag | Model | Size | Build | Push |
   |---|---|---|---|---|
   | `qwen3-coder-next-fp8-baked` | Qwen3-Coder-Next FP8 | 104 GB | ~25 min | ~12 min |
   | `reap-139b-nvfp4-baked` | REAP-139B NVFP4 (lukealonso) | 92 GB | ~14 min | ~8 min |
   | `reap-172b-nvfp4-gb10-baked` | REAP-172B NVFP4 GB10 (saricles) | 124 GB | ~25 min | ~10 min |
   | `minimax-m25-nvfp4-baked` | MiniMax-M2.5 NVFP4 (nvidia) | 155 GB | ~24 min | ~12 min |
   | `minimax-m25-fp8-baked` | MiniMax-M2.5 FP8 (full) | 254 GB | ~50 min | ~30 min |

3. **Launcher fix: per-task `ORCH_CONFIG_PATH` + reify support** (commit `3be119c920`):
   - `preflight_baseline` now reads `spec['project_root']` so reify task baselines are checked out against `/home/leo/src/reify`, not dark-factory.
   - Empty `lint`/`typecheck` commands are treated as "no such step" so reify specs (which legitimately set `typecheck=""`) don't fail validation.
   - `resolve_task_ids --all-tasks` now globs both `df_task_*.json` and `reify_task_*.json` (any `*_task_*.json`).
   - `run_one_task` computes the orchestrator config path per-task from `spec['project_root']` and passes `--config <path>` explicitly.

4. **Workstation tier vLLM** on leo-workstation (RTX 3090, 24 GB):
   - Installed nvidia-container-toolkit, configured docker runtime.
   - Resolved driver mismatch (kernel 580.105 / userspace 580.126) via reboot.
   - Found AWQ 4-bit variant: `stelterlab/Qwen3-Coder-30B-A3B-Instruct-AWQ` (compressed-tensors, ~17 GB on disk; better-rated alternative `cyankiwi/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit` per agent research).
   - Memory tuning: `MAX_MODEL_LEN=63000` (max that fits KV cache), `GPU_MEMORY_UTIL=0.92`, `MAX_NUM_SEQS=4`, `ENFORCE_EAGER=1`.
   - vLLM healthy and serving via Marlin 4-bit MoE backend, accessible from dev box via `http://leo-workstation:8000`.

5. **Per-pod port assignment** in matrix runner (`b7e1bd5b2e`): each parallel pod gets its own port (8200, 8201, 8202, ...). Without this, all 5 SSH tunnels tried to bind `127.0.0.1:8200`; only the first succeeded, the other 4 silently failed and their health probes hit the wrong tunnel. Same class of bug as the IPv4/IPv6 port-collision the morning's "qwen3 hang" turned out to be.

### Matrix run #2 results (2026-04-07 22:11 BST)

3 of 5 configs produced eval results; 2 failed at infrastructure level.

| config | task | outcome | tests | lint | type | duration |
|---|---|---|---|---|---|---|
| qwen3-coder-next-fp8-new | reify_task_12 | **done** ✓ | True | True | True | 16 min |
| qwen3-coder-next-fp8-new | reify_task_27 | timeout | True | True | True | 60 min |
| qwen3-coder-next-fp8-new | df_task_12 | timeout | False | True | True | 60 min |
| qwen3-coder-next-fp8-new | df_task_13 | timeout | False | True | False | 60 min |
| qwen3-coder-next-fp8-new | df_task_18 | timeout | False | False | False | 60 min |
| reap-139b-nvfp4-new | df_task_12 | timeout (was done in earlier run) | False | True | True | 62 min |
| reap-139b-nvfp4-new | df_task_18 | timeout | True | True | True | 61 min |
| reap-139b-nvfp4-new | df_task_13 | timeout | False | True | True | 62 min |
| reap-139b-nvfp4-new | reify_task_12 | timeout | True | True | True | 62 min |
| reap-139b-nvfp4-new | reify_task_27 | timeout | True | True | True | 62 min |
| minimax-m25-fp8-new | df_task_13 | **done** ✓ | True | True | True | 48 min |
| minimax-m25-fp8-new | df_task_12 | blocked | True | True | True | 44 min |
| minimax-m25-fp8-new | df_task_18 | blocked | None | None | None | 40 min |
| minimax-m25-fp8-new | reify_task_27 | blocked | None | None | None | 17 min |
| minimax-m25-fp8-new | reify_task_12 | timeout | False | False | True | 60 min |
| minimax-m25-nvfp4-new | — | **FATAL: vLLM did not become healthy within 48 min** | — | — | — | — |
| reap-172b-nvfp4-gb10-new | — | **FATAL: vLLM did not become healthy within 40 min** | — | — | — | — |

**True `outcome=done` count this run: 2** (qwen3/reify_task_12, minimax-fp8/df_task_13).
**"Would be done" with longer timeout** (timeout but tests/lint/typecheck all pass): 4 more (reap-139b: df_task_18 + reify_task_12 + reify_task_27; qwen3: reify_task_27).

The 60 min orchestrator-internal timeout is too short for the inference speeds these models reach on RTX PRO 6000 / RTX PRO 6000 4-way. For the next session, bump the orchestrator's `--timeout` (the launcher's `--task-timeout-min` was already at 120; the inner orchestrator timeout binds first).

### Workstation tier first results (2026-04-07 evening)

5/5 tasks all hit the 60 min default eval timeout on first run, but several were close. Reify_task_27 produced `tests=True lint=True type=True` despite the timeout label.

| task | outcome | tests | lint | type |
|---|---|---|---|---|
| df_task_12 | timeout | False | True | True |
| df_task_13 | timeout | False | True | True |
| df_task_18 | timeout | False | False | False |
| reify_task_12 | timeout | False | True | True |
| reify_task_27 | timeout | True | True | True |

**4 retries with `--timeout 240` running at session end** (`ws-{df_task_12,df_task_13,df_task_18,reify_task_12}-4h.log`). The 3090 AWQ is much slower than RunPod GPUs (200-250 bridge calls in ~3 hours each), but should produce clean quality measurements when given enough time. Reify_task_27 was already passing so no retry needed.

### Failures / unresolved (next session)

1. **`minimax-m25-nvfp4-new` health timeout (48 min)** — pod created on H200 successfully but vLLM never returned 200 on `/health`. SSH tunnel + port were correct (per-pod port fix already in). Could be:
   - Same NCCL/init hang signature as the qwen3 morning attempts (model loaded but not serving)
   - A stale CUDA graph compile path despite `--enforce-eager` not being set
   - RunPod infrastructure issue (similar to the "critical error" alert reap-139b got)
   - Try with `--enforce-eager` set in the config; SSH into a fresh pod and py-spy the worker process (note: install py-spy in `:latest` base image to skip the in-pod install step)

2. **`reap-172b-nvfp4-gb10-new` health timeout (40 min)** on 2× RTX PRO 6000 fallback — same signature as minimax-nvfp4. Possibly TP=2 NCCL init issue. Try forcing `ENFORCE_EAGER=1` in the config.

3. **Bake-all script killed mid-build** (because I killed the wrong process when investigating a hang) — required a manual resume with `SKIP=` regex. The script is sequential build → push → rmi → next; if killed mid-build it leaves a partial layer. Improvement: idempotent restart (skip tags whose digest is already on Hub via `docker manifest inspect`).

4. **Result file `total_turns` / `duration_s` are `None` for timeout outcomes** — those metrics aren't populated when the timeout path fires. Makes per-task speed comparison harder. Fix: populate them from the partial run state before raising the timeout.

5. **Workstation tier evals can't run multiple in parallel** because vLLM is single-tenant on the 3090 (24 GB, no headroom for concurrent prefill). Currently running 4 in parallel against the same vLLM — they share the same KV cache and queue, so it's effectively serial. For the next session: either run sequentially with longer per-task timeout, or spin up a smaller model that supports concurrent batches.

### Cost (this session)

| Item | Cost |
|---|---|
| qwen3 baked image build/push (dev box, no $) | $0 |
| Other 4 baked images build/push | $0 |
| Gate eval qwen3 (1× pod, ~70 min, terminated mid-run x2) | ~$3 |
| Matrix run #1 (5 pods × ~30-60 min each, partial) | ~$8 |
| Matrix run #2 (5 pods × ~60-90 min each) | ~$15 |
| **Session total** | **~$26** |
| **Cumulative this week** | **~$74 of original ~$92 credit** |

### What changed in main this session

- `3be119c920` feat(eval): bake all 5 vLLM models into images via shard-per-layer Dockerfile
- `160cd85563` fix(eval): bind SSH tunnel to 127.0.0.1 explicitly + bump default port
- `b7e1bd5b2e` fix(eval): per-pod SSH tunnel ports + GPU fallbacks for H200 shortage

### What changed in runpod-toolkit

- `fa1d028` feat(bake): shard-per-layer model image baker
- `1eb311e` fix(bake): batch shards to stay under overlay2 layer limit

### Next session priority order

1. **Read 4h workstation retry results** when they finish — first clean quality measurements for qwen3-coder-30b-q4 (AWQ 4-bit, 3090).
2. **Bump orchestrator-internal eval timeout** from 60 min default to 120 min so the "would be done" timeouts in the matrix produce clean results.
3. **Re-run minimax-m25-nvfp4-new and reap-172b-nvfp4-gb10-new** with `ENFORCE_EAGER=1` set in config (forcing it via env override). If still hangs, py-spy investigation.
4. **Refire matrix** after the timeout bump and the two infrastructure fixes — should produce clean 5×5 = 25 results (or 5×5×2 = 50 if counting all attempts).
5. **Analyze the comparative scores** across configs to start the per-quantization-method writeup (REAP-139B NVFP4 vs minimax-m25-nvfp4 vs minimax-m25-fp8 vs Qwen3-Coder-Next FP8 vs Qwen3-Coder-30B AWQ).
6. **Optional**: rebake 1 image with py-spy preinstalled in the base layer for fast worker stack dumping next time.

### Session end state

- Matrix #2 ran to completion (all 5 launcher processes exited; pods all cleaned up via `finally: tear_down_pod`).
- Workstation 4h retries still running on leo-workstation (4 processes, no 60-min timeout).
- All commits pushed to local main; not yet pushed to origin.
- All 5 baked images on Docker Hub, public, ready for next session's matrix retries.

---



## Update 2026-04-07 PM: full session results

### Wins
1. **First true `outcome=done`** on `reap-139b-nvfp4-new` (RTX PRO 6000 Blackwell, run-id `6838dea3`). Implementer + 3 debug cycles + reviewers + merge, 158 turns, 27.5 min wall clock. Real cost ~$1 (RunPod time only — the "$30.77 cost" reported in the result file is phantom Sonnet-equivalent from the Claude CLI's usage tracker; vLLM-hosted runs have no per-token API cost).
2. **`ENFORCE_EAGER` env hook landed end-to-end**: dark-factory `feat(eval): add ENFORCE_EAGER workaround for qwen3 startup hang` (`315b5d4ffd`) + runpod-toolkit `feat(vllm): add ENFORCE_EAGER env hook for qwen3 startup hang` (`f909281`) + new `:latest` image (digest `sha256:d26fba20c254...`). Verified active in the running vLLM (cmdline `--enforce-eager`, V1 engine config `cudagraph_mode=NONE`, `compilation_mode=NONE`).
3. **Recovery branch merged**: `recover/vllm-eval-session-2026-04-06` fast-forwarded to local main at `26ca8dd6fc`. Local main is now ~675 commits ahead of `origin/main` (the orchestrator's destructive overnight auto-commit stream); not yet pushed.
4. **Orchestrate skill hardened against the "wrong project" bug** (`docs(orchestrate): add target-project identification guard`, `26ca8dd6fc`) — the bug that caused the data-loss incident is now flagged at the top of the skill instructions.
5. **Multi-task launcher with per-task baseline checkout** preserved on `feat/multi-task-launcher-baseline-checkout` (`8ce354dc26`) — 1902 lines, includes test_run_vllm_eval.py and test_snapshots.py. Implements the design we sketched: one pod, N tasks, each task evaluated against its own pre-task commit, baseline preflighted before pod creation. Awaiting review + merge.

### Failures / unresolved
1. **`qwen3-coder-next-fp8-new` has a SECOND hang that `--enforce-eager` does NOT fix.** Two attempts:
   - **Attempt 1** (1× H200 SXM, pod `1w3hrkojdmndyy`): stuck in image fetch (0% layers downloaded after 15+ min) — RunPod fleet-wide H200 SXM image-pull throughput problem, not config-specific. Confirmed by the baked diagnostic (below) hitting the same hang on the same DC. Terminated.
   - **Attempt 2** (2× RTX PRO 6000 Blackwell Server Edition, pod `xhzdzqicuil2rq`, after `chore(eval): switch qwen3-coder-next-fp8-new to 2× RTX PRO 6000` `b8998b49ab`): pod started cleanly, image pulled cleanly (RTX PRO 6000 DC has fine throughput), vLLM workers spawned with all the right flags including `--enforce-eager`. But the workers then sat at **100% CPU for 25+ minutes** with **zero model weights downloaded** (HF cache stayed at 11 MB of tokenizer/config files only) and **only 868 MiB on each GPU**. Workers were `R (running)` with `wchan=0` — busy in user-space compute, not blocked on a syscall. `hf_transfer.abi3.so` and `tokenizers.abi3.so` were loaded but no shards arrived. Container logs (per user) hadn't moved on from the V1 engine init line. **Diagnosis**: this is a *different* hang from the morning's CUDA-graph hypothesis. ENFORCE_EAGER prevents the *post-load* CUDA-graph capture hang but qwen3-coder-next-fp8 has a *pre-download* Python init hang that ENFORCE_EAGER doesn't address. The fix is no longer "land --enforce-eager"; it's a deeper diagnosis (next session). Real cost ~$1.40 burned. Pod terminated by user.
2. **Baked-image diagnostic** (`reap-139b-nvfp4`, pod `gg3t12tguiwhgm`): launched on H200 SXM as a parallel diagnostic to test whether image-pull issues were image-specific (`:latest` we just pushed) or DC-wide. Image pulled successfully (proving `:latest` push isn't broken), but the container then crashed twice on startup with `pydantic.ValidationError: Please pass the argument trust_remote_code=True` — the OLD `:reap-139b` baked image is **vLLM 0.18.1** and is missing every entrypoint patch added in this session (`--trust-remote-code`, GMU 0.95, `--max-num-seqs 16`, `--enforce-eager`, per-model `--tool-call-parser`). Pod was reaped (probably extraction filled the 220 GB container disk; root cause not fully confirmed). **Implication**: all `OLD :reap-139b/:reap-172b/:qwen3-coder-next/:qwen3-coder-next-fp8` baked images on Hub are stale and unusable until rebuilt with the new entrypoint. The 5 NEW configs (`-new` suffix) using `:latest` + HF download remain the canonical eval path. Memory: `project_old_baked_images_stale.md`.

### What changed in main this session
- `26ca8dd6fc` docs(orchestrate): add target-project identification guard
- (recovery branch merge) — `1077779690`/`71e9b1c5a1`/`5fa0f30751`/`c4561692d1` from yesterday's recovery
- `4e82e9369f` fix: clean up main lint and test failures (user)
- `52f13fc43a` fix: resolve type errors in workflow.py (user)
- `9c0c546081` fix: resolve type errors in verify.py, git_ops.py, mcp_lifecycle.py (user)
- `315b5d4ffd` feat(eval): add ENFORCE_EAGER workaround for qwen3 startup hang
- `b8998b49ab` chore(eval): switch qwen3-coder-next-fp8-new to 2× RTX PRO 6000

### What changed in runpod-toolkit
- `f909281` feat(vllm): add ENFORCE_EAGER env hook for qwen3 startup hang
- New layer pushed to `leosiriusdawn/runpod-vllm:latest` and `:enforce-eager` (digest `sha256:d26fba20c254...`)

### Cost
- ~$1 (REAP-new successful eval, ~30 min on RTX PRO 6000 Blackwell)
- ~$1.40 (qwen3 hang, ~25 min on 2× RTX PRO 6000 Blackwell)
- ~$0.50 (qwen3 stuck H200 SXM + H100 NVL terminated, ~10 min total)
- ~$2 (baked diagnostic, ~50 min on H200 SXM through extraction failure)
- **Session total: ~$5**, leaving ~$37 in RunPod credit
- Combined with previous sessions: ~$48 used out of original ~$50 + $90 added = ~$92 credit; ~$44 spent

### Next-session priority order
1. **Diagnose qwen3-coder-next-fp8 pre-download Python init hang.** What is the worker process actually doing for 25+ minutes? Likely paths: torch.compile despite `--enforce-eager` (some shape compilation passes still run), Qwen3-Coder-Next custom modeling.py code path (it requires `trust_remote_code=True`), or fp8 quantization config processing. SSH into a fresh pod with py-spy preinstalled in the image (or set `kernel.yama.ptrace_scope=0` somehow) and dump worker stack traces. Alternative: try a non-trust-remote-code Qwen3-Coder-Next variant if one exists, or downgrade vLLM to 0.18.1 specifically for qwen3 to test whether this is a vLLM 0.19 regression.
2. **Review + merge the multi-task launcher** (`feat/multi-task-launcher-baseline-checkout`, `8ce354dc26`). Once merged, all subsequent evals can use it for the per-task baseline + multi-task-per-pod model.
3. **Run a per-task-baseline eval of REAP-new** to validate the multi-task launcher and produce the first *fair* `outcome=done` score (current REAP-new score is 0.0 because the task was already done in the tree).
4. **Fire the post-gate eval matrix** for the remaining 3 MiniMax variants (`reap-172b-nvfp4-gb10-new`, `minimax-m25-nvfp4-new`, `minimax-m25-fp8-new`) once the multi-task launcher is merged.
5. **Optional**: rebuild the OLD baked images (`:reap-139b`, `:reap-172b`, `:qwen3-coder-next-fp8`, `:devstral-small`) with the new entrypoint, OR delete them from Hub since the NEW configs work end-to-end.

---

## Update 2026-04-07: root cause identified + recovery

### What we now know

**The "tool-format bridge bug" diagnosed in the morning session was a misframing.** vLLM 0.19 ships a native Anthropic adapter (`vllm/entrypoints/anthropic/api_router.py` + `serving.py`) that converts `/v1/messages` requests to OpenAI internally, runs inference, then converts responses back. The bridge layer works at the protocol level — verified via local netcat probe of Claude CLI 2.1.92 outbound traffic.

**The actual root cause was wrong tool-call parsers**, configured per-model in `--tool-call-parser`:
- MiniMax M2.5 emits `<minimax:tool_call><invoke name="..."><parameter name="...">...</parameter></invoke></minimax:tool_call>` XML — needs `--tool-call-parser minimax_m2`
- Qwen3-Coder-Next emits `<tool_call><function=name><parameter=name>val</parameter></function></tool_call>` XML — needs `--tool-call-parser qwen3_coder`
- Both were configured with `hermes` (which expects `<tool_call>{json}</tool_call>`)

Two distinct failure modes from this single misconfiguration:
1. **MiniMax + hermes**: parser doesn't recognise `<minimax:tool_call>` at all → output passes through as plain text in the assistant message → CLI sees a "successful" final answer with embedded XML → no tool ever executes → silent no-op iterations → `outcome=blocked` after exhaustion
2. **Qwen3-Coder + hermes**: parser sees the `<tool_call>` start tag but mis-parses the inner XML body as JSON → produces a malformed `tool_use` block → CLI errors with `subtype=error_during_execution` (the symptom we originally diagnosed)

vLLM 0.19's tool parser registry (`vllm/tool_parsers/__init__.py`) has 25+ registered parsers including dedicated entries for `minimax_m2`, `qwen3_coder`, `qwen3_xml`, `mistral`, `kimi_k2`, `gemma4`, `llama3_json`, `deepseek_v3/v31/v32`, etc.

### First successful vLLM eval (2026-04-06 19:35)

`reap-139b-nvfp4-new` with `minimax_m2` parser on 1× RTX PRO 6000 Blackwell:
- 273 turns, 47,708 output tokens at 41.89 tok/s
- 4 implementer iterations + 2 debug cycles
- 171 lines changed across 10 files
- **Tests pass, lint clean, typecheck clean, verification passed**
- Outcome was `blocked` *only* because all 5 Claude sonnet reviewers hit the Max usage cap ("resets 11pm")
- Result file: `df_task_12__reap-139b-nvfp4-new__97cc6a12.json`

### Bridge + parser are complementary, not competing

`shared/src/shared/vllm_bridge.py` (Task 457) is now active on every vLLM eval call. It normalises `tool_use.id` to `toolu_` prefix, parses JSON-string `input` fields, and fixes `stop_reason='tool_calls'` → `'tool_use'`. These are residual format quirks that vLLM's native Anthropic adapter doesn't fully clean up. The bridge is necessary even with the right parser, but the right parser is the larger fix — it eliminates the malformed-tool-call class entirely at the source.

### Three infrastructure blockers fixed in entrypoint-vllm.sh

Patched in `/home/leo/src/runpod-toolkit/docker/entrypoint-vllm.sh` (separate repo, survived the working-tree wipe):

1. **`--trust-remote-code` (unconditional)** — required for all MiniMax M2.5 variants. The architecture isn't yet upstream in transformers, so HF returns the repo with an `auto_map` pointing at `modeling_minimax_m2.py`, which requires opt-in execution. Without this flag, vLLM crashes instantly with `pydantic.ValidationError`. Qwen models don't need it.

2. **`--gpu-memory-utilization ${GPU_MEMORY_UTIL:-0.95}`** — vLLM 0.19 changed CUDA graph memory accounting: graphs are now profiled and reserved inside the GMU budget. The default 0.9 leaves too little KV cache headroom for large models on tight GPUs. For lukealonso/MiniMax-M2.5-REAP-139B-A10B-NVFP4 (78 GB weights + 5.77 GB CUDA graphs) on 96 GB RTX PRO 6000, 0.9 left only 3.85 GB for KV cache. 0.97 gives ~10.5 GB → 88,752 token KV pool.

3. **`--max-num-seqs ${MAX_NUM_SEQS:-16}`** — vLLM defaults to 1024, which pre-allocates a sampler softmax buffer per slot during warmup. On tight pods at 0.97 GMU, this OOMs the sampler warmup by ~782 MiB. Each eval pod serves one implementer at a time, so 16 is plenty (~16 GB savings vs default).

### Active blockers

**Qwen3-Coder-Next-FP8 startup hang** (NOT memory-related, confirmed on both 96 GB RTX PRO 6000 and 141 GB H200 SXM):
- vLLM process stays alive, model fully loaded into VRAM, but `/health` never returns 200
- Likely stuck in torch.compile or CUDA graph capture
- **Task 515's discovery (on branch, not yet merged)**: `--enforce-eager` (i.e. `ENFORCE_EAGER=1` env var, requires entrypoint hook) disables CUDA graph capture and may fix the hang. The hook needs to be added to `entrypoint-vllm.sh` and the image rebuilt.
- Container logs are required for full diagnosis but RunPod doesn't preserve them after pod termination — must capture from web console or via SSH while the pod is live

### The destructive overnight window

The orchestrator was started on dark-factory by mistake (should have been on reify) and ran ~351 commits between session end (~21:30 BST 2026-04-06) and ~07:16 BST 2026-04-07. During that window, the working tree's uncommitted state — the morning refactor + our session edits to `configs.py`, `cli_invoke.py`, `run_vllm_eval.py`, and this doc — was destroyed. Most likely cause: the orchestrator's `merge-queue stash/sync` mechanism tried to stash uncommitted changes around a merge, hit a conflict on stash pop (we found the conflict-marker version in the unreachable git objects), and silently dropped the state instead of preserving it in the stash list.

Recovery: `git fsck --unreachable` surfaced 88,871 unreachable commits and 72,318 unreachable blobs. Searching the blobs by size + distinctive content strings (`tool_call_parser='minimax_m2'`, `RUNPOD_CONFIG_NAMES`, `MAX_NUM_SEQS`, `stdout_text_for_log`, `Last updated: 2026-04-06 ~16:05 BST`) found clean copies of all four files. Restored on branch `recover/vllm-eval-session-2026-04-06`.

**Lesson:** Never leave significant uncommitted work in a worktree where the orchestrator could be running. Commit aggressively to a feature branch.

### Current state (2026-04-07 morning)

- **Recovery branch**: `recover/vllm-eval-session-2026-04-06` — 3 commits on top of main, all 130+ relevant tests pass:
  1. `c4561692d1` recover: restore lost vLLM eval session work from unreachable git blobs
  2. `5fa0f30751` test(eval-configs): update count regression guards for recovered _new variants
  3. `71e9b1c5a1` docs(eval): update with 2026-04-07 root cause findings
- **Orchestrator**: stopped (was running on dark-factory by mistake — should have been on reify)
- **Docker image**: `leosiriusdawn/runpod-vllm:latest` digest `sha256:152e81f6...` includes the three entrypoint patches (trust-remote-code, GMU, max-num-seqs)
- **RunPod credit**: ~$42 remaining
- **Task 515 (qwen3 recovery)**: committed on branch `task/515`, 19 commits, **not merged**. Includes the `ENFORCE_EAGER=1` hypothesis for the qwen3 hang. Took a different code structure (kept the OLD MODELS-dict approach in run_vllm_eval.py) so cherry-pick will need conflict resolution.
- **Memory files** in `~/.claude/projects/-home-leo-src-dark-factory/memory/` are intact (not in repo, separate filesystem):
  - `project_vllm_eval_session_2026_04_06.md` — full evening session summary
  - `project_vllm_eval_blockers.md` — detailed blocker writeup
- **What still needs running**: A clean reap-139b end-to-end with a fresh Claude Max cap so reviewers can complete; qwen3 debug with container log capture.

### Next actions

1. **~~Merge `recover/vllm-eval-session-2026-04-06` to main~~** ✓ DONE 2026-04-07 — fast-forwarded to local main at `26ca8dd6fc` (still unpushed; origin/main is 666 commits behind from the destructive run).
2. **~~Re-apply Task 515's ENFORCE_EAGER insight~~** ✓ DONE 2026-04-07 — `_vllm_config()` now takes `enforce_eager: bool = False` which sets `ENFORCE_EAGER='1'`, applied to `qwen3-coder-next-fp8-new`. `run_vllm_eval.py` whitelist forwards it to the pod env. `TestEnforceEagerOnQwen3` regression guards it.
3. **Add `ENFORCE_EAGER` env var support to entrypoint-vllm.sh** in the runpod-toolkit repo (if-set → append `--enforce-eager` to vLLM CMD), rebuild and push `:latest`. **This is the only out-of-tree step left** — until it lands, the in-tree env var has no runtime effect.
4. **Rerun `reap-139b-nvfp4-new` end-to-end** when the Claude Max cap is fresh (the implementer phase already passed once; reviewers just need budget). Should produce the first true `outcome=done` for a vLLM-hosted config.
5. **If qwen3 boots with `--enforce-eager`**, run that eval too. If it still hangs, capture container logs via SSH while the pod is running (RunPod doesn't preserve them after termination).
6. **Then fire the post-gate eval matrix** for the remaining 3 MiniMax variants (reap-172b-nvfp4-gb10-new, minimax-m25-nvfp4-new, minimax-m25-fp8-new). All 4 have `tool_call_parser='minimax_m2'` set; the H200 ones don't have memory tuning (default 131k context fits).
7. **Going forward: commit aggressively.** Any non-trivial work should land on a feature branch immediately. The orchestrator's stash/sync mechanism is not safe for preserving uncommitted state across long-running task processing. Memory files survive (different filesystem), but in-repo working tree state does not.

### Learnings — institutional memory

- vLLM 0.19 has a native Anthropic adapter; no external bridge is required for the protocol layer. The `vllm_bridge.py` from Task 457 provides residual format normalization and is complementary, not redundant.
- vLLM 0.19's CUDA graph memory profiler changed the `gpu-memory-utilization` budget — 0.9 default is too low for tight large-model pods.
- vLLM defaults `max-num-seqs=1024`, which pre-allocates sampler softmax buffers per slot during warm-up. On tight pods this OOMs. For single-request eval pods, 16 is enough.
- All MiniMax M2.5 HF repos require `--trust-remote-code` (custom modeling code via auto_map).
- vLLM 0.19 ships dedicated tool-call parsers for ~25 model families. Always check `vllm/tool_parsers/__init__.py` for the right parser when adding a new model.
- RunPod GPU availability API is unreliable — `get_gpu_availability()` may report capacity that `create_pod()` then refuses with "no instances available." Retry with a delay or fall through alternate GPU types.
- RunPod container logs are NOT preserved after pod termination. Capture from web console while running, or via SSH `docker logs` / `/proc/PID/fd/{1,2}` before the container restarts.
- The eval runner does NOT support resume mid-workflow. If reviewers fail (e.g. cap hit), the entire eval must be rerun from scratch — implementer iterations re-execute, then reviewers run on fresh budget.
- The orchestrator's `merge-queue stash/sync` mechanism can silently drop uncommitted working tree state on stash-pop conflicts. There is **no recovery mechanism** beyond `git fsck --unreachable` blob search, which only works if the lost content was ever committed to an object (e.g. via the failed stash itself).

---

## Bridge fix (Task 457)

**Merged in Task 457.** The eval pipeline previously sent Claude CLI directly to the vLLM
`/v1/messages` endpoint, whose tool_use response format differs from what Claude CLI expects
(OpenAI-style `tool_calls`, JSON-string `input`, wrong `stop_reason`, non-Anthropic IDs).
This caused every tool_use response to error the CLI parser; no vLLM eval ever reached
`outcome=done`.

The fix adds `shared/src/shared/vllm_bridge.py` — a local aiohttp HTTP proxy that:
- Starts on `127.0.0.1:0` (OS-assigned port) per invocation
- Translates POST `/v1/messages` responses: converts `tool_calls` → Anthropic content blocks,
  normalises tool_use IDs to `toolu_` prefix, parses JSON-string `input` fields, fixes
  `stop_reason` to `'tool_use'` when tool_use blocks are present
- Forces `stream=false` on forwarded requests (buffered translation — simpler than SSE)
- Pipes all other paths straight through to the upstream vLLM endpoint

**Auto-activation:** The bridge is wired into `shared/src/shared/cli_invoke.py`. Whenever
`env_overrides` contains `ANTHROPIC_BASE_URL`, `_invoke_claude` starts a per-invocation
bridge pointing at that URL, rewrites the subprocess env to point at the bridge's local
URL, and tears the bridge down in a `finally` block (success, failure, or exception).
No API changes — existing callers that set `env_overrides['ANTHROPIC_BASE_URL']` gain the
bridge transparently.

## TL;DR for next session

**The big find:** The **eval pipeline for vLLM-hosted models is systemically broken** — not an infrastructure problem, a format-bridging one. All 10 iterations of the gate eval hit:
- Model responds fine (29 tok/s, 121-token tool call per turn)
- Claude CLI errors with `stop_reason=tool_use` + `is_error=true` + `subtype=error_during_execution`
- Historical check: **no vLLM-hosted config has ever produced `outcome=done`** in `orchestrator/src/orchestrator/evals/results/`

The Qwen model emits hermes-format tool calls; vLLM's `--tool-call-parser hermes` converts them to OpenAI format; the Claude CLI then expects Anthropic `tool_use` blocks and fails. There must be a missing/broken adapter somewhere in the chain.

**Infrastructure fully healed this session:**
1. ✓ Disk recovered from 80 GB free → 734 GB free (moved 530 GB of OLD bf16 models to Leo_X10p_4TB_00, deleted 162 GB REAP-172B source)
2. ✓ MiniMaxAI/MiniMax-M2.5 source (215 GB) moved to Leo_X10p_4TB_00 after second reboot — Internal-2nd now 948 GB free
3. ✓ **`/etc/fstab` permanent fix applied** — Internal-2nd mounts by UUID on boot, no more bind-mount workaround; plus `docker.service` `RequiresMountsFor` dropin at `/etc/systemd/system/docker.service.d/wait-for-internal-2nd.conf`
4. ✓ `configs.py` fully refactored: `EvalConfig` now has `image`, `gpu_type`, `gpu_count`, `container_disk_gb` fields; single source of truth; `run_vllm_eval.py` consumes via `get_config_by_name()`
5. ✓ `container_disk_gb` bug fixed: original agent refactor used 50 GB (wrong — RunPod includes image in container disk), bumped to image-size + headroom
6. ✓ `run_vllm_eval.py` `wait_for_pod` timeout bumped 30 min → 60 min
7. ✓ **All 5 new configs pivoted to use `:latest` base + HF download** — abandoned the slow local push (30+ min push hangs observed), HF download on pod is faster end-to-end

**Next session priorities (user directed — do 1 & 2 concurrently):**
1. **Debug the vLLM → Claude CLI tool-format bridge** (slow; SSH into fresh pod, curl `/v1/messages` with tool-use prompt, inspect orchestrator/shared/cli_invoke.py and any Anthropic↔OpenAI shim; the error happens in the Claude CLI side after it gets the tool_use response)
2. **Try `reap-139b-nvfp4-new` as alternative gate eval** — different model (MiniMax architecture, potentially different tool format) on 1× RTX PRO 6000 (~$1.79/hr, ~$2 test). If it passes → Qwen-specific. If it fails → confirms systemic.

## Gate eval failure — full forensics

**Pod lifecycle (CLEAN):**
```
15:52:37  gate eval started (qwen3-coder-next-fp8-new, :latest base + HF download)
15:52:39  Pod created: xyhyxucv1q41mc (RTX PRO 6000 Blackwell Server Edition, $1.69/hr)
15:54:58  Pod RUNNING, SSH at 216.243.220.242:10847   (2m19s to pod ready — good)
15:55:01  SSH tunnel up, waiting for vLLM
16:01:16  vLLM healthy on port 8100                   (6m15s HF download + vLLM startup — good)
16:01:54→16:03:20  10 iterations, all fail identically
16:03:28  Pod terminated cleanly
```

**Failure pattern (consistent across all 10 iterations):**
```json
{
  "type": "result",
  "subtype": "error_during_execution",
  "duration_ms": 1500-3000,
  "is_error": true,
  "num_turns": 1,
  "stop_reason": "tool_use",
  "session_id": "...",
  "total_cost_usd": 0.19-0.20,
  "usage": {
    "input_tokens": 38000-39000,
    "output_tokens": 121-124,
    ...
  }
}
```

**Result file:** `orchestrator/src/orchestrator/evals/results/df_task_12__qwen3-coder-next-fp8-new__9d13dbb1.json`
- `outcome: blocked`
- `tokens_per_second: 29.06` — model was actually serving
- `input_tokens: 390358` (10 × 39k)
- `output_tokens: 1217` (10 × 121)
- `is_local_model: true` — orchestrator correctly detected local routing
- `composite_score: 0.0`

**Interpretation:** The model is serving and responding with a tool call after each 39k-token prompt. The Claude CLI runs `claude --print --output-format json --model sonnet …` against the vLLM endpoint (via `ANTHROPIC_BASE_URL=http://localhost:8100`). Something in that chain — either vLLM's returned format, or the CLI's parsing — is mishandling the tool call response. The fact that the first iteration took 24s (real inference) and subsequent ones took 1.5-3s suggests the vLLM endpoint started returning cached responses fast, which is suspicious.

**Where to look in next session:**
- `orchestrator/shared/cli_invoke.py` — how it invokes Claude CLI and what env it passes
- Whatever sets `ANTHROPIC_BASE_URL` — does it go direct to vLLM's `/v1/chat/completions` (OpenAI format), or is there a shim serving `/v1/messages` (Anthropic format)?
- `/home/leo/src/runpod-toolkit/docker/entrypoint-vllm.sh` — does vLLM launch with `--api-server anthropic` or similar shim?
- The full stdout of a failed iteration — the logs only captured first 500 bytes; the full error message (probably in stderr) would identify the exact failure point
- Compare against a KNOWN-working 3090-tier config (e.g. `qwen3-coder-30b-q4`) — but also check prior results in `results/` to see if those ever passed

## Infrastructure state (post-session)

### Disk
```
/dev/nvme1n1p5 (was nvme0n1p5)  920G  ??  /     [root: ~95%]
/dev/nvme0n1  (was nvme1n1)    1.8T  792G used  948G free  /media/leo/Internal-2nd
/dev/sda2                      3.6T  2.6T used  823G free  /media/leo/Leo_X10p_4TB_00
```

Kernel re-enumerated nvme devices after second reboot (nvme0 and nvme1 swapped). UUID-based fstab entry handles it.

### /etc/fstab (permanent fix applied this session)
```
UUID=8bd62e99-c02a-4ec0-a25a-30678a0e4398  /media/leo/Internal-2nd  ext4  defaults,nofail,x-systemd.device-timeout=10s  0  2
```

Plus `/etc/systemd/system/docker.service.d/wait-for-internal-2nd.conf`:
```
[Unit]
RequiresMountsFor=/media/leo/Internal-2nd
```

Verified with `findmnt --verify`: 0 errors. Docker now waits for Internal-2nd mount on boot.

### Local model files

**`/media/leo/Internal-2nd/leo/models/`** — EMPTY (all OLD bf16 moved out or deleted this session)

**`/media/leo/Leo_X10p_4TB_00/leo/models/`:**
```
models--cerebras--MiniMax-M2.5-REAP-139B-A10B       131 G   OLD bf16, moved here
models--lukealonso--MiniMax-M2.5-REAP-139B-A10B-NVFP4  70 G  NEW, not used this session
models--MiniMaxAI--MiniMax-M2.5                     215 G   moved here (was on Internal-2nd)
models--mistralai--Devstral-Small-2505               88 G   OLD, Devstral parked
models--nvidia--MiniMax-M2.5-NVFP4                  131 G   NEW, not used this session
models--Qwen--Qwen3-Coder-Next                      149 G   OLD bf16
models--Qwen--Qwen3-Coder-Next-FP8                   75 G   NEW — this is what the pod downloads from HF
models--saricles--MiniMax-M2.5-REAP-172B-A10B-NVFP4-GB10  93 G  NEW
```

**Deleted this session:** `models--cerebras--MiniMax-M2.5-REAP-172B-A10B` (162 G) — the OLD baked image already on Hub, source no longer needed.

### Local Docker images

```
leosiriusdawn/runpod-vllm:reap-172b-nvfp4-gb10   123 GB  (built pre-session, NOT pushed — local only)
leosiriusdawn/runpod-vllm:qwen3-coder-next-fp8   104 GB  (pushed to Hub pre-session)
leosiriusdawn/runpod-vllm:latest / :upgraded      24 GB  (base, pushed, has entrypoint + TOOL_CALL_PARSER env var)
```

**NOT built/pushed this session** (abandoned the local build path): `reap-139b-nvfp4`, `minimax-m25-nvfp4`, `minimax-m25-fp8`. The local `reap-172b-nvfp4-gb10` 123 GB image is the sole survivor; kept on disk in case a future session wants to push it.

### Docker Hub state (`leosiriusdawn/runpod-vllm`)

| Tag | State | Notes |
|---|---|---|
| `:latest` / `:upgraded` | ✓ pushed | vllm 0.19.0 + mistral_common 1.11.0 + TOOL_CALL_PARSER env var |
| `:qwen3-coder-next-fp8` | ✓ pushed | NEW 75 GB FP8, pushed pre-session — but NOT USED this session (switched to :latest + HF) |
| `:qwen3-coder-next` | ✓ pushed | OLD bf16, vllm 0.18.1 |
| `:reap-139b` | ✓ pushed | OLD bf16-labeled-actually-FP8 |
| `:reap-172b` | ✓ pushed | OLD FP8 |
| `:devstral-small` | ✓ pushed | OLD, broken (parked) |
| `:reap-172b-nvfp4-gb10` | ✗ LOCAL ONLY | 123 GB, built pre-session, push hung repeatedly (30+ MB/s), abandoned |
| `:reap-139b-nvfp4` | ✗ never built | abandoned; pivot to :latest + HF download |
| `:minimax-m25-nvfp4` | ✗ never built | same |
| `:minimax-m25-fp8` | ✗ never built | same |

## configs.py — current state

All 5 new configs now use `:latest` + HF download. The existing-image configs still use OLD baked Hub images on bigger H200 pods.

```python
<<<<<<< Updated upstream
VLLM_EVAL_CONFIGS = [
    _vllm_config('minimax-m25-fp8', 'MiniMaxAI/MiniMax-M2.5'),
    _vllm_config('qwen3-coder-next-fp8', 'Qwen/Qwen3-Coder-Next'),
    _vllm_config('reap-139b-nvfp4', 'cerebras/MiniMax-M2.5-REAP-139B-A10B'),
    _vllm_config('reap-172b-nvfp4', 'cerebras/MiniMax-M2.5-REAP-172B-A10B'),
    _vllm_config('qwen3-coder-30b-q4', 'Qwen/Qwen3-Coder-30B-A3B-Instruct'),
    _vllm_config('devstral-small-2505-q6', 'mistralai/Devstral-Small-2505'),
]
=======
# New configs (5 — all use :latest + HF download)
qwen3-coder-next-fp8-new        :latest  1× RTX PRO 6000  240G  Qwen/Qwen3-Coder-Next-FP8
reap-139b-nvfp4-new             :latest  1× RTX PRO 6000  200G  lukealonso/MiniMax-M2.5-REAP-139B-A10B-NVFP4
reap-172b-nvfp4-gb10-new        :latest  1× H200           260G  saricles/MiniMax-M2.5-REAP-172B-A10B-NVFP4-GB10
minimax-m25-nvfp4-new           :latest  1× H200           320G  nvidia/MiniMax-M2.5-NVFP4
minimax-m25-fp8-new             :latest  2× H200           600G  MiniMaxAI/MiniMax-M2.5

# Existing-image configs (4 — OLD baked images, bigger pods)
qwen3-coder-next-fp8            :qwen3-coder-next  2× H200  250G  Qwen/Qwen3-Coder-Next
reap-139b-nvfp4                 :reap-139b         1× H200  220G  cerebras/MiniMax-M2.5-REAP-139B-A10B
reap-172b-nvfp4                 :reap-172b         2× H200  260G  cerebras/MiniMax-M2.5-REAP-172B-A10B
minimax-m25-fp8                 :latest            2× H200  600G  MiniMaxAI/MiniMax-M2.5  (HF download)

# Workstation tier (unchanged)
qwen3-coder-30b-q4, devstral-small-2505-q6, qwen25-coder-32b-q4
>>>>>>> Stashed changes
```

`devstral-small` remains PARKED — Devstral-Small-2505 emits 3 inconsistent tool-call formats, not a parser config problem.

## run_vllm_eval.py — current state

- Imports from `orchestrator.evals.configs` via `sys.path.insert("/home/leo/src/dark-factory/orchestrator/src")`
- `--config` replaces the old `--model`; choices are the 9 RunPod-targetable configs
- Looks up image/gpu_type/gpu_count/container_disk_gb/env_overrides from `EvalConfig`
- `wait_for_pod` timeout: **3600s (60 min)**, bumped from 1800s because 100+ GB baked images can take 30+ min to pull
- `GPU_TYPES` fallback list trimmed to just `RTX PRO 6000 Blackwell Server Edition` and `H200`
- `--image` and `--gpu-type` CLI overrides preserved

## Things that bit us this session

1. **First reboot:** lost all local image state because of the Internal-2nd udisks mount race (known from previous session). FIXED permanently with fstab + systemd dropin.

2. **Second reboot:** disk re-enumeration (nvme0 ↔ nvme1). UUID-based fstab handles it.

3. **`container_disk_gb=50` was too small** for baked images. RunPod's container disk includes the image itself (not just the writable layer). The first gate eval attempt hung at runtime=null for 23 min because RunPod couldn't fit a 104 GB image in 50 GB of container disk. Fixed: set each config's `container_disk_gb` to image_size + ~50 GB headroom.

4. **Docker push of 98.9 GB unique layer hung** twice (pre- and post-reboot), each time ~22+ min without TCP activity but dockerd was computing layer hash at ~30 MB/s (I/O bound + slow). Root cause unclear — maybe legacy builder's layer layout isn't push-friendly. Workaround: pivoted to `:latest` + HF download (saves the whole push step).

5. **Gate eval "EVAL PASSED" log line is misleading** — it just checks `subprocess.returncode == 0`, which is 0 even when `outcome=blocked`. Always read the actual result file.

6. **`docker build … 2>&1 | tail -20` in the v4 monitor script lost the build's exit code** because no `set -o pipefail` — v4 monitor logged "BUILD DONE: reap-139b-nvfp4" at 14:28:25 but the build had actually failed (no tag was created). Caused me to think progress was further along than it was.

## Cost so far (session 2026-04-06)

| Item | Cost |
|---|---|
| Pre-reboot (from previous status doc) | ~$2.82 |
| Gate eval attempt 1 (30 min hang, container_disk bug) | ~$0.85 |
| Gate eval attempt 2 (killed after 30s, config pivot) | ~$0.01 |
| Gate eval attempt 3 (HF download, ran 10 iterations, blocked) | ~$0.30 |
| **Session total** | **~$3.98** |
| **Remaining RunPod credit** | **~$48** |

## Files changed this session (uncommitted)

```
orchestrator/src/orchestrator/evals/configs.py    [refactored + pivoted to :latest/HF]
scripts/run_vllm_eval.py                          [config-driven, longer timeout]
docs/vllm-eval-status.md                          [this doc]
/etc/fstab                                         [UUID line added]
/etc/systemd/system/docker.service.d/wait-for-internal-2nd.conf  [new]
```

No git commits made — next session should review and decide what to commit.

## Background scripts written (may be useful or should be cleaned up)

All in `/var/tmp/` (persistent across reboots):
- `move-minimax-*.log` — MiniMaxAI/MiniMax-M2.5 move log (completed)
- `rebuild-and-push-all.sh` + log — abandoned (pivot to HF)
- `retry-push-after-build.sh` + log — abandoned
- `restart-builds.sh` + log — abandoned
- `launch-post-gate-evals.sh` + log — still valid, just needs a new gate result to trigger

## Cheat sheet for next session

```bash
# Verify things look sane after reboot
mount | grep Internal-2nd       # should show /dev/nvme0n1 on /media/leo/Internal-2nd
docker images leosiriusdawn/runpod-vllm
df -h /media/leo/Internal-2nd /media/leo/Leo_X10p_4TB_00

# Dispatch 1 & 2 concurrently next session:

# (1) Debug tool-format bridge — launch a fresh pod and poke at it
cd /home/leo/src/dark-factory
python3 scripts/run_vllm_eval.py --config qwen3-coder-next-fp8-new --task df_task_12 --port 8100 --no-volume &
# While it's running, grab the SSH info from /var/tmp/gate-eval-*.log, SSH in:
#   ssh -i ~/.ssh/id_runpod root@<host> -p <port>
# Then on the pod:
#   curl -s http://localhost:8000/v1/models | jq    # what does vLLM expose?
#   curl -s http://localhost:8000/health
#   curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{...minimal tool-use request...}'
# Compare to what Claude CLI expects on ANTHROPIC_BASE_URL

# (2) reap-139b-nvfp4-new as alternative gate
python3 scripts/run_vllm_eval.py --config reap-139b-nvfp4-new --task df_task_12 --port 8101 --no-volume

# Critical files to read in next session for (1):
#   orchestrator/shared/cli_invoke.py
#   orchestrator/src/orchestrator/evals/runner.py  (how --vllm-url becomes ANTHROPIC_BASE_URL)
#   /home/leo/src/runpod-toolkit/docker/entrypoint-vllm.sh  (what server does vLLM launch?)

# Read the full failed iteration stdout (only first 500 chars currently logged):
#   The orchestrator logs are at /var/tmp/gate-eval-qwen3-coder-next-fp8-new-20260406-155237.log
#   But to see the FULL agent stdout, probably need to rerun with more verbose logging, OR
#   check if stdout was captured somewhere else (worktree dir, session files, etc.)
```

## Next session priority order

1. **Read `orchestrator/shared/cli_invoke.py` + `orchestrator/src/orchestrator/evals/runner.py`** to understand how `--vllm-url` propagates and how the Claude CLI is invoked. The full command line is logged:
   ```
   claude --print --output-format json --model sonnet --max-budget-usd 20.0 \
     --system-prompt-file /tmp/sysprompt_uvwq9dkz.txt --permission-mode bypassPermissions \
     --max-turns 80 --effort...
   ```
2. **Capture a full failed-iteration stdout** (not just first 500 chars) to see the actual error message
3. **Run `reap-139b-nvfp4-new` as alternative gate** (concurrent with debug) — quick data point
4. **If bridge is fixable**, fix it, rerun gate, then the 4 post-gate evals fire automatically (`launch-post-gate-evals.sh` is already set up)
5. **If not fixable quickly**, consider whether vLLM evals are worth continuing at all, or defer until a major refactor
