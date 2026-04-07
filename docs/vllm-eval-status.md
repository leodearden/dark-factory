# vLLM Local Model Evaluation — Status Report

**Last updated:** 2026-04-07 09:30 BST
**Status:** First successful vLLM-hosted eval achieved (reap-139b-nvfp4-new with minimax_m2 parser, 2026-04-06 19:35).

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
