# vLLM Local Model Evaluation — Status Report

**Last updated:** 2026-04-06 06:40 BST
**Plan:** `~/.claude/plans/zany-dancing-yeti.md`

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

**The pipeline works end-to-end except for two vLLM-level blockers:**
1. **Devstral** has a `Tekkenizer.encode() add_special_tokens` bug in vLLM 0.18.1 / mistral_common 1.10.0
2. **Large models** (Qwen3-Coder-Next 149GB, REAP-139B 131GB) crash inside the container before HF download finishes — root cause unknown, needs container/system log inspection

**Next-session task list (in order):**
1. Run option A in background: derive a new base image with `pip install -U mistral_common vllm`, push as `leosiriusdawn/runpod-vllm:upgraded`, then test devstral with it
2. Start a Qwen3-Coder-Next pod with default settings so the user can SSH in and inspect logs (`journalctl`, `dmesg`, `/proc/1/fd/{1,2}`, `nvidia-smi`) to figure out why the container crashes
3. Once one model works end-to-end, run all 5 evals in parallel on `df_task_12`

## Infrastructure State

### dark-factory
- **Branch:** `main`
- **Commit `53486e8023`** (2026-04-05): env_overrides scoping (only implementer/debugger) + 4 task_id mismatches fixed
- **Eval script:** `scripts/run_vllm_eval.py` (saved from `/tmp/run_eval.py`)
- **No uncommitted changes**

### runpod-toolkit
- **Branch:** `main`
- **Commit `d68d219`** (2026-04-06): TOKENIZER_MODE env var, Dockerfile.local-model, vllm_pod.py arg fixes
- **No uncommitted changes**

### Local models — `/media/leo/Internal-2nd/leo/models/`
All 5 models downloaded from RunPod volume:
| Model | Size | HF cache dir |
|-------|------|--------------|
| Devstral-Small-2505 | 88 GB | `models--mistralai--Devstral-Small-2505/` |
| Qwen3-Coder-Next | 149 GB | `models--Qwen--Qwen3-Coder-Next/` |
| REAP-139B | 131 GB | `models--cerebras--MiniMax-M2.5-REAP-139B-A10B/` |
| REAP-172B | 162 GB | `models--cerebras--MiniMax-M2.5-REAP-172B-A10B/` |
| MiniMax-M2.5 | 215 GB | `models--MiniMaxAI--MiniMax-M2.5/` |

### Docker images (in new data-root `/media/leo/Internal-2nd/leo/docker-data/`)
| Tag | Built | Pushed | Size |
|-----|-------|--------|------|
| `leosiriusdawn/runpod-vllm:latest` | ✅ | ✅ | 22.4 GB (modified entrypoint w/ TOKENIZER_MODE) |
| `leosiriusdawn/runpod-vllm:devstral-small` | ✅ | ✅ | 117 GB |
| `leosiriusdawn/runpod-vllm:qwen3-coder-next` | ✅ | ✅ | ~170 GB |
| `leosiriusdawn/runpod-vllm:reap-139b` | ✅ | ✅ | ~155 GB |
| `leosiriusdawn/runpod-vllm:reap-172b` | ✅ | ⏳ in progress | ~185 GB |
| `leosiriusdawn/runpod-vllm:minimax-m25` | ⏳ pending | ⏳ pending | (will start when build monitor sees minimax download done) |

**Note:** Baked Docker images turned out to be **impractical for RunPod evals** — RunPod cannot finish pulling 100GB+ images (container `uptime=0s` after 20+ min). Use base image + HF download approach instead. The baked images may still be useful for other clouds or local Docker deployments.

### Build monitor
- **Script:** `/tmp/build-as-complete.sh` running as background bash process
- **Output:** `/tmp/build-as-complete.out`
- **PID:** check `pgrep -f build-as-complete`
- **Behavior:** polls `/tmp/rsync-*.log` every 30s, builds + pushes Docker images as each download completes
- Will pick up minimax-m25 once that download finishes (see below)

### MiniMax download status
- **DONE** as of 2026-04-06 ~05:30 — but check `grep "total size is" /tmp/rsync-minimax.log`
- The build monitor will detect completion on its next 30s poll and start the build automatically

### RunPod
- **Credit remaining:** ~$55 of $65 (spent ~$10 total: ~$0.50 transfer pod over 9h, ~$9 on failed GPU pod attempts)
- **Currently running pods:** none (transfer pod terminated 2026-04-06 06:37)
- **API key:** `rpa_VLRVNJ8HB5CH7MQZL9WW2XPQBQO18V3PMA1H1BSM11niy2` (in `~/.secrets/runpod.env`)
- **SSH key:** `~/.ssh/id_runpod` (passphrase-free ed25519)
- **Volume `obxma9bf1b`** (US-NC-1, 900GB) still has all the models — keep it as backup for now

## What Works (validated end-to-end)

### The eval pipeline approach: base image + HF download
**Don't use baked Docker images on RunPod.** Use the 22GB base image and let vLLM download the model from HuggingFace at startup. This is faster and reliable for at least Devstral.

For Devstral specifically:
- Pod creation → SSH available: ~30 seconds (sometimes 2-3 min if image cache is cold)
- vLLM model download (88GB from HF) + load: ~4 minutes
- Total time to healthy vLLM endpoint: **~5 minutes**

### Eval script: `scripts/run_vllm_eval.py`
Creates GPU pod → waits for SSH → SSH tunnel → waits for vLLM health → runs eval → **always terminates pod in finally block**.

```bash
python3 scripts/run_vllm_eval.py --model devstral-small --task df_task_12 --port 8100 --no-volume
```

Key features:
- `--no-volume` (default): use base image + HF download (works for Devstral)
- `--use-volume`: mount volume `obxma9bf1b` in US-NC-1 (blocked: no GPU stock there as of 2026-04-05)
- GPU type fallback list (tries 96GB Blackwell variants then 80GB H100/A100)
- Pod always terminated on success, failure, or exception
- Loads `CLAUDE_OAUTH_TOKEN_G` from `.env` automatically
- Eval cwd is `/home/leo/src/dark-factory/orchestrator/` (must be — see "Stale eval-worktree shadowing" below)

### GPU type
**Use:** `NVIDIA RTX PRO 6000 Blackwell Server Edition` (~$1.89/hr secure)
- Available consistently when datacenter is NOT specified
- 96GB VRAM
- The string `NVIDIA RTX PRO 6000` (without "Blackwell Server Edition") is **invalid** — caused first failed attempt

**Fallbacks** (if Blackwell unavailable):
- `NVIDIA RTX PRO 6000 Blackwell Workstation Edition`
- `NVIDIA H100 80GB HBM3` (~$2.69/hr)
- `NVIDIA A100-SXM4-80GB` (~$1.49/hr)

### Datacenter
**Don't specify a datacenter** — let RunPod auto-pick. Specifying EUR-IS-1 or US-NC-1 has resulted in "no instances available" errors.

The volume `obxma9bf1b` is in US-NC-1, but US-NC-1 has had **no 80GB+ GPU stock** during all attempts. So volume-mount approach is currently blocked.

## Blockers

### Blocker 1: Devstral Tekkenizer bug
**Symptom:** vLLM loads and serves health endpoint, but every Claude Code agent call returns:
```
API Error: 500 {"type":"error","error":{"type":"internal_error",
  "message":"Tekkenizer.encode() got an unexpected keyword argument 'add_special_tokens'"}}
```
Each call takes ~3 min and the orchestrator marks each as `success=True` (because it got *a* response), then moves on. The eval will run forever without producing real code changes.

**Root cause:** vLLM 0.18.1 + mistral_common 1.10.0 ship with a Tekkenizer that doesn't accept `add_special_tokens` kwarg. vLLM passes this kwarg unconditionally.

**Failed fix:** Setting `TOKENIZER_MODE=slow` via the new entrypoint env var **crashes vLLM at startup**. The flag is added but vLLM 0.18.1 doesn't gracefully handle "slow" mode for Mistral models.

**Untested fixes (for next session):**
1. **Option A (recommended)**: Build a derived image:
   ```dockerfile
   FROM leosiriusdawn/runpod-vllm:latest
   RUN pip install -U mistral_common vllm
   ```
   Push as `leosiriusdawn/runpod-vllm:upgraded`. Test with Devstral.
2. Try `--tokenizer hf-internal-testing/llama-tokenizer` to override with a non-Mistral tokenizer (probably won't work — model has its own tokenizer)
3. Monkey-patch the Tekkenizer in a startup script

### Blocker 2: Large models crash before download completes
**Models affected:** Qwen3-Coder-Next (149GB), REAP-139B (131GB). Devstral (88GB) is unaffected.

**Symptom:**
1. Pod created, RUNNING
2. SSH becomes available within 2-4 min
3. SSH works briefly — diagnostics show GPU=0MiB used, HF cache=15MB or less, disk almost empty
4. ~5-10 min after creation, SSH stops responding (`Connection refused`)
5. Container's `uptime=0s` (PID 1 exited)
6. Eval script's vLLM health timeout eventually fires and pod is terminated

**Hypotheses (untested):**
- vLLM crashes during pre-flight model architecture check before downloading
- KV cache pre-allocation fails with default `MAX_MODEL_LEN=131072` on these large models
- Container OOM (CPU memory, not GPU) during HF download for very large models — RunPod has limited container memory by default
- Custom REAP/MiniMax model architecture not fully supported by vLLM 0.18.1

**What we tried that didn't help:**
- `MAX_MODEL_LEN=65536` reduced context — still crashed
- `VLLM_LOGGING_LEVEL=DEBUG` — **made it crash faster** (don't use)
- Container disk 350GB (way more than 2.5× model size) — still crashed
- `DTYPE=float8` env var — needs more testing in clean environment

**What needs investigation:**
- **Get actual container/system logs while it's crashing.** SSH in immediately after pod creation, run `tail -f /proc/1/fd/{1,2}` and `journalctl -fk`, watch what happens
- Maybe the new entrypoint with TOKENIZER_MODE block has a bug? It looked fine but worth re-checking
- Try setting `container_disk_gb=600` and `min_memory_gb=128` to rule out memory pressure

### Blocker 3 (resolved): Stale eval-worktree shadowing
The `orchestrator eval` subcommand was missing from `--help` because Python was importing `orchestrator.cli` from a stale `.eval-worktrees/df_task_13/run-25a0871e/` directory.

**Fix:** `cd orchestrator/` before running `uv run orchestrator eval ...`. The eval script does this automatically.

## Critical learnings (don't repeat the mistakes)

1. **Container crashes silently** — RunPod's pod-status API shows `RUNNING` even when the container's PID 1 has died. Check `runtime.uptimeInSeconds` in `runpod.get_pods()` — `0s` after several minutes means the container is dead.

2. **Always terminate pods in `finally` blocks** — the eval script does this. Failed attempts so far have been cleaned up correctly. Check `runpod.get_pods()` after any error to verify nothing is leaking.

3. **`pkill -f run_eval.py` is unreliable** — the SIGTERM doesn't always trigger the script's cleanup logic, leaving orphaned pods. Better to call `client.terminate_pod(pod_id)` directly.

4. **Don't specify datacenter** unless required by volume — RunPod has limited stock per DC.

5. **Don't use `VLLM_LOGGING_LEVEL=DEBUG`** — confirmed via A/B test to cause crashes.

6. **`TOKENIZER_MODE=slow` crashes vLLM 0.18.1** — added the env var hook in entrypoint for future use, but currently broken.

7. **Baked Docker images don't work on RunPod** — pulling 100GB+ images never finishes. Use base image + HF download.

## Where things are

| Thing | Path |
|-------|------|
| Eval script | `/home/leo/src/dark-factory/scripts/run_vllm_eval.py` |
| Eval script (working copy) | `/tmp/run_eval.py` |
| Build monitor script | `/tmp/build-as-complete.sh` |
| Build monitor output log | `/tmp/build-as-complete.out` |
| Rsync logs | `/tmp/rsync-{devstral,qwen3,reap139,reap172,minimax}.log` |
| Docker data-root | `/media/leo/Internal-2nd/leo/docker-data/` |
| Local model cache | `/media/leo/Internal-2nd/leo/models/` |
| RunPod toolkit | `/home/leo/src/runpod-toolkit/` |
| Modified entrypoint | `/home/leo/src/runpod-toolkit/docker/entrypoint-vllm.sh` |
| Local-model Dockerfile | `/home/leo/src/runpod-toolkit/docker/Dockerfile.local-model` |
| Eval configs | `/home/leo/src/dark-factory/orchestrator/src/orchestrator/evals/configs.py` |
| Eval task files | `/home/leo/src/dark-factory/orchestrator/src/orchestrator/evals/tasks/df_task_*.json` |
| Memory: blockers note | `/home/leo/.claude/projects/-home-leo-src-dark-factory/memory/project_vllm_eval_blockers.md` |

## Eval configs (from `orchestrator/src/orchestrator/evals/configs.py`)

```python
VLLM_EVAL_CONFIGS = [
    _vllm_config('minimax-m25-fp8', 'MiniMaxAI/MiniMax-M2.5'),
    _vllm_config('qwen3-coder-next-fp8', 'Qwen/Qwen3-Coder-Next'),
    _vllm_config('reap-139b-nvfp4', 'cerebras/MiniMax-M2.5-REAP-139B-A10B'),
    _vllm_config('reap-172b-nvfp4', 'cerebras/MiniMax-M2.5-REAP-172B-A10B'),
    _vllm_config('qwen3-coder-30b-q4', 'Qwen/Qwen3-Coder-30B-A3B-Instruct'),
    _vllm_config('devstral-small-2505-q6', 'mistralai/Devstral-Small-2505'),
    _vllm_config('qwen25-coder-32b-q4', 'Qwen/Qwen2.5-Coder-32B-Instruct'),
]
```

## Eval task files

All 5 task files have embedded plans (task_id mismatches fixed in `53486e8023`):
- `df_task_12.json` — Bug fix: verify.py shell issue (3 steps, "small" complexity) — **use this for canary**
- `df_task_13.json` — Bug fix: empty diff in reviewers (8 steps)
- `df_task_18.json` — Larger feature task ("medium")
- `reify_task_12.json` — Reify project task ("high")
- `reify_task_27.json` — Reify project task ("high")

## Account for Evals

Use account G (`CLAUDE_OAUTH_TOKEN_G` in `.env`) for architect/reviewer Claude calls. The eval script loads `.env` automatically.

## Concrete next-session commands

```bash
# 1. Verify state
cd /home/leo/src/dark-factory
git log --oneline -3            # should show 53486e8023
cd /home/leo/src/runpod-toolkit
git log --oneline -3            # should show d68d219
python3 -c "
import sys; sys.path.insert(0, '/home/leo/src/runpod-toolkit')
import runpod
runpod.api_key = 'rpa_VLRVNJ8HB5CH7MQZL9WW2XPQBQO18V3PMA1H1BSM11niy2'
print([p['id'] for p in runpod.get_pods()])  # should be empty
"
tail -5 /tmp/build-as-complete.out  # check if minimax build/push has started

# 2. Option A: build upgraded image (do this in background)
mkdir -p /tmp/upgraded-image && cd /tmp/upgraded-image
cat > Dockerfile <<EOF
FROM leosiriusdawn/runpod-vllm:latest
RUN pip install --no-cache-dir -U mistral_common vllm
EOF
docker build -t leosiriusdawn/runpod-vllm:upgraded . &
# After build: docker push leosiriusdawn/runpod-vllm:upgraded
# Then update scripts/run_vllm_eval.py to use the :upgraded tag
# Test: python3 scripts/run_vllm_eval.py --model devstral-small --task df_task_12 --no-volume

# 3. Start Qwen3-Coder-Next pod for the user to inspect
# Use base image (not baked) so SSH comes up fast
# Use defaults — no DTYPE, no MAX_MODEL_LEN, no debug logging
python3 scripts/run_vllm_eval.py --model qwen3-coder-next --task df_task_12 --port 8101 --no-volume &
# As soon as SSH is available, the user will SSH in to inspect logs:
#   ssh -i ~/.ssh/id_runpod root@<ip> -p <port>
#   tail -f /proc/1/fd/{1,2}    # vllm stdout/stderr
#   journalctl -fk              # kernel ring buffer
#   dmesg -wT                   # OOM kills, etc
#   nvidia-smi -l 5             # GPU memory pressure
#   df -h                       # disk usage
```

## Account costs (so far)

| Item | Cost |
|------|------|
| CPU transfer pod (~9h @ $0.13/hr) | ~$1.20 |
| Failed GPU eval attempts (~6 pods, ~30 min total) | ~$2-3 |
| Successful Devstral load + Tekkenizer-error eval loop | ~$3 |
| Other GPU starts that crashed | ~$2-3 |
| **Total spent** | **~$10** |
| **Remaining** | **~$55** |

$55 is plenty for 5 successful evals (1 each model, ~$1-2 each on RTX PRO 6000 Blackwell).
