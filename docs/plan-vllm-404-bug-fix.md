# Plan: fix the vLLM `/v1/messages` 404 bug — SSH tunnel port collision

**Start here.** Self-contained prompt for a new Claude Code session in
`/home/leo/src/dark-factory`. Read `docs/plan-vllm-404-bug-diagnosis.md`
first only if you want the original framing — the diagnosis below
supersedes it.

## TL;DR diagnosis

The 2026-04-08 12:19-12:31 BST matrix retry #3 launched **four**
`run_vllm_eval.py` processes in parallel, each with `--port 8200` (the
default). Only one SSH `-L 127.0.0.1:8200:127.0.0.1:8000` bind can
succeed; the launcher spawns ssh **without `-o ExitOnForwardFailure=yes`**,
so the three losers keep running with **no forward**. Each loser's
`wait_for_vllm()` then hits the winner's tunnel, gets HTTP 200 from
`/health`, and reports `"vLLM healthy on port 8200"` — a false positive.

Every eval subprocess then bridges requests to `127.0.0.1:8200`, which
routes to the **winner's** pod. The Claude CLI substitutes
`ANTHROPIC_DEFAULT_SONNET_MODEL` into the request (each config has its
own HF model name), the winner's pod has a different model loaded, and
vLLM returns **404 "model not found"**. The bridge forwards the 404
verbatim. The CLI emits `is_error=true` with
`"There's an issue with the selected model (X). It may not exist..."`,
exits in ~1-3s, produces 0-cost / 0-turn / 0-line-changed results, and
the workflow loops through 20 iterations with zero agent work. Verify
runs against the clean baseline and reports PASS/PASS/PASS — the false
T/T/T the original plan flagged.

## Evidence summary

**Matrix log correlation** — four runs tried port 8200 in the same window:

| Matrix run                                       | ssh -L started | "vLLM healthy" |
|--------------------------------------------------|----------------|----------------|
| `qwen3-coder-next-fp8-new-20260408-104340`       | 10:56:06       | 12:19:31       |
| `minimax-m25-fp8-new-20260408-121522`            | 12:17:33       | 12:19:28       |
| `reap-139b-nvfp4-new-20260408-120905`            | 12:17:59       | 12:19:23       |
| `minimax-m25-nvfp4-new-20260408-120723`          | 12:23:30       | 12:23:33       |

All four report "healthy on port 8200" within seconds of each other
(12:19:23-12:23:33) because they are all probing the same physical
tunnel. The older working example
`matrix-reap-139b-nvfp4-new-20260407-221202.log` used `--port 8201` and
ran cleanly; similarly `22:51:48` fp8 used `--port 8204`.

**Per-eval log correlation** — three of the four failed identically:

| Config                  | POST `/v1/messages` 200 | 404 | 404 body size |
|-------------------------|-------------------------|-----|---------------|
| `reap-139b-nvfp4-new`   | 0                       | 40  | 307           |
| `qwen3-coder-next-fp8-new` | 0                    | 40  | 288           |
| `minimax-m25-nvfp4-new` | 0                       | 40  | 288           |
| `minimax-m25-fp8-new`   | 392                     | 0   | n/a           |

20 iterations × 2 requests per iteration = 40 requests. The fp8 case is
the **winner** of the race. Its requests go to its own pod and
succeed — that's why the original plan mislabeled it as the "intermittent
404 case." It's not.

**Model-name length ↔ 404 body size**: `lukealonso/MiniMax-M2.5-REAP-139B-A10B-NVFP4`
is 44 chars, vs `Qwen/Qwen3-Coder-Next-FP8` (25) and
`nvidia/MiniMax-M2.5-NVFP4` (25). Body-size delta = 307 − 288 = 19 =
44 − 25. The 404 body inlines the requested model name, proving the
upstream sees the loser's model name (not the winner's), meaning each
eval DID send its own configured model name and vLLM rejected it.

**The fp8 case is a red herring for this bug.** The fp8 log shows:
- Iter 1 (12:19:53 → 12:39:39): 200s throughout; exit reason
  `error_max_budget_usd` ($20.18, 82 turns). **Natural budget cap.**
- Iter 2 (12:39:50 → 12:56:14): 200s then `ConnectionRefusedError:
  Cannot connect to host 127.0.0.1:8200`. **SSH tunnel died** mid-run.
- Iter 3+: fast-cycle failures from the dead tunnel.

The fp8 "intermittent failure" the original plan flagged is a separate
issue (likely SSH keepalive timeout on a stressed pod), **not** the
404 bug. It is still worth fixing but not as part of this work.

## Code sites

`scripts/run_vllm_eval.py:525-556` — the broken ssh spawn:

```python
log(f"Starting SSH tunnel (127.0.0.1:{args.port} → pod:8000)")
tunnel_proc = subprocess.Popen(
    [
        "ssh",
        "-N",
        "-L", f"127.0.0.1:{args.port}:127.0.0.1:8000",
        "-i", SSH_KEY,
        "-o", "StrictHostKeyChecking=no",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
        # MISSING: -o ExitOnForwardFailure=yes
        f"root@{pod.ssh_host}",
        "-p", str(pod.ssh_port),
    ],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.PIPE,
)
time.sleep(3)
```

`scripts/run_vllm_eval.py:984` — the default port:

```python
p.add_argument("--port", type=int, default=8200)
```

`scripts/run_vllm_eval.py:403-416` — `wait_for_vllm()`, which cannot
distinguish "my tunnel" from "someone else's tunnel":

```python
url = f"http://127.0.0.1:{port}/health"
# ... urllib.request.urlopen → 200 means "something is listening"
```

## Fixes (three layers, all belt-and-braces)

### Layer 1 — Make ssh fail loudly on bind collision (required)

Add `-o ExitOnForwardFailure=yes` to the `ssh -L` args.
Behavior change: if `-L 127.0.0.1:<port>:...` fails to bind, ssh exits
with code 255 instead of running without a forward. The launcher's
existing error path (`vllm_healthy() returns False` → `PodBringupFailed`)
then kicks in, and the pod is cleaned up instead of routing requests to
whatever else is on that port. **Blast radius**: 1 line in
`run_vllm_eval.py`. No test file needs to change, but a regression test
should be added (see below).

### Layer 2 — Auto-pick a free port (recommended)

Replace `--port` default 8200 with a port-finder:

```python
# When --port is not explicitly set, bind to 127.0.0.1:0 to ask the OS
# for a free port, read it back, close the probe socket, and use that
# port for the ssh tunnel. Race window is small because the subsequent
# ssh -L binds within ~10 ms and ExitOnForwardFailure=yes (Layer 1) is
# the safety net.
import socket
def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
```

Explicit `--port N` from the CLI still honors the user's choice. This
removes the need for humans to coordinate parallel matrix runs (which
is what went wrong in retry #3).

**Blast radius**: ~15 lines. Touch sites: `_pick_free_port()` helper,
`bring_up_pod()` consumes `args.port or _pick_free_port()`, `--port`
default → `None`, tests in `scripts/test_run_vllm_eval.py` (if that
exists — check first).

### Layer 3 — Health probe with a model-list sanity check (nice-to-have)

Strengthen `wait_for_vllm()` to also hit `GET /v1/models` and confirm
the expected `MODEL_NAME` appears in the list. If the health probe hits
someone else's tunnel, the model list will not match the config's
`hf_model`. This catches even the ExitOnForwardFailure-bypasses-us case
(e.g. some future change accidentally strips the option).

**Blast radius**: ~20 lines. Adds ~1 extra HTTP call per health probe
cycle, trivial cost. Only useful if we're paranoid — Layers 1 and 2 are
already sufficient.

## Regression tests

### Unit: bind collision → subprocess exit

Under `scripts/test_run_vllm_eval.py` (create if missing), test that
when port 8200 is already bound by a listening socket in the test
fixture, the launcher's ssh command (stubbed with a fake ssh that
respects `ExitOnForwardFailure=yes`) exits non-zero, `vllm_healthy()`
returns False, and `bring_up_pod()` raises `PodBringupFailed`.

### Unit: `_pick_free_port()` returns an unused port

Trivial — bind two sockets and assert the ports differ.

### Integration (manual, optional): two concurrent local fakes

Spin up two local aiohttp mock servers on ports `0` (OS-assigned), start
two `run_vllm_eval.py` subprocesses with `--port 0` (requests auto),
confirm they route to **different** mock servers. This is the end-to-end
regression guard against the 2026-04-08 retry #3 failure mode.

## Acceptance criteria

1. `-o ExitOnForwardFailure=yes` added to `scripts/run_vllm_eval.py:540`
   ssh args, with a comment citing this plan / the 404 bug.
2. `--port` default switched to auto-pick via `_pick_free_port()`; the
   `wait_for_vllm` / `vllm_healthy` call sites take the chosen port
   through `PodHandle.local_port` (already plumbed).
3. At least one regression test in `scripts/test_run_vllm_eval.py` that
   fails against pre-fix code and passes after.
4. A smoke matrix: launch **two** `run_vllm_eval.py --config X` in
   parallel against trivial configs (or `--plan-only` equivalents) and
   confirm they pick distinct ports from the launcher log output. No
   pod spin needed for this check.
5. Do **not** touch `orchestrator/src/orchestrator/evals/results/_quarantine_404_bug/`
   — the quarantined results stay quarantined until a fresh matrix with
   the fix produces replacements.

## Non-goals (punted to follow-ups)

- The **fp8 mid-run tunnel death** (`ConnectionRefusedError` at 12:56:13
  in `eval-minimax-m25-fp8-new-reify_task_12-20260408-121929-b7b5f2.log`).
  Different bug — likely SSH keepalive killing the connection when a
  CPU-bound or memory-pressured pod stops responding within
  `ServerAliveCountMax × ServerAliveInterval = 90 s`. Deserves its own
  plan once Layer 1+2 land.
- The **false T/T/T scoring** that would have propagated garbage results
  as `composite_score=1.0`. That's already covered by the quarantine in
  `_quarantine_404_bug/` and the ε scoring fix (`plan-scoring-and-judge.md`).
- The **zero-cost instant-exit heuristic** in `cli_invoke.py:253-286`
  that was flagged by a reviewer trial as potentially mis-firing on
  vLLM failures. Orthogonal — it kicks in only when a `usage_gate` is
  present, and vLLM configs may or may not use one. Evaluate separately.

## Budget

- Code edits: ~20 lines across `run_vllm_eval.py` and a test file.
- Pod time: $0 — the fix can be tested locally with stubbed ssh and
  port-binding fakes.
- Session estimate: 30-60 min.
