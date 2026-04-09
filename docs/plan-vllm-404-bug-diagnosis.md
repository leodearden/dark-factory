# Plan: diagnose the vLLM `/v1/messages?beta=true` 404 bug

**Read this first, then start.** This file is a self-contained prompt for
a new Claude Code session in `/home/leo/src/dark-factory`. It assumes you
have not read `docs/vllm-eval-status.md`, though that has broader
context.

## The bug

When the eval runner invokes the Claude CLI against a vLLM-hosted model,
**some runs** fail immediately on every iteration. The CLI emits an
error result and returns in ~1-3 seconds per attempt, runs 20 times
(`max_execute_iterations`), makes **zero changes to code**, and the
workflow exits `blocked`. Since the worktree is unchanged, verify runs
against the clean baseline and reports PASS on tests/lint/typecheck,
so the result JSON ends up `tests_pass=True lint_clean=True
typecheck_clean=True outcome=blocked lines_changed=0 cost_usd=0.0
iterations=20` — a false T/T/T that would get `composite_score=1.0`
under the ε scoring fix if not caught.

## Concrete evidence

The per-task eval log
`/var/tmp/dark-factory-evals/eval-reap-139b-nvfp4-new-reify_task_12-20260408-121925-b72f65.log`
shows the pattern clearly. For each of 20 iterations:

```
[shared.cli_invoke] Invoking claude agent: model=sonnet cwd=...
[shared.cli_invoke] Command: claude --print --output-format json --model sonnet
  --max-budget-usd 20.0 --system-prompt-file /tmp/sysprompt_*.txt
  --permission-mode bypassPermissions --max-turns 80 --effort...
[aiohttp.access] 127.0.0.1 ... "HEAD / HTTP/1.1" 404 131 "-" "Bun/1.3.11"
[aiohttp.access] 127.0.0.1 ... "POST /v1/messages?beta=true HTTP/1.1" 404 307 "-" "claude-cli/2.1.96"
[aiohttp.access] 127.0.0.1 ... "POST /v1/messages?beta=true HTTP/1.1" 404 307 "-" "claude-cli/2.1.96"
[shared.cli_invoke] Agent exit code: 1
[shared.cli_invoke] Agent stdout length: 869 bytes (full, returncode=1):
{"type":"result","subtype":"success","is_error":true,"duration_ms":1176,
 "num_turns":1,"result":"There's an issue with the selected model
 (lukealonso/MiniMax-M2.5-REAP-139B-A10B-NVFP4). It may not exist or you
 may not have access to it. Run --model to pick a different model.",
 "stop_reason":"stop_sequence", ...}
[orchestrator.workflow] Task reify_task_12 [implementer]: success=False cost=$0.00 turns=1 timeout=1200s
```

Two curiosities in that log block:

1. **The command says `--model sonnet` but the CLI error references the
   upstream HF model name.** The error message format is
   `"There's an issue with the selected model (X)"` where X is
   `lukealonso/MiniMax-M2.5-REAP-139B-A10B-NVFP4` — the vLLM upstream
   model name, not `sonnet`. How does the CLI know that name? Possible
   paths: (a) a server response substituted it in, (b) the CLI reads
   vLLM's `/v1/models` list endpoint and picks the first entry when
   `sonnet` isn't found, (c) ANTHROPIC_MODEL env var somewhere.

2. **The aiohttp bridge is logging 404 responses to `POST
   /v1/messages?beta=true`.** The bridge is `shared/src/shared/vllm_bridge.py`
   — an aiohttp proxy that sits between the Claude CLI and vLLM's
   `/v1/messages` endpoint. It's not crashing, just returning 404 with a
   307-byte body. That's the key signal: the bridge is running but not
   routing this request to the upstream.

## Scope of the bug

**2026-04-08 matrix retry #3** (12:27-12:31 BST) hit the bug on 3 entire
configs (every task × every iteration):

- `reap-139b-nvfp4-new` — 5/5 tasks garbage
- `minimax-m25-nvfp4-new` — 5/5 tasks garbage
- `qwen3-coder-next-fp8-new` — 5/5 tasks garbage

The nine false-T/T/T result files have been quarantined to
`orchestrator/src/orchestrator/evals/results/_quarantine_404_bug/`
(see that directory's README for details). Six other files from those
configs have `tests_pass=False` naturally (baseline wasn't clean for
`df_task_12`/`df_task_13`) and weren't contaminated.

**Intermittent occurrence on `minimax-m25-fp8-new`** (the same run did
NOT hit the bug): the 2026-04-08 13:53 BST retry of fp8 completed 5
tasks. Four worked correctly with real implementer work, real costs
($10-$22 per task), real line changes. But `reify_task_12` (result
`reify_task_12__minimax-m25-fp8-new__982eb336.json`) has `0 lines
changed, 139 turns, $29.30 cost, 20 iterations, 90 min wall clock` —
the same bug hit that one task and not the others.

**That single fp8 case is the most important investigation target**
because it rules out hypotheses tied to config or model family.

## What's known about the plumbing

- **Bridge source:** `shared/src/shared/vllm_bridge.py`. It's an aiohttp
  proxy started per invocation by `shared/src/shared/cli_invoke.py`.
  When `env_overrides` contains `ANTHROPIC_BASE_URL`, `_invoke_claude`
  spawns a bridge pointing at that URL, rewrites the subprocess env to
  point at the bridge's local URL, and tears the bridge down in a
  `finally` block. Started on `127.0.0.1:0` (OS-assigned port).
- **Entry points:** the eval runner at
  `orchestrator/src/orchestrator/evals/runner.py` sets `ANTHROPIC_BASE_URL`
  to the vLLM URL (e.g., `http://127.0.0.1:8200`) via `env_overrides`
  on the config. That flows through `_invoke` and eventually to the
  bridge startup.
- **vLLM 0.19 has a native Anthropic adapter** at `/v1/messages` — you
  can in principle point the Claude CLI directly at vLLM without a
  bridge, but the bridge is still wired in because it normalizes some
  residual format quirks (tool_use IDs, JSON-string inputs, stop_reason).
  Both the bridge and vLLM claim to handle `/v1/messages`.
- **Claude CLI 2.1.96** (from the log) sends
  `POST /v1/messages?beta=true`. The `?beta=true` query param is a
  newer CLI addition — unclear if older bridge/vLLM code handles the
  query-parameter variant of the route.

## Hypotheses, ranked

1. **Bridge route does not match `?beta=true` query string.** aiohttp
   route matching in `vllm_bridge.py` may use a literal path match that
   `/v1/messages?beta=true` doesn't match (even though the query string
   shouldn't affect routing at the aiohttp level, a naive route check
   on the full request line or a path regex that accidentally
   anchors could cause 404). **Fastest to test:** grep the bridge
   source for route registration.

2. **Bridge's upstream call proxies the full path including query
   params** and vLLM's native Anthropic adapter doesn't handle the
   `?beta=true` variant, returning 404. **Test:** curl the pod's vLLM
   endpoint directly from inside the pod to see if vLLM accepts or
   rejects `?beta=true`.

3. **`/v1/models` endpoint returning the wrong model name.** When the
   CLI can't find its requested model (`sonnet`), it may query
   `/v1/models` and get back the upstream HF name, which it then
   inserts into its error message. The 404 is the upstream rejection,
   and the error wording is a CLI-side formatting artifact. **Test:**
   check vLLM's `/v1/models` output and see if it lists a model that
   Claude CLI would accept as "sonnet."

4. **Bridge process dies or gets into a bad state after N invocations.**
   The intermittent fp8 case is suggestive: 4 tasks on the same pod
   worked, then 1 task hit the bug for 20 straight iterations. If the
   bridge were truly config-systemic, the 4 working tasks wouldn't
   exist. If it were per-task, the failure would start at iteration 1
   on the same task and not the next. **Test:** re-read
   `/var/tmp/dark-factory-evals/eval-minimax-m25-fp8-new-reify_task_12-20260408-121929-b7b5f2.log`
   and find the transition point where the bridge started 404-ing —
   did it happen after the tunnel survived for N successful calls?
   Was there a network blip? Was it always broken for this task?

5. **Port collision between bridge instances on the tunnel.** If
   multiple tasks run concurrently (concurrency=5 in the matrix), they
   each spawn their own bridge. Each bridge binds `127.0.0.1:0` so
   the OS assigns unique ports — no collision there. But if the per-pod
   tunnel uses a shared local port (e.g., 8200 in the matrix log), the
   CLI's `ANTHROPIC_BASE_URL` and the bridge URL could be confused.
   **Test:** verify the bridge URL rewriting logic in cli_invoke.py
   distinguishes bridge-local-url from upstream-vllm-url cleanly.

6. **Claude CLI 2.1.96 changed its model handling.** The `--model sonnet`
   arg might no longer map cleanly to an upstream model — earlier CLI
   versions fell back gracefully; 2.1.96 might fail harder. **Test:**
   pin to an older CLI version (2.1.92 is referenced in the doc) and
   rerun.

Prioritize 1, 2, and 4 — those are the most diagnosable with the least
cost.

## Investigation steps

**Do NOT spin up a new full matrix run.** Investigation should cost at
most one fresh pod × 30-60 min (~$1.50-$3) to isolate the bug.

### Phase 1 — read code (free, no pod needed)

1. Read `shared/src/shared/vllm_bridge.py` in full. Look for aiohttp
   route registration. What path patterns does it accept? Does it
   handle query parameters on `/v1/messages`?
2. Read `shared/src/shared/cli_invoke.py` around where it spawns the
   bridge and rewrites env. Exact line range: grep for
   `vllm_bridge` and `ANTHROPIC_BASE_URL` in that file.
3. Read `orchestrator/src/orchestrator/evals/runner.py` around the part
   that sets `env_overrides`. Confirm that the bridge-vs-vllm URL
   distinction is crisp.
4. Tail the full log from the intermittent fp8 case:
   `/var/tmp/dark-factory-evals/eval-minimax-m25-fp8-new-reify_task_12-20260408-121929-b7b5f2.log`
   Find the exact iteration where the 404s begin. Check if it's
   iteration 1 (systemic for this task) or a later iteration (state
   drift). Check timestamps — is there a gap between the last working
   call and the first failing call?

### Phase 2 — reproduce on a small pod (~$3 budget)

5. Spin up one pod with `minimax-m25-fp8-new` (the config that proves
   the bug is intermittent, so it works often and we can observe
   transitions):
   ```bash
   cd /home/leo/src/dark-factory
   python3 scripts/run_vllm_eval.py --config minimax-m25-fp8-new \
     --task df_task_12 --port 8200 --no-volume
   ```
6. While it runs, in another terminal, SSH into the pod (pod ID visible
   in `/var/tmp/dark-factory-evals/matrix-*.log` and SSH info from
   `runpodctl get pod <id>` or the launcher log).
7. From inside the pod, curl vLLM directly — bypass the bridge:
   ```bash
   curl -sS -X POST http://127.0.0.1:8000/v1/messages \
     -H 'content-type: application/json' -H 'x-api-key: test' \
     -d '{"model":"sonnet","max_tokens":50,"messages":[{"role":"user","content":"hi"}]}'
   ```
   Then with the `?beta=true` variant:
   ```bash
   curl -sS -X POST 'http://127.0.0.1:8000/v1/messages?beta=true' \
     -H 'content-type: application/json' -H 'x-api-key: test' \
     -d '{"model":"sonnet","max_tokens":50,"messages":[{"role":"user","content":"hi"}]}'
   ```
   Compare responses. Does vLLM handle `?beta=true`? If it 404s, the
   upstream is the problem, not the bridge.
8. From the dev box (not the pod), through the SSH tunnel, hit the
   bridge URL directly. You'll need to find the bridge port the
   current eval invocation is using — it's logged by `shared.cli_invoke`
   on bridge startup. Curl the same requests through the bridge and
   compare against direct-to-vLLM responses.
9. Also hit `/v1/models`:
   ```bash
   curl -sS http://127.0.0.1:8000/v1/models | jq .
   ```
   What model names does vLLM expose? Is `sonnet` one of them, or only
   the upstream HF name? The CLI's model-lookup failure mode is likely
   involved.

### Phase 3 — write the fix

The shape of the fix depends on which hypothesis wins:

- **Hypothesis 1 won (bridge route):** add a wildcard route in
  `vllm_bridge.py` that accepts `/v1/messages` regardless of query
  string. Test with a unit test that POSTs to `/v1/messages?beta=true`
  and confirms forwarding.
- **Hypothesis 2 won (upstream vLLM):** strip `?beta=true` from the
  outgoing request in the bridge, or add an explicit route pattern on
  the vLLM side. The bridge-side strip is easier.
- **Hypothesis 3 won (model name):** add a model-name rewrite in the
  bridge or in `cli_invoke.py` — the CLI thinks it's calling `sonnet`
  but the upstream expects the HF model name. The bridge can substitute
  it on the outgoing request.
- **Hypothesis 4 won (state drift):** investigate what state accumulates
  in the bridge process after N requests. Possible culprit: aiohttp
  connection pool, keep-alive handling, or a per-session state dict.
  Fix is to make the bridge stateless per request.
- **Hypothesis 5 won (port collision):** tighten the bridge URL vs
  upstream URL handling in `cli_invoke.py`.

## Critical files (one-line index)

- `shared/src/shared/vllm_bridge.py` — the aiohttp proxy (start here)
- `shared/src/shared/cli_invoke.py` — where the bridge is started and
  torn down per Claude CLI invocation; search for `vllm_bridge` and
  `ANTHROPIC_BASE_URL`
- `orchestrator/src/orchestrator/evals/runner.py` — where
  `env_overrides['ANTHROPIC_BASE_URL']` is set for vLLM configs
- `/var/tmp/dark-factory-evals/eval-minimax-m25-fp8-new-reify_task_12-20260408-121929-b7b5f2.log`
  — the intermittent-failure log (the most diagnostic evidence)
- `/var/tmp/dark-factory-evals/eval-reap-139b-nvfp4-new-reify_task_12-20260408-121925-b72f65.log`
  — a representative systemic-failure log
- `orchestrator/src/orchestrator/evals/results/_quarantine_404_bug/README.md`
  — the quarantine report with the full signature and file list
- `/home/leo/src/runpod-toolkit/docker/entrypoint-vllm.sh` — the pod
  entrypoint; it launches vLLM with specific flags that may affect
  which routes are exposed

## Non-goals

- **Do not rerun the full matrix.** The investigation should cost ≤ $3
  in pod time. A full matrix is $15-$30 and should only follow a fix.
- **Do not touch the quarantined results directory.** Those files stay
  quarantined until a matrix with the bug fixed produces fresh
  replacements.
- **Do not attempt to "rescore" the quarantined files.** The data
  underneath is bogus; no scoring function can rescue them.
- **Do not investigate from the dev box alone.** You need SSH into a
  live pod to bypass the SSH tunnel and test vLLM directly. This is
  critical for distinguishing bridge-layer from vLLM-layer failures.

## Acceptance criteria

1. **Reproducible minimal case.** Phase 2 curl commands that reliably
   trigger the 404, and the equivalent that succeeds, committed or
   pasted into the session output.
2. **Hypothesis identified with evidence.** Which of 1-6 is the root
   cause, and why the others are ruled out.
3. **Fix proposed with estimated blast radius.** File, function, diff
   sketch. Don't ship the fix in the same session as the diagnosis —
   get the diagnosis right first, then write a follow-up session
   prompt for the fix.
4. **Regression test sketched.** How would we catch this bug next
   time? A unit test on the bridge that POSTs `/v1/messages?beta=true`
   and asserts forwarding? An integration test that curls through the
   bridge during eval startup?
5. **Follow-up plan prompt written.** If the fix is non-trivial, write
   a new `docs/plan-vllm-404-bug-fix.md` for the implementation
   session.

## Budget

- Code reading: free
- Pod time: ≤ $3 (one fresh pod × ≤ 60 min)
- Total session estimate: 30-90 min of dev time + ≤ $3 pod spend

If the investigation exceeds 2 hours or $5 in pod time without
converging on a hypothesis, stop and write up what you've learned —
don't throw money at a dead end.

## Related

- `docs/vllm-eval-status.md` — broader context on the 2026-04-08 session
- `docs/plan-scoring-and-judge.md` — the ε scoring fix this bug
  contaminates if garbage data isn't quarantined first
- `docs/plan-runpod-retry.md` — unrelated retry work for runpod-toolkit
