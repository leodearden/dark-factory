# Plan: RunPod client retry on transient network errors

**Read this first, then start.** This file is a self-contained prompt for
a new Claude Code session working in `/home/leo/src/runpod-toolkit/` (a
**separate repo** from dark-factory). Scope is tight — the entire change
is three decorator applications plus one small refactor per method.

## Context: the failures this fixes

The RunPod GraphQL API (`api.runpod.io`) and the `runpod` Python SDK it
backs occasionally throw transient network errors during otherwise-healthy
operations. Observed in a dark-factory eval run on 2026-04-08:

- **`get_pod` hit `ConnectionResetError`** during pod polling → launcher
  exception path tore down a perfectly good pod. Cost: wasted cold start.
- **`terminate_pod` failed on DNS resolution:**
  `"Failed to resolve 'api.runpod.io' ([Errno -3] Temporary failure in name
  resolution)"`. The launcher hit a vLLM health timeout, tried to
  terminate the pod, and on the DNS hiccup printed
  `"MANUAL CLEANUP NEEDED: runpodctl remove pod r64fywcfjemmzx"`. The
  pod leaked. At ~$1.69/hr this is the worst-case scenario.

Both failures are clearly transient — retry with backoff almost certainly
succeeds on the second attempt. No caller currently retries.

## Good news: `backoff` v2.2.1 is already a dependency

Check `/home/leo/src/runpod-toolkit/pyproject.toml` line 8 — `backoff` is
listed as a dependency but **grep shows zero usages across the repo**.
It's already installed, ready to use. No dep changes, no `uv sync` needed.

The entire retry implementation is three decorator applications plus a
small ordering refactor inside each target method. Do not write a custom
retry helper.

## Targets — exactly three methods

**File:** `/home/leo/src/runpod-toolkit/runpod_toolkit/compute/runpod_client.py`

- **`get_pod(pod_id)` at line 645** — called in a tight polling loop from
  `wait_for_pod()` at line 709. Retry on `get_pod` means `wait_for_pod`
  gains resilience for free.
- **`list_pods()` at line 670** — less hot but same exception surface.
- **`terminate_pod(pod_id)` at line 687** — **CRITICAL for cost safety.**
  More retries here (5 vs 3) because a leaked pod costs real money.

### Methods NOT in scope — do not touch them

- **`create_pod()` (line 386) and `create_cpu_pod()` (line 518)** — higher-
  level retry logic already exists in
  `runpod_toolkit/compute/pod_manager.py` at lines 168-222 and 277-331.
  Don't add retries here; you'd double-retry.
- **`get_gpu_availability()` (line 242), `get_datacenters()` (line 343),
  `create_volume()` (line 776), `list_volumes()` (line 870),
  `delete_volume()` (line 919)** — informational or irreversible. These
  use `requests.post` directly, not the SDK. Not in this plan's scope.

## The fix

### Top of `runpod_client.py` — imports and constants

Add near the existing imports (the file already imports `logging`, `time`,
etc.):

```python
import socket
import backoff
import requests.exceptions

_TRANSIENT = (
    requests.exceptions.ConnectionError,
    socket.gaierror,
    ConnectionResetError,
    TimeoutError,
)
```

The `_TRANSIENT` tuple lists exactly the classes that should be retried.
Do **not** include `requests.exceptions.RequestException` — it's the
parent of all requests exceptions including auth and malformed-request
errors, which should NOT be retried. Explicit list only.

### Refactor each target method

Each target method currently wraps the SDK call in
`try/except Exception: ... raise RunPodError(...)`. That wrapping catches
transient exceptions and converts them to `RunPodError` **before** the
backoff decorator at the function boundary can see them. Backoff will
never retry because the exception class it sees is `RunPodError`, not
`ConnectionResetError`.

**Fix:** add an explicit `except _TRANSIENT: raise` clause **before** the
generic `except Exception` clause. Transient exceptions propagate raw;
everything else still gets wrapped as `RunPodError`.

#### get_pod

```python
@backoff.on_exception(backoff.expo, _TRANSIENT, max_tries=3, logger=logger)
def get_pod(self, pod_id: str) -> PodInfo:
    """Get pod information.

    Retries on transient network errors (ConnectionError, DNS failures,
    ConnectionReset, TimeoutError) up to 3 times with exponential backoff.
    """
    try:
        response = self._runpod.get_pod(pod_id)
        if not response:
            raise RunPodError(f"Pod not found: {pod_id}")
        return PodInfo.from_api(response)
    except _TRANSIENT:
        raise  # let backoff retry
    except Exception as e:
        if isinstance(e, RunPodError):
            raise
        raise RunPodError(f"Failed to get pod {pod_id}: {e}") from e
```

#### list_pods

```python
@backoff.on_exception(backoff.expo, _TRANSIENT, max_tries=3, logger=logger)
def list_pods(self) -> List[PodInfo]:
    """List all pods.

    Retries on transient network errors up to 3 times with exponential
    backoff.
    """
    try:
        response = self._runpod.get_pods()
        if not response:
            return []
        return [PodInfo.from_api(pod) for pod in response]
    except _TRANSIENT:
        raise  # let backoff retry
    except Exception as e:
        raise RunPodError(f"Failed to list pods: {e}") from e
```

#### terminate_pod

```python
@backoff.on_exception(backoff.expo, _TRANSIENT, max_tries=5, logger=logger)
def terminate_pod(self, pod_id: str) -> bool:
    """Terminate a pod.

    Retries on transient network errors up to 5 times (more than read
    operations) because a leaked pod costs real money (~$1.69/hr). Uses
    exponential backoff.
    """
    try:
        logger.info(f"Terminating pod {pod_id}")
        response = self._runpod.terminate_pod(pod_id)
        # Response is typically empty on success
        return True
    except _TRANSIENT:
        raise  # let backoff retry
    except Exception as e:
        raise RunPodError(f"Failed to terminate pod {pod_id}: {e}") from e
```

### Logger wiring

`runpod_client.py:36` already has
`logger = logging.getLogger(__name__)`. Passing `logger=logger` to each
decorator routes retry warnings through the existing logger so they
appear in the same log stream as the rest of the file's output.

### Why this works — SDK is stateless

The runpod SDK's `run_graphql_query()` at
`.venv/lib/python3.12/site-packages/runpod/api/graphql.py:43` creates a
fresh `requests.post()` call per invocation. There is **no shared
Session object**, no connection pool, no state to clean up between
retries. A retry is just "call the same function again" — completely
safe.

### Exception surface (verified)

From the SDK source:
- `runpod.error.AuthenticationError` — HTTP 401 (NOT retryable — real
  error)
- `runpod.error.QueryError` — GraphQL error in response body (NOT
  retryable — real error)
- `requests.exceptions.ConnectionError` — base for socket errors, DNS
  failures, connection resets (retryable)
- `socket.gaierror` — DNS resolution failures (retryable)
- `ConnectionResetError` — OS-level (retryable)
- `TimeoutError` — OS-level socket timeouts (retryable)

The explicit `_TRANSIENT` tuple covers all the retryable classes without
overreaching into `RequestException`.

## No tests

The runpod-toolkit repo has **no test directory, no pytest, no CI** (it's
a v0.1.0 internal scratch repo). Do not set up test infrastructure.
Manual smoke testing via an actual eval launcher run is sufficient.

## Acceptance criteria

1. **Happy path unchanged.** Run a real dark-factory eval launcher once:
   ```bash
   cd /home/leo/src/dark-factory
   python3 scripts/run_vllm_eval.py \
     --config minimax-m25-fp8-new --task df_task_12 --no-volume
   ```
   Expected: pod creates, runs, terminates cleanly just like before. No
   behavior change in the happy path.

2. **Transient error survives.** Smoke-test a simulated transient failure
   by temporarily monkey-patching the SDK at the top of
   `runpod_client.py` **for one test run only** (revert before commit):
   ```python
   _call_count = {'n': 0}
   _real_get_pod = self._runpod.get_pod
   def _flaky_get_pod(pod_id):
       _call_count['n'] += 1
       if _call_count['n'] <= 2:
           raise ConnectionResetError("simulated transient")
       return _real_get_pod(pod_id)
   self._runpod.get_pod = _flaky_get_pod
   ```
   Run `wait_for_pod()` on a live pod. Expected: two retry warnings
   in the log, third call succeeds, pod reaches RUNNING normally.
   **Revert the monkey-patch immediately.**

3. **`wait_for_pod` polling resilience confirmed.** The polling loop at
   line 709 calls `get_pod()` repeatedly. With the decorator on
   `get_pod`, a transient failure during polling no longer crashes
   the loop. No code change needed in `wait_for_pod` itself.

4. **Log messages route through existing logger.** Check that backoff
   warnings (e.g., "Backing off get_pod for 5.0s after 1 tries") appear
   in the same log stream as existing `logger.info/warning` calls from
   `runpod_client.py`.

## Files NOT to touch

- `pyproject.toml` — `backoff` is already a dependency, no changes needed
- `uv.lock` — no dependency changes
- `runpod_toolkit/compute/pod_manager.py` — has its own higher-level
  retry logic for `create_pod` / `create_cpu_pod`; leave alone
- Anything outside `runpod_client.py`

## Commit and merge

1. Change should live in `/home/leo/src/runpod-toolkit/` (separate repo).
2. Branch from `main` in that repo.
3. Single focused commit: "feat(runpod-client): retry transient network errors".
4. Push and open PR / merge locally — follow whatever workflow that repo
   uses. No GitHub Actions to wait for since there's no CI.

## Why this is the whole plan

Before simplification, I considered writing a custom retry decorator with
hand-rolled exponential backoff. That was an unnecessary reinvention —
`backoff` v2.2.1 is already installed and provides exactly the API we
need. The entire change is:

- 4 new lines of imports at the top of `runpod_client.py`
- 5 lines defining `_TRANSIENT`
- 3 decorator applications (one line each)
- 3 `except _TRANSIENT: raise` clauses (one added to each target method)

Total: ~15 lines added, ~0 lines removed, 1 file touched. Zero behavior
change in the happy path. Transient failures self-heal.

## Critical files (one-line index)

- `/home/leo/src/runpod-toolkit/runpod_toolkit/compute/runpod_client.py:36` — logger setup
- `/home/leo/src/runpod-toolkit/runpod_toolkit/compute/runpod_client.py:645-668` — `get_pod`
- `/home/leo/src/runpod-toolkit/runpod_toolkit/compute/runpod_client.py:670-685` — `list_pods`
- `/home/leo/src/runpod-toolkit/runpod_toolkit/compute/runpod_client.py:687-707` — `terminate_pod`
- `/home/leo/src/runpod-toolkit/runpod_toolkit/compute/runpod_client.py:709-770` — `wait_for_pod` (will gain resilience for free)
- `/home/leo/src/runpod-toolkit/pyproject.toml:8` — `backoff` already listed
- `/home/leo/src/runpod-toolkit/runpod_toolkit/compute/pod_manager.py:168-222,277-331` — **DO NOT TOUCH** (has its own retry for create_*)
