# serve_escalation_mcp readiness bound: the measurement, and the value derived from it

Task 5868, measured 2026-10-13 on branch `task/5868` at base `8ab92d5067`
(CPython 3.13.9, the worktree's own `.venv`, a 32-core host under real fleet
load).

## What it governs

`escalation/tests/conftest.py::_READY_TIMEOUT_S`, the readiness deadline of the
`serve_escalation_mcp` harness. `_serve_escalation_mcp_impl`'s `_start` polls a
real MCP handshake until it succeeds, the serving thread exits, or this
deadline passes, and then raises `EscalationServerNotReady`. It is pinned by
`escalation/tests/test_serve_escalation_mcp_fixture.py::test_the_readiness_bound_clears_the_measured_first_start_with_headroom`
and `::test_the_readiness_bound_expires_before_the_verify_per_test_budget`.

## The derivation rule

    bound = round_up_to_whole_minute(worst measured first-start readiness latency
                                     x UNDER_LOAD_HEADROOM_FACTOR)

Adopted wholesale from task 5442 (`plans/pytest-per-test-timeout-measurement-2026-09-17.md`),
not invented here. The factor is
`orchestrator/tests/_orch_helpers.py::UNDER_LOAD_HEADROOM_FACTOR` (8); the
rationale for 8x rather than 2x lives in that note. The worst run is kept,
never the freshest. The bound is never-narrow: lower it only on a new
measurement taken at comparable or higher load, never on judgement.

## Corpus

| source | date | code shape | loadavg | process state | path | figure |
|---|---|---|---|---|---|---|
| task 5060 architect | 2026-09-24 | pre-`96dd5b79c8`, `ws='websockets-sansio'` | 162-335 | first start in a fresh process | startup FAILURE against an already-held port | **5.46s** to the uvicorn bind failure (later starts 0.67-0.81s; one poll overran the deadline by up to 2.6s, before per-attempt bounding landed in `70a5ec8beb`) |
| task 5868 architect | 2026-10-13 | base `8ab92d5067`, `ws='none'` | 90/107/138 | fresh process, no suite running | happy path | first ready 0.599s, second 0.140s |
| task 5868 architect | 2026-10-13 | same | 175-196 | 12 fresh processes alongside the full escalation suite under `-n auto --dist loadgroup` (2412 passed in 33.67s) | happy path | first ready 0.177-0.235s, second 0.119-0.185s |
| same run | 2026-10-13 | same | 175-196 | `--durations=0` setup of the three consumer modules (`test_capability_guard_http.py`, `test_status_authority_gate.py`, `test_legibility_census_escalation_e2e.py`) | happy path | <= 0.33s |

**Why a failure-path figure bounds the happy path.** A successful start does
every step the failed one did (imports, `create_server`, thread start, uvicorn
startup up to the bind) and then more: lifespan startup and the first
handshake. So under the conditions it was measured in, 5.46s is a strict lower
bound on a successful first start.

**Why it is kept although today's happy path is ~25x faster.** It is the
highest-load measurement on record (loadavg up to 335, against today's 196),
and it was taken on a code shape that did more in-thread work. Today's figures
come from a lower load and a lighter shape, so they do not refute it, and the
rule keeps the worst run.

## Arithmetic

    worst measured first start              =  5.46s
    x UNDER_LOAD_HEADROOM_FACTOR (8)        = 43.68s
    round up to a whole minute              = 60s    -> _READY_TIMEOUT_S

The previous 10s kept only 1.8x headroom over 5.46s.

Upper edge: 60 < 300, the `--timeout=300` in `escalation/orchestrator.yaml`'s
`test_command`. Escalation runs `timeout_method = "signal"`
(`escalation/pyproject.toml`). Because of that ordering, an
alive-but-never-ready server fails with the structured
`EscalationServerNotReady`, which names the side of the wire that failed,
instead of a bare pytest-timeout kill.

## Cost

Only an alive-but-never-ready server pays the full bound. A healthy server
returns on its first successful handshake, and a dead one returns when its
thread exits. `test_a_server_that_dies_during_startup_ends_the_wait_and_is_named`
keeps the production bound on purpose, so it would pay 60s once, and only on a
regression of the end-on-thread-death behaviour.

## Method

The probe loads the conftest under a unique module name, so it never collides
with pytest's own `conftest`. It wraps `_bind_escalation_listener` to timestamp
the bind, and times two `_start` calls. `ready` is bind -> `_start` return,
which approximates the deadline window. `total` is the whole call, including
`create_server`.

```python
import importlib.util, json, sys, tempfile, time
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    '_readiness_probe_conftest', Path(sys.argv[1]).resolve())
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

real_bind, bound_at = mod._bind_escalation_listener, []
def _timed_bind():
    bound_at.append(time.monotonic())
    return real_bind()
mod._bind_escalation_listener = _timed_bind

result = {'loadavg': Path('/proc/loadavg').read_text().split()[:3]}
with tempfile.TemporaryDirectory() as tmp:
    gen = mod._serve_escalation_mcp_impl()
    start = next(gen)
    try:
        for label in ('first', 'second'):
            t0 = time.monotonic()
            start(Path(tmp) / label)
            done = time.monotonic()
            result[label] = {'ready': round(done - bound_at[-1], 3),
                             'total': round(done - t0, 3)}
    finally:
        gen.close()
print(json.dumps(result))
```

Invocation, from the worktree. `env -u VIRTUAL_ENV` stops an agent shell
resolving the main checkout's code:

    cd escalation && env -u VIRTUAL_ENV ../.venv/bin/python probe.py tests/conftest.py

For the under-load rows, loop it over fresh processes while this runs:

    env -u VIRTUAL_ENV ../.venv/bin/python -m pytest tests -q --durations=0 --timeout=0 -p no:cacheprovider

The uvicorn lifespan `CancelledError` tracebacks printed at teardown are
expected noise. Read the JSON line printed last.

## Residue

No happy-path measurement exists yet at loadavg >= 250 on the current
`ws='none'` shape. That is the measurement that would show whether 5.46s still
binds. Raising the bound is always safe; narrowing it needs that measurement.
