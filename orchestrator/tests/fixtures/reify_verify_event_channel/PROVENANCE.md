# Provenance — `reify_verify_event_channel` fixtures

These fixtures exist so dark-factory's reader of reify's verify event channel
(`REIFY_VERIFY_EVENT_LOG`, contract reify
`docs/prds/v0_6/run-all-runner-infra-harness.md` §4 C2/C3, amended by reify
retro-critic ruling R1) is tested against **real producer bytes**: real
`run_all.sh` transcripts with the channel each run wrote beside them, and real
`@@REIFY_SLOT_TIMEOUT@@` stderr lines with the record their emitter wrote.
Nothing here is hand-typed or re-sliced.

## The rule

> **If a test that reads these bytes fails, fix the consumer
> (`orchestrator/src/orchestrator/verify_event_channel.py`,
> `verify_classify.py`, `merge_lane/shadow.py`), never the fixture.**
> Re-capture with the recipe below if reify's producer has genuinely moved.

## These are PRODUCER-only bytes

Every `.events.jsonl` here holds only what reify wrote. In production DF
allocates the channel file and writes its own `channel_open` header record
FIRST (C3 rule 1 as restated by ruling R1: the channel is present when DF
allocated it for this verify and wrote its header). So a test composes
`encode_channel_open(...) + <these bytes>`, which is the production file
shape, and **these bytes alone read as channel-ABSENT** — a reify `run_start`
no longer makes a channel present.

## Capture

| | |
|---|---|
| **Grounding** | **Captured run** (strongest) |
| **Captured** | 2026-10-23, dark-factory task **6856** |
| **reify main at capture** | `246222addcffc2edb303fa54873357421e65dc41` |
| **Scratch dir** | `/tmp/df6856-capture/` (outside every worktree; never committed) |

### (1) `run_all-pool.*` and `run_all-serial.*`

Driven through reify's own harness, `tests/infra/test_verify_events.py`
(`RunAllFixture`, `B1_MEMBERS`, `B1_ROWS`, `B1_BRANCHES` — the same inputs as
its `RunAllChannelTests` B1 case). Members: `test_fail.sh` (prints
`  FAIL: x`, exits 1), `test_pass.sh` and `test_skippable.sh` (print
`member <name> ok`, exit 0). Each `.log` is run_all's stdout with stderr merged;
each `.events.jsonl` is the channel that same run wrote. Both runs exited 1.

Driver, verbatim (`python3 -I driver.py /tmp/df6856-capture/out`):

```python
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, '/home/leo/src/reify/tests/infra')
import test_verify_events as t  # noqa: E402

OUT = Path(sys.argv[1])


class StandIn:
    def __init__(self):
        self.cleanups = []

    def addCleanup(self, f, *a):
        self.cleanups.append((f, a))

    def assertEqual(self, a, b, msg=None):
        assert a == b, (a, b, msg)

    def close(self):
        for f, a in reversed(self.cleanups):
            f(*a)


for branch, knobs in t.B1_BRANCHES:
    T = StandIn()
    try:
        fx = t.RunAllFixture(T, t.B1_MEMBERS)
        env = fx.env(fx.manifest(t.B1_ROWS), **knobs)
        result = subprocess.run(
            fx.argv(), env=env, cwd=fx.cwd, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, errors='surrogateescape', timeout=600,
        )
        assert result.returncode == 1, result.returncode
        (OUT / f'run_all-{branch}.log').write_text(result.stdout, errors='surrogateescape')
        shutil.copyfile(fx.channel, OUT / f'run_all-{branch}.events.jsonl')
        print(branch, 'rc', result.returncode)
    finally:
        T.close()
```

The pool branch retried `test_fail.sh` serially, so its log carries the
`--- attempt 1 (concurrent pool) ---` / `--- attempt 2 (serial retry) ---`
delimiters inside the one `test_fail.sh` block.

### (2) `slot_timeout-fatal.*` and `slot_timeout-soft.*`

The real emitter, `scripts/lib_slot_acquire.sh::slot_emit_timeout` (it sources
`lib_event_log.sh` itself), run from `/home/leo/src/reify` with
`OUT=/tmp/df6856-capture/out`:

```sh
REIFY_VERIFY_EVENT_LOG=$OUT/slot_timeout-fatal.events.jsonl bash -c 'source scripts/lib_slot_acquire.sh; slot_emit_timeout test_slot 8 1800 fatal /tmp/reify-test-semaphore.lock lib_test_semaphore.sh' 2> $OUT/slot_timeout-fatal.stderr.txt
REIFY_VERIFY_EVENT_LOG=$OUT/slot_timeout-soft.events.jsonl bash -c 'source scripts/lib_slot_acquire.sh; slot_emit_timeout pool_slot 2 30 soft /tmp/reify-run-all-pool.lock run_all_pool' 2> $OUT/slot_timeout-soft.stderr.txt
```

Each `.stderr.txt` is the one column-0 `@@REIFY_SLOT_TIMEOUT@@ ...` sentinel
line; each `.events.jsonl` is exactly one `slot_timeout` record.

## Files

| File | Lines / bytes | sha256 |
|---|---|---|
| `run_all-pool.events.jsonl` | 9 / 1433 | `2626862f6e539b781e34c20a8d69c7d3a3b9bb790558f1fa3a38fb598eda9f6b` |
| `run_all-pool.log` | 25 / 714 | `aadf87a534afe7fe0b271fd1c2a32d16be0fd71df3f3cf63a35f7f822a4a4d52` |
| `run_all-serial.events.jsonl` | 8 / 1307 | `e4360d480c802f63f1852e29ee553ef0ac5b61bdb0f41d9945a1b878f19d9552` |
| `run_all-serial.log` | 20 / 496 | `e337dd7a0420cc2ac7eb6832401f5efcbb0f5d81edd0f98214c30a5dce7d91f8` |
| `slot_timeout-fatal.events.jsonl` | 1 / 214 | `a5f8ecf0ceabda2fb68439f4b24676a8c603dbc9327ab5d0d9a7d8a1cc78927e` |
| `slot_timeout-fatal.stderr.txt` | 1 / 114 | `2cf80fe39c63213ae96c6db75c5a30e2497deb4ed8f91a3455e89b4670982478` |
| `slot_timeout-soft.events.jsonl` | 1 / 200 | `7aacef14e8b7271d29a4de4a95283d8937d303a7365b0612575bac9ca81f12ca` |
| `slot_timeout-soft.stderr.txt` | 1 / 109 | `21d33febcf54f3aae34c7212c850355ec489152d088095cc18f4b04cf84fbe47` |

## Validation with reify's reference reader

`python3 -I /home/leo/src/reify/tests/infra/verify_events.py summary <file>`,
run in the scratch dir. Its `channel=` line is reify's own `run_start` notion of
presence, not DF's header rule.

```
$ python3 -I /home/leo/src/reify/tests/infra/verify_events.py summary run_all-pool.events.jsonl
channel=present
runs=1
discovered=3
failures=1
failed=test_fail.sh
flaky=
skipped=
interrupted=false
slot_timeout_fatal=0
slot_timeout_soft=0
skipped_lines=0
$ python3 -I /home/leo/src/reify/tests/infra/verify_events.py summary run_all-serial.events.jsonl
channel=present
runs=1
discovered=3
failures=1
failed=test_fail.sh
flaky=
skipped=
interrupted=false
slot_timeout_fatal=0
slot_timeout_soft=0
skipped_lines=0
$ python3 -I /home/leo/src/reify/tests/infra/verify_events.py summary slot_timeout-fatal.events.jsonl
channel=absent
runs=0
discovered=0
failures=0
failed=
flaky=
skipped=
interrupted=false
slot_timeout_fatal=1
slot_timeout_soft=0
skipped_lines=0
$ python3 -I /home/leo/src/reify/tests/infra/verify_events.py summary slot_timeout-soft.events.jsonl
channel=absent
runs=0
discovered=0
failures=0
failed=
flaky=
skipped=
interrupted=false
slot_timeout_fatal=0
slot_timeout_soft=1
skipped_lines=0
```
