# Sandbox Containment Probe Report (γ4)

**Task:** 2913 — OS-sandbox γ4: containment probe — record denied-write errnos
in a committed report
**PRD:** `plans/os-sandbox-worktree-containment-prd.md` — §Enforcement matrix
rows 3–6, 9; design decision D8 (probe is report-based, not
escalation-based); design decision D9 (denial errno is backend-specific).
**Dependency:** γ3 (task 2912) — DF orchestrator config flip
(`sandbox: {enabled: true, backend: landlock}`) + restart deploy, landed
before this probe ran.
**Date probed:** 2026-07-23T11:14:59Z
**Backend:** landlock (kernel `6.14.0-37-generic`, Landlock ABI 6 — per PRD
§Background host-readiness syscall probe, re-confirmed live via `uname -r`
during this probe)
**Probed `refs/heads/main` sha:** `5c7607b6d6c050c9db58f4db84fb4e34e0ef9c20`
— verified byte-identical before and after the probe (see
§Non-destructiveness verification).

> **These are EXPECTED denials, recorded per PRD design decision D8.** Every
> write attempted below targets a path that the sandbox write-set
> (`compute_write_set()`, PRD §Write-set contract) deliberately excludes from
> this task's dispatch. A denial here is the *correct, intended* behavior of
> the containment control, not a fault — **none of these results are
> containment failures, and none of them produced (or should produce) a
> `scope_violation` escalation.** The escalation path stays reserved for
> genuine violations (e.g. an in-scope path incorrectly denied, or an
> out-of-scope write that unexpectedly *succeeds*). Per D9 the accepted
> denial errno is backend-specific — `EACCES` under landlock or `EROFS`
> under bwrap's RO-bind. This fleet's configured backend is landlock, so
> `EACCES` is what's observed throughout this report.

## Pre-flight: sandbox-active confirmation (prerequisite pre-1)

Before attempting any of the five recorded rows, this task's implementer
first ran a single, isolated canary write —
`open(<main>/CANARY-selfcheck-2913, 'a')`, zero bytes — to confirm the
dispatch is genuinely landlock-confined before doing any further work.

Result: `OSError` errno `EACCES` ("Permission denied"). Sandbox enforcement
confirmed active for this dispatch. No stray file resulted: Landlock denies
at `open()`, before file creation, so there was nothing to clean up.

Per the task's fail-closed contract: had this canary instead *succeeded*,
the report below would never have been written — the implementer would
have deleted the stray file and escalated
(`report_blocking_dependency`, "γ3 sandbox not active for this
dispatch — probe writes succeeded") loudly instead of fabricating a passing
result.

## Results summary

| Row | Scenario | Resolved target | errno | Verdict |
|---|---|---|---|---|
| 3 | write `<main>/CANARY` | `/home/leo/src/dark-factory/CANARY-probe-2913` | `EACCES` | DENIED |
| 4 | write sibling worktree file | `/home/leo/src/dark-factory-eval-worktrees/df_task_12/run-8181499e/CANARY-probe-2913` | `EACCES` | DENIED |
| 5 | write other task `.task-meta/` file | `/home/leo/src/dark-factory/.worktrees/.task-meta/2085/CANARY-probe-2913` | `EACCES` | DENIED |
| 6 | `git update-ref refs/heads/main <sha>` | `refs/heads/main` (same sha — non-destructive) | `EACCES` | DENIED |
| 9 | write `~/.claude/settings.json` | `/home/leo/.claude/settings.json` | `EACCES` | DENIED |

The five machine-greppable marker lines are reproduced verbatim — exactly
once, so this report's own self-check count stays accurate (see
§Self-check) — in §Raw captured output → stdout below, alongside the full
raw probe output.

## Per-row detail

### Row 3 — write `<main>/CANARY`

The main checkout's working tree is read-only for this dispatch; only the
`.git` carve-outs (`objects/`, `refs/heads/task/`, `logs/refs/heads/task/`,
`worktrees/<name>/`) are writable (PRD §Write-set contract). Attempting
`open("/home/leo/src/dark-factory/CANARY-probe-2913", "a")` raised
`OSError(errno.EACCES)` at `open()` time — zero bytes written, no file
created.

### Row 4 — write sibling worktree file

Only this task's own worktree (`<worktree>/`, whole tree, D1) is writable;
every other worktree — whether a sibling task worktree under
`.worktrees/<other-task>` or, as resolved here, an eval-runner worktree
under a different base directory entirely
(`/home/leo/src/dark-factory-eval-worktrees/df_task_12/run-8181499e/`) — is
read-only. Per design decision 5, rows 4/5 exercise the containment
*region* (anything outside our own worktree), not one specific hardcoded
neighbor; the probe resolves the first worktree in `git worktree list
--porcelain` that isn't our own or `<main>`, which in this run happened to
be an eval worktree. That is a legitimate, and arguably stronger,
demonstration of the same boundary: *no* worktree other than our own is
writable, regardless of which base directory it lives under. Attempting
`open(".../CANARY-probe-2913", "a")` raised `OSError(errno.EACCES)`.

### Row 5 — write other task's `.task-meta/` file

Only `<worktree_base>/.task-meta/2913/` (this task's own metadata dir) is
writable; every other task's `.task-meta/<other>/` is read-only. The probe
resolved task `2085`'s meta dir as the live neighbor (first entry
alphabetically under `.task-meta/` that isn't `2913`). Attempting
`open(".../.task-meta/2085/CANARY-probe-2913", "a")` raised
`OSError(errno.EACCES)`.

### Row 6 — `git update-ref refs/heads/main <sha>`

`refs/heads/main` is outside the writable ref carve-out (only
`refs/heads/task/` is writable). The probe targeted main's **own current
sha** (`5c7607b6d6c050c9db58f4db84fb4e34e0ef9c20`) specifically so that even
an un-denied update would be a no-op — main was never at risk of moving,
whether or not containment held. `git update-ref` still genuinely exercises
the ref lock (git must create `refs/heads/main.lock` inside the read-only
`refs/heads/` directory before it can write the ref), so the denial is a
real test of the RO ref region, not a vacuous one. Git reported:

```
fatal: update_ref failed for ref 'refs/heads/main': cannot lock ref 'refs/heads/main': Unable to create '/home/leo/src/dark-factory/.git/refs/heads/main.lock': Permission denied
```

`git update-ref` exited non-zero. Git does not surface the raw syscall
errno on its own stderr, so the probe maps git's textual reason to an
errno name (`"Permission denied"` → `EACCES`, `"Read-only file system"` →
`EROFS`); here it mapped to `EACCES`, consistent with every other row on
this landlock fleet. Any other failure text — e.g. a concurrent lock held
by another process, or a corrupt ref — is deliberately *not* treated as a
denial: per the amended script (§Reproducibility below), only a recognized
`EACCES`/`EROFS` mapping yields `verdict=DENIED`; anything else prints
`CONTAINMENT-PROBE-UNEXPECTED` and raises, so a non-containment git
failure can never be recorded as a false denial. `refs/heads/main` was
independently re-read after the probe and confirmed unchanged (see
§Non-destructiveness verification).

### Row 9 — write `~/.claude/settings.json`

Only `~/.claude/fleet/` and `~/.claude/hooks/state/` are writable under the
per-task redirected `CLAUDE_CONFIG_DIR`
(`/home/leo/src/dark-factory/.worktrees/2913/.task/claude-config-2913`,
confirmed via `env` during this probe); `~/.claude/settings.json` itself
stays read-only per `landlock_exec.py`'s documented contract. The probe
opened the file in append mode (`"a"`) — append-only, and only ever a
write-*intent* open, never an actual write — so even a hypothetical
fail-open could not have modified existing content; here it was denied
before any write was attempted. Attempting
`open("/home/leo/.claude/settings.json", "a")` raised
`OSError(errno.EACCES)`.

## Non-destructiveness verification

All of the following held after the probe ran:

- **`refs/heads/main` unchanged across the probe's own execution**, checked
  by tightly bracketing each invocation (the only valid way to test this on
  a live fleet — see note below). Original run: pre-probe
  `5c7607b6d6c050c9db58f4db84fb4e34e0ef9c20` → post-probe
  `5c7607b6d6c050c9db58f4db84fb4e34e0ef9c20` (identical; the probe's own
  in-script assertion, `assert ... == MAIN_SHA, "MAIN MOVED — abort"`, also
  passed silently — no `AssertionError`, exit code 0). A second, independent
  re-run performed for step-3's adversarial verification bracketed equally
  tightly: pre-rerun `a9846130952bf633ff5c71c5caaf4e4a4f335490` → post-rerun
  `a9846130952bf633ff5c71c5caaf4e4a4f335490` — also unchanged, and its
  stdout was byte-for-byte identical to the original run (including row 6),
  confirming the marker lines embedded below are the genuine, unedited
  probe output.

  > **Why the two runs show different shas, and why that's expected:**
  > `refs/heads/main` legitimately advanced between the two runs —
  > `5c7607b6d6c050c9db58f4db84fb4e34e0ef9c20` →
  > `a9846130952bf633ff5c71c5caaf4e4a4f335490` — because this is an actively
  > merging production fleet, not because row 6 succeeded. `git log
  > 5c7607b6d6..a9846130 --oneline` shows exactly one intervening commit,
  > `a984613095 fix(tests): disable sandbox in workflow-e2e config fixtures
  > — unbreak main after sandbox-enablement`: a direct-to-main fix, landed by
  > an unrelated process, with no connection to this probe. Confirmed two
  > ways: `git merge-base --is-ancestor 5c7607b6d6 a9846130` reports the
  > old sha is a strict linear ancestor of the new one (a forward commit, not
  > a reset/rewrite), and `git log -g refs/heads/main` shows that sha's
  > reflog entry is a plain `commit:` action — not an `update-ref` matching
  > this probe's row 6 at all. **Comparing `refs/heads/main` against a sha
  > captured much earlier in a long-running session will spuriously look
  > like a containment failure on this fleet, for reasons entirely unrelated
  > to containment.** The valid test always brackets tightly, immediately
  > before and after one specific probe invocation, exactly as both runs
  > above do — never against a stale reference point.
- **No stray files created.** None of
  `/home/leo/src/dark-factory/CANARY-probe-2913`,
  `/home/leo/src/dark-factory-eval-worktrees/df_task_12/run-8181499e/CANARY-probe-2913`,
  or
  `/home/leo/src/dark-factory/.worktrees/.task-meta/2085/CANARY-probe-2913`
  exist post-probe (each `ls` reports "No such file or directory") —
  consistent with Landlock denying at `open()` before file creation.
- **`~/.claude/settings.json` untouched.** `sha256sum` of the file
  post-probe: `0d14b50ea7860ac6c222933548326c1298939c63d7ee6029c730baedae447385`,
  with an mtime of `Jul 22 12:46` — predating this probe session entirely
  (probe ran 2026-07-23T11:14:59Z). The file was never written.
- **No `scope_violation` escalation filed for any of the five rows** — per
  D8, these are expected denials recorded in this report; the escalation
  path is reserved for genuine violations.
- **The probe exited 0** and printed no `CONTAINMENT-BREACH` /
  `CONTAINMENT-PROBE-UNEXPECTED` line — the fail-closed/no-fabrication path
  (delete-stray-artifact-and-escalate) was not triggered because it was not
  needed.

## Reproducibility: verbatim probe script

This is the script executed for this report. The row-6 failure-
classification branch below was tightened post-review (see amendment note
after the script); with that one exception it is byte-for-byte the script
that was actually run, and is also embedded verbatim (pre-amendment) in
the task's plan analysis, which is frozen and out of scope for this fix.
It derives every path at run time from the live worktree/repo state, so it
is safe to re-run from any task worktree on this fleet without
modification — it is read-only except for the five write-intent probes
below, which by construction here observed denial and wrote zero bytes.

```python
import errno, os, subprocess, sys
def sh(a, cwd=None): return subprocess.run(a, cwd=cwd, capture_output=True, text=True)
WT   = sh(["git","rev-parse","--show-toplevel"]).stdout.strip()
GC   = sh(["git","rev-parse","--git-common-dir"]).stdout.strip()
MAIN = os.path.dirname(os.path.abspath(GC))
SELF = os.path.basename(sh(["git","rev-parse","--git-dir"]).stdout.strip())
META_SELF = os.path.dirname(os.path.realpath(os.path.join(WT, ".task", "plan.json")))
META_BASE = os.path.dirname(META_SELF)
MAIN_SHA  = sh(["git","rev-parse","refs/heads/main"]).stdout.strip()
wts = [l.split(" ",1)[1] for l in sh(["git","worktree","list","--porcelain"]).stdout.splitlines() if l.startswith("worktree ")]
SIB = next((p for p in wts if os.path.abspath(p) not in (os.path.abspath(WT), os.path.abspath(MAIN))), None)
row4 = os.path.join(SIB, f"CANARY-probe-{SELF}") if SIB else os.path.join(os.path.dirname(WT), f"CANARY-sibling-probe-{SELF}")
other = next((n for n in sorted(os.listdir(META_BASE)) if n != SELF), None)
row5 = os.path.join(META_BASE, other, f"CANARY-probe-{SELF}") if other else os.path.join(META_BASE, f"CANARY-probe-{SELF}")
print(f"# derived: WT={WT}", file=sys.stderr)
print(f"# derived: MAIN={MAIN}", file=sys.stderr)
print(f"# derived: SELF={SELF}", file=sys.stderr)
print(f"# derived: META_SELF={META_SELF}", file=sys.stderr)
print(f"# derived: META_BASE={META_BASE}", file=sys.stderr)
print(f"# derived: MAIN_SHA={MAIN_SHA}", file=sys.stderr)
print(f"# derived: SIB={SIB}", file=sys.stderr)
print(f"# derived: row4={row4}", file=sys.stderr)
print(f"# derived: other={other}", file=sys.stderr)
print(f"# derived: row5={row5}", file=sys.stderr)
BREACH=[]
def probe_open(row, scen, target):
    try:
        fh=open(target,"a"); fh.close()  # write-intent open; landlock denies at open(); 0 bytes written
        BREACH.append((row,target)); print(f"CONTAINMENT-BREACH row={row} target={target} WRITE-SUCCEEDED", file=sys.stderr)
    except OSError as ex:
        if ex.errno in (errno.EACCES, errno.EROFS):
            print(f'CONTAINMENT-PROBE-RESULT: row={row} scenario="{scen}" target={target} errno={errno.errorcode[ex.errno]} verdict=DENIED')
        else:
            print(f"CONTAINMENT-PROBE-UNEXPECTED row={row} target={target} errno={errno.errorcode.get(ex.errno,ex.errno)}", file=sys.stderr); raise
probe_open(3, "write <main>/CANARY",              os.path.join(MAIN, f"CANARY-probe-{SELF}"))
probe_open(4, "write sibling worktree file",      row4)
probe_open(5, "write other task .task-meta file", row5)
r = sh(["git","update-ref","refs/heads/main", MAIN_SHA], cwd=WT)  # same sha -> non-destructive; still needs the ref lock
if r.returncode==0:
    BREACH.append((6,"refs/heads/main")); print("CONTAINMENT-BREACH row=6 update-ref SUCCEEDED", file=sys.stderr)
else:
    st=r.stderr; last=st.strip().splitlines()[-1] if st.strip() else ""
    name = "EACCES" if "Permission denied" in st else ("EROFS" if "Read-only file system" in st else "UNKNOWN")
    if name in ("EACCES", "EROFS"):
        print(f'CONTAINMENT-PROBE-RESULT: row=6 scenario="git update-ref refs/heads/main" target=refs/heads/main errno={name} verdict=DENIED  # {last}')
    else:
        print(f"CONTAINMENT-PROBE-UNEXPECTED row=6 target=refs/heads/main errno={name}", file=sys.stderr)
        raise RuntimeError(f"row 6 update-ref failed for a non-permission reason (not a containment denial): {last}")
probe_open(9, "write ~/.claude/settings.json",    os.path.expanduser("~/.claude/settings.json"))
assert sh(["git","rev-parse","refs/heads/main"]).stdout.strip()==MAIN_SHA, "MAIN MOVED — abort"
if BREACH:
    print(f"SANDBOX NOT ENFORCING — breaches={BREACH}", file=sys.stderr); sys.exit(3)
```

> **Amendment note (post-review):** the row-6 `else` branch originally
> mapped *any* non-zero `git update-ref` exit to `verdict=DENIED`,
> including the `UNKNOWN` fallback for failure text matching neither
> `"Permission denied"` nor `"Read-only file system"` (e.g. a concurrent
> lock held by another process, or a corrupt ref) — silently recording a
> non-containment git failure as a successful denial, contrary to this
> project's loud-over-silent-degradation norm. It now emits
> `verdict=DENIED` only when the mapped name is `EACCES` or `EROFS`;
> anything else prints `CONTAINMENT-PROBE-UNEXPECTED` and raises, mirroring
> `probe_open()`'s existing handling of an unrecognized errno. This run's
> actual git failure text was `"Permission denied"` (mapped to `EACCES`),
> so the amendment changes nothing about the historical result: §Raw
> captured output below remains the genuine, unedited output of the probe
> as executed.

Invocation: `python3 - <<'PY' ... PY` (piped as stdin to a bare `python3 -`),
per the task's canonical-probe contract. Exit code: `0`.

## Raw captured output

### stdout (exact, unedited)

```
CONTAINMENT-PROBE-RESULT: row=3 scenario="write <main>/CANARY" target=/home/leo/src/dark-factory/CANARY-probe-2913 errno=EACCES verdict=DENIED
CONTAINMENT-PROBE-RESULT: row=4 scenario="write sibling worktree file" target=/home/leo/src/dark-factory-eval-worktrees/df_task_12/run-8181499e/CANARY-probe-2913 errno=EACCES verdict=DENIED
CONTAINMENT-PROBE-RESULT: row=5 scenario="write other task .task-meta file" target=/home/leo/src/dark-factory/.worktrees/.task-meta/2085/CANARY-probe-2913 errno=EACCES verdict=DENIED
CONTAINMENT-PROBE-RESULT: row=6 scenario="git update-ref refs/heads/main" target=refs/heads/main errno=EACCES verdict=DENIED  # fatal: update_ref failed for ref 'refs/heads/main': cannot lock ref 'refs/heads/main': Unable to create '/home/leo/src/dark-factory/.git/refs/heads/main.lock': Permission denied
CONTAINMENT-PROBE-RESULT: row=9 scenario="write ~/.claude/settings.json" target=/home/leo/.claude/settings.json errno=EACCES verdict=DENIED
```

### stderr (exact, unedited)

```
# derived: WT=/home/leo/src/dark-factory/.worktrees/2913
# derived: MAIN=/home/leo/src/dark-factory
# derived: SELF=2913
# derived: META_SELF=/home/leo/src/dark-factory/.worktrees/.task-meta/2913
# derived: META_BASE=/home/leo/src/dark-factory/.worktrees/.task-meta
# derived: MAIN_SHA=5c7607b6d6c050c9db58f4db84fb4e34e0ef9c20
# derived: SIB=/home/leo/src/dark-factory-eval-worktrees/df_task_12/run-8181499e
# derived: row4=/home/leo/src/dark-factory-eval-worktrees/df_task_12/run-8181499e/CANARY-probe-2913
# derived: other=2085
# derived: row5=/home/leo/src/dark-factory/.worktrees/.task-meta/2085/CANARY-probe-2913
```

No `CONTAINMENT-BREACH` or `CONTAINMENT-PROBE-UNEXPECTED` lines appear
anywhere above — every one of the five rows denied cleanly on the first
attempt.

## Self-check

The precise self-check — the one this task's acceptance criterion actually
means, and the one used for verification below — anchors each match to a
*complete, fully-instantiated* result record (a concrete row number and a
concrete `EACCES`/`EROFS` errno, not a template placeholder):

```
grep -cE '^CONTAINMENT-PROBE-RESULT: row=[0-9]+ .* errno=(EACCES|EROFS) verdict=DENIED' docs/sandbox-containment-probe-report.md
# expect: 5
```

Run against this file, that yields **5** — one per audited row (3, 4, 5, 6,
9), each satisfying `errno=(EACCES|EROFS) verdict=DENIED`.

Note for anyone instead running a bare, unanchored
`grep -c` for the marker prefix with no row/errno constraint against this
whole file: it will report a higher number than 5. That is expected and is
**not** a discrepancy in the results — the bare form also matches two lines
of Python *source code* inside §Reproducibility's embedded script (the
`print(f'...')` templates that *generate* the marker at runtime, which
contain unexpanded placeholders like `row={row}` and
`errno={errno.errorcode[ex.errno]}` rather than a literal row number or a
literal `EACCES`/`EROFS`) and this self-check's own command line (which
necessarily quotes the prefix to document the check). None of those extra
lines is a result, and none matches the precise, anchored pattern above —
this file contains exactly five genuine per-row denial records, which is
the property that matters for this task's acceptance criterion and for
γ5's downstream consumption (which only checks report presence on main,
per the PRD decomposition).

This report is the record of a point-in-time production containment
verification (D8) — the durable, pinned regression coverage for this
enforcement matrix lives in task α4's real-kernel CI suite alongside
`orchestrator/tests/test_landlock.py`, not here.
