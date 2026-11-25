# Verify-host preflight

`orchestrator verify-host-preflight` runs a project's merge gate on one named
verify host, through the same remote path the merge lane uses, and records the
verdict in a file. A host's bring-up gates cite these records
(`plans/verify-host-roster-and-rented-trial-prd.md` §6.2 and §9). The code is
in `orchestrator/src/orchestrator/verify_host_preflight.py`.

## What it runs

It verifies one commit. The default is the tip of `git.main_branch`, and
`--sha` takes any revision that names a commit in the project checkout.

- **The gate.** It builds the merge-verify spec with the lane's own builder,
  `verify_runner.py::build_merge_verify_spec`, over every registered module.
  It then sets the breadth to `full`, so no module is scoped out. `task_files`
  is unset and no verify cache is shipped, so every unit runs on the host and
  no replayed green stands in for the host's own run. Every other field of the
  merge-gate profile (workspace, env, cold timeouts, pre-provision) is exactly
  what the lane ships.
- **The path.** The runner is built by `verify_hosts.py::remote_runner_for`,
  the same builder the harness uses. The preflight first calls
  `RemoteRunner.ensure_gate`, timed on its own: this publishes the CLI
  checkout's own sha as the gate and checks the gate's handshake. It then calls
  `RemoteRunner.run_merge_verify`, which pushes `refs/merge-verify/<id>` and
  runs `orchestrator verify-merge` over ssh in that gate.
- **Where it runs on the host.** `verify-merge` makes its own ephemeral
  `_merge-<uuid>` worktree. It runs as its own process, separate from any
  orchestrator, so it holds no host slot. On a busy host it is slower, not
  unsafe.
- **Archives.** The runner's red, green and no-verdict archives land in
  `data/verify-logs/preflight/`, beside the records.

Any declared runner in `verify_runners` is accepted, whether enabled or not, so
a host can be preflighted before it is enabled.

With `--reference <runner>`, the same commit and the same spec are verified on
a second declared runner at the same time, and the skipped tests are compared
(see "Skip parity" below).

## Usage and exit codes

```
orchestrator verify-host-preflight --runner <name> [--sha <rev>] [--reference <name>] [--json] [--config <yaml>]
```

Without `--json` it prints a short summary. With `--json` it prints the
runner's record.

| Exit | Meaning |
|------|---------|
| 0 | The gate passed and, with `--reference`, skip parity holds. |
| 1 | The gate is red, there was no verdict, the pass was trivial (vacuous), or parity is broken or undetermined. |
| 2 | Refused before any dispatch. |

These cases are refused with exit 2:

- an undeclared runner;
- a reference equal to the runner;
- a runner without usable gate paths (`df_checkout_path`);
- a runner name that is not one path segment;
- `git.persistent_merge_worktree` on in the project config;
- a `--sha` that names no commit;
- a malformed exceptions file;
- a process that cannot read its own deployed sha.

## What it does not prove

- **It is never trust evidence or trial data.** No event store reaches the
  preflight, so it emits no `merge_verify` or other event. Its record file is
  its only output.
- **One green is not agreement.** It does not show that the host agrees with
  the workstation over time. That is probation (PRD §6.4).
- **No task verifies.** It runs the merge gate only.
- **The gate is the CLI's sha.** The gate built is the commit of the checkout
  the CLI runs from, not necessarily the one a running orchestrator was
  deployed from.
- **The persistent-worktree check reads the project config only.** It does not
  read the host's own yaml. If that yaml turns `git.persistent_merge_worktree`
  on, `verify-merge` shares the lane's worktree lock, and a busy host answers
  with a `flock_contention` red.
- **A stale remote main can give a trivial pass.** The spec carries no
  `task_files`, so the host derives the diff against its own main. If that
  main is stale, a docs-only diff can make it pass without running anything.
  Such a result is recorded as `trivial: true` and `passed: false`.

## Reading the record

Each runner's record is at `data/verify-logs/preflight/<runner>-<sha>.json`
under the project root. The fields are:

| Field | Meaning |
|-------|---------|
| `runner` | The declared runner name. |
| `sha` | The full commit verified. |
| `gate` | The deployed sha the gate was built from: `sha`, `indeterminate`, `dirty_wire_path`. |
| `passed` | True only for a non-trivial green verdict. |
| `trivial` | The host ran nothing (see above). |
| `wall_secs` | From the start of the gate ensure to the end of the verify. |
| `ensure_gate_secs` | The first gate ensure alone; null when it raised. |
| `gate_built` | Whether that ensure had to build the gate; null when it raised. |
| `summary`, `category` | The verdict's own summary and failure category. |
| `no_verdict` | `{kind, message}` when the host produced no usable verdict; `kind` is a `RunnerUnavailable` kind (`transport`, `gate`, `unparseable`, `no_verdict`, `wire_shape`). |
| `modules.<prefix>.failing_test_ids` | That module's failing junit ids; null when the verdict carries no per-module attribution. |
| `modules.<prefix>.failing_count` | Their count, or null. |
| `failing_test_ids` | The run-wide failing ids. |
| `skipped_test_ids` | The run-wide skipped junit ids, or null when no junit was collected. |
| `skip_parity` | The parity block with `--reference`, else null. |

The test ids use junit's `classname::name` spelling, for example
`tests.test_verify_role_integration::test_role_env_propagates_to_reify_verify_plan[merge]`.
Skipped ids are not attributed per module, because `VerifyResult` carries no
per-module skip attribution.

## Skip parity

With `--reference`, the two runners' skipped vectors are compared:

- `skipped_only_on_runner`: ids skipped on the runner and not on the
  reference;
- `skipped_only_on_reference`: the reverse.

Both runners verify the same commit, so they collect the same tests, and
"skipped on one only" means "skipped on one and run on the other". An id in
either list is excused only by an entry in
`config/verify-host-skip-parity-exceptions.yaml` under the project root. Those
entries appear in `excepted`, each with its owner. Any other id is
`unexplained`, and parity holds only when there are none.

Parity is `undetermined`, and the command exits 1, when either run had no
verdict or collected no junit. A missing vector licenses no conclusion.

### The exceptions file

The file is optional, and an absent file means no exceptions. Each entry names
a test id in junit spelling, the task or ticket that will remove it, and why:

```yaml
exceptions:
  - test_id: "tests.test_verify_role_integration::test_role_env_propagates_to_reify_verify_plan[merge]"
    owner: {type: task, id: 7030}
    reason: Needs a reify checkout at the workstation's reify sha, which task 7030 provisions on verify hosts.
```

Every entry must be owned (INV-12). `owner` is the house owner record,
`{type: task, id: <n>}` or `{type: ticket, id: tkt_...}`. An entry with no
owner, no reason or no test id, or a duplicated test id, fails the load, and
the preflight exits 2 before it dispatches anything.

To add an exception:

1. Take the id from a record's `skip_parity.unexplained`.
2. File or name the task that will make the test run on that host.
3. Add the entry with that owner and a one-line reason.

`orchestrator/tests/test_verify_host_preflight.py` loads this repository's
tracked file, so an ownerless entry fails the suite.
