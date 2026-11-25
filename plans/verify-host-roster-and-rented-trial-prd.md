# A rented verify host: fresh-host readiness, a live host roster, and a two-step throughput trial

**Status:** authored 2026-11-25 (`/team` study, then `/prd` author mode with Leo attending; one design seat verified the roster contract against the code). Decomposed 2026-11-25 into tasks α1 to λ; their ids are stamped in `plans/verify-host-roster-and-rented-trial-prd.capability-manifest.yaml`.
**Type:** extension of the shipped multi-host verify lever (`plans/merge-throughput-multihost-verify-prd.md`, `plans/remote-task-leg-verify-prd.md`) to a third host, plus one new ownership seam: the verify-host roster becomes live configuration.
**Approach:** B+H. G5 applies on every count: the roster change touches `config.py`, `harness.py`, `verify_runner.py`, `verify_hosts.py`, `merge_lane/host_quarantine.py` and `merge_lane/post_merge_verify.py`; it has three consumers (merge dispatch, the drift check, the task leg); and a wrong verdict from an unproven host can land a red commit.
**Code anchors** verified against main `17fe85747c` (2026-11-25). Main moves fast: cite by symbol and re-locate at implementation time.

**Rulings this PRD executes (Leo, 2026-11-25):**
- The goal is landings per week. Dark-factory landings count most for now.
- Trial on rented capacity before any capital purchase. A rented server anywhere in the world is acceptable: dark-factory, reify and solar-challenge are public, and solar-challenge-platform is headed for open source.
- Python projects go on the rented host first; reify joins as a second step inside the same trial.
- A new host earns trust per host, automatically: three agreeing full-breadth verifies, then trusted. No human step.
- The host verifies as soon as it is ready. Once live enable and disable land, the trial alternates blocks of days with the host on and off.
- The workstation reaches the host over Tailscale, under a tag whose ACL lets the workstation reach the host's ssh port and lets the host reach nothing.
- Adding and removing verify capacity should not need a restart.
- Config that describes only this operator's machines is not checked in or pushed. These projects are meant to become open-source releases that others can run portably.
- During probation the local verdict is acted on in both directions, and a disagreement benches the new host without halting the lane (decisions 4 and 5, approved).

**Standing rulings it stays inside:** never build host-wide verify admission across projects; merge verifies get no cap and no share; one merge-verify slot per host per project; the task cap stays 36; file locks stay held until the merge lands; plan-cap exhaustion never falls back to paid API use.

## 1. Goal

A third machine verifies for the factory, and the factory can tell whether it was worth the money.

What an operator observes once this lands:

- One private file, `$XDG_CONFIG_HOME/orchestrator/verify-hosts.yaml`, names every verify host on this workstation and which projects use it. No repository carries a host name, a user name or a home path.
- `scripts/deploy/provision_verify_host.sh <host-name>` takes a blank Ubuntu 24.04 machine to a verify host, and `scripts/deploy/check_verify_host.py <host-name>` prints one verdict per requirement of the host contract (§6.1).
- `orchestrator verify-host-preflight --runner <name>` runs the project's real full gate on that host through the production dispatch path and prints a structured verdict. It works against `laptop` today.
- Setting `enabled: false` on a host in the roster file and calling `mcp__escalation__reload_config` takes that host out of service without a restart or a commit: `mcp__escalation__get_merge_queue` shows it `draining`, then gone. Setting it back shows it benched for admission, then free. The orchestrator's run id does not change.
- A runner added with `trust: probation` shows `trust_agreements` climbing in `get_merge_queue` as its first verdicts are confirmed locally, becomes `graduated` after three, and stays graduated across a restart.
- `scripts/verify_host_trial_report.py` prints, per on-block and off-block and per project, landings per day and the mechanism measures in §6.5, and one decision line: the relative gain in fleet landings against the relative cost of the host.

## 2. Premise (G6), measured 2026-11-11 to 2026-11-25

Rules are stated with each figure. The study's scratch outputs were in `/tmp/team-cloud-verify-1125/` and are not durable; the figures below are the record.

- **The workstation is saturated.** Week of 11-18: mean 30.5 of 32 threads busy, 93% of 10-minute periods at or above 90% busy (atop over all 168 h; `sar` agrees within 0.1%). Verify scopes are 50% of that CPU (2,587 core-hours), agents 26%, everything else 24%.
- **The laptop is full too.** Same week: mean 13.3 of 16 threads busy, 62% of 10-minute periods at or above 90%. Idle thread-hours were 17% (451 core-hours), and 243 of those fell in the 12% of the time it was under half busy. Its dark-factory merge verifies took a median 35.2 minutes against about 19 uncontended.
- **Local merge verifies are stretched about three times.** Dark-factory `merge_verify` median wall on the workstation: 34.1 minutes in the week of 11-11 (n=178), 74.6 in the week of 11-18 (n=93).
- **Dispatch is held by host pressure, mostly IO and memory.** Dark-factory `dispatch_deferred` rows in the week of 11-18: 1,567, of which `io_some_avg10` 719, `mem_full_avg10` 587, `runqueue_ratio` 261. reify legs peak at a median 29.5 GiB and up to 64 GiB of memory on the workstation (systemd scope `memory.peak`, page cache included).
- **Verify compute is the cheap input.** A dark-factory landing used about 5.4 core-hours of verify (workstation plus laptop scope CPU divided by landings, week of 11-18; contended, so an upper figure). On a server of about $252 a month with 32 threads that is $0.06 to $0.10 per landing. An agent seat costs GBP 150 a month and paid for 92 to 170 landings a week.
- **The decision threshold.** Seven seats are about $1,375 a month. A $252 host adds 18%. On landings per dollar it pays when fleet landings rise by more than 18%, which was about 200 a week on the 1,073 landed in the week of 11-18.

What this PRD does not claim: that the host will clear that threshold. The study's estimate was +100 to +350 landings a week and was not independently reviewed. Two other limits were measured in the same window and are not addressed here: agent allowance (all seven accounts were capped at once from 2026-11-24T21:13Z; about 10% of allowance expired unused over the two weeks, which the allowance-reserve tasks 10430 to 10434 address) and file locks (on 2026-11-25, 306 of 454 ready dark-factory candidates were lock-refused). The trial exists to measure the host's effect with those limits present.

## 3. Sketch of approach

Three parts, each usable without the next.

**Part 1, fresh-host readiness.** What a verify host must provide is written down once, as a provisioning script and a checker that agree by construction (the checker's requirement table is the script's own list). Both are generic: everything specific to a machine comes from a private roster file that lives outside every repository. The parts of the laptop's setup that exist only on the laptop today split accordingly: the PATH wrapper becomes a tracked template, `tzdata-legacy` and the git receive setting become script steps, and the per-host YAML is rendered from the roster. Tests that read unrecorded host state are fixed in the tests, not papered over on the host. A preflight command runs the real gate on a named runner through the production path, so "this host can verify this project" is a command's exit code instead of a session's judgment.

**Part 2, a live roster.** The roster file is the one home of which hosts exist and whether each is in service. A config reload re-reads it and reconciles the running allocator, host pool and quarantine tracker against it: a newly enabled host is admitted benched and joins only after the gate handshake and a clean probe; a disabled host drains its in-flight lease and is then dropped. Each runner carries a trust state. While a host is on probation, every merge verdict it returns is re-run locally before it is acted on.

**Part 3, the trial.** One rented server. Dark-factory, solar-challenge and solar-challenge-platform use it first; reify joins after its one-time host build. The host runs continuously until the live roster lands, then the trial alternates off and on blocks. A report script computes the comparison from the run stores.

## 4. Consumers (G1)

| Mechanism | Consumer |
|---|---|
| The private roster file and its loader | Every orchestrator's runner list; the provisioning script and checker; the trial's toggles |
| `provision_verify_host.sh`, `check_verify_host.py`, the wrapper template and the rendered per-host YAML | The bring-up sittings (θ, ι in §9); `SETUP.md` §12 and `OPERATIONS.md`; any later owned host; anyone else running the factory with a second machine |
| Fixes to host-tied tests | The preflight on a blank host (θ) |
| `orchestrator verify-host-preflight` | The bring-up sittings; the predicate that closes θ and ι |
| Live roster reconcile (`enabled`, per-host `task_slots`) | The trial's off and on toggles (κ1 to κ4); the operator benching a host; adding the rented host without a restart |
| Per-host trust and probation | The rented host's admission in each project that uses it |
| `verify_host_*` events and the `get_merge_queue` host fields | The escalation watchers; the trial report (block boundaries) |
| `verify_host_trial_report.py` | The reading gate (λ) |

## 5. Resolved design decisions

1. **Scope of the first step: dark-factory, solar-challenge, solar-challenge-platform.** reify joins in a second step (Leo). The trial report keeps the two steps apart.
2. **Trust is per host and automatic** (Leo). A host's entry for a project has `trust: probation | trusted`, default `probation`, so a new entry is never trusted by omission. The laptop's entries are written `trusted` when they move into the roster file (the sitting ζ). An inline entry left in a project's yaml, which cannot say, re-earns trust with three confirmed verifies at its next restart.
3. **Three agreeing full-breadth verifies graduate a host** (Leo: "a quick smoke test, but full suite breadth required"). While a host is on probation its merge verifies run at full breadth whatever the project's `merge_verify_breadth`, and so does the local re-run. The count is `verify_host_probation_agreements_to_graduate`, default 3, reloadable; 0 means never graduate automatically.
4. **During probation the local verdict is the one acted on, in both directions.** A green from the host is re-run locally before landing (the existing cross-check, scoped to that host). A red from the host is re-run locally before the task is bounced. Agreement counts toward graduation. This is `corroborate-before-acting` applied to an unproven host: the laptop's first day as a primary host produced four false reds after a passing pre-flight.
5. **A disagreement benches the probation host and does not halt the lane.** The local verdict stands, the host is quarantined in the `divergence` class until an operator clears it, the count resets, and one escalation is filed. This differs on purpose from task 11000's ruling for trusted hosts (escalate only, because a divergence between proven hosts is almost always noise): a new host's disagreements have been real environment differences every time, and an unbenched disagreeing host would spend a local verify on every verdict.
6. **A probation host takes no task verifies.** A false red in a task verify sends an agent after a failure that does not exist. Probation lasts three merge verifies, so the cost is hours.
7. **The roster's one home is a private file outside every repository** (Leo, 2026-11-25: instance-specific config is not checked in or pushed, because these projects are meant to become portable open-source releases). The file is `$XDG_CONFIG_HOME/orchestrator/verify-hosts.yaml`, overridable by `ORCH_VERIFY_HOSTS_FILE`; every orchestrator on the workstation reads its own project's slice of it (§6.3). An absent file means no remote hosts, which is what a fresh clone gets. Benching a host on purpose is `enabled: false` in that file plus a reload: the same mechanism as removal, and it survives restarts because it is config. No second store of operator benches (`one-fact-one-home`). Three things follow: a host's identity is written once instead of once per project; a toggle is one edit to an untracked file, so it dirties no checkout, makes no commit and costs the merge lane no re-verify; and the history of changes is the `config_reload` audit rows and `verify_host_*` events, not git. An inline `verify_runners` block in a project's yaml stays accepted so existing setups keep working, but naming hosts in both places is a config error.
8. **Disabling drains; it does not cancel.** An in-flight verify on a leaving host runs to its verdict and the verdict is honoured. A slot that would be parked on a leaving host is dropped instead, and re-admission always requires the gate handshake and a clean probe, so a stale verify left behind cannot meet a fresh lease.
9. **Identity changes need a restart.** `ssh_host`, `git_remote`, `config_path`, `df_checkout_path` and `gate_checkouts_path` on a live name report `restart_required` per host. To move a host, add it under a new name and disable the old one.
10. **Trust belongs to a machine, not a name.** The admission probe reads the host's `/etc/machine-id`; a change opens a new probation, so a replacement server under an old name starts unproven.
11. **Speculation depth is sized at start from declared runners**, enabled or not. A host declared `enabled: false` before a restart can be enabled live at full depth. A host added beyond the declared count is admitted and used; it will sometimes idle until the next restart, and the reload report says so (`speculation_headroom`). Correctness does not depend on the depth.
12. **Trust evidence lives in the event store** and is rebuilt at start. Graduation is never written back to the roster file. No new persisted file (the 2026-11-12 ruling against a cap ledger applies in spirit).
13. **The per-host YAML is rendered from the private roster file, never tracked.** Each project block in the roster carries a `host_config` map holding that project's host-side keys. The provisioning script renders it to the host's `config_path`, and the checker fails when the installed file differs from a fresh render. The repositories ship only generic code and `config/verify-hosts.example.yaml` with placeholder names. Moving the per-host tunables into the wire spec, so that no file is installed at all, is the better end state and belongs to task 9710 (S16 ζ5b) and task 5746; this PRD does not pre-empt them.
14. **Tests do not depend on unrecorded host state, and nothing tracked names this operator's machines.** A test that reads a file in the operator's home at import, hard-codes a home directory, assumes a time zone without setting it, or waits without a bound is fixed in the test. The provisioning script, the checker and the wrapper take the user, paths and host names from the roster file; none of them contains a literal user name, host name or home path. A genuine platform requirement (Landlock, a user systemd manager, a writable `/var/tmp`) goes in the host contract and is checked by `check_verify_host.py`.
15. **Bring-up is an attended sitting.** Renting, joining the tailnet and adding the ssh alias are Leo's. Everything after runs from the workstation in one interactive session per step, launched from a gate task whose predicate is the preflight's recorded green. Dispatched task agents do not reach the rented host.
16. **The trial runs on, then alternates** (Leo): continuous until the live roster lands and at least five days have passed, then OFF 3 days, ON 3, OFF 3, ON 3. Toggles are dated gate tasks executed by the standing L2 watcher: one edit to the roster file, then a reload of each orchestrator that uses the host.
17. **The decision rule is relative gain against relative cost.** The reading reports fleet landings per day in on-blocks against off-blocks with a block-bootstrap interval, beside host cost as a share of the seat bill. Keeping, buying or stopping is Leo's decision at the reading gate.
18. **Server floor:** 16 cores and 32 threads of Zen 4 or newer, 128 GB of memory, 2 TB or more of local NVMe, hourly or monthly billing without a long term. The $252 figure in §2 is a 16-core Zen 5 server with 2×3.84 TB billed hourly with no setup fee, read 2026-11-25. The pick is made at the rent gate from that day's prices.
19. **No host-wide admission.** Each project's task slots on the host are that project's own number. Their sum is the operator's sizing, recorded in the per-host YAML files and shown by the checker.

## 6. Contracts (H)

### 6.1 Host contract

`check_verify_host.py <host-name> [--json]` reads the host's entry from the roster file and prints one row per requirement: `requirement`, `verdict` (`ok` | `fail` | `warn`), `observed`, `needed`. Exit 0 iff no `fail`. The provisioning script satisfies exactly these rows, and a test asserts that the script's step list and the checker's requirement list are the same table (`no-lockstep-duplication`).

| Requirement | Needed |
|---|---|
| OS and user | Ubuntu 24.04; a non-root user (the roster's ssh user); git identity set |
| Time zone and locale | the same time zone and locale as the dispatching workstation, read from both; `tzdata-legacy` installed |
| Toolchain | git, bash, flock, uv at the pinned version, uv-managed CPython 3.13, node and npm at the major the project's `verify_env_probes` expects |
| systemd | `systemd-run --user --scope true` succeeds over a non-interactive ssh session (linger on) |
| Kernel | Landlock ABI 2 or higher; `/var/tmp` writable |
| Memory | swap present; `MemAvailable` at least the sum of the per-project scope `max_gib` values in the roster's `host_config` maps |
| Wrapper | `/usr/local/bin/orchestrator` equals the tracked template `scripts/deploy/verify-host-wrapper.template.sh` rendered with this host's paths, and honours `ORCH_GATE_BIN` |
| Checkouts | the Dark Factory checkout and each project checkout exist at the roster's paths with `receive.denyCurrentBranch=updateInstead` |
| Per-host YAML | for each project using the host, the file at its `config_path` exists and equals a fresh render of the roster's `host_config` |
| Disk | free space at least the project's stated footprint; a synchronous-write probe reported in ms (`warn` above 20 ms: one rented host measured 126 ms and more than doubled the dark-factory gate) |
| Network | reachable with the dispatcher's own ssh options; the host key is pinned in the workstation's `known_hosts` |

No credential is required on the host, and the checker fails if an agent OAuth token file is present in the verify user's home.

### 6.2 Preflight

`orchestrator verify-host-preflight --runner <name> [--sha <sha>] [--json]` builds the merge-verify spec for the given commit (default: the tip of `main`) with `verify_runner.py::build_merge_verify_spec`, dispatches it to the named runner through the production `RemoteRunner` path (publish, `ensure_gate`, `verify-merge`) at full breadth, and prints `runner`, `sha`, `passed`, `wall_secs`, `ensure_gate_secs`, and per module `failing_test_ids` and the skipped vector. It does not take the orchestrator's host slot: the verify runs in its own ephemeral worktree beside any verify in flight, so on a busy host it is slower, not unsafe, and it refuses a project whose config turns on a persistent merge worktree. It emits no `merge_verify` event, so it never counts as trust evidence or as trial data. Exit 0 iff passed. It writes its JSON to `data/verify-logs/preflight/<runner>-<sha>.json`; the bring-up gates in §9 are resolved by citing those files.

Skip parity: `--reference <runner>` runs the same commit on a second runner and fails if a test is skipped on one and run on the other, except for ids in a tracked list whose every entry names its owning task (`exceptions-owned-or-ratified`; the reify-dependent guards are owned by task 7030).

### 6.3 Roster

**Roster file.** One private file for the workstation, read by every orchestrator on it:

```yaml
hosts:
  <host-name>:
    ssh_host: <ssh alias>
    df_checkout_path: <path on the host>
    gate_checkouts_path: <path on the host>
    enabled: true
    projects:
      <project_id>:
        git_remote: <remote name in that project's repository>
        config_path: <path on the host of the rendered per-host YAML>
        task_slots: <int, optional>
        trust: probation | trusted
        host_config: { <host-side keys for this project> }
```

`orchestrator/src/orchestrator/verify_host_roster_file.py` parses it and returns, for one `project_id`, the same `VerifyRunnerConfig` list an inline block yields today: one runner per host that names the project, with the host's `enabled`. `host_config` is opaque to the orchestrator; only the provisioning script and the checker read it. A file that fails to parse fails the load loudly; at a reload it leaves the running roster untouched and the report says why. A project with both an inline `verify_runners` block and an entry in the file fails the load.

**Config.** `VerifyRunnerConfig` gains `task_slots: int | None` (dispatcher-side task-verify turns on this host; `None` means the global `verify_admission_task_slots`; 0 means merge verifies only) and `trust: 'probation' | 'trusted'` (default `probation`). A module constant partitions the model's fields into identity fields (decision 9) and live fields (`enabled`, `task_slots`, `trust`), and a test asserts the partition is exact. `OrchestratorConfig` gains a validator (runner names unique; none is `local`) and `verify_host_probation_agreements_to_graduate: int = 3`. `verify_runners` joins `config.py::RELOADABLE_FIELDS`; only the reconcile writes it. A reload re-reads the roster file as well as the project yaml.

**Planner.** A pure module `orchestrator/src/orchestrator/verify_host_roster.py`:

```python
HostAction = Literal['admit', 'undrain', 'retire', 'retune', 'refused', 'unchanged']
def plan_host_changes(live, fresh, *, gate_available: bool, draining: frozenset[str]) -> HostPlan
```

Keyed by name. Enabled in `fresh` only: `refused` when no deployed sha exists, `undrain` when the name is draining with unchanged identity, else `admit`. Enabled in `live` only: `retire`. Enabled in both: `refused` on an identity change, `retune` when a live field differs, else `unchanged`. `HostPlan.runners_to_write` keeps the live entry for every refused name.

**Applying.** `harness.py::Harness.reload_config` computes the plan, substitutes `runners_to_write` into the fresh config, runs `apply_reload`, and applies the plan to the task-side pool and the merge worker in the same turn, with no await between plan and apply. The merge lane's single writer is `merge_lane/host_quarantine.py::HostQuarantine`; the worker only delegates.

**Slot and membership states (merge lane).** Membership is a separate dimension from slot state.

| State | Meaning | Exits, and who performs them |
|---|---|---|
| absent | not managed | admitted by the reconcile: free, benched in class `admission` |
| benched `admission` | quarantined, tracked | readmitted by the reprobe sweep or an operator clear once `ensure_gate` passes, the probe is clean and the machine id is read; alarms on the existing unreachable ladder |
| active, free | acquirable unless quarantined | busy on acquire; dropped at once on retire |
| active, busy | lease held | free on release; parked on a failed cancel; draining on retire |
| active, parked | cancel failed | free on a clean probe; dropped on retire |
| draining, busy | not acquirable; lease runs to its verdict | dropped on any release path, including one that would have parked; active again on undrain |
| dropped | both allocator maps and the park epoch popped; tracker, bench and probe records forgotten; open alarms resolved with evidence `host_retired` | absent |

A bench requested for a draining or unmanaged host records nothing and emits `verify_host_bench_ignored`.

**Task side.** `verify_hosts.py::HostPool` admits and retires members. A retired host's gate is kept until its held turns leave; it grants no new turn. `task_verify_dispatch.py::TaskVerifyDispatcher` re-checks membership in the same step as the claim, and the runner for a turn is built at the claim from the entry captured at admit. `HostPool.policy` returns zero slots for a host that is not a member, is quarantined, or is on probation; otherwise the entry's `task_slots`, else the global value.

**Restart dispositions** (`scheduling-state-survives-restart`): membership is rebuilt from the roster file; draining and the admission bench are dropped by construction, because a restart ends every lease and builds remotes free as today; trust is rebuilt from event rows.

### 6.4 Trust

`merge_lane/host_trust.py::HostTrust` holds, per host: the declared trust, the machine id, and the agreement count since the later of the last probation declaration, the last disagreement and the last machine-id change. Effective trust is `trusted` when declared so; else `graduated` when the count has reached the configured number and that number is above 0; else `probation`.

While effective trust is `probation`:

- the host's merge verifies are dispatched at full breadth;
- `merge_lane/post_merge_verify.py` re-runs every verdict it returns on the local host at full breadth and acts on the local verdict;
- agreement emits `verify_host_trust_evidence` with `agree: true` and raises the count; reaching the number emits `verify_host_trust_changed` to `graduated`;
- disagreement emits evidence with `agree: false`, resets the count, quarantines the host in class `divergence`, and files one escalation keyed on the host;
- an inconclusive local run is no evidence.

The global `verify_cross_check_remote_green` keeps its meaning and its halt behaviour for trusted hosts.

At worker start `HostTrust` rebuilds from `verify_host_trust_declared` and `verify_host_trust_evidence` rows across runs, off the event-loop thread, through a read that raises on failure. A failed rebuild emits `verify_host_trust_rebuilt` with `ok: false`, leaves every non-trusted host on probation, and shows `trust_evidence_ok: false` in the snapshot (`no-silent-fail-soft`).

### 6.5 What a reload reports, events, snapshot, trial measures

`reload_config` returns `verify_hosts: [{name, action, disposition, reason, speculation_headroom}]` with `disposition` one of `applied`, `restart_required`, `refused`. The `config_reload` audit row is written for any `verify_runners` edit; today it is lost because the list is not JSON-renderable (the bug task 10441 describes for `usage_cap.accounts`).

Events: `verify_host_admitted`, `verify_host_retiring`, `verify_host_retired` (`drained_secs`, `parked_at_retire`, `resolved_alarms`), `verify_host_retuned`, `verify_host_bench_ignored`, `verify_host_trust_declared`, `verify_host_trust_evidence`, `verify_host_trust_changed`, `verify_host_trust_rebuilt`. Each carries `host`.

`get_merge_queue` `hosts[]` rows gain `membership` (`active` | `draining`), `draining_since`, `trust`, `trust_agreements`, `trust_agreements_needed`, `trust_evidence_ok`, `task_slots`. `quarantine_class` gains `admission`.

`verify_host_trial_report.py --host <runner> --since <date> --host-cost-usd <n> --seat-bill-usd <n>` reads every project's `data/orchestrator/runs.db` and `data/load-samples.db`. Blocks are bounded by `verify_host_admitted` and `verify_host_retired` rows; before the first such row a block starts at the host's first `merge_verify` row. Per block and project it prints: landings per day (distinct task id with a `merge_finalized` done row or a `task_results` done row, counted once); `merge_verify` median wall by runner; share of run-slot time waiting for a task-verify turn; `dispatch_deferred` hours by metric; workstation CPU pressure from the sampler, naming its metrics only through `sampler/src/sampler/metrics.py::METRIC_GROUPS` (`sampler-vocabulary-one-table`); lock-conflict requeues; agent spend per day and allowance expired unused. It prints one decision line: on-block against off-block fleet landings per day, the relative difference with a block-bootstrap interval, and the host's cost as a share of the seat bill. A block with no production row for the host is printed as "no verify ran on this host", never as a zero effect (`readers-prove-their-producer`).

## 7. Boundary-test sketch (H)

Tests extend `orchestrator/tests/test_host_allocator.py`, `test_merge_lane_host_quarantine.py`, `test_verify_hosts.py`, `test_task_verify_host_race.py`, `test_harness_config_reload.py`, `test_merge_queue_host_observability.py`, `test_multihost_verify_integration.py` and `escalation/tests/test_reload_config_integration.py`, using `_verify_host_drive.py::probe_free_orch_config`, `_merge_lane_fakes.py::make_lane` and `_host_health_helpers.py`.

| # | Precondition | Act | Postcondition |
|---|---|---|---|
| 1 | Harness started with a deployed sha; yaml has `laptop` | add enabled `box`; reload | report row admit, applied; snapshot row `box` free, quarantined, class `admission`; pool grants `box` zero task turns |
| 2 | row 1; `ensure_gate` passes and the probe is clean | reprobe sweep | `box` not quarantined; recovery evidence `ensure_gate`; a merge acquire can return `box` |
| 3 | row 1; `ensure_gate` fails | sweep past the alarm threshold | still benched; one L1; no lease ever granted |
| 4 | no deployed sha | add enabled `box`; reload | disposition `restart_required` with a reason; no snapshot row |
| 5 | `laptop` free | disable; reload | row gone; `verify_host_retired`; neither lane offers it |
| 6 | `laptop` busy with a merge lease | disable; reload; complete the verify | while busy `membership: draining` and acquire returns local; the verdict lands; after release the row is gone and `drained_secs > 0` |
| 7 | `laptop` busy and draining | cancel fails | no parked slot, no tracker entry; `parked_at_retire: true` |
| 8 | `laptop` benched unreachable with an open L1 | disable; reload | tracker gone; L1 resolved `host_retired` |
| 9 | task turn granted on `laptop` | disable mid-verify | the verify completes and its sidecar names `laptop` |
| 10 | a waiter queued on `laptop`'s gate | disable | no grant; local wins; nothing raises |
| 11 | grant resolved, claim pending | disable before the claim | the attempt leaves; the local verdict is returned |
| 12 | `laptop` draining and busy | re-enable; reload | undrain; same lease |
| 13 | live `laptop` | change `ssh_host`; reload | per-host `restart_required`; runner and config entry unchanged |
| 14 | `laptop` | `task_slots` 0, then 2 | pool grants 0, then 2 at the next grant; merge lane unaffected |
| 15 | — | duplicate name, or a runner named `local` | load fails; `reloaded: false`; nothing changes |
| 16 | start with `box` declared disabled | enable live | `speculation_headroom >= 0` |
| 17 | `box` on probation, number 2 | two greens from `box`, local re-runs green | two evidence rows; `verify_host_trust_changed` to graduated; the third green is not re-run |
| 18 | `box` on probation | a red from `box`, local re-run red | the task is bounced on the local verdict; one agreeing evidence row |
| 19 | `box` on probation | a red from `box`, local re-run green | the request proceeds on the local green; `box` quarantined `divergence`; one escalation; the lane is not halted |
| 20 | `box` graduated | a new worker over the same event store | snapshot still shows graduated |
| 21 | `box` graduated | the admission probe reads a different machine id | probation, count 0 |
| 22 | the event-store read raises | worker start | `verify_host_trust_rebuilt ok: false`; probation holds |
| 23 | `box` on probation; project breadth scoped | a merge verify dispatched to `box` | the remote leg and the local re-run are full breadth |
| 24 | any `verify_runners` edit | reload | a `config_reload` row exists |

## 8. Pre-conditions and substrate (G3)

Verified on main `17fe85747c`:

- `verify_runners` is restart-only and diffed as one leaf: `config.py::_iter_leaves`, `config.py::apply_reload`, `config.py::RELOADABLE_FIELDS`. `harness.py::Harness.reload_config` has no host hook.
- `RemoteRunner` can be built at reload time: it needs the deployed sha (held on the harness), the code checkout resolver and `git.main_branch`.
- The speculation depth bounds two semaphores and the permit ledger; nothing couples it to the host count, so more hosts than the depth expects is a throughput matter only.
- Hazards the roster must handle, each read in the code: `HostAllocator._is_acquirable` and `_local_fallback_reason` index the slot map by name; `cancel_and_release` parks by name and would resurrect a removed slot; `HostPool._known` raises for an unknown name, and a gate's leave path calls the pool's policy; `remote_runner_factory` searches the live config at build time.
- The local re-run of a remote green exists (`merge_lane/verify_cross_check.py::cross_check_remote_green`, called from `merge_lane/post_merge_verify.py::_cross_check_remote` with the runner in hand). The local re-run of a remote red is built here.
- `verify.py` carries per-module failing test ids and a skipped vector, which the preflight prints.
- Dated and delayed milestone gates exist (`docs/task-authoring.md`, `metadata.milestone`), which the toggle gates use.
- `orchestrator/tests/conftest.py` pins `ORCH_VERIFY_RUNNERS=[]`, so a runner added to the live yaml does not reach the test suite. The roster file needs the same guard: α's loader must not read the operator's real file under test, and a test asserts it.
- A config value that names an external file already exists (`usage_cap.accounts_file`, with `${VAR:default}` interpolation), so reading a second file at load and reload is an established shape. `~/.config/orchestrator/` does not exist on the workstation today; the sitting ζ creates it.

Not verified, carried as tactical questions in §12: whether `runs.db` indexes `events.event_type` (cost of the trust rebuild); the exact fields of the existing recovery events; whether the MCP reload tool's own return survives a pydantic leaf.

Outside the code: the rented server, the tailnet tag and its ACL, and the ssh alias are Leo's to create at gate η.

## 9. Decomposition plan

Sizes follow the overlay bands. Same-file edits are serialized by the dependencies shown.

| Task | What | Signal (leaf) or what it unlocks | Prerequisites |
|---|---|---|---|
| α1 | Roster file and host contract, touching no hot file: `verify_host_roster_file.py` (the parser of §6.3), `config/verify-hosts.example.yaml`, `provision_verify_host.sh`, `check_verify_host.py`, the wrapper template, rendering of the per-host YAML from `host_config`, the shared requirement table and its agreement test, fixes to the known host-tied tests (`scripts/run_vllm_eval.py` reading a key file at import and hard-coding a home path; the unbounded wait in `fused-memory/tests/test_main_ticket_janitor_wiring.py`; time-zone assumptions in `shared/tests/test_invocation_outcome.py`), `SETUP.md` §12 and `OPERATIONS.md` | unlocks η, θ, ζ and α2; `check_verify_host.py <host> --roster <file>` run against the laptop with a hand-written roster prints the table | — |
| α2 | The orchestrator reads the roster: the hook in config loading (restart-tier in this task), the both-sources error, and the guard that keeps the operator's real file out of tests | A test starts an orchestrator config from a roster file and gets the same runner list as from the equivalent inline block; unlocks ζ and γ | α1 |
| ζ | Operator sitting, once α2 is deployed: move the laptop's entries for the four projects that use it out of their tracked yamls into the roster file (`trust: trusted`), with each project's `host_config` copied from the laptop's existing per-host YAML; remove the inline blocks in one commit per project; restart-free where γ has landed, else at the next restart | `check_verify_host.py laptop` prints the table with no `fail`; each of the four orchestrators still shows `laptop` in `get_merge_queue`; none of the four projects' orchestrator yamls declares a runner | α2 |
| β | `orchestrator verify-host-preflight` (§6.2) with skip parity and the owned exception list | The command run against `laptop` at the tip of `main` exits 0 and writes its JSON record | — |
| γ | Live roster (§6.3, §6.5 report, events and snapshot fields; boundary rows 1 to 16 and 24) | On the live dark-factory orchestrator, `enabled: false` on `laptop` in the roster file plus a reload returns a `retire`, `applied` row; `get_merge_queue` shows draining, then gone; re-enabling shows `admission`, then free; the run id is unchanged and `git status` in the checkout is unchanged | α2 |
| δ | Per-host trust (§6.4; boundary rows 17 to 23) | With the rented host admitted on probation, `get_merge_queue` shows `trust_agreements` reaching 3 on real merge verifies and `verify_host_trust_changed` to graduated; the state survives the next restart | γ |
| ε | `verify_host_trial_report.py` (§6.5) | Run with `--host laptop` over 2026-11-11 to 2026-11-25 it prints the laptop's blocks from production rows and the per-project measures | — |
| η | Operator gate (Leo): rent the server to the floor in decision 18, join it to the tailnet under the restricted tag, add the ssh alias | unlocks θ | α1, β |
| θ | Bring-up sitting, Python step: add the host to the roster file with dark-factory, solar-challenge and solar-challenge-platform blocks (`trust: probation`); provision, check, preflight green for each; enable by reload when γ has landed, else at the next restart | Three preflight records with `passed: true`; the host's first `merge_verify` rows in each project. The gate is resolved by citing those records | η |
| ι | Bring-up sitting, reify step: reify's host build on the same server, reify's block in the roster file, preflight green through reify's gate | reify's preflight record and its first `merge_verify` row on the host | θ |
| κ1 to κ4 | Dated toggle gates: OFF, ON, OFF, ON at three-day spacing; each is one edit to the roster file and one reload per orchestrator that uses the host, executed by the standing L2 watcher | The matching `verify_host_retired` or `verify_host_admitted` row in every project that uses the host | γ, δ, θ; κ1 not before five days after θ |
| λ | Reading gate (Leo's decision): run ε's report over the trial, attach it, rule keep, buy or stop | The report with at least two on-blocks and two off-blocks carrying production rows | κ4 plus three days; ε |

α2, γ and δ all edit `config.py`, so they run in that order; γ and δ also share `harness.py`, `merge_lane/host_quarantine.py` and `merge_lane/worker.py`. α1 and β touch none of those files. α is split in two for that reason: `config.py` is one of the most contended files in the repository (168 open tasks named it on 2026-11-25, and a pinned task's park reserves it), and the bring-up needs only α1.

The trial does not wait for α2, γ or δ. The rented host is never written into a tracked file either way: until α2 is deployed, the sitting gives each orchestrator its runner list through its systemd unit's environment (`ORCH_VERIFY_RUNNERS`, which already overrides the yaml because environment settings rank above the file in `config.py::OrchestratorConfig.settings_customise_sources`), which is private and untracked; the provisioning script and checker read the roster file from the start. Once α2 is deployed the sitting ζ makes the roster file the orchestrators' source and removes the environment entries. If θ completes before γ, the host is enabled at the next restart with the global cross-check left off and `trust: probation` inert, and the sitting hand-runs three preflights at successive tips of `main` as the smoke test. When δ lands, the entry's probation takes effect at the following restart.

## 10. Cross-PRD and cross-task seams (G4)

| Other work | Direction | Seam | Owner | Status |
|---|---|---|---|---|
| `plans/review-all-s4-verify-run-context-and-runner-contract-prd.md`, tasks 9137 (θ) and 9163 (ο) | they restructure what γ edits | `HostAllocator` states and where quarantine lives | S4 owns the structure; this PRD adds membership and draining inside today's structure, and 9137 and 9163 carry them when they move | wire 9137 and 9163 to depend on γ at decompose |
| `plans/review-all-s16-orchestrator-config-shape-defaults-and-routing-prd.md`, task 9710 (ζ5b) | it moves verify-host knobs into a submodel | field paths of `verify_runners`, `task_slots`, `trust` | S16 owns the shape; this PRD adds fields in today's shape | wire 9710 to depend on δ |
| task 5746 | consumes | a per-host leaf on `VerifyRunnerConfig` | 5746 | independent; same file as γ |
| tasks 6775, 6776 | they edit `HostAllocator` | lease identity; module extraction | theirs | after γ |
| task 9050 | edits `verify_hosts.py` | battery-based admission | 9050 | independent; ordered by lock |
| task 10441 | produces | JSON-safe `config_reload` audit | γ delivers the general fix; 10441 keeps the `usage_cap.accounts` case if anything remains | re-read at decompose |
| tasks 11000, 11001 (pinned) | they change drift handling | escalate-only divergence; an asynchronous second leg | theirs; δ does not use drift checks as trust evidence | no dependency |
| task 11062 | consumes | the lane's restart-state register | 11062; γ and δ declare their dispositions (§6.3) | whichever lands second adds the entries |
| task 7030 | produces | a pinned reify checkout on verify hosts | 7030 owns the skip-parity exceptions β lists | not a prerequisite |
| task 4196 | produces | cleanup of leaked worktrees on hosts | 4196 | hygiene is by hand until it lands (`scripts/verify_host_hygiene.py`) |
| reify, solar-challenge, solar-challenge-platform repositories | consume | each loses its inline `verify_runners` block; nothing is added | the sitting ζ makes that one commit per project | — |
| `usage_cap.accounts_file` (`config/usage-accounts.yaml`, tracked) and each project's `dark-factory-orchestrator.yaml` | same concern | other instance-specific config still checked in | not this PRD; noted for Leo in §12 question 7 | — |

## 11. Out of scope

- More than one merge-verify slot per host.
- Any admission that sums load across projects on a host.
- Hosts created per verify, or a pool that starts and stops by queue depth.
- Buying hardware. The reading gate may rule on it.
- Moving per-host tunables into the runner entry and the wire spec (tasks 9710, 5746).
- The allowance reserve and file-lock contention, measured in §2 and owned elsewhere.
- Changing drift-check cadence or what a divergence between trusted hosts does.

## 12. Open questions (tactical)

1. **Initial slot numbers on the rented host.** Suggested: dark-factory 3 task turns at pytest width 8 with one module at a time for merges; each solar project 1 task turn; reify 1. Tune during the first on-block. Decide in θ and ι.
2. **The synchronous-write threshold.** 20 ms is a placeholder between the one good and the one bad host measured. Decide in α from the laptop's and the rented host's readings; it only warns.
3. **Block length.** Three days is a guess at the shortest block that averages over the daily account resets. Decide at κ1 from the first on-period's day-to-day spread; λ may order further blocks.
4. **Cost of the trust rebuild.** If `events.event_type` is not indexed, bound the scan by the earliest probation declaration. Decide in δ.
5. **Admission probe timing.** The next reprobe sweep (up to 120 s) or an immediate probe at reload. Decide in γ.
6. **Reloading every orchestrator that uses a host.** By hand through each project's escalation endpoint, or a small `orchestrator verify-host reload` helper that walks the roster. Decide in γ.
7. **Other instance config still tracked.** The accounts file and the per-project orchestrator yamls also describe this operator's setup. Moving them out is a separate piece of work under the same ruling; this PRD only stops adding to it. For Leo to schedule.
8. **Keeping the roster file safe.** It is untracked by design. Whether it lives in a private repository of Leo's or is only backed up is his call; the event store keeps the history of changes either way.

## 13. Gate notes

- **G1:** every mechanism has a consumer inside this PRD (§4).
- **G3:** §8. The one capability built rather than found is the local re-run of a remote red.
- **G4:** §10. No reciprocal ownership; S4 and S16 keep their structure and shape, and are wired to follow.
- **G5:** B+H; §6 and §7.
- **G6:** the 18% threshold is a decision rule derived from §2's prices, not a task's pass criterion. The number 3 is Leo's. The 20 ms threshold is provisional and only warns. θ's green preflight on a blank host depends on fixes not yet known; decision 15 makes that an attended step and α fixes the known ones first.
- **G7 (advisory walk):** hits found and cleared in the design: the trust rebuild runs off the loop thread (`loop-thread-occupancy-bounded`); the rebuild read raises on failure (`no-silent-fail-soft`); a disagreeing probation host is benched so it cannot burn a local verify per verdict (`storm-escape-required`); probation with the number set to 0 is unbounded by choice and visible in the snapshot (`holds-owned-and-bounded`).
- **G7, added with decision 7:** a roster file that fails to parse must not read as "no hosts" (`no-silent-fail-soft`): the load fails, and a reload keeps the running roster and reports the error.
- **META:** yes. Leo approved decisions 4 and 5 on 2026-11-25 and replaced decision 13's tracked per-host file with the private roster file (decisions 7, 13, 14).
