# Per-project cgroup slice topology

This file is the single normative home of the per-project slice contract, for
dark-factory and for reify. Both repos cite it and neither restates it. It was
assigned to task 3394 by `plans/load-throttle-harmonisation-prd.md` D5. The
PRD's §6.3 table stays the home of the `/proc/self/cgroup` parity fixtures.

## Name

- A project's slice is `df-<project_id>.slice`, where `<project_id>` is
  `OrchestratorConfig.fused_memory.project_id`. The rule has one
  implementation: `shared/src/shared/cgroup_slice.py::project_slice_name`.
- The id is lower-cased, and every character outside `[a-z0-9_]` becomes `_`.
  systemd derives a slice's parent by stripping its last `-` segment, so
  `df-my-proj.slice` would silently nest under a bogus `df-my.slice`.
  Flattening keeps exactly one project level under `df.slice`. An empty id is
  an error.
- The name is never derived from
  `orchestrator/src/orchestrator/verify.py::_scope_tag_for`. That tag is
  dash-sanitised and names verify scope UNITS only.

## Topology

```
user@UID.service/
├── app.slice/                              desktop apps, interactive terminals
└── df.slice/
    └── df-<project_id>.slice/
        ├── orchestrator-<project>.service  the daemon and its ungoverned agents
        └── df-verify-<tag>-<uuid>.scope    one per verify leg or segment
```

- The service is placed by `Slice=df-<project_id>.slice` in its committed unit,
  `scripts/orchestrator-<project>.service`.
  `tests/scripts/test_orchestrator_service_files.py::test_project_unit_runs_under_its_project_slice`
  pins each value to the name rule through the unit's `--config` root.
- Ungoverned agents are children of the daemon. They inherit its service
  cgroup, and nothing places them separately.
- Verify scopes are placed by `systemd-run --user --scope --slice=<slice>`
  (`orchestrator/src/orchestrator/verify.py::_scope_launch_argv`), alongside
  their per-role `-p CPUWeight=` and memory properties.

## How a child finds its project slice

Walk the `0::` path in `/proc/self/cgroup` leaf-upward to the segment EQUAL to
`project_slice_name(project_id)`. Equivalently, that is the `df-*.slice`
segment directly under `df.slice`.

It is NOT the nearest `df-*.slice` ancestor. Every governed sub-slice
(§Governed sub-slices) also matches `df-*.slice` and sits nearer the leaf, so
"nearest" would resolve `df-reify-governed-agents.slice` and read the wrong
`cpu.pressure`. This supersedes the "nearest `df-*.slice`" wording PRD §6.3
once carried; §6.3 row 4 is the governed shape.

The python side is `shared/src/shared/psi.py::resolve_own_cgroup`, and reify's
`@own-slice` is the bash side, which must apply the same exact-segment rule
(reify tasks 9998 and 7335). Their shapes are the PRD §6.3 table. No
environment variable carries the slice name, because it would be a second
discovery mechanism for the same fact.

## Governed sub-slices

Sub-slices for governed work nest under the project slice by dash-naming. For
example, `df-reify-governed-agents.slice` sits under `df-reify-governed.slice`,
which sits under `df-reify.slice`. That keeps reify's C-G2 true across the seam
between the two repos. Re-homing `scripts/lib_cgroup.sh::cgroup_role_slice`
there is reify task 9998.

## Slice weight

Startup step `1c1i`
(`orchestrator/src/orchestrator/project_slice_weight.py::ensure_project_slice_weight`)
runs `systemctl --user set-property --runtime <slice> CPUWeight=100`. It acts
only when both of these hold:

- (i) `Harness.run` got `apply_project_slice_weight=True`. Only
  `orchestrator/src/orchestrator/cli.py::run`, every unit's ExecStart, passes
  it. This opt-in is what keeps tests off the production slice: agents are
  children of the daemon and share its service leaf, and so does every pytest
  they run, `Harness.run` tests included.
- (ii) The daemon's own leaf cgroup is a `.service` directly under its project
  slice. This is a ROLLOUT check (was the unit reinstalled with `Slice=`?), not
  an identity check. When it fails, the step logs one WARNING and changes
  nothing.

About the property itself:

- 100 is systemd's default. What matters is that the property is present: a
  fresh slice has no `cpu.weight` until some CPU property is set on it.
- `--runtime` writes the drop-in under `$XDG_RUNTIME_DIR`, so it ends with the
  user manager. The next daemon start sets it again.
- Choosing per-project values is dark_factory task 9984.

## Consequences

- **(a) Two layers of weight.** Task 5205's per-role scope weights (merge 100,
  task 33, background 10) now split ONE project's slice among its verify scopes
  and its daemon. The slice weights set the split between projects.
  Re-deriving both layers is dark_factory task 9984.
- **(b) Policy: factory work against the desktop.** `df.slice` is a sibling of
  `app.slice` under `user@UID.service`. Under contention, all factory work
  together therefore shares one weight against the desktop and interactive
  terminals. That is deliberate, and consistent with Leo's 2026-08-03 ruling
  that interactive work is privileged. Interactive terminal processes are
  deliberately NOT placed in project slices.

## Laptop and remote verify hosts

Verify scopes there land under `df-<project_id>.slice` too, because the name
comes from the requesting project's config. No daemon runs there to set that
slice's weight, so it keeps systemd's default.

## Rollout

The committed units take effect only when an operator reinstalls them with
`scripts/setup-host.sh` and restarts each orchestrator (dark_factory task
9985). Verify scopes move as soon as the code lands, because their placement is
code. Until the reinstall, a project's verify scopes sit under its slice while
its daemon is still in `app.slice`. In that window step `1c1i` logs its
WARNING, naming the daemon's actual cgroup, and the unit parity checker
reports the installed copies as drift.

Two more effects last until the reinstall:

- The sampler's leaf set switches to slices that hold only verify work.
  `sampler/src/sampler/metrics.py::discover_pressure_cgroups` falls back to
  `orchestrator-*.service` leaves only when no `df-*.slice` exists, so from the
  first verify scope on it stops sampling the daemons and their agents. Task
  5220's `df-*.slice` calibration therefore waits on task 9985 as well as on
  this one.
- `df.slice` holds only verify work, at systemd's default weight, against
  `app.slice`, which still holds every daemon, its agents and the desktop.
  That is not the verify-versus-agent split task 5206 tuned, when verify scopes
  and daemons were siblings in `app.slice`.
