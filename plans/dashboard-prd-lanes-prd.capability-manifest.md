# Capability manifest — dashboard PRD view: lanes by need, chain attribution, honest progress

Binds each task's asserted capabilities to evidence, mechanizing G3 + G6.
Machine-readable twin: `dashboard-prd-lanes-prd.capability-manifest.yaml`.
PRD: `plans/dashboard-prd-lanes-prd.md`.
As of main `f6e7ea4b3a`, 2026-10-09 (the PRD's own anchors were verified on its parent `99ad2d7ddb`; no code moved between them).

**Verdict summary: 45 bindings, all PASS; 28 mechanical checks, zero polarity findings at `main`.**

The 27 grep/path checks were linted with `shared.delivered_check_polarity.lint_delivered_checks`
against `main`, using each task's declared `metadata.files`. The linter skips script checks, so
the one script check (α) was run by hand: it exits 1 for `fields` and 0 for the existing
`statuses`. All 28 fail today, as required.
The DAG has no inversion:

- α → γ1 → γ2 → δ → ζ
- β → γ2
- β → ε1–ε4 → ζ
- out-of-batch premises: 5276 → β, and 6524, 6527 → γ2.

## Filed batch (2026-10-09, committed pending in one `commit_planning`)

| Label | Task | Depends on | Mechanical checks copied |
|---|---|---|---|
| α | 6581 | — | 1 (script) |
| β | 6582 | 5276 | 5 |
| γ1 | 6583 | α | 8 |
| γ2 | 6584 | γ1, β, 6524, 6527 | 6 |
| δ | 6585 | γ2 | 3 |
| ε1 | 6586 | β | 1 |
| ε2 | 6588 | β | 2 |
| ε3 | 6590 | β | 1 |
| ε4 | 6591 | β | 1 |
| ζ | 6592 | δ, ε1–ε4 | — (human gate) |
| follow-up (no label) | 6593 | γ1 | — |

Also wired: 4390 → α. Cancelled as superseded by δ: 6336 and 6013.
`commit_planning` stamped all ten labels; it reported no missing labels and no errors.

## Substrate findings that shaped the bindings

| Capability | Status at decompose | Consequence |
|---|---|---|
| PRD §7 symbols | **CONFIRMED**, all of them, by a substrate seat on main. One wording drift: `dashboard/src/dashboard/data/tasks.py::_shape_task` turns a missing title into `''` but passes `status` through as `None` | γ1's brief states the drift exactly |
| `get_tasks` has no `fields` | **CONFIRMED**: `fused-memory/src/fused_memory/server/tools.py::get_tasks` takes `project_root, tag, page_size, offset, statuses`, builds the full list, then slices | α's check is the parameter itself |
| Stamper write-back | **CONFIRMED** `yaml.safe_dump(raw, sort_keys=False)` in `manifest_stamping.py::_stamp_capability_manifests_impl` today. Task 5276 (in progress) rewrites it; its files add `manifest_sidecar_text.py`, which suggests a task_id-lines-only text stamp | β depends on 5276 and re-verifies "the field survives a stamp" against what landed |
| Deferral record and `needs_human` | **NOT on main**: `shared/src/shared/task_deferral.py` is task 6524 (in progress) | γ2 depends on 6524 |
| Holder liveness | Task 6527 (pending) computes it per snapshot **inside** `active_tasks.py` and names no function | γ2 depends on 6527 and extracts the read into its own access module (PRD §4.8) |
| `TestThePrdBoxCount` (task 6013) | Still present in `dashboard/tests/test_tab_tasks_terminal_window.py`. 6013's other half (`countMayUndercount` in the retired list) is **already on main** | 6013 is cancelled as superseded by δ, which rewrites that class |
| Open-work PRD set (ε tranches) | Recounted read-only under §4.1's rule (appendix), at 2026-10-09T16:01Z and again at 17:04Z just before filing; both reads gave the same 74 tracked keys, against 77 at authoring. At 17:04Z 1,619 tasks were open and 1,067 were unattributed (1,091 at authoring) | The ε tranches are fixed from that set plus this PRD's own key: 75 keys cut 19/19/19/18. The final lists are in each ε task's description |

## Per-leaf bindings

Producer/consumer bindings name the task label; all producers are upstream of their consumers.

| Leaf | Capability | Evidence | Check |
|---|---|---|---|
| α | `get_tasks(fields=…)` declared and forwarded | producer α; absent on main | script `scripts/check_method_param_wiring.py --file …/server/tools.py --function get_tasks --param fields --forwards-to get_tasks` |
| α | projection row shape; invalid-entry rejection | α's own real-store tests (boundary rows 1–3) | manual |
| β | `signal_leaves` on `CapabilityManifestDoc` | producer β; `extra='forbid'` today | grep in `capability_manifest.py` |
| β | `sidecar_path` one home | producer β | grep `^def sidecar_path(` |
| β | stamper uses `sidecar_path` | producer β (wired) | grep in `manifest_stamping.py` |
| β | field survives a stamp | premise 5276, upstream; boundary row 5 | manual |
| β | drift audit reports signal bindings | producer β | grep in `audit_manifest_descriptor_drift.py` |
| β | `/prd` decompose authors the field | producer β | grep in `skills/prd/references/decompose-mode.md` |
| γ1 | bulk terminal link fields | α upstream; γ1's lineage datum | grep `def acquire_lineage(` |
| γ1 | chain attribution | producer γ1 | grep `^def attribute(` in `prd_attribution.py` |
| γ1 | coalescing has one home | producer γ1 moves `_coalesce_prd` | grep **absent** `def _coalesce_prd(` in `active_tasks.py` |
| γ1 | `Milestone.fired` | producer γ1 | grep `def fired(` in `task_metadata.py` |
| γ1 | ready predicate, ready age | producer γ1; `pending_since` populated on 1,058 of 1,062 ready (field-population PASS) | grep `def is_ready(` |
| γ1 | rule per figure | producer γ1 | grep `^FIGURE_RULES` |
| γ1 | `/prds` serves `PRD_VIEW` | producer γ1 | grep in `api/prds.py` |
| γ1 | access-path grant | producer γ1 (INV-12, ratified PRD §4.12) | grep in `test_datum_access_paths.py` |
| γ1 | exact per-PRD counts, census identity | `validate_prd_view` + boundary rows 9, 12 | manual |
| γ2 | lane rules | producer γ2 | grep `^LANE_RULES` |
| γ2 | scheduler state access datum | producer γ2 (extraction) | grep `def acquire_scheduler_state(` |
| γ2 | signal-mark reader | β schema upstream; ε data upstream of ζ | grep `load_capability_manifest` in `prd_signal.py` |
| γ2 | runs.db activity | `task_started` live (9,954 rows) | grep `task_started` in `task_activity.py` |
| γ2 | `PRD_CONTENTION` | producer γ2 | grep in `api/prds.py` |
| γ2 | Parked reads the deferral record | 6524 upstream | grep `needs_human` in `prd_view.py` |
| γ2 | holder liveness | 6527 upstream; γ2 extracts | manual (module name is the implementer's) |
| γ2 | pending-L2 view | `CorpusRecord` level/status/task_id on main | manual |
| δ | PRD boxes retired | producer δ (7 references today) | grep **absent** `groupTasksByPrd` under `static/` |
| δ | lanes page exists | γ2 upstream; filename fixed by PRD §4.12 | path `static/redux/prd_lanes.jsx` |
| δ | data registry carries `PRD_CONTENTION` | producer δ | grep in `data.js` |
| δ | rules on hover | γ1 `FIGURE_RULES` upstream; boundary row 20 | manual |
| ε1–ε4 | tranche completion note | producer εN | path `plans/dashboard-prd-lanes-prd.signal-backfill-epsilonN.md` |
| ε1–ε4 | labels named from PRD text | β upstream; reading judgment | manual |
| ε2 | this PRD names ζ | producer ε2 | grep `^signal_leaves:` in this sidecar |
| ζ | live checks (a)–(f) | every producer upstream | manual (human gate) |

## Decompose decisions

1. **ε's "completion note" is a committed file**,
   `plans/dashboard-prd-lanes-prd.signal-backfill-epsilonN.md`, one per tranche. That makes
   it checkable (a `path` check gating ζ) and gives ζ (e) a source for "ε's named count".
2. **ε tranches** are cut from the open-work recount ordered by key, 19/19/19/rest. This PRD's
   own key is included, and ε2 holds it. The final lists are in each ε task's description.
3. **Keys outside ε.** The recount at filing also found `docs/prds/verify-retry-failed-only.md`,
   named by task 6575, which was filed minutes before this decompose. That path is **not a file on main**. It is
   left out of ε, since there is nothing to read. The view will show it with its key as the title and
   "no sidecar" as the reason.
4. **γ2's liveness extraction** is conditional. The brief extracts the read only if 6527 left it
   inside `active_tasks.py`, which 6527's brief does.
5. **The PRD's figure of ~13 files for γ1** becomes 16 declared files, because the
   12 moved tests, the budget test and the shared `Milestone` test are listed explicitly. That
   crosses the 15-file review trigger. Accepted: the code is four new modules and two small edits.
6. **Companions** (PRD §9):
   - 4390 → α edge plus a dated note;
   - 6336 and 6013 cancelled, citing δ;
   - dated notes on `plans/dashboard-one-datum-one-path-prd.md` (decision 8 and two
     out-of-scope bullets) and `plans/dashboard-taskgraph-legibility-prd.md` (Part 2);
   - one low follow-up repointing `Scheduler._milestone_time_gated` at `Milestone.fired`,
     filed against the PRD with no plan label.

## Fresh review (critic seat, before filing)

A fresh opus reviewer read the drafted batch and this sidecar against the PRD. It raised 22
findings: one blocker, seven major and fourteen minor. All were integrated except one, which
was declined. The ones that changed a binding or an edge:

- **Blocker: α's original check.** It was a file-scoped grep for the annotation
  `fields: list[str] | None`. `get_tasks(statuses: Any = None)` sets the house precedent of
  typing `Any` so that INV-2 rejection works, so an implementer following it would deliver the
  contract and leave the check red forever, holding γ1, 4390 and everything downstream. It is
  replaced by the function-scoped, annotation-agnostic param-wiring script.
- **δ's page check.** It grepped `PRD_VIEW` inside `prd_lanes.jsx`, which would stay red if the
  view arrives as a prop. It became a `path` check.
- **Brief changes, no change to any binding:**
  - ε checks bindings with read-only sqlite, because dispatched roles hold no `get_task`;
  - γ1 gets an exact validator subset and absent-field list, the `Milestone.fired` tz
    normalisation and call-site parse rule, a `wait_for`-bounded lineage refresh with a
    hung-MCP sweep entry, and a GUARDED name for its access grant;
  - γ2 inputs sit inside the `/prds` budget, and γ2 gets provenance, rules and activity for
    every new figure, plus "a malformed milestone is not a hold";
  - δ gets its remaining pins, and the segment vocabulary is served (INV-5);
  - F keeps the scheduler's malformed-milestone WARNING.
- **Declined: an optional restructure.** It would move `Milestone.fired` into the follow-up F
  and make γ1 depend on F. That would put the orchestrator's `scheduler.py` lock on γ1's
  critical path, against PRD §9's assignment of `fired` to γ1. Instead, γ1's brief now tells it
  to re-read `task_metadata.py`'s landed tail, which tasks 6524 and 6558 restructure.

## Appendix — the open-work recount (ζ (a) reuses it)

Read-only. Run `python3 -I recount.py --db /home/leo/src/dark-factory/.taskmaster/tasks/tasks.db`
(not the 0-byte `.taskmaster/tasks.db` decoy). It implements PRD §4.1 exactly and is independent
of the dashboard code, by construction: it was written before γ1.

```python
"""Read-only recount of open-work PRD keys under plans/dashboard-prd-lanes-prd.md §4.1.

Usage: python3 recount.py [--db TASKS_DB] [--root REPO_ROOT]  → JSON on stdout.
"""
import argparse
import json
import re
import sqlite3
from collections import Counter
from datetime import datetime, timezone

MAX_HOPS = 8
TERMINAL = {"done", "cancelled"}
PRD_FIELDS = ("prd_path", "prd", "prd_ref")
LINK_FIELDS = ("spawned_from", "x_recovered_from_task", "cross_repo_refile_of")


def load_tasks(db_path):
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    rows = conn.execute("select id, status, metadata from tasks where tag='master'").fetchall()
    conn.close()
    tasks = {}
    for task_id, status, raw in rows:
        meta = json.loads(raw) if raw else {}
        tasks[task_id] = (status, meta if isinstance(meta, dict) else {})
    return tasks


def own_prd(meta, root):
    for field in PRD_FIELDS:
        value = meta.get(field)
        if isinstance(value, str) and value.strip():
            key = re.split(r"[#§]", value, maxsplit=1)[0].strip().removeprefix("./")
            key = key.removeprefix(root.rstrip("/") + "/")
            return (None, "foreign_prd") if key.startswith("/") or not key else (key, None)
    return None, None


def parse_link(value):
    if isinstance(value, int) and not isinstance(value, bool):
        return None, value
    if isinstance(value, str) and value.strip().isdigit():
        return None, int(value.strip())
    if isinstance(value, str) and re.fullmatch(r"[^:\s]+:\d+", value.strip()):
        return "cross_project_link", None
    return "unparseable_link", None


def attribute(task_id, tasks, root):
    seen, current, hops = set(), task_id, 0
    while True:
        seen.add(current)
        meta = tasks[current][1]
        key, foreign = own_prd(meta, root)
        if key or foreign:
            return key, hops, foreign
        links = [f for f in LINK_FIELDS if meta.get(f) not in (None, "")]
        if not links:
            return None, hops, "no_link" if hops == 0 else "root_without_prd"
        reason, target = parse_link(meta[links[0]])
        if reason:
            return None, hops, reason
        if target not in tasks:
            return None, hops, "missing_target"
        if target in seen:
            return None, hops, "cycle"
        if hops == MAX_HOPS:
            return None, hops, "depth_exceeded"
        current, hops = target, hops + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="/home/leo/src/dark-factory/.taskmaster/tasks/tasks.db")
    parser.add_argument("--root", default="/home/leo/src/dark-factory")
    args = parser.parse_args()
    tasks = load_tasks(args.db)
    open_ids = [t for t, (status, _) in tasks.items() if status not in TERMINAL]
    per_key, dark = {}, Counter()
    for task_id in open_ids:
        key, hops, reason = attribute(task_id, tasks, args.root)
        if key is None:
            dark[reason] += 1
            continue
        counts = per_key.setdefault(key, {"direct": 0, "chain": 0})
        counts["direct" if hops == 0 else "chain"] += 1
    print(json.dumps({
        "as_of_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "open": len(open_ids),
        "dark_matter": {"total": sum(dark.values()), "by_reason": dict(dark)},
        "prd_keys": dict(sorted(per_key.items())),
    }, indent=1))


if __name__ == "__main__":
    main()
```
