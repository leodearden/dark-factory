"""OS-sandbox rollout soak predicate (PRD γ1/γ5).

The real implementation behind ``scripts/check_sandbox_soak.sh`` — the
``before_done.kind='predicate'`` check consumed by the γ5 soak gate. The soak
is GREEN (exit 0 → task done) iff ALL of:

  (a) >=10 DISTINCT tasks that have a ``sandbox_applied`` event reached
      ``done``;
  (b) the containment probe report is tracked on main at
      ``docs/sandbox-containment-probe-report.md``;
  (c) 0 sandbox-attributable blocks.

Everything is derived from STRUCTURED queries over the event store + task
records — never transcript-grep (INV-2). See the module design in the task
2910 plan and ``plans/os-sandbox-worktree-containment-prd.md``.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from orchestrator.event_store import EventType

# Canonical location of the containment probe report (γ4 commits it to main).
PROBE_REPORT_PATH = "docs/sandbox-containment-probe-report.md"

# PRD-D6 spec constant: the soak requires >=10 distinct done sandboxed tasks.
MIN_DONE_DEFAULT = 10

# Arm-2 attribution token: an out-of-set path denial (D9 errnos) surfaced
# through a structured escalation summary — matched on the errno token only
# (never the word "sandbox"), keeping arm 2 precise and non-overlapping with
# arm 1.
_ERRNO_RE = re.compile(r"\b(EACCES|EROFS)\b")


class SoakInputError(Exception):
    """A soak input could not be measured — a missing/unreadable store, a bad
    argument, or a git error resolving the report. Distinct from an exit-1
    not-yet-green verdict: it maps to the CLI's exit 2 (usage/infra error) so an
    infra failure is never misread as a soak verdict. Loud-over-silent (project
    norm): a mispointed path must not degrade to an empty '0 tasks, keep
    waiting' for the entire >=3-day soak.
    """


@dataclass
class SoakVerdict:
    """Structured verdict of the soak predicate.

    ``ok`` is True iff all three soak conditions hold. ``reason`` is a single
    human-readable line (the sole stdout line the CLI prints). ``metrics``
    carries the derived counts for observability / debugging.
    """

    ok: bool
    reason: str
    metrics: dict


def _sandbox_attributable_blocks(task_status, sandbox_unavailable_task_ids, escalations):
    r"""Return the sorted task_ids that are a sandbox-attributable block.

    A block is sandbox-attributable iff the task's CURRENT status is ``blocked``
    AND either:
      * arm 1 — its task_id has a ``sandbox_unavailable`` event (a fail-closed
        refusal), or
      * arm 2 — it has an ``escalation_created`` event whose structured summary
        matches ``\b(EACCES|EROFS)\b`` (a denial on an out-of-set path).

    Correlation is strictly by task_id — both event types carry task_id as a
    first-class column, so no fuzzy timestamp window is needed (INV-2). A
    ``sandbox_unavailable`` event for a task that later recovered (not currently
    ``blocked``) is not counted.
    """
    unavailable = {str(t) for t in sandbox_unavailable_task_ids}
    errno_task_ids = set()
    for esc in escalations:
        tid = esc.get("task_id")
        if tid is None:
            continue
        if _ERRNO_RE.search(esc.get("summary") or ""):
            errno_task_ids.add(str(tid))
    blocked = {str(t) for t, status in task_status.items() if status == "blocked"}
    attributable = {t for t in blocked if t in unavailable or t in errno_task_ids}
    # Numeric ids sort numerically (5 before 10); non-numeric ids fall after.
    return sorted(
        attributable, key=lambda t: (0, int(t)) if t.isdigit() else (1, t)
    )


def evaluate_soak(
    sandbox_applied_task_ids,
    task_status,
    sandbox_unavailable_task_ids,
    escalations,
    report_present,
    min_done=MIN_DONE_DEFAULT,
):
    """Pure verdict function over already-read structured inputs.

    Args:
        sandbox_applied_task_ids: iterable of task_id (str) with a
            ``sandbox_applied`` event (condition-a numerator candidates).
        task_status: mapping ``{task_id(str): status(str)}`` from tasks.db.
        sandbox_unavailable_task_ids: iterable of task_id (str) with a
            ``sandbox_unavailable`` event (condition-c arm 1).
        escalations: list of ``{task_id, summary, category}`` dicts
            (condition-c arm 2).
        report_present: bool — is the probe report tracked on main (condition
            b).
        min_done: the >=N distinct-done bound (PRD-D6, default 10).

    Returns:
        SoakVerdict.
    """
    applied = {str(t) for t in sandbox_applied_task_ids}
    done_count = sum(1 for t in applied if task_status.get(t) == "done")

    # Condition (c) — sandbox-attributable blocks (PRD Open Q2).
    attributable = _sandbox_attributable_blocks(
        task_status, sandbox_unavailable_task_ids, escalations
    )

    clauses: list[str] = []
    if done_count < min_done:
        clauses.append(
            f"only {done_count}/{min_done} distinct sandboxed tasks reached done"
        )
    if not report_present:
        clauses.append(
            f"containment probe report absent on main ({PROBE_REPORT_PATH})"
        )
    if attributable:
        clauses.append(
            f"{len(attributable)} sandbox-attributable block(s) "
            f"[{', '.join(attributable)}]"
        )

    metrics = {
        "done_count": done_count,
        "min_done": min_done,
        "sandboxed_task_count": len(applied),
        "report_present": bool(report_present),
        "attributable_block_count": len(attributable),
        "attributable_block_task_ids": list(attributable),
    }

    if clauses:
        return SoakVerdict(
            ok=False, reason="FAIL — " + "; ".join(clauses) + ".", metrics=metrics
        )
    return SoakVerdict(
        ok=True,
        reason=(
            f"PASS — sandbox soak green: {done_count}/{min_done} distinct "
            "sandboxed tasks reached done; containment probe report present on "
            "main; 0 sandbox-attributable blocks."
        ),
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# Read-only structured readers over the event store + task records.
#
# Both stores are opened strictly read-only via the `file:<db>?mode=ro` URI
# (mirrors b3_gate._read_task_description / evals.task_sampler) — a predicate
# must never mutate the stores it measures. A missing/unreadable store raises
# SoakInputError rather than degrading to an empty result.
# ---------------------------------------------------------------------------

def _connect_ro(db) -> sqlite3.Connection:
    """Open *db* strictly read-only, or raise SoakInputError."""
    path = Path(db)
    if not path.exists():
        raise SoakInputError(f"database not found: {path}")
    try:
        return sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        raise SoakInputError(f"cannot open database {path}: {exc}") from exc


def _read_distinct_task_ids(db, event_type: str) -> set[str]:
    """DISTINCT non-null task_ids for *event_type* across ALL runs (no run_id
    scope — the soak spans >=3 days and orchestrator restarts, so a run-scoped
    query would undercount; same rationale as fetch_events_by_type_all_runs)."""
    conn = _connect_ro(db)
    try:
        rows = conn.execute(
            "SELECT DISTINCT task_id FROM events "
            "WHERE event_type = ? AND task_id IS NOT NULL",
            (event_type,),
        ).fetchall()
    except sqlite3.Error as exc:
        raise SoakInputError(f"failed reading events from {db}: {exc}") from exc
    finally:
        conn.close()
    return {str(r[0]) for r in rows}


def read_sandbox_applied_task_ids(db) -> set[str]:
    """Distinct task_ids with a ``sandbox_applied`` event (condition-a pool)."""
    return _read_distinct_task_ids(db, EventType.sandbox_applied.value)


def read_sandbox_unavailable_task_ids(db) -> set[str]:
    """Distinct task_ids with a ``sandbox_unavailable`` event (arm-1 pool)."""
    return _read_distinct_task_ids(db, EventType.sandbox_unavailable.value)


def read_escalations(db) -> list[dict]:
    """All ``escalation_created`` rows as ``{task_id, summary, category}`` dicts,
    with the ``data`` JSON decoded (arm-2 pool). Run-agnostic."""
    conn = _connect_ro(db)
    try:
        rows = conn.execute(
            "SELECT task_id, data FROM events WHERE event_type = ?",
            (EventType.escalation_created.value,),
        ).fetchall()
    except sqlite3.Error as exc:
        raise SoakInputError(f"failed reading escalations from {db}: {exc}") from exc
    finally:
        conn.close()
    out = []
    for task_id, raw in rows:
        summary = ""
        category = ""
        if raw:
            try:
                data = json.loads(raw)
            except (ValueError, TypeError):
                data = {}
            if isinstance(data, dict):
                summary = data.get("summary") or ""
                category = data.get("category") or ""
        out.append(
            {
                "task_id": None if task_id is None else str(task_id),
                "summary": summary,
                "category": category,
            }
        )
    return out


def read_task_status(db, tag: str = "master") -> dict[str, str]:
    """``{task_id(str): status}`` for the given *tag* (default 'master')."""
    conn = _connect_ro(db)
    try:
        rows = conn.execute(
            "SELECT id, status FROM tasks WHERE tag = ?", (tag,)
        ).fetchall()
    except sqlite3.Error as exc:
        raise SoakInputError(f"failed reading tasks from {db}: {exc}") from exc
    finally:
        conn.close()
    return {str(row[0]): row[1] for row in rows}


def _report_present_on_main(repo_root, ref: str = "main", path: str = PROBE_REPORT_PATH) -> bool:
    """True iff *path* is tracked in the tree of *ref* (default ``main``).

    Uses ``git cat-file -e <ref>:<path>`` — a tracked-on-main check, not bare
    filesystem existence, so a stray untracked/uncommitted docs/ file does not
    false-pass. A git invocation error (not a git repository, unknown ref, git
    missing) raises SoakInputError (-> exit 2); a path simply absent at an
    otherwise-valid ref returns False. Never lets a git failure masquerade as
    False.
    """
    def _run(args):
        try:
            return subprocess.run(
                ["git", "-C", str(repo_root), *args],
                capture_output=True, text=True,
            )
        except OSError as exc:  # git binary missing / repo_root unusable
            raise SoakInputError(
                f"git invocation failed in {repo_root}: {exc}"
            ) from exc

    # 1) Validate the ref resolves — separates a real git error (not a repo /
    #    unknown ref -> SoakInputError) from a path merely absent at a valid ref
    #    (-> False). rev-parse --verify --quiet exits non-zero (silently) on an
    #    unknown ref and 128 (with stderr) outside a git repo.
    ref_check = _run(["rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"])
    if ref_check.returncode != 0:
        raise SoakInputError(
            f"cannot resolve git ref {ref!r} in {repo_root} "
            "(not a git repository or unknown ref): "
            f"{(ref_check.stderr or '').strip() or 'no such ref'}"
        )
    # 2) The ref is valid — presence of the path in its tree IS the verdict.
    return _run(["cat-file", "-e", f"{ref}:{path}"]).returncode == 0


# ---------------------------------------------------------------------------
# CLI entry point — the before_done.kind='predicate' exit-code contract:
#   0 = green (all three conditions met)     -> task done
#   1 = measured-but-not-yet-green           -> the legitimate pre-soak state
#   2 = usage/infra error (SoakInputError, argparse error)
# Every non-zero exit prints ONE reason line: the 0/1 verdict to stdout, the 2
# error to stderr.
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="orchestrator.sandbox_soak",
        description=(
            "OS-sandbox rollout soak predicate (PRD γ1/γ5). Exit 0 iff >=N "
            "distinct sandboxed tasks are done, the containment probe report is "
            "tracked on main, and there are 0 sandbox-attributable blocks."
        ),
    )
    parser.add_argument("--repo-root", default=None, help="repo root (default: cwd)")
    parser.add_argument("--events-db", default=None,
                        help="event store (default: <repo-root>/data/orchestrator/runs.db)")
    parser.add_argument("--tasks-db", default=None,
                        help="task records (default: <repo-root>/.taskmaster/tasks/tasks.db)")
    parser.add_argument("--report-ref", default="main",
                        help="git ref the probe report must be tracked on (default: main)")
    parser.add_argument("--report-path", default=PROBE_REPORT_PATH,
                        help=f"probe report path (default: {PROBE_REPORT_PATH})")
    parser.add_argument("--tag", default="master", help="tasks.db tag (default: master)")
    parser.add_argument("--min-done", type=int, default=MIN_DONE_DEFAULT,
                        help=f"min distinct done sandboxed tasks (default: {MIN_DONE_DEFAULT})")
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        # argparse already printed a usage error to stderr and raised
        # SystemExit(2) (or 0 for --help). Preserve the code as a usage error
        # (2) without killing an in-process caller.
        return exc.code if isinstance(exc.code, int) else 2

    repo_root = Path(args.repo_root) if args.repo_root else Path.cwd()
    events_db = (
        Path(args.events_db) if args.events_db
        else repo_root / "data" / "orchestrator" / "runs.db"
    )
    tasks_db = (
        Path(args.tasks_db) if args.tasks_db
        else repo_root / ".taskmaster" / "tasks" / "tasks.db"
    )

    try:
        applied = read_sandbox_applied_task_ids(events_db)
        unavailable = read_sandbox_unavailable_task_ids(events_db)
        escalations = read_escalations(events_db)
        task_status = read_task_status(tasks_db, tag=args.tag)
        report_present = _report_present_on_main(
            repo_root, ref=args.report_ref, path=args.report_path
        )
    except SoakInputError as exc:
        print(f"check_sandbox_soak: ERROR — {exc}", file=sys.stderr)
        return 2

    verdict = evaluate_soak(
        applied, task_status, unavailable, escalations, report_present,
        min_done=args.min_done,
    )
    print(verdict.reason)
    return 0 if verdict.ok else 1


if __name__ == "__main__":
    sys.exit(main())
