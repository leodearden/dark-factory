#!/usr/bin/env python3
"""Audit the reviewer findings that the off-contract verdict shape dropped.

READ-ONLY / REPORT-ONLY: this module and its CLI open no database, mutate no
task, verdict or plan artifact, and write nothing outside stdout. The triage it
supports is a read-and-report exercise; any actual fix belongs in a per-defect
follow-up task.

Background (task 5430, origin esc-2896-10): between 2026-07-19 and 2026-08-10
`reviewer_comprehensive` emitted verdicts whose `verdict.issues[]` entries used
an off-contract shape — `severity` outside {blocking, suggestion}, and
`file`+`line` instead of `location`. Both review gates key on the contract
fields, so those findings were read as suggestions AND skipped by the in-scope
filter: the amendment gate never fired and no implementer ever saw them. The
companion task prevents recurrence; task 5430 accounts for the residue, and this
script does the mechanical half of that accounting.

It does three things and NO JUDGEMENT:

  - `normalize_issue` collapses any of the six observed issue schemas into one
    record. It is the single place that knows about the shape variance, so a
    caller never re-derives "where is the headline in this one".
  - `census` / `select_population` measure a verdict tree.
  - `validate_report` checks a finished triage report against its frozen roster.

Whether a given finding is still live on today's main is a judgement over code
that has moved for seven weeks. No function here answers that, and none should:
the dispositions live in the triage report, with their reasoning.

A NOTE ON THE NUMBERS THIS PRINTS. The verdict corpus is untracked runtime state
under a gitignored `.worktrees/`, and later re-reviews overwrite a verdict file
in place. It is therefore NOT a stable population: task 2896's verdict was
overwritten on 2026-09-12, destroying the highest-severity finding in task
5430's own roster. Every census this script emits is a dated measurement of a
moving target, never a fact about a closed set.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import NamedTuple

DEFAULT_ROOT = Path("/home/leo/src/dark-factory/.worktrees/.task-meta")

#: The only two severities the review gates understand. Anything else is read as
#: a suggestion and skipped by the in-scope filter — the drop this task triages.
CONTRACT_SEVERITIES = frozenset({"blocking", "suggestion"})

#: The severities task 5430 triages. Everything below this floor was emitted at
#: low/minor/nit/trivial and is deliberately out of scope.
TRIAGE_SEVERITIES = frozenset({"high", "major", "medium", "moderate"})

#: Ordered key preference for the headline. `title` is the common spelling;
#: `short_summary` is the one the ReportFindings-shaped verdicts use.
HEADLINE_KEYS = ("title", "short_summary")

#: Ordered key preference for the primary detail body. First populated key wins.
DETAIL_KEYS = ("description", "detail", "summary")

#: Supplementary detail keys, appended in this order when populated. These are
#: additive rather than alternatives: a shape can carry several at once.
SUPPLEMENT_KEYS = ("failure_scenario", "reproduce", "suggestion", "suggested_fix", "recommendation")


def _text(raw: dict, key: str) -> str | None:
    """Return raw[key] as stripped text, or None when absent/blank.

    `title: None` and `title: "   "` are both treated as absent — five of the
    23 dropped findings carry an explicit null here.
    """
    value = raw.get(key)
    if not isinstance(value, str):
        return None
    return value.strip() or None


def _resolve_location(raw: dict) -> str:
    """Resolve a location from either the on-contract or off-contract spelling.

    Raises rather than returning a partly-None string: an unlocatable finding
    that stringifies to "None:None" reads as located, and then nobody checks it.
    """
    on_contract = _text(raw, "location")
    if on_contract:
        return on_contract
    file = _text(raw, "file")
    if not file:
        raise ValueError(
            f"issue carries no location signal at all: neither `location` nor `file` "
            f"(keys present: {sorted(raw)})"
        )
    line = raw.get("line")
    return f"{file}:{line}" if line is not None else file


def _resolve_statement(raw: dict) -> str:
    """Join every populated text key into one readable statement.

    Nothing is dropped: the shapes disagree about which key carries the detail,
    so preferring one and discarding the rest would lose real content.
    """
    parts: list[str] = []
    for key in HEADLINE_KEYS:
        if headline := _text(raw, key):
            parts.append(headline)
            break
    for key in DETAIL_KEYS:
        if detail := _text(raw, key):
            parts.append(detail)
            break
    parts.extend(f"{key}: {text}" for key in SUPPLEMENT_KEYS if (text := _text(raw, key)))
    return "\n\n".join(parts)


def normalize_issue(task_id: str, index: int, raw: dict) -> dict:
    """Collapse one reviewer issue of any observed schema into one record.

    Pure: takes a dict, returns a dict, touches no disk. `index` is the issue's
    position in `verdict.issues[]`, which is what makes the derived `id` stable
    and re-checkable against a frozen verdict file.
    """
    return {
        "id": f"{task_id}-{raw.get('category')}-{index}",
        "task": task_id,
        "index": index,
        "severity": raw.get("severity"),
        "category": raw.get("category"),
        "location": _resolve_location(raw),
        "statement": _resolve_statement(raw),
        "off_contract": raw.get("severity") not in CONTRACT_SEVERITIES,
    }


class VerdictIssue(NamedTuple):
    """One normalized issue plus the facts that belong to its verdict FILE.

    Role, emission time and source path are properties of the verdict, not of
    the issue, so `normalize_issue` cannot supply them — the walker does.
    """
    path: Path
    role: str
    emitted_at: str | None
    location_less: bool
    record: dict


class Census(NamedTuple):
    """A DATED MEASUREMENT of a verdict tree, never a fact about a closed set.

    `issues` counts normalized issues; `unlocatable` counts the residue the
    normalizer refused, so `issues + unlocatable` is every issue entry present.
    `verdicts_with_issues` reads the raw list, so a verdict whose only entry was
    refused still counts as carrying one, and `sum(roles.values()) == issues`
    for the same reason. `off_contract_severity` and
    `location_less` are deliberately separate tallies: the 23 findings task 5430
    triages were in both, which is how the two review gates compounded, but a
    census that folded them together could never show the two diverging.
    """
    files: int
    unparseable: int
    unlocatable: int
    verdicts_with_issues: int
    issues: int
    off_contract_severity: int
    location_less: int
    roles: dict[str, int]
    emitted_first: str | None
    emitted_last: str | None


class Scan(NamedTuple):
    issues: list[VerdictIssue]
    files: int
    unparseable: int
    unlocatable: int
    verdicts_with_issues: int
    emitted_at: list[str]
    roles: Counter


def _scan(root: Path) -> Scan:
    """Walk `<root>/*/verdicts/*.json` once. The SPOT for tree traversal.

    Tolerant by tallying, never by dropping: a file that will not parse and an
    issue the normalizer refuses are both counted, so a caller is told what the
    walk could not read rather than being handed a quietly short number.
    """
    issues: list[VerdictIssue] = []
    files = unparseable = unlocatable = verdicts_with_issues = 0
    emitted_at: list[str] = []
    roles: Counter = Counter()

    for path in sorted(root.glob("*/verdicts/*.json")):
        files += 1
        try:
            payload = json.loads(path.read_text())
        except (OSError, ValueError):
            unparseable += 1
            continue

        raw_issues = payload.get("verdict", {}).get("issues") or []
        if raw_issues:
            verdicts_with_issues += 1
        role = payload.get("role") or path.stem
        stamp = payload.get("emitted_at")
        if stamp:
            emitted_at.append(stamp)

        for index, raw in enumerate(raw_issues):
            # Tallied before normalization can refuse the issue, so the role
            # breakdown reconciles with the issue total rather than quietly
            # omitting whatever the walk could not read.
            roles[role] += 1
            try:
                record = normalize_issue(path.parent.parent.name, index, raw)
            except ValueError:
                unlocatable += 1
                continue
            issues.append(VerdictIssue(
                path=path,
                role=role,
                emitted_at=stamp,
                location_less=not _text(raw, "location"),
                record=record,
            ))

    return Scan(issues, files, unparseable, unlocatable, verdicts_with_issues, emitted_at, roles)


def census(root: Path) -> Census:
    """Measure a verdict tree. Makes no selection and no judgement."""
    scan = _scan(root)
    return Census(
        files=scan.files,
        unparseable=scan.unparseable,
        unlocatable=scan.unlocatable,
        verdicts_with_issues=scan.verdicts_with_issues,
        issues=len(scan.issues) + scan.unlocatable,
        off_contract_severity=sum(i.record["off_contract"] for i in scan.issues) + scan.unlocatable,
        location_less=sum(i.location_less for i in scan.issues) + scan.unlocatable,
        roles=dict(scan.roles),
        emitted_first=min(scan.emitted_at, default=None),
        emitted_last=max(scan.emitted_at, default=None),
    )


def select_population(root: Path) -> list[VerdictIssue]:
    """The issues task 5430 triages: off-contract severity AND in the band.

    Never filters on role. Role is a census observation, and excluding a role
    here would understate the residue while looking like it had measured it.
    """
    return [
        issue for issue in _scan(root).issues
        if issue.record["off_contract"] and issue.record["severity"] in TRIAGE_SEVERITIES
    ]


#: The only three dispositions the triage may reach: (a) already fixed
#: incidentally, (b) still a live defect, (c) not a defect / no longer applicable.
DISPOSITIONS = frozenset({"a", "b", "c"})

#: The one disposition that obliges a follow-up task. A (b) with no ticket has
#: been dropped a second time, by the task that exists to account for the first.
LIVE_DEFECT = "b"


def validate_report(report: dict) -> list[str]:
    """Check a triage report against its frozen roster. [] means valid.

    Pure: reads the parsed dict, touches no disk. This is the mechanical form of
    the task's "honest accounting" — completeness is a structural property of a
    data file, so it is enforced rather than asserted in prose. What it
    deliberately does NOT check is whether any disposition is CORRECT; that is a
    judgement over seven weeks of moved code, and a test encoding it would be
    pinning a conclusion rather than a behaviour.
    """
    violations: list[str] = []
    roster_ids = [entry.get("id") for entry in report.get("roster", [])]
    seen: Counter = Counter()

    for entry in report.get("dispositions", []):
        rid = entry.get("id")
        seen[rid] += 1
        if rid not in roster_ids:
            violations.append(f"{rid}: dispositioned but absent from the frozen roster")
            continue
        if seen[rid] > 1:
            violations.append(f"{rid}: dispositioned {seen[rid]} times — one of them is unread")
            continue

        disposition = entry.get("disposition")
        deferred = bool(entry.get("deferred_by_gate"))
        note = entry.get("note")
        ticket = entry.get("followup_ticket")

        if not (isinstance(note, str) and note.strip()):
            violations.append(f"{rid}: no note — a disposition without reasoning cannot be checked")
        if deferred and disposition is not None:
            violations.append(f"{rid}: claims both a disposition and a gate deferral")
        elif not deferred and disposition not in DISPOSITIONS:
            violations.append(f"{rid}: disposition {disposition!r} is not one of {sorted(DISPOSITIONS)}")

        has_ticket = isinstance(ticket, str) and ticket.strip()
        if disposition == LIVE_DEFECT and not has_ticket:
            violations.append(f"{rid}: judged a live defect but cites no follow-up ticket")
        if disposition in DISPOSITIONS - {LIVE_DEFECT} and has_ticket:
            violations.append(f"{rid}: dispositioned ({disposition}) yet carries a follow-up ticket")

    violations.extend(
        f"{rid}: on the roster but never dispositioned"
        for rid in roster_ids if rid not in seen
    )
    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="task-meta tree to census (default: %(default)s)")
    parser.add_argument("--json", action="store_true", help="emit the census as JSON")
    parser.add_argument("--validate", type=Path, metavar="REPORT",
                        help="validate a triage report against its frozen roster")
    args = parser.parse_args(argv)

    if args.validate:
        violations = validate_report(json.loads(args.validate.read_text()))
        for violation in violations:
            print(violation)
        print(f"{len(violations)} violation(s) in {args.validate}")
        return 1 if violations else 0

    measured = census(args.root)
    if args.json:
        print(json.dumps({"root": str(args.root), **measured._asdict()}, indent=2))
    else:
        print(f"root: {args.root}")
        for field, value in measured._asdict().items():
            print(f"  {field}: {value}")
        print(f"  triage_population: {len(select_population(args.root))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
