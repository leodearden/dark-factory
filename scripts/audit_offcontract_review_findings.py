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
  - `census` / `select_population` measure a verdict tree. Both REFUSE a root
    that is not a directory rather than returning zeros, and both report the
    residue they could not read — a count the caller is handed, never a number
    quietly shortened.
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

#: The one place that knows the live tree's layout. Named rather than inlined
#: because a zero-file census has to be able to SAY what it looked for: the
#: frozen `corpus/` in this task's artifact directory is stored flat as
#: `<task>.json`, so pointing `--root` at it matches nothing, and an unnamed
#: zero there reads as a measurement instead of a layout mismatch.
VERDICT_GLOB = "*/verdicts/*.json"

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


def _is_off_contract(severity: object) -> bool:
    """Whether a severity is one the review gates cannot read.

    Named rather than inlined because the REFUSED issues have to be classified
    the same way as the read ones. Two spellings of this test is how a tally
    starts counting a population it never inspected.
    """
    return severity not in CONTRACT_SEVERITIES


def _is_selectable(severity: object) -> bool:
    """Whether a severity puts an issue in the population task 5430 triages."""
    return _is_off_contract(severity) and severity in TRIAGE_SEVERITIES


def _text(raw: dict, key: str) -> str | None:
    """Return raw[key] as stripped text, or None when absent/blank.

    `title: None` and `title: "   "` are both treated as absent — five of the
    23 dropped findings carry an explicit null here.
    """
    value = raw.get(key)
    if not isinstance(value, str):
        return None
    return value.strip() or None


def _raw_issues(payload: object) -> list | None:
    """The `verdict.issues[]` list in a payload, or None when the shape is wrong.

    Valid JSON of the wrong TYPE is exactly as unreadable as broken JSON, and
    refusing it here is what keeps that a TALLY rather than an abort: reaching
    `.get` on a top-level list raises `AttributeError`, which the walk's
    `except (OSError, ValueError)` does not catch, so one malformed file would
    kill the walk of every other one. The input is untracked runtime state from
    a pipeline already measured emitting six mutually incompatible issue
    schemas, so this is a live class of malformation, not a hypothetical.

    An ABSENT or null `verdict`/`issues` is benign and yields `[]`, not a
    refusal — 29 of the 1,271 live verdict files legitimately carry no `issues`
    key, and counting those as malformed would invent unparseable files.
    """
    if not isinstance(payload, dict):
        return None
    verdict = payload.get("verdict") or {}
    if not isinstance(verdict, dict):
        return None
    issues = verdict.get("issues") or []
    return issues if isinstance(issues, list) else None


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
    if not isinstance(raw, dict):
        raise ValueError(
            f"issue entry is not an object but a {type(raw).__name__}: {raw!r}"
        )
    return {
        "id": f"{task_id}-{raw.get('category')}-{index}",
        "task": task_id,
        "index": index,
        "severity": raw.get("severity"),
        "category": raw.get("category"),
        "location": _resolve_location(raw),
        "statement": _resolve_statement(raw),
        "off_contract": _is_off_contract(raw.get("severity")),
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

    The refused residue folds into the two tallies ASYMMETRICALLY, and the
    asymmetry is the point. `location_less` absorbs all of it, because an issue
    the normalizer refused is by definition location-less. `off_contract_severity`
    absorbs only the part whose severity was ACTUALLY off-contract: a refused
    issue can carry `severity: "blocking"`, and folding it in wholesale would
    contaminate this tally with a population whose severity was never inspected
    — defeating the separation the paragraph above exists to preserve.
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
    """One walk's raw result. `unlocatable_severities` counts the REFUSED issues
    by the severity they carried, so both consumers classify that residue from
    one place: the census asks which of them were off-contract, the population
    asks which would have been selected. A bare refusal count could answer
    neither without re-inspecting the raw dicts the walk has already discarded.
    """
    issues: list[VerdictIssue]
    files: int
    unparseable: int
    unlocatable_severities: Counter
    verdicts_with_issues: int
    emitted_at: list[str]
    roles: Counter

    @property
    def unlocatable(self) -> int:
        return sum(self.unlocatable_severities.values())


def _scan(root: Path) -> Scan:
    """Walk `<root>/*/verdicts/*.json` once. The SPOT for tree traversal.

    Tolerant by tallying, never by dropping: a file that will not parse and an
    issue the normalizer refuses are both counted, so a caller is told what the
    walk could not read rather than being handed a quietly short number.

    That contract has one input it cannot honour by tallying, so it refuses it
    instead: `Path.glob` on a missing directory yields nothing rather than
    raising, which would hand back an all-zero census indistinguishable from a
    real tree that happens to hold no verdicts. An unreadable root is a broken
    QUESTION, not a measurable answer, so it raises.
    """
    if not root.is_dir():
        raise NotADirectoryError(
            f"not a directory, so there is nothing to census: {root}"
        )
    issues: list[VerdictIssue] = []
    files = unparseable = verdicts_with_issues = 0
    emitted_at: list[str] = []
    roles: Counter = Counter()
    unlocatable_severities: Counter = Counter()

    for path in sorted(root.glob(VERDICT_GLOB)):
        files += 1
        try:
            payload = json.loads(path.read_text())
        except (OSError, ValueError):
            unparseable += 1
            continue

        raw_issues = _raw_issues(payload)
        if raw_issues is None:
            unparseable += 1
            continue
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
                # The raw dict is still in hand, so the severity it carried is
                # recorded rather than assumed. A non-dict entry has none.
                unlocatable_severities[raw.get("severity") if isinstance(raw, dict) else None] += 1
                continue
            issues.append(VerdictIssue(
                path=path,
                role=role,
                emitted_at=stamp,
                location_less=not _text(raw, "location"),
                record=record,
            ))

    return Scan(issues, files, unparseable, unlocatable_severities,
                verdicts_with_issues, emitted_at, roles)


def census(root: Path) -> Census:
    """Measure a verdict tree. Makes no selection and no judgement."""
    scan = _scan(root)
    return Census(
        files=scan.files,
        unparseable=scan.unparseable,
        unlocatable=scan.unlocatable,
        verdicts_with_issues=scan.verdicts_with_issues,
        issues=len(scan.issues) + scan.unlocatable,
        off_contract_severity=(
            sum(i.record["off_contract"] for i in scan.issues)
            + sum(count for severity, count in scan.unlocatable_severities.items()
                  if _is_off_contract(severity))
        ),
        location_less=sum(i.location_less for i in scan.issues) + scan.unlocatable,
        roles=dict(scan.roles),
        emitted_first=min(scan.emitted_at, default=None),
        emitted_last=max(scan.emitted_at, default=None),
    )


class Population(NamedTuple):
    """The issues a triage roster can be frozen from, plus the ones that cannot.

    `unselectable` is not a footnote. A roster is frozen from `issues`, and a
    finding absent from a roster is never dispositioned — which is precisely how
    the 23 findings this task exists to account for were lost. An issue the
    normalizer refused is unrosterable for a different reason (no location) but
    with the identical consequence, so the count travels WITH the population
    rather than being available only from a separate census call.

    It counts the refused issues whose severity would have SELECTED them, not
    every refusal: a refused `low` was never in this population anyway, and
    reporting it here would overstate what the roster is missing.
    """
    issues: list[VerdictIssue]
    unselectable: int


def select_population(root: Path) -> Population:
    """The issues task 5430 triages: off-contract severity AND in the band.

    Never filters on role. Role is a census observation, and excluding a role
    here would understate the residue while looking like it had measured it.
    """
    scan = _scan(root)
    return Population(
        issues=[issue for issue in scan.issues
                if _is_selectable(issue.record["severity"])],
        unselectable=sum(count for severity, count in scan.unlocatable_severities.items()
                         if _is_selectable(severity)),
    )


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

    It must also never FAIL OPEN, which is a sharper requirement than checking
    the entries it is given. Every check below is "each roster id appears once,
    validly" — a shape an EMPTY roster satisfies vacuously, so a report with no
    roster, or one whose roster key is misspelled, would be certified complete.
    A completeness gate that passes a file containing nothing inverts its own
    purpose, so the roster's own usability is checked first.
    """
    violations: list[str] = []
    roster_ids: list[str] = []
    seen: Counter = Counter()

    for entry in report.get("roster") or []:
        # A blank id is not a benign omission: `None` matches `None`, so a
        # roster entry and a disposition entry that both lack one pair up and
        # validate each other. An unusable id is reported and never matched.
        if rid := _text(entry, "id"):
            roster_ids.append(rid)
        else:
            violations.append(f"roster entry carries no usable id: {entry!r}")

    if not roster_ids:
        violations.append(
            "report carries no usable roster — every check below is satisfied "
            "vacuously by an empty one, so there is nothing to validate against"
        )

    for entry in report.get("dispositions") or []:
        rid = _text(entry, "id")
        if not rid:
            violations.append(f"disposition entry carries no usable id: {entry!r}")
            continue
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

    try:
        measured = census(args.root)
        population = select_population(args.root)
    except NotADirectoryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # The glob is reported alongside the counts so a zero names what it looked
    # for. Both modes carry it: `--json` is what provenance.json quotes as the
    # reproduction command, so a field missing there is a number a reader
    # cannot re-derive.
    if args.json:
        print(json.dumps({
            "root": str(args.root), "glob": VERDICT_GLOB,
            **measured._asdict(),
            "triage_population": len(population.issues),
            "triage_unselectable": population.unselectable,
        }, indent=2))
    else:
        print(f"root: {args.root}")
        print(f"  glob: {VERDICT_GLOB}")
        for field, value in measured._asdict().items():
            print(f"  {field}: {value}")
        print(f"  triage_population: {len(population.issues)} "
              f"(+{population.unselectable} unlocatable, not selectable)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
