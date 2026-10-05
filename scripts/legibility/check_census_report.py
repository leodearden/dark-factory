#!/usr/bin/env python3
"""Check the newest census report against docs/quality-findings-contract.md
§1 and §5 as plans/census-incremental-prd.md §4.3 C3 applies them (leaf L11).

Exit 0 when the newest plans/confusion-census-<YYYY-MM-DD>[-<n>] report
conforms, 1 otherwise. Every line goes to stdout and the last one is a
compact JSON verdict (docs/task-authoring.md §6). This is an independent
oracle: it imports nothing from census.py.
"""
from __future__ import annotations

import argparse
import enum
import json
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

PLANS_DIR = "plans"
REPORT_NAME_RE = re.compile(
    r"^confusion-census-(?P<date>\d{4}-\d{2}-\d{2})(?:-(?P<n>\d+))?\.(?P<ext>json|md)$"
)
SECTIONS = (
    "Method",
    "Findings",
    "Filed Tasks",
    "Dispositions",
    "Screened",
    "Adjudication",
    "Structural",
    "Synthesis",
)
NOTE_CAP = 400


class Verdict(enum.StrEnum):
    CONFORMS = "conforms"
    NO_CONFORMING_REPORT_YET = "no_conforming_report_yet"
    NONCONFORMING = "nonconforming"


class GapKind(enum.StrEnum):
    MISSING = "missing"
    MALFORMED = "malformed"


@dataclass(frozen=True)
class Gap:
    kind: GapKind
    name: str
    detail: str = ""


@dataclass(frozen=True)
class Report:
    date: str
    suffix: str | None
    record: Path | None
    rendering: Path | None

    @property
    def stem(self) -> str:
        return f"confusion-census-{self.date}" + (f"-{self.suffix}" if self.suffix else "")

    @property
    def shown_path(self) -> str:
        return f"{PLANS_DIR}/{self.stem}" + (".md" if self.rendering else ".json")


@dataclass(frozen=True)
class Result:
    verdict: Verdict
    report: str | None
    gaps: tuple[Gap, ...]
    headline: str


def newest_report(plans_dir: Path) -> Report | None:
    if not plans_dir.is_dir():
        return None
    halves: dict[tuple[str, str | None], dict[str, Path]] = {}
    for path in sorted(plans_dir.iterdir()):
        match = REPORT_NAME_RE.match(path.name)
        if match:
            halves.setdefault((match["date"], match["n"]), {})[match["ext"]] = path
    if not halves:
        return None
    (date, suffix), paths = max(halves.items(), key=lambda item: (item[0][0], int(item[0][1] or 1)))
    return Report(date, suffix, record=paths.get("json"), rendering=paths.get("md"))


def unfenced_lines(md_text: str) -> Iterator[tuple[int, str]]:
    in_fence = False
    for index, line in enumerate(md_text.splitlines()):
        if line.startswith("```"):
            in_fence = not in_fence
        elif not in_fence:
            yield index, line


def level2_titles(md_text: str) -> list[str]:
    return [line[3:].strip() for _, line in unfenced_lines(md_text) if line.startswith("## ")]


def check_sections(md_text: str) -> list[Gap]:
    present = set(level2_titles(md_text))
    return [Gap(GapKind.MISSING, f"## {title}") for title in SECTIONS if title not in present]


def check_rendering(report: Report) -> list[Gap]:
    if report.rendering is None:
        return []
    return check_sections(report.rendering.read_text(encoding="utf-8"))


def names(gaps: tuple[Gap, ...], kind: GapKind) -> list[str]:
    return [gap.name for gap in gaps if gap.kind is kind]


def nonconforming_headline(report: str, gaps: tuple[Gap, ...]) -> str:
    parts = []
    if missing := names(gaps, GapKind.MISSING):
        parts.append("keys missing: " + ", ".join(missing))
    if malformed := names(gaps, GapKind.MALFORMED):
        parts.append("malformed: " + ", ".join(malformed))
    return f"{report} does not conform -- " + "; ".join(parts)


def check(project_root: Path) -> Result:
    report = newest_report(project_root / PLANS_DIR)
    if report is None:
        return Result(Verdict.NONCONFORMING, None, (), f"no census report under {project_root / PLANS_DIR}")
    gaps = tuple(check_rendering(report))
    if gaps:
        return Result(Verdict.NONCONFORMING, report.shown_path, gaps, nonconforming_headline(report.shown_path, gaps))
    return Result(Verdict.CONFORMS, report.shown_path, (), f"{report.shown_path} conforms")


def gap_line(gap: Gap) -> str:
    return f"  {gap.kind} {gap.name}" + (f": {gap.detail}" if gap.detail else "")


def compact_verdict(result: Result) -> str:
    payload = {
        "verdict": str(result.verdict),
        "report": result.report,
        "missing": names(result.gaps, GapKind.MISSING),
        "malformed": names(result.gaps, GapKind.MALFORMED),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def render(result: Result) -> list[str]:
    return [result.headline, *(gap_line(gap) for gap in result.gaps), compact_verdict(result)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check the newest census report's conformance to the findings contract.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.project_root.is_dir():
        parser.error(f"--project-root is not a directory: {args.project_root}")
    result = check(args.project_root)
    for line in render(result):
        print(line)
    return 0 if result.verdict is Verdict.CONFORMS else 1


if __name__ == "__main__":
    sys.exit(main())
