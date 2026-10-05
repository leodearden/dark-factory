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
METHOD_TITLE = "Method"
SECTIONS = (
    METHOD_TITLE,
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

    def shown(self, ext: str) -> str:
        return f"{PLANS_DIR}/{self.stem}.{ext}"

    @property
    def shown_path(self) -> str:
        return self.shown("md" if self.rendering else "json")


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


def heading(title: str) -> str:
    return f"## {title}"


def check_sections(md_text: str) -> list[Gap]:
    present = set(level2_titles(md_text))
    return [Gap(GapKind.MISSING, heading(title)) for title in SECTIONS if title not in present]


def check_halves(report: Report) -> list[Gap]:
    absent = [ext for ext, path in (("json", report.record), ("md", report.rendering)) if path is None]
    return [Gap(GapKind.MISSING, report.shown(ext)) for ext in absent]


def predates_header(report: Report, titles: list[str]) -> bool:
    return report.record is None and METHOD_TITLE not in titles


def names(gaps: tuple[Gap, ...], kind: GapKind) -> list[str]:
    return [gap.name for gap in gaps if gap.kind is kind]


def no_report_result(project_root: Path) -> Result:
    pattern = f"{PLANS_DIR}/confusion-census-<YYYY-MM-DD>[-<n>].{{json,md}}"
    headline = f"no conforming report yet -- no {pattern} report under {project_root}"
    return Result(Verdict.NO_CONFORMING_REPORT_YET, None, (), headline)


def predates_header_result(report: Report) -> Result:
    gaps = (Gap(GapKind.MISSING, heading(METHOD_TITLE)), Gap(GapKind.MISSING, report.shown("json")))
    headline = (
        f"no conforming report yet -- {report.shown_path} predates the contract §5 header; "
        f"missing: {', '.join(names(gaps, GapKind.MISSING))}"
    )
    return Result(Verdict.NO_CONFORMING_REPORT_YET, report.shown_path, gaps, headline)


def nonconforming_headline(report: str, gaps: tuple[Gap, ...]) -> str:
    parts = []
    if missing := names(gaps, GapKind.MISSING):
        parts.append("keys missing: " + ", ".join(missing))
    if malformed := names(gaps, GapKind.MALFORMED):
        parts.append("malformed: " + ", ".join(malformed))
    return f"{report} does not conform -- " + "; ".join(parts)


def post_header_result(report: Report, gaps: tuple[Gap, ...]) -> Result:
    if gaps:
        return Result(Verdict.NONCONFORMING, report.shown_path, gaps, nonconforming_headline(report.shown_path, gaps))
    return Result(Verdict.CONFORMS, report.shown_path, (), f"{report.shown_path} conforms")


def post_header_gaps(report: Report, md_text: str | None) -> list[Gap]:
    gaps = check_halves(report)
    if md_text is not None:
        gaps += check_sections(md_text)
    return gaps


def check(project_root: Path) -> Result:
    report = newest_report(project_root / PLANS_DIR)
    if report is None:
        return no_report_result(project_root)
    md_text = report.rendering.read_text(encoding="utf-8") if report.rendering else None
    if predates_header(report, level2_titles(md_text or "")):
        return predates_header_result(report)
    return post_header_result(report, tuple(post_header_gaps(report, md_text)))


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
