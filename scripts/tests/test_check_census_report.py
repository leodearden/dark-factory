"""Tests for scripts/legibility/check_census_report.py, the census report
conformance oracle (plans/census-incremental-prd.md leaf L11).

Fixtures are real report files under tmp_path/plans, built here from the
contract (docs/quality-findings-contract.md §1 and §5, PRD §4.3 C3) rather
than from census.py, so the checker is judged against the contract's own
words. Section titles and finding fields are restated literally on purpose:
a checker that silently drops one is caught.
"""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import check_census_report
import pytest
import yaml
from cli_subprocess_timeout import cli_timeout_from_env

SCRIPT = Path(__file__).parent.parent / "legibility" / "check_census_report.py"
CLI_TIMEOUT = cli_timeout_from_env("CHECK_CENSUS_REPORT_TEST_TIMEOUT")

METHOD = {
    "run_id": "census-dark_factory-20261020",
    "as_of_sha": "0" * 40,
    "since": "none",
    "evidence": {"sessions_enumerated": 12, "mined": 4},
    "verification": {"confirmed": 1, "weakened": 0, "refuted": 0, "unverified": 0},
    "cost": {"miner_calls": 4, "verify_calls": 1, "wall_clock_secs": 90},
    "inputs_consumed": [],
    "extra": {},
}

FINDING = {
    "key": "fk-0123456789ab",
    "area": "shared",
    "sub_area": None,
    "anchor": "scripts/legibility/x.py::f",
    "tags": ["h13", "kind:confusion"],
    "severity": "medium",
    "evidence_source": ["agent-transcripts"],
    "statement": "Agents misread f's return value in three sessions.",
    "proposal": None,
    "verdict": "confirmed",
    "disposition": "open",
    "first_seen": "census-dark_factory-20261020",
    "last_seen": ["census-dark_factory-20261020"],
    "supersedes": [],
}

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


def method_for(stem: str) -> dict:
    """METHOD with the run_id the contract §5 basename rule gives *stem*."""
    suffix = stem.removeprefix("confusion-census-")
    date, _, n = suffix[:10], suffix[10:11], suffix[11:]
    run_id = "census-dark_factory-" + date.replace("-", "") + (f"-{n}" if n else "")
    return {**copy.deepcopy(METHOD), "run_id": run_id}


def method_section(method: dict) -> str:
    return "```yaml\n" + yaml.safe_dump(method, sort_keys=False) + "```\n"


def write_report(
    plans_dir: Path,
    stem: str,
    *,
    method: dict | None = None,
    findings: list | None = None,
    sections: tuple[str, ...] | list[str] = SECTIONS,
    bodies: dict[str, str] | None = None,
    record: bool = True,
    md: bool = True,
) -> None:
    method = method_for(stem) if method is None else method
    findings = [copy.deepcopy(FINDING)] if findings is None else findings
    plans_dir.mkdir(parents=True, exist_ok=True)
    if record:
        doc = {"method": method, "findings": findings}
        (plans_dir / f"{stem}.json").write_text(json.dumps(doc), encoding="utf-8")
    if md:
        overrides = bodies or {}
        parts = ["# confusion census\n\n"]
        for title in sections:
            default = method_section(method) if title == "Method" else f"{title} prose for the fixture run.\n"
            parts.append(f"## {title}\n\n{overrides.get(title, default)}\n")
        (plans_dir / f"{stem}.md").write_text("".join(parts), encoding="utf-8")


def run(root: Path, capsys: pytest.CaptureFixture[str]) -> tuple[int, list[str], dict]:
    rc = check_census_report.main(["--project-root", str(root)])
    lines = capsys.readouterr().out.splitlines()
    return rc, lines, json.loads(lines[-1])


def without(titles: tuple[str, ...], *dropped: str) -> list[str]:
    return [t for t in titles if t not in dropped]


def test_complete_newest_report_conforms(tmp_path, capsys):
    write_report(tmp_path / "plans", "confusion-census-2026-10-20")

    rc, _, verdict = run(tmp_path, capsys)

    assert rc == 0
    assert verdict["verdict"] == "conforms"
    assert verdict["report"] == "plans/confusion-census-2026-10-20.md"
    assert verdict["missing"] == []
    assert verdict["malformed"] == []


def write_row_16_fixture(root: Path) -> None:
    write_report(root / "plans", "confusion-census-2026-10-15")
    write_report(root / "plans", "confusion-census-2026-10-20", sections=without(SECTIONS, "Screened"))


def test_newest_lacking_screened_exits_1_naming_it(tmp_path, capsys):
    write_row_16_fixture(tmp_path)

    rc, lines, verdict = run(tmp_path, capsys)

    assert rc == 1
    assert verdict["verdict"] == "nonconforming"
    assert verdict["report"] == "plans/confusion-census-2026-10-20.md"
    assert "## Screened" in verdict["missing"]
    assert any("plans/confusion-census-2026-10-20.md" in line and "## Screened" in line for line in lines)
    assert "keys missing" in lines[0]


@pytest.mark.parametrize("title", without(SECTIONS, "Method"))
def test_each_required_section_is_named_when_absent(tmp_path, capsys, title):
    write_report(tmp_path / "plans", "confusion-census-2026-10-20", sections=without(SECTIONS, title))

    rc, _, verdict = run(tmp_path, capsys)

    assert rc == 1
    assert f"## {title}" in verdict["missing"]


def test_heading_inside_code_fence_does_not_count(tmp_path, capsys):
    fenced = "Synthesis prose.\n\n```\n## Screened\n3 clusters attached\n```\n"
    write_report(
        tmp_path / "plans",
        "confusion-census-2026-10-20",
        sections=without(SECTIONS, "Screened"),
        bodies={"Synthesis": fenced},
    )

    rc, _, verdict = run(tmp_path, capsys)

    assert rc == 1
    assert "## Screened" in verdict["missing"]


def test_cli_entry_point_exit_status(tmp_path):
    write_row_16_fixture(tmp_path)

    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--project-root", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=CLI_TIMEOUT,
        check=False,
    )

    assert proc.returncode == 1
    assert json.loads(proc.stdout.splitlines()[-1])["verdict"] == "nonconforming"
