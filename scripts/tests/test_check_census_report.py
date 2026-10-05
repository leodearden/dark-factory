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
    record_text: str | None = None,
    md: bool = True,
) -> None:
    method = method_for(stem) if method is None else method
    findings = [copy.deepcopy(FINDING)] if findings is None else findings
    plans_dir.mkdir(parents=True, exist_ok=True)
    if record:
        text = json.dumps({"method": method, "findings": findings}) if record_text is None else record_text
        (plans_dir / f"{stem}.json").write_text(text, encoding="utf-8")
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


PRE_CONTRACT_REPORT = (
    "# confusion census 2026-10-03\n\n"
    "## Saturation\n\nprose\n\n"
    "## Verification\n\nprose\n\n"
    "## Synthesis\n\nprose\n\n"
    "## Filed Tasks\n\nprose\n\n"
    "## Cost\n\nprose\n"
)


def test_newest_report_predating_the_header_is_no_conforming_report_yet(tmp_path, capsys):
    write_report(tmp_path / "plans", "confusion-census-2026-09-30")
    (tmp_path / "plans" / "confusion-census-2026-10-03.md").write_text(PRE_CONTRACT_REPORT, encoding="utf-8")

    rc, lines, verdict = run(tmp_path, capsys)

    assert rc == 1
    assert verdict["verdict"] == "no_conforming_report_yet"
    assert verdict["report"] == "plans/confusion-census-2026-10-03.md"
    assert verdict["missing"] == ["## Method", "plans/confusion-census-2026-10-03.json"]
    assert "no conforming report yet" in lines[0]
    assert "plans/confusion-census-2026-10-03.md" in lines[0]
    assert "keys missing" not in lines[0]


@pytest.mark.parametrize("plans_present", [False, True], ids=["no-plans-dir", "only-unrelated-files"])
def test_no_report_at_all_is_no_conforming_report_yet(tmp_path, capsys, plans_present):
    if plans_present:
        plans = tmp_path / "plans"
        plans.mkdir()
        (plans / "confusion-census-2026-10-03-payloads.json").write_text("{}", encoding="utf-8")
        (plans / "census-incremental-prd.md").write_text("# PRD\n", encoding="utf-8")

    rc, lines, verdict = run(tmp_path, capsys)

    assert rc == 1
    assert verdict["verdict"] == "no_conforming_report_yet"
    assert verdict["report"] is None
    assert "no conforming report yet" in lines[0]
    assert "plans/confusion-census-" in lines[0]


def test_rendering_with_method_but_no_record_is_nonconforming(tmp_path, capsys):
    write_report(tmp_path / "plans", "confusion-census-2026-10-20", record=False)

    rc, lines, verdict = run(tmp_path, capsys)

    assert rc == 1
    assert verdict["verdict"] == "nonconforming"
    assert "plans/confusion-census-2026-10-20.json" in verdict["missing"]
    assert "keys missing" in lines[0]
    assert "no conforming report yet" not in lines[0]


def test_rendering_without_method_heading_is_named_missing(tmp_path, capsys):
    write_report(tmp_path / "plans", "confusion-census-2026-10-20", sections=without(SECTIONS, "Method"))

    rc, _, verdict = run(tmp_path, capsys)

    assert rc == 1
    assert verdict["verdict"] == "nonconforming"
    assert "## Method" in verdict["missing"]


STEM = "confusion-census-2026-10-20"


def method_lacking(*keys: str) -> dict:
    return {k: v for k, v in method_for(STEM).items() if k not in keys}


def assert_nonconforming(rc: int, verdict: dict) -> None:
    assert rc == 1
    assert verdict["verdict"] == "nonconforming"


def test_method_block_missing_keys_are_named(tmp_path, capsys):
    write_report(tmp_path / "plans", STEM, bodies={"Method": method_section(method_lacking("extra", "inputs_consumed"))})

    rc, _, verdict = run(tmp_path, capsys)

    assert_nonconforming(rc, verdict)
    assert set(verdict["missing"]) == {"method.extra", "method.inputs_consumed"}
    assert verdict["malformed"] == []


@pytest.mark.parametrize(
    "body",
    [
        "Run notes first.\n\n" + method_section(method_for(STEM)),
        "```json\n" + json.dumps(method_for(STEM)) + "\n```\n",
    ],
    ids=["prose-before-fence", "json-fence"],
)
def test_method_first_element_must_be_a_yaml_fence(tmp_path, capsys, body):
    write_report(tmp_path / "plans", STEM, bodies={"Method": body})

    rc, _, verdict = run(tmp_path, capsys)

    assert_nonconforming(rc, verdict)
    assert "## Method yaml block" in verdict["missing"]


UNPARSEABLE_YAML = "run_id: [unclosed\n"


def yaml_error_text(text: str) -> str:
    try:
        yaml.safe_load(text)
    except yaml.YAMLError as exc:
        return str(exc).splitlines()[0]
    raise AssertionError(f"expected {text!r} not to parse")


@pytest.mark.parametrize(
    "body",
    [
        "```yaml\nrun_id: census-dark_factory-20261020\n",
        "```yaml\n" + UNPARSEABLE_YAML + "```\n",
        "```yaml\n- run_id\n- as_of_sha\n```\n",
    ],
    ids=["unclosed-fence", "invalid-yaml", "yaml-list"],
)
def test_method_block_unparseable_or_not_a_mapping_is_malformed(tmp_path, capsys, body):
    write_report(tmp_path / "plans", STEM, bodies={"Method": body})

    rc, lines, verdict = run(tmp_path, capsys)

    assert_nonconforming(rc, verdict)
    assert "## Method yaml block" in verdict["malformed"]
    if UNPARSEABLE_YAML in body:
        error = yaml_error_text(UNPARSEABLE_YAML)
        assert any("## Method yaml block" in line and error in line for line in lines)


def test_method_key_outside_extra_is_malformed(tmp_path, capsys):
    write_report(tmp_path / "plans", STEM, bodies={"Method": method_section({**method_for(STEM), "screened": 3})})

    rc, lines, verdict = run(tmp_path, capsys)

    assert_nonconforming(rc, verdict)
    assert "method.screened" in verdict["malformed"]
    assert any("method.screened" in line and "extra" in line for line in lines[1:-1])


def test_method_key_nested_under_extra_conforms(tmp_path, capsys):
    write_report(tmp_path / "plans", STEM, method={**method_for(STEM), "extra": {"screened": 3}})

    rc, _, verdict = run(tmp_path, capsys)

    assert rc == 0
    assert verdict["verdict"] == "conforms"


def test_record_method_missing_keys_are_named(tmp_path, capsys):
    write_report(tmp_path / "plans", STEM, method=method_lacking("cost"), bodies={"Method": method_section(method_for(STEM))})

    rc, _, verdict = run(tmp_path, capsys)

    assert_nonconforming(rc, verdict)
    assert verdict["missing"] == ["record.method.cost"]


def test_record_without_method_is_named(tmp_path, capsys):
    write_report(tmp_path / "plans", STEM, record_text=json.dumps({"findings": [FINDING]}))

    rc, _, verdict = run(tmp_path, capsys)

    assert_nonconforming(rc, verdict)
    assert "record.method" in verdict["missing"]


CONTRACT_FINDING_FIELDS = (
    "key",
    "area",
    "sub_area",
    "anchor",
    "tags",
    "severity",
    "evidence_source",
    "statement",
    "proposal",
    "verdict",
    "disposition",
    "first_seen",
    "last_seen",
    "supersedes",
)


def finding_lacking(*fields: str, **values: object) -> dict:
    return {k: v for k, v in {**copy.deepcopy(FINDING), **values}.items() if k not in fields}


def test_finding_missing_contract_fields_is_named(tmp_path, capsys):
    second = finding_lacking("proposal", "supersedes", key="fk-aaaaaaaaaaaa")
    write_report(tmp_path / "plans", STEM, findings=[copy.deepcopy(FINDING), second])

    rc, lines, verdict = run(tmp_path, capsys)

    assert_nonconforming(rc, verdict)
    assert "findings[1].proposal" in verdict["missing"]
    assert "findings[1].supersedes" in verdict["missing"]
    assert not [name for name in verdict["missing"] if name.startswith("findings[0]")]
    for field in ("proposal", "supersedes"):
        assert any("fk-aaaaaaaaaaaa" in line and f"findings[1].{field}" in line for line in lines[1:-1])


def test_the_fixture_finding_carries_every_contract_field():
    assert set(FINDING) == set(CONTRACT_FINDING_FIELDS)


@pytest.mark.parametrize("field", CONTRACT_FINDING_FIELDS)
def test_each_contract_field_is_required(tmp_path, capsys, field):
    write_report(tmp_path / "plans", STEM, findings=[finding_lacking(field)])

    rc, _, verdict = run(tmp_path, capsys)

    assert_nonconforming(rc, verdict)
    assert f"findings[0].{field}" in verdict["missing"]


def test_null_valued_optional_fields_conform(tmp_path, capsys):
    finding = {**copy.deepcopy(FINDING), "sub_area": None, "proposal": None}
    write_report(tmp_path / "plans", STEM, findings=[finding])

    rc, _, verdict = run(tmp_path, capsys)

    assert rc == 0
    assert verdict["verdict"] == "conforms"


@pytest.mark.parametrize(
    ("record_doc", "kind", "name"),
    [
        ({}, "missing", "findings"),
        ({"findings": {}}, "malformed", "findings"),
        ({"findings": [FINDING, "x"]}, "malformed", "findings[1]"),
    ],
    ids=["no-findings-key", "findings-not-a-list", "finding-not-an-object"],
)
def test_record_without_findings_list(tmp_path, capsys, record_doc, kind, name):
    write_report(tmp_path / "plans", STEM, record_text=json.dumps({"method": method_for(STEM), **record_doc}))

    rc, _, verdict = run(tmp_path, capsys)

    assert_nonconforming(rc, verdict)
    assert name in verdict[kind]


def test_empty_findings_list_conforms(tmp_path, capsys):
    write_report(tmp_path / "plans", STEM, findings=[])

    rc, lines, verdict = run(tmp_path, capsys)

    assert rc == 0
    assert verdict["verdict"] == "conforms"
    assert "0 findings" in lines[0]


def test_unparseable_record_is_malformed(tmp_path, capsys):
    write_report(tmp_path / "plans", STEM, record_text="{not json")
    with pytest.raises(json.JSONDecodeError) as decode_error:
        json.loads("{not json")

    rc, lines, verdict = run(tmp_path, capsys)

    assert_nonconforming(rc, verdict)
    assert f"plans/{STEM}.json" in verdict["malformed"]
    assert any(f"plans/{STEM}.json" in line and str(decode_error.value) in line for line in lines[1:-1])
