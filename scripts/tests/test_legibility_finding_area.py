"""Tests for scripts/legibility/finding_area.py — contract §3's one path → area rule,
read from a project's review briefing and its /review-all overlay."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from legibility import finding_area

REPO_ROOT = Path(__file__).resolve().parents[2]

_NESTED_OVERLAY = """\
# review-all overlay — fixture

- **Briefing**: `review/briefing.yaml` (default); area keys are its `subprojects`.
- **Path → area, paths under no member** — CONFIRMED (fixture):
  - `scripts/legibility/**` → `shared` (the legibility pipeline's home).
  - other `scripts/**` → `repo` (operator and CI helpers with no owning member).
  - `cockpit/` → **not mapped**: a top-level package the briefing omits is a briefing defect.
- **Slice seeds**: `orchestrator` → `merge-lane` (`merge_lane/`).
"""

_REORDERED_OVERLAY = """\
- **Path → area, paths under no member**:
  - other `scripts/**` → `repo`.
  - `scripts/legibility/**` → `shared`.
"""

_SKELETON_OVERLAY = """\
```markdown
- Path → area (paths under no member): `scripts/legibility/**` → `shared`; other `scripts/**` -> `repo`; `<dir>/` → briefing defect.
- Slice seeds: `orchestrator` → `merge-lane`.
```
"""

_UNKNOWN_AREA_OVERLAY = """\
- **Path → area, paths under no member**:
  - `scripts/legibility/**` → `shared`.
  - `tools/**` → `tooling`.
"""

_MEMBER_SHADOW_OVERLAY = """\
- **Path → area, paths under no member**:
  - `orchestrator/**` → `shared`.
"""


def _briefing(root: Path, members: list[str]) -> None:
    lines = ["project: fixture", "subprojects:"]
    lines += [f"  {key}:\n    description: {key} member" for key in members]
    _write(root / "review" / "briefing.yaml", "\n".join(lines) + "\n")


def _overlay(root: Path, text: str) -> None:
    _write(root / ".claude" / "skills" / "review-all" / "project.md", text)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _rules(area_map: finding_area.AreaMap) -> set[tuple[str, str]]:
    return {(rule.glob, rule.area) for rule in area_map.rules}


def test_a_member_directory_contains_its_paths(tmp_path):
    _briefing(tmp_path, ["orchestrator", "shared"])
    area_map = finding_area.load_area_map(tmp_path)
    assert area_map.area_of("orchestrator/src/x.py::f") == "orchestrator"
    assert area_map.area_of("orchestrator") == "orchestrator"
    assert area_map.area_of("orchestratorx/y.py") == "repo"


def test_the_nested_overlay_slot_maps_paths_under_no_member(tmp_path):
    _briefing(tmp_path, ["orchestrator", "shared"])
    _overlay(tmp_path, _NESTED_OVERLAY)
    area_map = finding_area.load_area_map(tmp_path)
    assert area_map.area_of("scripts/legibility/x.py") == "shared"
    assert area_map.area_of("scripts/other.py") == "repo"
    assert area_map.area_of("cockpit/x.py") == "repo"


def test_the_longest_matching_glob_wins_whatever_the_document_order(tmp_path):
    _briefing(tmp_path, ["orchestrator", "shared"])
    _overlay(tmp_path, _REORDERED_OVERLAY)
    area_map = finding_area.load_area_map(tmp_path)
    assert area_map.area_of("scripts/legibility/x.py") == "shared"
    assert area_map.area_of("scripts/other.py") == "repo"


def test_the_one_line_skeleton_slot_parses_to_the_same_rules(tmp_path):
    _briefing(tmp_path, ["orchestrator", "shared"])
    _overlay(tmp_path, _SKELETON_OVERLAY)
    area_map = finding_area.load_area_map(tmp_path)
    assert _rules(area_map) == {("scripts/legibility/**", "shared"), ("scripts/**", "repo")}


def test_backticked_pairs_outside_the_slot_are_not_rules():
    rules = finding_area.parse_overlay_rules(_NESTED_OVERLAY)
    assert {(rule.glob, rule.area) for rule in rules} == {
        ("scripts/legibility/**", "shared"),
        ("scripts/**", "repo"),
    }


def test_a_rule_naming_an_unknown_area_is_dropped_with_a_note(tmp_path):
    _briefing(tmp_path, ["orchestrator", "shared"])
    _overlay(tmp_path, _UNKNOWN_AREA_OVERLAY)
    area_map = finding_area.load_area_map(tmp_path)
    assert _rules(area_map) == {("scripts/legibility/**", "shared")}
    assert area_map.area_of("tools/x.py") == "repo"
    assert any("tools/**" in note and "tooling" in note for note in area_map.notes)


def test_a_member_directory_beats_an_overlay_glob(tmp_path):
    _briefing(tmp_path, ["orchestrator", "shared"])
    _overlay(tmp_path, _MEMBER_SHADOW_OVERLAY)
    area_map = finding_area.load_area_map(tmp_path)
    assert area_map.area_of("orchestrator/src/x.py") == "orchestrator"


def test_slugs_and_unmapped_paths_get_repo(tmp_path):
    _briefing(tmp_path, ["orchestrator", "shared"])
    area_map = finding_area.load_area_map(tmp_path)
    assert area_map.overlay is None
    assert area_map.area_of("slug:merge-lane-practice") == "repo"
    assert area_map.area_of("docs/x.md") == "repo"
    assert area_map.area_of("scripts/legibility/x.py") == "repo"


def test_no_briefing_makes_every_area_repo_and_says_so(tmp_path):
    _overlay(tmp_path, _NESTED_OVERLAY)
    area_map = finding_area.load_area_map(tmp_path)
    assert area_map.briefing is None
    assert area_map.members == ()
    assert area_map.area_of("orchestrator/src/x.py::f") == "repo"
    assert area_map.area_of("scripts/legibility/x.py") == "repo"
    assert any(finding_area.BRIEFING_RELPATH in note for note in area_map.notes)
    assert any("scripts/legibility/**" in note and "shared" in note for note in area_map.notes)


@pytest.mark.parametrize(
    "briefing_text, problem",
    [
        ("subprojects: [orchestrator\n", "malformed"),
        ("subprojects:\n  - orchestrator\n  - shared\n", "subprojects"),
    ],
    ids=["invalid-yaml", "subprojects-is-a-list"],
)
def test_a_malformed_briefing_behaves_like_no_briefing(tmp_path, briefing_text, problem):
    _write(tmp_path / "review" / "briefing.yaml", briefing_text)
    area_map = finding_area.load_area_map(tmp_path)
    assert area_map.briefing is None
    assert area_map.members == ()
    assert area_map.area_of("orchestrator/src/x.py") == "repo"
    assert any(finding_area.BRIEFING_RELPATH in note and problem in note for note in area_map.notes)


def test_to_record_is_json_safe_and_names_what_the_map_used(tmp_path):
    _briefing(tmp_path, ["shared", "orchestrator"])
    _overlay(tmp_path, _UNKNOWN_AREA_OVERLAY)
    record = finding_area.load_area_map(tmp_path).to_record()
    assert json.loads(json.dumps(record)) == record
    assert record["briefing"] == finding_area.BRIEFING_RELPATH
    assert record["areas"] == ["orchestrator", "repo", "shared"]
    assert record["overlay"] == finding_area.OVERLAY_RELPATH
    assert record["rules"] == [["scripts/legibility/**", "shared"]]
    assert record["notes"] and all(isinstance(note, str) for note in record["notes"])


def test_to_record_names_no_overlay_as_none(tmp_path):
    _briefing(tmp_path, ["shared"])
    assert finding_area.load_area_map(tmp_path).to_record()["overlay"] is None


def test_every_area_is_a_plain_key(tmp_path):
    _briefing(tmp_path, ["orchestrator", "shared"])
    _overlay(tmp_path, _NESTED_OVERLAY)
    area_map = finding_area.load_area_map(tmp_path)
    anchors = [
        "orchestrator/src/orchestrator/merge_lane/gates.py::Gate.check",
        "shared/src/shared/x.py",
        "scripts/legibility/census.py::run_census",
        "scripts/x.py",
        "slug:a-practice",
        "docs/x.md",
        "orchestrator",
    ]
    for anchor in anchors:
        assert re.fullmatch(finding_area.AREA_KEY_PATTERN, area_map.area_of(anchor)), anchor


def test_this_checkouts_briefing_and_overlay_give_the_contract_areas():
    area_map = finding_area.load_area_map(REPO_ROOT)
    assert area_map.area_of("orchestrator/src/orchestrator/harness.py::run") == "orchestrator"
    assert area_map.area_of("fused-memory/src/x.py") == "fused-memory"
    assert area_map.area_of("scripts/legibility/x.py") == "shared"
    assert area_map.area_of("scripts/foo.py") == "repo"
    assert area_map.area_of("cockpit/x.py") == "repo"
    assert set(area_map.areas) == {
        "fused-memory", "orchestrator", "shared", "dashboard", "escalation", "sampler", "repo",
    }
