"""The area a finding's anchor belongs to: contract §3's one path → area rule.

docs/quality-findings-contract.md §3 fixes the rule. A path inside a briefing
subproject's member directory (``<key>/`` at the project root) takes that key.
A path under no member takes the area the project's /review-all overlay names
for it, in the "Path → area map" slot that
skills/review-all/references/project-overlay.md defines. Any other path, and
every ``slug:`` anchor, takes ``repo``. The map records the files and rules it
used, and every degradation as a note, so a report can show them (INV-11).
"""
from __future__ import annotations

import fnmatch
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_AREA = "repo"
AREA_KEY_PATTERN = r"^[^/\s]+$"
BRIEFING_RELPATH = "review/briefing.yaml"
OVERLAY_RELPATH = ".claude/skills/review-all/project.md"

_SLOT_TITLE_RE = re.compile(r"Path\s*(?:→|->)\s*area")
_LIST_ITEM_RE = re.compile(r"^(?P<indent>\s*)[-*+]\s")
_RULE_PAIR_RE = re.compile(r"`([^`\s]+)`\s*(?:→|->)\s*`([^`\s]+)`")


@dataclass(frozen=True)
class OverlayRule:
    glob: str
    area: str


@dataclass(frozen=True)
class AreaMap:
    briefing: str | None
    members: tuple[str, ...]
    overlay: str | None
    rules: tuple[OverlayRule, ...]
    notes: tuple[str, ...]

    @property
    def areas(self) -> tuple[str, ...]:
        return tuple(sorted({*self.members, REPO_AREA}))

    def area_of(self, anchor: str) -> str:
        if anchor.startswith("slug:"):
            return REPO_AREA
        path = anchor.partition("::")[0]
        for key in self.members:
            if path == key or path.startswith(key + "/"):
                return key
        matching = [rule for rule in self.rules if fnmatch.fnmatchcase(path, rule.glob)]
        if not matching:
            return REPO_AREA
        return max(matching, key=lambda rule: len(rule.glob)).area

    def to_record(self) -> dict[str, Any]:
        return {
            "briefing": self.briefing,
            "areas": list(self.areas),
            "overlay": self.overlay,
            "rules": [[rule.glob, rule.area] for rule in self.rules],
            "notes": list(self.notes),
        }


class _BriefingUnusable(Exception):
    """Why the briefing yields no members; the message completes a note."""


def parse_overlay_rules(text: str) -> tuple[OverlayRule, ...]:
    slot = "\n".join(_path_area_slot(text.splitlines()))
    return tuple(OverlayRule(glob, area) for glob, area in _RULE_PAIR_RE.findall(slot))


def _path_area_slot(lines: list[str]) -> list[str]:
    for index, line in enumerate(lines):
        item = _LIST_ITEM_RE.match(line)
        if item and _SLOT_TITLE_RE.search(line):
            return [line, *_deeper_lines(lines[index + 1:], len(item["indent"]))]
    return []


def _deeper_lines(lines: list[str], indent: int) -> list[str]:
    deeper: list[str] = []
    for line in lines:
        if line.strip() and len(line) - len(line.lstrip()) <= indent:
            break
        deeper.append(line)
    return deeper


def load_area_map(project_root: str | os.PathLike[str]) -> AreaMap:
    """The area map *project_root* declares. Never raises: a defect becomes a note."""
    root = Path(project_root)
    briefing, members, member_notes = _read_members(root)
    overlay, rules, rule_notes = _read_rules(root, members)
    return AreaMap(
        briefing=briefing,
        members=members,
        overlay=overlay,
        rules=rules,
        notes=(*member_notes, *rule_notes),
    )


def _read_members(root: Path) -> tuple[str | None, tuple[str, ...], tuple[str, ...]]:
    try:
        subprojects = _briefing_subprojects(root / BRIEFING_RELPATH)
    except _BriefingUnusable as reason:
        return None, (), (f"{BRIEFING_RELPATH} {reason}: every area is {REPO_AREA} (contract §3)",)
    members = tuple(key for key in subprojects if re.fullmatch(AREA_KEY_PATTERN, key))
    skipped = tuple(
        f"{BRIEFING_RELPATH}: subproject key {key!r} is not a plain area key; skipped"
        for key in subprojects
        if key not in members
    )
    return BRIEFING_RELPATH, members, skipped


def _briefing_subprojects(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise _BriefingUnusable("absent")
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        raise _BriefingUnusable(f"unreadable: {exc}") from exc
    except yaml.YAMLError as exc:
        raise _BriefingUnusable(f"malformed: {type(exc).__name__}") from exc
    subprojects = doc.get("subprojects") if isinstance(doc, dict) else None
    if not isinstance(subprojects, dict) or not all(isinstance(key, str) for key in subprojects):
        raise _BriefingUnusable("malformed: subprojects is not a mapping with string keys")
    return subprojects


def _read_rules(
    root: Path, members: tuple[str, ...],
) -> tuple[str | None, tuple[OverlayRule, ...], tuple[str, ...]]:
    path = root / OVERLAY_RELPATH
    if not path.is_file():
        return None, (), ()
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return None, (), (f"{OVERLAY_RELPATH} unreadable ({exc}): no overlay rule applies",)
    parsed = parse_overlay_rules(text)
    known = {*members, REPO_AREA}
    kept = tuple(rule for rule in parsed if rule.area in known)
    notes = tuple(
        f"{OVERLAY_RELPATH}: rule `{rule.glob}` → `{rule.area}` dropped: {rule.area!r} is not a briefing area"
        for rule in parsed
        if rule.area not in known
    )
    if not parsed:
        notes = (f"{OVERLAY_RELPATH}: no `glob → area` rule found in a Path → area slot",)
    return OVERLAY_RELPATH, kept, notes
