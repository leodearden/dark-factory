"""The finding key: one stable id per location and lens, minted the same way by every instrument.

docs/quality-findings-contract.md §2 is the spec, and this module is its one
implementation. The interim skills/hotspot-survey/scripts/findings_artefact.py::finding_key
stays as skill-local tooling (Leo, 2026-10-05); shared/tests/test_finding_key.py
holds the two to the same committed fixture.
"""

from __future__ import annotations

import hashlib
import re

KEY_PATTERN = r"^fk-[0-9a-f]{12}$"

_LINE_PIN_RE = re.compile(r":\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*$")


def normalise_anchor(anchor: str) -> str:
    return _LINE_PIN_RE.sub("", anchor.strip().replace("\\", "/"))


def finding_key(area: str, anchor: str, primary_tag: str) -> str:
    canonical = f"{area}|{normalise_anchor(anchor)}|{primary_tag}".lower()
    return "fk-" + hashlib.sha256(canonical.encode()).hexdigest()[:12]


def is_finding_key(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(KEY_PATTERN, value) is not None
