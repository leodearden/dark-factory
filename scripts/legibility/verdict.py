"""The census verifier's typed verdict on one cluster, and the parser that
normalises a verifier reply into one: plans/census-incremental-prd.md §4.4
(C4); docs/quality-findings-contract.md §1 (anchor, tags), §4 (severity) and
§6 (route).
"""
from __future__ import annotations

import enum
import os
import re
import reprlib
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from legibility import codebook, filing_policy

SEVERITIES: tuple[str, ...] = tuple(
    codebook.V2_SCHEMA["properties"]["entries"]["items"]["properties"]["severity"]["enum"]
)
DEFAULT_SEVERITY = "medium"
KIND_TAG = "kind:confusion"


class Route(enum.Enum):
    MECHANICAL = "mechanical"
    STRUCTURAL = "structural"


class NormalisationKind(enum.Enum):
    ANCHOR_REWRITTEN = "anchor_rewritten"
    ANCHOR_FROM_REMEDIATION = "anchor_from_remediation"
    ANCHOR_FROM_TITLE = "anchor_from_title"
    TAG_DROPPED = "tag_dropped"
    SEVERITY_SUBSTITUTED = "severity_substituted"
    SEVERITY_REASON_MISSING = "severity_reason_missing"
    ROUTE_DEFAULTED = "route_defaulted"
    REMEDIATION_REJECTED = "remediation_rejected"
    REASON_MISSING = "reason_missing"


class VerdictError(ValueError):
    """A verifier reply, or a Verdict's fields, outside the C4 contract."""


_BOUNDED = reprlib.Repr()
_BOUNDED.maxstring = _BOUNDED.maxother = 120


def _bounded_repr(value: object) -> str:
    return _BOUNDED.repr(value)


@dataclass(frozen=True)
class Normalisation:
    kind: NormalisationKind
    offered: str
    used: str

    def to_record(self) -> dict[str, str]:
        return {"kind": self.kind.value, "offered": self.offered, "used": self.used}


def _normalised(kind: NormalisationKind, offered: object, used: str) -> tuple[Normalisation, ...]:
    return (Normalisation(kind, _bounded_repr(offered), used),)


_VERIFIER_TAG_RE = re.compile(r"h(?:[1-9]|1[0-4])|comments|tests|inv-\d+|inv-[a-z]+-\d+")
_KEBAB_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_SYMBOL_RE = re.compile(r"[A-Za-z_][\w.]*")
_LINE_PIN_RE = re.compile(r":\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*$")
_SLUG_PREFIX = "slug:"


def _is_verifier_tag(tag: str) -> bool:
    return _VERIFIER_TAG_RE.fullmatch(tag) is not None


def is_finding_tag(tag: str) -> bool:
    """Whether *tag* is in the contract §1 vocabulary this instrument emits."""
    return tag == KIND_TAG or _is_verifier_tag(tag)


def _is_repo_relative_path(path: str) -> bool:
    if "\\" in path or "\x00" in path or path.startswith("/"):
        return False
    return all(segment not in ("", ".", "..") for segment in path.split("/"))


def is_anchor(anchor: str) -> bool:
    """Whether *anchor* has the contract §1 shape: ``slug:<kebab>``, or a
    repo-relative posix path with no line pin and an optional ``::symbol``."""
    if anchor.startswith(_SLUG_PREFIX):
        return _KEBAB_RE.fullmatch(anchor.removeprefix(_SLUG_PREFIX)) is not None
    path, separator, symbol = anchor.partition("::")
    if separator and _SYMBOL_RE.fullmatch(symbol) is None:
        return False
    return _is_repo_relative_path(path) and _LINE_PIN_RE.search(path) is None


@dataclass(frozen=True)
class Remediation:
    path: str
    change: str

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not _is_repo_relative_path(self.path):
            raise VerdictError(
                f"Remediation.path must be a repo-relative posix path, got {_bounded_repr(self.path)}"
            )
        if not isinstance(self.change, str) or not self.change.strip():
            raise VerdictError(
                f"Remediation.change must be a non-empty str, got {_bounded_repr(self.change)}"
            )

    def to_record(self) -> dict[str, str]:
        return {"path": self.path, "change": self.change}


def _are_finding_tags(tags: object) -> bool:
    return (
        isinstance(tags, tuple)
        and all(isinstance(tag, str) and is_finding_tag(tag) for tag in tags)
        and len(set(tags)) == len(tags)
        and tags[-1:] == (KIND_TAG,)
    )


_VERDICT_INVARIANTS: tuple[tuple[str, Callable[[object], bool], str], ...] = (
    ("verified", lambda value: isinstance(value, bool), "a bool"),
    ("reason", lambda value: isinstance(value, str), "a str"),
    (
        "anchor",
        lambda value: isinstance(value, str) and is_anchor(value),
        "slug:<kebab> or a repo-relative path[::symbol] with no line pin",
    ),
    ("tags", _are_finding_tags, f"distinct finding tags ending in {KIND_TAG!r}"),
    ("severity", lambda value: value in SEVERITIES, f"one of {SEVERITIES}"),
    (
        "severity_reason",
        lambda value: isinstance(value, str) and bool(value.strip()),
        "a non-empty str",
    ),
    ("route", lambda value: isinstance(value, Route), "a Route"),
    (
        "remediation",
        lambda value: value is None or isinstance(value, Remediation),
        "a Remediation or None",
    ),
    (
        "normalisations",
        lambda value: isinstance(value, tuple)
        and all(isinstance(note, Normalisation) for note in value),
        "a tuple of Normalisation",
    ),
)


@dataclass(frozen=True)
class Verdict:
    verified: bool
    reason: str
    anchor: str
    tags: tuple[str, ...]
    severity: str
    severity_reason: str
    route: Route
    remediation: Remediation | None
    normalisations: tuple[Normalisation, ...] = ()

    def __post_init__(self) -> None:
        for field_name, holds, expected in _VERDICT_INVARIANTS:
            value = getattr(self, field_name)
            if not holds(value):
                raise VerdictError(
                    f"Verdict.{field_name} must be {expected}, got {_bounded_repr(value)}"
                )

    def to_record(self) -> dict[str, object]:
        return {
            "verified": self.verified,
            "reason": self.reason,
            "anchor": self.anchor,
            "tags": list(self.tags),
            "severity": self.severity,
            "severity_reason": self.severity_reason,
            "route": self.route.value,
            "remediation": None if self.remediation is None else self.remediation.to_record(),
            "normalisations": [note.to_record() for note in self.normalisations],
        }


def count_normalisations(verdicts: Iterable[Verdict]) -> dict[str, int]:
    """Normalisations per kind across *verdicts*, every kind present."""
    counts = {kind.value: 0 for kind in NormalisationKind}
    for found in verdicts:
        for note in found.normalisations:
            counts[note.kind.value] += 1
    return counts


def _verified(raw: Mapping[str, object]) -> bool:
    value = raw.get("verified")
    if not isinstance(value, bool):
        raise VerdictError(
            f"verifier reply field 'verified' must be a JSON bool, got {_bounded_repr(value)}"
        )
    return value


def _reason(raw: object) -> tuple[str, tuple[Normalisation, ...]]:
    return (raw.strip() if isinstance(raw, str) else ""), ()


def _remediation(raw: object, root: Path) -> tuple[Remediation | None, tuple[Normalisation, ...]]:
    shaped = filing_policy.proposed_remediation({"remediation": raw})
    if shaped is None:
        return None, ()
    return Remediation(shaped["path"], shaped["change"].strip()), ()


def _kebab(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _anchor(
    raw: object, *, root: Path, remediation: Remediation | None, title: str,
) -> tuple[str, tuple[Normalisation, ...]]:
    if isinstance(raw, str) and is_anchor(raw.strip()):
        return raw.strip(), ()
    return _SLUG_PREFIX + _kebab(title), ()


def _tags(raw: object) -> tuple[tuple[str, ...], tuple[Normalisation, ...]]:
    offered = raw if isinstance(raw, list) else []
    kept = tuple(tag for tag in offered if isinstance(tag, str) and _is_verifier_tag(tag))
    return (*kept, KIND_TAG), ()


def _severity(
    raw: object, raw_reason: object,
) -> tuple[str, str, tuple[Normalisation, ...]]:
    reason = raw_reason.strip() if isinstance(raw_reason, str) else ""
    if raw in SEVERITIES and reason:
        return str(raw), reason, ()
    return DEFAULT_SEVERITY, f"verifier offered severity {_bounded_repr(raw)}", ()


def _route(
    raw: object, *, remediation: Remediation | None,
) -> tuple[Route, tuple[Normalisation, ...]]:
    if isinstance(raw, str) and raw in {route.value for route in Route}:
        return Route(raw), ()
    return (Route.MECHANICAL if remediation is not None else Route.STRUCTURAL), ()


def parse_verdict(
    raw: Mapping[str, object], *, title: str, project_root: str | os.PathLike[str],
) -> Verdict:
    """The Verdict a verifier reply states, every repair it needed recorded
    on ``Verdict.normalisations``. Raises VerdictError only when ``verified``
    is not a JSON bool: without it the reply is no verdict at all."""
    root = Path(project_root).resolve()
    verified = _verified(raw)
    reason, reason_notes = _reason(raw.get("reason"))
    remediation, remediation_notes = _remediation(raw.get("remediation"), root)
    anchor, anchor_notes = _anchor(
        raw.get("anchor"), root=root, remediation=remediation, title=title,
    )
    tags, tag_notes = _tags(raw.get("tags"))
    severity, severity_reason, severity_notes = _severity(
        raw.get("severity"), raw.get("severity_reason"),
    )
    route, route_notes = _route(raw.get("route"), remediation=remediation)
    return Verdict(
        verified=verified,
        reason=reason,
        anchor=anchor,
        tags=tags,
        severity=severity,
        severity_reason=severity_reason,
        route=route,
        remediation=remediation,
        normalisations=(
            *anchor_notes,
            *tag_notes,
            *severity_notes,
            *route_notes,
            *remediation_notes,
            *reason_notes,
        ),
    )
