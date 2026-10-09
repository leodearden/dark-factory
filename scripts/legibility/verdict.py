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
    SEVERITY_MISSING = "severity_missing"
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


_HEURISTIC_TAGS = tuple(f"h{number}" for number in range(1, 15))
_STANCE_TAGS = ("comments", "tests")
_INVARIANT_TAG_RE = re.compile(r"inv-(?:[a-z]+-)?\d+")
_KEBAB_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_SYMBOL_RE = re.compile(r"[A-Za-z_][\w.]*")
_LINE_PIN_RE = re.compile(r":\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*$")
_SLUG_PREFIX = "slug:"


def _is_verifier_tag(tag: str) -> bool:
    return (
        tag in _HEURISTIC_TAGS
        or tag in _STANCE_TAGS
        or _INVARIANT_TAG_RE.fullmatch(tag) is not None
    )


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
    if isinstance(raw, str):
        return raw.strip(), ()
    return "", _normalised(NormalisationKind.REASON_MISSING, raw, "")


def _path_below_root(raw: str, root: Path) -> str | None:
    """*raw* as a posix path relative to *root*, or ``None`` unless it names
    something that EXISTS below that root. The root itself names no fix
    surface: it would let a vague reply pass the singleton gate."""
    try:
        target = (root / raw).resolve()
        below_root = target != root and target.is_relative_to(root) and target.exists()
    except (OSError, ValueError):
        return None
    if not below_root:
        return None
    relative = target.relative_to(root).as_posix()
    return relative if _is_repo_relative_path(relative) else None


def _remediation(raw: object, root: Path) -> tuple[Remediation | None, tuple[Normalisation, ...]]:
    if raw is None:
        return None, ()
    shaped = filing_policy.proposed_remediation({"remediation": raw})
    path = None if shaped is None else _path_below_root(shaped["path"], root)
    if shaped is None or path is None:
        offered = raw.get("path") if isinstance(raw, dict) else raw
        return None, _normalised(NormalisationKind.REMEDIATION_REJECTED, offered, "")
    return Remediation(path, shaped["change"].strip()), ()


def _kebab(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


_UNTITLED_SLUG = "untitled"


def _repaired_anchor(anchor: str, root: Path) -> str | None:
    if anchor.startswith(_SLUG_PREFIX):
        slug = _kebab(anchor.removeprefix(_SLUG_PREFIX))
        return _SLUG_PREFIX + slug if slug else None
    unpinned = _LINE_PIN_RE.sub("", anchor.replace("\\", "/"))
    raw_path, _, symbol = unpinned.partition("::")
    path = _path_below_root(_LINE_PIN_RE.sub("", raw_path), root)
    if path is None:
        return None
    repaired = f"{path}::{symbol}" if _SYMBOL_RE.fullmatch(symbol) else path
    return repaired if is_anchor(repaired) else None


def _anchor(
    raw: object, *, root: Path, remediation: Remediation | None, title: str,
) -> tuple[str, tuple[Normalisation, ...]]:
    stripped = raw.strip() if isinstance(raw, str) else None
    repaired = None if stripped is None else _repaired_anchor(stripped, root)
    if repaired is not None:
        notes = () if repaired == stripped else _normalised(
            NormalisationKind.ANCHOR_REWRITTEN, raw, repaired,
        )
        return repaired, notes
    if remediation is not None:
        return remediation.path, _normalised(
            NormalisationKind.ANCHOR_FROM_REMEDIATION, raw, remediation.path,
        )
    slug = _SLUG_PREFIX + (_kebab(title) or _UNTITLED_SLUG)
    return slug, _normalised(NormalisationKind.ANCHOR_FROM_TITLE, raw, slug)


def _tags(raw: object) -> tuple[tuple[str, ...], tuple[Normalisation, ...]]:
    if raw is None:
        return (KIND_TAG,), ()
    if not isinstance(raw, list):
        return (KIND_TAG,), _normalised(NormalisationKind.TAG_DROPPED, raw, "")
    kept: list[str] = []
    dropped: list[Normalisation] = []
    for tag in raw:
        canonical = tag.strip().lower() if isinstance(tag, str) else ""
        if _is_verifier_tag(canonical) and canonical not in kept:
            kept.append(canonical)
        else:
            dropped.extend(_normalised(NormalisationKind.TAG_DROPPED, tag, ""))
    return (*kept, KIND_TAG), tuple(dropped)


def _default_severity_note(raw: object) -> tuple[NormalisationKind, str]:
    if raw is None:
        return (
            NormalisationKind.SEVERITY_MISSING,
            f"verifier gave no severity; census used {DEFAULT_SEVERITY!r}",
        )
    return NormalisationKind.SEVERITY_SUBSTITUTED, (
        f"verifier offered severity {_bounded_repr(raw)}, outside {SEVERITIES} "
        f"(contract §4: critical is never a finding severity); census used "
        f"{DEFAULT_SEVERITY!r}"
    )


def _severity(
    raw: object, raw_reason: object,
) -> tuple[str, str, tuple[Normalisation, ...]]:
    reason = raw_reason.strip() if isinstance(raw_reason, str) else ""
    canonical = raw.strip().lower() if isinstance(raw, str) else ""
    if canonical not in SEVERITIES:
        kind, substitution = _default_severity_note(raw)
        return (
            DEFAULT_SEVERITY,
            f"{substitution}; verifier's reason: {reason}" if reason else substitution,
            _normalised(kind, raw, DEFAULT_SEVERITY),
        )
    if reason:
        return canonical, reason, ()
    stated = f"verifier gave severity {canonical!r} without a severity_reason"
    return canonical, stated, _normalised(
        NormalisationKind.SEVERITY_REASON_MISSING, raw_reason, stated,
    )


_ROUTES_BY_VALUE = {route.value: route for route in Route}


def _route(
    raw: object, *, remediation: Remediation | None,
) -> tuple[Route, tuple[Normalisation, ...]]:
    named = _ROUTES_BY_VALUE.get(raw.strip().lower()) if isinstance(raw, str) else None
    if named is not None:
        return named, ()
    default = Route.MECHANICAL if remediation is not None else Route.STRUCTURAL
    return default, _normalised(NormalisationKind.ROUTE_DEFAULTED, raw, default.value)


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


_SEVERITY_MEANINGS = {
    "high": "the next change in this area is likely to be wrong or expensive without the fix",
    "medium": "a real cost, contained",
    "low": "worth doing when the area is next touched",
}
"""docs/quality-findings-contract.md §4's one-line meaning of each SEVERITIES value."""

_ROUTE_MEANINGS = {
    Route.MECHANICAL: "one anchor, no design choice; a competent agent can fix it from the reason",
    Route.STRUCTURAL: (
        "spans modules, chooses between designs, changes a contract, or proposes a split"
    ),
}
"""docs/quality-findings-contract.md §6's meaning of each Route."""


def _require_a_meaning_for_each(
    table: str, stated: Iterable[str], required: Iterable[str],
) -> None:
    stated_values, required_values = sorted(stated), sorted(required)
    if stated_values != required_values:
        raise RuntimeError(
            f"verdict.{table} must state a meaning for exactly {required_values}, "
            f"got {stated_values}"
        )


_require_a_meaning_for_each("_SEVERITY_MEANINGS", _SEVERITY_MEANINGS, SEVERITIES)
_require_a_meaning_for_each(
    "_ROUTE_MEANINGS",
    (route.value for route in _ROUTE_MEANINGS),
    (route.value for route in Route),
)

_TAG_VOCABULARY = (
    f"tags: {_HEURISTIC_TAGS[0]}..{_HEURISTIC_TAGS[-1]} for the heuristics of the "
    "quality definition the finding rests on, by number, the heuristic it most "
    f"rests on FIRST; {' or '.join(_STANCE_TAGS)} for its two stances; inv-<id> for "
    "an invariant of this tree's docs/legibility/design-invariants.md.\n"
)
_SEVERITY_LINES = "".join(
    f'  "{severity}": {_SEVERITY_MEANINGS[severity]}.\n' for severity in SEVERITIES
)
_ROUTE_LINES = "".join(f'  "{route.value}": {_ROUTE_MEANINGS[route]}.\n' for route in Route)


def reply_instructions(project_root: str) -> str:
    """The verify prompt's statement of the C4 reply: its JSON shape and the
    anchors, tags, severities and routes ``parse_verdict`` keeps."""
    severities = " | ".join(f'"{severity}"' for severity in SEVERITIES)
    routes = " | ".join(f'"{route.value}"' for route in Route)
    return (
        "Respond with STRICT JSON ONLY (no prose, no markdown fences), for a "
        "verified and a refuted claim alike, exactly this shape: "
        '{"verified": true|false, "reason": "...", '
        '"anchor": "path/to/file.py::symbol | path/to/file | slug:<kebab>", '
        '"tags": ["h<n>", ...], '
        f'"severity": {severities}, "severity_reason": "...", "route": {routes}, '
        f'"remediation": {{"path": "<path relative to {project_root}>", '
        '"change": "<one sentence>"} | null}.\n\n'
        f"anchor: relative to {project_root}; the enclosing top-level or "
        "class-level definition, else the file, else slug:<kebab> for a non-code "
        "subject (a prompt, a contract, an operating practice).\n"
        + _TAG_VOCABULARY
        + "severity: never critical, which is reserved for operational breakage.\n"
        + _SEVERITY_LINES
        + "route:\n"
        + _ROUTE_LINES
    )
