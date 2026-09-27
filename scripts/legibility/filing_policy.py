"""Whether and where the legibility census files a verified confusion cluster.

A census observes one project, but a confusion seen there can be fixable only
in the harness that serves it (dark-factory's fused-memory, escalation server,
orchestrator). Filing such a cluster into the observed project hands its
curator a task it cannot act on.

Routing favours precision over recall. A missed harness marker leaves the
cluster where it was observed, which was the behaviour before routing existed.
A false marker would move a hosted project's own confusion into the harness.

Markers read a cluster's DESCRIPTIVE fields (what the confusion is about),
not its evidence. Evidence quotes are verbatim transcript excerpts, and
harness tool calls show up in them incidentally. The one evidence marker is a
harness error that can only come from the harness itself.
"""
from __future__ import annotations

import enum
import logging
import re
from dataclasses import dataclass

logger = logging.getLogger("legibility.filing_policy")


@dataclass(frozen=True)
class ProjectRef:
    project_root: str
    project_id: str


@dataclass(frozen=True)
class FixSurfaceMatch:
    component: str
    evidence: str


@dataclass(frozen=True)
class FilingTarget:
    project: ProjectRef
    fix_surface: tuple[FixSurfaceMatch, ...] = ()


class _Scope(enum.Enum):
    DESCRIPTIVE = ("title", "cause", "summary", "area")
    EVIDENCE = ("evidence",)


@dataclass(frozen=True)
class _Marker:
    component: str
    pattern: re.Pattern[str]
    scope: _Scope


_MARKERS: tuple[_Marker, ...] = (
    _Marker(
        "fused-memory",
        re.compile(r"\bfused[-_]memory(?![-\w])", re.IGNORECASE),
        _Scope.DESCRIPTIVE,
    ),
    _Marker("fused-memory-mcp-tool", re.compile(r"\bmcp__fused-memory__\w+"), _Scope.DESCRIPTIVE),
    _Marker("escalation-mcp-tool", re.compile(r"\bmcp__escalation__\w+"), _Scope.DESCRIPTIVE),
    _Marker(
        "escalation-mcp-server",
        re.compile(r"\bescalation[- ]mcp\b", re.IGNORECASE),
        _Scope.DESCRIPTIVE,
    ),
    _Marker(
        "dark-factory-package-source",
        re.compile(r"(?<![\w./-])(?:orchestrator|escalation|shared|fused-memory)/src/"),
        _Scope.DESCRIPTIVE,
    ),
    _Marker(
        "legibility-census-scripts",
        re.compile(r"(?<![\w./-])scripts/legibility/"),
        _Scope.DESCRIPTIVE,
    ),
    _Marker(
        "reconciliation-verifier-pseudo-tool",
        re.compile(
            r"No such tool available: (?:read_file|grep_search|glob_search|git_log|git_show)\b"
        ),
        _Scope.EVIDENCE,
    ),
)
"""Harness fix-surface markers, in reporting order.

The pseudo-tool names are the reconciliation CodebaseVerifier's own tools,
defined in fused-memory/src/fused_memory/reconciliation/verify.py. A bare
tool name is not a marker, because hosted projects use those identifiers
themselves.
"""


def _checkout_path_marker(harness_root: str) -> _Marker:
    return _Marker(
        "dark-factory-checkout-path",
        re.compile(re.escape(harness_root.rstrip("/") + "/")),
        _Scope.DESCRIPTIVE,
    )


def _scanned_texts(cluster: dict, scope: _Scope) -> list[str]:
    texts: list[str] = []
    for field_name in scope.value:
        value = cluster.get(field_name)
        items = value if isinstance(value, list) else [value]
        texts.extend(item for item in items if isinstance(item, str))
    return texts


def _first_match(marker: _Marker, cluster: dict) -> FixSurfaceMatch | None:
    for text in _scanned_texts(cluster, marker.scope):
        found = marker.pattern.search(text)
        if found:
            return FixSurfaceMatch(component=marker.component, evidence=found.group(0))
    return None


def harness_fix_surface(cluster: dict, *, harness_root: str) -> tuple[FixSurfaceMatch, ...]:
    """One match per harness marker the cluster fires, in marker order."""
    markers = (*_MARKERS, _checkout_path_marker(harness_root))
    matches = (_first_match(marker, cluster) for marker in markers)
    return tuple(match for match in matches if match is not None)


def proposed_remediation(cluster: dict) -> dict | None:
    """The cluster's in-tree remediation, when it has the ``{path, change}`` shape."""
    remediation = cluster.get("remediation")
    if not isinstance(remediation, dict):
        return None
    for key in ("path", "change"):
        value = remediation.get(key)
        if not isinstance(value, str) or not value.strip():
            return None
    return remediation


def _override_project(cluster: dict) -> ProjectRef | None:
    """The explicit ``target_project_root``/``target_project_id`` pair (PRD
    decision 4). The two name one project and move together: a partial pair
    is warned and treated as absent, never mixed with the observed project."""
    has_root = "target_project_root" in cluster
    has_id = "target_project_id" in cluster
    if has_root and has_id:
        return ProjectRef(cluster["target_project_root"], cluster["target_project_id"])
    if has_root or has_id:
        logger.warning(
            "census: cluster %r supplies only one of "
            "target_project_root/target_project_id (must move together, "
            "PRD decision 4) -- ignoring the partial override",
            cluster.get("title"),
        )
    return None


def resolve_target(
    cluster: dict, *, observed: ProjectRef, harness: ProjectRef | None,
) -> FilingTarget:
    """Where *cluster* files. An explicit override pair wins. Otherwise a
    harness fix surface moves it to *harness*, unless the verifier proposed a
    remediation inside the observed tree (an ambiguous cluster stays put)."""
    override = _override_project(cluster)
    if override is not None:
        return FilingTarget(override)
    if harness is None or harness.project_id == observed.project_id:
        return FilingTarget(observed)
    if proposed_remediation(cluster) is not None:
        return FilingTarget(observed)
    fix_surface = harness_fix_surface(cluster, harness_root=harness.project_root)
    if fix_surface:
        return FilingTarget(harness, fix_surface)
    return FilingTarget(observed)


MIN_UNREMEDIATED_SIGHTINGS = 2


@dataclass(frozen=True)
class WithheldCluster:
    title: str | None
    sighting_count: int


def is_fileable(cluster: dict, *, sighting_count: int) -> bool:
    """A verified cluster files when the verifier proposed an in-tree
    remediation, or when it has recurred; a bare singleton is only recorded."""
    return (
        proposed_remediation(cluster) is not None
        or sighting_count >= MIN_UNREMEDIATED_SIGHTINGS
    )
