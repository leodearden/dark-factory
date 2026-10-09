"""Tests for scripts/legibility/verdict.py — the census verifier's typed
verdict and its parser (plans/census-incremental-prd.md §4.4 C4, §4.8 row 6).

Every test drives the public surface (``parse_verdict``, ``Verdict``,
``count_normalisations``) against a real tmp_path tree.
"""
from __future__ import annotations

import dataclasses
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import codebook
import pytest
from legibility import verdict

_FULL_RAW = {
    "verified": True,
    "reason": "seen",
    "anchor": "pkg/mod.py::f",
    "tags": ["h13", "inv-11", "tests"],
    "severity": "high",
    "severity_reason": "next change likely wrong",
    "route": "mechanical",
    "remediation": {"path": "pkg/mod.py", "change": " Rename f "},
}


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text("def f():\n    pass\n")
    return tmp_path


def _parse(raw: dict, root: Path, *, title: str = "A confusion") -> verdict.Verdict:
    return verdict.parse_verdict(raw, title=title, project_root=root)


def _valid_fields(**overrides) -> dict:
    fields = {
        "verified": True,
        "reason": "seen",
        "anchor": "pkg/mod.py::f",
        "tags": ("h13", verdict.KIND_TAG),
        "severity": "high",
        "severity_reason": "next change likely wrong",
        "route": verdict.Route.MECHANICAL,
        "remediation": verdict.Remediation("pkg/mod.py", "Rename f"),
    }
    fields.update(overrides)
    return fields


def test_parse_verdict_keeps_a_fully_valid_reply_verbatim(tree: Path) -> None:
    v = _parse(_FULL_RAW, tree)

    assert v.verified is True
    assert v.reason == "seen"
    assert v.anchor == "pkg/mod.py::f"
    assert v.tags == ("h13", "inv-11", "tests", "kind:confusion")
    assert v.severity == "high"
    assert v.severity_reason == "next change likely wrong"
    assert v.route is verdict.Route.MECHANICAL
    assert v.remediation == verdict.Remediation("pkg/mod.py", "Rename f")
    assert v.normalisations == ()


def test_parse_verdict_parses_a_refuted_reply(tree: Path) -> None:
    v = _parse({**_FULL_RAW, "verified": False}, tree)

    assert v.verified is False
    assert v.normalisations == ()


@pytest.mark.parametrize("offered", [None, "true", 1])
def test_parse_verdict_rejects_a_non_bool_verified(tree: Path, offered) -> None:
    with pytest.raises(verdict.VerdictError) as excinfo:
        _parse({**_FULL_RAW, "verified": offered}, tree)

    assert isinstance(excinfo.value, ValueError)
    assert "verified" in str(excinfo.value)
    assert repr(offered) in str(excinfo.value)


def test_parse_verdict_rejects_an_absent_verified(tree: Path) -> None:
    raw = {k: v for k, v in _FULL_RAW.items() if k != "verified"}

    with pytest.raises(verdict.VerdictError, match="verified"):
        _parse(raw, tree)


def _codebook_with_severity(severity: str) -> dict:
    return {
        "version": 2,
        "entries": [
            {
                "id": "entry-a",
                "title": "Some confusion cluster",
                "severity": severity,
                "status": "open",
                "origin_phase": "unknown",
                "manifested_phase": "unknown",
                "sightings": [],
            },
        ],
    }


def test_severities_are_exactly_the_codebook_entry_severities() -> None:
    for severity in verdict.SEVERITIES:
        assert codebook.validate(_codebook_with_severity(severity)) == []
    for severity in ("critical", "urgent"):
        assert severity not in verdict.SEVERITIES
        assert codebook.validate(_codebook_with_severity(severity)) != []
    assert verdict.DEFAULT_SEVERITY == "medium"
    assert verdict.DEFAULT_SEVERITY in verdict.SEVERITIES


@pytest.mark.parametrize(
    ("field", "offered"),
    [
        ("severity", "critical"),
        ("severity_reason", ""),
        ("tags", ("h13",)),
        ("tags", ("h13", "kind:confusion", "kind:confusion")),
        ("tags", ("kind:confusion", "h13")),
        ("tags", ("bogus", "kind:confusion")),
        ("anchor", "a.py:12"),
        ("anchor", "pkg\\mod.py"),
        ("route", "mechanical"),
    ],
)
def test_verdict_construction_enforces_its_invariants(field: str, offered) -> None:
    with pytest.raises(verdict.VerdictError) as excinfo:
        verdict.Verdict(**_valid_fields(**{field: offered}))

    assert field in str(excinfo.value)
    assert repr(offered) in str(excinfo.value)


def test_verdict_construction_accepts_valid_fields() -> None:
    v = verdict.Verdict(**_valid_fields())

    assert v.normalisations == ()


def test_verdict_is_frozen() -> None:
    v = verdict.Verdict(**_valid_fields())

    with pytest.raises(dataclasses.FrozenInstanceError):
        v.severity = "low"  # type: ignore[misc]


def test_count_normalisations_of_nothing_names_every_kind_at_zero() -> None:
    assert verdict.count_normalisations([]) == {
        kind.value: 0 for kind in verdict.NormalisationKind
    }


def test_to_record_is_json_safe(tree: Path) -> None:
    record = _parse(_FULL_RAW, tree).to_record()

    assert json.loads(json.dumps(record)) == record
    assert record["route"] == "mechanical"
    assert record["tags"] == ["h13", "inv-11", "tests", "kind:confusion"]
    assert record["remediation"] == {"path": "pkg/mod.py", "change": "Rename f"}


def test_to_record_of_a_verdict_without_remediation_carries_none() -> None:
    record = verdict.Verdict(**_valid_fields(remediation=None)).to_record()

    assert json.loads(json.dumps(record)) == record
    assert record["remediation"] is None


# ---------------------------------------------------------------------------
# Anchor and remediation normalisation, each repair counted (C4).
# ---------------------------------------------------------------------------

_TITLE = "Docs omit the X convention!"
_TITLE_SLUG = "slug:docs-omit-the-x-convention"
_GUIDE = {"path": "docs/guide.md", "change": "Document X"}


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """An observed tree holding pkg/mod.py and docs/guide.md, with
    tmp_path/outside.py beside it, outside the tree."""
    tree_root = tmp_path / "tree"
    (tree_root / "pkg").mkdir(parents=True)
    (tree_root / "pkg" / "mod.py").write_text("def f():\n    pass\n")
    (tree_root / "docs").mkdir()
    (tree_root / "docs" / "guide.md").write_text("guide\n")
    (tmp_path / "outside.py").write_text("outside\n")
    return tree_root


def _kinds(v: verdict.Verdict) -> list[verdict.NormalisationKind]:
    return [note.kind for note in v.normalisations]


@pytest.mark.parametrize(
    ("offered", "expected"),
    [
        pytest.param("{root}/pkg/mod.py::f", "pkg/mod.py::f", id="absolute"),
        pytest.param("pkg\\mod.py::f", "pkg/mod.py::f", id="backslash"),
        pytest.param("pkg/mod.py:12", "pkg/mod.py", id="line-pin"),
        pytest.param("pkg/mod.py:3-9", "pkg/mod.py", id="range-pin"),
        pytest.param("pkg/mod.py:1,4-5", "pkg/mod.py", id="list-pin"),
        pytest.param("pkg/mod.py::f:12", "pkg/mod.py::f", id="symbol-pin"),
        pytest.param("pkg/mod.py::f()", "pkg/mod.py", id="non-identifier-symbol"),
        pytest.param("slug:Prompt Contract", "slug:prompt-contract", id="slug-kebabed"),
    ],
)
def test_parse_verdict_rewrites_a_repairable_anchor_and_counts_it(
    root: Path, offered: str, expected: str,
) -> None:
    raw_anchor = offered.format(root=root)
    v = _parse({**_FULL_RAW, "anchor": raw_anchor}, root)

    assert v.anchor == expected
    assert v.normalisations == (
        verdict.Normalisation(
            verdict.NormalisationKind.ANCHOR_REWRITTEN, repr(raw_anchor), expected,
        ),
    )


@pytest.mark.parametrize(
    ("offered", "expected"),
    [
        pytest.param(" pkg/mod.py ", "pkg/mod.py", id="surrounding-whitespace"),
        pytest.param("slug:valid-one", "slug:valid-one", id="valid-slug"),
        pytest.param("pkg", "pkg", id="directory"),
    ],
)
def test_parse_verdict_keeps_a_canonical_anchor_uncounted(
    root: Path, offered: str, expected: str,
) -> None:
    v = _parse({**_FULL_RAW, "anchor": offered}, root)

    assert v.anchor == expected
    assert v.normalisations == ()


_INVALID_ANCHORS = [
    pytest.param(lambda root: "pkg/missing.py::f", id="nonexistent"),
    pytest.param(lambda root: "../outside.py", id="outside-dot-dot"),
    pytest.param(lambda root: str(root.parent / "outside.py"), id="outside-absolute"),
    pytest.param(lambda root: ".", id="root-dot"),
    pytest.param(lambda root: "./", id="root-dot-slash"),
    pytest.param(lambda root: "pkg/..", id="root-up"),
    pytest.param(lambda root: "pkg/\x00mod.py", id="nul-byte"),
    pytest.param(lambda root: "x" * 5000, id="too-long"),
    pytest.param(lambda root: None, id="null"),
    pytest.param(lambda root: 7, id="non-str"),
    pytest.param(lambda root: "", id="empty"),
]


def _without(raw: dict, key: str) -> dict:
    return {k: v for k, v in raw.items() if k != key}


@pytest.mark.parametrize("anchor_of", _INVALID_ANCHORS)
def test_parse_verdict_falls_back_to_the_remediation_path(root: Path, anchor_of) -> None:
    raw = {**_FULL_RAW, "anchor": anchor_of(root), "remediation": _GUIDE}

    v = _parse(raw, root, title=_TITLE)

    assert v.anchor == "docs/guide.md"
    assert _kinds(v) == [verdict.NormalisationKind.ANCHOR_FROM_REMEDIATION]
    assert v.normalisations[0].used == "docs/guide.md"


def test_parse_verdict_falls_back_to_the_remediation_path_for_an_absent_anchor(
    root: Path,
) -> None:
    v = _parse({**_without(_FULL_RAW, "anchor"), "remediation": _GUIDE}, root)

    assert v.anchor == "docs/guide.md"
    assert _kinds(v) == [verdict.NormalisationKind.ANCHOR_FROM_REMEDIATION]


@pytest.mark.parametrize("anchor_of", _INVALID_ANCHORS)
def test_parse_verdict_falls_back_to_the_title_slug(root: Path, anchor_of) -> None:
    raw = {**_FULL_RAW, "anchor": anchor_of(root), "remediation": None}

    v = _parse(raw, root, title=_TITLE)

    assert v.anchor == _TITLE_SLUG
    assert _kinds(v) == [verdict.NormalisationKind.ANCHOR_FROM_TITLE]
    assert v.normalisations[0].used == _TITLE_SLUG


def test_parse_verdict_falls_back_to_the_title_slug_for_an_absent_anchor(root: Path) -> None:
    v = _parse(_without(_without(_FULL_RAW, "anchor"), "remediation"), root, title=_TITLE)

    assert v.anchor == _TITLE_SLUG
    assert _kinds(v) == [verdict.NormalisationKind.ANCHOR_FROM_TITLE]


@pytest.mark.parametrize("title", ["", "!!! ???", "日本語"])
def test_parse_verdict_title_slug_is_well_formed_for_any_title(root: Path, title: str) -> None:
    v = _parse({**_FULL_RAW, "anchor": None, "remediation": None}, root, title=title)

    assert re.fullmatch(r"slug:[a-z0-9]+(?:-[a-z0-9]+)*", v.anchor)
    assert _kinds(v) == [verdict.NormalisationKind.ANCHOR_FROM_TITLE]


@pytest.mark.parametrize("absolute", [False, True], ids=["relative", "absolute"])
def test_parse_verdict_keeps_an_in_tree_remediation_repo_relative(
    root: Path, absolute: bool,
) -> None:
    path = str(root / "docs" / "guide.md") if absolute else "docs/guide.md"

    v = _parse({**_FULL_RAW, "remediation": {"path": path, "change": "  Document X "}}, root)

    assert v.remediation == verdict.Remediation("docs/guide.md", "Document X")
    assert v.normalisations == ()


def test_parse_verdict_resolves_a_symlinked_remediation_to_its_in_tree_target(
    root: Path,
) -> None:
    (root / "link.md").symlink_to(Path("docs") / "guide.md")

    v = _parse({**_FULL_RAW, "remediation": {"path": "link.md", "change": "Document X"}}, root)

    assert v.remediation == verdict.Remediation("docs/guide.md", "Document X")


_REJECTED_REMEDIATION_PATHS = [
    pytest.param(lambda root: "docs/missing.md", id="nonexistent"),
    pytest.param(lambda root: "../outside.py", id="outside-dot-dot"),
    pytest.param(lambda root: str(root.parent / "outside.py"), id="outside-absolute"),
    pytest.param(lambda root: ".", id="root-dot"),
    pytest.param(lambda root: "./", id="root-dot-slash"),
    pytest.param(lambda root: "docs/..", id="root-up"),
    pytest.param(str, id="root-absolute"),
    pytest.param(lambda root: "docs/a\x00b.md", id="nul-byte"),
]


@pytest.mark.parametrize("path_of", _REJECTED_REMEDIATION_PATHS)
def test_parse_verdict_rejects_an_out_of_tree_remediation_and_counts_it(
    root: Path, path_of,
) -> None:
    path = path_of(root)

    v = _parse({**_FULL_RAW, "remediation": {"path": path, "change": "Document X"}}, root)

    assert v.remediation is None
    assert v.normalisations == (
        verdict.Normalisation(verdict.NormalisationKind.REMEDIATION_REJECTED, repr(path), ""),
    )


def test_parse_verdict_rejects_a_too_long_remediation_path_with_a_bounded_offer(
    root: Path,
) -> None:
    v = _parse({**_FULL_RAW, "remediation": {"path": "x" * 5000, "change": "Document X"}}, root)

    assert v.remediation is None
    [note] = v.normalisations
    assert note.kind is verdict.NormalisationKind.REMEDIATION_REJECTED
    assert note.offered.startswith("'xxx")
    assert len(note.offered) < 200


@pytest.mark.parametrize(
    ("remediation", "offered"),
    [
        pytest.param({"path": "docs/guide.md"}, repr("docs/guide.md"), id="no-change"),
        pytest.param(
            {"path": "docs/guide.md", "change": "  "}, repr("docs/guide.md"), id="blank-change",
        ),
        pytest.param({"change": "Document X"}, repr(None), id="no-path"),
        pytest.param("docs/guide.md", repr("docs/guide.md"), id="bare-string"),
        pytest.param(["docs/guide.md"], repr(["docs/guide.md"]), id="list"),
    ],
)
def test_parse_verdict_rejects_a_malformed_remediation_and_counts_it(
    root: Path, remediation, offered: str,
) -> None:
    v = _parse({**_FULL_RAW, "remediation": remediation}, root)

    assert v.remediation is None
    assert v.normalisations == (
        verdict.Normalisation(verdict.NormalisationKind.REMEDIATION_REJECTED, offered, ""),
    )


def test_parse_verdict_takes_a_null_or_absent_remediation_uncounted(root: Path) -> None:
    for raw in ({**_FULL_RAW, "remediation": None}, _without(_FULL_RAW, "remediation")):
        v = _parse(raw, root)

        assert v.remediation is None
        assert v.normalisations == ()


def test_count_normalisations_sums_each_kind_across_verdicts(root: Path) -> None:
    pinned = _parse({**_FULL_RAW, "anchor": "pkg/mod.py:12"}, root)
    pinned_and_rejected = _parse(
        {
            **_FULL_RAW,
            "anchor": "pkg/mod.py:3",
            "remediation": {"path": "docs/missing.md", "change": "Document X"},
        },
        root,
    )

    counts = verdict.count_normalisations([pinned, pinned_and_rejected])

    assert counts == {
        **{kind.value: 0 for kind in verdict.NormalisationKind},
        "anchor_rewritten": 2,
        "remediation_rejected": 1,
    }


# ---------------------------------------------------------------------------
# Tags, severity, route and reason normalisation (PRD §4.8 row 6).
# ---------------------------------------------------------------------------

_K = verdict.NormalisationKind


def _offers(v: verdict.Verdict, kind: verdict.NormalisationKind) -> list[str]:
    return [note.offered for note in v.normalisations if note.kind is kind]


def test_parse_verdict_keeps_vocabulary_tags_and_counts_every_drop(root: Path) -> None:
    raw_tags = [
        "h13", "bogus", "H2 ", "inv-11", "inv-sf-3", "kind:review", "kind:confusion", "h13", 7,
    ]

    v = _parse({**_FULL_RAW, "tags": raw_tags}, root)

    assert v.tags == ("h13", "h2", "inv-11", "inv-sf-3", "kind:confusion")
    assert _kinds(v) == [_K.TAG_DROPPED] * 5
    assert _offers(v, _K.TAG_DROPPED) == [
        repr("bogus"), repr("kind:review"), repr("kind:confusion"), repr("h13"), repr(7),
    ]


def test_parse_verdict_primary_tag_is_the_first_surviving_verifier_tag(root: Path) -> None:
    v = _parse({**_FULL_RAW, "tags": ["bogus", "tests", "h3"]}, root)

    assert v.tags == ("tests", "h3", verdict.KIND_TAG)


def test_parse_verdict_drops_a_non_list_tags_value_as_one(root: Path) -> None:
    v = _parse({**_FULL_RAW, "tags": "h13"}, root)

    assert v.tags == (verdict.KIND_TAG,)
    assert _kinds(v) == [_K.TAG_DROPPED]
    assert _offers(v, _K.TAG_DROPPED) == [repr("h13")]


@pytest.mark.parametrize(
    ("tag", "kept"),
    [
        ("h1", True), ("h14", True), ("h0", False), ("h15", False), ("h01", False),
        ("comments", True), ("tests", True), ("comment", False),
        ("inv-7", True), ("inv-sf-3", True), ("inv-", False), ("inv-sf", False),
    ],
)
def test_parse_verdict_keeps_exactly_the_verifier_tag_vocabulary(
    root: Path, tag: str, kept: bool,
) -> None:
    v = _parse({**_FULL_RAW, "tags": [tag]}, root)

    assert (tag in v.tags) is kept


@pytest.mark.parametrize("raw", [{"tags": []}, {"tags": None}, {}], ids=["empty", "null", "absent"])
def test_parse_verdict_takes_no_tags_uncounted(root: Path, raw: dict) -> None:
    v = _parse({**_without(_FULL_RAW, "tags"), **raw}, root)

    assert v.tags == (verdict.KIND_TAG,)
    assert v.normalisations == ()


@pytest.mark.parametrize(
    ("offered", "expected"),
    [("high", "high"), ("medium", "medium"), ("low", "low"), (" HIGH ", "high")],
)
def test_parse_verdict_keeps_an_enum_severity_uncounted(
    root: Path, offered: str, expected: str,
) -> None:
    v = _parse({**_FULL_RAW, "severity": offered, "severity_reason": "because"}, root)

    assert v.severity == expected
    assert v.severity_reason == "because"
    assert v.normalisations == ()


_OFF_ENUM_SEVERITIES = [
    pytest.param({"severity": "critical"}, "critical", id="critical"),
    pytest.param({"severity": "urgent"}, "urgent", id="urgent"),
    pytest.param({"severity": "severe"}, "severe", id="severe"),
    pytest.param({"severity": ""}, "", id="empty"),
    pytest.param({"severity": 3}, 3, id="non-str"),
]


@pytest.mark.parametrize(("raw", "offered"), _OFF_ENUM_SEVERITIES)
def test_parse_verdict_substitutes_medium_for_an_off_enum_severity(
    root: Path, raw: dict, offered,
) -> None:
    base = _without(_without(_FULL_RAW, "severity"), "severity_reason")

    v = _parse({**base, **raw}, root)

    assert v.severity == "medium"
    assert repr(offered) in v.severity_reason
    assert "medium" in v.severity_reason
    assert _kinds(v) == [_K.SEVERITY_SUBSTITUTED]
    assert _offers(v, _K.SEVERITY_SUBSTITUTED) == [repr(offered)]


@pytest.mark.parametrize(("raw", "offered"), _OFF_ENUM_SEVERITIES)
def test_parse_verdict_keeps_the_verifiers_reason_inside_a_substitution(
    root: Path, raw: dict, offered,
) -> None:
    base = _without(_FULL_RAW, "severity")

    v = _parse({**base, **raw, "severity_reason": "the merge lane halts"}, root)

    assert v.severity == "medium"
    assert "the merge lane halts" in v.severity_reason
    assert repr(offered) in v.severity_reason
    assert _kinds(v) == [_K.SEVERITY_SUBSTITUTED]


_MISSING_SEVERITIES = [
    pytest.param({}, id="absent"),
    pytest.param({"severity": None}, id="null"),
]


@pytest.mark.parametrize("raw", _MISSING_SEVERITIES)
def test_parse_verdict_states_a_missing_severity_as_none_given(root: Path, raw: dict) -> None:
    base = _without(_without(_FULL_RAW, "severity"), "severity_reason")

    v = _parse({**base, **raw}, root)

    assert v.severity == "medium"
    assert "no severity" in v.severity_reason
    assert "medium" in v.severity_reason
    assert "critical" not in v.severity_reason
    assert "offered" not in v.severity_reason
    assert v.normalisations == (
        verdict.Normalisation(_K.SEVERITY_MISSING, repr(None), "medium"),
    )


@pytest.mark.parametrize("raw", _MISSING_SEVERITIES)
def test_parse_verdict_keeps_the_verifiers_reason_beside_a_missing_severity(
    root: Path, raw: dict,
) -> None:
    base = _without(_FULL_RAW, "severity")

    v = _parse({**base, **raw, "severity_reason": "the merge lane halts"}, root)

    assert v.severity == "medium"
    assert "no severity" in v.severity_reason
    assert "the merge lane halts" in v.severity_reason
    assert _kinds(v) == [_K.SEVERITY_MISSING]


@pytest.mark.parametrize("raw", [{}, {"severity_reason": "  "}, {"severity_reason": 4}],
                         ids=["absent", "blank", "non-str"])
def test_parse_verdict_states_a_missing_severity_reason_and_counts_it(
    root: Path, raw: dict,
) -> None:
    v = _parse({**_without(_FULL_RAW, "severity_reason"), **raw}, root)

    assert v.severity == "high"
    assert v.severity_reason.strip()
    assert _kinds(v) == [_K.SEVERITY_REASON_MISSING]


@pytest.mark.parametrize(
    ("offered", "expected"),
    [
        ("mechanical", verdict.Route.MECHANICAL),
        ("structural", verdict.Route.STRUCTURAL),
        ("MECHANICAL", verdict.Route.MECHANICAL),
        (" Structural ", verdict.Route.STRUCTURAL),
    ],
)
def test_parse_verdict_keeps_a_named_route_uncounted(
    root: Path, offered: str, expected: verdict.Route,
) -> None:
    v = _parse({**_FULL_RAW, "route": offered}, root)

    assert v.route is expected
    assert v.normalisations == ()


_UNNAMED_ROUTES = [
    pytest.param({}, None, id="absent"),
    pytest.param({"route": None}, None, id="null"),
    pytest.param({"route": "maybe"}, "maybe", id="unknown"),
]


@pytest.mark.parametrize(("raw", "offered"), _UNNAMED_ROUTES)
def test_parse_verdict_defaults_the_route_to_mechanical_with_a_remediation(
    root: Path, raw: dict, offered,
) -> None:
    v = _parse({**_without(_FULL_RAW, "route"), **raw}, root)

    assert v.route is verdict.Route.MECHANICAL
    assert v.normalisations == (
        verdict.Normalisation(_K.ROUTE_DEFAULTED, repr(offered), "mechanical"),
    )


@pytest.mark.parametrize(("raw", "offered"), _UNNAMED_ROUTES)
def test_parse_verdict_defaults_the_route_to_structural_without_a_remediation(
    root: Path, raw: dict, offered,
) -> None:
    v = _parse({**_without(_FULL_RAW, "route"), "remediation": None, **raw}, root)

    assert v.route is verdict.Route.STRUCTURAL
    assert v.normalisations == (
        verdict.Normalisation(_K.ROUTE_DEFAULTED, repr(offered), "structural"),
    )


def test_parse_verdict_defaults_the_route_to_structural_for_a_rejected_remediation(
    root: Path,
) -> None:
    raw = {
        **_without(_FULL_RAW, "route"),
        "remediation": {"path": "docs/missing.md", "change": "Document X"},
    }

    v = _parse(raw, root)

    assert v.route is verdict.Route.STRUCTURAL
    assert sorted(kind.value for kind in _kinds(v)) == ["remediation_rejected", "route_defaulted"]


@pytest.mark.parametrize("raw", [{}, {"reason": None}, {"reason": 5}],
                         ids=["absent", "null", "non-str"])
def test_parse_verdict_records_a_missing_reason(root: Path, raw: dict) -> None:
    v = _parse({**_without(_FULL_RAW, "reason"), **raw}, root)

    assert v.reason == ""
    assert _kinds(v) == [_K.REASON_MISSING]


def test_parse_verdict_normalises_the_row_6_malformed_verdict(root: Path) -> None:
    raw = {
        "verified": True,
        "reason": "r",
        "anchor": "pkg/missing.py::f",
        "tags": ["h13", "bogus"],
        "severity": "critical",
        "severity_reason": "s",
    }

    v = _parse(raw, root, title=_TITLE)

    assert v.anchor == _TITLE_SLUG
    assert v.tags == ("h13", verdict.KIND_TAG)
    assert v.severity == "medium"
    assert v.route is verdict.Route.STRUCTURAL
    assert verdict.count_normalisations([v]) == {
        **{kind.value: 0 for kind in verdict.NormalisationKind},
        "anchor_from_title": 1,
        "tag_dropped": 1,
        "severity_substituted": 1,
        "route_defaulted": 1,
    }


# ---------------------------------------------------------------------------
# reply_instructions: the verify prompt's statement of the C4 reply, built from
# the same vocabulary parse_verdict keeps.
# ---------------------------------------------------------------------------

_OBSERVED_ROOT = "/observed/tree"


def test_reply_instructions_give_every_severity_and_route_a_meaning() -> None:
    text = verdict.reply_instructions(_OBSERVED_ROOT)

    for severity in verdict.SEVERITIES:
        assert re.search(rf'^  "{severity}": \S', text, re.MULTILINE), severity
    for route in verdict.Route:
        assert re.search(rf'^  "{route.value}": \S', text, re.MULTILINE), route
    assert _OBSERVED_ROOT in text
    assert verdict.KIND_TAG not in text


_SCRIPTS_DIR = Path(__file__).resolve().parents[1]


def test_importing_verdict_fails_on_a_codebook_severity_without_a_meaning() -> None:
    probe = (
        "from legibility import codebook\n"
        "schema = codebook.V2_SCHEMA['properties']['entries']['items']['properties']\n"
        "schema['severity']['enum'].append('critical')\n"
        "from legibility import verdict\n"
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(
        [str(_SCRIPTS_DIR), str(_SCRIPTS_DIR / "legibility")],
    )}

    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True, text=True, env=env, timeout=60, check=False,
    )

    assert result.returncode != 0
    assert "RuntimeError" in result.stderr
    assert "['critical', 'high', 'low', 'medium']" in result.stderr
    assert "['high', 'low', 'medium']" in result.stderr
