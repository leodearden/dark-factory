"""Tests for scripts/legibility/prescreen.py: the census pre-screen of novel
clusters against what is already catalogued (plans/census-incremental-prd.md
L6), through the module's public API only."""
from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest
from legibility import census_findings, codebook, prescreen, verdict
from shared.finding_key import finding_key

_RUN = "census-dark_factory-20261019"
_NO_SLUGS: tuple[str, ...] = ()


def _sighting(session: str) -> dict:
    return {
        "date": "2026-07-01",
        "project": "dark_factory",
        "session": session,
        "origin_phase": "implement",
        "manifested_phase": "verify",
    }


def _entry(entry_id: str, title: str, session: str, *, status: str = "open") -> dict:
    return {
        "id": entry_id,
        "title": title,
        "severity": "medium",
        "status": status,
        "origin_phase": "implement",
        "manifested_phase": "verify",
        "sightings": [_sighting(session)],
    }


def _candidate(cand_id: str, title: str, session: str, disposition: str, **fields: Any) -> dict:
    return {
        "id": cand_id,
        "title": title,
        "first_seen": "2026-07-01",
        "disposition": disposition,
        "sightings": [_sighting(session)],
        **fields,
    }


def _catalogued() -> dict:
    return {
        "version": 2,
        "entries": [
            _entry("entry-cand-20260701-2", "Promoted shape", "sess-promoted"),
            _entry("entry-only", "Entry only shape", "sess-entry"),
            _entry("entry-fixed", "Fixed shape", "sess-fixed", status="fixed"),
            _entry("entry-cand-20260701-4", "Fixed via candidate", "sess-fixed-cand", status="fixed"),
        ],
        "candidates": [
            _candidate("cand-20260701-1", "Pending shape", "sess-pending", "pending"),
            _candidate(
                "cand-20260701-2", "Promoted shape", "sess-promoted", "promoted",
                promoted_to="entry-cand-20260701-2",
            ),
            _candidate("cand-20260701-3", "Rejected shape", "sess-rejected", "rejected"),
            _candidate(
                "cand-20260701-4", "Fixed via candidate", "sess-fixed-cand", "promoted",
                promoted_to="entry-cand-20260701-4",
            ),
        ],
    }


def _cluster(title: str, session: str = "sess-new") -> dict:
    return {
        "title": title,
        "summary": title,
        "cause": "a cause",
        "area": None,
        "origin_phase": "implement",
        "manifested_phase": "verify",
        "evidence": [],
        "sightings": [{"session": session, "origin_phase": "implement", "manifested_phase": "verify"}],
    }


def _screen(*clusters: dict) -> prescreen.Screen:
    return prescreen.screen(list(clusters), prescreen.Catalogue.of(_catalogued()))


# ---------------------------------------------------------------------------
# normalised_title
# ---------------------------------------------------------------------------

def test_normalised_titles_ignore_case_and_whitespace():
    assert prescreen.normalised_title("  Rejected\tSHAPE \n") == prescreen.normalised_title("rejected shape")


def test_normalised_titles_keep_different_words_apart():
    assert prescreen.normalised_title("Rejected shape") != prescreen.normalised_title("Rejected shapes")


# ---------------------------------------------------------------------------
# screen(): the title rule, surfacing, and the session rule
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "title, matched",
    [
        ("Pending shape", ("cand-20260701-1",)),
        ("promoted  SHAPE", ("cand-20260701-2", "entry-cand-20260701-2")),
        ("Rejected Shape", ("cand-20260701-3",)),
        ("entry only shape", ("entry-only",)),
    ],
    ids=["pending", "promoted", "rejected", "entry-only"],
)
def test_a_catalogued_title_is_screened_by_the_title_rule(title, matched):
    cluster = _cluster(title)

    screen = _screen(cluster)

    assert screen.to_verify == ()
    assert screen.surfaced == ()
    assert screen.screened == (
        prescreen.Screened(rule=prescreen.Rule.TITLE, title=title, session="sess-new", matched=matched),
    )


def test_a_novel_title_from_an_unseen_session_stays_to_verify_in_order():
    first, second = _cluster("Brand new shape"), _cluster("Another new shape", session="sess-other")

    screen = _screen(first, second)

    assert len(screen.to_verify) == 2
    assert screen.to_verify[0] is first
    assert screen.to_verify[1] is second
    assert screen.screened == ()
    assert screen.surfaced == ()
    assert screen.uncatalogued == screen.to_verify
    assert all(got is want for got, want in zip(screen.uncatalogued, screen.to_verify, strict=True))


@pytest.mark.parametrize(
    "title, fixed_entry_id",
    [("FIXED shape", "entry-fixed"), ("Fixed via candidate", "entry-cand-20260701-4")],
    ids=["fixed-entry", "candidate-promoted-to-a-fixed-entry"],
)
def test_a_title_matching_a_fixed_entry_is_surfaced_and_verified(title, fixed_entry_id):
    cluster = _cluster(title)

    screen = _screen(cluster)

    assert screen.to_verify == (cluster,)
    assert screen.screened == ()
    assert screen.surfaced == (
        prescreen.Surfaced(rule=prescreen.Rule.TITLE, title=title, fixed_entry_ids=(fixed_entry_id,)),
    )
    assert screen.uncatalogued == ()


def test_only_clusters_no_catalogued_record_matches_are_uncatalogued():
    novel = _cluster("Brand new shape")
    surfaced = _cluster("FIXED shape", session="sess-other")

    screen = _screen(novel, surfaced)

    assert screen.to_verify == (novel, surfaced)
    assert screen.uncatalogued == (novel,)
    assert screen.uncatalogued[0] is novel


def test_a_title_matching_a_retired_entry_is_surfaced_as_fixed():
    cb = _catalogued()
    cb["entries"].append(_entry("entry-retired", "Retired shape", "sess-retired", status="retired"))
    cluster = _cluster("retired shape")

    screen = prescreen.screen([cluster], prescreen.Catalogue.of(cb))

    assert screen.to_verify == (cluster,)
    assert [item.fixed_entry_ids for item in screen.surfaced] == [("entry-retired",)]


def test_a_novel_title_from_an_already_sighted_session_is_screened_by_the_session_rule():
    cluster = _cluster("Brand new shape", session="sess-pending")

    screen = _screen(cluster)

    assert screen.to_verify == ()
    assert screen.screened == (
        prescreen.Screened(
            rule=prescreen.Rule.SESSION, title="Brand new shape", session="sess-pending",
            matched=("cand-20260701-1",),
        ),
    )


def test_the_session_rule_names_every_record_carrying_the_session():
    screen = _screen(_cluster("Brand new shape", session="sess-promoted"))

    assert screen.screened[0].rule is prescreen.Rule.SESSION
    assert set(screen.screened[0].matched) == {"cand-20260701-2", "entry-cand-20260701-2"}


def test_the_title_rule_wins_when_the_session_rule_also_applies():
    screen = _screen(_cluster("Rejected shape", session="sess-pending"))

    assert [screened.rule for screened in screen.screened] == [prescreen.Rule.TITLE]


def test_the_attached_by_methods_are_rule_values():
    assert prescreen.Rule.TITLE.value == "title"
    assert prescreen.Rule.SESSION.value == "session"
    assert prescreen.Rule.KEY.value == "key"
    assert prescreen.Rule.CROSS_INSTRUMENT.value == "cross_instrument"


# ---------------------------------------------------------------------------
# routed_records(): title-screened payloads attach through the merger's routing
# ---------------------------------------------------------------------------

_TITLE_ATTACH = {"method": "title", "run_id": _RUN}


def _payload(title: str) -> dict:
    return {
        "title": title,
        "cause": "a cause",
        "area": "orchestrator",
        "origin_phase": "implement",
        "manifested_phase": "verify",
        "evidence_quote": f"quote for {title}",
    }


def _record(*titles: str, session: str = "sess-new") -> dict:
    return {
        "session": session,
        "date": "2026-10-19",
        "project": "dark_factory",
        "agent_class": "orchestrated-task",
        "matches": [],
        "candidates": [_payload(title) for title in titles],
    }


def _routed(*records: dict) -> list[dict]:
    return prescreen.routed_records(list(records), prescreen.Catalogue.of(_catalogued()), run_id=_RUN)


@pytest.mark.parametrize(
    "mined, routing_title",
    [
        ("rejected  SHAPE", "Rejected shape"),
        ("Promoted shape", "Promoted shape"),
        ("pending shape", "Pending shape"),
    ],
)
def test_a_title_screened_payload_takes_the_candidates_exact_title_and_an_attach(mined, routing_title):
    (record,) = _routed(_record(mined))

    assert record["candidates"] == [{**_payload(mined), "title": routing_title, "attached_by": _TITLE_ATTACH}]
    assert record["matches"] == []


def test_a_payload_matching_only_an_entry_becomes_a_match_on_it():
    (record,) = _routed(_record("ENTRY only shape"))

    assert record["candidates"] == []
    assert record["matches"] == [
        {
            "entry_id": "entry-only",
            "origin_phase": "implement",
            "manifested_phase": "verify",
            "evidence_quote": "quote for ENTRY only shape",
            "attached_by": _TITLE_ATTACH,
        }
    ]


def test_surfaced_session_screened_and_novel_payloads_are_untouched():
    surfaced = _record("fixed SHAPE")
    session_screened = _record("Brand new shape", session="sess-pending")
    novel = _record("Another new shape")

    assert _routed(surfaced, session_screened, novel) == [surfaced, session_screened, novel]


def test_routing_never_mutates_its_input_records():
    records = [_record("rejected  SHAPE", "ENTRY only shape", "Brand new shape")]
    before = copy.deepcopy(records)

    prescreen.routed_records(records, prescreen.Catalogue.of(_catalogued()), run_id=_RUN)

    assert records == before


def _merged(*records: dict) -> dict:
    merge = codebook.apply_coding_records(_catalogued(), _routed(*records), invariant_slugs=_NO_SLUGS)
    assert codebook.validate(merge.codebook) == []
    return merge.codebook


def _holder(cb: dict, record_id: str) -> dict:
    return next(record for record in (*cb["entries"], *cb["candidates"]) if record["id"] == record_id)


def _new_sighting(cb: dict, record_id: str) -> dict:
    return next(s for s in _holder(cb, record_id)["sightings"] if s["session"] == "sess-new")


@pytest.mark.parametrize(
    "mined, holder_id",
    [
        ("rejected  SHAPE", "cand-20260701-3"),
        ("Promoted Shape", "entry-cand-20260701-2"),
        ("entry ONLY shape", "entry-only"),
        ("Pending  shape", "cand-20260701-1"),
    ],
    ids=["rejected", "promoted", "entry-only", "pending"],
)
def test_a_title_screened_sighting_lands_through_the_merger_carrying_its_attach(mined, holder_id):
    merged = _merged(_record(mined))

    assert len(merged["candidates"]) == len(_catalogued()["candidates"])
    assert _new_sighting(merged, holder_id)["attached_by"] == _TITLE_ATTACH


def test_only_the_novel_payload_mints_a_pending_candidate():
    merged = _merged(_record("rejected  SHAPE", "ENTRY only shape", "Brand new shape"))

    minted = merged["candidates"][len(_catalogued()["candidates"]):]
    assert [(candidate["title"], candidate["disposition"]) for candidate in minted] == [
        ("Brand new shape", "pending")
    ]
    assert "attached_by" not in minted[0]["sightings"][0]


# ---------------------------------------------------------------------------
# place(): the key rule (a verdict at a live keyed entry's (area, anchor)
# joins it) and the cross-instrument rule, decided once per cluster
# ---------------------------------------------------------------------------

_ANCHOR = "orchestrator/x.py::f"
_STANDING = census_findings.FindingIdentity.mint("orchestrator", _ANCHOR, ("h13", verdict.KIND_TAG))
_VERIFIER_TAGS = ("h11", verdict.KIND_TAG)


def _found(tags: tuple[str, ...] = _VERIFIER_TAGS, anchor: str = _ANCHOR) -> verdict.Verdict:
    return verdict.Verdict(
        verified=True,
        reason="the verifier read orchestrator/x.py",
        anchor=anchor,
        tags=tags,
        severity="medium",
        severity_reason="r",
        route=verdict.Route.MECHANICAL,
        remediation=None,
    )


def _minted(area: str = "orchestrator", anchor: str = _ANCHOR, tags=_VERIFIER_TAGS):
    return census_findings.FindingIdentity.mint(area, anchor, tags)


def _keyed_codebook(*candidate_sessions: str) -> dict:
    entry = {
        **_entry("entry-cand-20261001-1", "A title", "sess-a"),
        **_STANDING.to_fields(),
        "first_seen_run": "census-dark_factory-20261001",
        "last_seen_runs": ["census-dark_factory-20261001"],
    }
    promoted = _candidate(
        "cand-20261001-1", "A title", "sess-a", "promoted",
        promoted_to="entry-cand-20261001-1", **_STANDING.to_fields(),
    )
    pending = _candidate("cand-20261019-1", "B title", "sess-b", "pending")
    pending["sightings"] = [_sighting(session) for session in candidate_sessions or ("sess-b",)]
    return {
        "version": 2,
        "entries": [_entry("entry-unkeyed", "Unkeyed", "sess-u"), entry],
        "candidates": [promoted, pending],
    }


_NO_INPUTS = prescreen.InstrumentFindings((), None, ())


def _place(cb: dict, identity=None, inputs: prescreen.InstrumentFindings = _NO_INPUTS) -> prescreen.Placement:
    return prescreen.place(cb, "B title", identity or _minted(), inputs)


def test_a_verdict_at_a_keyed_entrys_location_joins_it_whatever_the_primary_tag():
    placement = _place(_keyed_codebook())

    assert placement.standing_entry_id == "entry-cand-20261001-1"
    assert placement.identity == _STANDING
    assert placement.adoption is None
    assert placement.surfaced is None


@pytest.mark.parametrize(
    "identity",
    [_minted(anchor="orchestrator/x.py::g"), _minted(area="shared")],
    ids=["other-anchor", "other-area"],
)
def test_a_verdict_at_another_location_is_promoted_under_its_minted_identity(identity):
    placement = _place(_keyed_codebook(), identity)

    assert placement.standing_entry_id is None
    assert placement.identity == identity
    assert placement.surfaced is None
    assert placement.supersedes == ()
    assert placement.first_seen_run(_RUN) == _RUN


def test_an_unkeyed_entry_is_never_joined():
    cb = _keyed_codebook()
    cb["entries"] = cb["entries"][:1]

    assert _place(cb).standing_entry_id is None


@pytest.mark.parametrize("status", ["fixed", "retired"])
def test_a_verdict_at_only_a_fixed_entrys_location_is_promoted_afresh_and_surfaced(status):
    cb = _keyed_codebook()
    cb["entries"][1]["status"] = status

    placement = _place(cb)

    assert placement.standing_entry_id is None
    assert placement.identity == _minted()
    assert placement.surfaced == prescreen.Surfaced(
        rule=prescreen.Rule.KEY, title="B title", fixed_entry_ids=("entry-cand-20261001-1",),
    )


def test_a_live_entry_at_a_fixed_entrys_location_is_joined_and_nothing_is_surfaced():
    cb = _keyed_codebook()
    cb["entries"][1]["status"] = "fixed"
    cb["entries"].append({**copy.deepcopy(cb["entries"][1]), "id": "entry-cand-20261010-1", "status": "open"})

    placement = _place(cb)

    assert placement.standing_entry_id == "entry-cand-20261010-1"
    assert placement.surfaced is None


def _attach(cb: dict, found: verdict.Verdict | None = None):
    placement = _place(cb)
    assert placement.standing_entry_id is not None
    return prescreen.attach_by_key(cb, "cand-20261019-1", placement.standing_entry_id, found or _found(), run_id=_RUN)


def _by_id(cb: dict, record_id: str) -> dict:
    return next(record for record in (*cb["entries"], *cb["candidates"]) if record["id"] == record_id)


def test_a_key_attach_promotes_the_candidate_onto_the_standing_entry():
    cb = _keyed_codebook()
    before = copy.deepcopy(cb)

    attached, _ = _attach(cb)

    assert cb == before
    candidate = _by_id(attached, "cand-20261019-1")
    assert candidate["disposition"] == "promoted"
    assert candidate["promoted_to"] == "entry-cand-20261001-1"
    assert candidate["attached_by"] == {"method": "key", "run_id": _RUN}
    assert census_findings.FindingIdentity.from_record(candidate) == _STANDING
    entry = _by_id(attached, "entry-cand-20261001-1")
    assert census_findings.FindingIdentity.from_record(entry) == _STANDING
    assert len(attached["entries"]) == len(cb["entries"])
    assert codebook.validate(attached) == []


def test_a_key_attach_lands_the_candidates_sightings_on_the_entry_with_the_verifier_tags():
    attached, _ = _attach(_keyed_codebook("sess-a", "sess-b"))

    sightings = _by_id(attached, "entry-cand-20261001-1")["sightings"]
    assert [sighting["session"] for sighting in sightings] == ["sess-a", "sess-b"]
    assert "attached_by" not in sightings[0]
    assert sightings[1]["attached_by"] == {"method": "key", "run_id": _RUN}
    assert sightings[1]["verifier_tags"] == list(_VERIFIER_TAGS)
    assert codebook.validate(attached) == []


def test_a_key_attach_names_what_it_joined_and_the_tag_disagreement():
    _, key_attach = _attach(_keyed_codebook())

    assert key_attach.title == "B title"
    assert key_attach.candidate_id == "cand-20261019-1"
    assert key_attach.entry_id == "entry-cand-20261001-1"
    assert key_attach.identity == _STANDING
    assert key_attach.verifier_tags == _VERIFIER_TAGS
    assert key_attach.tag_disagreement == ("h13", "h11")


def test_a_key_attach_with_the_same_primary_tag_reports_no_disagreement():
    _, key_attach = _attach(_keyed_codebook(), _found(tags=("h13", verdict.KIND_TAG)))

    assert key_attach.tag_disagreement is None


# ---------------------------------------------------------------------------
# The cross-instrument rule's input: the newest /review, /hotspot-survey and
# /review-all records (docs/quality-findings-contract.md §5, §9)
# ---------------------------------------------------------------------------

_REVIEW_RUN = "review-dark_factory-20261009-2"
_HOTSPOT_RUN = "hotspot-survey-dark_factory-20261009"
_REVIEW_ALL_RUN = "review-all-dark_factory-20261008"
_EXTERNAL_KEY = finding_key("orchestrator", _ANCHOR, "h4")
_OLD_KEY = finding_key("orchestrator", "orchestrator/x.py::old", "h4")


def _external(key: str, area: str, anchor: str, **fields: Any) -> dict:
    return {
        "key": key,
        "area": area,
        "anchor": anchor,
        "tags": ["h4", "kind:defect"],
        "severity": "high",
        "statement": "s",
        "first_seen": "review-all-dark_factory-20261001",
        "supersedes": [],
        **fields,
    }


def _write(root: Path, relpath: str, content: Any) -> None:
    path = root / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")


def _contract_record(run_id: str, *findings: dict) -> dict:
    return {"method": {"run_id": run_id, "as_of_sha": "abc1234"}, "findings": list(findings)}


def _review_all_finding() -> dict:
    return _external(
        _EXTERNAL_KEY, "orchestrator", _ANCHOR,
        supersedes=[_OLD_KEY, "positional:clusters[0].findings[0]"],
    )


def _all_instruments(root: Path) -> None:
    _write(root, "review/reports/review-dark_factory-20261009.json", "{not json")
    _write(
        root, "review/reports/review-dark_factory-20261009-2.json",
        _contract_record(_REVIEW_RUN, _external(finding_key("dashboard", "dashboard/a.py", "h11"), "dashboard", "dashboard/a.py")),
    )
    _write(root, "plans/bug-hotspot-survey-2026-07-06-full-findings.json", {"clusters": []})
    _write(
        root, "plans/bug-hotspot-survey-2026-10-09-full-findings.json",
        _contract_record(_HOTSPOT_RUN, _external(finding_key("shared", "shared/b.py", "h10"), "shared", "shared/b.py")),
    )
    _write(root, "plans/review-all-dark_factory-20261008.json", _contract_record(_REVIEW_ALL_RUN, _review_all_finding()))
    _write(root, "plans/quality-metrics/review-all-dark_factory-20261008.json", {"run_id": "metrics-only"})


def test_only_each_instruments_newest_record_is_consumed(tmp_path):
    _all_instruments(tmp_path)

    found = prescreen.load_instrument_findings(tmp_path, "dark_factory")

    assert found.consumed == (_REVIEW_RUN, _HOTSPOT_RUN, _REVIEW_ALL_RUN)
    assert found.note is None
    assert {external.run_id for external in found.findings} == {_REVIEW_RUN, _HOTSPOT_RUN, _REVIEW_ALL_RUN}


def test_an_external_finding_keeps_its_identity_and_only_its_key_supersedes(tmp_path):
    _all_instruments(tmp_path)

    found = prescreen.load_instrument_findings(tmp_path, "dark_factory")

    (external,) = [finding for finding in found.findings if finding.run_id == _REVIEW_ALL_RUN]
    assert external == prescreen.ExternalFinding(
        key=_EXTERNAL_KEY,
        area="orchestrator",
        anchor=_ANCHOR,
        first_seen="review-all-dark_factory-20261001",
        supersedes=(_OLD_KEY,),
        run_id=_REVIEW_ALL_RUN,
    )


@pytest.mark.parametrize(
    "malformed",
    [
        {"key": "fk-nothex"},
        {"area": "orchestrator/merge-queue"},
        {"anchor": "orchestrator/x.py:12"},
        {"anchor": None},
    ],
    ids=["key", "area", "anchor-line-pin", "anchor-missing"],
)
def test_a_finding_without_a_valid_identity_is_skipped_and_counted(tmp_path, malformed):
    bad = {**_review_all_finding(), **malformed}
    _write(tmp_path, "plans/review-all-dark_factory-20261008.json", _contract_record(_REVIEW_ALL_RUN, bad))

    found = prescreen.load_instrument_findings(tmp_path, "dark_factory")

    assert found.findings == ()
    assert found.consumed == (_REVIEW_ALL_RUN,)
    assert found.note is not None
    assert "1 finding" in found.note
    assert "plans/review-all-dark_factory-20261008.json" in found.note


def test_with_no_records_nothing_is_consumed_and_the_note_says_where_it_looked(tmp_path):
    found = prescreen.load_instrument_findings(tmp_path, "dark_factory")

    assert found.consumed == ()
    assert found.findings == ()
    assert found.note is not None
    for named in ("/review", "review/reports", "/hotspot-survey", "/review-all", "plans"):
        assert named in found.note


@pytest.mark.parametrize(
    "content, why",
    [
        ("{not json", "unreadable"),
        ({"findings": []}, "method.run_id"),
        ({"method": {"run_id": _REVIEW_ALL_RUN}, "findings": {}}, "findings"),
        ([1, 2], "method.run_id"),
    ],
    ids=["bad-json", "no-run-id", "findings-not-a-list", "not-an-object"],
)
def test_an_unusable_newest_record_is_noted_never_consumed_and_never_raises(tmp_path, content, why):
    _write(tmp_path, "plans/review-all-dark_factory-20261008.json", content)

    found = prescreen.load_instrument_findings(tmp_path, "dark_factory")

    assert found.consumed == ()
    assert found.note is not None
    assert "plans/review-all-dark_factory-20261008.json" in found.note
    assert why in found.note


def _loaded(tmp_path: Path) -> prescreen.InstrumentFindings:
    _all_instruments(tmp_path)
    return prescreen.load_instrument_findings(tmp_path, "dark_factory")


def test_an_equal_key_is_adopted_with_the_external_supersedes_and_first_seen(tmp_path):
    minted = census_findings.FindingIdentity.mint("orchestrator", _ANCHOR, ("h4", verdict.KIND_TAG))

    adoption = _loaded(tmp_path).match(minted)

    assert adoption is not None
    assert adoption.matched_by is prescreen.AdoptedBy.KEY
    assert adoption.identity == minted
    assert adoption.supersedes == (_OLD_KEY,)
    assert adoption.first_seen == "review-all-dark_factory-20261001"
    assert adoption.external.run_id == _REVIEW_ALL_RUN


def test_an_equal_location_is_adopted_with_only_the_key_replaced(tmp_path):
    minted = _minted(tags=("h13", verdict.KIND_TAG))

    adoption = _loaded(tmp_path).match(minted)

    assert adoption is not None
    assert adoption.matched_by is prescreen.AdoptedBy.LOCATION
    assert adoption.identity == dataclasses.replace(minted, key=_EXTERNAL_KEY)
    assert adoption.external.run_id == _REVIEW_ALL_RUN


def test_an_equal_key_wins_over_an_equal_location(tmp_path):
    _all_instruments(tmp_path)
    elsewhere = finding_key("orchestrator", "orchestrator/y.py", "h4")
    assert elsewhere != _EXTERNAL_KEY
    _write(
        tmp_path, "review/reports/review-dark_factory-20261009-2.json",
        _contract_record(_REVIEW_RUN, _external(elsewhere, "orchestrator", _ANCHOR)),
    )
    minted = census_findings.FindingIdentity.mint("orchestrator", _ANCHOR, ("h4", verdict.KIND_TAG))

    adoption = prescreen.load_instrument_findings(tmp_path, "dark_factory").match(minted)

    assert adoption is not None
    assert adoption.matched_by is prescreen.AdoptedBy.KEY
    assert adoption.external.run_id == _REVIEW_ALL_RUN


def test_an_unmatched_identity_adopts_nothing(tmp_path):
    assert _loaded(tmp_path).match(_minted(anchor="orchestrator/x.py::g")) is None


def test_a_promotion_at_an_externally_keyed_location_is_placed_under_the_adopted_key(tmp_path):
    cb = _keyed_codebook()
    cb["entries"] = cb["entries"][:1]
    minted = _minted(tags=("h13", verdict.KIND_TAG))

    placement = _place(cb, minted, _loaded(tmp_path))

    assert placement.standing_entry_id is None
    assert placement.adoption is not None
    assert placement.identity == dataclasses.replace(minted, key=_EXTERNAL_KEY)
    assert placement.supersedes == (_OLD_KEY,)
    assert placement.first_seen_run(_RUN) == "review-all-dark_factory-20261001"


def test_a_live_entry_at_the_location_wins_over_the_external_finding(tmp_path):
    placement = _place(_keyed_codebook(), inputs=_loaded(tmp_path))

    assert placement.standing_entry_id == "entry-cand-20261001-1"
    assert placement.adoption is None
    assert placement.identity == _STANDING


# ---------------------------------------------------------------------------
# ScreenReport: per-rule counts, the run record, and the ## Screened section
# rendered from the record alone (contract §5)
# ---------------------------------------------------------------------------

_NO_INPUTS_NOTE = "/review: no record matching review/reports/review-dark_factory-<YYYYMMDD>[-<n>].json"


def _full_report(tmp_path: Path) -> prescreen.ScreenReport:
    inputs = _loaded(tmp_path)
    minted = _minted(tags=("h13", verdict.KIND_TAG))
    adoption = inputs.match(minted)
    assert adoption is not None
    _, key_attach = _attach(_keyed_codebook())
    return prescreen.ScreenReport(
        screen=_screen(
            _cluster("Rejected shape"),
            _cluster("ENTRY only shape"),
            _cluster("Brand new shape", session="sess-pending"),
            _cluster("Fixed shape"),
            _cluster("Novel shape"),
        ),
        surfaced_by_key=(
            prescreen.Surfaced(rule=prescreen.Rule.KEY, title="D title", fixed_entry_ids=("entry-cand-20260901-1",)),
        ),
        attached=(key_attach,),
        re_observed=(prescreen.ReObserved(title="C title", entry_id="entry-cand-20261019-3", adoption=adoption),),
        inputs=inputs,
    )


def _empty_report() -> prescreen.ScreenReport:
    return prescreen.ScreenReport(
        screen=prescreen.Screen((), (), ()),
        surfaced_by_key=(),
        attached=(),
        re_observed=(),
        inputs=prescreen.InstrumentFindings((), _NO_INPUTS_NOTE, ()),
    )


def test_the_report_counts_each_rule(tmp_path):
    assert _full_report(tmp_path).counts() == {"title": 2, "session": 1, "key": 1, "cross_instrument": 1}


def test_an_empty_report_counts_zero_for_every_rule():
    assert _empty_report().counts() == {"title": 0, "session": 0, "key": 0, "cross_instrument": 0}


def test_the_report_record_round_trips_through_json(tmp_path):
    record = _full_report(tmp_path).to_record()

    assert json.loads(json.dumps(record)) == record


def _section(report: prescreen.ScreenReport) -> list[str]:
    return prescreen.report_lines(json.loads(json.dumps(report.to_record())))


def _lines_naming(lines: list[str], *needles: str) -> list[str]:
    return [line for line in lines if all(needle in line for needle in needles)]


def test_the_section_opens_with_its_heading_and_one_count_line(tmp_path):
    lines = _section(_full_report(tmp_path))

    assert lines[:3] == ["", "## Screened", ""]
    assert len(_lines_naming(lines, "title", "session", "key", "cross_instrument")) == 1


def test_the_section_names_the_consumed_runs(tmp_path):
    lines = _section(_full_report(tmp_path))

    assert _lines_naming(lines, _REVIEW_RUN, _HOTSPOT_RUN, _REVIEW_ALL_RUN)


def test_the_section_names_every_item_and_where_it_went(tmp_path):
    lines = _section(_full_report(tmp_path))

    assert _lines_naming(lines, "Rejected shape", "cand-20260701-3")
    assert _lines_naming(lines, "ENTRY only shape", "entry-only")
    assert _lines_naming(lines, "Brand new shape", "cand-20260701-1")
    assert _lines_naming(lines, "Fixed shape", "entry-fixed", "surfaced by title")
    assert _lines_naming(lines, "D title", "entry-cand-20260901-1", "surfaced by key")
    assert not _lines_naming(lines, "Novel shape")
    assert _lines_naming(lines, "B title", "entry-cand-20261001-1")
    assert _lines_naming(lines, "C title", "entry-cand-20261019-3", _REVIEW_ALL_RUN)


def test_a_key_attach_with_a_tag_disagreement_names_both_tags(tmp_path):
    lines = _section(_full_report(tmp_path))

    assert _lines_naming(lines, "B title", "h13", "h11")


def test_an_empty_report_still_renders_zero_counts_and_the_inputs_note():
    lines = _section(_empty_report())

    assert lines[:3] == ["", "## Screened", ""]
    (count_line,) = _lines_naming(lines, "title", "session", "key", "cross_instrument")
    assert "1" not in count_line
    assert _lines_naming(lines, _NO_INPUTS_NOTE)
