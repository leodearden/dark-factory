"""Tests for scripts/legibility/prescreen.py: the census pre-screen of novel
clusters against what is already catalogued (plans/census-incremental-prd.md
L6), through the module's public API only."""
from __future__ import annotations

import copy
from typing import Any

import pytest
from legibility import census_findings, codebook, prescreen, verdict

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
    assert screen.surfaced == (prescreen.Surfaced(title=title, fixed_entry_ids=(fixed_entry_id,)),)


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
# The key rule: a verdict at a keyed entry's (area, anchor) attaches to it
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


def test_keyed_entry_matches_area_and_anchor_whatever_the_primary_tag():
    cb = _keyed_codebook()

    entry = prescreen.keyed_entry(cb, _minted())

    assert entry is not None
    assert entry["id"] == "entry-cand-20261001-1"


@pytest.mark.parametrize(
    "identity",
    [_minted(anchor="orchestrator/x.py::g"), _minted(area="shared")],
    ids=["other-anchor", "other-area"],
)
def test_keyed_entry_finds_nothing_at_another_location(identity):
    assert prescreen.keyed_entry(_keyed_codebook(), identity) is None


def test_keyed_entry_never_matches_an_unkeyed_entry():
    cb = _keyed_codebook()
    cb["entries"] = cb["entries"][:1]

    assert prescreen.keyed_entry(cb, _minted()) is None


def _attach(cb: dict, found: verdict.Verdict | None = None):
    entry = prescreen.keyed_entry(cb, _minted())
    assert entry is not None
    return prescreen.attach_by_key(cb, "cand-20261019-1", entry, found or _found(), run_id=_RUN)


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
