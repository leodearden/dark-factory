"""Tests for scripts/legibility/synthesis.py: the census synthesis brief, the
C6 document's one validator, its application to the run, and the report
section rendered from it (plans/census-incremental-prd.md C6, leaf L10),
through the module's public API only."""
from __future__ import annotations

import copy
import json

import pytest
from legibility import census_findings, codebook, synthesis, verdict

_RUN = "census-dark_factory-20261021"
_EXTERNAL_RUN = "review-dark_factory-20261019"
_RUN_SESSIONS = frozenset({"s-run-1", "s-run-2", "s-run-3", "s-run-4", "s-run-5"})
_COUNTS = {"mined": 5, "confirmed": 3, "refuted": 0}


def _sighting(
    session: str, *, invariant: str | None = None, origin: str = "implement", manifested: str = "verify",
) -> dict:
    sighting = {
        "date": "2026-10-21",
        "project": "dark_factory",
        "session": session,
        "origin_phase": origin,
        "manifested_phase": manifested,
    }
    return sighting if invariant is None else {**sighting, "invariant_violated": invariant}


def _identity(anchor: str) -> census_findings.FindingIdentity:
    return census_findings.FindingIdentity.mint("orchestrator", anchor, ("h13", verdict.KIND_TAG))


def _entry(entry_id: str, *sightings: dict, anchor: str | None = None, **fields) -> dict:
    entry = {
        "id": entry_id,
        "title": f"title of {entry_id}",
        "severity": "low",
        "status": "open",
        "origin_phase": "implement",
        "manifested_phase": "verify",
        "sightings": list(sightings),
        **fields,
    }
    return entry if anchor is None else {**entry, **_identity(anchor).to_fields()}


def _found(
    *, severity: str = "high", remediation: verdict.Remediation | None = None, reason: str = "seen in the tree",
) -> verdict.Verdict:
    return verdict.Verdict(
        verified=True,
        reason=reason,
        anchor="orchestrator/n.py::h",
        tags=("h13", verdict.KIND_TAG),
        severity=severity,
        severity_reason=f"{severity} because it costs the next change",
        route=verdict.Route.MECHANICAL if remediation else verdict.Route.STRUCTURAL,
        remediation=remediation,
    )


_ANCHOR_ATTACHED = "orchestrator/a.py::f"
_ANCHOR_MATCHED = "orchestrator/b.py::g"
_ANCHOR_QUIET = "orchestrator/c.py"
_ANCHOR_NEW = "orchestrator/n.py::h"
_ANCHOR_ADOPTED = "orchestrator/x.py"
_NEW_ID = "entry-cand-20261021-1"
_ADOPTED_ID = "entry-cand-20261021-2"
_NEW_FOUND = _found(remediation=verdict.Remediation("orchestrator/n.py", "split h into two functions"))
_ATTACH_FOUND = _found(severity="medium", reason="still there at a.py")
_ADOPTED_FOUND = _found(severity="low", reason="the review saw it too")


def _before() -> dict:
    return {
        "version": 2,
        "entries": [
            _entry(
                "entry-attached", _sighting("s-old-a", invariant="inv-a"), anchor=_ANCHOR_ATTACHED,
                status="partially", cause="standing cause a",
            ),
            _entry("entry-matched", _sighting("s-old-b"), anchor=_ANCHOR_MATCHED),
            _entry("entry-quiet", _sighting("s-old-c", invariant="inv-old"), anchor=_ANCHOR_QUIET),
            _entry("entry-unkeyed", _sighting("s-old-d")),
        ],
        "candidates": [],
    }


def _new_sighting() -> dict:
    return _sighting("s-run-4", invariant="inv-a", origin="architect", manifested="review")


def _after() -> dict:
    after = copy.deepcopy(_before())
    attached, matched, _quiet, unkeyed = after["entries"]
    attached["sightings"].append(_sighting("s-run-1"))
    matched["sightings"].append(_sighting("s-run-2", invariant="inv-b"))
    unkeyed["sightings"].append(_sighting("s-run-3"))
    after["candidates"] = [
        {
            "id": "cand-20261021-1", "title": f"title of {_NEW_ID}", "cause": "new cause",
            "first_seen": "2026-10-21", "disposition": "promoted", "promoted_to": _NEW_ID,
            "sightings": [_new_sighting()],
        },
        {
            "id": "cand-20261021-2", "title": f"title of {_ADOPTED_ID}",
            "first_seen": "2026-10-21", "disposition": "promoted", "promoted_to": _ADOPTED_ID,
            "sightings": [_sighting("s-run-5")],
        },
    ]
    after["entries"].extend([
        _entry(
            _NEW_ID, _new_sighting(), anchor=_ANCHOR_NEW, severity="high",
            origin_phase="architect", manifested_phase="review",
        ),
        _entry(_ADOPTED_ID, _sighting("s-run-5"), anchor=_ANCHOR_ADOPTED),
    ])
    return after


def _verdicts() -> list[census_findings.Adjudication]:
    return [
        census_findings.Adjudication(_NEW_ID, _NEW_FOUND, "new cause"),
        census_findings.Adjudication("entry-attached", _ATTACH_FOUND, None),
        census_findings.Adjudication(_ADOPTED_ID, _ADOPTED_FOUND, None),
    ]


def _brief(**overrides) -> synthesis.Brief:
    arguments = {
        "run_id": _RUN,
        "before": _before(),
        "after": _after(),
        "verdicts": _verdicts(),
        "adopted": {_ADOPTED_ID: _EXTERNAL_RUN},
        "run_sessions": _RUN_SESSIONS,
        "counts": _COUNTS,
    }
    return synthesis.Brief.of(**{**arguments, **overrides})


def _key(anchor: str) -> str:
    return _identity(anchor).key


def _re_observed(brief: synthesis.Brief, entry_id: str) -> synthesis.ReObservedFinding:
    (finding,) = (finding for finding in brief.re_observed if finding.entry_id == entry_id)
    return finding


# ---------------------------------------------------------------------------
# The brief
# ---------------------------------------------------------------------------

def test_a_keyed_entry_this_run_created_is_a_new_finding_with_its_verdict():
    (new,) = _brief().new

    assert (new.key, new.entry_id, new.title, new.cause) == (
        _key(_ANCHOR_NEW), _NEW_ID, f"title of {_NEW_ID}", "new cause",
    )
    assert (new.area, new.anchor, new.tags) == ("orchestrator", _ANCHOR_NEW, ("h13", verdict.KIND_TAG))
    assert (new.origin_phase, new.manifested_phase, new.sessions) == ("architect", "review", ("s-run-4",))
    assert new.verdict == synthesis.Judgement(
        severity="high",
        severity_reason=_NEW_FOUND.severity_reason,
        reason="seen in the tree",
        proposal="split h into two functions",
        route="mechanical",
    )


def test_a_standing_keyed_entry_that_gained_a_session_is_re_observed_with_only_what_it_gained():
    brief = _brief()
    attached = _re_observed(brief, "entry-attached")
    matched = _re_observed(brief, "entry-matched")

    assert (attached.key, attached.title, attached.cause) == (
        _key(_ANCHOR_ATTACHED), "title of entry-attached", "standing cause a",
    )
    assert attached.gained_sessions == ("s-run-1",)
    assert attached.standing == synthesis.Standing(
        severity="low", status="partially", origin_phase="implement", manifested_phase="verify",
    )
    assert attached.verdict is not None and attached.verdict.reason == "still there at a.py"
    assert attached.adopted_from is None
    assert (matched.gained_sessions, matched.verdict) == (("s-run-2",), None)


def test_an_entry_promoted_under_another_instruments_key_is_re_observed_not_new():
    brief = _brief()
    adopted = _re_observed(brief, _ADOPTED_ID)

    assert adopted.adopted_from == _EXTERNAL_RUN
    assert adopted.gained_sessions == ("s-run-5",)
    assert adopted.verdict is not None and adopted.verdict.reason == "the review saw it too"
    assert _ADOPTED_ID not in {finding.entry_id for finding in brief.new}
    assert [finding.entry_id for finding in brief.re_observed] == ["entry-attached", "entry-matched", _ADOPTED_ID]


def test_an_entry_with_no_new_session_is_never_in_the_brief_nor_is_a_standing_count():
    record = _brief().to_record()
    text = json.dumps(record)

    assert "entry-quiet" not in text
    assert _key(_ANCHOR_QUIET) not in text
    assert set(record) == {"run_id", "new", "re_observed", "unkeyed_re_observed", "invariant_counts", "counts"}
    assert record["counts"] == _COUNTS


def test_an_unkeyed_entry_that_gained_a_session_is_counted_but_not_listed():
    brief = _brief()

    assert brief.unkeyed_re_observed == 1
    assert "entry-unkeyed" not in json.dumps(brief.to_record())


def test_a_brief_is_empty_exactly_when_it_has_no_new_and_no_re_observed_finding():
    before = _before()
    only_unkeyed = copy.deepcopy(before)
    only_unkeyed["entries"][3]["sightings"].append(_sighting("s-run-3"))
    only_matched = copy.deepcopy(before)
    only_matched["entries"][1]["sightings"].append(_sighting("s-run-2"))

    assert not _brief().is_empty
    assert _brief(after=before, verdicts=[], adopted={}).is_empty
    assert _brief(after=only_unkeyed, verdicts=[], adopted={}).is_empty
    assert not _brief(after=only_matched, verdicts=[], adopted={}).is_empty


def test_invariant_counts_are_distinct_slug_session_pairs_over_this_runs_sessions():
    assert _brief().invariant_counts == (("inv-a", 1), ("inv-b", 1))
    assert _brief().to_record()["invariant_counts"] == {"inv-a": 1, "inv-b": 1}


def test_only_standing_re_observed_entries_are_correctable_and_only_judged_findings_refinable():
    brief = _brief()

    assert brief.correctable_entry_ids == frozenset({"entry-attached", "entry-matched"})
    assert brief.refinable_keys == frozenset({_key(_ANCHOR_NEW), _key(_ANCHOR_ATTACHED), _key(_ANCHOR_ADOPTED)})


def test_the_brief_record_round_trips_through_json():
    record = _brief().to_record()

    assert json.loads(json.dumps(record)) == record


def test_a_verdict_naming_no_entry_of_the_run_is_refused():
    with pytest.raises(ValueError, match="entry-missing"):
        _brief(verdicts=[census_findings.Adjudication("entry-missing", _NEW_FOUND, None)])


# ---------------------------------------------------------------------------
# The C6 document's validator
# ---------------------------------------------------------------------------

_DROP = object()


def _shaped(base: dict, overrides: dict) -> dict:
    return {name: value for name, value in {**base, **overrides}.items() if value is not _DROP}


def _new_item(**overrides) -> dict:
    return _shaped({"key": _key(_ANCHOR_NEW), "observation": "h does two jobs"}, overrides)


def _change(**overrides) -> dict:
    return _shaped({"field": "severity", "from": "low", "to": "medium", "why": "it recurred"}, overrides)


def _re_item(**overrides) -> dict:
    return _shaped({"key": _key(_ANCHOR_ATTACHED), "sessions": ["s-run-1"], "changed": [_change()]}, overrides)


def _refinement(**overrides) -> dict:
    return _shaped(
        {"key": _key(_ANCHOR_NEW), "origin_phase": "merge", "manifested_phase": "review", "why": "a merge made it"},
        overrides,
    )


def _correction(**overrides) -> dict:
    return _shaped({"entry_id": "entry-matched", "note": "the framing was refuted", "cause": "a truer cause"}, overrides)


def _note(**overrides) -> dict:
    return _shaped({"claim": "a.py and b.py share a cause", "evidence": "both name f", "verified": False}, overrides)


def _document(**overrides) -> dict:
    return _shaped(
        {
            "new": [_new_item()],
            "re_observed": [_re_item()],
            "phase_refinements": [_refinement()],
            "corrections": [_correction()],
            "notes": [_note()],
        },
        overrides,
    )


_EMPTY_DOCUMENT = {"new": [], "re_observed": [], "phase_refinements": [], "corrections": [], "notes": []}


@pytest.mark.parametrize("raw", [
    pytest.param(json.dumps(_document()), id="bare"),
    pytest.param("Here it is:\n```json\n" + json.dumps(_document()) + "\n```\n", id="fenced"),
])
def test_a_valid_document_parses_to_typed_values(raw):
    result = synthesis.parse(raw, _brief())

    assert result == synthesis.Synthesis(
        new=(synthesis.NewObservation(_key(_ANCHOR_NEW), "h does two jobs"),),
        re_observed=(
            synthesis.ReObservation(
                key=_key(_ANCHOR_ATTACHED),
                sessions=("s-run-1",),
                changed=(synthesis.Change("severity", from_value="low", to_value="medium", why="it recurred"),),
            ),
        ),
        phase_refinements=(synthesis.PhaseRefinement(_key(_ANCHOR_NEW), "merge", "review", "a merge made it"),),
        corrections=(
            synthesis.Correction("entry-matched", note="the framing was refuted", title=None, cause="a truer cause"),
        ),
        notes=(synthesis.Note("a.py and b.py share a cause", "both name f"),),
    )


@pytest.mark.parametrize("brief", [
    pytest.param(_brief(), id="full-brief"),
    pytest.param(_brief(after=_before(), verdicts=[], adopted={}), id="empty-brief"),
])
def test_the_empty_document_is_valid_against_any_brief(brief):
    assert synthesis.parse(json.dumps(_EMPTY_DOCUMENT), brief) == synthesis.Synthesis()


@pytest.mark.parametrize(("raw", "fragments"), [
    pytest.param("not json at all", ["not a JSON object"], id="non-json"),
    pytest.param(json.dumps([_document()]), ["not a JSON object"], id="top-level-array"),
    pytest.param(json.dumps(_document(notes=_DROP)), ["$: missing key 'notes'"], id="missing-top-level-key"),
    pytest.param(json.dumps(_document(standing=[])), ["$: unexpected key 'standing'"], id="extra-top-level-key"),
    pytest.param(json.dumps(_document(new={})), ["new: must be a list", "{}"], id="list-of-wrong-type"),
    pytest.param(
        json.dumps(_document(new=[_new_item(severity="high")])), ["new[0]: unexpected key 'severity'"],
        id="extra-item-key",
    ),
    pytest.param(
        json.dumps(_document(new=[_new_item(key=_key(_ANCHOR_ATTACHED))])),
        ["new[0].key: not a new finding of this run", _key(_ANCHOR_ATTACHED)],
        id="re-observed-key-in-new",
    ),
    pytest.param(
        json.dumps(_document(re_observed=[_re_item(key=_key(_ANCHOR_NEW))])),
        ["re_observed[0].key: not a re-observed finding of this run", _key(_ANCHOR_NEW)],
        id="new-key-in-re-observed",
    ),
    pytest.param(
        json.dumps(_document(new=[_new_item(), _new_item()])),
        ["new[1].key: duplicate", _key(_ANCHOR_NEW)],
        id="duplicate-key",
    ),
    pytest.param(
        json.dumps(_document(re_observed=[_re_item(sessions=["s-old-a"])])),
        ["re_observed[0].sessions[0]: not a session this finding gained", "s-old-a"],
        id="session-not-gained",
    ),
    pytest.param(
        json.dumps(_document(re_observed=[_re_item(changed=[_change(field="title")])])),
        ["re_observed[0].changed[0].field", "'title'"],
        id="changed-field-outside-vocabulary",
    ),
    pytest.param(
        json.dumps(_document(re_observed=[_re_item(changed=[_change(to="critical")])])),
        ["re_observed[0].changed[0].to", "'critical'"],
        id="severity-outside-vocabulary",
    ),
    pytest.param(
        json.dumps(_document(re_observed=[_re_item(changed=[_change(field="verdict", **{"from": "unverified"}, to="maybe")])])),
        ["re_observed[0].changed[0].to", "'maybe'"],
        id="verdict-outside-vocabulary",
    ),
    pytest.param(
        json.dumps(_document(re_observed=[_re_item(changed=[_change(field="origin_phase", **{"from": "implement"}, to="coding")])])),
        ["re_observed[0].changed[0].to", "'coding'"],
        id="phase-change-outside-vocabulary",
    ),
    pytest.param(
        json.dumps(_document(phase_refinements=[_refinement(key=_key(_ANCHOR_MATCHED))])),
        ["phase_refinements[0].key: no verdict this run", _key(_ANCHOR_MATCHED)],
        id="refinement-of-an-unjudged-finding",
    ),
    pytest.param(
        json.dumps(_document(phase_refinements=[_refinement(origin_phase="coding")])),
        ["phase_refinements[0].origin_phase", "'coding'"],
        id="refinement-phase-outside-vocabulary",
    ),
    pytest.param(
        json.dumps(_document(phase_refinements=[_refinement(why="  ")])),
        ["phase_refinements[0].why: must be a non-empty string"],
        id="empty-why",
    ),
    pytest.param(
        json.dumps(_document(new=[_new_item(observation="")])),
        ["new[0].observation: must be a non-empty string"],
        id="empty-observation",
    ),
    pytest.param(
        json.dumps(_document(corrections=[_correction(note="")])),
        ["corrections[0].note: must be a non-empty string"],
        id="empty-note",
    ),
    pytest.param(
        json.dumps(_document(corrections=[_correction(entry_id=_ADOPTED_ID)])),
        ["corrections[0].entry_id: not a correctable entry", _ADOPTED_ID],
        id="correction-of-an-uncorrectable-entry",
    ),
    pytest.param(
        json.dumps(_document(corrections=[_correction(status="fixed")])),
        ["corrections[0]: unexpected key 'status'", "census-owned"],
        id="correction-carrying-status",
    ),
    pytest.param(
        json.dumps(_document(corrections=[_correction(cause=_DROP)])),
        ["corrections[0]: writes neither title nor cause"],
        id="correction-writing-nothing",
    ),
    pytest.param(
        json.dumps(_document(corrections=[_correction(), _correction(cause="another cause")])),
        ["corrections[1].entry_id: duplicate", "entry-matched"],
        id="two-corrections-for-one-entry",
    ),
    pytest.param(
        json.dumps(_document(notes=[_note(verified=True)])),
        ["notes[0].verified: must be false", "no tools"],
        id="verified-note",
    ),
])
def test_an_invalid_document_is_unavailable_naming_the_path_and_value(raw, fragments):
    result = synthesis.parse(raw, _brief())

    assert isinstance(result, synthesis.Unavailable)
    for fragment in fragments:
        assert fragment in result.reason


def test_every_problem_of_an_invalid_document_is_named():
    raw = json.dumps(_document(new=[_new_item(observation="")], notes=[_note(verified=True)]))

    result = synthesis.parse(raw, _brief())

    assert isinstance(result, synthesis.Unavailable)
    assert "new[0].observation" in result.reason
    assert "notes[0].verified" in result.reason


# ---------------------------------------------------------------------------
# Applying the document, and the standing count
# ---------------------------------------------------------------------------

def _cluster(title: str, *sessions: str) -> dict:
    return {
        "title": title,
        "origin_phase": "implement",
        "manifested_phase": "verify",
        "sightings": [
            {"session": session, "origin_phase": "implement", "manifested_phase": "verify"} for session in sessions
        ],
    }


def _refined_phases(cluster: dict) -> set[tuple[str, str]]:
    return {
        (record["origin_phase"], record["manifested_phase"]) for record in (cluster, *cluster["sightings"])
    }


def _entry_by_id(cb: dict, entry_id: str) -> dict:
    (entry,) = (entry for entry in cb["entries"] if entry["id"] == entry_id)
    return entry


def test_refinements_rewrite_the_runs_clusters_and_the_entries_it_created_only():
    after = _after()
    pristine = copy.deepcopy(after)
    document = synthesis.Synthesis(phase_refinements=(
        synthesis.PhaseRefinement(_key(_ANCHOR_NEW), "merge", "review", "a merge made it"),
        synthesis.PhaseRefinement(_key(_ANCHOR_ATTACHED), "prd", "ops", "the PRD set it up"),
    ))
    clusters = [
        (_cluster("new", "s-run-4"), _NEW_ID),
        (_cluster("attached", "s-run-1"), "entry-attached"),
        (_cluster("unplaced", "s-run-6"), None),
        (_cluster("adopted", "s-run-5"), _ADOPTED_ID),
    ]

    refined = synthesis.apply_refinements(after, document, clusters=clusters, created={_NEW_ID, _ADOPTED_ID})

    new, attached, unplaced, adopted = refined.clusters
    assert _refined_phases(new) == {("merge", "review")}
    assert _refined_phases(attached) == {("prd", "ops")}
    assert unplaced == _cluster("unplaced", "s-run-6")
    assert adopted == _cluster("adopted", "s-run-5")
    created = _entry_by_id(refined.codebook, _NEW_ID)
    standing = _entry_by_id(refined.codebook, "entry-attached")
    assert (created["origin_phase"], created["manifested_phase"]) == ("merge", "review")
    assert (standing["origin_phase"], standing["manifested_phase"]) == ("implement", "verify")
    assert (refined.clusters_refined, refined.entries_refined) == (2, 1)
    assert after == pristine
    assert clusters[0][0] == _cluster("new", "s-run-4")
    assert codebook.validate(refined.codebook) == []


def test_no_refinement_leaves_the_codebook_and_clusters_equal():
    clusters = [(_cluster("new", "s-run-4"), _NEW_ID)]

    refined = synthesis.apply_refinements(_after(), synthesis.Synthesis(), clusters=clusters, created={_NEW_ID})

    assert refined.codebook == _after()
    assert refined.clusters == (_cluster("new", "s-run-4"),)
    assert (refined.clusters_refined, refined.entries_refined) == (0, 0)


def test_a_document_without_corrections_has_no_corrections_record():
    assert synthesis.corrections_record(
        synthesis.Synthesis(), run_id=_RUN, date="2026-10-21", project="dark_factory",
    ) is None


def test_the_corrections_record_is_a_coding_record_the_merger_applies():
    document = synthesis.Synthesis(corrections=(
        synthesis.Correction("entry-matched", note="the framing was refuted", title=None, cause="a truer cause"),
        synthesis.Correction("entry-attached", note="renamed", title="a clearer title", cause=None),
    ))

    record = synthesis.corrections_record(document, run_id=_RUN, date="2026-10-21", project="dark_factory")

    assert record is not None
    assert codebook.validate_coding_record(record) == []
    assert (record["session"], record["agent_class"]) == (_RUN, "census-synthesis")
    merged = codebook.apply_coding_records(_after(), [record], invariant_slugs=())
    matched = _entry_by_id(merged.codebook, "entry-matched")
    attached = _entry_by_id(merged.codebook, "entry-attached")
    assert matched["cause"] == "a truer cause"
    assert matched["title"] == "title of entry-matched"
    assert attached["title"] == "a clearer title"
    assert [sighting.get("note") for sighting in matched["sightings"] if sighting["session"] == _RUN] == [
        "the framing was refuted",
    ]
    assert merged.counts.corrections_applied == 2


def test_the_standing_count_is_open_entries_that_stood_and_gained_no_session():
    before = _before()
    before["entries"].extend([
        _entry("entry-still", _sighting("s-old-e"), status="partially"),
        _entry("entry-fixed", _sighting("s-old-f"), status="fixed"),
        _entry("entry-retired-now", _sighting("s-old-g")),
        _entry("entry-accepted", _sighting("s-old-h"), status="accepted", accepted_by=7),
    ])
    after = _after()
    after["entries"].extend(copy.deepcopy(before["entries"][4:]))
    _entry_by_id(after, "entry-retired-now")["status"] = "retired"

    assert synthesis.standing_count(before, after) == 2
