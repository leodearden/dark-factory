"""Tests for scripts/legibility/synthesis.py: the census synthesis brief, the
C6 document's one validator, its application to the run, and the report
section rendered from it (plans/census-incremental-prd.md C6, leaf L10),
through the module's public API only."""
from __future__ import annotations

import copy
import json

import pytest
from legibility import census_findings, synthesis, verdict

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
