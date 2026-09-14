"""Tests for scripts/audit_offcontract_review_findings.py — the READ-ONLY audit
behind task 5430's triage of the reviewer findings that the off-contract verdict
shape silently dropped.

Origin: esc-2896-10. Between 2026-07-19 and 2026-08-10 `reviewer_comprehensive`
emitted verdicts whose `verdict.issues[]` entries used an off-contract shape —
`severity` outside {blocking, suggestion}, and `file`+`line` instead of
`location`. Both review gates key on the contract fields, so those findings were
read as suggestions AND skipped by the in-scope filter: no implementer ever saw
them. The audit under test does the mechanical half of the triage (normalize,
census, validate) and makes no judgement about whether a finding is live.

Mirrors test_audit_wiped_metadata_files.py: pure functions get direct pytest
coverage; `main()` gets subprocess coverage.

NO TEST HERE ASSERTS A COUNT DERIVED FROM THE LIVE VERDICT TREE, OR FROM THE
FROZEN corpus/ EITHER. `.worktrees/` is gitignored runtime state that later
re-reviews overwrite in place, and it demonstrably moved under measurement while
this task was being planned — the off-contract population shrank 76 -> 74 and the
location-less population 123 -> 121 in roughly three days. A test pinning "the
corpus yields N findings" would be pinning a conclusion, not a behaviour. Every
assertion below runs against synthetic issue dicts and tmp_path verdict trees
the test builds itself. Live numbers belong in provenance.json as dated
measurements, never in an assertion.
"""
from __future__ import annotations

import pytest

from audit_offcontract_review_findings import (
    CONTRACT_SEVERITIES,
    normalize_issue,
)

# ---------------------------------------------------------------------------
# Synthetic issue dicts — one per schema variant measured across the 14 frozen
# verdicts. The KEY SETS are transcribed from the real corpus; the text is
# invented, so no assertion here depends on a real finding's wording.
#
#   {location, description, suggested_fix}                      2896 (on-contract)
#   {title, description, failure_scenario, file, line}           3031 3075 3757
#   {short_summary, summary, failure_scenario, file, line, verdict} 3041 (3340 sans verdict)
#   {title, description, reproduce, file, line}                  3064
#   {title, detail, file, line}                                  3142 3453
#   {title, description, blocking, file, line}                   3308
#   {title, description, suggestion, file, line}                 3363 3454
#   {title, description, failure_scenario, recommendation, file, line} 3679
# ---------------------------------------------------------------------------

ON_CONTRACT_LOCATION_SHAPE = {
    "severity": "suggestion",
    "category": "robustness",
    "location": "orchestrator/src/orchestrator/workflow.py:6090",
    "description": "The retry ladder swallows the transport error.",
    "suggested_fix": "Re-raise after the third attempt.",
}

TITLE_DESCRIPTION_FAILURE_SHAPE = {
    "severity": "high",
    "category": "correctness",
    "file": "fused-memory/src/fused_memory/reconciliation/mem0_tombstone.py",
    "line": 4,
    "title": "Tombstone coverage is overclaimed",
    "description": "Only two of six delete paths record a tombstone.",
    "failure_scenario": "A recon delete on the other four paths leaves no trace.",
}

SHORT_SUMMARY_SUMMARY_SHAPE = {
    "severity": "medium",
    "category": "architectural-coherence",
    "file": "fused-memory/src/fused_memory/reconciliation/summary_pool.py",
    "line": 365,
    "short_summary": "Eviction order makes narratives single-trim transient",
    "summary": "The (is_ledger_stamp, ...) key evicts every narrative first.",
    "failure_scenario": "One trim drops the durable resolution Stage 2 just wrote.",
    "verdict": "CONFIRMED",
}

TITLE_REPRODUCE_SHAPE = {
    "severity": "medium",
    "category": "efficiency",
    "file": "fused-memory/src/fused_memory/server/tools.py",
    "line": 3827,
    "title": "Tool description grew to 8,328 chars",
    "description": "Roughly 2k tokens are loaded into every agent session.",
    "reproduce": "len(get_statuses.__doc__)",
}

TITLE_DETAIL_SHAPE = {
    "severity": "medium",
    "category": "blast-radius",
    "file": "orchestrator/src/orchestrator/proc_supervision.py",
    "line": 847,
    "title": "--setenv=PYTHONPATH= applies to the entire transient unit",
    "detail": "The deploy payload runs with a PYTHONPATH it never had.",
}

TITLE_BLOCKING_SHAPE = {
    "severity": "moderate",
    "category": "contract-drift",
    "file": "plans/dashboard-availability-prd.md",
    "line": 195,
    "title": "Two normative Contract rows are superseded but still stated",
    "description": "The PRD documents the behaviour the code deliberately replaced.",
    "blocking": False,
}

TITLE_SUGGESTION_SHAPE = {
    "severity": "major",
    "category": "correctness",
    "file": "dashboard/src/dashboard/data/memory_evals.py",
    "line": 805,
    "title": "Narrowing recovered_open drops the escalation-open badge",
    "description": "insufficient_data, grandfathered and unjudged rows lose it.",
    "failure_scenario": "An open escalation renders as closed on the dashboard.",
    "suggestion": "Keep the badge keyed on the escalation, not the judgement.",
}

TITLE_RECOMMENDATION_SHAPE = {
    "severity": "medium",
    "category": "test-coverage",
    "file": "orchestrator/src/orchestrator/verify_classify.py",
    "line": 380,
    "title": "Docstring cites a pin test that does not exist",
    "description": "The load-bearing classification ordering is unpinned.",
    "failure_scenario": "A reordering lands green.",
    "recommendation": "Add the pin test the docstring already promises.",
}

ALL_SHAPES = [
    ON_CONTRACT_LOCATION_SHAPE,
    TITLE_DESCRIPTION_FAILURE_SHAPE,
    SHORT_SUMMARY_SUMMARY_SHAPE,
    TITLE_REPRODUCE_SHAPE,
    TITLE_DETAIL_SHAPE,
    TITLE_BLOCKING_SHAPE,
    TITLE_SUGGESTION_SHAPE,
    TITLE_RECOMMENDATION_SHAPE,
]


# ---------------------------------------------------------------------------
# normalize_issue — the SPOT for the six mutually incompatible issue schemas.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw", ALL_SHAPES, ids=lambda r: "+".join(sorted(r)))
def test_every_observed_shape_yields_a_usable_record(raw):
    """No schema variant may normalize to an unreadable or unlocatable record."""
    record = normalize_issue("3041", 1, raw)

    assert record["task"] == "3041"
    assert record["index"] == 1
    assert record["id"] == f"3041-{raw['category']}-1"
    assert record["severity"] == raw["severity"]
    assert record["category"] == raw["category"]
    assert record["statement"].strip(), "a normalized issue must carry readable text"
    assert record["location"], "a normalized issue must carry a location"


@pytest.mark.parametrize("raw", ALL_SHAPES, ids=lambda r: "+".join(sorted(r)))
def test_statement_carries_every_text_key_the_shape_populated(raw):
    """Detail keys differ per shape, so none of them may be dropped on the floor."""
    statement = normalize_issue("3041", 0, raw)["statement"]

    text_keys = {
        "title", "short_summary",
        "description", "detail", "summary",
        "failure_scenario", "reproduce", "suggestion", "suggested_fix", "recommendation",
    }
    for key in text_keys & set(raw):
        assert raw[key] in statement, f"{key} was dropped from the statement"


def test_location_prefers_the_on_contract_field():
    record = normalize_issue("2896", 0, ON_CONTRACT_LOCATION_SHAPE)
    assert record["location"] == "orchestrator/src/orchestrator/workflow.py:6090"


def test_location_falls_back_to_file_and_line():
    record = normalize_issue("3453", 1, TITLE_DETAIL_SHAPE)
    assert record["location"] == "orchestrator/src/orchestrator/proc_supervision.py:847"


def test_on_contract_severity_is_not_flagged_off_contract():
    assert normalize_issue("2896", 0, ON_CONTRACT_LOCATION_SHAPE)["off_contract"] is False


@pytest.mark.parametrize("severity", sorted(CONTRACT_SEVERITIES))
def test_off_contract_is_false_for_exactly_the_contract_severities(severity):
    raw = {**TITLE_DETAIL_SHAPE, "severity": severity}
    assert normalize_issue("3453", 1, raw)["off_contract"] is False


@pytest.mark.parametrize("severity", ["high", "major", "medium", "moderate", "low", "minor", "nit", "trivial", None])
def test_off_contract_is_true_for_everything_outside_the_contract(severity):
    raw = {**TITLE_DETAIL_SHAPE, "severity": severity}
    assert normalize_issue("3453", 1, raw)["off_contract"] is True


# ---------------------------------------------------------------------------
# The two cases that caused the original loss. These are regression pins on the
# failure this task exists to account for, not hypotheticals: five of the 23
# dropped findings carry `title: None`, and all 23 lacked `location`.
# ---------------------------------------------------------------------------

def test_null_title_falls_back_to_the_detail_keys():
    """Five of the 23 dropped findings carry `title: None` — an empty statement
    there is how a finding becomes invisible even once someone goes looking."""
    raw = {
        "severity": "medium",
        "category": "correctness",
        "file": "scripts/legibility/nightly.py",
        "line": 872,
        "title": None,
        "short_summary": None,
        "summary": "Streak escalation is one-shot for a persistent condition.",
    }
    statement = normalize_issue("3340", 0, raw)["statement"]
    assert "one-shot for a persistent condition" in statement


@pytest.mark.parametrize("detail_key", ["description", "detail", "summary"])
def test_null_title_falls_back_to_each_detail_key_in_turn(detail_key):
    raw = {
        "severity": "medium",
        "category": "correctness",
        "file": "scripts/legibility/nightly.py",
        "line": 872,
        "title": None,
        detail_key: "The condition persists but escalates once.",
    }
    assert "escalates once" in normalize_issue("3340", 0, raw)["statement"]


def test_an_issue_with_no_location_signal_at_all_raises():
    """Silently yielding "None:None" is how an unlocatable finding gets read as
    located and then never checked. Fail loudly instead."""
    raw = {
        "severity": "medium",
        "category": "correctness",
        "title": "Something is wrong somewhere",
        "description": "But the reviewer never said where.",
    }
    with pytest.raises(ValueError, match="location"):
        normalize_issue("3340", 0, raw)


def test_a_file_without_a_line_does_not_stringify_the_missing_line():
    """Same silent-None failure class as above, one field narrower."""
    raw = {
        "severity": "medium",
        "category": "test-coverage",
        "file": "scripts/legibility/check_trickle_progress.py",
        "title": "Ships with no caller anywhere in the repo",
        "description": "Nothing in the repo invokes it.",
    }
    record = normalize_issue("3340", 1, raw)
    assert record["location"] == "scripts/legibility/check_trickle_progress.py"
    assert "None" not in record["location"]


def test_an_empty_string_title_is_treated_as_absent():
    raw = {**TITLE_DETAIL_SHAPE, "title": "   "}
    statement = normalize_issue("3453", 1, raw)["statement"]
    assert statement.strip().startswith("The deploy payload")
