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

import json
import subprocess
import sys
from pathlib import Path

import pytest
from audit_offcontract_review_findings import (
    CONTRACT_SEVERITIES,
    TRIAGE_SEVERITIES,
    census,
    normalize_issue,
    select_population,
    validate_report,
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


@pytest.mark.parametrize("raw", ["just a string", ["a", "list"], 42, None])
def test_an_issue_entry_that_is_not_an_object_at_all_raises(raw):
    """`normalize_issue` is the SPOT for issue shape, so "this is not an issue"
    is its refusal to make, not the walker's. Reaching `.get` on a str instead
    raises AttributeError, which the walk does not catch — one such entry would
    abort the census of every other file."""
    with pytest.raises(ValueError, match="not an object"):
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


# ---------------------------------------------------------------------------
# census / select_population — measurement over a verdict tree.
#
# The fixture tree below is built by the test and is the ONLY thing these
# assertions read. See the module docstring: the live tree and the frozen
# corpus are both moving targets, so pinning a number from either would pin a
# conclusion rather than a behaviour.
# ---------------------------------------------------------------------------

def _write_verdict(root, task, role, emitted_at, issues):
    path = root / task / "verdicts" / f"{role}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "role": role,
        "emitted_at": emitted_at,
        "schema_version": 1,
        "verdict": {"issues": issues},
    }))
    return path


def _fixture_tree(tmp_path):
    """A tree carrying one instance of every case the census must survive."""
    root = tmp_path / ".task-meta"

    # On-contract: a `suggestion` with the contract `location` spelling.
    _write_verdict(root, "2896", "reviewer_comprehensive", "2026-08-01T00:00:00+00:00", [
        {"severity": "suggestion", "category": "robustness",
         "location": "a/b.py:1", "description": "on contract"},
    ])
    # Off-contract but BELOW the triage floor.
    _write_verdict(root, "3031", "reviewer_comprehensive", "2026-07-19T00:00:00+00:00", [
        {"severity": "low", "category": "documentation",
         "file": "c/d.py", "line": 2, "title": "too minor to triage"},
    ])
    # Off-contract and inside the triage band.
    _write_verdict(root, "3041", "reviewer_comprehensive", "2026-08-10T00:00:00+00:00", [
        {"severity": "medium", "category": "correctness",
         "file": "e/f.py", "line": 3, "title": "in band"},
    ])
    # Off-contract, in band, and from a role that is NOT reviewer_comprehensive.
    _write_verdict(root, "3075", "reviewer_scoped", "2026-08-05T00:00:00+00:00", [
        {"severity": "high", "category": "correctness",
         "file": "g/h.py", "line": 4, "title": "foreign role"},
    ])
    # A verdict that parses but carries no issues — the shape task 2896's
    # archived `reviews/` copies actually have, which is why `files` and
    # `verdicts_with_issues` must be separate numbers.
    _write_verdict(root, "3142", "reviewer_comprehensive", "2026-08-09T00:00:00+00:00", [])
    # An issue with no location signal at all: normalize_issue refuses it, so it
    # must be TALLIED rather than crashing the census or vanishing from it.
    _write_verdict(root, "3308", "reviewer_comprehensive", "2026-08-02T00:00:00+00:00", [
        {"severity": "medium", "category": "correctness", "title": "nowhere"},
    ])
    # Malformed JSON.
    unparseable = root / "3363" / "verdicts" / "reviewer_comprehensive.json"
    unparseable.parent.mkdir(parents=True, exist_ok=True)
    unparseable.write_text("{not json at all")
    # Valid JSON of the WRONG TYPE — the malformation class that used to abort
    # the whole walk with an uncaught AttributeError rather than be tallied.
    wrong_type = root / "3340" / "verdicts" / "reviewer_comprehensive.json"
    wrong_type.parent.mkdir(parents=True, exist_ok=True)
    wrong_type.write_text('["a list, not an object"]')

    return root


def test_census_counts_files_verdicts_and_issues_separately(tmp_path):
    result = census(_fixture_tree(tmp_path))

    assert result.files == 8
    assert result.verdicts_with_issues == 5, "the empty verdict must not be counted"
    assert result.issues == 5


def test_census_counts_off_contract_severity_and_location_less_separately(tmp_path):
    """These are different populations. All 23 dropped findings were in BOTH,
    which is exactly why the two gates compounded — but the census must not
    assume that, or it cannot ever show the two diverging."""
    result = census(_fixture_tree(tmp_path))

    assert result.off_contract_severity == 4, "low + medium + high + unlocatable medium"
    assert result.location_less == 4, "everything except the on-contract suggestion"


def test_census_breaks_issues_down_by_emitting_role(tmp_path):
    result = census(_fixture_tree(tmp_path))
    assert result.roles == {"reviewer_comprehensive": 4, "reviewer_scoped": 1}


def test_the_role_breakdown_reconciles_with_the_issue_total(tmp_path):
    """A census whose parts do not add up is worse than no census: it reads as
    measured. The refused issue must appear in SOME role bucket, or the
    breakdown silently omits whatever the walk could not read."""
    result = census(_fixture_tree(tmp_path))
    assert sum(result.roles.values()) == result.issues


def test_census_reports_the_emission_window(tmp_path):
    result = census(_fixture_tree(tmp_path))
    assert result.emitted_first == "2026-07-19T00:00:00+00:00"
    assert result.emitted_last == "2026-08-10T00:00:00+00:00"


def test_census_tallies_a_malformed_file_instead_of_crashing_or_dropping_it(tmp_path):
    result = census(_fixture_tree(tmp_path))

    assert result.unparseable == 2, "broken JSON and valid-JSON-of-the-wrong-type"
    assert result.files == 8, "an unparseable file is still a file that was there"


@pytest.mark.parametrize("payload", [
    '["a list, not an object"]',
    '{"verdict": "not an object either"}',
    '{"verdict": {"issues": {"keyed": {"severity": "high"}}}}',
    '"a bare string"',
    '7',
])
def test_census_tallies_a_wrongly_shaped_payload_instead_of_aborting_the_walk(tmp_path, payload):
    """Valid JSON of the wrong TYPE is as unreadable as broken JSON. Each of
    these used to reach `.get` on a non-dict and raise AttributeError, which
    `except (OSError, ValueError)` does not catch."""
    root = tmp_path / ".task-meta"
    path = root / "9001" / "verdicts" / "reviewer_comprehensive.json"
    path.parent.mkdir(parents=True)
    path.write_text(payload)

    result = census(root)

    assert result.files == 1
    assert result.unparseable == 1
    assert result.issues == 0


def test_one_malformed_file_does_not_cost_the_walk_of_the_others(tmp_path):
    """The failure this tolerance exists for is not the bad file's own count —
    it is the 1,270 good files whose measurement a single abort would destroy."""
    root = tmp_path / ".task-meta"
    _write_verdict(root, "3041", "reviewer_comprehensive", "2026-08-10T00:00:00+00:00", [
        {"severity": "medium", "category": "correctness",
         "file": "e/f.py", "line": 3, "title": "readable, and after the bad file"},
    ])
    broken = root / "3031" / "verdicts" / "reviewer_comprehensive.json"
    broken.parent.mkdir(parents=True, exist_ok=True)
    broken.write_text('["sorts before 3041, so it is walked first"]')

    result = census(root)

    assert result.unparseable == 1
    assert result.issues == 1, "the good file after the bad one was still read"
    assert [i.record["severity"] for i in select_population(root).issues] == ["medium"]


def test_a_non_dict_issue_entry_is_tallied_as_unlocatable_not_a_crash(tmp_path):
    """A non-dict ENTRY is an issue-level refusal, so it lands in `unlocatable`
    where the normalizer's other refusals do — not in the file-level tally."""
    root = tmp_path / ".task-meta"
    _write_verdict(root, "3041", "reviewer_comprehensive", "2026-08-10T00:00:00+00:00",
                   ["just a string, not an issue object"])

    result = census(root)

    assert result.unparseable == 0, "the FILE parsed and was well-shaped"
    assert result.unlocatable == 1
    assert result.issues == 1, "the refused entry is still counted as present"


def test_census_tallies_an_unlocatable_issue_instead_of_crashing_or_dropping_it(tmp_path):
    """Same no-silent-fail-soft reasoning as `unparseable`: an issue the
    normalizer refuses is a measurement the census must report, not lose."""
    result = census(_fixture_tree(tmp_path))

    assert result.unlocatable == 1
    assert result.issues == 5, "the refused issue is still counted as present"
    assert len(result.roles) == 2


def test_census_of_an_empty_tree_is_all_zeroes(tmp_path):
    empty = tmp_path / ".task-meta"
    empty.mkdir()
    result = census(empty)

    assert (result.files, result.issues, result.unparseable) == (0, 0, 0)
    assert result.roles == {}
    assert result.emitted_first is None
    assert result.emitted_last is None


@pytest.mark.parametrize("walker", [census, select_population],
                         ids=["census", "select_population"])
def test_a_root_that_does_not_exist_is_refused_rather_than_censused(tmp_path, walker):
    """`Path.glob` on a missing directory yields nothing rather than raising, so
    an unchecked root hands back an all-zero census indistinguishable from a
    real tree holding no verdicts. Both entry points must refuse it: a caller
    that reads zeros as a measurement is the no-silent-fail-soft failure."""
    missing = tmp_path / "moved-or-renamed"

    with pytest.raises(NotADirectoryError, match=str(missing)):
        walker(missing)


@pytest.mark.parametrize("walker", [census, select_population],
                         ids=["census", "select_population"])
def test_a_root_that_is_a_file_is_refused_too(tmp_path, walker):
    not_a_tree = tmp_path / "task-meta.json"
    not_a_tree.write_text("{}")

    with pytest.raises(NotADirectoryError):
        walker(not_a_tree)


def test_an_existing_root_with_the_wrong_layout_still_measures_zero(tmp_path):
    """The distinction the guard draws: a MISSING root is a broken question, an
    existing one holding no matching files is a real (if empty) answer. The
    frozen corpus/ in this task's artifact directory is the live instance —
    stored flat as `<task>.json`, so it matches nothing and must not raise."""
    flat = tmp_path / "corpus"
    flat.mkdir()
    (flat / "3041.json").write_text('{"verdict": {"issues": []}}')

    assert census(flat).files == 0


def test_select_population_keeps_only_off_contract_issues_in_the_triage_band(tmp_path):
    selected = select_population(_fixture_tree(tmp_path)).issues

    assert {issue.record["severity"] for issue in selected} == {"medium", "high"}
    assert all(issue.record["off_contract"] for issue in selected)
    assert all(issue.record["severity"] in TRIAGE_SEVERITIES for issue in selected)


def test_select_population_excludes_the_on_contract_suggestion(tmp_path):
    selected = select_population(_fixture_tree(tmp_path)).issues
    assert "suggestion" not in {issue.record["severity"] for issue in selected}


def test_select_population_excludes_severities_below_the_triage_floor(tmp_path):
    selected = select_population(_fixture_tree(tmp_path)).issues
    assert "low" not in {issue.record["severity"] for issue in selected}


def test_select_population_never_filters_on_role(tmp_path):
    """Role is a census OBSERVATION, not a selection criterion. The foreign-role
    `high` is in the population; dropping it would understate the residue."""
    selected = select_population(_fixture_tree(tmp_path)).issues

    assert {issue.role for issue in selected} == {"reviewer_comprehensive", "reviewer_scoped"}
    assert any(issue.role == "reviewer_scoped" for issue in selected)


def test_select_population_carries_the_verdict_facts_the_issue_itself_lacks(tmp_path):
    """Role, emission time and source path belong to the verdict file, not the
    issue, so `normalize_issue` cannot supply them and the walker must."""
    foreign = next(i for i in select_population(_fixture_tree(tmp_path)).issues
                   if i.role == "reviewer_scoped")

    assert foreign.emitted_at == "2026-08-05T00:00:00+00:00"
    assert foreign.path.name == "reviewer_scoped.json"
    assert foreign.location_less is True
    assert foreign.record["location"] == "g/h.py:4"


def _tree_with_one_unlocatable(tmp_path, severity):
    """A tree whose ONLY issue is unlocatable, carrying `severity`.

    Deliberately not folded into `_fixture_tree`: that fixture's unlocatable
    issue is a `medium`, so every fold of the refused residue looks correct
    there whether or not the severity is ever inspected. Varying the severity
    is the only way to tell the two apart."""
    root = tmp_path / ".task-meta"
    _write_verdict(root, "3308", "reviewer_comprehensive", "2026-08-02T00:00:00+00:00", [
        {"severity": severity, "category": "correctness", "title": "nowhere"},
    ])
    return root


def test_an_on_contract_refusal_is_not_counted_into_the_off_contract_tally(tmp_path):
    """The refused residue folds into `location_less` soundly (a refusal IS
    location-less) but into `off_contract_severity` only on inspection. A
    `blocking` issue with no location is refused AND on-contract; counting it
    off-contract contaminates the tally with a severity nobody read."""
    result = census(_tree_with_one_unlocatable(tmp_path, "blocking"))

    assert (result.issues, result.unlocatable) == (1, 1)
    assert result.location_less == 1, "a refused issue is by definition location-less"
    assert result.off_contract_severity == 0, "its severity was on-contract"


@pytest.mark.parametrize("severity", ["high", "medium", "low", None])
def test_an_off_contract_refusal_is_still_counted_into_the_off_contract_tally(tmp_path, severity):
    """The other half of the same pin: inspecting the severity must not turn
    into dropping the residue. Everything outside the contract still counts."""
    result = census(_tree_with_one_unlocatable(tmp_path, severity))

    assert result.off_contract_severity == 1
    assert result.location_less == 1


def test_the_population_reports_the_refusals_it_could_not_put_on_a_roster(tmp_path):
    """A finding absent from the roster is never dispositioned — the exact loss
    this task accounts for. So the refusal count travels WITH the population,
    not only via a separate census call a caller may not make."""
    root = tmp_path / ".task-meta"
    _write_verdict(root, "3075", "reviewer_comprehensive", "2026-08-05T00:00:00+00:00", [
        {"severity": "high", "category": "correctness",
         "file": "g/h.py", "line": 4, "title": "rosterable"},
        {"severity": "high", "category": "correctness", "title": "unlocatable, and in band"},
    ])

    population = select_population(root)

    assert len(population.issues) == 1
    assert population.unselectable == 1, "the in-band refusal must be visible"


@pytest.mark.parametrize("severity", ["low", "blocking", "suggestion"])
def test_a_refusal_outside_the_population_is_not_reported_as_missing_from_it(tmp_path, severity):
    """`unselectable` must not overstate what the roster is missing: a refused
    `low` or on-contract issue was never in this population to begin with."""
    population = select_population(_tree_with_one_unlocatable(tmp_path, severity))

    assert population.issues == []
    assert population.unselectable == 0


def test_the_fixture_trees_refusal_is_reported_by_both_walkers(tmp_path):
    """`census.unlocatable` and `Population.unselectable` answer different
    questions over the same residue, and both must see the fixture's one
    refused `medium`: it is location-less AND would have been selected."""
    root = _fixture_tree(tmp_path)

    assert census(root).unlocatable == 1
    assert select_population(root).unselectable == 1


def test_the_contract_and_triage_severity_sets_are_disjoint():
    """If they ever overlapped, an issue could be both on-contract and in the
    triage band, and `select_population` would silently return nothing."""
    assert not (CONTRACT_SEVERITIES & TRIAGE_SEVERITIES)


# ---------------------------------------------------------------------------
# validate_report — "honest accounting" made mechanical.
#
# The claim this task must not make on trust is "all 23 findings were
# dispositioned". Completeness is a structural property of a data file, so it
# gets enforced rather than asserted in prose. Every case below runs against
# in-memory dicts; nothing reads the real report.
# ---------------------------------------------------------------------------

ROSTER_IDS = ("3041-correctness-0", "3363-design-1")


def _report(*dispositions, roster_ids=ROSTER_IDS):
    return {
        "roster": [{"id": rid} for rid in roster_ids],
        "dispositions": list(dispositions),
    }


def _entry(rid: str, disposition: str | None = "c",
           note: str | None = "Checked against main today.", **extra):
    return {"id": rid, "disposition": disposition, "note": note, **extra}


def test_a_complete_report_validates_clean():
    report = _report(
        _entry("3041-correctness-0", "b", followup_ticket="tkt_abc123"),
        _entry("3363-design-1", "a"),
    )
    assert validate_report(report) == []


@pytest.mark.parametrize("disposition", ["a", "c"])
def test_a_and_c_entries_need_no_ticket(disposition):
    report = _report(
        _entry("3041-correctness-0", disposition),
        _entry("3363-design-1", disposition),
    )
    assert validate_report(report) == []


def test_a_roster_id_missing_from_dispositions_is_rejected():
    report = _report(_entry("3041-correctness-0", "a"))
    violations = validate_report(report)
    assert any("3363-design-1" in v for v in violations)


def test_a_disposition_for_an_unknown_id_is_rejected():
    """An id that is not on the frozen roster means the roster moved under the
    triage, which is the one thing freezing it was supposed to prevent."""
    report = _report(
        _entry("3041-correctness-0", "a"),
        _entry("3363-design-1", "a"),
        _entry("9999-invented-0", "a"),
    )
    violations = validate_report(report)
    assert any("9999-invented-0" in v for v in violations)


def test_a_duplicated_disposition_id_is_rejected():
    """Two dispositions for one finding means one of them is unread."""
    report = _report(
        _entry("3041-correctness-0", "a"),
        _entry("3041-correctness-0", "b", followup_ticket="tkt_abc123"),
        _entry("3363-design-1", "a"),
    )
    violations = validate_report(report)
    assert any("3041-correctness-0" in v for v in violations)


@pytest.mark.parametrize("disposition", ["d", "A", "", None, "b?", "fixed"])
def test_a_disposition_outside_abc_is_rejected(disposition):
    report = _report(
        _entry("3041-correctness-0", disposition),
        _entry("3363-design-1", "a"),
    )
    violations = validate_report(report)
    assert any("3041-correctness-0" in v for v in violations)


@pytest.mark.parametrize("note", [None, "", "   ", "\n\t "])
def test_a_missing_or_blank_note_is_rejected(note):
    """A disposition without reasoning is a verdict, and a verdict nobody can
    check is how these 23 findings got lost in the first place."""
    report = _report(
        _entry("3041-correctness-0", "a", note=note),
        _entry("3363-design-1", "a"),
    )
    violations = validate_report(report)
    assert any("3041-correctness-0" in v for v in violations)


@pytest.mark.parametrize("ticket", [None, "", "   "])
def test_a_live_defect_without_a_followup_ticket_is_rejected(ticket):
    """THE accounting invariant this task exists to enforce: a finding judged
    still-live with no follow-up filed has been dropped a second time."""
    report = _report(
        _entry("3041-correctness-0", "b", followup_ticket=ticket),
        _entry("3363-design-1", "a"),
    )
    violations = validate_report(report)
    assert any("3041-correctness-0" in v for v in violations)


def test_a_live_defect_with_a_ticket_is_accepted():
    report = _report(
        _entry("3041-correctness-0", "b", followup_ticket="tkt_abc123"),
        _entry("3363-design-1", "b", followup_ticket="tkt_def456"),
    )
    assert validate_report(report) == []


@pytest.mark.parametrize("disposition", ["a", "c"])
def test_a_ticket_on_a_not_live_finding_is_rejected(disposition):
    """The task says file follow-ups for group (b) and ONLY group (b). A ticket
    hung off an (a) or (c) means either the disposition or the filing is wrong."""
    report = _report(
        _entry("3041-correctness-0", disposition, followup_ticket="tkt_abc123"),
        _entry("3363-design-1", "a"),
    )
    violations = validate_report(report)
    assert any("3041-correctness-0" in v for v in violations)


def test_a_gate_deferred_entry_with_a_note_is_accepted():
    """The stop path is a first-class recorded outcome, not an abandoned one."""
    report = _report(
        _entry("3041-correctness-0", None, note="Deferred by the gate: all five "
               "priority findings dispositioned (a)/(c), so the 18 mediums are "
               "low yield.", deferred_by_gate=True),
        _entry("3363-design-1", "a"),
    )
    assert validate_report(report) == []


@pytest.mark.parametrize("note", [None, "", "   "])
def test_a_gate_deferred_entry_without_a_note_is_rejected(note):
    """Deferral must cite the gate's rationale. Otherwise the stop path becomes
    a way to leave a finding unaccounted for while still validating."""
    report = _report(
        _entry("3041-correctness-0", None, note=note, deferred_by_gate=True),
        _entry("3363-design-1", "a"),
    )
    violations = validate_report(report)
    assert any("3041-correctness-0" in v for v in violations)


def test_a_null_disposition_is_only_excused_by_an_explicit_deferral():
    """`disposition: null` with no `deferred_by_gate` is the seeded starting
    state — it must NOT validate, or the report could ship untriaged."""
    report = _report(
        _entry("3041-correctness-0", None),
        _entry("3363-design-1", None),
    )
    violations = validate_report(report)
    assert len(violations) >= 2


def test_a_gate_deferred_entry_may_not_also_claim_a_disposition():
    """Deferred and dispositioned are mutually exclusive: an entry that claims
    both leaves a reader unable to say whether the finding was read."""
    report = _report(
        _entry("3041-correctness-0", "a", deferred_by_gate=True),
        _entry("3363-design-1", "a"),
    )
    violations = validate_report(report)
    assert any("3041-correctness-0" in v for v in violations)


def test_the_seeded_report_shape_fails_loudly():
    """Step 7 seeds all 23 ids with `disposition: null`. That report MUST be
    rejected — a validator that passed it would certify an untriaged file."""
    report = _report(*[_entry(rid, None, note=None) for rid in ROSTER_IDS])
    assert validate_report(report)


# ---------------------------------------------------------------------------
# The degenerate cases, which are a sharper requirement than the ones above.
# Every check in `validate_report` reads "each roster id appears once, validly"
# — a shape an EMPTY roster satisfies vacuously. investigation.md leans on
# `--validate ... exits 0` as the proof that all 23 findings were accounted
# for, so a gate that certifies an empty or structurally-broken file inverts
# the one thing it exists to do. Same principle as
# `test_the_seeded_report_shape_fails_loudly`, one step further down.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("report", [
    {},
    {"roster": [], "dispositions": []},
    {"roster": []},
    {"dispositions": [{"id": "3041-correctness-0", "disposition": "a", "note": "n"}]},
    {"rooster": [{"id": "3041-correctness-0"}], "dispositions": []},
    {"roster": None, "dispositions": None},
], ids=["empty", "both-empty", "roster-only", "no-roster-key", "misspelled-roster", "nulls"])
def test_a_report_with_no_usable_roster_is_never_certified_complete(report):
    """A completeness gate that passes a file containing nothing is worse than
    no gate: it reads as proof. Each of these used to return [] and exit 0."""
    assert validate_report(report), "an empty roster must not validate"


def test_a_roster_entry_without_an_id_is_rejected():
    report = _report(_entry("3363-design-1", "a"),
                     roster_ids=ROSTER_IDS)
    report["roster"].append({"task": "3041", "severity_as_emitted": "high"})
    violations = validate_report(report)
    assert any("no usable id" in v for v in violations)


@pytest.mark.parametrize("blank", [None, "", "   "])
def test_a_nameless_disposition_cannot_pair_with_a_nameless_roster_entry(blank):
    """One field narrower than the empty-roster case, and the reason a blank id
    is excluded from the match rather than merely reported: `None` matches
    `None`, so two entries that both lack an id would satisfy each other and
    the report would validate with nothing actually dispositioned."""
    report = {
        "roster": [{"id": blank, "task": "3041"}],
        "dispositions": [{"id": blank, "disposition": "a", "note": "Checked."}],
    }
    violations = validate_report(report)

    assert len(violations) >= 2, violations
    assert any("roster entry carries no usable id" in v for v in violations)
    assert any("disposition entry carries no usable id" in v for v in violations)


def test_a_nameless_disposition_does_not_excuse_a_real_roster_id():
    """The nameless entry must not be counted as having dispositioned anything:
    the roster ids it does not name are still owed a disposition."""
    report = {
        "roster": [{"id": rid} for rid in ROSTER_IDS],
        "dispositions": [{"disposition": "a", "note": "Checked."}],
    }
    violations = validate_report(report)

    assert all(any(rid in v for v in violations) for rid in ROSTER_IDS)


def test_the_real_reports_shape_still_validates_after_the_guard():
    """The guard must reject the degenerate shapes without rejecting the shape
    the closing step actually runs — a roster and one disposition per id."""
    assert validate_report(_report(
        _entry("3041-correctness-0", "b", followup_ticket="tkt_abc123"),
        _entry("3363-design-1", "c"),
    )) == []


def test_violations_are_readable_strings():
    report = _report(_entry("3041-correctness-0", "a"))
    violations = validate_report(report)
    assert violations and all(isinstance(v, str) and v.strip() for v in violations)


# ---------------------------------------------------------------------------
# main() --validate, through the CLI, because the exit code is the contract a
# closing step actually relies on.
# ---------------------------------------------------------------------------

SCRIPT = Path(__file__).resolve().parents[1] / "audit_offcontract_review_findings.py"


def _run_validate(tmp_path, report):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--validate", str(path)],
        capture_output=True, text=True, check=False,
    )


def test_cli_validate_exits_zero_on_a_complete_report(tmp_path):
    result = _run_validate(tmp_path, _report(
        _entry("3041-correctness-0", "b", followup_ticket="tkt_abc123"),
        _entry("3363-design-1", "c"),
    ))
    assert result.returncode == 0, result.stdout + result.stderr


def test_cli_validate_exits_nonzero_and_names_the_violations(tmp_path):
    result = _run_validate(tmp_path, _report(
        _entry("3041-correctness-0", "b", followup_ticket=None),
    ))
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "3041-correctness-0" in combined
    assert "3363-design-1" in combined, "the missing roster id must be named too"
