"""Tests for how scripts/audit_manifest_descriptor_drift.py renders and exits:
the pure formatters (format_report, format_json), _is_dirty's non-vacuity rule,
and main() driven by subprocess, for every reported dimension.

Mirrors test_audit_combine_gate_marker_loss.py: pure functions get direct pytest
coverage; ``main()`` gets subprocess coverage.
"""
from __future__ import annotations

import json

from _task_db_scan import (
    AUDIT_EXIT_FINDINGS,
    AUDIT_EXIT_NO_ROOT,
    AUDIT_EXIT_NOTHING_AUDITED,
    AUDIT_EXIT_OK,
    format_kv_line,
)
from audit_manifest_descriptor_drift import (
    _COVERAGE_CAVEAT,
    _DISCOVERY_FAILED_NOTICE,
    _UNBOUND_CAVEAT,
    EXIT_DRIFT,
    EXIT_NO_ROOT,
    EXIT_NOTHING_AUDITED,
    EXIT_OK,
    _is_dirty,
    audit_project,
    format_json,
    format_report,
)
from manifest_drift_fixtures import (
    GREP_CHECK,
    check_entry,
    declaring_sidecar,
    git_init,
    labelled_row,
    make_project,
    manifest_doc,
    one_project,
    run_cli,
    task_row,
    write_manifest,
)

# ---------------------------------------------------------------------------
# Non-vacuity / loudness. A silently-empty corpus must NEVER render as a clean
# zero: an empty corpus and a clean corpus are indistinguishable in the finding
# count, and only one of them is good news.
# ---------------------------------------------------------------------------

def test_manifest_root_that_is_not_a_git_checkout_is_DIRTY_not_clean(
        tmp_path, make_tasks_db):
    """git discovery failure is recorded, not degraded to an empty sweep."""
    root = one_project(tmp_path, make_tasks_db,
                        sidecar_check=GREP_CHECK,
                        task_entry=check_entry("gate", GREP_CHECK))
    not_a_checkout = tmp_path / "bare"
    not_a_checkout.mkdir()

    audit = audit_project(str(root), str(not_a_checkout))

    assert audit.coverage.git_discovery_failed is True
    assert audit.findings == []
    assert audit.coverage.manifests_swept == 0
    # NAMED, not merely flagged.
    assert any("ls-files" in d for d in audit.coverage.uncomparable_details)
    assert _is_dirty([audit]) is True


def test_a_root_with_zero_tracked_manifests_is_clean_not_dirty(
        tmp_path, make_tasks_db):
    """THE OTHER HALF, pinned so the distinction cannot collapse.

    A foreign project may legitimately have no capability manifests at all.
    That is a real, complete, clean sweep of an empty corpus — unlike the
    discovery FAILURE above, where the corpus size is unknown.
    """
    root = make_project(tmp_path, make_tasks_db, tasks=[task_row(100, [])])

    audit = audit_project(str(root))

    assert audit.coverage.manifests_swept == 0
    assert audit.coverage.git_discovery_failed is False
    assert _is_dirty([audit]) is False


# ---------------------------------------------------------------------------
# Formatting. Both formatters are PURE — they return str and print nothing;
# main() does the single print.
# ---------------------------------------------------------------------------

def _drifted_audit(tmp_path, make_tasks_db):
    root = one_project(tmp_path, make_tasks_db,
                        sidecar_check=GREP_CHECK,
                        task_entry=check_entry("gate", {**GREP_CHECK, "pattern": "def bar"}))
    return audit_project(str(root))


def _unbound_audit(tmp_path, make_tasks_db):
    """Nothing drifted; one historical and one LIVE unbound label on one sidecar."""
    root = make_project(
        tmp_path, make_tasks_db, name="unbound",
        tasks=[labelled_row(7, status="done", label="ω"),
               labelled_row(8, status="pending", label="ψ")],
        manifests=[("plans/x-prd.capability-manifest.yaml", declaring_sidecar("η0", "α"))],
    )
    return audit_project(str(root))


def test_formatters_are_pure_and_print_nothing(tmp_path, make_tasks_db, capsys):
    """Both return str and neither prints — main() does the single print."""
    audits = [_drifted_audit(tmp_path, make_tasks_db),
              _unbound_audit(tmp_path, make_tasks_db)]
    capsys.readouterr()

    assert isinstance(format_report(audits), str)
    assert isinstance(format_json(audits), str)
    assert capsys.readouterr().out == ""


def test_coverage_block_is_printed_even_on_a_zero_finding_sweep(
        tmp_path, make_tasks_db):
    """ALWAYS printed. The whole point is that the finding list is a comparison
    of matched pairs, and a reader must be told the size of the remainder."""
    root = one_project(tmp_path, make_tasks_db,
                        sidecar_check=GREP_CHECK,
                        task_entry=check_entry("gate", GREP_CHECK))

    report = format_report([audit_project(str(root))])

    assert "COVERAGE" in report
    assert _COVERAGE_CAVEAT in report


def test_report_coverage_rows_render_with_their_column_alignment(
        tmp_path, make_tasks_db):
    """WHOLE LINES with their alignment, never a bare digit substring.

    A bare `assert "32" in report` passes off any number anywhere in the text,
    including a task id, so it cannot tell a correct count from a coincidence.
    """
    root = make_project(
        tmp_path, make_tasks_db,
        tasks=[task_row(100, [check_entry("other", GREP_CHECK)])],
        manifests=[("plans/a-prd.capability-manifest.yaml", manifest_doc(100))],
    )

    lines = format_report([audit_project(str(root))]).splitlines()

    assert "    manifests swept:                    1" in lines
    # SEEN and COMPARED render as two ADJACENT rows: the eligible population
    # and the matched-pair count are different numbers, and a report whose
    # thesis is "the findings are not the whole corpus" must not overstate the
    # one figure that says how much was actually compared.
    assert "    mechanical capabilities seen:       1" in lines
    assert "    mechanical capabilities compared:   0" in lines
    assert "    capabilities with no task entry:    1" in lines
    assert "    task entries with no capability:    1" in lines
    assert "    manifest tasks with no db row:      0" in lines
    # The two absence classes render ADJACENTLY and on the same gutter, so an
    # operator reading a nonzero no-db-row count sees, in the next line, the
    # class it is NOT. The 30-char label is under _COVERAGE_LABEL_WIDTH (36),
    # so its value lands at column 40 like every sibling row.
    assert "    external-registry task blocks:      0" in lines
    assert lines.index("    external-registry task blocks:      0") == (
        lines.index("    manifest tasks with no db row:      0") + 1)
    # And the MIS-AUTHORED form of that class renders directly beneath it, on
    # the same gutter, for the mirror-image reason: printed apart, a nonzero
    # count here would read as more of the explained kind. 27-char label, so
    # its value lands at column 40 too.
    assert "    self-bound external blocks:         0" in lines
    assert lines.index("    self-bound external blocks:         0") == (
        lines.index("    external-registry task blocks:      0") + 1)
    assert "    unvalidatable task entries:         0" in lines
    assert "    unconvertible sidecar descriptors:  0" in lines
    assert "    manifests that failed to parse:     0" in lines


def test_finding_line_names_manifest_task_capability_and_fields(
        tmp_path, make_tasks_db):
    """Every finding line carries the identity triple AND what differs."""
    report = format_report([_drifted_audit(tmp_path, make_tasks_db)])

    assert format_kv_line([
        ("manifest", "plans/a-prd.capability-manifest.yaml"),
        ("task_id", 100),
        ("capability", "gate"),
        ("fields", "pattern"),
    ]) in report.splitlines()


def test_report_names_both_roots_so_a_decoupled_run_is_unambiguous(
        tmp_path, make_tasks_db):
    root = make_project(tmp_path, make_tasks_db, tasks=[task_row(100, [])])
    other = make_project(tmp_path, make_tasks_db, name="other")

    report = format_report([audit_project(str(root), str(other))])

    assert str(root) in report
    assert str(other) in report


def test_report_says_so_prominently_when_git_discovery_failed(
        tmp_path, make_tasks_db):
    """A silently-empty corpus must never RENDER as a clean zero either."""
    root = make_project(tmp_path, make_tasks_db, tasks=[task_row(100, [])])
    not_a_checkout = tmp_path / "bare"
    not_a_checkout.mkdir()

    report = format_report([audit_project(str(root), str(not_a_checkout))])

    # On the CONSTANT, matching the _COVERAGE_CAVEAT test's shape. A hardcoded
    # substring pins WORDING rather than behaviour: it fails a reword that is
    # just as loud, and survives a rewrite that guts the warning.
    assert _DISCOVERY_FAILED_NOTICE in report
    # ABOVE the rows, so a reader sees the zero is UNKNOWN before reading it.
    lines = report.splitlines()
    assert lines.index(_DISCOVERY_FAILED_NOTICE) < next(
        i for i, ln in enumerate(lines) if "drifted descriptors" in ln)


def test_format_json_emits_an_object_with_projects_coverage_and_findings(
        tmp_path, make_tasks_db):
    """An OBJECT, not an array: coverage travels WITH the findings, so a
    machine consumer cannot read the finding list without the caveat."""
    payload = json.loads(format_json([_drifted_audit(tmp_path, make_tasks_db)]))

    assert isinstance(payload, dict)
    assert set(payload) >= {"projects"}
    (project,) = payload["projects"]
    assert project["project_root"].endswith("proj")
    assert project["manifest_root"] == project["project_root"]
    assert project["coverage"]["mechanical_capabilities_compared"] == 1
    # Every coverage class travels, including the ones that are zero on this
    # sweep: a machine consumer that only ever sees a key when it is nonzero
    # cannot tell "none of these" from "this build does not report them".
    assert project["coverage"]["external_registry_task_blocks"] == 0
    assert project["coverage"]["self_bound_external_task_blocks"] == 0
    assert project["coverage"]["git_discovery_failed"] is False
    (finding,) = project["findings"]
    assert finding["manifest"] == "plans/a-prd.capability-manifest.yaml"
    assert finding["task_id"] == 100
    assert finding["capability"] == "gate"
    assert finding["differing_fields"] == ["pattern"]
    # BOTH normalized descriptors travel, so a consumer never has to re-read
    # the two sources to learn what the two spellings actually were.
    assert finding["sidecar_check"]["pattern"] == "def foo"
    assert finding["task_check"]["pattern"] == "def bar"
    # The caveat travels in the payload too, for the same reason.
    assert _COVERAGE_CAVEAT in json.dumps(payload)


# ---------------------------------------------------------------------------
# main() — driven by SUBPROCESS, never in-process.
#
# Shelling out to the script PATH (never `python -m`) is also what PROVES the
# flat-sibling `import _task_db_scan` resolves: a directly-executed script puts
# its own directory at sys.path[0], and scripts/tests/conftest.py's sys.path
# insertion does not reach a child process.
#
# Every test passes an explicit tmp_path --project-root, so the LIVE default
# root (/home/leo/src/dark-factory) is never reached from this suite.
#   exit 0 = no drift; 1 = drift or a failed discovery; 2 = no root resolved;
#   3 = roots resolved but NOTHING was audited.
# ---------------------------------------------------------------------------

def test_main_exit_0_on_a_clean_project_and_still_prints_coverage(
        tmp_path, make_tasks_db):
    """A clean sweep still shows its coverage — see _format_coverage."""
    root = one_project(tmp_path, make_tasks_db,
                        sidecar_check=GREP_CHECK,
                        task_entry=check_entry("gate", GREP_CHECK))

    result = run_cli("--project-root", str(root))

    assert result.returncode == 0, result.stderr
    assert "COVERAGE" in result.stdout


def test_main_exit_1_and_names_the_drifted_row(tmp_path, make_tasks_db):
    root = one_project(tmp_path, make_tasks_db,
                        sidecar_check=GREP_CHECK,
                        task_entry=check_entry("gate", {**GREP_CHECK, "pattern": "def bar"}))

    result = run_cli("--project-root", str(root))

    assert result.returncode == 1
    assert "plans/a-prd.capability-manifest.yaml" in result.stdout
    assert "task_id=100" in result.stdout
    assert "capability=gate" in result.stdout
    assert "fields=pattern" in result.stdout


def test_main_exit_2_when_no_project_root_resolves(tmp_path):
    """The literal 2, NOT the imported constant.

    Deliberate, and the sibling suite records why: importing the constant lets
    a renumber stay green while the epilog keeps promising 2 to operators and
    CI. The number in the contract is what a consumer branches on.
    """
    result = run_cli("--project-root", str(tmp_path / "no-such-project"))

    assert result.returncode == 2
    assert "no project root" in result.stderr.lower()


def test_main_exit_3_when_the_only_tasks_db_is_unreadable(tmp_path):
    """Roots resolved but every one failed, so NOTHING was audited.

    3, hardcoded, for the same reason as 2 above. Never treat it as clean:
    stdout on this path is a well-formed EMPTY payload.
    """
    root = tmp_path / "corrupt"
    (root / ".taskmaster" / "tasks").mkdir(parents=True)
    (root / ".taskmaster" / "tasks" / "tasks.db").write_text("this is not a database")

    result = run_cli("--project-root", str(root))

    assert result.returncode == 3
    assert "nothing was" in result.stderr.lower()


def test_main_json_payload_shape(tmp_path, make_tasks_db):
    root = one_project(tmp_path, make_tasks_db,
                        sidecar_check=GREP_CHECK,
                        task_entry=check_entry("gate", {**GREP_CHECK, "pattern": "def bar"}))

    result = run_cli("--project-root", str(root), "--json")

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    (project,) = payload["projects"]
    assert project["coverage"]["mechanical_capabilities_compared"] == 1
    assert project["coverage"]["external_registry_task_blocks"] == 0
    assert project["coverage"]["self_bound_external_task_blocks"] == 0
    assert [f["capability"] for f in project["findings"]] == ["gate"]


def test_main_project_root_is_repeatable(tmp_path, make_tasks_db):
    """Each root produces its own audit block in ONE render."""
    a = one_project(tmp_path, make_tasks_db,
                     sidecar_check=GREP_CHECK,
                     task_entry=check_entry("gate", GREP_CHECK))
    b = make_project(
        tmp_path, make_tasks_db, name="b",
        tasks=[task_row(200, [check_entry("gate", {**GREP_CHECK, "pattern": "drifted"})])],
        manifests=[("plans/b-prd.capability-manifest.yaml",
                    manifest_doc(200, prd="plans/b-prd.md"))],
    )

    result = run_cli("--project-root", str(a), "--project-root", str(b), "--json")

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert [p["project_root"] for p in payload["projects"]] == [str(a), str(b)]
    assert [len(p["findings"]) for p in payload["projects"]] == [0, 1]


def test_manifest_root_decouples_the_manifest_tree_from_the_task_store(
        tmp_path, make_tasks_db):
    """THE FLAG'S REASON FOR EXISTING, with both halves asserted.

    `.taskmaster/` is gitignored and exists only in the primary checkout, so a
    task WORKTREE has no tasks.db at all — and discover_project_roots DROPS a
    root without one. Sweeping a worktree's sidecars therefore requires reading
    the manifests from tree B while reading the task store from root A.

    The second half is what makes this non-vacuous: WITHOUT the flag the same
    invocation reports A's own (empty) manifest tree, so the flag is shown to
    genuinely change discovery rather than merely being accepted.
    """
    store = make_project(tmp_path, make_tasks_db, name="store",
                          tasks=[task_row(300, [check_entry("gate", {**GREP_CHECK,
                                                             "pattern": "drifted"})])])
    sidecars = tmp_path / "sidecars"
    write_manifest(sidecars, "plans/w-prd.capability-manifest.yaml",
                    manifest_doc(300, prd="plans/w-prd.md"))
    git_init(sidecars)

    decoupled = run_cli("--project-root", str(store),
                         "--manifest-root", str(sidecars), "--json")
    assert decoupled.returncode == 1
    (project,) = json.loads(decoupled.stdout)["projects"]
    assert project["project_root"] == str(store)
    assert project["manifest_root"] == str(sidecars)
    assert [f["capability"] for f in project["findings"]] == ["gate"]

    # The report NAMES both roots, so a decoupled run is unambiguous on stdout.
    named = run_cli("--project-root", str(store), "--manifest-root", str(sidecars))
    assert str(store) in named.stdout and str(sidecars) in named.stdout

    # WITHOUT the flag: A's own manifest tree, which is empty. Same task store,
    # same drifted record, zero findings — so the flag changed discovery.
    default = run_cli("--project-root", str(store), "--json")
    assert default.returncode == 0
    (only,) = json.loads(default.stdout)["projects"]
    assert only["manifest_root"] == str(store)
    assert only["findings"] == []
    assert only["coverage"]["manifests_swept"] == 0


def test_manifest_root_with_two_project_roots_warns_and_still_runs(
        tmp_path, make_tasks_db):
    """The on_roots warn-not-fail shape: ONE manifest tree over MANY task
    stores is a coherent thing to ask for, so it is warned about, not
    rejected."""
    a = make_project(tmp_path, make_tasks_db, name="a", tasks=[task_row(1, [])])
    b = make_project(tmp_path, make_tasks_db, name="b", tasks=[task_row(2, [])])
    sidecars = make_project(tmp_path, make_tasks_db, name="side")

    result = run_cli("--project-root", str(a), "--project-root", str(b),
                      "--manifest-root", str(sidecars), "--json")

    assert result.returncode == 0
    assert "warning" in result.stderr.lower()
    assert len(json.loads(result.stdout)["projects"]) == 2


def test_main_manifest_root_that_is_not_a_checkout_is_loudly_non_zero(
        tmp_path, make_tasks_db):
    """It must NOT report a clean zero — the corpus size is UNKNOWN, not zero."""
    root = make_project(tmp_path, make_tasks_db, tasks=[task_row(100, [])])
    not_a_checkout = tmp_path / "bare"
    not_a_checkout.mkdir()

    result = run_cli("--project-root", str(root), "--manifest-root", str(not_a_checkout))

    assert result.returncode != 0
    # On the CONSTANT, not a hardcoded substring — see the format_report
    # counterpart above for why.
    assert _DISCOVERY_FAILED_NOTICE in result.stdout


def test_main_run_is_strictly_read_only(tmp_path, make_tasks_db):
    """THE READ-ONLY CLAIM, CHECKED. The tasks.db AND every manifest YAML have
    their (mtime, sha256) captured before and after a full run that reports
    BOTH a drifted descriptor and an unbound label, and the whole mapping must
    be unchanged."""
    import hashlib

    root = make_project(
        tmp_path, make_tasks_db,
        tasks=[task_row(100, [check_entry("gate", {**GREP_CHECK, "pattern": "drifted"})]),
               labelled_row(200, prd_path="plans/a-prd.md", label="omega")],
        manifests=[
            ("plans/a-prd.capability-manifest.yaml", manifest_doc(100)),
            ("docs/prds/b-prd.capability-manifest.yaml",
             manifest_doc(100, label="γ", prd="docs/prds/b-prd.md")),
        ],
    )
    inputs = [
        root / ".taskmaster" / "tasks" / "tasks.db",
        root / "plans" / "a-prd.capability-manifest.yaml",
        root / "docs" / "prds" / "b-prd.capability-manifest.yaml",
    ]

    def fingerprint():
        return {
            str(p): (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest())
            for p in inputs
        }

    before = fingerprint()
    result = run_cli("--project-root", str(root))
    assert result.returncode == 1
    # The run genuinely exercised the label-binding direction too.
    assert "label=omega" in result.stdout

    assert fingerprint() == before


# The label-binding direction's exit contract. A LIVE unbound label is dirty; a
# historical one is reported but not dirty (see UnboundLabel.is_live). Labels
# here are ASCII so no assertion depends on the child's stdout encoding.

def test_main_exit_1_when_a_live_unbound_label_is_the_only_issue(
        tmp_path, make_tasks_db):
    root = make_project(
        tmp_path, make_tasks_db,
        tasks=[labelled_row(7, status="pending", label="kappa-followup")],
        manifests=[("plans/x-prd.capability-manifest.yaml", declaring_sidecar("kappa"))],
    )

    result = run_cli("--project-root", str(root))

    assert result.returncode == 1
    assert _unbound_line(7, "kappa-followup", "pending", "LIVE") in result.stdout.splitlines()


def test_main_exit_0_when_every_unbound_label_is_historical_but_names_them_all(
        tmp_path, make_tasks_db):
    """Clean, but never silent: a done or cancelled row cannot be touched by a
    future planning batch, so it does not make the run dirty, and it is still
    in the report."""
    root = make_project(
        tmp_path, make_tasks_db,
        tasks=[labelled_row(7, status="done", label="omega"),
               labelled_row(8, status="cancelled", label="psi")],
        manifests=[("plans/x-prd.capability-manifest.yaml", declaring_sidecar("alpha"))],
    )

    result = run_cli("--project-root", str(root))

    assert result.returncode == 0, result.stdout
    lines = result.stdout.splitlines()
    assert _unbound_line(7, "omega", "done", "historical") in lines
    assert _unbound_line(8, "psi", "cancelled", "historical") in lines


def test_main_drift_plus_a_historical_unbound_label_still_exits_1(
        tmp_path, make_tasks_db):
    root = make_project(
        tmp_path, make_tasks_db,
        tasks=[task_row(100, [check_entry("gate", {**GREP_CHECK, "pattern": "def bar"})]),
               labelled_row(7, status="done", prd_path="plans/a-prd.md", label="omega")],
        manifests=[("plans/a-prd.capability-manifest.yaml",
                    manifest_doc(100, prd="plans/a-prd.md"))],
    )

    assert run_cli("--project-root", str(root)).returncode == 1


def test_main_exit_0_with_neither_drift_nor_an_unbound_label(tmp_path, make_tasks_db):
    """A live task whose label its sidecar declares is BOUND, so nothing is
    wrong."""
    root = make_project(
        tmp_path, make_tasks_db,
        tasks=[labelled_row(7, status="pending", label="alpha")],
        manifests=[("plans/x-prd.capability-manifest.yaml", declaring_sidecar("alpha"))],
    )

    assert run_cli("--project-root", str(root)).returncode == 0


def test_main_discovery_failure_exits_1_whatever_the_unbound_list_holds(
        tmp_path, make_tasks_db):
    """Only historical labels here, which alone would be clean, but the corpus
    was never enumerated, so no label was classified at all."""
    root = make_project(tmp_path, make_tasks_db,
                         tasks=[labelled_row(7, status="done", label="omega")])
    not_a_checkout = tmp_path / "bare"
    not_a_checkout.mkdir()

    result = run_cli("--project-root", str(root), "--manifest-root", str(not_a_checkout))

    assert result.returncode == 1


def test_main_json_carries_unbound_labels_with_their_liveness(tmp_path, make_tasks_db):
    root = make_project(
        tmp_path, make_tasks_db,
        tasks=[labelled_row(7, status="done", label="omega"),
               labelled_row(8, status="pending", label="psi")],
        manifests=[("plans/x-prd.capability-manifest.yaml", declaring_sidecar("alpha"))],
    )

    result = run_cli("--project-root", str(root), "--json")

    assert result.returncode == 1
    (project,) = json.loads(result.stdout)["projects"]
    assert [(row["task_id"], row["label"], row["is_live"])
            for row in project["unbound_labels"]] == [(7, "omega", False), (8, "psi", True)]


def test_exit_constants_alias_the_shared_tier_3_codes():
    """The per-script NAMES survive; the VALUES have ONE home.

    Keeps the aliases honest, exactly as the sibling audit's counterpart does:
    a local re-spelling would drift from what run_audit_cli actually returns.
    """
    assert (EXIT_OK, EXIT_DRIFT, EXIT_NO_ROOT, EXIT_NOTHING_AUDITED) == (
        AUDIT_EXIT_OK, AUDIT_EXIT_FINDINGS, AUDIT_EXIT_NO_ROOT, AUDIT_EXIT_NOTHING_AUDITED)


# ---------------------------------------------------------------------------
# REPORTING THE LABEL-BINDING DIRECTION (task 4907).
#
# Every unbound row is NAMED, historical and live alike, and the section prints
# even when it is empty: a reader must be able to tell "looked and found none"
# from "never looked".
# ---------------------------------------------------------------------------

def _unbound_line(task_id, label, status, hazard):
    return format_kv_line([
        ("manifest", "plans/x-prd.capability-manifest.yaml"),
        ("task_id", task_id),
        ("label", label),
        ("status", status),
        ("hazard", hazard),
    ])


def test_report_names_every_unbound_row_and_marks_the_live_one(
        tmp_path, make_tasks_db):
    """Each row line carries (manifest, task_id, label, status) and a hazard that
    tells LIVE from historical at a glance. The sidecar's declared labels ride
    on the continuation line, in the sidecar's own order."""
    lines = format_report([_unbound_audit(tmp_path, make_tasks_db)]).splitlines()

    for row_line in (_unbound_line(7, "ω", "done", "historical"),
                     _unbound_line(8, "ψ", "pending", "LIVE")):
        assert row_line in lines
        assert lines[lines.index(row_line) + 1] == "      declared labels: η0, α"


def test_the_unbound_section_and_its_caveat_print_even_when_empty(
        tmp_path, make_tasks_db):
    """Never silent, for the reason the coverage block always prints. The caveat
    is asserted by its CONSTANT, as the coverage caveat's test does."""
    root = one_project(tmp_path, make_tasks_db,
                        sidecar_check=GREP_CHECK,
                        task_entry=check_entry("gate", GREP_CHECK))

    report = format_report([audit_project(str(root))])

    assert "  -- unbound task labels (0, 0 live) --" in report.splitlines()
    assert _UNBOUND_CAVEAT in report


def test_the_total_line_counts_every_dimension_and_the_live_rows(
        tmp_path, make_tasks_db):
    """A report carrying several kinds of row must not close on a drift-only total."""
    audits = [_drifted_audit(tmp_path, make_tasks_db),
              _unbound_audit(tmp_path, make_tasks_db)]

    last = format_report(audits).splitlines()[-1]

    assert last == (
        "1 drifted descriptor(s), 2 unbound task label(s) (1 live), "
        "0 stale signal binding(s) across 2 project(s)")


def test_discovery_failed_notice_sits_above_the_unbound_section_too(
        tmp_path, make_tasks_db):
    """An empty unbound list from an unenumerable corpus is as meaningless as an
    empty finding list, so the notice precedes both."""
    root = make_project(tmp_path, make_tasks_db, tasks=[labelled_row(7)])
    not_a_checkout = tmp_path / "bare"
    not_a_checkout.mkdir()

    lines = format_report([audit_project(str(root), str(not_a_checkout))]).splitlines()

    assert lines.index(_DISCOVERY_FAILED_NOTICE) < next(
        i for i, ln in enumerate(lines) if "unbound task labels" in ln)


def test_format_json_carries_unbound_rows_with_an_explicit_liveness_flag(
        tmp_path, make_tasks_db):
    """``is_live`` is a property, so ``_asdict()`` alone would drop it; the JSON
    writer carries it explicitly. The new caveat travels under its own key,
    leaving the drift caveat as it was."""
    payload = json.loads(format_json([_unbound_audit(tmp_path, make_tasks_db)]))

    (project,) = payload["projects"]
    assert project["unbound_labels"] == [
        {"task_id": 7, "label": "ω", "status": "done",
         "manifest": "plans/x-prd.capability-manifest.yaml",
         "declared_labels": ["η0", "α"], "is_live": False},
        {"task_id": 8, "label": "ψ", "status": "pending",
         "manifest": "plans/x-prd.capability-manifest.yaml",
         "declared_labels": ["η0", "α"], "is_live": True},
    ]
    assert payload["unbound_labels_caveat"] == _UNBOUND_CAVEAT
    assert payload["caveat"] == _COVERAGE_CAVEAT
