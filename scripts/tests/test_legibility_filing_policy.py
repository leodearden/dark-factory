"""Tests for scripts/legibility/filing_policy.py — whether and where the
legibility census files a verified confusion cluster (task 5931).

Imported by bare name (`import filing_policy`), resolved via
scripts/tests/conftest.py's scripts/legibility/ sys.path insertion.

The regression fixture copies, verbatim and hermetically, the fields of real
candidates from reify's docs/legibility/confusion-codebook.yaml: four whose
census tasks were misfiled into reify although their fix surface is
dark-factory's, and five genuinely reify-actionable (or singleton) ones.
"""
from __future__ import annotations

import filing_policy
import pytest

HARNESS_ROOT = "/home/leo/src/dark-factory"


def _cluster(*, title, cause, area, evidence_quote):
    """A cluster shaped like census.py::_novel_clusters output."""
    return {
        "title": title,
        "summary": cause,
        "cause": cause,
        "area": area,
        "evidence": [evidence_quote],
    }


REIFY_7894 = _cluster(  # cand-20260807-1
    title="Dependency edge added 'for bookkeeping' renders the new task self-blocking by construction",
    cause=(
        "Task dependency edges in the fused-memory task system have only one enforced (blocking) "
        "semantics — there is no non-blocking/informational edge type. When a fix task is filed "
        "and given a dependency purely to preserve transitive-dependency history ('for "
        "bookkeeping'), that edge is nonetheless enforced as a real blocking constraint. If the "
        "fix task's dependency chain loops back onto the defect it exists to remediate, the task "
        "becomes self-blocking by construction, forcing otherwise-avoidable L2 escalation-watcher "
        "involvement until a human catches and removes the edge."
    ),
    area="task-dependency-management / fused-memory task system",
    evidence_quote=(
        '"6084 is self-blocking by construction": Thanks for spotting that, it makes perfect '
        "sense now you explain it. That's not what I intended, drop the dep on 5261. The "
        "bookkeeping of that dep is less important than 6084 getting to do the work it's "
        "designed to do without bothering the L2 escalation watcher or me."
    ),
)

REIFY_7895 = _cluster(  # cand-20260921-14
    title="fused-memory get_task call times out with no diagnostic detail",
    cause=(
        "An orchestrated-task session's mcp__fused-memory__get_task call for a task record "
        "returns a bare 'operation timed out' with no indication of server load, retry "
        "guidance, or whether the read partially succeeded, leaving the agent unable to "
        "distinguish transient latency from a stuck server."
    ),
    area="fused-memory MCP / task-store reads",
    evidence_quote=(
        'mcp__fused-memory__get_task({"id": "6591", "project_root": "/home/leo/src/reify"}) '
        "-> The operation timed out."
    ),
)

REIFY_7900 = _cluster(  # cand-20260909-13
    title=(
        "Agent invokes hallucinated generic tool names (read_file, git_log) absent from its "
        "actual toolset"
    ),
    cause=(
        "Agent assumed conventional/generic tool names (read_file, git_log) exist rather than "
        "using the session's actual registered tools (Read, Bash git log), producing repeated "
        "'No such tool available' errors mid-verification"
    ),
    area="tooling",
    evidence_quote=(
        'read_file({"max_lines": "50", "path": "crates/reify-compiler/src/type_resolution.rs"}) '
        "-> <tool_use_error>Error: No such tool available: read_file</tool_use_error>"
    ),
)

REIFY_7901_RETRY = _cluster(  # cand-20260918-12
    title=(
        "add_memory MCP call times out and is retried with an identical payload, risking a "
        "duplicate write"
    ),
    cause=(
        "The fused-memory add_memory call for a procedural_knowledge entry returned 'The "
        "operation timed out.'; the agent immediately retried the exact same content rather "
        "than checking whether the write had actually landed server-side before the timeout "
        "was surfaced client-side, risking a duplicate memory entry if the original attempt in "
        "fact succeeded."
    ),
    area="fused-memory MCP tool reliability",
    evidence_quote=(
        "(turn 276) mcp__fused-memory__add_memory(...) -> The operation timed out. / (turn 281) "
        "mcp__fused-memory__add_memory(... same content ...) -> The operation timed out."
    ),
)

REIFY_7901_SESSION_END = _cluster(  # cand-20260819-20
    title="fused-memory add_memory call times out on session-end reflection write",
    cause=(
        "mcp__fused-memory__add_memory invoked near session end (category "
        "observations_and_summaries) returns 'The operation timed out.' with no retry, "
        "fallback, or surfaced recovery path, silently discarding the distilled session summary"
    ),
    area="memory-write reliability",
    evidence_quote=(
        'mcp__fused-memory__add_memory({"agent_id": "claude-interactive-prd-5429-residuals", '
        '"category": "observations_and_summaries", "content": "Session 2026-08-20 '
        '(enum-shadow-coherence PRD authoring): the #5429 \\"local enum shadows pre...) -> The '
        "operation timed out."
    ),
)

REIFY_7909 = _cluster(  # cand-20260901-17
    title="sed -n <start>,<end> without trailing print command yields 'missing command' error",
    cause=(
        "Agent invoked `sed -n <start>,<end> <file>` intending to print a line range but "
        "omitted the required `p` (or other) command after the address range, so sed parses it "
        "as an incomplete expression and errors rather than printing anything"
    ),
    area="shell-tool-invocation",
    evidence_quote=(
        "sed -n 900,945 crates/reify-constraints/src/registry.rs ... sed: -e expression #1, "
        "char 7: missing command"
    ),
)

REIFY_7896 = _cluster(  # cand-20260823-13
    title="Redundant duplicate full-suite test invocation to extract two derived metrics",
    cause=(
        "Agent chained two separate `timeout 2400 cargo test -p reify-eval ...` pipelines in one "
        "Bash call — one piped through grep+awk+paste to sum per-binary pass counts, a second "
        "piped through grep -c to count 'test result' lines — instead of capturing the test "
        "output once and deriving both numbers from the same capture. This doubles wall-clock "
        "cost and doubles exposure to being killed (exit 137) before either pipeline finishes."
    ),
    area="test-execution",
    evidence_quote=(
        'timeout 2400 cargo test -p reify-eval 2>&1 | grep -E "^test result" | awk '
        "-F'[.;]' '{print $2}' | paste -sd+ | head -3; timeout 2400 cargo test -p reify-eval "
        '2>&1 | grep -c "^test resu'
    ),
)

REIFY_7898 = _cluster(  # cand-20260805-1
    title=(
        "Read tool's file-size cap on large generated source files forces fallback to "
        "offset/limit or search"
    ),
    cause=(
        "Agent attempted a full-file Read on a 345.5KB source file (diagnostics.rs) and hit the "
        "tool's 256KB hard cap, which surfaces a clear remediation message (use offset/limit or "
        "search) but the session shows no self-correction signal and multiple subsequent 'not "
        "found' misses, suggesting the fallback search strategy struggled to locate the "
        "intended content without ever reading the file directly"
    ),
    area="tool-usage",
    evidence_quote=(
        'Read({"file_path": ".../reify-core/src/diagnostics.rs"}) -> File content (345.5KB) '
        "exceeds maximum allowed size (256KB). Use offset and limit parameters to read specific "
        "portions of the file, or search for specific content instead of reading the whole file."
    ),
)

REIFY_7923 = _cluster(  # cand-20260813-12
    title="Meta-tests assert on source-code comment/prose text instead of constructed runtime behavior",
    cause=(
        "Test author (agent) implemented guard tests by grepping the target script's own "
        "comments or the test file's own prose for a banned/required substring, rather than "
        "asserting on the actual constructed argv/behavior — even to the point of splitting "
        'string literals (e.g. `chang""ed=`) purely to avoid the check matching its own source. '
        "This pins cosmetic wording (comment rewording breaks the gate; a flag leaking into a "
        "non-comment string still passes) instead of the real guarantee, which was already "
        "covered behaviorally by adjacent argv-assertions."
    ),
    area="test-infra",
    evidence_quote=(
        "It pins prose, not behaviour: a maintainer who reworded or trimmed the header comment "
        "while never constructing the flag would fail the gate, and conversely a maintainer who "
        "wrote `# --paths-from` in a comment while the flag leaked into a non-comment "
        "heredoc/string could still satisfy the shape... the check has to assemble its own "
        'patterns from split fragments (`UPSTREAM_TOKEN_EQ="chang""ed="`) purely so it does '
        "not flag its own source — a self-exemption tell"
    ),
)

REIFY_7930 = _cluster(  # cand-20260826-33
    title="Reify .ri module declaration must exactly match file basename or eval rejects it",
    cause=(
        "Agents authoring .ri probe/fixture files assume `module <name>` is a free-form label, "
        "but eval derives the expected module path from the file's location and hard-fails "
        "with E_MODULE_PATH_MISMATCH on any mismatch, burning a wasted eval cycle before the "
        "convention is learned; the probing agent flagged it as a 'gotcha' to warn future runs."
    ),
    area="reify-dsl-authoring",
    evidence_quote=(
        "One gotcha up front: `module <name>` MUST match the file basename or eval fails with "
        "`error: E_MODULE_PATH_MISMATCH: declared module path 'probe01' does not match expected "
        "path 'probe_01_wall_named_cut' (derived from file location)`."
    ),
)

MISFILED_INTO_REIFY = [
    pytest.param(REIFY_7894, id="reify-7894"),
    pytest.param(REIFY_7895, id="reify-7895"),
    pytest.param(REIFY_7900, id="reify-7900"),
    pytest.param(REIFY_7901_RETRY, id="reify-7901-retry"),
    pytest.param(REIFY_7901_SESSION_END, id="reify-7901-session-end"),
]

REIFY_ACTIONABLE = [
    pytest.param(REIFY_7909, id="reify-7909"),
    pytest.param(REIFY_7896, id="reify-7896"),
    pytest.param(REIFY_7898, id="reify-7898"),
    pytest.param(REIFY_7923, id="reify-7923"),
    pytest.param(REIFY_7930, id="reify-7930"),
]


def _components(cluster, *, harness_root=HARNESS_ROOT):
    return [m.component for m in filing_policy.harness_fix_surface(cluster, harness_root=harness_root)]


def _synthetic(**fields):
    return {"title": "t", "summary": None, "cause": "c", "area": "a", "evidence": [], **fields}


# ---------------------------------------------------------------------------
# harness_fix_surface — the regression fixture
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cluster", MISFILED_INTO_REIFY)
def test_misfiled_reify_clusters_carry_a_harness_fix_surface(cluster):
    assert len(filing_policy.harness_fix_surface(cluster, harness_root=HARNESS_ROOT)) >= 1


@pytest.mark.parametrize("cluster", REIFY_ACTIONABLE)
def test_reify_actionable_clusters_carry_no_harness_fix_surface(cluster):
    assert filing_policy.harness_fix_surface(cluster, harness_root=HARNESS_ROOT) == ()


def test_reconciliation_pseudo_tool_error_is_the_only_signal_for_7900():
    matches = filing_policy.harness_fix_surface(REIFY_7900, harness_root=HARNESS_ROOT)
    assert matches == (
        filing_policy.FixSurfaceMatch(
            component="reconciliation-verifier-pseudo-tool",
            evidence="No such tool available: read_file",
        ),
    )


# ---------------------------------------------------------------------------
# harness_fix_surface — precision edge cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "own_file",
    [
        "fused-memory-config.yaml",
        "crates/reify-audit/src/fused_memory_client.rs",
        "crates/reify-audit/src/fused_memory.rs",
        "fused_memory.rs",
        "tools/fused-memory/bridge.py",
    ],
)
def test_hosted_projects_own_fused_memory_named_files_do_not_match(own_file):
    cluster = _synthetic(cause=f"Agent edited {own_file} and the build broke")
    assert _components(cluster) == []


@pytest.mark.parametrize(
    "harness_text",
    [
        pytest.param("fused-memory/config/config.yaml sets a 30s MCP timeout", id="path-head"),
        pytest.param(  # reify cand-20260926-2
            "handling that only exists in dark-factory/fused-memory server code", id="path-tail",
        ),
        pytest.param("mcp-tool-call-serialization/fused-memory", id="area-tail"),  # cand-20260827-18
    ],
)
def test_a_path_naming_the_harness_component_still_matches(harness_text):
    assert _components(_synthetic(cause=harness_text)) == ["fused-memory"]


def test_bare_dark_factory_and_orchestrator_words_do_not_match():
    cluster = _synthetic(cause="The dark-factory orchestrator restarted the job mid-run")
    assert _components(cluster) == []


def test_fused_memory_tool_call_only_in_evidence_does_not_match():
    cluster = _synthetic(
        evidence=['mcp__fused-memory__get_task({"id": "12"}) -> {"status": "done"}'],
    )
    assert _components(cluster) == []


def test_pseudo_tool_name_without_the_harness_error_does_not_match():
    cluster = _synthetic(
        title="Agent calls read_file on a missing path",
        cause="read_file returned an empty buffer for a path that was never written",
    )
    assert _components(cluster) == []


def test_absolute_path_under_the_harness_root_matches_only_that_root():
    cluster = _synthetic(
        cause="Crash in /home/leo/src/dark-factory/orchestrator/src/orchestrator/scheduler.py",
    )
    assert "dark-factory-checkout-path" in _components(cluster)
    assert "dark-factory-checkout-path" not in _components(cluster, harness_root="/srv/other-harness")


def test_escalation_mcp_tool_in_the_title_matches():
    cluster = _synthetic(title="mcp__escalation__merge_request rejects a rebased branch")
    assert _components(cluster) == ["escalation-mcp-tool"]


@pytest.mark.parametrize(
    "cluster",
    [
        pytest.param({}, id="empty"),
        pytest.param(
            {"title": None, "summary": None, "cause": None, "area": None, "evidence": None},
            id="all-none",
        ),
        pytest.param({"title": "t", "evidence": [None]}, id="none-evidence-item"),
    ],
)
def test_missing_or_none_fields_do_not_raise(cluster):
    assert filing_policy.harness_fix_surface(cluster, harness_root=HARNESS_ROOT) == ()


# ---------------------------------------------------------------------------
# resolve_target — where a verified cluster files
# ---------------------------------------------------------------------------

OBSERVED = filing_policy.ProjectRef("/home/leo/src/reify", "reify")
HARNESS = filing_policy.ProjectRef(HARNESS_ROOT, "dark_factory")
IN_TREE_REMEDIATION = {"path": "docs/x.md", "change": "Add a note on the timeout"}


def _resolve(cluster, *, harness: filing_policy.ProjectRef | None = HARNESS):
    return filing_policy.resolve_target(cluster, observed=OBSERVED, harness=harness)


def test_harness_marked_cluster_files_into_the_harness():
    target = _resolve(REIFY_7895)
    assert target.project == HARNESS
    assert target.fix_surface != ()


def test_marker_less_cluster_stays_observed():
    assert _resolve(REIFY_7909) == filing_policy.FilingTarget(project=OBSERVED, fix_surface=())


def test_no_harness_keeps_every_cluster_observed():
    assert _resolve(REIFY_7895, harness=None).project == OBSERVED


def test_harness_censusing_itself_is_a_no_op():
    same_project = filing_policy.ProjectRef(HARNESS_ROOT, "reify")
    assert _resolve(REIFY_7895, harness=same_project) == filing_policy.FilingTarget(
        project=OBSERVED, fix_surface=(),
    )


def test_in_tree_remediation_keeps_a_harness_marked_cluster_observed():
    cluster = {**REIFY_7895, "remediation": IN_TREE_REMEDIATION}
    assert _resolve(cluster).project == OBSERVED


def test_complete_override_pair_wins_over_markers_and_remediation():
    cluster = {
        **REIFY_7895,
        "remediation": IN_TREE_REMEDIATION,
        "target_project_root": "/srv/elsewhere",
        "target_project_id": "elsewhere",
    }
    assert _resolve(cluster).project == filing_policy.ProjectRef("/srv/elsewhere", "elsewhere")


@pytest.mark.parametrize("partial_key", ["target_project_root", "target_project_id"])
@pytest.mark.parametrize(
    ("cluster", "expected"),
    [pytest.param(REIFY_7895, HARNESS, id="marked"), pytest.param(REIFY_7909, OBSERVED, id="unmarked")],
)
def test_partial_override_is_warned_and_ignored(caplog, partial_key, cluster, expected):
    with caplog.at_level("WARNING", logger="legibility.filing_policy"):
        target = _resolve({**cluster, partial_key: "/srv/elsewhere"})
    assert target.project == expected
    assert any(record.levelname == "WARNING" for record in caplog.records)


@pytest.mark.parametrize("cluster", MISFILED_INTO_REIFY)
def test_misfiled_reify_clusters_resolve_to_the_harness(cluster):
    assert _resolve(cluster).project == HARNESS


@pytest.mark.parametrize("cluster", REIFY_ACTIONABLE)
def test_reify_actionable_clusters_resolve_to_the_observed_project(cluster):
    assert _resolve(cluster).project == OBSERVED


# ---------------------------------------------------------------------------
# is_fileable — the singleton filing gate
# ---------------------------------------------------------------------------

def test_unremediated_singleton_is_not_fileable():
    assert filing_policy.is_fileable(REIFY_7909, sighting_count=1) is False


def test_unremediated_recurrence_is_fileable():
    assert filing_policy.is_fileable(REIFY_7909, sighting_count=2) is True


def test_remediated_singleton_is_fileable():
    cluster = {**REIFY_7909, "remediation": IN_TREE_REMEDIATION}
    assert filing_policy.is_fileable(cluster, sighting_count=1) is True


@pytest.mark.parametrize(
    "remediation",
    [
        pytest.param("docs/x.md: add a note", id="non-dict"),
        pytest.param({"path": "docs/x.md"}, id="missing-change"),
        pytest.param({"path": "", "change": "Add a note"}, id="empty-path"),
        pytest.param({"path": 7, "change": "Add a note"}, id="non-str-path"),
    ],
)
def test_malformed_remediation_counts_as_absent(remediation):
    cluster = {**REIFY_7909, "remediation": remediation}
    assert filing_policy.is_fileable(cluster, sighting_count=1) is False


def test_zero_sightings_without_remediation_is_not_fileable():
    assert filing_policy.is_fileable(REIFY_7909, sighting_count=0) is False
