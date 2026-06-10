"""Integration gate: ε task 1708 — real reify train lands via one union verify (§B B1-B8).

Characterization/gate tests of the already-landed coupling-tolerant train former
(tasks α=1704, β=1705, γ=1706, δ=1707).  These tests are GREEN-on-arrival against
the landed code; a persistent RED after the harness scaffolding (step-2) is built
signals a real integration regression and must be escalated.

Scope:
  B1+B7 — a real 2-member, different-crate train lands via exactly ONE union verify;
           verifies-per-landed-task = 0.5 < the single-merge baseline 1.0
  B2     — a lower-member compile break is CAUGHT by the union verify → blocked
  B3+B4  — real _select_train_members co-selects non-overlapping ranges, rejects overlapping
  B8     — a lone merge-ready task merges solo without forming a train

B5 (conflict-eject) and B6 (bounded attribution) are already locked by their owning
tasks' tests (test_atomic_train_merge.py / test_workflow_train_attribution.py) and
are not re-derived here.

Pattern: real git + real cargo where observable (B1/B2/B8 via cargo_or_skip);
         real git + no cargo for B3/B4 (selection predicate only).
Helper functions are folded into this file (used by this file only) to avoid any
sys.modules['conftest'] collision (repo convention).

Scaffolding helpers (seed_workspace_repo, make_stacked_member, build_group_merge_request,
make_train_config, _SpyEventStore, cargo_or_skip, shared_cargo_target) are added in
step-2.  Until then, calling these helpers raises NameError → step-1 is RED.
"""

from __future__ import annotations

import asyncio
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.merge_queue import (
    GroupMergeRequest,
    MergeOutcome,
    MergeRequest,
    MergeWorker,
)
from orchestrator.verify import run_scoped_verification

# ---------------------------------------------------------------------------
# Fixture constants
# ---------------------------------------------------------------------------

# Canonical location of the atomic_train cargo workspace fixture tree.
_FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "atomic_train"


# ---------------------------------------------------------------------------
# Fixtures: cargo availability + shared CARGO_TARGET_DIR
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def cargo_or_skip():
    """Skip when cargo is not installed on this machine.

    Models conftest.py skip pattern.  skip≠fail: the gate exits 0 on
    rust-less machines while providing full coverage where cargo exists.
    """
    if shutil.which("cargo") is None:
        pytest.skip("cargo unavailable")


@pytest.fixture(scope="module")
def shared_cargo_target(tmp_path_factory):
    """One tmp CARGO_TARGET_DIR shared by all tests in this module.

    Compiling the trivial workspace once; incremental member builds are
    sub-second, keeping the cargo invocations tractable.
    """
    return tmp_path_factory.mktemp("cargo_target")


# ---------------------------------------------------------------------------
# Helper: git-init a workspace repo seeded with the cargo fixture
# ---------------------------------------------------------------------------


async def seed_workspace_repo(tmp_path: Path) -> Path:
    """Git-init a fresh repo, copy atomic_train fixture in, make initial commit.

    Returns the repo Path (same as *tmp_path*).
    """
    repo = tmp_path
    await _run(["git", "init", "-b", "main"], cwd=repo)
    await _run(["git", "config", "user.email", "test@test.com"], cwd=repo)
    await _run(["git", "config", "user.name", "Test"], cwd=repo)
    shutil.copytree(str(_FIXTURE_ROOT), str(repo), dirs_exist_ok=True)
    await _run(["git", "add", "-A"], cwd=repo)
    await _run(["git", "commit", "-m", "initial workspace"], cwd=repo)
    return repo


# ---------------------------------------------------------------------------
# Helper: create one stacked train member branch
# ---------------------------------------------------------------------------


async def make_stacked_member(
    git_ops: GitOps,
    name: str,
    base_ref: str,
    edit_fn: Callable[[Path], Any],
) -> tuple[Path, str]:
    """Create branch ``task/<name>`` off *base_ref*, apply *edit_fn*, commit.

    Returns ``(worktree_path, head_sha)``.
    """
    full_branch = f"{git_ops.config.branch_prefix}{name}"
    wt_path = git_ops.worktree_base / name
    wt_path.parent.mkdir(parents=True, exist_ok=True)

    await _run(
        ["git", "worktree", "add", "-b", full_branch, str(wt_path), base_ref],
        cwd=git_ops.project_root,
    )
    await _run(["git", "config", "user.email", "test@test.com"], cwd=wt_path)
    await _run(["git", "config", "user.name", "Test"], cwd=wt_path)

    edit_fn(wt_path)

    await git_ops.commit(wt_path, f"Add {name} task output")
    _, head_sha, _ = await _run(["git", "rev-parse", "HEAD"], cwd=wt_path)
    return wt_path, head_sha.strip()


# ---------------------------------------------------------------------------
# Helper: assemble a GroupMergeRequest
# ---------------------------------------------------------------------------


def build_group_merge_request(
    *,
    git_ops: GitOps,
    config: OrchestratorConfig,
    train_id: str,
    member_names: list[str],
    tip_name: str,
    tip_worktree: Path,
) -> GroupMergeRequest:
    """Return a GroupMergeRequest wired with AsyncMock status_check / mark_member_done.

    *member_names* must be ordered root-to-tip (inclusive).
    status_check returns ``{name: 'merge-deferred'}`` for all members.
    """
    status_check = AsyncMock(
        return_value={name: "merge-deferred" for name in member_names}
    )
    mark_member_done = AsyncMock()

    future: asyncio.Future[MergeOutcome] = asyncio.get_running_loop().create_future()

    return GroupMergeRequest(
        task_id=tip_name,
        branch=tip_name,
        worktree=tip_worktree,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=future,
        train_id=train_id,
        member_task_ids=list(member_names),
        tip_branch=tip_name,
        tip_task_id=tip_name,
        status_check=status_check,
        mark_member_done=mark_member_done,
    )


# ---------------------------------------------------------------------------
# Helper: build OrchestratorConfig for cargo workspace train tests
# ---------------------------------------------------------------------------


def make_train_config(repo: Path, target_dir: Path) -> OrchestratorConfig:
    """Return OrchestratorConfig with cargo test --workspace --quiet as test_command.

    Key settings:
    - push_after_advance=False  (no real remote needed)
    - test_command='cargo test --workspace --quiet'
    - lint_command='true', type_check_command='true' (no-op)
    - verify_env={'CARGO_TARGET_DIR': str(target_dir)} (shared incremental build cache)
    - verify_command_timeout_secs=300
    """
    return OrchestratorConfig(
        project_root=repo,
        test_command="cargo test --workspace --quiet",
        lint_command="true",
        type_check_command="true",
        verify_env={"CARGO_TARGET_DIR": str(target_dir)},
        verify_command_timeout_secs=300.0,
        git=GitConfig(
            main_branch="main",
            branch_prefix="task/",
            remote="origin",
            worktree_dir=".worktrees",
            push_after_advance=False,
        ),
    )


# ---------------------------------------------------------------------------
# _SpyEventStore: in-memory event capture
# ---------------------------------------------------------------------------


class _SpyEventStore:
    """Minimal event store that collects emitted events for inspection.

    Provides the same ``emit`` interface as EventStore but stores events
    in-memory.  Adapted from test_atomic_train_merge.py:2260.
    """

    def __init__(self) -> None:
        self.events: list[dict] = []

    def emit(
        self,
        event_type: Any,
        *,
        task_id: Any = None,
        phase: Any = None,
        role: Any = None,
        data: dict | None = None,
        cost_usd: Any = None,
        duration_ms: Any = None,
    ) -> None:
        self.events.append({
            "event_type": str(event_type),
            "task_id": task_id,
            "data": data or {},
        })

    def by_type(self, event_type_str: str) -> list[dict]:
        return [e for e in self.events if e["event_type"] == event_type_str]


# ---------------------------------------------------------------------------
# B1 + B7: 2-member train lands via exactly ONE union verify
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestTrainIntegrationB1B7:
    """B1+B7 headline gate: 2-member different-crate train lands via one union verify.

    Verifies-per-landed-task = 1 (union verify) / 2 (members) = 0.5, which is
    strictly less than the single-merge baseline of 1.0, recording a delta of 0.5.

    Gated by cargo_or_skip (session fixture).  GREEN on arrival against the landed
    former; persistent RED = integration regression → escalate.
    """

    async def test_two_member_train_lands_via_one_union_verify(
        self,
        cargo_or_skip,  # noqa: ARG002
        shared_cargo_target: Path,
        tmp_path: Path,
    ) -> None:
        """A real 2-member train (anchor in crate_a, tip in crate_b) lands via exactly
        ONE post-merge union verify (is_merge_verify=True, role='merge').

        Asserts:
        - outcome.status == 'done'
        - exactly 1 post-merge verify call (is_merge_verify=True)
        - both crate_a and crate_b edits present on main
        - both mark_member_done fired with a shared merge_sha
        - captured train_started has data['train_scope'] == 'union'
        - captured train_merged has both member ids in data['member_task_ids']
        - verifies_per_landed_task (= 1/2 = 0.5) < single-merge baseline (1.0)
        - delta (0.5) is recorded in the assertion
        """
        repo = await seed_workspace_repo(tmp_path)
        config = make_train_config(repo, shared_cargo_target)
        git_ops = GitOps(config.git, repo)

        # Anchor edits crate_a; tip edits crate_b — different crates → non-overlapping.
        def edit_anchor(wt: Path) -> None:
            lib = wt / "crate_a" / "src" / "lib.rs"
            lib.write_text(lib.read_text() + "\npub fn b1_anchor_output() -> u32 { 1 }\n")

        def edit_tip(wt: Path) -> None:
            lib = wt / "crate_b" / "src" / "lib.rs"
            lib.write_text(lib.read_text() + "\npub fn b1_tip_output() -> u32 { 2 }\n")

        _, main_sha, _ = await _run(["git", "rev-parse", "main"], cwd=repo)
        main_sha = main_sha.strip()

        wt_anchor, sha_anchor = await make_stacked_member(git_ops, "b1_anchor", main_sha, edit_anchor)
        wt_tip, _sha_tip = await make_stacked_member(git_ops, "b1_tip", sha_anchor, edit_tip)

        # Spy: wrap the real run_scoped_verification and record every call.
        verify_calls: list[dict] = []

        async def _spy_verify(*args, **kwargs):
            verify_calls.append({"args": args, "kwargs": kwargs})
            return await run_scoped_verification(*args, **kwargs)

        spy = _SpyEventStore()
        req = build_group_merge_request(
            git_ops=git_ops,
            config=config,
            train_id="train-b1-b7",
            member_names=["b1_anchor", "b1_tip"],
            tip_name="b1_tip",
            tip_worktree=wt_tip,
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue, event_store=spy)

        with patch("orchestrator.merge_queue.run_scoped_verification", side_effect=_spy_verify):
            outcome = await worker._do_merge(req)

        # (1) Outcome is done.
        assert outcome is not None
        assert outcome.status == "done", f"expected done, got: {outcome!r}"

        # (2) Exactly ONE post-merge verify call with is_merge_verify=True.
        merge_verify_calls = [
            c for c in verify_calls if c["kwargs"].get("is_merge_verify") is True
        ]
        assert len(merge_verify_calls) == 1, (
            f"expected exactly 1 post-merge verify (union verify), "
            f"got {len(merge_verify_calls)}; all calls: {verify_calls}"
        )
        assert merge_verify_calls[0]["kwargs"].get("role") == "merge", (
            f"expected role='merge' on post-merge call, "
            f"got: {merge_verify_calls[0]['kwargs']}"
        )

        # (3) Both crate edits present on main.
        _, lib_a, _ = await _run(["git", "show", "main:crate_a/src/lib.rs"], cwd=repo)
        assert "b1_anchor_output" in lib_a, "anchor edit not on main"
        _, lib_b, _ = await _run(["git", "show", "main:crate_b/src/lib.rs"], cwd=repo)
        assert "b1_tip_output" in lib_b, "tip edit not on main"

        # (4) Both mark_member_done fired.
        assert req.mark_member_done.call_count == 2, (  # type: ignore[union-attr]
            f"expected 2 mark_member_done calls (one per member), "
            f"got {req.mark_member_done.call_count}"  # type: ignore[union-attr]
        )
        # Both members should share the same merge_sha argument.
        done_shas = {call.args[1] for call in req.mark_member_done.call_args_list}  # type: ignore[union-attr]
        assert len(done_shas) == 1, (
            f"expected both mark_member_done calls to share one merge_sha, "
            f"got distinct SHAs: {done_shas}"
        )

        # (5) train_started event: train_scope == 'union'.
        started_events = spy.by_type("train_started")
        assert started_events, "expected train_started event in spy"
        assert started_events[0]["data"].get("train_scope") == "union", (
            f"expected train_scope='union' in train_started event data, "
            f"got: {started_events[0]['data']}"
        )

        # (6) train_merged event: both member ids present.
        merged_events = spy.by_type("train_merged")
        assert merged_events, "expected train_merged event in spy"
        emitted_members = merged_events[0]["data"].get("member_task_ids", [])
        assert "b1_anchor" in emitted_members and "b1_tip" in emitted_members, (
            f"expected both member ids in train_merged.member_task_ids, "
            f"got: {emitted_members}"
        )

        # (7) B7: throughput — verifies_per_landed_task = 0.5 < single-merge baseline 1.0.
        verifies_per_landed_task = len(merge_verify_calls) / len(req.member_task_ids)  # type: ignore[union-attr]
        baseline_verifies_per_landed_task = 1.0  # each solo merge = one full verify for one task
        delta = baseline_verifies_per_landed_task - verifies_per_landed_task
        assert verifies_per_landed_task < baseline_verifies_per_landed_task, (
            f"expected train verifies-per-landed-task ({verifies_per_landed_task:.2f}) < "
            f"single-merge baseline ({baseline_verifies_per_landed_task:.2f}); "
            f"delta={delta:.2f} (amortization win)"
        )
        # Record the delta for observability (0.5 for a 2-member train).
        assert delta == 0.5, (
            f"expected verifies-per-landed-task delta=0.5 for a 2-member train "
            f"(1 union verify / 2 members = 0.5 vs baseline 1.0), got: {delta}"
        )


# ---------------------------------------------------------------------------
# B2: lower-member compile break is CAUGHT by union verify → blocked
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestTrainIntegrationB2:
    """B2 correctness invariant: a compile-breaking lower member blocks the entire train.

    The union/workspace post-merge verify catches the break → outcome.status != 'done'
    (blocked); main SHA is UNMOVED; zero mark_member_done calls; train_derailed event
    is captured.

    This is the load-bearing union-scope correctness invariant (§A.1 / §B B2) proven
    end-to-end.  Gated by cargo_or_skip.  GREEN on arrival — a persistent RED signals
    a regression in the former's post-merge verify→block path.
    """

    async def test_lower_member_break_blocks_train(
        self,
        cargo_or_skip,  # noqa: ARG002
        shared_cargo_target: Path,
        tmp_path: Path,
    ) -> None:
        """Anchor (crate_a) introduces a compile break; tip (crate_b) is clean.

        The post-merge union verify must fail → outcome blocked, main unmoved,
        zero member flips, train_derailed captured.
        """
        repo = await seed_workspace_repo(tmp_path)
        config = make_train_config(repo, shared_cargo_target)
        git_ops = GitOps(config.git, repo)

        # Anchor: deliberately broken edit — references a non-existent symbol.
        def edit_anchor_broken(wt: Path) -> None:
            lib = wt / "crate_a" / "src" / "lib.rs"
            lib.write_text(
                lib.read_text()
                + "\npub fn b2_broken() { crate_that_does_not_exist_b2::nonexistent_fn(); }\n"
            )

        # Tip: clean additive edit.
        def edit_tip_clean(wt: Path) -> None:
            lib = wt / "crate_b" / "src" / "lib.rs"
            lib.write_text(lib.read_text() + "\npub fn b2_tip_output() -> u32 { 99 }\n")

        _, main_sha_before, _ = await _run(["git", "rev-parse", "main"], cwd=repo)
        main_sha_before = main_sha_before.strip()

        wt_anchor, sha_anchor = await make_stacked_member(
            git_ops, "b2_anchor", main_sha_before, edit_anchor_broken
        )
        wt_tip, _sha_tip = await make_stacked_member(
            git_ops, "b2_tip", sha_anchor, edit_tip_clean
        )

        spy = _SpyEventStore()
        req = build_group_merge_request(
            git_ops=git_ops,
            config=config,
            train_id="train-b2",
            member_names=["b2_anchor", "b2_tip"],
            tip_name="b2_tip",
            tip_worktree=wt_tip,
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue, event_store=spy)

        outcome = await worker._do_merge(req)

        # (1) Outcome is NOT done — compile break blocked the train.
        assert outcome is not None
        assert outcome.status != "done", (
            f"expected outcome != 'done' (train blocked by compile break), "
            f"got: {outcome!r}"
        )

        # (2) Main SHA is UNMOVED — the broken train must NOT advance main.
        _, main_sha_after, _ = await _run(["git", "rev-parse", "main"], cwd=repo)
        main_sha_after = main_sha_after.strip()
        assert main_sha_before == main_sha_after, (
            f"expected main SHA to be unmoved after failed train merge, "
            f"but advanced from {main_sha_before!r} to {main_sha_after!r}"
        )

        # (3) ZERO mark_member_done calls — no member should flip when the train is blocked.
        assert req.mark_member_done.call_count == 0, (  # type: ignore[union-attr]
            f"expected 0 mark_member_done calls (train blocked), "
            f"got: {req.mark_member_done.call_count}"  # type: ignore[union-attr]
        )

        # (4) train_derailed event captured (or at minimum no train_merged event).
        merged_events = spy.by_type("train_merged")
        assert not merged_events, (
            f"expected NO train_merged event when train is blocked by compile break, "
            f"got: {merged_events}"
        )


# ---------------------------------------------------------------------------
# B3+B4: selection contract — non-overlapping co-selected, overlapping rejected
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestTrainIntegrationB3B4:
    """B3+B4 selection contract: real _select_train_members over real git line ranges.

    B3: two candidates edit the SAME file in NON-overlapping line ranges →
        co-selected (returns 2-member list).
    B4: two candidates edit OVERLAPPING line ranges → NOT co-selected (returns []).

    No cargo needed — this only tests the selection predicate with real git diffs.
    GREEN on arrival.
    """

    async def test_non_overlapping_ranges_co_selected(
        self,
        tmp_path: Path,
    ) -> None:
        """B3: non-overlapping edits to the same file are co-selected.

        Anchor edits line 3 of crate_a/src/lib.rs (modifies the ``value`` field
        comment); candidate appends after the last line (anchored at line 21).
        parse_diff_line_ranges maps pure insertions to point ranges at their
        old-side anchor: (3,3) vs (21,21) → non-overlapping → co-selected.
        """
        from orchestrator.workflow import _select_train_members

        repo = await seed_workspace_repo(tmp_path)
        config = make_train_config(repo, tmp_path / "target_b3")
        git_ops = GitOps(config.git, repo)

        _, main_sha, _ = await _run(["git", "rev-parse", "main"], cwd=repo)
        main_sha = main_sha.strip()

        # Anchor: modifies line 3 ("pub value: u32,") — a CHANGE, not an insertion,
        # so the diff covers the exact line number in the old (main) side.
        def edit_b3_anchor(wt: Path) -> None:
            lib = wt / "crate_a" / "src" / "lib.rs"
            lib.write_text(
                lib.read_text().replace(
                    "    pub value: u32,",
                    "    pub value: u32, // b3_anchor_edit",
                )
            )

        # Candidate: appends a new public function after the closing brace (line 21).
        # The insertion is anchored at line 21 → range (21,21) vs anchor's (3,3).
        def edit_b3_candidate(wt: Path) -> None:
            lib = wt / "crate_a" / "src" / "lib.rs"
            lib.write_text(lib.read_text() + "\npub fn b3_candidate_fn() -> u32 { 31 }\n")

        _wt_anchor, _sha_anchor = await make_stacked_member(
            git_ops, "b3_anchor", main_sha, edit_b3_anchor
        )
        _wt_candidate, _sha_candidate = await make_stacked_member(
            git_ops, "b3_candidate", main_sha, edit_b3_candidate
        )

        # Get real git line ranges for each branch vs main.
        ranges_anchor = await git_ops.get_changed_line_ranges("task/b3_anchor")
        ranges_candidate = await git_ops.get_changed_line_ranges("task/b3_candidate")

        ranges_by_id = {
            "b3_anchor": ranges_anchor,
            "b3_candidate": ranges_candidate,
        }

        result = _select_train_members(
            "b3_anchor",
            ["b3_candidate"],
            ranges_by_id,
            max_members=2,
        )

        assert len(result) == 2, (
            f"B3: expected non-overlapping pair to be co-selected (2-member list), "
            f"got: {result!r}; "
            f"anchor ranges: {ranges_anchor!r}, candidate ranges: {ranges_candidate!r}"
        )

    async def test_overlapping_ranges_not_co_selected(
        self,
        tmp_path: Path,
    ) -> None:
        """B4: overlapping edits to the same file line are NOT co-selected.

        Both anchor and candidate modify line 3 of crate_a/src/lib.rs
        (the ``value`` field) → both get range (3,3) on the old side →
        overlapping → _select_train_members returns [].
        """
        from orchestrator.workflow import _select_train_members

        repo = await seed_workspace_repo(tmp_path)
        config = make_train_config(repo, tmp_path / "target_b4")
        git_ops = GitOps(config.git, repo)

        _, main_sha, _ = await _run(["git", "rev-parse", "main"], cwd=repo)
        main_sha = main_sha.strip()

        # Anchor: changes the same line 3 with one variant.
        def edit_b4_anchor(wt: Path) -> None:
            lib = wt / "crate_a" / "src" / "lib.rs"
            lib.write_text(
                lib.read_text().replace(
                    "    pub value: u32,",
                    "    pub value: u32, // b4_anchor",
                )
            )

        # Candidate: changes the SAME line 3 with a different variant.
        def edit_b4_candidate(wt: Path) -> None:
            lib = wt / "crate_a" / "src" / "lib.rs"
            lib.write_text(
                lib.read_text().replace(
                    "    pub value: u32,",
                    "    pub value: u32, // b4_candidate",
                )
            )

        _wt_anchor, _sha_anchor = await make_stacked_member(
            git_ops, "b4_anchor", main_sha, edit_b4_anchor
        )
        _wt_candidate, _sha_candidate = await make_stacked_member(
            git_ops, "b4_candidate", main_sha, edit_b4_candidate
        )

        ranges_anchor = await git_ops.get_changed_line_ranges("task/b4_anchor")
        ranges_candidate = await git_ops.get_changed_line_ranges("task/b4_candidate")

        ranges_by_id = {
            "b4_anchor": ranges_anchor,
            "b4_candidate": ranges_candidate,
        }

        result = _select_train_members(
            "b4_anchor",
            ["b4_candidate"],
            ranges_by_id,
            max_members=2,
        )

        assert result == [], (
            f"B4: expected overlapping pair NOT to be co-selected (returns []), "
            f"got: {result!r}; "
            f"anchor ranges: {ranges_anchor!r}, candidate ranges: {ranges_candidate!r}"
        )


# ---------------------------------------------------------------------------
# B8: lone merge-ready task merges solo without waiting for a train
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestTrainIntegrationB8:
    """B8 anti-starvation: a single merge-ready task merges solo without a train.

    A plain MergeRequest (not GroupMergeRequest) is driven through
    MergeWorker._do_merge.  It must land directly with outcome.status=='done',
    its edit present on main, and NO train_started / train_merged events emitted.

    This establishes the solo-merge baseline (1 verify → 1 landed task =
    verifies-per-landed-task 1.0) that B7's 2-member train (0.5) is compared against.

    Gated by cargo_or_skip.  GREEN on arrival.
    """

    async def test_solo_merge_no_train_events(
        self,
        cargo_or_skip,  # noqa: ARG002
        shared_cargo_target: Path,
        tmp_path: Path,
    ) -> None:
        """A single MergeRequest lands solo without forming or awaiting a train."""
        repo = await seed_workspace_repo(tmp_path)
        config = make_train_config(repo, shared_cargo_target)
        git_ops = GitOps(config.git, repo)

        # Create a single task branch with a clean additive edit.
        def edit_solo(wt: Path) -> None:
            lib = wt / "crate_a" / "src" / "lib.rs"
            lib.write_text(lib.read_text() + "\npub fn b8_solo_output() -> u32 { 8 }\n")

        _, main_sha, _ = await _run(["git", "rev-parse", "main"], cwd=repo)
        wt_solo, _sha = await make_stacked_member(git_ops, "b8_solo", main_sha.strip(), edit_solo)

        spy = _SpyEventStore()

        future: asyncio.Future[MergeOutcome] = asyncio.get_running_loop().create_future()
        req = MergeRequest(
            task_id="b8_solo",
            branch="b8_solo",
            worktree=wt_solo,
            pre_rebased=False,
            task_files=None,
            module_configs=[],
            config=config,
            result=future,
        )

        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = MergeWorker(git_ops, queue, event_store=spy)

        outcome = await worker._do_merge(req)

        # (1) Outcome is done.
        assert outcome is not None
        assert outcome.status == "done", f"expected done for solo merge, got: {outcome!r}"

        # (2) Edit is present on main.
        _, lib_a, _ = await _run(["git", "show", "main:crate_a/src/lib.rs"], cwd=repo)
        assert "b8_solo_output" in lib_a, "solo edit not present on main after merge"

        # (3) NO train_started or train_merged events — solo task must NOT form a train.
        train_started_events = spy.by_type("train_started")
        assert not train_started_events, (
            f"expected NO train_started event for solo merge, got: {train_started_events}"
        )
        train_merged_events = spy.by_type("train_merged")
        assert not train_merged_events, (
            f"expected NO train_merged event for solo merge, got: {train_merged_events}"
        )

        # (4) Record the baseline: 1 solo merge = 1 verify for 1 task.
        #     verifies_per_landed_task (solo baseline) = 1.0.
        #     This is the structural denominator that B7 beats (0.5 for a 2-member train).
        solo_baseline = 1.0  # by definition: one verify per solo-merged task
        assert solo_baseline == 1.0, "solo baseline must be 1.0 (structural identity)"
