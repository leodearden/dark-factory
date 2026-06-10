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
from pathlib import Path
from unittest.mock import patch

import pytest

from orchestrator.merge_queue import (
    GroupMergeRequest,
    MergeRequest,
    MergeWorker,
)
from orchestrator.verify import run_scoped_verification


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
        from orchestrator.git_ops import GitOps
        git_ops = GitOps(config.git, repo)

        # Anchor edits crate_a; tip edits crate_b — different crates → non-overlapping.
        def edit_anchor(wt: Path) -> None:
            lib = wt / "crate_a" / "src" / "lib.rs"
            lib.write_text(lib.read_text() + "\npub fn b1_anchor_output() -> u32 { 1 }\n")

        def edit_tip(wt: Path) -> None:
            lib = wt / "crate_b" / "src" / "lib.rs"
            lib.write_text(lib.read_text() + "\npub fn b1_tip_output() -> u32 { 2 }\n")

        from orchestrator.git_ops import _run
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
