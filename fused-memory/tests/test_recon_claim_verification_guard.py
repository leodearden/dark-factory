"""Tests for the recon claim-verification guard (task 2438).

Hardens Stage 2 recon task-filing against fabricated commit/diff/stamp
claims attributed to a completed task/commit/ACTION — mirrors the shape of
tests/test_recon_code_fix_premise_guard.py, but the target here is
retrospective FALSE-FACT claims embedded in a filed task's own description,
not a self-referential recon code-fix premise.

Motivating incident: task 2433 asserted, as verified fact, that "task 2372
added an ACTION #5 stamp (metadata.done_provenance_invalidated=true) on task
reopen." No such stamp/token has ever existed in tree or history — a
fabrication. See recon_claim_verification_guard.py's module docstring.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

# ──────────────────────────────────────────────────────────────────────────────
# task-2438 step-01 RED: TestExtractAttributedClaims
# ──────────────────────────────────────────────────────────────────────────────


class TestExtractAttributedClaims:
    """Tests for extract_attributed_claims() + AttributedClaim from
    fused_memory.middleware.recon_claim_verification_guard.
    """

    def test_incident_sentence_yields_one_claim_with_attribution(self):
        """(a) The verbatim task-2433 sentence yields one AttributedClaim with
        token=='done_provenance_invalidated' and an attribution naming both
        task 2372 and ACTION #5."""
        from fused_memory.middleware.recon_claim_verification_guard import (
            extract_attributed_claims,
        )

        text = (
            "This is the same site that stamps "
            "metadata.done_provenance_invalidated=true per task 2372 ACTION #5 "
            "on task reopen."
        )

        claims = extract_attributed_claims(text)

        assert len(claims) == 1
        claim = claims[0]
        assert claim.token == "done_provenance_invalidated"
        assert "task 2372" in claim.attribution
        assert "ACTION #5" in claim.attribution

    def test_prospective_field_without_anchor_yields_no_claims(self):
        """(b) A prospective feature description that names a
        `metadata.foo_bar` field with NO task/commit/ACTION anchor yields []
        — this is a new-work spec, not a retrospective fact claim."""
        from fused_memory.middleware.recon_claim_verification_guard import (
            extract_attributed_claims,
        )

        text = (
            "We should add a new `metadata.foo_bar` field so operators can "
            "override the default routing behaviour going forward."
        )

        assert extract_attributed_claims(text) == []

    def test_empty_and_plain_prose_yield_no_claims(self):
        """(c) Empty/whitespace/plain-prose text yields []."""
        from fused_memory.middleware.recon_claim_verification_guard import (
            extract_attributed_claims,
        )

        assert extract_attributed_claims("") == []
        assert extract_attributed_claims("   \n\t  ") == []
        assert extract_attributed_claims(
            "The scheduler dispatches tasks once their dependencies are done."
        ) == []

    def test_multiple_distinct_tokens_dedupe_in_first_seen_order(self):
        """(d) Multiple distinct tokens dedupe in first-seen order; a repeated
        token keeps its FIRST occurrence's attribution. The filler paragraphs
        are wide enough that the bounded co-occurrence window cannot bridge
        across them, so this test does not depend on the exact window size."""
        from fused_memory.middleware.recon_claim_verification_guard import (
            extract_attributed_claims,
        )

        filler = (
            "This paragraph exists purely to separate the two mentions of "
            "the same token by a wide margin so the bounded co-occurrence "
            "window used for attribution matching cannot bridge across it, "
            "keeping this test independent of the exact window size chosen. "
        ) * 2

        text = (
            "The guard stamps metadata.alpha_flag=true per task 100 during "
            "the first pass. "
            + filler
            + "metadata.beta_flag=true per task 200 was added afterwards. "
            + filler
            + "Finally metadata.alpha_flag=true per task 300 repeats the "
            "earlier stamp."
        )

        claims = extract_attributed_claims(text)

        assert [c.token for c in claims] == ["alpha_flag", "beta_flag"]
        assert "task 100" in claims[0].attribution
        assert "task 300" not in claims[0].attribution
        assert "task 200" in claims[1].attribution


# ──────────────────────────────────────────────────────────────────────────────
# task-2438 step-03 RED: TestVerifyAttributedClaims / TestUnverifiedClaimsInText
# ──────────────────────────────────────────────────────────────────────────────


class TestVerifyAttributedClaims:
    """Tests for verify_attributed_claims(claims, probe) with an injected
    fake probe (dict/set membership) — no git, no I/O.
    """

    def test_absent_token_is_returned_as_unverified(self):
        """(a) A claim whose token the probe reports absent is returned."""
        from fused_memory.middleware.recon_claim_verification_guard import (
            AttributedClaim,
            verify_attributed_claims,
        )

        claim = AttributedClaim(
            token="done_provenance_invalidated", attribution="task 2372", span="...",
        )
        probe = (lambda present: lambda token: token in present)(set())

        result = verify_attributed_claims([claim], probe)

        assert result == [claim]

    def test_present_token_is_filtered_out(self):
        """(b) A claim whose token the probe reports present is filtered out
        (self-correcting — an existing token like `get_statuses` verifies clean)."""
        from fused_memory.middleware.recon_claim_verification_guard import (
            AttributedClaim,
            verify_attributed_claims,
        )

        claim = AttributedClaim(token="get_statuses", attribution="task 100", span="...")
        probe = (lambda present: lambda token: token in present)({"get_statuses"})

        result = verify_attributed_claims([claim], probe)

        assert result == []

    def test_empty_claims_returns_empty(self):
        """(c) Empty claims => []."""
        from fused_memory.middleware.recon_claim_verification_guard import (
            verify_attributed_claims,
        )

        result = verify_attributed_claims([], lambda token: True)

        assert result == []

    def test_mixed_claims_returns_only_absent_subset(self):
        """Mixed input: only the tokens the probe reports absent survive."""
        from fused_memory.middleware.recon_claim_verification_guard import (
            AttributedClaim,
            verify_attributed_claims,
        )

        present = AttributedClaim(token="get_statuses", attribution="task 1", span="...")
        absent = AttributedClaim(token="totally_fake_stamp", attribution="task 2", span="...")
        probe = (lambda present_set: lambda token: token in present_set)({"get_statuses"})

        result = verify_attributed_claims([present, absent], probe)

        assert result == [absent]


class TestUnverifiedClaimsInText:
    """Tests for unverified_claims_in_text(text, probe) — composes
    extract_attributed_claims + verify_attributed_claims, mirroring
    premise_refuted_entry's composition shape.
    """

    INCIDENT_TEXT = (
        "This is the same site that stamps "
        "metadata.done_provenance_invalidated=true per task 2372 ACTION #5 "
        "on task reopen."
    )

    def test_incident_sentence_unverified_when_probe_lacks_token(self):
        """(d1) A probe lacking the token => one unverified claim."""
        from fused_memory.middleware.recon_claim_verification_guard import (
            unverified_claims_in_text,
        )

        probe = (lambda present: lambda token: token in present)(set())

        result = unverified_claims_in_text(self.INCIDENT_TEXT, probe)

        assert len(result) == 1
        assert result[0].token == "done_provenance_invalidated"

    def test_incident_sentence_verified_when_probe_has_token(self):
        """(d2) A probe that has the token => []."""
        from fused_memory.middleware.recon_claim_verification_guard import (
            unverified_claims_in_text,
        )

        probe = (lambda present: lambda token: token in present)(
            {"done_provenance_invalidated"},
        )

        result = unverified_claims_in_text(self.INCIDENT_TEXT, probe)

        assert result == []

    def test_text_with_no_claims_returns_empty_without_calling_probe(self):
        """No attributed claims in the text => [] and the probe is never invoked."""
        from fused_memory.middleware.recon_claim_verification_guard import (
            unverified_claims_in_text,
        )

        def probe(token: str) -> bool:
            raise AssertionError("probe must not be called when there are no claims")

        result = unverified_claims_in_text("Plain prose with no claims at all.", probe)

        assert result == []


# ──────────────────────────────────────────────────────────────────────────────
# task-2438 step-05 RED: TestMakeSourceAndHistoryProbe
# ──────────────────────────────────────────────────────────────────────────────


def _run_git(args: list[str], cwd) -> None:
    subprocess.run(
        ["git", *args], cwd=str(cwd), check=True, capture_output=True, text=True,
    )


def _require_git() -> None:
    if shutil.which("git") is None:
        pytest.skip("git is not available")


@pytest.fixture
def git_repo_with_history(tmp_path):
    """A real temp git repo (git is available in the worktree;
    orchestrator/tests/test_git_ops.py is the established real-git test
    precedent): commit 1 adds two tracked files, one containing
    KNOWN_LIVE_TOKEN and the other containing REMOVED_HISTORY_TOKEN;
    commit 2 deletes the REMOVED_HISTORY_TOKEN file, so that token survives
    only in history, not in the working tree.
    """
    _require_git()

    repo = tmp_path / "repo"
    repo.mkdir()
    _run_git(["init", "-b", "main"], cwd=repo)
    _run_git(["config", "user.email", "test@test.com"], cwd=repo)
    _run_git(["config", "user.name", "Test"], cwd=repo)

    (repo / "present.py").write_text("KNOWN_LIVE_TOKEN = 1\n", encoding="utf-8")
    (repo / "removed.py").write_text("REMOVED_HISTORY_TOKEN = 2\n", encoding="utf-8")
    _run_git(["add", "-A"], cwd=repo)
    _run_git(["commit", "-m", "add present.py and removed.py"], cwd=repo)

    (repo / "removed.py").unlink()
    _run_git(["add", "-A"], cwd=repo)
    _run_git(["commit", "-m", "delete removed.py"], cwd=repo)

    return repo


class TestMakeSourceAndHistoryProbe:
    """Tests for make_source_and_history_probe(repo_root) — the one impure
    git-backed adapter. Encodes the architect's exact manual refutation
    check for the task-2433 incident: git grep (tree) then, on miss,
    git log --all -S (history pickaxe).
    """

    def test_tree_present_token_returns_true(self, git_repo_with_history):
        """(a) True for a token present in a tracked working-tree file."""
        from fused_memory.middleware.recon_claim_verification_guard import (
            make_source_and_history_probe,
        )

        probe = make_source_and_history_probe(git_repo_with_history)

        assert probe("KNOWN_LIVE_TOKEN") is True

    def test_never_committed_token_returns_false(self, git_repo_with_history):
        """(b) False for a token that was never committed — absent from tree
        AND history."""
        from fused_memory.middleware.recon_claim_verification_guard import (
            make_source_and_history_probe,
        )

        probe = make_source_and_history_probe(git_repo_with_history)

        assert probe("TOTALLY_FABRICATED_TOKEN_XYZ") is False

    def test_removed_from_tree_but_in_history_returns_true(self, git_repo_with_history):
        """(c) True for a token removed from the tree but still reachable via
        `git log --all -S` (history hit) — proves the guard does not
        false-flag a token legitimately removed from the tree."""
        from fused_memory.middleware.recon_claim_verification_guard import (
            make_source_and_history_probe,
        )

        probe = make_source_and_history_probe(git_repo_with_history)

        assert probe("REMOVED_HISTORY_TOKEN") is True

    def test_non_git_repo_fails_open_true(self, tmp_path):
        """(d) fail-open True when repo_root is not a git repo / git errors."""
        _require_git()
        from fused_memory.middleware.recon_claim_verification_guard import (
            make_source_and_history_probe,
        )

        not_a_repo = tmp_path / "not_a_repo"
        not_a_repo.mkdir()

        probe = make_source_and_history_probe(not_a_repo)

        assert probe("ANY_TOKEN_AT_ALL") is True


# ──────────────────────────────────────────────────────────────────────────────
# Amendment: TestMakeSourceAndHistoryProbePathExclusion
#
# Regression tests for the reviewer's efficacy finding: `git grep` (no
# pathspec) searches the WHOLE tracked tree, which now includes this task's
# own artifacts (e.g. `done_provenance_invalidated` used as an example in
# this module's docstring, both test files, stage2.py's prompt guardrail,
# and docs/legibility/confusion-codebook.yaml). Without excluding
# tests/docs/prompts paths, a token that merges into ONLY those paths would
# verify "present" and the guard would never again flag a genuine
# re-fabrication of that identical token — silently defeating the exact
# incident (task 2433) this guard exists to catch.
# ──────────────────────────────────────────────────────────────────────────────


class TestMakeSourceAndHistoryProbePathExclusion:
    """tests/, docs/, and prompts/ paths are excluded from both the tree grep
    and the history pickaxe, so a match confined to them never counts as
    "present"."""

    def test_token_only_in_excluded_paths_returns_false(self, tmp_path):
        """(a) A token that exists ONLY under tests/, docs/, and prompts/
        paths verifies ABSENT."""
        _require_git()
        from fused_memory.middleware.recon_claim_verification_guard import (
            make_source_and_history_probe,
        )

        repo = tmp_path / "repo"
        repo.mkdir()
        _run_git(["init", "-b", "main"], cwd=repo)
        _run_git(["config", "user.email", "test@test.com"], cwd=repo)
        _run_git(["config", "user.name", "Test"], cwd=repo)

        (repo / "tests").mkdir()
        (repo / "docs").mkdir()
        (repo / "prompts").mkdir()
        (repo / "tests" / "foo_test.py").write_text(
            "EXAMPLE_ONLY_TOKEN = 1\n", encoding="utf-8",
        )
        (repo / "docs" / "notes.md").write_text(
            "EXAMPLE_ONLY_TOKEN appears here too\n", encoding="utf-8",
        )
        (repo / "prompts" / "guardrail.py").write_text(
            "# mentions EXAMPLE_ONLY_TOKEN as an example\n", encoding="utf-8",
        )
        _run_git(["add", "-A"], cwd=repo)
        _run_git(["commit", "-m", "add example-only mentions"], cwd=repo)

        probe = make_source_and_history_probe(repo)

        assert probe("EXAMPLE_ONLY_TOKEN") is False

    def test_token_in_real_source_path_still_returns_true(self, tmp_path):
        """(b) Control: a token present in a NON-excluded source path is
        still detected — the exclusion is scoped, not a blanket disable."""
        _require_git()
        from fused_memory.middleware.recon_claim_verification_guard import (
            make_source_and_history_probe,
        )

        repo = tmp_path / "repo"
        repo.mkdir()
        _run_git(["init", "-b", "main"], cwd=repo)
        _run_git(["config", "user.email", "test@test.com"], cwd=repo)
        _run_git(["config", "user.name", "Test"], cwd=repo)

        (repo / "src").mkdir()
        (repo / "src" / "real.py").write_text("REAL_SOURCE_TOKEN = 1\n", encoding="utf-8")
        _run_git(["add", "-A"], cwd=repo)
        _run_git(["commit", "-m", "add real source"], cwd=repo)

        probe = make_source_and_history_probe(repo)

        assert probe("REAL_SOURCE_TOKEN") is True

    def test_incident_token_confined_to_excluded_paths_still_flagged(self, tmp_path):
        """(c) Directly regresses the task-2433 incident scenario the review
        comment identified: even once `done_provenance_invalidated` merges
        into this guard's own tests/docstrings/prompt guardrail (all
        excluded paths), a genuinely fabricated re-use of the identical
        token in a NEW candidate must still be flagged."""
        _require_git()
        from fused_memory.middleware.recon_claim_verification_guard import (
            make_source_and_history_probe,
        )

        repo = tmp_path / "repo"
        repo.mkdir()
        _run_git(["init", "-b", "main"], cwd=repo)
        _run_git(["config", "user.email", "test@test.com"], cwd=repo)
        _run_git(["config", "user.name", "Test"], cwd=repo)

        (repo / "tests").mkdir()
        (repo / "tests" / "test_guard.py").write_text(
            "# example: metadata.done_provenance_invalidated=true\n", encoding="utf-8",
        )
        _run_git(["add", "-A"], cwd=repo)
        _run_git(["commit", "-m", "add example token in tests"], cwd=repo)

        probe = make_source_and_history_probe(repo)

        assert probe("done_provenance_invalidated") is False


# ──────────────────────────────────────────────────────────────────────────────
# Amendment: TestMakeSourceAndHistoryProbeRepoRootResolution
#
# Regression test for the reviewer's correctness/consistency finding: `git
# grep` (no pathspec) scopes to repo_root's subtree while `git log --all`
# scopes to the whole repo regardless of cwd — so a repo_root that is a
# package subdirectory (rather than the repo top level) made the two probes
# search inconsistent scopes.
# ──────────────────────────────────────────────────────────────────────────────


class TestMakeSourceAndHistoryProbeRepoRootResolution:
    """make_source_and_history_probe resolves repo_root to its git top level
    once, up front, so both calls search the same scope regardless of which
    subdirectory repo_root points at."""

    def test_staged_uncommitted_token_outside_subdir_root_is_still_found(self, tmp_path):
        """A token that is STAGED (tracked via `git add`, so it is visible to
        `git grep`'s default working-tree search) but NOT YET committed (so
        the history pickaxe alone cannot rescue it), and lives OUTSIDE the
        subdirectory passed as repo_root, must still be found once grep is
        rooted at the resolved repo top level rather than the subdirectory.

        Staging (rather than leaving the file fully untracked) isolates the
        cwd-scope bug this test targets from `git grep`'s separate, unrelated
        default of not searching untracked files at all."""
        _require_git()
        from fused_memory.middleware.recon_claim_verification_guard import (
            make_source_and_history_probe,
        )

        repo = tmp_path / "repo"
        repo.mkdir()
        _run_git(["init", "-b", "main"], cwd=repo)
        _run_git(["config", "user.email", "test@test.com"], cwd=repo)
        _run_git(["config", "user.name", "Test"], cwd=repo)

        pkg = repo / "pkg"
        pkg.mkdir()
        (pkg / "placeholder.py").write_text("PLACEHOLDER = 1\n", encoding="utf-8")
        _run_git(["add", "-A"], cwd=repo)
        _run_git(["commit", "-m", "init pkg"], cwd=repo)

        # Staged but NOT committed — outside pkg/, at the repo root. The
        # history pickaxe alone cannot see this (never committed); only a
        # tree grep rooted at the true top level can.
        (repo / "outside_pkg.py").write_text("SUBDIR_SCOPE_TOKEN = 1\n", encoding="utf-8")
        _run_git(["add", "outside_pkg.py"], cwd=repo)

        probe = make_source_and_history_probe(pkg)

        assert probe("SUBDIR_SCOPE_TOKEN") is True
