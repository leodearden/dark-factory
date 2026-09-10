"""Tests for the extracted git-authority landing tier.

THIS FILE COULD NOT HAVE EXISTED BEFORE THE EXTRACTION (task 4887, PRD
`docs/prds/landed-not-done-recovery.md` label ζ).  Every member it exercises
lived as a 4-space-indent sibling inside ``escalation.server.create_server``
and was reachable only by building a whole MCP server and driving the
``merge_status`` tool through it.  So the fact that nothing here constructs a
server IS the workstream's signal — ζ's stated intermediate signal is that
"the tier is importable outside ``create_server``'s scope", and an import at
module level plus direct calls is the most direct way to assert it.

The behavioural pins for the tier as it is reached THROUGH ``merge_status``
stay in ``test_merge_status_git_authority.py``; this file pins the extracted
module's own interface.
"""
from __future__ import annotations

import dataclasses
import enum
import logging
import types
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from shared.merge_state import MergeState

from escalation.git_authority import (
    GitAuthorityArm,
    GitAuthorityOutcome,
    GitAuthorityVerdict,
    TaskMetadataResult,
    found_on_main_response,
    probe_landing,
    task_metadata,
)


class TestFoundOnMainResponse:
    """The Tier-3.5 done/found_on_main renderer, called with no server."""

    def test_returns_the_exact_response_shape(self) -> None:
        """Whole-dict equality — no key may be added, dropped or renamed.

        A partial ``result['state'] == 'done'`` style assertion would let a
        silently-added key through, and the response shape is task 4831's to
        extend (`plans/merge-status-durable-non-landed-prd.md` D1/D2), not
        this task's.
        """
        assert found_on_main_response('req-1', 'a' * 40) == {
            'state': MergeState.done,
            'request_id': 'req-1',
            'generation': 1,
            'kind': 'found_on_main',
            'merge_sha': 'a' * 40,
            'outcome': 'found_on_main',
        }

    def test_accepts_a_none_request_id(self) -> None:
        """``request_id`` is ``str | None`` — merge_status may resolve by branch."""
        assert found_on_main_response(None, 'b' * 40) == {
            'state': MergeState.done,
            'request_id': None,
            'generation': 1,
            'kind': 'found_on_main',
            'merge_sha': 'b' * 40,
            'outcome': 'found_on_main',
        }

    def test_state_is_a_genuine_str(self) -> None:
        """The wire-compatibility property the shared vocabulary relies on.

        ``MergeState`` is a ``StrEnum``, so the emitted ``state`` is a real
        ``str`` instance and JSON-encodes as its plain spelling.  That is why
        the whole-dict pins above hold against plain-string expectations, and
        why moving the renderer out of server.py cannot change a byte on the
        wire.
        """
        state = found_on_main_response(None, 'c' * 40)['state']

        assert isinstance(state, str), f'state must be a genuine str, got {type(state)}'
        assert state == 'done'


class TestGitAuthorityOutcome:
    """The verdict vocabulary task 4831 (PRD label β) will switch on."""

    def test_is_a_str_enum(self) -> None:
        assert issubclass(GitAuthorityOutcome, enum.StrEnum)

    def test_has_exactly_three_members(self) -> None:
        """A fourth outcome must not appear without a reviewer seeing this move.

        ``landed_unconfirmed`` vs ``no_signal`` is this task's answer to "are
        'landed but unattributable' and 'we know nothing' the same
        proposition?" — they are NOT, and the distinction is materialized
        here, on the verdict, rather than on the ``merge_status`` MCP
        response, whose vocabulary belongs to task 4831.
        """
        assert {m.value for m in GitAuthorityOutcome} == {
            'found_on_main', 'landed_unconfirmed', 'no_signal',
        }

    @pytest.mark.parametrize(
        'member,spelling',
        [
            (GitAuthorityOutcome.found_on_main, 'found_on_main'),
            (GitAuthorityOutcome.landed_unconfirmed, 'landed_unconfirmed'),
            (GitAuthorityOutcome.no_signal, 'no_signal'),
        ],
    )
    def test_members_round_trip_as_their_plain_spelling(
        self, member: GitAuthorityOutcome, spelling: str
    ) -> None:
        assert member == spelling
        assert isinstance(member, str)


class TestGitAuthorityArm:
    """The other closed set on the verdict — which git signal reached it."""

    def test_is_a_str_enum(self) -> None:
        assert issubclass(GitAuthorityArm, enum.StrEnum)

    def test_has_exactly_two_members(self) -> None:
        """There are two git signals and no third: the branch ref still
        exists and its tip is an ancestor of main, or the ref is gone and a
        merge marker was found.  A third arm must not appear without a
        reviewer seeing this move.
        """
        assert {m.value for m in GitAuthorityArm} == {'ancestor', 'marker'}

    def test_members_round_trip_as_their_plain_spelling(self) -> None:
        """``arm`` was a bare ``str`` before, and every existing pin compares
        it against a plain spelling.  ``StrEnum`` is what keeps those pins —
        and any JSON a consumer builds from a verdict — byte-identical while
        the type becomes a closed set (heuristic 12).
        """
        assert GitAuthorityArm.ancestor == 'ancestor'
        assert GitAuthorityArm.marker == 'marker'
        assert isinstance(GitAuthorityArm.ancestor, str)


def _harness(
    *, task: Any = None, raises: bool = False, with_scheduler: bool = True,
) -> types.SimpleNamespace:
    """Build the minimal duck-typed harness ``task_metadata`` reads.

    The function touches nothing but ``harness.scheduler.get_task``, which is
    why it can be exercised with a two-attribute SimpleNamespace and no MCP
    server anywhere in sight.
    """
    harness = types.SimpleNamespace()
    if with_scheduler:
        harness.scheduler = types.SimpleNamespace(
            get_task=(
                AsyncMock(side_effect=RuntimeError('scheduler unreachable'))
                if raises else AsyncMock(return_value=task)
            ),
        )
    return harness


@pytest.mark.asyncio
class TestTaskMetadata:
    """The G7 fetch-outcome tri-state — a FAULT is not a FACT.

    Before the extraction this helper returned a bare ``{}`` on every failure
    mode, so a caller could not tell "the task genuinely declares no
    metadata" from "we could not read it".  The PRD's G7 requirement
    (``docs/prds/landed-not-done-recovery.md`` label ζ) is that the extracted
    API return the fetch outcome alongside the metadata, so leaf η's writer
    can treat "could not read metadata" as "cannot verify the degeneracy
    guard" rather than as "no degeneracy".
    """

    async def test_healthy_read_with_metadata(self) -> None:
        result = await task_metadata(
            _harness(task={'metadata': {'branch_base_sha': 'b' * 40}}),
            '4887', site='merge_status',
        )

        assert result.metadata == {'branch_base_sha': 'b' * 40}
        assert result.unavailable is False

    @pytest.mark.parametrize(
        'task', [{}, {'metadata': None}, {'metadata': {}}],
        ids=['key_absent', 'key_none', 'key_empty'],
    )
    async def test_healthy_read_without_metadata_is_not_unavailable(
        self, task: dict
    ) -> None:
        """The load-bearing half of the tri-state, with its twin below.

        The read SUCCEEDED and the answer is "this task declares nothing".
        That is a FACT, and it must not be reported with the same shape as
        the FAULT cases — otherwise ``{}`` is exactly as uninformative as it
        was before the extraction.
        """
        result = await task_metadata(_harness(task=task), '4887', site='merge_status')

        assert result.metadata == {}
        assert result.unavailable is False

    @pytest.mark.parametrize(
        'harness,label',
        [
            (None, 'harness absent'),
            (_harness(with_scheduler=False), 'no .scheduler attribute'),
            (_harness(raises=True), 'get_task raised'),
        ],
        ids=['no_harness', 'no_scheduler', 'get_task_raises'],
    )
    async def test_fault_modes_are_flagged_unavailable(
        self, harness: Any, label: str
    ) -> None:
        """The twin: ``{}`` here means "we could not read", not "nothing there".

        All three must ALSO honour the never-raises guarantee — a fault in
        the task store must degrade a single guard, not swallow the whole
        probe.
        """
        result = await task_metadata(harness, '4887', site='merge_status')

        assert result.metadata == {}, f'{label}: must fail open to an empty dict'
        assert result.unavailable is True, f'{label}: must be flagged unavailable'

    async def test_absent_task_is_a_fact_not_a_fault(self) -> None:
        """THE ONE GENUINELY ARGUABLE CELL — ``get_task`` answering None.

        Decided as ``unavailable is False``, i.e. "no such task" is a FACT.

        The flag's discriminant is READ SUCCESS, not guard-verifiability.
        That is forced by the cell above: a task record that is present but
        carries no metadata is pinned False, and it sits in exactly the same
        guard-verifiability position as this one (no ``branch_base_sha``
        either way).  So "cannot verify the degeneracy guard" cannot be what
        the flag means, and the only consistent reading left is fault vs.
        fact.  Here the scheduler was healthy and answered definitively.

        The counter-argument was weighed and rejected: one could call an
        absent record "unreadable" and flag it True on fail-safe grounds.
        But that would make ``unavailable is True`` mean two different things
        — a fault OR a legitimately-absent task — which is precisely the
        conflation this flag exists to remove, and it would leave η unable to
        tell them apart. The module docstring states the same rule.
        """
        result = await task_metadata(_harness(task=None), '4887', site='merge_status')

        assert result.metadata == {}
        assert result.unavailable is False

    async def test_degradation_is_logged_with_the_site_label(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Task 3103 review #3: without ``site`` a submit-path fault was logged
        as a merge_status failure, so an operator grepping for a submit-path
        degradation would not find it.
        """
        with caplog.at_level(logging.WARNING):
            await task_metadata(_harness(raises=True), '4887', site='merge_request')

        assert 'merge_request' in caplog.text, (
            f'the site label must be interpolated into the warning, got: {caplog.text!r}'
        )
        assert 'merge_status' not in caplog.text


class TestTaskMetadataResult:
    """The ``{}``-is-not-proof carrier itself."""

    def test_is_frozen(self) -> None:
        """Immutable so a consumer cannot rewrite the fetch outcome downstream."""
        result = TaskMetadataResult(metadata={})

        with pytest.raises(dataclasses.FrozenInstanceError):
            result.unavailable = True   # type: ignore[misc]

    def test_unavailable_defaults_to_false(self) -> None:
        """Mirrors ``TruthReport.escalation_store_unavailable``: defaulting keeps
        hand-construction ergonomic, so the common healthy case reads cleanly.
        """
        assert TaskMetadataResult(metadata={}).unavailable is False


# ---------------------------------------------------------------------------
# probe_landing — the extracted tier itself.
#
# Every case below runs with NO create_server and NO request_id, which IS
# ζ's user-observable signal.  The stub shapes are LIFTED from
# ``test_merge_status_git_authority.py`` rather than reinvented, so there is
# one stub vocabulary for this tier and not two.
#
# HAZARD carried over from that module's ``_stub_git_ops`` docstring: any NEW
# sub-method the real code starts calling must be given a default here, or it
# raises inside the fire-safe wrapper and every case silently degrades to
# ``no_signal`` — which looks like a passing guard.  That is why each test
# below asserts POSITIVELY on its expected outcome instead of merely
# asserting "not found_on_main".
# ---------------------------------------------------------------------------

_MAIN = 'm' * 40


def _probe_git_ops(**overrides: Any) -> types.SimpleNamespace:
    stub = types.SimpleNamespace(
        resolve_branch_sha=AsyncMock(return_value=None),
        is_ancestor=AsyncMock(return_value=False),
        find_merge_marker=AsyncMock(return_value=None),
        find_task_citation_commit=AsyncMock(return_value=None),
        commit_effect_present_in_main=AsyncMock(return_value=True),
    )
    for name, fn in overrides.items():
        setattr(stub, name, fn)
    return stub


def _ancestor_git_ops(
    *, tip: str, citation: str | None = None, effect_present: bool = True,
) -> types.SimpleNamespace:
    return _probe_git_ops(
        resolve_branch_sha=AsyncMock(
            side_effect=lambda b: tip if b.startswith('task/') else _MAIN
        ),
        is_ancestor=AsyncMock(return_value=True),
        find_task_citation_commit=AsyncMock(return_value=citation),
        commit_effect_present_in_main=AsyncMock(return_value=effect_present),
    )


def _marker_git_ops(
    *, marker: str | None, predates_base: bool = False, effect_present: bool = True,
) -> types.SimpleNamespace:
    return _probe_git_ops(
        resolve_branch_sha=AsyncMock(return_value=None),   # branch ref gone
        is_ancestor=AsyncMock(return_value=predates_base),
        find_merge_marker=AsyncMock(return_value=marker),
        commit_effect_present_in_main=AsyncMock(return_value=effect_present),
    )


def _config(tmp_path: Path, *, commit_citation_pattern: str | None = None) -> Any:
    from orchestrator.config import (  # type: ignore[reportMissingImports]
        GitConfig,
        OrchestratorConfig,
    )
    return OrchestratorConfig(
        project_root=tmp_path,
        max_concurrent_tasks=1,
        git=GitConfig(
            main_branch='main', branch_prefix='task/', remote='origin',
            worktree_dir='.worktrees',
            commit_citation_pattern=commit_citation_pattern,
        ),
    )


@pytest.mark.asyncio
class TestProbeLanding:
    """The tier, called directly — no MCP server, no request_id."""

    async def test_ancestor_arm_accept(self, tmp_path: Path) -> None:
        verdict = await probe_landing(
            _ancestor_git_ops(tip='a' * 40, citation='c' * 40), '900',
            orch_config=_config(tmp_path),
            harness=_harness(task={'metadata': {'branch_base_sha': 'b' * 40}}),
        )

        assert verdict.outcome is GitAuthorityOutcome.found_on_main
        assert verdict.merge_sha == 'c' * 40, 'merge_sha is the CITATION commit on main'
        assert verdict.arm is GitAuthorityArm.ancestor
        assert verdict.citation_gate_skipped is False, (
            'the gate RAN here — the flag is the discriminator a provenance '
            'writer branches on, so the fully-guarded case must clear it'
        )

    async def test_marker_arm_accept(self, tmp_path: Path) -> None:
        verdict = await probe_landing(
            _marker_git_ops(marker='d' * 40), '801',
            orch_config=_config(tmp_path),
            harness=_harness(task={'metadata': {'branch_base_sha': 'b' * 40}}),
        )

        assert verdict.outcome is GitAuthorityOutcome.found_on_main
        assert verdict.merge_sha == 'd' * 40
        assert verdict.arm is GitAuthorityArm.marker
        assert verdict.citation_gate_skipped is False, (
            'the marker arm never runs the citation gate to begin with, so it '
            'must not claim the gate was SKIPPED — that flag means the '
            'commit_citation_pattern opt-out weakened merge_sha'
        )

    async def test_citation_gate_opt_out_arm(self, tmp_path: Path) -> None:
        """``commit_citation_pattern == ''`` — the ONLY path that answers
        ``found_on_main`` with a sha that is not a commit discovered on main.

        ``test_merge_status_git_authority.py`` covers this through the MCP
        tool, where all that is observable is ``state == 'done'`` and the
        sha.  The verdict shape is what leaf η and task 4831 consume, so the
        weakening has to be pinned HERE: ``merge_sha`` is the raw branch tip,
        neither citation-discovered nor effect-present-checked, and
        ``citation_gate_skipped`` is the executable discriminator that says
        so.  Without this pin a future edit could drop the flag, repoint
        ``merge_sha``, or relabel the arm and every other test in both files
        would still pass while a provenance writer silently changed what it
        stamps.
        """
        tip = 'a' * 40
        git_ops = _ancestor_git_ops(tip=tip, citation=None)

        verdict = await probe_landing(
            git_ops, '904', orch_config=_config(tmp_path, commit_citation_pattern=''),
            harness=_harness(task={'metadata': {'branch_base_sha': 'b' * 40}}),
        )

        assert verdict.outcome is GitAuthorityOutcome.found_on_main
        assert verdict.merge_sha == tip, 'merge_sha is the raw BRANCH TIP here'
        assert verdict.arm is GitAuthorityArm.ancestor
        assert verdict.citation_gate_skipped is True, (
            'a caller stamping provenance must be able to SEE that the gate '
            'was skipped, rather than inferring it from evidence_sha is None'
        )
        assert verdict.evidence_sha is None
        assert verdict.reason is None
        git_ops.find_task_citation_commit.assert_not_called()

    async def test_ancestor_effect_absent_is_landed_unconfirmed(
        self, tmp_path: Path
    ) -> None:
        """THE MEASURED 95.4% COMMON CASE, and the reason both outcomes exist.

        A citing commit was FOUND on main — git positively established this
        branch's work landed — and only effect-survival failed.  Reporting
        that as ``no_signal`` would claim we know nothing, which is false.
        """
        verdict = await probe_landing(
            _ancestor_git_ops(tip='a' * 40, citation='c' * 40, effect_present=False),
            '901', orch_config=_config(tmp_path),
            harness=_harness(task={'metadata': {'branch_base_sha': 'b' * 40}}),
        )

        assert verdict.outcome is GitAuthorityOutcome.landed_unconfirmed
        assert verdict.reason == 'effect_absent'
        assert verdict.evidence_sha == 'c' * 40
        assert verdict.merge_sha is None, 'merge_sha is for ACCEPTED landings only'

    async def test_ancestor_no_citation_is_landed_unconfirmed(
        self, tmp_path: Path
    ) -> None:
        verdict = await probe_landing(
            _ancestor_git_ops(tip='a' * 40, citation=None), '3031',
            orch_config=_config(tmp_path),
            harness=_harness(task={'metadata': {'branch_base_sha': 'b' * 40}}),
        )

        assert verdict.outcome is GitAuthorityOutcome.landed_unconfirmed
        assert verdict.reason == 'no_citation'
        assert verdict.evidence_sha is None
        assert verdict.merge_sha is None

    async def test_marker_effect_absent_is_landed_unconfirmed(
        self, tmp_path: Path
    ) -> None:
        verdict = await probe_landing(
            _marker_git_ops(marker='d' * 40, effect_present=False), '803',
            orch_config=_config(tmp_path),
            harness=_harness(task={'metadata': {'branch_base_sha': 'b' * 40}}),
        )

        assert verdict.outcome is GitAuthorityOutcome.landed_unconfirmed
        assert verdict.arm is GitAuthorityArm.marker
        assert verdict.evidence_sha == 'd' * 40
        assert verdict.merge_sha is None

    @pytest.mark.parametrize(
        'git_ops,task,label',
        [
            (
                _ancestor_git_ops(tip='b' * 40, citation='c' * 40),
                {'metadata': {'branch_base_sha': 'b' * 40}},
                'degenerate branch — tip still at its creation point',
            ),
            (
                _probe_git_ops(
                    resolve_branch_sha=AsyncMock(return_value='a' * 40),
                    is_ancestor=AsyncMock(return_value=False),
                ),
                {'metadata': {}},
                'branch present but not an ancestor of main',
            ),
            (
                _probe_git_ops(resolve_branch_sha=AsyncMock(return_value=_MAIN)),
                {'metadata': {}},
                'branch sitting at exactly main HEAD',
            ),
            (
                _marker_git_ops(marker=None),
                {'metadata': {}},
                'branch deleted, no merge marker found',
            ),
            (
                _marker_git_ops(marker='d' * 40, predates_base=True),
                {'metadata': {'branch_base_sha': 'b' * 40}},
                'marker PREDATES branch_base_sha',
            ),
        ],
        ids=['degenerate', 'not_ancestor', 'at_main_head', 'no_marker', 'predates_base'],
    )
    async def test_genuinely_no_knowledge_paths_are_no_signal(
        self, tmp_path: Path, git_ops: Any, task: Any, label: str
    ) -> None:
        """No landing was ESTABLISHED on any of these — nothing to be
        unconfirmed about.

        The predates-veto case is the sharpest: a marker belonging to a
        PREVIOUS incarnation of a reused task id is evidence about a
        different run entirely, so dressing it up as ``landed_unconfirmed``
        would assert a landing this task never made.
        """
        verdict = await probe_landing(
            git_ops, '800',
            orch_config=_config(tmp_path), harness=_harness(task=task),
        )

        assert verdict.outcome is GitAuthorityOutcome.no_signal, label
        assert verdict.merge_sha is None, label

    async def test_is_fire_safe(self, tmp_path: Path) -> None:
        """A git fault degrades to the honest ``no_signal``, never propagates.

        The wrapper lives INSIDE probe_landing so leaf η inherits the
        fire-safety instead of having to re-add it.
        """
        verdict = await probe_landing(
            _probe_git_ops(
                resolve_branch_sha=AsyncMock(side_effect=RuntimeError('git exploded')),
            ),
            '905', orch_config=_config(tmp_path),
            harness=_harness(task={'metadata': {}}),
        )

        assert verdict.outcome is GitAuthorityOutcome.no_signal

    async def test_site_labels_every_degradation_this_probe_logs(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """``site`` defaults to merge_status, and a NON-merge_status consumer
        must be able to find its own faults in the log.

        Task 3103 review #3 added the label for exactly this: an unlabelled
        degradation is invisible to an operator grepping for the caller they
        actually invoked.  The extraction's whole purpose is a second
        consumer (leaf η's periodic writer, task 4651), so hardcoding
        ``merge_status`` here would re-create that defect for the new caller
        on day one — its faults would be logged against a tool nobody ran.
        Both warnings this probe can emit are checked: its own fire-safe
        wrapper, and the one ``task_metadata`` emits underneath it.
        """
        with caplog.at_level(logging.WARNING):
            await probe_landing(
                _probe_git_ops(
                    resolve_branch_sha=AsyncMock(side_effect=RuntimeError('git exploded')),
                ),
                '4651', orch_config=_config(tmp_path),
                harness=_harness(raises=True), site='landed_reconciler',
            )

        assert 'landed_reconciler' in caplog.text, (
            f'the fire-safe warning must name the CALLER, got: {caplog.text!r}'
        )
        assert 'merge_status' not in caplog.text

    async def test_site_reaches_the_metadata_fetch(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The label is threaded THROUGH to ``task_metadata``, not just used
        by the wrapper — a scheduler fault under a reconciler probe is logged
        against the reconciler.
        """
        with caplog.at_level(logging.WARNING):
            verdict = await probe_landing(
                _ancestor_git_ops(tip='a' * 40, citation='c' * 40), '4651',
                orch_config=_config(tmp_path), harness=_harness(raises=True),
                site='landed_reconciler',
            )

        assert verdict.metadata_unavailable is True, 'precondition: the fetch faulted'
        assert 'scheduler.get_task' in caplog.text
        assert 'landed_reconciler' in caplog.text, (
            f'the metadata degradation must name the CALLER, got: {caplog.text!r}'
        )
        assert 'merge_status' not in caplog.text

    async def test_metadata_unavailable_is_threaded_onto_the_verdict(
        self, tmp_path: Path
    ) -> None:
        """THE G7 PAYLOAD, and it can only be pinned here.

        merge_status fails open identically whether the metadata read
        succeeded-and-found-nothing or failed outright, so the distinction is
        invisible through the MCP tool.  The two runs below differ ONLY in
        that one's scheduler raises: the outcome is identical and the flag is
        not.
        """
        healthy = await probe_landing(
            _ancestor_git_ops(tip='a' * 40, citation='c' * 40), '906',
            orch_config=_config(tmp_path), harness=_harness(task={'metadata': {}}),
        )
        faulted = await probe_landing(
            _ancestor_git_ops(tip='a' * 40, citation='c' * 40), '906',
            orch_config=_config(tmp_path), harness=_harness(raises=True),
        )

        assert healthy.outcome is faulted.outcome, (
            'precondition: the fault must not change the OUTCOME, only the flag'
        )
        assert healthy.metadata_unavailable is False
        assert faulted.metadata_unavailable is True


class TestGitAuthorityVerdict:
    def test_is_frozen(self) -> None:
        verdict = GitAuthorityVerdict(outcome=GitAuthorityOutcome.no_signal)

        with pytest.raises(dataclasses.FrozenInstanceError):
            verdict.merge_sha = 'x' * 40   # type: ignore[misc]


def test_module_does_not_statically_import_orchestrator() -> None:
    """THE LAYERING CONSTRAINT, pinned directly rather than trusted to a comment.

    escalation/pyproject.toml declares no orchestrator dependency; the reverse
    import resolves at runtime only because the escalation server is hosted
    inside the orchestrator process.  Hoisting it to module level would turn
    today's graceful Tier-4 degradation into a server-CONSTRUCTION failure —
    a fail-open becoming a fail-closed, silently, on a refactor.

    Checked in a FRESH subprocess because this test session has orchestrator
    imported already, so an in-process ``sys.modules`` check would pass
    vacuously.
    """
    import subprocess
    import sys

    probe = (
        'import sys; import escalation.git_authority; '
        "print(any(m == 'orchestrator' or m.startswith('orchestrator.') "
        'for m in sys.modules))'
    )
    out = subprocess.run(
        [sys.executable, '-c', probe],
        capture_output=True, text=True, check=True,
    )

    assert out.stdout.strip() == 'False', (
        f'escalation.git_authority must not import orchestrator at module '
        f'level; got {out.stdout.strip()!r}'
    )
