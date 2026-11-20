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
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from _landing_verdicts import stub_landing_producer
from _rebased_landing_git import (
    add_main_merged_branch,
    add_net_empty_branch,
    add_unlanded_branch,
    build_rebased_solo_landing,
    new_repo,
    reapply_tip_on_main,
)
from _rebased_landing_git import commit as landing_commit
from _rebased_landing_git import git as landing_git
from shared.merge_state import MergeState
from shared.task_read import TaskAbsent, TaskFound, TaskReadFailed

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

    def test_has_exactly_four_members(self) -> None:
        """A fifth outcome must not appear without a reviewer seeing this move.

        ``landed_unconfirmed`` vs ``no_signal`` vs ``probe_failed`` are three
        different propositions — "landed but unattributable", "we looked and
        found nothing", and "we could not look" — and merge_status renders
        each differently (task 4831).
        """
        assert {m.value for m in GitAuthorityOutcome} == {
            'found_on_main', 'landed_unconfirmed', 'no_signal', 'probe_failed',
        }

    @pytest.mark.parametrize(
        'member,spelling',
        [
            (GitAuthorityOutcome.found_on_main, 'found_on_main'),
            (GitAuthorityOutcome.landed_unconfirmed, 'landed_unconfirmed'),
            (GitAuthorityOutcome.no_signal, 'no_signal'),
            (GitAuthorityOutcome.probe_failed, 'probe_failed'),
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

    def test_has_exactly_three_members(self) -> None:
        """There are three git signals and no fourth: the branch ref still
        exists and its tip is an ancestor of main; the ref is gone and a
        merge marker was found; or the ref still exists, its tip is NOT an
        ancestor of main, and every one of its commits is patch-equivalent
        on main — the worker-rebased landing.  A fourth arm must not appear
        without a reviewer seeing this move.
        """
        assert {m.value for m in GitAuthorityArm} == {'ancestor', 'marker', 'patch_id'}

    def test_members_round_trip_as_their_plain_spelling(self) -> None:
        """``arm`` was a bare ``str`` before, and every existing pin compares
        it against a plain spelling.  ``StrEnum`` is what keeps those pins —
        and any JSON a consumer builds from a verdict — byte-identical while
        the type becomes a closed set (heuristic 12).
        """
        assert GitAuthorityArm.ancestor == 'ancestor'
        assert GitAuthorityArm.marker == 'marker'
        assert GitAuthorityArm.patch_id == 'patch_id'
        assert isinstance(GitAuthorityArm.ancestor, str)


_BACKEND_FAULT_REASON = "get_task error='boom', error_type='BackendUnavailable'"


_NO_SUCH_TASK = TaskAbsent()


def _harness(
    *, read: Any = _NO_SUCH_TASK, raises: bool = False, with_scheduler: bool = True,
) -> types.SimpleNamespace:
    """Build the minimal duck-typed harness ``task_metadata`` reads.

    The function touches nothing but ``harness.scheduler.read_task``, which is
    why it can be exercised with a two-attribute SimpleNamespace and no MCP
    server anywhere in sight.  *read* is what ``read_task`` answers: a
    ``shared.task_read`` outcome, by default the healthy "no such task".
    """
    harness = types.SimpleNamespace()
    if with_scheduler:
        harness.scheduler = types.SimpleNamespace(
            read_task=(
                AsyncMock(side_effect=RuntimeError('scheduler unreachable'))
                if raises else AsyncMock(return_value=read)
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
            _harness(read=TaskFound({'metadata': {'branch_base_sha': 'b' * 40}})),
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
        result = await task_metadata(_harness(read=TaskFound(task)), '4887', site='merge_status')

        assert result.metadata == {}
        assert result.unavailable is False

    @pytest.mark.parametrize(
        'harness,label',
        [
            (None, 'harness absent'),
            (_harness(with_scheduler=False), 'no .scheduler attribute'),
            (_harness(raises=True), 'read_task raised'),
            (
                _harness(read=TaskReadFailed(_BACKEND_FAULT_REASON)),
                'read_task reported a failed read',
            ),
            (_harness(read={'metadata': {}}), 'read_task answered an unexpected value'),
            (
                types.SimpleNamespace(
                    scheduler=types.SimpleNamespace(
                        get_task=AsyncMock(return_value={'metadata': {}}),
                    ),
                ),
                'scheduler has no read_task',
            ),
        ],
        ids=[
            'no_harness', 'no_scheduler', 'read_task_raises',
            'read_task_reports_failure', 'read_task_unexpected_value',
            'scheduler_without_read_task',
        ],
    )
    async def test_fault_modes_are_flagged_unavailable(
        self, harness: Any, label: str
    ) -> None:
        """The twin: ``{}`` here means "we could not read", not "nothing there".

        ``read_task_reports_failure`` is the live fault in production: the
        real ``Scheduler.read_task`` never raises, so a fused-memory error
        reply arrives as a ``TaskReadFailed``, not as an exception.  Every
        mode must ALSO honour the never-raises guarantee — a fault in the
        task store must degrade a single guard, not swallow the whole probe.
        """
        result = await task_metadata(harness, '4887', site='merge_status')

        assert result.metadata == {}, f'{label}: must fail open to an empty dict'
        assert result.unavailable is True, f'{label}: must be flagged unavailable'

    async def test_absent_task_is_a_fact_not_a_fault(self) -> None:
        """THE ONE GENUINELY ARGUABLE CELL — ``read_task`` answering TaskAbsent.

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
        result = await task_metadata(_harness(read=TaskAbsent()), '4887', site='merge_status')

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

    async def test_failed_read_is_logged_with_the_site_label_not_the_reason(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """``Scheduler.read_task`` owns the reason's WARNING; repeating it here
        logged one fault twice.
        """
        with caplog.at_level(logging.WARNING):
            await task_metadata(
                _harness(read=TaskReadFailed(_BACKEND_FAULT_REASON)),
                '4887', site='merge_request',
            )

        assert 'merge_request' in caplog.text
        assert '4887' in caplog.text
        assert 'BackendUnavailable' not in caplog.text


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


def _ancestor_git_ops(*, tip: str) -> types.SimpleNamespace:
    """A live ref whose tip is an ancestor of main.

    Past its own guards the ancestor arm asks ``branch_work_landed``, which a
    stub cannot answer; pair this with
    :func:`_landing_verdicts.stub_landing_producer`.
    """
    return _probe_git_ops(
        resolve_branch_sha=AsyncMock(
            side_effect=lambda b: tip if b.startswith('task/') else _MAIN
        ),
        is_ancestor=AsyncMock(return_value=True),
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


def _config(
    tmp_path: Path, *, commit_citation_pattern: str | None = None,
    delivered_checks_enabled: bool = True,
) -> Any:
    from orchestrator.config import (  # type: ignore[reportMissingImports]
        DeliveredChecksConfig,
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
        delivered_checks=DeliveredChecksConfig(enabled=delivered_checks_enabled),
    )


def _task_declaring_checks() -> dict[str, Any]:
    """A task record whose metadata declares a delivered check.

    Non-empty is the whole point: ``_evidence_verdict`` forwards
    ``metadata.delivered_checks or []``, and ``validate_landing_evidence``
    guards its second accept path on that list being TRUTHY, so an empty
    list leaves the path unreachable — today's behaviour for every task
    that declares nothing.
    """
    return {
        'metadata': {
            'branch_base_sha': 'b' * 40,
            'delivered_checks': [
                {'name': 'cap', 'kind': 'grep', 'pattern': 'x', 'expect': 'present'},
            ],
        },
    }


def _confirming_check_runner(refs_run: list[str]) -> Callable[..., Any]:
    """A ``run_delivered_check`` fake that makes the REAL differential confirm.

    FALSE before the landing (``<sha>^1``), TRUE at it and at main: the
    contract of ``orchestrator/tests/_landing_evidence_helpers.py::differential_confirms``,
    which escalation tests cannot import.  Every ref asked about is appended
    to *refs_run*, so a test can tell a check that never ran from one that
    ran and declined.
    """
    from orchestrator.delivered_checks import (
        DeliveredCheckResult,
    )

    async def _run(
        check: dict[str, Any], *, project_root: Any, ref: str = 'main', runner: Any = None,
    ) -> Any:
        refs_run.append(ref)
        if ref.endswith('^1'):
            return DeliveredCheckResult.FAILED
        return DeliveredCheckResult.DELIVERED

    return _run


def _force_differential_confirm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the three-leg delivered-checks differential confirm.

    Patches the DEFINING module — ``validate_landing_evidence`` resolves
    ``_delivered_checks_differential`` as a module global, and
    ``_evidence_verdict``'s own import is per-call — so the REAL accept/reject
    decision logic is exercised and only the leg-running subprocess work is
    replaced.  The stub writes the same ``probe`` key the real function
    writes on a confirmation
    (``orchestrator/src/orchestrator/landing_evidence.py::_delivered_checks_differential``)
    and takes ``**kwargs`` because the DISCOVERY arm passes an extra
    ``branch_tip`` keyword the CANDIDATE arm does not.
    """
    async def _confirm(
        git_ops: Any, effect_check_sha: str, delivered_checks: Any,
        probe: dict[str, Any], **kwargs: Any,
    ) -> bool:
        probe['delivered_checks_legs'] = []
        probe['delivered_checks_outcome'] = 'confirmed'
        return True

    monkeypatch.setattr(
        'orchestrator.merge_lane.landing_evidence._delivered_checks_differential', _confirm,
    )


@pytest.mark.asyncio
class TestProbeLanding:
    """The tier, called directly — no MCP server, no request_id."""

    async def test_ancestor_arm_accept(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        stub_landing_producer(monkeypatch, 'landed', evidence_sha='c' * 40)

        verdict = await probe_landing(
            _ancestor_git_ops(tip='a' * 40), '900',
            orch_config=_config(tmp_path),
            harness=_harness(read=TaskFound({'metadata': {'branch_base_sha': 'b' * 40}})),
        )

        assert verdict.outcome is GitAuthorityOutcome.found_on_main
        assert verdict.merge_sha == 'c' * 40, "merge_sha is the producer's evidence on main"
        assert verdict.arm is GitAuthorityArm.ancestor
        assert verdict.citation_gate_skipped is False, (
            'the gate RAN here — the flag is the discriminator a provenance '
            'writer branches on, so the fully-guarded case must clear it'
        )

    async def test_marker_arm_accept(self, tmp_path: Path) -> None:
        verdict = await probe_landing(
            _marker_git_ops(marker='d' * 40), '801',
            orch_config=_config(tmp_path),
            harness=_harness(read=TaskFound({'metadata': {'branch_base_sha': 'b' * 40}})),
        )

        assert verdict.outcome is GitAuthorityOutcome.found_on_main
        assert verdict.merge_sha == 'd' * 40
        assert verdict.arm is GitAuthorityArm.marker
        assert verdict.citation_gate_skipped is False, (
            'the marker arm never runs the citation gate to begin with, so it '
            'must not claim the gate was SKIPPED — that flag means the '
            'commit_citation_pattern opt-out weakened merge_sha'
        )

    @pytest.mark.parametrize('make_git_ops', [
        pytest.param(lambda: _marker_git_ops(marker='d' * 40), id='marker'),
        pytest.param(
            lambda: _ancestor_git_ops(tip='a' * 40), id='ancestor',
        ),
    ])
    @pytest.mark.parametrize('key', ['801', 'task/801'], ids=['bare', 'prefixed'])
    async def test_metadata_is_read_by_the_bare_task_id_for_either_key_shape(
        self, tmp_path: Path, key: str, monkeypatch: pytest.MonkeyPatch,
        make_git_ops: Callable[[], types.SimpleNamespace],
    ) -> None:
        """probe_landing derives the task id it reads metadata by from either key shape."""
        stub_landing_producer(monkeypatch, 'landed', evidence_sha='c' * 40)
        git_ops = make_git_ops()
        harness = _harness(read=TaskFound({'metadata': {'branch_base_sha': 'b' * 40}}))

        verdict = await probe_landing(
            git_ops, key, orch_config=_config(tmp_path), harness=harness,
        )

        assert verdict.outcome is GitAuthorityOutcome.found_on_main
        harness.scheduler.read_task.assert_awaited_once_with('801')

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
        git_ops = _ancestor_git_ops(tip=tip)

        verdict = await probe_landing(
            git_ops, '904', orch_config=_config(tmp_path, commit_citation_pattern=''),
            harness=_harness(read=TaskFound({'metadata': {'branch_base_sha': 'b' * 40}})),
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

    async def test_ancestor_no_attribution_is_landed_unconfirmed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The reason both outcomes exist.

        The producer proved the branch's work is on main and could not
        attribute it to the task.  Reporting that as ``no_signal`` would
        claim we know nothing, which is false.
        """
        stub_landing_producer(monkeypatch, 'no_attribution')

        verdict = await probe_landing(
            _ancestor_git_ops(tip='a' * 40), '3031',
            orch_config=_config(tmp_path),
            harness=_harness(read=TaskFound({'metadata': {'branch_base_sha': 'b' * 40}})),
        )

        assert verdict.outcome is GitAuthorityOutcome.landed_unconfirmed
        assert verdict.arm is GitAuthorityArm.ancestor
        assert verdict.reason == 'no_attribution'
        assert verdict.evidence_sha is None
        assert verdict.merge_sha is None, 'merge_sha is for ACCEPTED landings only'

    @pytest.mark.parametrize(('reason', 'outcome'), [
        ('not_landed', GitAuthorityOutcome.no_signal),
        ('no_op_landing', GitAuthorityOutcome.no_signal),
        ('degenerate_branch', GitAuthorityOutcome.no_signal),
        ('git_error', GitAuthorityOutcome.probe_failed),
    ])
    async def test_ancestor_arm_maps_every_other_reject(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        reason: str, outcome: GitAuthorityOutcome,
    ) -> None:
        """No landing established is ``no_signal``; a detector that could not
        look is ``probe_failed``, never a negative."""
        stub_landing_producer(monkeypatch, reason)

        verdict = await probe_landing(
            _ancestor_git_ops(tip='a' * 40), '3031',
            orch_config=_config(tmp_path),
            harness=_harness(read=TaskFound({'metadata': {'branch_base_sha': 'b' * 40}})),
        )

        assert verdict.outcome is outcome
        assert verdict.arm is GitAuthorityArm.ancestor
        assert verdict.reason == reason
        assert verdict.merge_sha is None

    async def test_marker_effect_absent_is_landed_unconfirmed(
        self, tmp_path: Path
    ) -> None:
        verdict = await probe_landing(
            _marker_git_ops(marker='d' * 40, effect_present=False), '803',
            orch_config=_config(tmp_path),
            harness=_harness(read=TaskFound({'metadata': {'branch_base_sha': 'b' * 40}})),
        )

        assert verdict.outcome is GitAuthorityOutcome.landed_unconfirmed
        assert verdict.arm is GitAuthorityArm.marker
        assert verdict.evidence_sha == 'd' * 40
        assert verdict.merge_sha is None

    # ---------------------------------------------------------------
    # The delivered-checks rescue — the SECOND way ``merge_sha``'s
    # guarantee can be weakened, and the one THIS module's own wiring
    # made reachable for the first time.
    #
    # ``_evidence_verdict`` passes ``delivered_checks=`` into
    # ``validate_landing_evidence``.  That activates the SECOND ACCEPT
    # PATH which sits INSIDE the ``commit_effect_present_in_main(...) is
    # False`` branch, so a ``found_on_main`` with a populated ``merge_sha``
    # can name a landing whose effect is ABSENT at main HEAD.  Since task
    # 4648 only the marker arm reaches ``_evidence_verdict``; the live-ref
    # arms are decided by ``branch_work_landed``, which never consults the
    # checks, so they can never be rescued.
    #
    # The differential is forced through the DEFINING module
    # (``orchestrator.merge_lane.landing_evidence``), the same way
    # ``test_merge_status_git_authority.py``'s
    # ``_record_validate_landing_evidence`` patches its own target, so the
    # REAL ``validate_landing_evidence`` decision logic stays in the loop.
    # That is the point: these pin that the second accept path is
    # REACHABLE FROM HERE, which a stubbed-out ``validate_landing_evidence``
    # returning a hand-built accepted verdict would not.
    # ---------------------------------------------------------------

    @pytest.mark.parametrize(('reason', 'outcome'), [
        ('landed', GitAuthorityOutcome.found_on_main),
        ('no_attribution', GitAuthorityOutcome.landed_unconfirmed),
    ])
    async def test_the_ancestor_arm_is_never_rescued_by_delivered_checks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        reason: str, outcome: GitAuthorityOutcome,
    ) -> None:
        """A task declaring checks, and a differential that WOULD confirm.

        The ancestor arm's producer never consults the checks, so its accept
        stays fully guarded and its reject stays a reject.
        """
        _force_differential_confirm(monkeypatch)
        stub_landing_producer(
            monkeypatch, reason, evidence_sha='c' * 40 if reason == 'landed' else None,
        )

        verdict = await probe_landing(
            _ancestor_git_ops(tip='a' * 40), '907', orch_config=_config(tmp_path),
            harness=_harness(read=TaskFound(_task_declaring_checks())),
        )

        assert verdict.outcome is outcome
        assert verdict.arm is GitAuthorityArm.ancestor
        assert verdict.rescued_by_delivered_checks is False
        assert verdict.merge_sha_fully_guarded is (outcome is GitAuthorityOutcome.found_on_main)

    async def test_marker_arm_accept_rescued_by_delivered_checks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Effect ABSENT, accepted anyway — and the verdict says which.

        Same git facts as ``test_marker_effect_absent_is_landed_unconfirmed``
        above; the ONLY differences are a task that declares delivered_checks
        and a differential that confirms.  Without
        ``rescued_by_delivered_checks`` this verdict is byte-identical to a
        fully-guarded accept, and a provenance writer stamping ``merge_sha``
        cannot tell a live landing from one main no longer carries.
        """
        _force_differential_confirm(monkeypatch)

        verdict = await probe_landing(
            _marker_git_ops(marker='d' * 40, effect_present=False), '908',
            orch_config=_config(tmp_path),
            harness=_harness(read=TaskFound(_task_declaring_checks())),
        )

        assert verdict.outcome is GitAuthorityOutcome.found_on_main, (
            'precondition: the differential must actually rescue the reject'
        )
        assert verdict.merge_sha == 'd' * 40
        assert verdict.arm is GitAuthorityArm.marker
        assert verdict.rescued_by_delivered_checks is True
        assert verdict.merge_sha_fully_guarded is False

    @pytest.mark.parametrize('enabled', [True, False])
    async def test_the_project_kill_switch_governs_the_marker_arms_rescue(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, enabled: bool,
    ) -> None:
        """``delivered_checks.enabled`` reaches the differential from orch_config.

        The REAL differential runs, with only its leaf check runner faked,
        because the kill switch lives inside it.  A real ``GitOps.config`` is
        the ``git`` sub-config and cannot carry the leaf, so the probe must
        pass it from the project config.  The marker arm is the only one that
        reaches the differential (task 4648).
        """
        git_ops = _marker_git_ops(marker='d' * 40, effect_present=False)
        git_ops.project_root = tmp_path
        refs_run: list[str] = []
        monkeypatch.setattr(
            'orchestrator.delivered_checks.run_delivered_check',
            _confirming_check_runner(refs_run),
        )

        verdict = await probe_landing(
            git_ops, '910',
            orch_config=_config(tmp_path, delivered_checks_enabled=enabled),
            harness=_harness(read=TaskFound(_task_declaring_checks())),
        )

        if enabled:
            assert verdict.outcome is GitAuthorityOutcome.found_on_main
            assert verdict.rescued_by_delivered_checks is True
            assert verdict.merge_sha_fully_guarded is False
        else:
            assert verdict.outcome is GitAuthorityOutcome.landed_unconfirmed
            assert verdict.rescued_by_delivered_checks is False
            assert verdict.merge_sha is None
            assert refs_run == [], 'a disabled feature runs no declared check'

    async def test_declaring_checks_does_not_by_itself_set_the_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """NEGATIVE CONTROL: the flag reports the PROBE OUTCOME, not the
        task's declaration.

        Identical task metadata to the rescued case; the only change is
        that the effect-present guard ACCEPTS, so the differential is never
        consulted at all (``landing_evidence.py``'s consumer documents the
        absent ``delivered_checks_outcome`` as "the survival check
        accepted, so there was nothing to rescue").  A flag that merely
        echoed ``metadata.delivered_checks`` would fail here — and would
        wrongly warn about every capability task's fully-guarded landing.
        """
        _force_differential_confirm(monkeypatch)

        verdict = await probe_landing(
            _marker_git_ops(marker='d' * 40, effect_present=True),
            '909', orch_config=_config(tmp_path),
            harness=_harness(read=TaskFound(_task_declaring_checks())),
        )

        assert verdict.outcome is GitAuthorityOutcome.found_on_main
        assert verdict.merge_sha == 'd' * 40
        assert verdict.rescued_by_delivered_checks is False
        assert verdict.merge_sha_fully_guarded is True

    async def test_citation_gate_opt_out_leaves_the_rescue_flag_clear(
        self, tmp_path: Path
    ) -> None:
        """The two weakenings stay independent in the other direction too.

        The opt-out path returns straight out of ``probe_landing`` without
        reaching ``_evidence_verdict``, so no differential can have run —
        claiming a rescue here would be a fabricated provenance fact.
        """
        verdict = await probe_landing(
            _ancestor_git_ops(tip='a' * 40), '910',
            orch_config=_config(tmp_path, commit_citation_pattern=''),
            harness=_harness(read=TaskFound(_task_declaring_checks())),
        )

        assert verdict.outcome is GitAuthorityOutcome.found_on_main
        assert verdict.citation_gate_skipped is True
        assert verdict.rescued_by_delivered_checks is False
        assert verdict.merge_sha_fully_guarded is False, (
            'either weakening alone is enough to disqualify merge_sha from '
            'being stamped as-is'
        )

    @pytest.mark.parametrize(
        'git_ops,task,label',
        [
            (
                _ancestor_git_ops(tip='b' * 40),
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
            orch_config=_config(tmp_path), harness=_harness(read=TaskFound(task)),
        )

        assert verdict.outcome is GitAuthorityOutcome.no_signal, label
        assert verdict.merge_sha is None, label

    async def test_is_fire_safe(self, tmp_path: Path) -> None:
        """A git fault degrades to ``probe_failed``, never propagates.

        The wrapper lives INSIDE probe_landing so leaf η inherits the
        fire-safety instead of having to re-add it.  The outcome is NOT
        ``no_signal``: a probe that could not look must stay distinguishable
        from one that looked and found nothing.
        """
        verdict = await probe_landing(
            _probe_git_ops(
                resolve_branch_sha=AsyncMock(side_effect=RuntimeError('git exploded')),
            ),
            '905', orch_config=_config(tmp_path),
            harness=_harness(read=TaskFound({'metadata': {}})),
        )

        assert verdict.outcome is GitAuthorityOutcome.probe_failed
        assert verdict.merge_sha is None

    async def test_is_fire_safe_when_is_ancestor_raises(self, tmp_path: Path) -> None:
        verdict = await probe_landing(
            _probe_git_ops(
                resolve_branch_sha=AsyncMock(
                    side_effect=lambda b: 'a' * 40 if b.startswith('task/') else _MAIN
                ),
                is_ancestor=AsyncMock(side_effect=RuntimeError('git exploded')),
            ),
            '905', orch_config=_config(tmp_path),
            harness=_harness(read=TaskFound({'metadata': {}})),
        )

        assert verdict.outcome is GitAuthorityOutcome.probe_failed
        assert verdict.merge_sha is None

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
            verdict = await probe_landing(
                _probe_git_ops(
                    resolve_branch_sha=AsyncMock(side_effect=RuntimeError('git exploded')),
                ),
                '4651', orch_config=_config(tmp_path),
                harness=_harness(raises=True), site='landed_reconciler',
            )

        assert verdict.outcome is GitAuthorityOutcome.probe_failed
        assert 'landed_reconciler' in caplog.text, (
            f'the fire-safe warning must name the CALLER, got: {caplog.text!r}'
        )
        assert 'merge_status' not in caplog.text

    async def test_site_reaches_the_metadata_fetch(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The label is threaded THROUGH to ``task_metadata``, not just used
        by the wrapper — a scheduler fault under a reconciler probe is logged
        against the reconciler.
        """
        stub_landing_producer(monkeypatch, 'landed', evidence_sha='c' * 40)
        with caplog.at_level(logging.WARNING):
            verdict = await probe_landing(
                _ancestor_git_ops(tip='a' * 40), '4651',
                orch_config=_config(tmp_path), harness=_harness(raises=True),
                site='landed_reconciler',
            )

        assert verdict.metadata_unavailable is True, 'precondition: the fetch faulted'
        assert 'scheduler.read_task' in caplog.text
        assert 'landed_reconciler' in caplog.text, (
            f'the metadata degradation must name the CALLER, got: {caplog.text!r}'
        )
        assert 'merge_status' not in caplog.text

    async def test_metadata_unavailable_is_threaded_onto_the_verdict(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """THE G7 PAYLOAD, and it can only be pinned here.

        merge_status fails open identically whether the metadata read
        succeeded-and-found-nothing or failed outright, so the distinction is
        invisible through the MCP tool.  The two runs below differ ONLY in
        that one's scheduler raises: the outcome is identical and the flag is
        not.
        """
        stub_landing_producer(monkeypatch, 'landed', evidence_sha='c' * 40)
        healthy = await probe_landing(
            _ancestor_git_ops(tip='a' * 40), '906',
            orch_config=_config(tmp_path), harness=_harness(read=TaskFound({'metadata': {}})),
        )
        faulted = await probe_landing(
            _ancestor_git_ops(tip='a' * 40), '906',
            orch_config=_config(tmp_path), harness=_harness(raises=True),
        )

        assert healthy.outcome is faulted.outcome is GitAuthorityOutcome.found_on_main, (
            'precondition: the fault must not change the OUTCOME, only the flag'
        )
        assert healthy.metadata_unavailable is False
        assert faulted.metadata_unavailable is True


@pytest.fixture
def fresh_landing_tally(monkeypatch: pytest.MonkeyPatch) -> None:
    """Charge the real producer's verdicts to a throwaway tally.

    ``branch_work_landed`` charges the process-lifetime
    ``orchestrator.merge_lane.landing_evidence.LANDING_TALLY`` on every
    verdict; that attribute's own comment says tests replace it for isolation.
    """
    from orchestrator.merge_lane.landing_evidence import LandingTally
    monkeypatch.setattr(
        'orchestrator.merge_lane.landing_evidence.LANDING_TALLY', LandingTally(),
    )


async def _probe_repo(
    repo: Path, tid: str, *, metadata: dict[str, Any] | None = None, read_raises: bool = False,
    commit_citation_pattern: str | None = None,
) -> GitAuthorityVerdict:
    """Probe *tid*'s branch in the real repo at *repo*.

    The task read answers *metadata* (none by default), or raises when
    *read_raises*.  *commit_citation_pattern* is the project's
    ``git.commit_citation_pattern``.
    """
    from orchestrator.git_ops import GitOps
    config = _config(repo, commit_citation_pattern=commit_citation_pattern)
    return await probe_landing(
        GitOps(config.git, repo), tid, orch_config=config,
        harness=_harness(read=TaskFound({'metadata': metadata or {}}), raises=read_raises),
    )


@pytest.mark.asyncio
@pytest.mark.usefixtures('fresh_landing_tally')
class TestProbeLandingPatchIdArm:
    """A live ref that is NOT an ancestor of main, decided in real git.

    ``GitOps.advance_main``'s rebased path leaves exactly this: every branch
    commit rewritten onto main, the ref retained at its pre-rebase tip.  The
    arm delegates the decision to the real ``branch_work_landed``, so these
    run real git rather than stubs.
    """

    async def test_cited_rebased_landing_is_found_on_main_at_the_citing_commit(
        self, tmp_path: Path,
    ) -> None:
        landing = build_rebased_solo_landing(tmp_path / 'repo', '6978', cited=True)

        verdict = await _probe_repo(landing.repo, '6978')

        assert verdict.outcome is GitAuthorityOutcome.found_on_main, verdict
        assert verdict.arm is GitAuthorityArm.patch_id
        assert verdict.merge_sha == verdict.evidence_sha == landing.citing_sha, (
            'merge_sha is the rebased CITING commit on main, never the retained '
            'pre-rebase tip and never main HEAD'
        )
        assert verdict.reason == 'landed'
        assert verdict.citation_gate_skipped is False
        assert verdict.rescued_by_delivered_checks is False
        assert verdict.merge_sha_fully_guarded is True

    async def test_uncited_rebased_landing_is_found_on_main_at_the_tips_patch_id_twin(
        self, tmp_path: Path,
    ) -> None:
        landing = build_rebased_solo_landing(tmp_path / 'repo', '6978', cited=False)

        verdict = await _probe_repo(landing.repo, '6978')

        assert verdict.outcome is GitAuthorityOutcome.found_on_main, verdict
        assert verdict.arm is GitAuthorityArm.patch_id
        assert verdict.merge_sha == landing.main_tip, (
            "with no citing commit, merge_sha is the branch tip's patch-id twin on main"
        )
        assert verdict.merge_sha != landing.branch_tip

    async def test_unlanded_live_branch_is_no_signal(self, tmp_path: Path) -> None:
        repo = tmp_path / 'repo'
        new_repo(repo)
        add_unlanded_branch(repo, '6978')

        verdict = await _probe_repo(repo, '6978')

        assert verdict.outcome is GitAuthorityOutcome.no_signal, verdict
        assert verdict.merge_sha is None
        assert verdict.arm is GitAuthorityArm.patch_id
        assert verdict.reason == 'not_landed'

    async def test_commit_added_after_the_landing_is_no_signal(self, tmp_path: Path) -> None:
        """A branch commit main lacks keeps the arm from answering found_on_main.

        Such a commit reads '+' in ``git cherry``, and so does a genuine
        landing whose context drifted (task 6936).  Either way the answer is
        ``no_signal``: here the arm under-reports.
        """
        landing = build_rebased_solo_landing(tmp_path / 'repo', '6978', cited=True)
        landing_git(landing.repo, 'checkout', 'task/6978')
        landing_commit(landing.repo, 'amend: after the landing', 'after.txt')
        landing_git(landing.repo, 'checkout', 'main')

        verdict = await _probe_repo(landing.repo, '6978')

        assert verdict.outcome is GitAuthorityOutcome.no_signal, verdict
        assert verdict.merge_sha is None
        assert verdict.reason == 'not_landed'

    async def test_contained_but_unattributable_is_landed_unconfirmed(
        self, tmp_path: Path,
    ) -> None:
        """Containment proven, attribution refused: a landing, not a miss.

        Main holds two commits sharing the tip's patch-id and subject, so
        ``git cherry`` reports every branch commit contained while the twin
        lookup refuses to guess between them.
        """
        landing = build_rebased_solo_landing(tmp_path / 'repo', '6978', cited=False)
        reapply_tip_on_main(landing)

        verdict = await _probe_repo(landing.repo, '6978')

        assert verdict.outcome is GitAuthorityOutcome.landed_unconfirmed, verdict
        assert verdict.arm is GitAuthorityArm.patch_id
        assert verdict.reason == 'no_attribution'
        assert verdict.merge_sha is None
        assert verdict.merge_sha_fully_guarded is False

    async def test_a_detector_that_cannot_look_is_probe_failed_and_names_the_site(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """``git_error`` is a claim about the detector, never a negative.

        The stub answers the two calls the dispatch makes and nothing else,
        so the real ``branch_work_landed`` hits a missing primitive and
        contains it as ``git_error``.
        """
        git_ops = types.SimpleNamespace(
            resolve_branch_sha=AsyncMock(
                side_effect=lambda b: 'a' * 40 if b.startswith('task/') else _MAIN,
            ),
            is_ancestor=AsyncMock(return_value=False),
        )

        with caplog.at_level(logging.WARNING):
            verdict = await probe_landing(
                git_ops, '6978', orch_config=_config(tmp_path),
                harness=_harness(read=TaskFound({'metadata': {}})), site='landed_reconciler',
            )

        assert verdict.outcome is GitAuthorityOutcome.probe_failed, verdict
        assert verdict.merge_sha is None
        assert verdict.arm is GitAuthorityArm.patch_id
        assert verdict.reason == 'git_error'
        assert 'landed_reconciler' in caplog.text
        assert 'merge_status' not in caplog.text

    async def test_a_metadata_fault_reaches_accept_and_reject_verdicts(
        self, tmp_path: Path,
    ) -> None:
        landed = build_rebased_solo_landing(tmp_path / 'landed', '6978', cited=True)
        unlanded = tmp_path / 'unlanded'
        new_repo(unlanded)
        add_unlanded_branch(unlanded, '6978')

        accepted = await _probe_repo(landed.repo, '6978', read_raises=True)
        rejected = await _probe_repo(unlanded, '6978', read_raises=True)

        assert accepted.outcome is GitAuthorityOutcome.found_on_main, accepted
        assert rejected.outcome is GitAuthorityOutcome.no_signal, rejected
        assert accepted.arm is rejected.arm is GitAuthorityArm.patch_id
        assert accepted.metadata_unavailable is True
        assert rejected.metadata_unavailable is True

    async def test_a_net_empty_branch_is_a_no_op_landing_and_no_signal(
        self, tmp_path: Path,
    ) -> None:
        """The empty-net-diff guard stands between vacuous containment and ``done``."""
        repo = tmp_path / 'repo'
        new_repo(repo)
        add_net_empty_branch(repo, '6978')

        verdict = await _probe_repo(repo, '6978')

        assert verdict.outcome is GitAuthorityOutcome.no_signal, verdict
        assert verdict.arm is GitAuthorityArm.patch_id
        assert verdict.reason == 'no_op_landing'
        assert verdict.merge_sha is None

    async def test_a_degenerate_branch_is_no_signal(self, tmp_path: Path) -> None:
        """A tip still equal to the recorded ``branch_base_sha`` delivered nothing."""
        repo = tmp_path / 'repo'
        new_repo(repo)
        tip = add_unlanded_branch(repo, '6978')

        verdict = await _probe_repo(repo, '6978', metadata={'branch_base_sha': tip})

        assert verdict.outcome is GitAuthorityOutcome.no_signal, verdict
        assert verdict.arm is GitAuthorityArm.patch_id
        assert verdict.reason == 'degenerate_branch'

    async def test_main_merged_into_branch_without_recorded_base_is_a_no_op_landing(
        self, tmp_path: Path,
    ) -> None:
        """``git cherry`` prints nothing here, and the no-op guard catches it.

        With no recorded base the guard measures from merge-base(main, tip),
        which is main's tip, and the branch adds nothing beyond it.
        """
        repo = tmp_path / 'repo'
        new_repo(repo)
        add_main_merged_branch(repo, '6978', earlier_citation=True)

        verdict = await _probe_repo(repo, '6978')

        assert verdict.outcome is GitAuthorityOutcome.no_signal, verdict
        assert verdict.reason == 'no_op_landing'

    async def test_main_merged_into_branch_with_recorded_base_is_not_a_landing(
        self, tmp_path: Path,
    ) -> None:
        """The same branch, with the creation point every dispatched task records.

        Empty ``git cherry`` output carries no content proof: the branch has
        no non-merge commit beyond main, so nothing of its own is shown to be
        on main and it is not a landing, whatever an earlier round cited.
        """
        repo = tmp_path / 'repo'
        new_repo(repo)
        base = add_main_merged_branch(repo, '6978', earlier_citation=True)

        verdict = await _probe_repo(repo, '6978', metadata={'branch_base_sha': base})

        assert verdict.outcome is GitAuthorityOutcome.no_signal, verdict
        assert verdict.reason == 'not_landed'
        assert verdict.merge_sha is None

    async def test_an_opted_out_project_answers_at_the_patch_id_twin_not_the_citing_commit(
        self, tmp_path: Path,
    ) -> None:
        landing = build_rebased_solo_landing(tmp_path / 'repo', '6978', cited=True)

        verdict = await _probe_repo(landing.repo, '6978', commit_citation_pattern='')

        assert verdict.outcome is GitAuthorityOutcome.found_on_main, verdict
        assert verdict.arm is GitAuthorityArm.patch_id
        assert verdict.merge_sha == landing.main_tip
        assert verdict.merge_sha != landing.citing_sha
        assert verdict.merge_sha_fully_guarded is True
        assert verdict.citation_gate_skipped is False

    async def test_a_project_override_pattern_decides_the_patch_id_arms_citation(
        self, tmp_path: Path,
    ) -> None:
        landing = build_rebased_solo_landing(tmp_path / 'repo', '6978', cited=False)
        squash = landing_commit(landing.repo, 'Fix an unrelated thing (#6978)', 'squash.txt')

        overridden = await _probe_repo(
            landing.repo, '6978', commit_citation_pattern=r'^\[{tid}\]',
        )
        default = await _probe_repo(landing.repo, '6978')

        assert overridden.merge_sha == landing.main_tip, overridden
        assert default.merge_sha == squash, default


def _write_and_commit(root: Path, subject: str, files: dict[str, str]) -> str:
    for name, body in files.items():
        (root / name).write_text(body)
    landing_git(root, 'add', *files)
    landing_git(root, 'commit', '-m', subject)
    return landing_git(root, 'rev-parse', 'HEAD')


def _lines(stem: str, count: int) -> str:
    return ''.join(f'{stem} line {n}\n' for n in range(count))


def _merged_landing_then_same_path_churn(root: Path, task_id: str) -> tuple[str, str]:
    """``task/<task_id>`` merged ``--no-ff`` into main, then main rewrites its paths.

    Three later main commits each replace every line the task added, which is
    the churn a decaying effect-present check reads as "reverted".  Returns
    ``(merge_sha, branch_base_sha)``.
    """
    branch = f'task/{task_id}'
    new_repo(root)
    base = landing_git(root, 'rev-parse', 'main')
    landing_git(root, 'checkout', '-b', branch)
    _write_and_commit(root, f'feat({task_id}): add the module', {'work.py': _lines('feature', 40)})
    _write_and_commit(root, f'test({task_id}): cover the module', {'helper.py': _lines('helper', 30)})
    landing_git(root, 'checkout', 'main')
    landing_git(root, 'merge', '--no-ff', '-m', f'Merge {branch} into main', branch)
    merge_sha = landing_git(root, 'rev-parse', 'main')
    for n in range(3):
        _write_and_commit(root, f'refactor: churn {n}', {
            'work.py': _lines(f'churn {n}', 40), 'helper.py': _lines(f'churn {n}', 30),
        })
    return merge_sha, base


def _spy_on_landing_producers(monkeypatch: pytest.MonkeyPatch) -> tuple[AsyncMock, AsyncMock]:
    """Delegating spies on the DEFINING module's two landing producers.

    ``git_authority`` imports both per call, so patching
    ``orchestrator.merge_lane.landing_evidence`` is what its arms resolve.
    Returns ``(branch_work_landed, validate_landing_evidence)``.
    """
    from orchestrator.merge_lane import landing_evidence
    work_landed = AsyncMock(wraps=landing_evidence.branch_work_landed)
    validate = AsyncMock(wraps=landing_evidence.validate_landing_evidence)
    monkeypatch.setattr(landing_evidence, 'branch_work_landed', work_landed)
    monkeypatch.setattr(landing_evidence, 'validate_landing_evidence', validate)
    return work_landed, validate


@pytest.mark.asyncio
@pytest.mark.usefixtures('fresh_landing_tally')
class TestProbeLandingAncestorArmDecidesByBranchWorkLanded:
    """A live tip that IS an ancestor of main, decided by the non-decaying producer.

    Task 4648 repoints the ancestor arm from ``validate_landing_evidence``'s
    effect-present DISCOVERY policy to ``branch_work_landed``, the same
    producer the patch-id arm uses.  Its degeneracy guard and the ``''``
    citation opt-out still run first.  Which policy each landing-detection
    site keeps is recorded in
    ``docs/prds/landed-not-done-recovery.effect-present-caller-enumeration.md`` §7.
    """

    async def test_a_merged_landing_survives_same_path_churn(self, tmp_path: Path) -> None:
        merge_sha, base = _merged_landing_then_same_path_churn(tmp_path / 'repo', '4648')

        verdict = await _probe_repo(tmp_path / 'repo', '4648', metadata={'branch_base_sha': base})

        assert verdict.outcome is GitAuthorityOutcome.found_on_main, verdict
        assert verdict.arm is GitAuthorityArm.ancestor
        assert verdict.merge_sha == merge_sha
        assert verdict.merge_sha_fully_guarded is True
        assert verdict.rescued_by_delivered_checks is False

    async def test_a_reseeded_tip_matched_only_by_a_later_subject_is_unconfirmed(
        self, tmp_path: Path,
    ) -> None:
        """esc-4648-5: a later main commit repeating the tip's subject is not its landing."""
        repo = tmp_path / 'repo'
        new_repo(repo)
        base = landing_git(repo, 'rev-parse', 'main')
        foreign = landing_commit(repo, 'chore: tidy the census', 'census.md')
        landing_git(repo, 'branch', 'task/77', foreign)
        landing_commit(repo, 'chore: tidy the census', 'census-next.md')

        verdict = await _probe_repo(repo, '77', metadata={'branch_base_sha': base})

        assert verdict.outcome is GitAuthorityOutcome.landed_unconfirmed, verdict
        assert verdict.arm is GitAuthorityArm.ancestor
        assert verdict.merge_sha is None

    async def test_the_ancestor_arm_asks_branch_work_landed_not_the_effect_present_gate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _merge_sha, base = _merged_landing_then_same_path_churn(tmp_path / 'repo', '4648')
        metadata = {'branch_base_sha': base}
        work_landed, validate = _spy_on_landing_producers(monkeypatch)

        await _probe_repo(tmp_path / 'repo', '4648', metadata=metadata)

        work_landed.assert_awaited_once()
        assert work_landed.await_args is not None
        assert work_landed.await_args.kwargs['pattern_template'] is None, (
            "the project's git.commit_citation_pattern, None here, is forwarded as-is"
        )
        assert work_landed.await_args.kwargs['metadata'] == metadata
        validate.assert_not_awaited()

    async def test_the_citation_opt_out_still_answers_at_the_tip_without_the_producer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _merge_sha, base = _merged_landing_then_same_path_churn(tmp_path / 'repo', '4648')
        tip = landing_git(tmp_path / 'repo', 'rev-parse', 'task/4648')
        work_landed, _validate = _spy_on_landing_producers(monkeypatch)

        verdict = await _probe_repo(
            tmp_path / 'repo', '4648', metadata={'branch_base_sha': base},
            commit_citation_pattern='',
        )

        assert verdict.outcome is GitAuthorityOutcome.found_on_main, verdict
        assert verdict.arm is GitAuthorityArm.ancestor
        assert verdict.merge_sha == tip
        assert verdict.citation_gate_skipped is True
        work_landed.assert_not_awaited()

    async def test_a_degenerate_ancestor_tip_is_no_signal(self, tmp_path: Path) -> None:
        repo = tmp_path / 'repo'
        new_repo(repo)
        base = landing_git(repo, 'rev-parse', 'main')
        landing_git(repo, 'branch', 'task/4648', base)
        landing_commit(repo, 'chore: main moves on', 'later.txt')

        verdict = await _probe_repo(repo, '4648', metadata={'branch_base_sha': base})

        assert verdict.outcome is GitAuthorityOutcome.no_signal, verdict
        assert verdict.merge_sha is None

    async def test_a_raising_producer_degrades_to_probe_failed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _merge_sha, base = _merged_landing_then_same_path_churn(tmp_path / 'repo', '4648')
        monkeypatch.setattr(
            'orchestrator.merge_lane.landing_evidence.branch_work_landed',
            AsyncMock(side_effect=RuntimeError('producer exploded')),
        )

        verdict = await _probe_repo(tmp_path / 'repo', '4648', metadata={'branch_base_sha': base})

        assert verdict.outcome is GitAuthorityOutcome.probe_failed, verdict
        assert verdict.merge_sha is None


def _merged_landing_then_ref_deleted(root: Path, task_id: str) -> tuple[str, str]:
    """``task/<task_id>`` merged ``--no-ff`` into main, then its ref deleted.

    Only the ``Merge task/<task_id> into main`` marker is left to find.
    Returns ``(merge_sha, branch_base_sha)``.
    """
    branch = f'task/{task_id}'
    new_repo(root)
    base = landing_git(root, 'rev-parse', 'main')
    landing_git(root, 'checkout', '-b', branch)
    _write_and_commit(root, f'feat({task_id}): add the module', {'work.py': _lines('feature', 40)})
    landing_git(root, 'checkout', 'main')
    landing_git(root, 'merge', '--no-ff', '-m', f'Merge {branch} into main', branch)
    landing_git(root, 'branch', '-D', branch)
    return landing_git(root, 'rev-parse', 'main'), base


@pytest.mark.asyncio
@pytest.mark.usefixtures('fresh_landing_tally')
class TestProbeLandingMarkerArmKeepsTheCandidatePolicy:
    async def test_a_deleted_ref_marker_is_checked_as_the_candidate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The marker arm stays on ``validate_landing_evidence`` CANDIDATE (task 4648).

        A deleted ref leaves no live tip for ``branch_work_landed`` to judge,
        so the marker itself is the candidate the effect-present policy
        checks, and the producer the live-ref arms use is never asked.  The
        ruling is recorded in
        ``docs/prds/landed-not-done-recovery.effect-present-caller-enumeration.md`` §7.
        """
        merge_sha, base = _merged_landing_then_ref_deleted(tmp_path / 'repo', '4648')
        work_landed, validate = _spy_on_landing_producers(monkeypatch)

        verdict = await _probe_repo(tmp_path / 'repo', '4648', metadata={'branch_base_sha': base})

        assert verdict.outcome is GitAuthorityOutcome.found_on_main, verdict
        assert verdict.arm is GitAuthorityArm.marker
        validate.assert_awaited_once()
        assert validate.await_args is not None
        assert validate.await_args.kwargs['candidate_sha'] == merge_sha
        work_landed.assert_not_awaited()


class TestGitAuthorityVerdict:
    def test_is_frozen(self) -> None:
        verdict = GitAuthorityVerdict(outcome=GitAuthorityOutcome.no_signal)

        with pytest.raises(dataclasses.FrozenInstanceError):
            verdict.merge_sha = 'x' * 40   # type: ignore[misc]

    @pytest.mark.parametrize(
        'verdict,expected,label',
        [
            (
                GitAuthorityVerdict(
                    outcome=GitAuthorityOutcome.found_on_main,
                    merge_sha='c' * 40, arm=GitAuthorityArm.ancestor,
                ),
                True,
                'found_on_main with neither weakening — the only stampable case',
            ),
            (
                GitAuthorityVerdict(
                    outcome=GitAuthorityOutcome.found_on_main,
                    merge_sha='c' * 40, arm=GitAuthorityArm.patch_id,
                ),
                True,
                'patch-id path: the PRD D1 contract checks containment and '
                'deliberately not effect survival — not a weakening',
            ),
            (
                GitAuthorityVerdict(
                    outcome=GitAuthorityOutcome.found_on_main,
                    merge_sha='a' * 40, arm=GitAuthorityArm.ancestor,
                    citation_gate_skipped=True,
                ),
                False,
                'merge_sha is a raw branch tip, never discovered on main',
            ),
            (
                GitAuthorityVerdict(
                    outcome=GitAuthorityOutcome.found_on_main,
                    merge_sha='c' * 40, arm=GitAuthorityArm.ancestor,
                    rescued_by_delivered_checks=True,
                ),
                False,
                "merge_sha is on main but its effect is absent at main HEAD",
            ),
            (
                GitAuthorityVerdict(
                    outcome=GitAuthorityOutcome.found_on_main,
                    merge_sha='a' * 40, arm=GitAuthorityArm.ancestor,
                    citation_gate_skipped=True, rescued_by_delivered_checks=True,
                ),
                False,
                'both weakenings at once',
            ),
            (
                GitAuthorityVerdict(
                    outcome=GitAuthorityOutcome.landed_unconfirmed,
                    arm=GitAuthorityArm.ancestor, evidence_sha='c' * 40,
                ),
                False,
                'landed_unconfirmed — no merge_sha to stamp at all',
            ),
            (
                GitAuthorityVerdict(outcome=GitAuthorityOutcome.no_signal),
                False,
                'no_signal — nothing established',
            ),
            (
                GitAuthorityVerdict(outcome=GitAuthorityOutcome.probe_failed),
                False,
                'probe_failed — the probe could not look',
            ),
        ],
        ids=[
            'fully_guarded', 'patch_id_arm', 'citation_gate_skipped', 'rescued',
            'both_weakenings', 'landed_unconfirmed', 'no_signal', 'probe_failed',
        ],
    )
    def test_merge_sha_fully_guarded(
        self, verdict: GitAuthorityVerdict, expected: bool, label: str
    ) -> None:
        """ONE question a provenance writer asks instead of enumerating
        weakenings.

        Pinned on hand-constructed verdicts so the property is independent
        of probe plumbing: adding a THIRD weakening must change this one
        expression, and this table is what fails if a future edit adds the
        field but forgets to fold it in.  Non-``found_on_main`` outcomes are
        False because there is no sha to stamp — NOT a claim that the branch
        did not land.
        """
        assert verdict.merge_sha_fully_guarded is expected, label

    def test_weakening_flags_default_to_the_fully_guarded_reading(self) -> None:
        """Polarity pin: both flags mean A GUARANTEE WAS WEAKENED.

        So a hand-constructed verdict — leaf eta's writer, task 4831's
        mapper — is SAFE by default, and a future field spelled with the
        opposite polarity (``effect_present_confirmed``) would make the
        dataclass default assert the weakened case.
        """
        verdict = GitAuthorityVerdict(outcome=GitAuthorityOutcome.found_on_main)

        assert verdict.citation_gate_skipped is False
        assert verdict.rescued_by_delivered_checks is False


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
