"""The two invariants the write_triage flip gate rests on, pinned executably.

Neither pin here is prose. One EXECUTES ``select_judge_candidates`` and reads
the slate it returns; the other validates a real ``delivered_checks``
descriptor against a real path on disk. That distinction is the whole point of
the file: the defect it exists to prevent was a claim about behaviour that
lived only in a task description, where nothing could contradict it.

The refuted claim is that ``candidates[0]`` is the attach target. It is not,
on the hoisted path. ``select_judge_candidates``' rescue arm APPENDS the
band's winner rather than promoting it --
``selected = [*selected[: max(n - 1, 0)], winner]`` -- so a winner that fell
outside the top *n* lands LAST, not first, and the slate stays exactly *n*
long because the rescue EVICTS rather than widens. Worse, when the winner is a
hoisted parent that never appeared as a result of its own, the rescued record
is the CHILD carrying the evidence, and the canonical id is absent from the
returned slate entirely.

The full narrative -- the measurement, the corrected requirement, the
descriptor swap and the 4762/4810/3169 interlock -- is in
``plans/write-triage-attach-target-contradiction.md``.

No test in this file needs an API key, a network, or Qdrant.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from shared.capability_manifest import DeliveredCheckMeta

from fused_memory.models.enums import MemoryCategory, SourceStore
from fused_memory.models.memory import MemoryResult
from fused_memory.server.grouped_read import PARENT_ID_KEY
from fused_memory.server.write_triage_judge import select_judge_candidates
from fused_memory.services.memory_service import RRF_K

# A post-RRF rank-1 relevance score. Spelled out for the same reason
# ``test_write_triage_judge.py`` spells it out: `relevance_score` is NOT the
# cosine, and a test that passed by reading it would pass for the wrong
# reason.
_RRF_RANK1 = 1.0 / (RRF_K + 1)


def _result(
    id_: str,
    score: float,
    *,
    content: str = 'some procedural content',
    relevance_score: float = _RRF_RANK1,
    extra_metadata: dict | None = None,
) -> MemoryResult:
    """A POST-RRF ``MemoryResult``: *score* is the COSINE, in metadata.

    Same shape as ``test_write_triage_judge.py::_result`` -- kept local rather
    than imported across suites, matching how the triage suites already stand
    alone, and deliberately NOT imported from a file task 4762 is about to
    rewrite.
    """
    metadata: dict = {'store_rank': 1, 'store_score': score}
    if extra_metadata:
        metadata.update(extra_metadata)
    return MemoryResult(
        id=id_,
        content=content,
        category=MemoryCategory.procedural_knowledge,
        source_store=SourceStore.mem0,
        relevance_score=relevance_score,
        metadata=metadata,
    )


def _hoisted_slate() -> list[MemoryResult]:
    """Six ordinary results, plus one child whose hoisted parent is absent.

    ``child-1`` scores BELOW every ``m*``, so it falls outside a top-3 window
    and can only enter the slate through the rescue arm. This is precisely the
    shape task 4762's own frozen step-13 produces when it rescores the
    evidence child below the cut.
    """
    results = [_result(f'm{i}', 0.90 - i / 100) for i in range(6)]
    results.append(_result('child-1', 0.60, extra_metadata={PARENT_ID_KEY: 'parent-1'}))
    return results


class TestAttachTargetIsNotAlwaysFirst:
    """Position does not encode the attach target, so a prompt must not say it does.

    Every assertion here is on opaque ids and positions. Nothing asserts on
    prompt WORDING -- that class of source-text meta-test is the one task
    3128 steps 23-25 deleted, and the norm against re-adding it is recorded on
    task 4762.
    """

    def test_a_hoisted_attach_target_lands_LAST_not_first(self) -> None:
        """Refutes "candidates[0] is the attach target": here it is candidates[-1]."""
        selected = select_judge_candidates(_hoisted_slate(), 3, canonical_id='parent-1')

        assert [r.id for r in selected] == ['m0', 'm1', 'child-1']
        # The rescue EVICTS the weakest in-window candidate; it does not widen
        # the slate, so a caller cannot detect the rescue from the length.
        assert len(selected) == 3
        assert selected[0].id != 'child-1'
        assert selected[-1].id == 'child-1'

    def test_an_ordinary_band_winner_does_land_first(self) -> None:
        """The CONTROL: on the ordinary path the winner IS first.

        The point is not that position is always wrong -- it is that the two
        paths DISAGREE, which is what makes position unsound as an encoding.
        """
        selected = select_judge_candidates(_hoisted_slate(), 3, canonical_id='m0')

        assert [r.id for r in selected] == ['m0', 'm1', 'm2']
        assert selected[0].id == 'm0'

    def test_the_hoisted_canonical_id_is_absent_from_the_slate_entirely(self) -> None:
        """A naive ``r.id == canonical_id`` marker would mark NOTHING here.

        The rescued record is the child that carried the evidence, reachable
        only through ``PARENT_ID_KEY``. Any fix that marks the attach target
        must match on id OR parent id.
        """
        selected = select_judge_candidates(_hoisted_slate(), 3, canonical_id='parent-1')

        assert 'parent-1' not in [r.id for r in selected]
        assert not [r for r in selected if r.id == 'parent-1']

        by_parent = [
            r for r in selected if (r.metadata or {}).get(PARENT_ID_KEY) == 'parent-1'
        ]
        assert [r.id for r in by_parent] == ['child-1']


#: This file sits at ``<root>/fused-memory/tests/server/``.
_REPO_ROOT = Path(__file__).parents[3]

#: The ``metadata.delivered_checks`` entry task 4822 installed on task 4762,
#: replacing a ``{kind: 'grep', pattern: 'candidate_id', expect: 'present'}``
#: descriptor that asserted a symbol NAME rather than a behaviour -- and that
#: named the option the corrected plan does not take.
#:
#: This is byte-for-byte the same ``script``/``args``/``timeout_secs`` as task
#: 3169's ``metadata.before_done`` predicate. That is the point: the invariant
#: then has ONE encoding referenced from TWO enforcement points -- the
#: dispatch-time delivered-checks gate and the before_done gate -- instead of
#: two independent encodings that can drift apart, which is exactly how the
#: contradiction this file records came about.
#:
#: ``metadata.delivered_checks`` is invisible to git, so without this constant
#: the swap would have no auditable home in the repo.
FLIP_GATE_DELIVERED_CHECK = {
    'name': 'write_triage_pre_flip_preconditions_on_main',
    'kind': 'script',
    'script': 'scripts/check_write_triage_flip_preconditions.sh',
    'args': [],
    'timeout_secs': 120,
}


class TestFlipGateDeliveredCheckDescriptor:
    """The installed descriptor, pinned against the real tree.

    ``run_delivered_check`` degrades a malformed or unreachable ``script``
    descriptor to ``DeliveredCheckResult.ERRORED``, and per
    ``docs/task-authoring.md`` section 3.3 an ERRORED check is a fail-safe
    wait with NO streak bump and NO escalation. It therefore holds the
    dependent SILENTLY and indefinitely -- the one new failure mode the swap
    to ``kind='script'`` introduces, and the reason these three tests exist.
    """

    def test_the_repo_root_resolves(self) -> None:
        """Fail loudly if this file moves, rather than checking the wrong tree."""
        assert (_REPO_ROOT / 'dark-factory-orchestrator.yaml').exists()

    def test_the_descriptor_validates_as_a_DeliveredCheckMeta(self) -> None:
        """A descriptor that does not validate is a silent, unescalated hold."""
        meta = DeliveredCheckMeta(**FLIP_GATE_DELIVERED_CHECK)

        assert meta.kind == 'script'
        assert meta.script == 'scripts/check_write_triage_flip_preconditions.sh'
        assert meta.args == []
        assert meta.timeout_secs == 120

    def test_a_zero_timeout_descriptor_is_rejected(self) -> None:
        """The cross-field validator is what makes the assertion above load-bearing."""
        with pytest.raises(ValueError, match='timeout_secs is required and must be > 0'):
            DeliveredCheckMeta(**{**FLIP_GATE_DELIVERED_CHECK, 'timeout_secs': 0})

    def test_the_named_script_exists_and_is_executable(self) -> None:
        """``_run_script_check`` execs ``project_root / meta.script`` directly.

        A rename, a deletion, or a lost exec bit is the same silent hold as a
        malformed descriptor. This does NOT duplicate task 4810's
        ``scripts/tests/`` suite: that pins the script's exit-code BEHAVIOUR,
        this pins that the descriptor's path still resolves to it.
        """
        script = _REPO_ROOT / FLIP_GATE_DELIVERED_CHECK['script']

        assert script.exists(), f'delivered-check script is missing: {script}'
        assert os.access(script, os.X_OK), f'delivered-check script is not executable: {script}'

    def test_the_descriptor_is_not_a_bare_symbol_grep(self) -> None:
        """Fails the moment someone reintroduces the symbol-grep shape 4822 removed."""
        assert FLIP_GATE_DELIVERED_CHECK['kind'] == 'script'
        assert 'pattern' not in FLIP_GATE_DELIVERED_CHECK
        assert 'expect' not in FLIP_GATE_DELIVERED_CHECK
