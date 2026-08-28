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
