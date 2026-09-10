"""Tests for the deterministic auto-consolidation predicate + canonical builder.

Task 5237, PRD `plans/memory-auto-consolidation-prd.md` contracts C2 (the pure
predicate) and C3 (the canonical text builder).

Every assertion here RUNS the code under test against fixture inputs and
asserts on the returned verdict or string. None inspects source text, greps for
a symbol, or pins English phrasing — with the single deliberate exception of
``TestBuildAutoCanonical.test_renders_the_prd_template_verbatim``, where the
template IS the contract and duplicating it once in the test is the only way to
notice drift in its wording, backticks or punctuation.
"""

from __future__ import annotations

import dataclasses
import inspect
import subprocess
import sys
import uuid
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pytest

from fused_memory.config.schema import FusedMemoryConfig
from fused_memory.reconciliation.consolidation_auto import (
    CORRECTION_METADATA_KEYS,
    UNREADABLE,
    AutoOutcome,
    AutoProposal,
    AutoReasonCode,
    AutoVerdict,
    build_auto_canonical,
    evaluate_auto_predicate,
)
from fused_memory.topic_slug import TOPIC_SLUG_MAX_LEN

#: The topic every predicate fixture proposes unless a test says otherwise.
TOPIC = 'memory-consolidation'

#: A member body that trips no hazard: no correction banner, no index label.
#: Fixtures that need a hazard say so explicitly, so a test that fails is
#: always failing on the one fact it names.
BENIGN_BODY = (
    'The scheduler drains its queue before the watchdog fires, so a wedged unit is '
    'revived without any operator action.'
)


def _member(
    member_id: str,
    *,
    topic: str | None = None,
    canonical: bool = False,
    category: str | None = 'procedural_knowledge',
    content: str = BENIGN_BODY,
    **extra_meta: Any,
) -> dict[str, Any]:
    """Build one record in the shape ``MemoryService.get_memory_by_id`` returns.

    That shape is the plain ``{'id', 'content', 'metadata'}`` dict — there is no
    ``MemoryRecord`` class in the tree — so these fixtures are the real thing
    rather than a stand-in whose drift from the source would go unnoticed.

    An unstamped member carries no ``metadata.topic`` KEY at all (not a ``None``
    value), and a non-canonical member carries no ``canonical`` key, because
    that is how live records are shaped. ``extra_meta`` is how a test adds a
    correction-metadata key without a second builder.
    """
    metadata: dict[str, Any] = dict(extra_meta)
    if category is not None:
        metadata['category'] = category
    if topic is not None:
        metadata['topic'] = topic
    if canonical:
        metadata['canonical'] = True
    return {'id': member_id, 'content': content, 'metadata': metadata}


def _members(*records: dict[str, Any]) -> dict[str, Any]:
    """Key each record by its OWN id — the mapping the predicate is handed.

    Keying from the record keeps the mapping key and ``record['id']`` from ever
    disagreeing, which a hand-written literal would let drift silently.

    The value type is the predicate's own: a record, ``None`` when the store
    has no such id, or :data:`UNREADABLE` when the read did not answer. Tests
    inject the latter two by assigning over a key, so a narrower annotation
    here would contradict the very fixtures it exists to build.
    """
    return {record['id']: record for record in records}


def _proposal(
    member_ids: tuple[str, ...],
    *,
    topic: str = TOPIC,
    category: str = 'procedural_knowledge',
) -> AutoProposal:
    """Build the structured proposal task delta assembles from a ledger row."""
    return AutoProposal(topic=topic, member_ids=tuple(member_ids), category=category)


def _codes(verdict: AutoVerdict) -> list[AutoReasonCode]:
    """Every reason code the verdict gave, in order."""
    return [reason.code for reason in verdict.reasons]


def _reasons_for(verdict: AutoVerdict, code: AutoReasonCode) -> list[Any]:
    """The reasons carrying *code* — a list, because collecting is the contract.

    Asking for a code and getting back a list is deliberate: the hazard arm
    reports EVERY offender, so a test that wants "the one reason" says so by
    asserting the length rather than by silently reading the first.
    """
    return [reason for reason in verdict.reasons if reason.code is code]


def _judge(
    members: Mapping[str, Any],
    *,
    proposal_ids: Sequence[str] | None = None,
    topic: str = TOPIC,
    canonical_count: int | None = 0,
    open_gate_id: str | None = None,
    existing_canonical_slugs: Sequence[str] = (),
    config: Any = None,
) -> AutoVerdict:
    """Run the predicate over *members*, every unmentioned fact at its benign value.

    The defaults are the assertion technique: a test that names only its one
    hazard is thereby claiming that hazard is what decided the verdict, because
    nothing else it left unsaid could have. Proposing the mapping's own keys by
    default keeps the proposal and the reads from drifting apart.
    """
    ids = tuple(proposal_ids) if proposal_ids is not None else tuple(members)
    return evaluate_auto_predicate(
        _proposal(ids, topic=topic),
        members=members,
        canonical_count=canonical_count,
        open_gate_id=open_gate_id,
        existing_canonical_slugs=existing_canonical_slugs,
        config=config or _auto_config(),
    )


def _auto_config(**overrides: Any):
    """A REAL ``ConsolidationAutoConfig``, never a bare ``MagicMock``.

    The model IS the contract: its defaults, its bounds and its cross-field
    validator are what the predicate reads, so a mock would let this suite pass
    against knobs that cannot exist (the ``scripts/check_bare_magicmock_config.py``
    guard, task 1372).
    """
    config = FusedMemoryConfig().consolidation_auto
    return config.model_copy(update=overrides) if overrides else config


class TestImportLeafAndSingleHomes:
    """``reconciliation/consolidation_auto.py`` is a STDLIB-ONLY import leaf.

    Its stated precedent is ``reconciliation/consolidation_gate.py``, but this
    module holds a STRICTLY STRONGER property, which is the point of the test
    below. MEASURED on this branch: ``import
    fused_memory.reconciliation.consolidation_gate`` pulls 1030 modules,
    including pydantic, yaml, mem0 and ``fused_memory.config.schema`` — the
    last three arriving via ``memory_metadata`` -> ``backends.mem0_client`` ->
    ``config.schema``. ``import fused_memory.reconciliation`` alone pulls 38
    and none of them. So this module can hold ``topic_slug``'s
    ``test_module_is_import_light`` property that the gate cannot, and that is
    exactly what makes its ``ConsolidationAutoConfig`` reference a
    ``TYPE_CHECKING``-only import rather than a runtime one.

    PRD D4 records a MEASURED hard import cycle from a careless import of
    exactly this kind (``config/schema.py`` -> ``memory_metadata`` ->
    ``backends.mem0_client`` -> ``config.schema``, raising ``ImportError:
    cannot import name 'FusedMemoryConfig'``). A cycle is DIRECTIONAL — the
    measured failure only surfaced from one side — so the two orders below are
    two different probes, not one probe run twice.

    Every probe runs in a FRESH interpreter, because this test process has
    already imported everything at collection and an in-process ``sys.modules``
    check would pass vacuously.
    """

    #: Heavy modules the leaf must never pull in — the precedent's four.
    #: ``targeted``/``harness`` are the reconciliation runtime;
    #: ``services.memory_service`` and ``server.tools`` are the store and MCP
    #: layers.
    FORBIDDEN = (
        'fused_memory.reconciliation.targeted',
        'fused_memory.reconciliation.harness',
        'fused_memory.services.memory_service',
        'fused_memory.server.tools',
    )

    #: The strictly stronger set: nothing that would make this module's import
    #: cost a config load, a pydantic build or the mem0 SDK.
    HEAVY = (
        'pydantic',
        'yaml',
        'mem0',
        'fused_memory.config.schema',
        'fused_memory.memory_metadata',
    )

    @staticmethod
    def _probe(body: str):
        return subprocess.run(
            [sys.executable, '-c', body], capture_output=True, text=True, timeout=300,
        )

    def test_module_imports_stay_leaf(self):
        """Importing the leaf alone must not drag in the reconciliation
        runtime, the memory service or the MCP tool layer."""
        forbidden = ', '.join(repr(m) for m in self.FORBIDDEN)
        result = self._probe(
            'import sys\n'
            'import fused_memory.reconciliation.consolidation_auto  # noqa: F401\n'
            f'forbidden = [{forbidden}]\n'
            'present = [m for m in forbidden if m in sys.modules]\n'
            'assert not present, present\n',
        )
        assert result.returncode == 0, result.stderr

    def test_module_is_import_light(self):
        """Stdlib only: no pydantic, no yaml, no mem0, no config load.

        If this fails, someone made the ``ConsolidationAutoConfig`` reference a
        runtime import instead of a ``TYPE_CHECKING`` one, and the predicate's
        import cost is now a whole config build.
        """
        heavy = ', '.join(repr(m) for m in self.HEAVY)
        result = self._probe(
            'import sys\n'
            'import fused_memory.reconciliation.consolidation_auto  # noqa: F401\n'
            f'heavy = [{heavy}]\n'
            'present = [m for m in heavy if m in sys.modules]\n'
            'assert not present, present\n',
        )
        assert result.returncode == 0, result.stderr

    def test_imports_with_the_gate_first(self):
        """The gate leaf first, then this one."""
        result = self._probe(
            'import fused_memory.reconciliation.consolidation_gate as cg\n'
            'import fused_memory.reconciliation.consolidation_auto as ca\n'
            'assert cg is not None and ca is not None\n',
        )
        assert result.returncode == 0, result.stderr

    def test_imports_with_the_gate_second(self):
        """The reversed order. These two leaves are the pair task delta's
        executor will hold together, so both orders are production orders."""
        result = self._probe(
            'import fused_memory.reconciliation.consolidation_auto as ca\n'
            'import fused_memory.reconciliation.consolidation_gate as cg\n'
            'assert cg is not None and ca is not None\n',
        )
        assert result.returncode == 0, result.stderr


class TestBuildAutoCanonical:
    """PRD C3: the ONE home of auto-consolidated canonical text.

    The template is duplicated between the test below and the implementation
    DELIBERATELY, and only here: it is the one thing this class exists to pin,
    and a test that derived the expected string from the implementation would
    assert nothing about drift in its wording, backticks or punctuation.
    """

    CLAIM = (
        'Stage 1 keeps re-proposing the same cluster because the ledger row is '
        'never addressed.'
    )
    TOPIC = 'memory-consolidation'
    RUN_ID = 'ce8590f1-cc05-48da-9428-1cf1f54f3fff'

    def test_renders_the_prd_template_verbatim(self):
        expected = (
            f'{self.CLAIM}\n\n'
            f'Index canonical for topic `{self.TOPIC}` over 5 short peers; the '
            f'live metadata.topic scroll is the member list (auto-consolidated, '
            f'run {self.RUN_ID}).'
        )

        assert build_auto_canonical(
            self.CLAIM, self.TOPIC, 5, self.RUN_ID,
        ) == expected

    def test_the_claim_is_the_first_paragraph_verbatim(self):
        """PRD §2 measured that a template canonical retrieves within 0.015
        cosine of a hand-written one. That property rests on the claim LEADING
        the body, so the claim must open the string and be followed by exactly
        one blank line."""
        out = build_auto_canonical(self.CLAIM, self.TOPIC, 5, self.RUN_ID)

        assert out.startswith(self.CLAIM)
        assert out[len(self.CLAIM):len(self.CLAIM) + 3] == '\n\nI', (
            'exactly one blank line must separate the claim from the index line'
        )

    def test_bound_is_under_500_chars_at_the_extremes(self):
        """PRD D5's evidence, and why there is no `canonical_max_chars` leaf.

        Every input at its own cap: a claim at the `claim_max_chars` default
        (200), a slug at TOPIC_SLUG_MAX_LEN (100), N at the `member_max`
        default (20), and a real recon run id — `str(uuid4())`, 36 chars, the
        shape reconciliation/harness.py and targeted.py both generate.

        The lower bound is asserted too: a silent shrink to a truncating
        implementation would otherwise sail past a bare `< 500`.
        """
        out = build_auto_canonical(
            'x' * 200, 't' * TOPIC_SLUG_MAX_LEN, 20, str(uuid.uuid4()),
        )

        assert len(out) < 500, len(out)
        assert len(out) >= 460, (
            f'expected the measured 464-char maximum, got {len(out)} — a '
            'shorter maximum means the builder is clipping something'
        )

    def test_extreme_inputs_are_not_truncated(self):
        """The builder never clips: a 200-char claim appears in full."""
        claim = 'y' * 200
        out = build_auto_canonical(claim, 't' * TOPIC_SLUG_MAX_LEN, 20, self.RUN_ID)

        assert claim in out


class TestVerdictShapeAndGating:
    """The verdict's shape, and rung 1 of C2's binding evaluation order.

    `already_gated` is evaluated FIRST, before every hazard and every outcome
    rung. That order is the contract, not an optimisation: a topic a human
    already owns must never collect a second gate filing, which is task 3524's
    DECIDE-FIRST seam. The test below therefore hands the predicate a member
    list that ALSO trips a hazard and pins that the hazard never surfaces.
    """

    def test_already_gated_is_a_noop_naming_the_gate(self):
        """An open gate short-circuits everything, and names the gate it found.

        The fixture is deliberately hostile: `m2` is UNREADABLE, which on any
        other path is a fail-closed FAIL. Seeing NOOP here — with no hazard
        reason — is what proves rung 1 runs before rung 2 rather than merely
        beside it.
        """
        members = _members(_member('m1'), _member('m3'))
        members['m2'] = UNREADABLE

        verdict = evaluate_auto_predicate(
            _proposal(('m1', 'm2', 'm3')),
            members=members,
            canonical_count=0,
            open_gate_id='5183',
            existing_canonical_slugs=(),
            config=_auto_config(),
        )

        assert verdict.outcome is AutoOutcome.NOOP
        assert len(verdict.reasons) == 1, verdict.reasons
        assert verdict.reasons[0].code is AutoReasonCode.already_gated
        assert '5183' in verdict.reasons[0].ids
        assert verdict.retain_ids == ()
        assert verdict.stripped_ids == ()

    def test_verdict_is_frozen_and_typed(self):
        """Structured data, not prose: enums and tuples, and immutable.

        A caller serialises `code.value` and branches on it (PRD §6). Freezing
        the verdict is what lets one be passed to a ledger writer and a
        reporter without either being able to edit the other's copy.
        """
        gated = evaluate_auto_predicate(
            _proposal(('m1', 'm2')),
            members=_members(_member('m1'), _member('m2')),
            canonical_count=0,
            open_gate_id='5183',
            existing_canonical_slugs=(),
            config=_auto_config(),
        )
        ungated = evaluate_auto_predicate(
            _proposal(('m1', 'm2')),
            members=_members(_member('m1'), _member('m2')),
            canonical_count=0,
            open_gate_id=None,
            existing_canonical_slugs=(),
            config=_auto_config(),
        )

        for verdict in (gated, ungated):
            assert isinstance(verdict, AutoVerdict)
            assert isinstance(verdict.outcome, AutoOutcome)
            assert isinstance(verdict.reasons, tuple)
            assert isinstance(verdict.retain_ids, tuple)
            assert isinstance(verdict.stripped_ids, tuple)
            for reason in verdict.reasons:
                assert isinstance(reason.code, AutoReasonCode)
                assert isinstance(reason.ids, tuple)
            with pytest.raises(dataclasses.FrozenInstanceError):
                # setattr, not a direct attribute assignment, so this stays pyright-clean
                # (a direct assignment on a frozen dataclass is reportAttributeAccessIssue).
                setattr(verdict, 'outcome', AutoOutcome.PASS)  # noqa: B010

    def test_predicate_version_is_echoed_from_config(self):
        """The version is a LIVE READ of the green-tier leaf, not a constant.

        `consolidation_auto.predicate_version` reloads hot, so a verdict minted
        after a reload must carry the NEW version — that is how a ledger row
        records which ruleset judged it. A module constant would silently keep
        stamping the old one.
        """
        args: dict[str, Any] = dict(
            members=_members(_member('m1'), _member('m2')),
            canonical_count=0,
            open_gate_id='5183',
            existing_canonical_slugs=(),
        )
        shipped = _auto_config()
        retagged = _auto_config(predicate_version='2')

        assert shipped.predicate_version == '1'
        first = evaluate_auto_predicate(_proposal(('m1', 'm2')), config=shipped, **args)
        second = evaluate_auto_predicate(_proposal(('m1', 'm2')), config=retagged, **args)

        assert first.predicate_version == shipped.predicate_version
        assert second.predicate_version == '2'

    def test_the_predicate_reads_no_store(self):
        """The caller supplies every fact — there is nowhere to pass a service.

        C2's "never calls search" is pinned structurally rather than by
        watching for I/O: the signature admits exactly six names and every one
        of them is a plain fact. A verdict that could depend on a live read
        would be a verdict that depends on WHEN it ran.
        """
        params = inspect.signature(evaluate_auto_predicate).parameters

        assert set(params) == {
            'proposal',
            'members',
            'canonical_count',
            'open_gate_id',
            'existing_canonical_slugs',
            'config',
        }
        assert [
            name
            for name, p in params.items()
            if p.kind is not inspect.Parameter.KEYWORD_ONLY
        ] == ['proposal']

        verdict = evaluate_auto_predicate(
            _proposal(('m1', 'm2')),
            members=_members(_member('m1'), _member('m2')),
            canonical_count=0,
            open_gate_id=None,
            existing_canonical_slugs=(),
            config=_auto_config(),
        )

        assert isinstance(verdict, AutoVerdict)


class TestTerminalOutcomes:
    """Rungs 3-6: the strip, and the three non-FAIL outcomes.

    These are the shapes the auto path exists to produce. PRD D4 expects the
    RE-EMISSION shapes (B2/B3) to be the majority verdict over time — a topic
    gets consolidated once and then re-proposed for the rest of its life — so
    the incumbent-strip and the tag-only rung are the load-bearing cases, not
    the fresh-cluster one.
    """

    def test_fresh_cluster_passes(self):
        """PRD B1: nothing stamped, no canonical yet — mint one.

        `retain_ids` comes back in the PROPOSAL's order, not the mapping's, so
        the members are built here in a different order than they are proposed.
        The executor folds in that order, and an order that silently came from
        a dict would be an order no caller could predict.
        """
        members = _members(
            _member('m3'), _member('m1'), _member('m5'), _member('m4'), _member('m2'),
        )

        verdict = evaluate_auto_predicate(
            _proposal(('m1', 'm2', 'm3', 'm4', 'm5')),
            members=members,
            canonical_count=0,
            open_gate_id=None,
            existing_canonical_slugs=(),
            config=_auto_config(),
        )

        assert verdict.outcome is AutoOutcome.PASS
        assert verdict.retain_ids == ('m1', 'm2', 'm3', 'm4', 'm5')
        assert verdict.stripped_ids == ()
        assert verdict.reasons == ()

    def test_unstamped_member_beside_one_canonical_is_tag_only(self):
        """A topic that already has a canonical grows two new members.

        PASS_TAG_ONLY means the executor stamps the unstamped members and
        touches the incumbent's content and metadata NOT AT ALL (PRD D14).
        `retain_ids` still carries all five: the already-stamped ones are what
        makes the cluster this topic's, and dropping them would leave the
        caller unable to report what it judged.
        """
        members = _members(
            _member('m1', topic=TOPIC),
            _member('m2', topic=TOPIC),
            _member('m3', topic=TOPIC),
            _member('n1'),
            _member('n2'),
        )

        verdict = evaluate_auto_predicate(
            _proposal(('m1', 'm2', 'm3', 'n1', 'n2')),
            members=members,
            canonical_count=1,
            open_gate_id=None,
            existing_canonical_slugs=(),
            config=_auto_config(),
        )

        assert verdict.outcome is AutoOutcome.PASS_TAG_ONLY
        assert verdict.retain_ids == ('m1', 'm2', 'm3', 'n1', 'n2')
        assert verdict.stripped_ids == ()

    def test_fully_consolidated_re_emission_is_a_noop(self):
        """PRD B3: the topic is already done and the cluster is re-proposed.

        Every member is stamped and the canonical exists, so there is nothing
        to write. NOOP rather than PASS_TAG_ONLY is the difference between a
        cycle that does nothing and a cycle that rewrites a settled topic every
        time Stage 1 notices it again.
        """
        members = _members(*(_member(f'm{i}', topic=TOPIC) for i in range(1, 6)))

        verdict = evaluate_auto_predicate(
            _proposal(('m1', 'm2', 'm3', 'm4', 'm5')),
            members=members,
            canonical_count=1,
            open_gate_id=None,
            existing_canonical_slugs=(),
            config=_auto_config(),
        )

        assert verdict.outcome is AutoOutcome.NOOP
        assert [r.code for r in verdict.reasons] == [AutoReasonCode.already_consolidated]
        assert verdict.retain_ids == ('m1', 'm2', 'm3', 'm4', 'm5')
        assert verdict.stripped_ids == ()

    def test_re_emission_with_regrowth_strips_the_incumbent(self):
        """PRD B2, the shape the strip rung exists for.

        A consolidated topic grows three new members and the LLM re-proposes
        the cluster, enumerating the canonical `C` along with them — which is
        what an LLM looking at a topic scroll naturally does. `C` must be moved
        out of the retained set (the executor would otherwise tag or fold the
        canonical into itself) and DISCLOSED, because a silently shortened
        member list would drop a record the proposal named with nothing
        recording why (PRD §6 INV-11).

        `C` is proposed in the MIDDLE of the list, so the retained order is a
        real assertion about order preservation rather than a truncation that
        would pass by accident.
        """
        proposed = ('m1', 'm2', 'C', 'm3', 'n1', 'm4', 'n2', 'm5', 'n3')
        members = _members(
            *(_member(f'm{i}', topic=TOPIC) for i in range(1, 6)),
            *(_member(f'n{i}') for i in range(1, 4)),
            _member('C', topic=TOPIC, canonical=True),
        )

        verdict = evaluate_auto_predicate(
            _proposal(proposed),
            members=members,
            canonical_count=1,
            open_gate_id=None,
            existing_canonical_slugs=(),
            config=_auto_config(),
        )

        assert verdict.outcome is AutoOutcome.PASS_TAG_ONLY
        assert verdict.stripped_ids == ('C',)
        assert 'C' not in verdict.retain_ids
        assert verdict.retain_ids == ('m1', 'm2', 'm3', 'n1', 'm4', 'n2', 'm5', 'n3')

        stripped = [r for r in verdict.reasons if r.code is AutoReasonCode.incumbent_canonical_stripped]
        assert len(stripped) == 1, verdict.reasons
        assert stripped[0].ids == ('C',)

    def test_a_stripped_incumbent_does_not_make_the_cluster_look_consolidated(self):
        """The strip is rung 3 and runs BEFORE the already-consolidated rung.

        Same cluster as above with the regrowth removed. The verdict is a NOOP,
        and the incumbent is STILL stripped and disclosed: `retain_ids`
        describes the members the predicate actually judged, with the canonical
        accounted for separately rather than silently folded in among them.
        Rung 4 reads that retained set, not `proposal.member_ids`.
        """
        members = _members(
            *(_member(f'm{i}', topic=TOPIC) for i in range(1, 6)),
            _member('C', topic=TOPIC, canonical=True),
        )

        verdict = evaluate_auto_predicate(
            _proposal(('m1', 'm2', 'C', 'm3', 'm4', 'm5')),
            members=members,
            canonical_count=1,
            open_gate_id=None,
            existing_canonical_slugs=(),
            config=_auto_config(),
        )

        assert verdict.outcome is AutoOutcome.NOOP
        assert verdict.stripped_ids == ('C',)
        assert verdict.retain_ids == ('m1', 'm2', 'm3', 'm4', 'm5')

        codes = [r.code for r in verdict.reasons]
        assert AutoReasonCode.incumbent_canonical_stripped in codes
        assert AutoReasonCode.already_consolidated in codes

    def test_stamped_members_with_no_canonical_still_pass(self):
        """Stamped members, zero canonicals — mint the missing canonical.

        The fall-through a naive "all stamped -> NOOP" gets wrong. It is a real
        state: task theta's migration stamps members onto a topic before any
        canonical exists, and a NOOP here would leave a topic whose scroll has
        members and no index entry, which nothing else sweeps.
        """
        members = _members(*(_member(f'm{i}', topic=TOPIC) for i in range(1, 4)))

        verdict = evaluate_auto_predicate(
            _proposal(('m1', 'm2', 'm3')),
            members=members,
            canonical_count=0,
            open_gate_id=None,
            existing_canonical_slugs=(),
            config=_auto_config(),
        )

        assert verdict.outcome is AutoOutcome.PASS
        assert verdict.retain_ids == ('m1', 'm2', 'm3')


#: Correction banners taken from LIVE dark_factory records, not invented.
#:
#: Each is a real body prefix. Together they pin the four shapes the regex has
#: to cover: bracketed at offset 0, bracketed MID-BODY, the bare bracketed
#: stamp, and the un-bracketed line-leading form.
CORRECTION_BANNER_SPECIMENS = {
    # mem0 cedabf87-ae25-4acb-9331-19b13599e78d — the 5180 canonical the PRD
    # names as its motivating specimen.
    'bracketed_5180_canonical': (
        '[CORRECTION 2026-09-08 (curator sitting, gate 5180 / esc-5180-1, ruled by '
        'Leo) — READ THIS FIRST]: Two earlier banners sat here and are replaced by '
        'this one.'
    ),
    # mem0 0090d639-c325-490c-a432-866b60a26ba7 — a SECOND banner sitting after
    # an ordinary paragraph. This one is why the regex must not be \A-anchored:
    # an anchored pattern reads this record as clean.
    'bracketed_mid_body': (
        f'{BENIGN_BODY}\n\n'
        '[CORRECTION 2026-09-07, task 4899] Topic `dashboard-js-test-substrate`: '
        'the load-bearing premise of this entry is now FALSE.'
    ),
    'bracketed_bare_stamp': '[CORRECTION 2026-08-08] The measured figure below is stale.',
    # Pinned in-repo as
    # scripts/amend_stale_resume_cwd_records.py::_CORRECTED_RECORD_PREIMAGE.
    'unbracketed_superseded': (
        'SUPERSEDED 2026-08-30 (by Stage-1 memory consolidation, task 4610). This '
        "entry's original framing is CONTRADICTED by later measurement."
    ),
    # One of the 9 live kind='correction' records.
    'unbracketed_correction_to': (
        "CORRECTION to the parent entry's closing STATUS paragraph (esc-4377-4, "
        'measured 2026-09-04).'
    ),
}

#: Bodies that must NOT match. The AMENDMENT idiom is the load-bearing one: 53
#: live dark_factory records open with it, and it is benign ACCRETION rather
#: than retraction — `server/grouped_read.py::AMENDMENT_KIND` treats it as a
#: first-class child kind. Matching it would refuse the majority of real
#: clusters and turn the auto path back into the gate flood the PRD ends.
BENIGN_ACCRETION_SPECIMENS = {
    'amendment': (
        'AMENDMENT to the "KNOWN-RED ON MAIN: test_fleet_staleness_composition.py" '
        'record — the trio is NO LONGER RED.'
    ),
    'sharpening': (
        'SHARPENING of the "never pin the prose of a constant this codebase owns" '
        'norm: the rule is about ownership, not about prose.'
    ),
    'ordinary': BENIGN_BODY,
}


class TestMemberHazards:
    """Rung 2, member level: every code that sends a cluster to a human.

    The arm is REFUSAL-ONLY. A code firing adds a FAIL; no code firing certifies
    nothing (PRD D14). Every test here RUNS the predicate against fixture
    members and asserts on the returned verdict — none inspects the regex, the
    key set, or any source text, so the checks survive a reimplementation.
    """

    def test_unreadable_member_fails_closed(self):
        """A read that DID NOT ANSWER must never look benign.

        `MemoryService.get_memory_by_id` returns None for a genuine miss but
        PROPAGATES a backend read timeout as TimeoutError, precisely so callers
        can tell the two apart. Collapsing them would let a Qdrant timeout read
        as "record absent" — and absent is a benign-looking input the predicate
        would happily rule on.
        """
        members = _members(_member('m1'), _member('m3'))
        members['m2'] = UNREADABLE

        verdict = _judge(members, proposal_ids=('m1', 'm2', 'm3'))

        assert verdict.outcome is AutoOutcome.FAIL
        reasons = _reasons_for(verdict, AutoReasonCode.member_unreadable)
        assert [r.ids for r in reasons] == [('m2',)]

    def test_missing_member_fails(self):
        """A member the proposal named and the store does not have."""
        members = _members(_member('m1'), _member('m3'))
        members['m2'] = None

        verdict = _judge(members, proposal_ids=('m1', 'm2', 'm3'))

        assert verdict.outcome is AutoOutcome.FAIL
        reasons = _reasons_for(verdict, AutoReasonCode.member_not_found)
        assert [r.ids for r in reasons] == [('m2',)]

    def test_member_canonical_of_a_different_topic_fails(self):
        """Folding another topic's canonical in would destroy that topic's index.

        PRD D15: the detail names BOTH slugs, because the operator reading the
        refusal needs to know which other topic was about to be swallowed.

        The incumbent of THIS topic is in the same fixture and must NOT be
        reported under this code — it is stripped, not refused, and that
        distinction is the whole of rung 3.
        """
        members = _members(
            _member('m1'),
            _member('x1', topic='some-other-topic', canonical=True),
            _member('C', topic=TOPIC, canonical=True),
        )

        verdict = _judge(members, proposal_ids=('m1', 'x1', 'C'), canonical_count=1)

        assert verdict.outcome is AutoOutcome.FAIL
        reasons = _reasons_for(verdict, AutoReasonCode.member_already_canonical)
        assert [r.ids for r in reasons] == [('x1',)]
        assert 'some-other-topic' in reasons[0].detail
        assert TOPIC in reasons[0].detail

    def test_member_stamped_with_a_different_topic_fails(self):
        """The admitted set is `topic in (None, T)` — anything else is a refusal.

        A member already stamped with another topic belongs to that topic's
        scroll; re-stamping it would silently move it, and nothing sweeps a
        topic that lost a member.
        """
        members = _members(
            _member('m1'),
            _member('m2', topic=TOPIC),
            _member('x1', topic='some-other-topic'),
        )

        verdict = _judge(members, proposal_ids=('m1', 'm2', 'x1'))

        assert verdict.outcome is AutoOutcome.FAIL
        reasons = _reasons_for(verdict, AutoReasonCode.member_different_topic)
        assert [r.ids for r in reasons] == [('x1',)]

    @pytest.mark.parametrize('key', sorted(CORRECTION_METADATA_KEYS))
    def test_member_carrying_correction_metadata_fails(self, key):
        """Parametrised FROM the frozenset, so a key added later is covered.

        Hard-coding the six keys here would let a seventh ship untested — the
        one shape of drift this arm cannot afford, since it is the only
        machine-readable half of correction detection.
        """
        correction_meta: dict[str, Any] = {key: 'ce8590f1-cc05-48da-9428-1cf1f54f3fff'}
        members = _members(_member('m1'), _member('m2', **correction_meta))

        verdict = _judge(members)

        assert verdict.outcome is AutoOutcome.FAIL
        reasons = _reasons_for(verdict, AutoReasonCode.member_carries_correction_metadata)
        assert [r.ids for r in reasons] == [('m2',)]
        assert key in reasons[0].detail

    @pytest.mark.parametrize(
        'body',
        CORRECTION_BANNER_SPECIMENS.values(),
        ids=list(CORRECTION_BANNER_SPECIMENS),
    )
    def test_member_carrying_a_correction_banner_fails(self, body):
        """Every shape is a MEASURED live record body, not an invented one.

        A regex derived from imagined banners would miss the ones that exist;
        the mid-body specimen in particular is a real record whose second
        banner an anchored pattern reads straight past.
        """
        members = _members(_member('m1'), _member('m2', content=body))

        verdict = _judge(members)

        assert verdict.outcome is AutoOutcome.FAIL
        reasons = _reasons_for(verdict, AutoReasonCode.member_carries_correction_banner)
        assert [r.ids for r in reasons] == [('m2',)]

    @pytest.mark.parametrize(
        'body',
        BENIGN_ACCRETION_SPECIMENS.values(),
        ids=list(BENIGN_ACCRETION_SPECIMENS),
    )
    def test_benign_accretion_is_not_a_correction_banner(self, body):
        """The false-positive guard, and why the regex is measured not guessed.

        A cluster of ordinary amended records must pass. If AMENDMENT matched,
        the predicate would refuse 53 live records' worth of perfectly healthy
        accretion and the auto path would produce the very gate flood it exists
        to end.
        """
        members = _members(_member('m1'), _member('m2', content=body))

        verdict = _judge(members)

        assert AutoReasonCode.member_carries_correction_banner not in _codes(verdict)
        assert verdict.outcome is AutoOutcome.PASS

    def test_mixed_category_fails(self):
        """One canonical cannot index two categories' worth of records.

        A member with NO category does not by itself trip this: an unstamped
        record predates the vocabulary rather than contradicting it, and
        refusing on absence would refuse most old clusters.
        """
        members = _members(
            _member('m1', category='procedural_knowledge'),
            _member('m2', category='observations_and_summaries'),
        )

        verdict = _judge(members)

        assert verdict.outcome is AutoOutcome.FAIL
        reasons = _reasons_for(verdict, AutoReasonCode.mixed_category)
        assert len(reasons) == 1
        assert set(reasons[0].ids) == {'m1', 'm2'}

        uncategorised = _members(_member('n1', category=None), _member('n2'))
        assert AutoReasonCode.mixed_category not in _codes(_judge(uncategorised))

    def test_a_hazard_beside_a_live_canonical_still_fails(self):
        """PRD D4, the case the binding order exists for.

        A NEW member carrying a correction banner arrives at a topic that
        already has a healthy canonical. Were the outcome rungs tried first
        this would be a tag-only pass — and the executor would stamp a retracted
        record into a live topic's scroll. Hazards outrank rungs 3-6.
        """
        members = _members(
            _member('m1', topic=TOPIC),
            _member('m2', topic=TOPIC),
            _member('n1', content=CORRECTION_BANNER_SPECIMENS['bracketed_5180_canonical']),
        )

        verdict = _judge(members, canonical_count=1)

        assert verdict.outcome is AutoOutcome.FAIL
        assert AutoReasonCode.member_carries_correction_banner in _codes(verdict)

    def test_every_hazard_is_collected_not_short_circuited(self):
        """One human sitting must name every problem, not the first one found.

        Same no-short-circuit bar the validator holds. A predicate that stopped
        at the first hazard would send the cluster back three times.
        """
        members = _members(
            _member('m1'),
            _member('m4', content=CORRECTION_BANNER_SPECIMENS['unbracketed_superseded']),
        )
        members['m2'] = UNREADABLE
        members['m3'] = None

        verdict = _judge(members, proposal_ids=('m1', 'm2', 'm3', 'm4'))

        assert verdict.outcome is AutoOutcome.FAIL
        assert {
            AutoReasonCode.member_unreadable,
            AutoReasonCode.member_not_found,
            AutoReasonCode.member_carries_correction_banner,
        } <= set(_codes(verdict))


#: Slug fixtures whose token-Jaccard against COLLIDING_TOPIC is computed, not
#: guessed: 0.6 exactly (the shipped threshold, so the boundary is testable),
#: 0.75 (clearly over) and 0.1667 (clearly under).
COLLIDING_TOPIC = 'memory-auto-consolidation-gate'
SLUG_AT_THRESHOLD = 'memory-auto-consolidation-sweep'
SLUG_OVER_THRESHOLD = 'memory-auto-consolidation'
SLUG_UNDER_THRESHOLD = 'memory-metadata-census'


class TestCanonicalAndSlugHazards:
    """Rung 2, canonical and slug level: the hazards about the topic itself.

    The incumbent-level codes are reportable only when the proposal NAMES the
    incumbent. C2 gives the predicate per-member reads plus a count and nothing
    else, so a canonical outside the member list is covered by the count alone
    — and PRD D14 already assigns the undetectable case, a stale incumbent
    carrying no banner, to the human sitting rather than to code.
    """

    def test_incumbent_carrying_a_correction_banner_fails(self):
        """PRD B4, using the exact live record the PRD names.

        Reported as `canonical_carries_correction`, NOT as the member code:
        a corrected INCUMBENT is a different fact from a corrected member, and
        it is the one that says this topic's index entry is itself unsound.
        And it must not be silently stripped — stripping is what happens to a
        HEALTHY incumbent, and doing it here would quietly consolidate a topic
        around a canonical someone has retracted.
        """
        members = _members(
            _member('m1'),
            _member('m2'),
            _member(
                'C',
                topic=TOPIC,
                canonical=True,
                content=CORRECTION_BANNER_SPECIMENS['bracketed_5180_canonical'],
            ),
        )

        verdict = _judge(members, canonical_count=1)

        assert verdict.outcome is AutoOutcome.FAIL
        assert [r.ids for r in _reasons_for(verdict, AutoReasonCode.canonical_carries_correction)] == [('C',)]
        assert AutoReasonCode.member_carries_correction_banner not in _codes(verdict)
        assert AutoReasonCode.incumbent_canonical_stripped not in _codes(verdict)

    def test_incumbent_carrying_correction_metadata_fails(self):
        """The machine-readable half of the same fact."""
        members = _members(
            _member('m1'),
            _member('C', topic=TOPIC, canonical=True, superseded_by='cedabf87-ae25-4acb-9331-19b13599e78d'),
        )

        verdict = _judge(members, canonical_count=1)

        assert verdict.outcome is AutoOutcome.FAIL
        assert [r.ids for r in _reasons_for(verdict, AutoReasonCode.canonical_carries_correction)] == [('C',)]
        assert AutoReasonCode.member_carries_correction_metadata not in _codes(verdict)

    def test_incumbent_category_mismatch_fails(self):
        """An index entry filed under one category cannot index another's records.

        The detail names BOTH categories, because the operator's next question
        is always which of the two is wrong.
        """
        members = _members(
            _member('m1', category='procedural_knowledge'),
            _member('m2', category='procedural_knowledge'),
            _member('C', topic=TOPIC, canonical=True, category='observations_and_summaries'),
        )

        verdict = _judge(members, canonical_count=1)

        assert verdict.outcome is AutoOutcome.FAIL
        reasons = _reasons_for(verdict, AutoReasonCode.canonical_category_mismatch)
        assert [r.ids for r in reasons] == [('C',)]
        assert 'observations_and_summaries' in reasons[0].detail
        assert 'procedural_knowledge' in reasons[0].detail

    def test_multiple_canonicals_fails(self):
        """Canonical uniqueness ships in WARN mode, so the predicate probes it.

        `memory_metadata.enforce` is False (task 3626): nothing stops a second
        canonical existing. Trusting an unenforced invariant is how a topic ends
        up with two index entries and no way to tell which one is read.
        """
        verdict = _judge(_members(_member('m1'), _member('m2')), canonical_count=2)

        assert verdict.outcome is AutoOutcome.FAIL
        assert AutoReasonCode.multiple_canonicals in _codes(verdict)

    def test_unavailable_canonical_count_fails_closed(self):
        """A count the caller could not obtain must never read as zero.

        Zero is the MINT path. Reading "I could not find out" as "there is
        none" is precisely how a second canonical gets written for a topic that
        already has one.
        """
        verdict = _judge(_members(_member('m1'), _member('m2')), canonical_count=None)

        assert verdict.outcome is AutoOutcome.FAIL
        assert AutoReasonCode.canonical_count_unavailable in _codes(verdict)

    @pytest.mark.parametrize(
        ('slug', 'fires'),
        [
            (SLUG_AT_THRESHOLD, True),
            (SLUG_OVER_THRESHOLD, True),
            (SLUG_UNDER_THRESHOLD, False),
        ],
    )
    def test_slug_near_collision_fails(self, slug, fires):
        """Two slugs that mean the same thing split a topic nothing sweeps.

        SLUG_AT_THRESHOLD sits at EXACTLY the 0.6 default, pinning that the
        comparison is inclusive. `>=` is the fail-closed reading of "above a
        threshold": at the boundary a human gate costs one sitting, while a
        wrong auto-mint splits a topic across two canonicals permanently.
        """
        members = _members(_member('m1'), _member('m2'))

        verdict = _judge(members, topic=COLLIDING_TOPIC, existing_canonical_slugs=(slug,))

        reasons = _reasons_for(verdict, AutoReasonCode.slug_near_collision)
        assert bool(reasons) is fires
        if fires:
            assert verdict.outcome is AutoOutcome.FAIL
            assert slug in reasons[0].detail
            assert COLLIDING_TOPIC in reasons[0].detail
        else:
            assert verdict.outcome is AutoOutcome.PASS

    def test_the_topics_own_slug_is_never_a_collision(self):
        """The skip PRD B2 cannot live without.

        On EVERY re-emission the topic already has a canonical, so its own slug
        is necessarily in the existing-slug list and its self-Jaccard is 1.0.
        Without the skip the collision hazard would fire on every tag-only
        refresh and PASS_TAG_ONLY would be unreachable — the predicate would
        refuse precisely the case it was built for.
        """
        members = _members(
            _member('m1', topic=TOPIC), _member('m2', topic=TOPIC), _member('n1'),
        )

        verdict = _judge(
            members,
            canonical_count=1,
            existing_canonical_slugs=(TOPIC, 'dashboard-js-test-substrate'),
        )

        assert AutoReasonCode.slug_near_collision not in _codes(verdict)
        assert verdict.outcome is AutoOutcome.PASS_TAG_ONLY

    def test_slug_collision_threshold_is_read_from_config(self):
        """The threshold is a live green-tier read, calibrated during rollout.

        PRD §12 Q2 leaves 0.6 to be tuned in the supervised cycle, which is only
        possible if the predicate reads the leaf per call rather than baking it
        in at import.
        """
        members = _members(_member('m1'), _member('m2'))
        args: dict[str, Any] = dict(
            topic=COLLIDING_TOPIC, existing_canonical_slugs=(SLUG_AT_THRESHOLD,)
        )

        assert _judge(members, **args).outcome is AutoOutcome.FAIL

        relaxed = _judge(members, config=_auto_config(slug_collision_jaccard=0.7), **args)

        assert AutoReasonCode.slug_near_collision not in _codes(relaxed)
        assert relaxed.outcome is AutoOutcome.PASS


class TestCountAndMemberSetContradictions:
    """Two write-bearing paths that shipped without a guard (review finding 1).

    MEASURED on this branch before the fix, by running the shipped predicate:

    (i)   topic ``t-a``, members ``{m1 (topic=t-a), C (canonical=True,
          topic=t-a)}``, ``canonical_count=0`` -> ``PASS``,
          ``stripped_ids=('C',)``, ``retain_ids=('m1',)``. That is rung 6's
          MINT path: it tells the executor to write a SECOND canonical for a
          topic whose incumbent is visible in the very member list it was
          handed. Canonical uniqueness ships in WARN mode
          (``memory_metadata.enforce=False``, task 3626), so nothing downstream
          refuses it — this is the +1-per-pass ratchet the feature exists to
          end.
    (ii)  members ``{C}`` alone, ``canonical_count=0`` -> ``PASS``,
          ``retain_ids=()`` — mint a canonical over zero members.
    (iii) ``member_ids=()``, ``members={}``, ``canonical_count=0`` -> ``PASS``,
          ``retain_ids=()``, ``reasons=()`` — a write-bearing mint instruction
          carrying not one reason. Reachable with no incumbent anywhere, which
          is why the empty-retain guard is a SEPARATE fix rather than a
          corollary of the first.
    """

    def test_a_named_incumbent_contradicting_a_zero_count_fails(self):
        """Case (i): the count says none, a named member IS this topic's canonical.

        The two are separate, non-atomic reads, so they can disagree without
        any caller bug — and when they do, the topic's canonical state is not
        decidable here. Fail closed, exactly as the unavailable-count arm does.
        """
        members = _members(
            _member('m1', topic=TOPIC),
            _member('C', topic=TOPIC, canonical=True),
        )

        verdict = _judge(members, canonical_count=0)

        assert verdict.outcome is AutoOutcome.FAIL
        assert verdict.outcome is not AutoOutcome.PASS
        contradictions = _reasons_for(verdict, AutoReasonCode.canonical_count_contradicted)
        assert len(contradictions) == 1
        assert contradictions[0].ids == ('C',)
        assert verdict.stripped_ids == ()
        assert verdict.retain_ids == ()

    def test_a_proposal_of_only_the_incumbent_with_a_zero_count_fails(self):
        """Case (ii): same defect, so the same code — the reads disagree.

        Nothing about this cluster is decidable either: the one record the
        proposal names is the canonical the count claims does not exist.
        """
        members = _members(_member('C', topic=TOPIC, canonical=True))

        verdict = _judge(members, canonical_count=0)

        assert verdict.outcome is AutoOutcome.FAIL
        assert AutoReasonCode.canonical_count_contradicted in _codes(verdict)
        assert verdict.retain_ids == ()
        assert verdict.stripped_ids == ()

    def test_a_zero_count_with_no_incumbent_named_still_passes(self):
        """The over-refusal guard: the hazard keys on an INCUMBENT, not a stamp.

        Task theta's migration stamps members with a topic BEFORE any canonical
        exists, so a zero count beside stamped members is the migration's
        ordinary shape. Refusing it would break PRD B1 outright. This is
        ``test_stamped_members_with_no_canonical_still_pass`` re-asserted
        against the new code's ABSENCE.
        """
        members = _members(_member('m1', topic=TOPIC), _member('m2', topic=TOPIC))

        verdict = _judge(members, canonical_count=0)

        assert verdict.outcome is AutoOutcome.PASS
        assert AutoReasonCode.canonical_count_contradicted not in _codes(verdict)
        assert verdict.retain_ids == ('m1', 'm2')

    def test_a_canonical_of_another_topic_does_not_contradict_the_count(self):
        """The two codes stay disjoint: only THIS topic's incumbent counts.

        A canonical of some other topic is ``member_already_canonical`` — a
        member-level hazard — and says nothing about how many canonicals THIS
        topic has. ``_is_incumbent`` already draws that line and the new rung
        must not blur it.
        """
        members = _members(
            _member('m1'),
            _member('x1', topic='some-other-topic', canonical=True),
        )

        verdict = _judge(members, canonical_count=0)

        assert verdict.outcome is AutoOutcome.FAIL
        assert AutoReasonCode.member_already_canonical in _codes(verdict)
        assert AutoReasonCode.canonical_count_contradicted not in _codes(verdict)

    def test_mint_over_zero_members_is_refused(self):
        """Case (iii): a write-bearing rung reached with nothing to act on.

        The emit boundary refuses an empty member list as
        ``member_count_out_of_range`` (C1) and the predicate deliberately does
        not re-derive it (PRD D6) — but an aged ledger row or a mis-wired
        caller can still put one here, and a fail-closed predicate must not
        answer it with a write-bearing PASS.
        """
        verdict = _judge({}, proposal_ids=(), canonical_count=0)

        assert verdict.outcome is AutoOutcome.FAIL
        assert verdict.outcome is not AutoOutcome.PASS
        assert AutoReasonCode.no_retained_members in _codes(verdict)
        assert verdict.retain_ids == ()

    def test_a_zero_member_proposal_beside_a_live_canonical_stays_a_noop(self):
        """The deliberate scope boundary — measured, and left alone on purpose.

        A NOOP is not write-bearing: it instructs the executor to do nothing,
        which is safe. Only the write-bearing rung-6 PASS is guarded. Pinning
        this stops a later reader tidying the guard upward into rung 4 and
        turning a harmless no-op into a human gate.
        """
        verdict = _judge({}, proposal_ids=(), canonical_count=1)

        assert verdict.outcome is AutoOutcome.NOOP
        assert AutoReasonCode.already_consolidated in _codes(verdict)
        assert AutoReasonCode.no_retained_members not in _codes(verdict)

#: Every reason code, mapped to inputs that actually PRODUCE it.
#:
#: The table is the deliverable "one test per reason code" made
#: machine-checked: the test below RUNS each entry and asserts the code comes
#: back, and separately asserts the key set is the whole enum. A code added
#: without a producing fixture fails here rather than shipping unreachable, and
#: a code whose producing conditions drift out from under it fails here too —
#: neither of which a test-name convention could catch.
REASON_CODE_FIXTURES: dict[AutoReasonCode, Callable[[], AutoVerdict]] = {
    AutoReasonCode.already_gated: lambda: _judge(
        _members(_member('m1'), _member('m2')), open_gate_id='5183',
    ),
    AutoReasonCode.member_unreadable: lambda: _judge(
        {'m1': _member('m1'), 'm2': UNREADABLE},
    ),
    AutoReasonCode.member_not_found: lambda: _judge(
        {'m1': _member('m1'), 'm2': None},
    ),
    AutoReasonCode.member_already_canonical: lambda: _judge(
        _members(_member('m1'), _member('x1', topic='some-other-topic', canonical=True)),
    ),
    AutoReasonCode.member_different_topic: lambda: _judge(
        _members(_member('m1'), _member('x1', topic='some-other-topic')),
    ),
    AutoReasonCode.member_carries_correction_metadata: lambda: _judge(
        _members(_member('m1'), _member('m2', superseded_by='0090d639')),
    ),
    AutoReasonCode.member_carries_correction_banner: lambda: _judge(
        _members(
            _member('m1'),
            _member('m2', content=CORRECTION_BANNER_SPECIMENS['bracketed_bare_stamp']),
        ),
    ),
    AutoReasonCode.canonical_carries_correction: lambda: _judge(
        _members(
            _member('m1'),
            _member(
                'C',
                topic=TOPIC,
                canonical=True,
                content=CORRECTION_BANNER_SPECIMENS['unbracketed_superseded'],
            ),
        ),
        canonical_count=1,
    ),
    AutoReasonCode.mixed_category: lambda: _judge(
        _members(
            _member('m1', category='procedural_knowledge'),
            _member('m2', category='preferences_and_norms'),
        ),
    ),
    AutoReasonCode.canonical_category_mismatch: lambda: _judge(
        _members(
            _member('m1', category='procedural_knowledge'),
            _member('C', topic=TOPIC, canonical=True, category='observations_and_summaries'),
        ),
        canonical_count=1,
    ),
    AutoReasonCode.multiple_canonicals: lambda: _judge(
        _members(_member('m1'), _member('m2')), canonical_count=2,
    ),
    AutoReasonCode.canonical_count_unavailable: lambda: _judge(
        _members(_member('m1'), _member('m2')), canonical_count=None,
    ),
    AutoReasonCode.slug_near_collision: lambda: _judge(
        _members(_member('m1'), _member('m2')),
        topic=COLLIDING_TOPIC,
        existing_canonical_slugs=(SLUG_AT_THRESHOLD,),
    ),
    AutoReasonCode.incumbent_canonical_stripped: lambda: _judge(
        _members(
            _member('m1', topic=TOPIC),
            _member('n1'),
            _member('C', topic=TOPIC, canonical=True),
        ),
        canonical_count=1,
    ),
    AutoReasonCode.already_consolidated: lambda: _judge(
        _members(_member('m1', topic=TOPIC), _member('m2', topic=TOPIC)),
        canonical_count=1,
    ),
    AutoReasonCode.canonical_count_contradicted: lambda: _judge(
        _members(
            _member('m1', topic=TOPIC),
            _member('C', topic=TOPIC, canonical=True),
        ),
        canonical_count=0,
    ),
    AutoReasonCode.no_retained_members: lambda: _judge(
        {}, proposal_ids=(), canonical_count=0,
    ),
}


class TestEveryReasonCodeIsExercised:
    """The vocabulary and the fixtures cannot drift apart."""

    def test_the_table_covers_the_whole_enum(self):
        """A code with no producing fixture is a code nobody has ever seen."""
        assert set(REASON_CODE_FIXTURES) == set(AutoReasonCode)

    @pytest.mark.parametrize(
        'code', list(REASON_CODE_FIXTURES), ids=lambda code: code.value,
    )
    def test_no_reason_code_ships_without_a_fixture(self, code):
        """RUN the fixture and assert the code actually comes back.

        Running rather than reading is the whole point: a table entry that no
        longer produces its code — because a rung moved, or a hazard now
        outranks it — is exactly the drift a source-level check would miss.
        """
        verdict = REASON_CODE_FIXTURES[code]()

        assert code in _codes(verdict), verdict
