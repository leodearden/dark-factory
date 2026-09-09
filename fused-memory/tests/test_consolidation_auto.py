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
from typing import Any

import pytest

from fused_memory.config.schema import FusedMemoryConfig
from fused_memory.reconciliation.consolidation_auto import (
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


def _members(*records: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Key each record by its OWN id — the mapping the predicate is handed.

    Keying from the record keeps the mapping key and ``record['id']`` from ever
    disagreeing, which a hand-written literal would let drift silently.
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
                verdict.outcome = AutoOutcome.PASS

    def test_predicate_version_is_echoed_from_config(self):
        """The version is a LIVE READ of the green-tier leaf, not a constant.

        `consolidation_auto.predicate_version` reloads hot, so a verdict minted
        after a reload must carry the NEW version — that is how a ledger row
        records which ruleset judged it. A module constant would silently keep
        stamping the old one.
        """
        args = dict(
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
