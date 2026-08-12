"""Unit tests for fused_memory.utils.referent_resolution.

referent_resolution is the PRECEDENCE POLICY layer over utils/canonical_labels
— PRD plans/memory-referent-fidelity-prd.md, leaf γ (§Contract "Referent-set
resolution"), task 3668. canonical_labels stays THE single normative site for
the label vocabulary itself (INV-5 / resolved decision 5); this module decides
WHICH of the available referent sources is authoritative for a given write, in
the order declared > metadata.task_id > derived scan > none, and reports what
the prose contradicted.

The module under test compiles NO regex of its own and must never grow one:
the derived path IS ``scan_content``, the declared path builds ``Referent``s,
and the metadata bridge tries ``parse_node_name`` before its single bare-digit
branch. These tests therefore re-assert β's precision behaviour at γ's OWN
boundary (see :class:`TestDerivedPrecisionPassthrough`) rather than restating
it, so a future resolver change cannot widen it behind β's back.

Mirrors tests/test_canonical_labels.py and tests/test_cross_project_refs.py,
whose leaf-module shape and pytest conventions this module copies.
"""

from __future__ import annotations

import dataclasses

import pytest

from fused_memory.utils.canonical_labels import Referent
from fused_memory.utils.referent_resolution import (
    REFERENT_SOURCES,
    ReferentResolution,
    resolve_referents,
)

#: The group every derived-path test scans in, unless it needs a foreign one.
GROUP = 'dark_factory'


class TestReferentSourcesVocabulary:
    """``REFERENT_SOURCES`` is the CLOSED vocabulary of resolution sources.

    Exported as one tuple so task ι's declaration-rate telemetry iterates a
    single site instead of re-spelling four string literals — a second copy
    would be the same lockstep duplication INV-5 forbids for the label
    vocabulary itself.
    """

    def test_is_the_four_sources_in_precedence_order(self):
        """Order is meaningful: it IS the precedence chain, strongest first."""
        assert REFERENT_SOURCES == ('declared', 'metadata', 'derived', 'none')

    def test_is_a_tuple_so_the_vocabulary_cannot_be_extended_in_place(self):
        """A list would leave ``REFERENT_SOURCES.append('guessed')`` open to any
        importer, quietly widening a vocabulary consumers switch on."""
        assert isinstance(REFERENT_SOURCES, tuple)


class TestReferentResolutionIsFrozen:
    """A resolution is EVIDENCE for destructive edge surgery, not a mutable
    accumulator — the same rationale that freezes ``Referent`` and ``LabelScan``.

    ``frozen=True`` blocks attribute REBINDING only, so the result fields are
    tuples rather than lists: a list field would leave
    ``resolution.referents.append(...)`` wide open, letting a consumer quietly
    add a referent the resolver refused to infer.
    """

    def test_referents_cannot_be_rebound(self):
        resolution = ReferentResolution(source='none')
        with pytest.raises(dataclasses.FrozenInstanceError):
            resolution.referents = ()  # type: ignore[misc]

    def test_source_cannot_be_rebound(self):
        resolution = ReferentResolution(source='none')
        with pytest.raises(dataclasses.FrozenInstanceError):
            resolution.source = 'declared'  # type: ignore[misc]

    @pytest.mark.parametrize('attr', ['referents', 'conflicts', 'ambiguous'])
    def test_result_fields_are_tuples_not_lists(self, attr):
        """Tuples are what make the frozen-ness true rather than nominal."""
        resolution = ReferentResolution(source='none')
        assert isinstance(getattr(resolution, attr), tuple)
        with pytest.raises(AttributeError):
            getattr(resolution, attr).append('nope')

    def test_a_resolution_is_hashable(self):
        """A frozen dataclass holding lists silently is not; holding tuples it is."""
        assert hash(ReferentResolution(source='none')) is not None


class TestReferentResolutionFieldDefaults:
    """``source`` is REQUIRED; the three referent tuples default to empty."""

    def test_source_has_no_default(self):
        """The structural guarantee behind "``.source`` is set on EVERY
        resolution including the empty one": no construction path can omit it.
        A default would merely make omission convenient and invisible, and ι
        counts this field."""
        with pytest.raises(TypeError):
            ReferentResolution()  # type: ignore[call-arg]

    def test_the_empty_resolution_has_all_three_tuples_empty(self):
        resolution = ReferentResolution(source='none')
        assert resolution.referents == ()
        assert resolution.conflicts == ()
        assert resolution.ambiguous == ()

    def test_construction_is_keyword_only(self):
        """kw_only mirrors ``Referent``: four same-typed tuple-ish fields read
        as noise positionally, and a consumer swapping ``conflicts`` for
        ``ambiguous`` at a call site would type-check fine."""
        with pytest.raises(TypeError):
            ReferentResolution('none')  # type: ignore[misc]


class TestDerivedPath:
    """With nothing declared and no usable metadata, the content's own
    mentions resolve the write — ``source='derived'``.

    The derived path IS ``scan_content``: these tests assert the resolver
    passes β's answer THROUGH rather than re-deriving it, which is what keeps
    the label vocabulary at one site (INV-5).
    """

    def test_a_single_mention_resolves_to_that_referent(self):
        resolution = resolve_referents(
            declared=None, metadata={}, content='Fixed the bug in Task 3127.', group_id=GROUP
        )
        assert resolution.referents == (Referent(kind='task', number='3127'),)
        assert resolution.source == 'derived'
        assert resolution.conflicts == ()

    def test_order_and_dedup_are_exactly_the_scanners(self):
        """First-seen POSITIONAL order, de-duplicated on
        (kind, project_id, number) — proving the resolver hands through
        ``scan_content``'s answer instead of re-deriving one that could
        drift."""
        resolution = resolve_referents(
            declared=None,
            metadata={},
            content='Task 3127 blocks task 2500, and Task 3127 again.',
            group_id=GROUP,
        )
        assert resolution.referents == (
            Referent(kind='task', number='3127'),
            Referent(kind='task', number='2500'),
        )
        assert resolution.source == 'derived'

    def test_digits_are_preserved_verbatim(self):
        """A referent never invents or reformats a task number: '0132' is a
        DIFFERENT referent from '132', and int-normalizing it would silently
        repoint the write at another task."""
        resolution = resolve_referents(
            declared=None, metadata={}, content='see task 0132', group_id=GROUP
        )
        assert resolution.referents == (Referent(kind='task', number='0132'),)

    def test_a_foreign_qualified_mention_survives_as_a_foreign_referent(self):
        """The qualifier is a DIFFERENT-project signal and is never normalized
        away — that collapse is precisely the bug this PRD exists to detect."""
        resolution = resolve_referents(
            declared=None, metadata={}, content='mirrors reify:132', group_id=GROUP
        )
        assert resolution.referents == (
            Referent(kind='task', project_id='reify', number='132'),
        )
        assert resolution.source == 'derived'


class TestEmptyResolution:
    """Nothing declared, nothing bridged, nothing derivable — ``source='none'``.

    Pins the PRD's boundary row "Undeclared, underivable -> write succeeds;
    source='none'; counted": 'none' is a REAL, counted source, not a missing
    value. ι reads this share to measure how often referents are unknowable.
    """

    @pytest.mark.parametrize(
        'content',
        [
            '',
            'the merge-lane hardening task',
            'Refactored the scheduler and tightened the watcher backstop.',
        ],
        ids=['empty', 'title-only-reference', 'no-reference-at-all'],
    )
    def test_underivable_content_resolves_to_none(self, content):
        resolution = resolve_referents(
            declared=None, metadata={}, content=content, group_id=GROUP
        )
        assert resolution.referents == ()
        assert resolution.source == 'none'
        assert resolution.conflicts == ()
        assert resolution.ambiguous == ()


class TestDerivedPrecisionPassthrough:
    """β's precision narrowings, re-asserted at γ's OWN boundary.

    This is the capability manifest's ``scan-content-precision`` check — "a
    negative fixture property" — pinned through ``resolve_referents`` rather
    than through ``scan_content``, so a future change to the RESOLVER cannot
    widen β's precision behind its back. Consumers of this module perform
    destructive edge surgery: a false positive misattributes a fact, which is
    the same bug in the opposite direction.
    """

    @pytest.mark.parametrize(
        'content',
        [
            'raised at graphiti_client.py:2091',
            'see example.com:8080 for the dashboard',
            'landed at 12:30',
            'the w6:2 column',
            'subtask 5 is done',
            'multitask 3 is a lookalike',
        ],
        ids=[
            'source-location',
            'url-authority',
            'clock-time',
            'short-non-project-token',
            'word-glued-subtask',
            'word-glued-multitask',
        ],
    )
    def test_colon_bearing_and_glued_noise_yields_no_referents(self, content):
        resolution = resolve_referents(
            declared=None, metadata={}, content=content, group_id=GROUP
        )
        assert resolution.referents == ()
        assert resolution.source == 'none'

    def test_a_path_shaped_group_id_yields_the_empty_resolution(self):
        """Inherited verbatim from β's conservative refusal: without a
        trustworthy local project id, local and foreign references cannot be
        told apart, so the resolver reports nothing rather than guessing."""
        resolution = resolve_referents(
            declared=None,
            metadata={},
            content='blocked on task 3127',
            group_id='-home-leo-src-dark-factory',
        )
        assert resolution.referents == ()
        assert resolution.source == 'none'


class TestAmbiguousReferentsAreRecordedNotGuessed:
    """A number claimed BOTH bare and foreign-qualified is genuinely ambiguous.

    Pins the PRD row "ref routed to .ambiguous; treated as undeclared;
    recorded, not guessed". An ambiguous referent is NEVER promoted into
    ``.referents`` — refusing both sides is the honest answer, since both are
    derived from the same ambiguous prose and handing over the local one as
    clean evidence would be a confidently-wrong answer, not a safe fallback.
    """

    def test_a_contested_number_is_reported_but_never_resolved(self):
        resolution = resolve_referents(
            declared=None,
            metadata={},
            content='task 2500 duplicates dark_factory:2500',
            group_id='reify',
        )
        assert resolution.referents == ()
        assert resolution.source == 'none'
        assert resolution.ambiguous == (
            Referent(kind='task', number='2500'),
            Referent(kind='task', project_id='dark_factory', number='2500'),
        )

    def test_a_declaration_free_contest_is_still_reported_under_metadata(self):
        """``.ambiguous`` is the scan's verbatim answer on EVERY path, so a
        caller never has to check ``.source`` before trusting it — that would
        force it to re-derive the scan itself (a second scan site, INV-5)."""
        resolution = resolve_referents(
            declared=None,
            metadata={'task_id': 4242},
            content='task 2500 duplicates dark_factory:2500',
            group_id='reify',
        )
        assert resolution.source == 'metadata'
        assert resolution.ambiguous == (
            Referent(kind='task', number='2500'),
            Referent(kind='task', project_id='dark_factory', number='2500'),
        )

    def test_one_contested_number_never_suppresses_an_unrelated_clean_ref(self):
        """The contest is per-NUMBER, not per-content: a clean referent
        elsewhere in the same body still resolves, so one ambiguous mention
        cannot silently empty an otherwise usable referent set."""
        resolution = resolve_referents(
            declared=None,
            metadata={},
            content='Task 3127 landed. Separately, task 2500 duplicates dark_factory:2500.',
            group_id='reify',
        )
        assert resolution.referents == (Referent(kind='task', number='3127'),)
        assert resolution.source == 'derived'
        assert resolution.ambiguous == (
            Referent(kind='task', number='2500'),
            Referent(kind='task', project_id='dark_factory', number='2500'),
        )


class TestMetadataBridgeAcceptedShapes:
    """``metadata['task_id']`` bridges the ambient harness task into a referent.

    ``_normalize_task_id_metadata`` (services/memory_service.py) documents this
    path as carrying a SCALAR int or str, str-coerced at the write boundary, so
    those are the shapes the bridge accepts. Label spellings go through β's
    ``parse_node_name`` first, at zero vocabulary cost.
    """

    @pytest.mark.parametrize(
        'task_id',
        [3127, '3127', ' 3127 ', 'Task 3127', 'task #3127', 'Task: 3127'],
        ids=['int', 'str', 'padded', 'node-name', 'hash-spelled', 'colon-spelled'],
    )
    def test_scalar_and_label_spellings_bridge_to_the_task_referent(self, task_id):
        resolution = resolve_referents(
            declared=None, metadata={'task_id': task_id}, content='', group_id=GROUP
        )
        assert resolution.referents == (Referent(kind='task', number='3127'),)
        assert resolution.source == 'metadata'

    def test_zero_padding_is_preserved_verbatim(self):
        """Never int-normalized: '0132' is a DIFFERENT referent from '132', so
        coercing it would silently repoint the write at another task."""
        resolution = resolve_referents(
            declared=None, metadata={'task_id': '0132'}, content='', group_id=GROUP
        )
        assert resolution.referents == (Referent(kind='task', number='0132'),)

    def test_a_qualified_task_id_bridges_to_a_foreign_referent(self):
        """``parse_node_name`` already owns this spelling and preserves the
        qualifier as a different-project signal; reusing it keeps the
        vocabulary at one site."""
        resolution = resolve_referents(
            declared=None, metadata={'task_id': 'reify:3127'}, content='', group_id=GROUP
        )
        assert resolution.referents == (
            Referent(kind='task', project_id='reify', number='3127'),
        )
        assert resolution.source == 'metadata'


class TestMetadataBridgePrecedence:
    """Metadata OUTRANKS the derived scan, and can never conflict with it."""

    def test_metadata_wins_over_a_disagreeing_scan(self):
        resolution = resolve_referents(
            declared=None,
            metadata={'task_id': 3668},
            content='Task 2500 is the one that landed.',
            group_id=GROUP,
        )
        assert resolution.referents == (Referent(kind='task', number='3668'),)
        assert resolution.source == 'metadata'

    def test_metadata_never_produces_a_conflict(self):
        """Ambient harness state is NOT a claim about the prose: an agent
        working on task 3668 legitimately writes memories about Task 2500.
        Reconciling metadata against content would reject a large fraction of
        correct writes — the "reject on absence loses memories" failure
        resolved decision 3 forbids."""
        resolution = resolve_referents(
            declared=None,
            metadata={'task_id': 3668},
            content='Task 2500 is the one that landed.',
            group_id=GROUP,
        )
        assert resolution.conflicts == ()


class TestMetadataBridgeDropsUnusableValues:
    """An unparseable ``task_id`` is DROPPED — it falls through, never raises.

    Asymmetric with ``declared`` ON PURPOSE: ``declared`` is an explicit caller
    assertion, while ``metadata`` is ambient harness state the writer may not
    control, and failing a write over odd ambient metadata would lose memories
    for no gain. The degradation stays OBSERVABLE rather than silent — ι sees a
    lower 'metadata' share, and the resolution still falls through to the
    derived scan rather than short-circuiting to 'none'.
    """

    @pytest.mark.parametrize(
        'metadata',
        [
            {},
            {'task_id': None},
            {'task_id': ''},
            {'task_id': '   '},
            {'task_id': 'the merge-lane hardening task'},
            {'task_id': '3127a'},
            {'task_id': True},
            {'task_id': [3127]},
            {'task_id': {'id': 3127}},
            {'task_id': '٣'},
            {'task_id': 3.0},
        ],
        ids=[
            'absent',
            'none',
            'empty',
            'whitespace',
            'title-prose',
            'trailing-junk',
            'bool-must-not-become-task-1',
            'list',
            'dict',
            'non-ascii-digit',
            'float',
        ],
    )
    def test_unusable_values_yield_the_empty_resolution(self, metadata):
        resolution = resolve_referents(
            declared=None, metadata=metadata, content='', group_id=GROUP
        )
        assert resolution.referents == ()
        assert resolution.source == 'none'

    def test_a_non_scalar_task_id_is_never_str_coerced_into_a_garbage_referent(self):
        """``_normalize_task_id_metadata`` flags list/tuple values as out of
        contract precisely because they would ``str()``-coerce to a Python
        repr; the bridge must drop them, never mint ``'[5040, 5149]'``."""
        resolution = resolve_referents(
            declared=None, metadata={'task_id': [5040, 5149]}, content='', group_id=GROUP
        )
        assert resolution.referents == ()

    def test_a_dropped_task_id_falls_through_to_the_derived_scan(self):
        """Proves the bridge falls THROUGH rather than short-circuiting to
        'none' — an unusable ambient value must not blind the resolver to what
        the prose plainly says."""
        resolution = resolve_referents(
            declared=None,
            metadata={'task_id': 'the merge-lane hardening task'},
            content='Fixed the bug in Task 3127.',
            group_id=GROUP,
        )
        assert resolution.referents == (Referent(kind='task', number='3127'),)
        assert resolution.source == 'derived'
