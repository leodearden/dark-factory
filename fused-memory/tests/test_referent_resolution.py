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
from fused_memory.utils.validation import InputValidationError

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


class TestSelfQualifiedReclassificationIsSourceInvariant:
    """One spelling denotes ONE node, whichever source wins.

    ``scan_content`` reclassifies a SELF-qualified reference — one whose
    qualifier is the local project — back to an own-project referent, and the
    declared path mirrors it (see
    :meth:`TestDeclaredPath.test_a_self_qualified_project_id_is_reclassified_as_own_project`).
    The metadata bridge was the one path that skipped the rule, so
    ``metadata={'task_id': 'dark_factory:3127'}`` in group ``dark_factory``
    produced a FOREIGN referent whose ``node_name`` is ``'dark_factory:3127'``
    while the other two paths produced ``'Task 3127'``.

    That is the exact failure class this PRD exists to prevent, arriving
    through the identity axis rather than the attribution one: a downstream
    verifier or edge-surgery consumer keyed on ``node_name`` would hunt a
    foreign node that does not exist inside the dark_factory graph and conclude
    the fact is unattributable — a wrong answer produced by the mechanism meant
    to prevent wrong answers, varying with which source happened to win.
    ``.source`` must select the AUTHORITY, never the IDENTITY.
    """

    def test_all_three_paths_agree_on_a_self_qualified_spelling(self):
        """The load-bearing assertion. A future divergence in ANY path fails
        here, which is why this is one test over three inputs rather than three
        unrelated per-path tests."""
        by_path = {
            'declared': resolve_referents(
                declared=[{'kind': 'task', 'id': 3127, 'project_id': 'dark_factory'}],
                metadata={},
                content='',
                group_id=GROUP,
            ),
            'metadata': resolve_referents(
                declared=None,
                metadata={'task_id': 'dark_factory:3127'},
                content='',
                group_id=GROUP,
            ),
            'derived': resolve_referents(
                declared=None,
                metadata={},
                content='see dark_factory:3127',
                group_id=GROUP,
            ),
        }
        for source, resolution in by_path.items():
            assert resolution.source == source
            assert resolution.referents == (Referent(kind='task', number='3127'),), source
            assert resolution.referents[0].project_id == '', source
            assert resolution.referents[0].node_name == 'Task 3127', source

    def test_no_path_mints_a_foreign_node_for_the_task_2500_spelling(self):
        """The SECOND source-invariance axis: a task-VOCABULARY qualifier.

        ``'task'`` is task vocabulary, never a project key, so NO path may
        yield a foreign ``'task:2500'`` node — a node that cannot exist, which
        a consumer keying destructive edge surgery on ``node_name`` would hunt
        anyway. The metadata and derived paths reach the LOCAL ``Task 2500``
        (their parsers claim the local label first); the declared path REFUSES
        outright, because there ``'task'`` arrives in a field explicitly named
        ``project_id`` — a category error rather than an ambiguous spelling —
        and this path's contract is to raise on a malformed entry rather than
        silently rewrite the caller's own assertion.

        The declared path was the one that skipped the rule, mirroring the
        metadata bridge's break fixed in b2cfe5518f.
        """
        with pytest.raises(InputValidationError):
            resolve_referents(
                declared=[{'kind': 'task', 'id': 2500, 'project_id': 'task'}],
                metadata={},
                content='',
                group_id=GROUP,
            )

        by_path = {
            'metadata': resolve_referents(
                declared=None, metadata={'task_id': 'task:2500'}, content='', group_id=GROUP
            ),
            'derived': resolve_referents(
                declared=None, metadata={}, content='see task:2500', group_id=GROUP
            ),
        }
        for source, resolution in by_path.items():
            assert resolution.source == source, source
            assert resolution.referents == (Referent(kind='task', number='2500'),), source
            assert resolution.referents[0].node_name == 'Task 2500', source

    def test_a_non_canonical_self_qualified_task_id_is_canonicalized_first(self):
        """Case and hyphen/underscore spelling are rendering choices, not
        different projects, so canonicalization has to happen BEFORE the
        is-this-us comparison — mirroring the declared path's
        ``test_a_non_canonical_project_id_is_canonicalized``."""
        resolution = resolve_referents(
            declared=None,
            metadata={'task_id': 'Dark-Factory:3127'},
            content='',
            group_id=GROUP,
        )
        assert resolution.referents == (Referent(kind='task', number='3127'),)
        assert resolution.referents[0].node_name == 'Task 3127'
        assert resolution.source == 'metadata'

    def test_a_genuinely_foreign_qualifier_is_not_reclassified(self):
        """REGRESSION GUARD: the fix must not over-reach and collapse FOREIGN
        referents onto the local project. Re-pins
        ``test_a_qualified_task_id_bridges_to_a_foreign_referent`` under the
        reclassifying code path."""
        resolution = resolve_referents(
            declared=None, metadata={'task_id': 'reify:3127'}, content='', group_id=GROUP
        )
        assert resolution.referents == (
            Referent(kind='task', project_id='reify', number='3127'),
        )
        assert resolution.source == 'metadata'

    def test_a_path_shaped_group_id_neither_raises_nor_reclassifies(self):
        """Mirrors ``_declared_referents``'s ``''``-sentinel fallback: the scan
        is empty for a path-shaped group, but an explicit metadata value is
        still bridged, and the sentinel must never compare equal to a real
        canonical project id — otherwise an untrustworthy local project id
        would start silently collapsing foreign referents onto it, strictly
        worse than the bug being fixed."""
        resolution = resolve_referents(
            declared=None, metadata={'task_id': 'reify:3127'}, content='', group_id='../etc'
        )
        assert resolution.referents == (
            Referent(kind='task', project_id='reify', number='3127'),
        )
        assert resolution.source == 'metadata'

    def test_a_path_shaped_qualifier_is_still_dropped_rather_than_raised(self):
        """Pre-existing behaviour (``parse_node_name`` answers None for these),
        pinned here so the reclassification edit cannot turn a DROP into a RAISE
        on the ambient-metadata path — which is documented as never failing a
        write."""
        resolution = resolve_referents(
            declared=None, metadata={'task_id': '../etc:3127'}, content='', group_id=GROUP
        )
        assert resolution.referents == ()
        assert resolution.source == 'none'


class TestDeclaredPath:
    """An explicit declaration is the strongest source and is honoured verbatim.

    Entry shape is the PRD's: ``{'kind': 'task', 'id': 3127}``, with an
    optional ``project_id``.
    """

    @pytest.mark.parametrize(
        'entry',
        [
            {'kind': 'task', 'id': 3127},
            {'kind': 'task', 'id': '3127'},
            {'id': 3127},
        ],
        ids=['int-id', 'str-id', 'kind-omitted'],
    )
    def test_a_well_formed_entry_resolves_to_its_referent(self, entry):
        """``kind`` defaults to 'task' — the same default ``Referent`` itself
        carries, so canonical_labels stays the single site for it."""
        resolution = resolve_referents(
            declared=[entry], metadata={}, content='', group_id=GROUP
        )
        assert resolution.referents == (Referent(kind='task', number='3127'),)
        assert resolution.source == 'declared'

    def test_digits_are_preserved_verbatim(self):
        """'0132' is a DIFFERENT referent from '132'; int-normalizing a
        declared id would silently repoint the caller's own assertion."""
        padded = resolve_referents(
            declared=[{'id': '0132'}], metadata={}, content='', group_id=GROUP
        )
        bare = resolve_referents(declared=[{'id': 132}], metadata={}, content='', group_id=GROUP)
        assert padded.referents == (Referent(kind='task', number='0132'),)
        assert bare.referents == (Referent(kind='task', number='132'),)
        assert padded.referents != bare.referents

    def test_an_optional_project_id_declares_a_foreign_referent(self):
        resolution = resolve_referents(
            declared=[{'kind': 'task', 'id': 132, 'project_id': 'reify'}],
            metadata={},
            content='',
            group_id=GROUP,
        )
        assert resolution.referents[0].node_name == 'reify:132'

    def test_a_non_canonical_project_id_is_canonicalized(self):
        """Case and hyphen/underscore spelling are rendering choices, not
        different projects — the same normalization ``scan_content`` applies
        before comparing, so the two sides stay comparable in step-13."""
        resolution = resolve_referents(
            declared=[{'kind': 'task', 'id': 132, 'project_id': 'Reify-X'}],
            metadata={},
            content='',
            group_id=GROUP,
        )
        assert resolution.referents == (
            Referent(kind='task', project_id='reify_x', number='132'),
        )

    def test_a_self_qualified_project_id_is_reclassified_as_own_project(self):
        """Mirrors ``scan_content``'s self-qualified reclassification EXACTLY.
        Without it a self-qualified declaration could never compare equal to
        the scanned form, and the conflict check would fire on a caller who
        was right."""
        resolution = resolve_referents(
            declared=[{'kind': 'task', 'id': 132, 'project_id': 'Dark-Factory'}],
            metadata={},
            content='',
            group_id=GROUP,
        )
        assert resolution.referents == (Referent(kind='task', number='132'),)
        assert resolution.referents[0].node_name == 'Task 132'

    def test_duplicate_entries_collapse_preserving_first_seen_order(self):
        """De-duplicated on (kind, project_id, number) — the same key and the
        same discipline ``scan_content``'s dedup uses."""
        resolution = resolve_referents(
            declared=[
                {'kind': 'task', 'id': 3127},
                {'kind': 'task', 'id': 2500},
                {'kind': 'task', 'id': '3127'},
            ],
            metadata={},
            content='',
            group_id=GROUP,
        )
        assert resolution.referents == (
            Referent(kind='task', number='3127'),
            Referent(kind='task', number='2500'),
        )


class TestDeclaredPrecedence:
    """A declaration beats BOTH lower sources."""

    def test_declared_outranks_metadata_and_the_scan(self):
        resolution = resolve_referents(
            declared=[{'kind': 'task', 'id': 5000}],
            metadata={'task_id': 3668},
            content='Fixed the bug in Task 3127.',
            group_id=GROUP,
        )
        assert resolution.referents == (Referent(kind='task', number='5000'),)
        assert resolution.source == 'declared'

    def test_ambiguity_is_still_reported_under_a_declaration(self):
        """The always-scan contract: ``.ambiguous`` reads the same whatever
        source won, so a declaring caller never has to re-derive the scan
        itself to learn the content was contested."""
        resolution = resolve_referents(
            declared=[{'kind': 'task', 'id': 5000}],
            metadata={},
            content='task 2500 duplicates dark_factory:2500',
            group_id='reify',
        )
        assert resolution.source == 'declared'
        assert resolution.ambiguous == (
            Referent(kind='task', number='2500'),
            Referent(kind='task', project_id='dark_factory', number='2500'),
        )


class TestMalformedDeclarationIsRejectedLoudly:
    """A malformed ``declared`` entry RAISES; it is never dropped, and never
    smuggled into ``.conflicts``.

    Three failure shapes were available and only one is honest. Reporting a
    typo through ``.conflicts`` would tell the agent "your declaration
    contradicts your prose" and send it hunting a semantic disagreement that
    does not exist — and a malformed entry is not a ``Referent`` anyway, which
    is what ``.conflicts`` is typed as. Silently DROPPING a bad entry is worse:
    it flips ``.source`` down to 'metadata' or 'derived', loses the caller's
    intent invisibly, and corrupts the very declaration-rate telemetry ι is
    being built to read — the silent-degradation failure this repo's
    loud-over-silent norm forbids.

    ``InputValidationError`` is reused rather than a new type: it documents
    itself as exactly this, it subclasses ValueError so ``except ValueError``
    callers keep working, and δ's boundary then sees ONE exception type.
    """

    @pytest.mark.parametrize(
        'declared',
        [
            {'kind': 'task', 'id': 3127},
            'task:3127',
            3127,
        ],
        ids=['bare-dict', 'bare-string', 'bare-int'],
    )
    def test_a_non_list_declared_is_rejected(self, declared):
        with pytest.raises(InputValidationError):
            resolve_referents(
                declared=declared,  # type: ignore[arg-type]
                metadata={},
                content='',
                group_id=GROUP,
            )

    @pytest.mark.parametrize(
        'entry',
        [3127, 'Task 3127', None, ['task', 3127]],
        ids=['int', 'string', 'none', 'list'],
    )
    def test_a_non_dict_entry_is_rejected(self, entry):
        with pytest.raises(InputValidationError):
            resolve_referents(
                declared=[entry],  # type: ignore[list-item]
                metadata={},
                content='',
                group_id=GROUP,
            )

    @pytest.mark.parametrize(
        'entry',
        [
            {'kind': 'task'},
            {'id': ''},
            {'id': '  '},
            {'id': '3127a'},
            {'id': 'Task 3127'},
            {'id': '٣'},
            {'id': 3.0},
            {'id': None},
            {'id': True},
        ],
        ids=[
            'missing-id',
            'empty-id',
            'whitespace-id',
            'trailing-junk-id',
            'label-spelled-id',
            'non-ascii-digit-id',
            'float-id',
            'none-id',
            'bool-must-not-become-task-1',
        ],
    )
    def test_an_unusable_id_is_rejected(self, entry):
        with pytest.raises(InputValidationError):
            resolve_referents(declared=[entry], metadata={}, content='', group_id=GROUP)

    @pytest.mark.parametrize(
        'entry',
        [
            {'kind': ['task'], 'id': 1},
            {'kind': {'a': 1}, 'id': 1},
            {'kind': 3, 'id': 1},
            {'kind': None, 'id': 1},
            {'kind': True, 'id': 1},
        ],
        ids=['list', 'dict', 'int', 'none', 'bool'],
    )
    def test_a_non_string_kind_is_rejected(self, entry):
        """``declared`` is caller-supplied JSON off an MCP tool argument, so a
        confused agent can reach every one of these. An UNHASHABLE kind is the
        sharp case: ``Referent.__post_init__``'s ``self.kind not in
        _KIND_LABELS`` membership test raises ``TypeError``, which is not a
        ``ValueError``, so the existing ``except ValueError`` around the
        construction structurally cannot convert it — and δ's ``except
        InputValidationError`` gate would take an unhandled exception instead of
        emitting a remediation message."""
        with pytest.raises(InputValidationError):
            resolve_referents(declared=[entry], metadata={}, content='', group_id=GROUP)

    @pytest.mark.parametrize(
        'entry',
        [
            {'id': 1, 'project_id': 5},
            {'id': 1, 'project_id': ['reify']},
            {'id': 1, 'project_id': {'a': 1}},
            {'id': 1, 'project_id': True},
        ],
        ids=['int', 'list', 'dict', 'bool'],
    )
    def test_a_truthy_non_string_project_id_is_rejected(self, entry):
        """These leak ``AttributeError: ... has no attribute 'startswith'``
        from inside ``is_path_shaped_name`` — again a type δ's gate does not
        catch."""
        with pytest.raises(InputValidationError):
            resolve_referents(declared=[entry], metadata={}, content='', group_id=GROUP)

    @pytest.mark.parametrize(
        'entry',
        [
            {'id': 1, 'project_id': 0},
            {'id': 1, 'project_id': []},
            {'id': 1, 'project_id': {}},
            {'id': 1, 'project_id': False},
        ],
        ids=['zero', 'empty-list', 'empty-dict', 'false'],
    )
    def test_a_falsy_non_string_project_id_is_rejected(self, entry):
        """The sharper half, and the one a naive fix misses: ``entry.get(
        'project_id') or ''`` swallows every FALSY value, so a ``[]``/``0``
        typo SILENTLY mints an OWN-project referent — the identical
        confidently-wrong-about-which-project outcome the unknown-key
        ('projectId') rejection already guards against, arriving through a
        different door. Validating AFTER the ``or ''`` would still let these
        through, which is why the check must read the RAW value."""
        with pytest.raises(InputValidationError):
            resolve_referents(declared=[entry], metadata={}, content='', group_id=GROUP)

    @pytest.mark.parametrize(
        'entry',
        [
            {'id': 1},
            {'kind': 'task', 'id': 1},
            {'id': 1, 'project_id': None},
            {'id': 1, 'project_id': ''},
        ],
        ids=['bare', 'explicit-kind', 'none-project-id', 'empty-project-id'],
    )
    def test_legitimate_absence_is_still_accepted(self, entry):
        """POSITIVE non-regression, so the type checks harden types without
        over-reaching into rejecting a caller who simply omitted the key.
        ``None`` and ``''`` are legitimate absence, not a typo."""
        resolution = resolve_referents(
            declared=[entry], metadata={}, content='', group_id=GROUP
        )
        assert resolution.referents == (Referent(kind='task', number='1'),)
        assert resolution.source == 'declared'

    def test_an_unknown_key_is_rejected_rather_than_ignored(self):
        """A camelCase typo would otherwise SILENTLY resolve to an
        OWN-project referent — a confidently wrong answer about which project
        a fact belongs to, which is the exact class of error this PRD exists
        to eliminate."""
        with pytest.raises(InputValidationError):
            resolve_referents(
                declared=[{'kind': 'task', 'id': 3127, 'projectId': 'reify'}],
                metadata={},
                content='',
                group_id=GROUP,
            )

    def test_an_unregistered_kind_is_rejected_as_an_input_validation_error(self):
        """Caught from ``Referent.__post_init__`` and re-raised, so δ's
        boundary sees ONE exception type rather than a bare ValueError leaking
        through from the vocabulary module."""
        with pytest.raises(InputValidationError) as excinfo:
            resolve_referents(
                declared=[{'kind': 'escalation', 'id': 3127}],
                metadata={},
                content='',
                group_id=GROUP,
            )
        # __post_init__'s message already names the registered kinds and where
        # to add one; re-raising must preserve that remediation, not replace it.
        assert '_KIND_LABELS' in str(excinfo.value)

    def test_a_path_shaped_project_id_is_refused_not_canonicalized(self):
        """Normalizing a mangled path would mint a NEW, wrong canonical key
        (RCA §4) — the referent would then confidently name a project that
        does not exist."""
        with pytest.raises(InputValidationError):
            resolve_referents(
                declared=[{'kind': 'task', 'id': 132, 'project_id': '../etc'}],
                metadata={},
                content='',
                group_id=GROUP,
            )

    @pytest.mark.parametrize(
        'project_id', ['task', 'Task', 'TASK', 'tasks', 'subtask', 'sub-task', 'Sub-Tasks']
    )
    def test_a_task_vocabulary_project_id_is_refused(self, project_id):
        """A task-vocabulary word is not a project key.

        canonical_labels refuses this qualifier in BOTH ``parse_node_name``
        and ``scan_content``; the declared path was the one source that
        accepted it, so ``{'id': 2500, 'project_id': 'task'}`` minted a
        FOREIGN ``'task:2500'`` node that cannot exist while the identical
        spelling resolved to the LOCAL ``'Task 2500'`` through both other
        paths. Every spelling collapses onto the check because it is applied
        to the CANONICALIZED value.
        """
        with pytest.raises(InputValidationError) as excinfo:
            resolve_referents(
                declared=[{'kind': 'task', 'id': 2500, 'project_id': project_id}],
                metadata={},
                content='',
                group_id=GROUP,
            )
        assert 'project' in str(excinfo.value)

    def test_a_real_project_that_merely_starts_with_task_is_accepted(self):
        """The NEGATIVE half, and why the predicate is a ``fullmatch`` rather
        than a prefix test: 'taskmaster' is a real project."""
        resolution = resolve_referents(
            declared=[{'kind': 'task', 'id': 2500, 'project_id': 'taskmaster'}],
            metadata={},
            content='',
            group_id=GROUP,
        )
        assert resolution.referents == (
            Referent(kind='task', project_id='taskmaster', number='2500'),
        )

    @pytest.mark.parametrize(
        'project_id',
        ['   ', 'foo bar', '..', 'x;y', 'a\nb', '`id`', "it's"],
        ids=[
            'whitespace-only',
            'embedded-space',
            'dot-dot',
            'semicolon',
            'newline',
            'backtick',
            'quote',
        ],
    )
    def test_a_charset_invalid_project_id_is_refused(self, project_id):
        """``canonicalize_project_id`` is a NORMALIZER, not a validator — it
        says so itself and defers the charset allowlist to
        ``validate_project_id``. Without that second call each of these minted
        a foreign referent naming a project that cannot exist ('   ' ->
        node_name '   :132', 'foo bar' -> 'foo bar:132'), which is the one hole
        left in this path's otherwise TOTAL contract against caller-supplied
        MCP JSON. The newline/backtick/quote rows are the prompt-injection
        vectors the allowlist exists to block.
        """
        with pytest.raises(InputValidationError) as excinfo:
            resolve_referents(
                declared=[{'kind': 'task', 'id': 132, 'project_id': project_id}],
                metadata={},
                content='',
                group_id=GROUP,
            )
        assert 'project_id' in str(excinfo.value)

    def test_a_legitimate_hyphenated_project_id_is_still_accepted(self):
        """The NEGATIVE half: validation runs on the CANONICALIZED value, so a
        hyphenated spelling must survive it rather than being rejected for the
        hyphen it no longer has. 'reify-mirror' is a well-formed project key."""
        resolution = resolve_referents(
            declared=[{'kind': 'task', 'id': 132, 'project_id': 'reify-mirror'}],
            metadata={},
            content='',
            group_id=GROUP,
        )
        assert resolution.referents == (
            Referent(kind='task', project_id='reify_mirror', number='132'),
        )

    def test_a_long_project_id_is_accepted_because_no_length_rule_exists(self):
        """Pins the deliberate NON-rule. ``_validate_identifier`` enforces
        charset and non-emptiness only, so a long-but-well-formed qualifier is
        accepted here too. Inventing a length cap at THIS site would be exactly
        the second copy of a validation rule INV-5 forbids — if a cap is ever
        wanted it belongs in validation.py, and this test will fail loudly and
        point at the right module when that happens.
        """
        resolution = resolve_referents(
            declared=[{'kind': 'task', 'id': 132, 'project_id': 'a' * 300}],
            metadata={},
            content='',
            group_id=GROUP,
        )
        assert resolution.referents[0].project_id == 'a' * 300

    def test_the_message_names_the_offending_entry_and_a_remediation_hint(self):
        """Follows ``require_full_uuid``'s shape: a single module-level hint
        constant folded into the raised message, so the remediation text has
        one site."""
        with pytest.raises(InputValidationError) as excinfo:
            resolve_referents(
                declared=[{'kind': 'task', 'id': '3127a'}], metadata={}, content='', group_id=GROUP
            )
        message = str(excinfo.value)
        assert '3127a' in message
        assert "'kind'" in message and "'id'" in message and "'project_id'" in message

    def test_an_oversized_entry_cannot_blow_up_the_error_message(self):
        """``_safe_repr``'s truncation discipline, copied so a pathological
        declaration cannot produce an unbounded exception message."""
        with pytest.raises(InputValidationError) as excinfo:
            resolve_referents(
                declared=[{'id': 'x' * 10_000}], metadata={}, content='', group_id=GROUP
            )
        assert len(str(excinfo.value)) < 2_000

    def test_a_malformed_entry_never_degrades_the_source_or_lands_in_conflicts(self):
        """The NEGATIVE assertion: the call RAISES rather than returning, so a
        bad entry can neither flip ``.source`` down to a lower source nor be
        reported as a semantic conflict."""
        with pytest.raises(InputValidationError):
            resolve_referents(
                declared=[{'kind': 'task', 'id': 3127}, {'id': 'oops'}],
                metadata={'task_id': 3668},
                content='Fixed the bug in Task 3127.',
                group_id=GROUP,
            )

    @pytest.mark.parametrize(
        'declared',
        [
            'task:3127',
            [3127],
            [{'kind': 'task', 'id': 3127, 'projectId': 'reify'}],
            [{'kind': 'task'}],
            [{'id': '3127a'}],
            [{'id': True}],
            [{'kind': ['task'], 'id': 1}],
            [{'kind': {'a': 1}, 'id': 1}],
            [{'kind': 3, 'id': 1}],
            [{'id': 1, 'project_id': 5}],
            [{'id': 1, 'project_id': []}],
            [{'id': 1, 'project_id': False}],
            [{'kind': 'escalation', 'id': 3127}],
            [{'kind': 'task', 'id': 132, 'project_id': '../etc'}],
            [{'kind': 'task', 'id': 2500, 'project_id': 'task'}],
            [{'kind': 'task', 'id': 2500, 'project_id': 'Sub-Tasks'}],
            [{'id': 1, 'project_id': '   '}],
            [{'id': 1, 'project_id': 'foo bar'}],
        ],
        ids=[
            'non-list',
            'non-dict-entry',
            'unknown-key',
            'missing-id',
            'bad-id',
            'bool-id',
            'unhashable-kind-list',
            'unhashable-kind-dict',
            'non-str-kind',
            'truthy-non-str-project-id',
            'falsy-non-str-project-id',
            'false-project-id',
            'unregistered-kind',
            'path-shaped-project-id',
            'task-vocabulary-project-id',
            'task-vocabulary-project-id-noncanonical',
            'whitespace-only-project-id',
            'charset-invalid-project-id',
        ],
    )
    def test_every_malformed_shape_leaks_exactly_one_exception_type(self, declared):
        """THE CONTRACT ITSELF. ``_declared_referents``'s docstring promises
        "δ's boundary sees ONE exception type"; this is what makes the promise
        CHECKABLE rather than aspirational, and it is what the previously
        all-passing suite lacked.

        Caught as bare ``Exception`` and re-asserted, so a leak reports as a
        readable "left as TypeError" failure rather than as an error δ's gate
        would silently take unhandled in production.
        """
        try:
            resolve_referents(declared=declared, metadata={}, content='', group_id=GROUP)
        except Exception as exc:  # noqa: BLE001 — the assertion IS the type check
            assert isinstance(exc, InputValidationError), (
                f'{declared!r} left _declared_referents as '
                f'{type(exc).__name__}, which δ\'s `except InputValidationError` '
                f'gate does not catch: {exc}'
            )
        else:
            pytest.fail(f'{declared!r} was accepted rather than rejected')


class TestDeclaredTriState:
    """``[]`` and ``None`` are behaviourally DISTINCT (resolved decision 2).

    ``None`` means "never considered" and falls through the chain. ``[]`` means
    "considered, none apply" and is honoured as a genuine declaration, so ι's
    telemetry can separate "the agent considered referents and found none
    apply" from ``source='none'`` ("nobody declared anything and nothing was
    derivable"). Under ``if declared:`` the two collapse and the tri-state
    becomes a docstring claim rather than a behaviour — which is what these
    tests exist to prevent.
    """

    def test_an_empty_declaration_is_honoured_rather_than_falling_through(self):
        resolution = resolve_referents(
            declared=[],
            metadata={'task_id': 3668},
            content='Fixed the bug in Task 3127.',
            group_id=GROUP,
        )
        assert resolution.source == 'declared'
        assert resolution.referents == ()

    def test_an_empty_declaration_never_conflicts_with_the_scan(self):
        """Rejecting here would be rejecting on ABSENCE, which resolved
        decision 3 forbids: agents that do not retry — notably /reflect at
        session end — lose the memory outright."""
        resolution = resolve_referents(
            declared=[],
            metadata={},
            content='Fixed the bug in Task 3127.',
            group_id=GROUP,
        )
        assert resolution.conflicts == ()

    def test_none_falls_through_to_the_metadata_bridge(self):
        """The distinguishing half: identical inputs, only ``declared``
        differs, and the resolution lands on a different source."""
        resolution = resolve_referents(
            declared=None,
            metadata={'task_id': 3668},
            content='Fixed the bug in Task 3127.',
            group_id=GROUP,
        )
        assert resolution.source == 'metadata'
        assert resolution.referents == (Referent(kind='task', number='3668'),)

    def test_an_empty_declaration_is_distinguishable_from_source_none(self):
        """With nothing else available either, the two rows STILL differ — this
        is the distinction ι counts as "the agent considered referents" versus
        "the agent never looked"."""
        considered = resolve_referents(
            declared=[], metadata={}, content='', group_id=GROUP
        )
        never_looked = resolve_referents(
            declared=None, metadata={}, content='', group_id=GROUP
        )
        assert considered.source == 'declared'
        assert never_looked.source == 'none'
        assert considered.referents == never_looked.referents == ()

    def test_an_empty_declaration_still_reports_ambiguity_from_the_scan(self):
        resolution = resolve_referents(
            declared=[],
            metadata={},
            content='task 2500 duplicates dark_factory:2500',
            group_id='reify',
        )
        assert resolution.source == 'declared'
        assert resolution.ambiguous == (
            Referent(kind='task', number='2500'),
            Referent(kind='task', project_id='dark_factory', number='2500'),
        )


class TestConflictsAreReportedNotEnforced:
    """``.conflicts`` names the DECLARED referents the scanned prose contradicts.

    γ REPORTS; δ's ``_entities_gate`` decides whether to reject. A resolver
    that silently degraded to the scan on conflict would produce a write the
    caller never asked for, so ``.referents`` is deliberately left untouched.
    """

    def test_the_prd_headline_row_names_the_declared_referent(self):
        """Content cites Task 3127, the caller declared 3129 — the
        adjacent-number misattribution this PRD exists to prevent."""
        resolution = resolve_referents(
            declared=[{'kind': 'task', 'id': 3129}],
            metadata={},
            content='Fixed the bug in Task 3127.',
            group_id=GROUP,
        )
        assert resolution.conflicts == (Referent(kind='task', number='3129'),)

    def test_both_numbers_are_recoverable_so_the_gate_can_name_them_both(self):
        """δ's structured error must name the declared referent AND what the
        prose actually said. The declared side comes from ``.conflicts``; the
        prose side is the derived resolution of the same content, which is
        exactly what δ gets by resolving with ``declared=None``."""
        content = 'Fixed the bug in Task 3127.'
        declared = resolve_referents(
            declared=[{'kind': 'task', 'id': 3129}],
            metadata={},
            content=content,
            group_id=GROUP,
        )
        prose = resolve_referents(declared=None, metadata={}, content=content, group_id=GROUP)
        assert declared.conflicts == (Referent(kind='task', number='3129'),)
        assert prose.referents == (Referent(kind='task', number='3127'),)

    def test_a_conflict_leaves_the_referents_untouched(self):
        """γ is mechanism, δ is policy: degrading to the scan here would
        silently write to a task the caller never declared."""
        resolution = resolve_referents(
            declared=[{'kind': 'task', 'id': 3129}],
            metadata={},
            content='Fixed the bug in Task 3127.',
            group_id=GROUP,
        )
        assert resolution.referents == (Referent(kind='task', number='3129'),)
        assert resolution.source == 'declared'

    def test_agreement_is_not_a_conflict(self):
        """PRD row: declared referent, correct endpoint -> declaration
        honoured, no conflict."""
        resolution = resolve_referents(
            declared=[{'kind': 'task', 'id': 3127}],
            metadata={},
            content='Fixed the bug in Task 3127.',
            group_id=GROUP,
        )
        assert resolution.conflicts == ()
        assert resolution.referents == (Referent(kind='task', number='3127'),)

    def test_a_declared_subset_of_the_scanned_referents_is_not_a_conflict(self):
        """Prose that mentions MORE than the caller declared is normal: a
        memory may cite several tasks while being ABOUT one."""
        resolution = resolve_referents(
            declared=[{'kind': 'task', 'id': 3129}],
            metadata={},
            content='Task 3127 and task 3129 both landed.',
            group_id=GROUP,
        )
        assert resolution.conflicts == ()

    def test_a_superset_conflicts_per_ref(self):
        """The adjacent-number typo is caught even when another declared
        referent IS corroborated — which is precisely what a set-level
        "declared and scanned are disjoint" verdict would miss."""
        resolution = resolve_referents(
            declared=[{'kind': 'task', 'id': 3127}, {'kind': 'task', 'id': 3129}],
            metadata={},
            content='Fixed the bug in Task 3127.',
            group_id=GROUP,
        )
        assert resolution.conflicts == (Referent(kind='task', number='3129'),)
        assert resolution.referents == (
            Referent(kind='task', number='3127'),
            Referent(kind='task', number='3129'),
        )

    @pytest.mark.parametrize(
        'content',
        [
            'the merge-lane hardening task',
            'Task θ2=2184 is the one that landed.',
            'Reworked 1251 and the watcher backstop.',
        ],
        ids=['title-only', 'greek-alias', 'bare-digit-node-name'],
    )
    def test_the_scanners_blind_list_never_manufactures_a_conflict(self, content):
        """The load-bearing consequence of resolved decision 8: the scanner
        cannot see title-only references, Greek aliases or bare-digit node
        names, so its SILENCE is uninformative and must never reject an honest
        write. An empty per-kind scan never conflicts."""
        resolution = resolve_referents(
            declared=[{'kind': 'task', 'id': 3129}],
            metadata={},
            content=content,
            group_id=GROUP,
        )
        assert resolution.conflicts == ()

    def test_precision_fixtures_never_manufacture_a_conflict(self):
        """A source location is not a task mention, so declaring 2091 against
        'graphiti_client.py:2091' must not read as a contradiction."""
        resolution = resolve_referents(
            declared=[{'kind': 'task', 'id': 2091}],
            metadata={},
            content='raised at graphiti_client.py:2091',
            group_id=GROUP,
        )
        assert resolution.conflicts == ()

    def test_the_project_axis_conflicts(self):
        """The prose names a DIFFERENT project's task with the same number —
        the cross-project collapse this PRD exists to detect, and invisible to
        any rule that compared bare numbers."""
        resolution = resolve_referents(
            declared=[{'kind': 'task', 'id': 3129}],
            metadata={},
            content='mirrors reify:3129',
            group_id=GROUP,
        )
        assert resolution.conflicts == (Referent(kind='task', number='3129'),)

    def test_a_declaration_resolves_an_ambiguity_rather_than_contradicting_it(self):
        """Membership is tested against ``refs | ambiguous``, so naming an
        ambiguous referent SETTLES the contest the prose left open — the single
        most useful thing a declaration can do."""
        resolution = resolve_referents(
            declared=[{'kind': 'task', 'id': 2500}],
            metadata={},
            content='task 2500 duplicates dark_factory:2500',
            group_id='reify',
        )
        assert resolution.conflicts == ()
        assert resolution.source == 'declared'
        assert resolution.referents == (Referent(kind='task', number='2500'),)
        assert resolution.ambiguous == (
            Referent(kind='task', number='2500'),
            Referent(kind='task', project_id='dark_factory', number='2500'),
        )

    def test_an_empty_declaration_never_conflicts(self):
        """Re-pinned here alongside the positive cases: rejecting on ABSENCE is
        what resolved decision 3 forbids."""
        resolution = resolve_referents(
            declared=[], metadata={}, content='Fixed the bug in Task 3127.', group_id=GROUP
        )
        assert resolution.conflicts == ()

    def test_conflicts_are_deduped_and_in_declared_order(self):
        resolution = resolve_referents(
            declared=[
                {'kind': 'task', 'id': 3129},
                {'kind': 'task', 'id': 2500},
                {'kind': 'task', 'id': '3129'},
            ],
            metadata={},
            content='Fixed the bug in Task 3127.',
            group_id=GROUP,
        )
        assert resolution.conflicts == (
            Referent(kind='task', number='3129'),
            Referent(kind='task', number='2500'),
        )

    @pytest.mark.parametrize(
        'declared,metadata',
        [(None, {'task_id': 3129}), (None, {})],
        ids=['metadata', 'derived'],
    )
    def test_the_lower_sources_never_conflict(self, declared, metadata):
        """Conflicts are a property of what the CALLER declared, never of
        ambient harness state and never of the scan against itself."""
        resolution = resolve_referents(
            declared=declared,
            metadata=metadata,
            content='Fixed the bug in Task 3127.',
            group_id=GROUP,
        )
        assert resolution.conflicts == ()
