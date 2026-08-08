"""Unit tests for fused_memory.utils.canonical_labels.

canonical_labels is THE single normative site for the canonical task-label
vocabulary (INV-5 / task 3667): one compiled description of what a task label
looks like, which utils/task_naming.py and utils/cross_project_refs.py both
call rather than each carrying their own lockstep-duplicated copy. Two copies
of a vocabulary drift, and the drift is invisible until a destructive
consumer acts on the half that is behind — 'task #1153' being a task-node
name to a human, but not to the anchored pattern task_naming used to own.

Mirrors tests/test_cross_project_refs.py and tests/test_task_naming.py, whose
leaf-module shape and pytest conventions this module copies.
"""

from __future__ import annotations

import pytest

from fused_memory.utils.canonical_labels import (
    LabelScan,
    Referent,
    parse_node_name,
    scan_content,
)


class TestReferentNodeName:
    """``Referent.node_name`` renders the graph node name the referent denotes.

    It is a derived property rather than a field so it can never drift out of
    sync with the fields it is computed from (the CrossProjectRef.entity_name
    precedent).
    """

    def test_own_project_referent_renders_the_bare_task_name(self):
        """An EMPTY project_id denotes own-project / unqualified — which is
        what lets node_name discriminate 'Task 132' from 'reify:132' from the
        referent alone."""
        referent = Referent(kind='task', project_id='', number='132')
        assert referent.node_name == 'Task 132'

    def test_qualified_referent_renders_the_qualified_entity_name(self):
        """A project qualifier is a DIFFERENT-project signal and must never be
        normalized away to 'Task 132' — that collapse is precisely the bug
        cross_project_refs exists to detect."""
        referent = Referent(kind='task', project_id='reify', number='132')
        assert referent.node_name == 'reify:132'


class TestDigitsPreservedVerbatim:
    """Digits are stored and rendered verbatim, never int-normalized, mirroring
    canonicalize_task_node_name and CrossProjectRef.task_number."""

    def test_zero_padded_number_survives_rendering(self):
        assert Referent(kind='task', project_id='', number='0132').node_name == 'Task 0132'

    def test_zero_padded_and_unpadded_numbers_are_distinct_referents(self):
        assert Referent(kind='task', project_id='', number='0132') != Referent(
            kind='task', project_id='', number='132'
        )


class TestReferentFieldDefaults:
    """``kind`` defaults to 'task' and ``project_id`` to '' (own-project), so
    the overwhelmingly common referent is spelled with one argument."""

    def test_number_only_construction_is_the_own_project_task_referent(self):
        assert Referent(number='132') == Referent(kind='task', project_id='', number='132')

    def test_number_only_construction_renders_the_bare_task_name(self):
        assert Referent(number='132').node_name == 'Task 132'


class TestReferentIsFrozen:
    """Referent is frozen — a scan result is evidence for destructive graph
    surgery and a consumer must not be able to rewrite which node it names."""

    def test_project_id_is_immutable(self):
        referent = Referent(number='132')
        with pytest.raises(Exception):  # noqa: B017 - FrozenInstanceError subclasses AttributeError
            referent.project_id = 'other'  # type: ignore[misc]

    def test_number_is_immutable(self):
        referent = Referent(number='132')
        with pytest.raises(Exception):  # noqa: B017
            referent.number = '999'  # type: ignore[misc]


class TestParseNodeNameMatchesLocalForms:
    """``parse_node_name`` is ANCHORED (whole-string): it answers "is this
    entity name a task label?", so it must never fire on a name that merely
    mentions a task.

    The separator between the word and the digits is whitespace, '#' or ':'.
    'task #1153' and 'Task: 132' are the forms task_naming's whitespace-only
    pattern could not see — they are two of the PRD's 53 measured variant
    splits, and making them parse is the point of the extraction.
    """

    @pytest.mark.parametrize(
        ('name', 'expected'),
        [
            ('task 132', 'Task 132'),
            ('tasks 153', 'Task 153'),
            ('TASK 42', 'Task 42'),
            ('Task  7', 'Task 7'),
            (' tasks 9 ', 'Task 9'),
            ('Task 42', 'Task 42'),  # idempotence: already-canonical maps to itself
            # NEW in task 3667 — the '#' spelling task_naming was structurally
            # unable to match, since '^\s*tasks?\s+(\d+)\s*$' requires whitespace.
            ('task #1153', 'Task 1153'),
            ('task#1153', 'Task 1153'),
            # A task-vocabulary qualifier is LOCAL task vocabulary, never a
            # project id — the same rule cross_project_refs already enforces.
            ('Task: 132', 'Task 132'),
        ],
    )
    def test_parses_to_an_own_project_referent(self, name, expected):
        referent = parse_node_name(name)
        assert referent is not None
        assert referent.kind == 'task'
        assert referent.project_id == ''
        assert referent.node_name == expected

    def test_digits_are_preserved_verbatim(self):
        referent = parse_node_name('task 0132')
        assert referent is not None
        assert referent.number == '0132'
        assert referent != parse_node_name('Task 132')

    def test_is_idempotent_through_node_name(self):
        """Re-parsing a referent's own node_name yields the same name — required
        by the normalization hook's ``canonical != name`` guard, which would
        otherwise rename the same node on every pass."""
        first = parse_node_name('task 132')
        assert first is not None
        second = parse_node_name(first.node_name)
        assert second is not None
        assert second.node_name == 'Task 132'


class TestParseNodeNameMatchesQualifiedForms:
    """A project-qualified node name parses to a FOREIGN referent whose
    qualifier is preserved, never normalized away to the bare 'Task N' form."""

    def test_qualified_name_keeps_its_project(self):
        referent = parse_node_name('reify:132')
        assert referent is not None  # narrows Referent | None -> Referent
        assert referent == Referent(kind='task', project_id='reify', number='132')
        assert referent.node_name == 'reify:132'

    def test_qualifier_is_canonicalized(self):
        referent = parse_node_name('Dark-Factory:2500')
        assert referent is not None
        assert referent.project_id == 'dark_factory'
        assert referent.number == '2500'


class TestParseNodeNameNonMatches:
    """Anchoring plus the precision narrowings: anything that is not itself a
    whole task label returns None, so a caller leaves that node untouched."""

    @pytest.mark.parametrize(
        'name',
        [
            'Alice',
            '',
            'task',  # no number
            'subtask 5',
            'multitask 3',
            'taskforce 9',
            # The anchoring invariant: a name that MENTIONS a task is not a
            # task-node name and must never be renamed to one.
            'Task 3331 dashboard index',
            'Task 42 orchestrator',
            'reify task 12',
            # No separator at all — preserves task_naming's original '\s+'
            # requirement, so the added '#'/':' alternation did not widen the
            # pattern into matching glued digits.
            'task132',
            # Bare-digit names are an explicit PRD blind spot, not a task label.
            '1251',
            # A path-shaped qualifier must be REFUSED, never silently
            # canonicalized into a new, wrong project key (RCA §4).
            '/home/leo/src/dark-factory:2500',
            # Task vocabulary is never a project id, in the qualified form too.
            'subtask: 2500',
            'sub-task: 2500',
        ],
    )
    def test_returns_none(self, name):
        assert parse_node_name(name) is None


class TestLabelScanIsFrozen:
    """LabelScan is frozen for the same reason Referent is: it is evidence for
    destructive graph surgery, not a mutable accumulator."""

    def test_both_lists_default_empty(self):
        scan = LabelScan()
        assert scan.refs == []
        assert scan.ambiguous == []

    def test_refs_is_immutable(self):
        scan = LabelScan()
        with pytest.raises(Exception):  # noqa: B017 - FrozenInstanceError subclasses AttributeError
            scan.refs = []  # type: ignore[misc]


class TestScanContentFindsOwnProjectReferents:
    """An unqualified prose mention of a task, and a SELF-qualified one, are
    both referents of the LOCAL task — project_id '' , node_name 'Task N'."""

    @pytest.mark.parametrize(
        ('content', 'group_id', 'number'),
        [
            ('Reify task 5181 was cancelled', 'reify', '5181'),
            # The '#' form the pre-3667 mention pattern could not see.
            ('task #1153 is the variant split', 'reify', '1153'),
        ],
    )
    def test_bare_mention_yields_an_own_project_referent(self, content, group_id, number):
        scan = scan_content(content, group_id=group_id)
        assert [(r.project_id, r.number) for r in scan.refs] == [('', number)]

    def test_colon_spelled_mention_is_local_vocabulary_not_a_project(self):
        """'Task: 2500' is a LOCAL task mention. Reading 'task' as a project id
        would send destructive surgery at a bogus 'task:2500' entity — the one
        rejection no downstream guard can make (see cross_project_refs)."""
        scan = scan_content('Task: 2500 is now done', group_id='dark_factory')
        assert [r.node_name for r in scan.refs] == ['Task 2500']
        assert [r for r in scan.refs if r.project_id == 'task'] == []

    @pytest.mark.parametrize(
        ('content', 'group_id'),
        [
            ('reify:5181 was cancelled', 'reify'),
            # Both sides are canonicalized before the comparison, so case and
            # hyphen/underscore spelling differences still count as SELF.
            ('REIFY:5181 was cancelled', 'reify'),
            ('Reify-Factory:5181 was cancelled', 'reify_factory'),
            ('reify_factory:5181 was cancelled', 'Reify-Factory'),
        ],
    )
    def test_self_qualified_ref_is_reclassified_as_own_project(self, content, group_id):
        """A qualifier naming the CURRENT group is not cross-project, but it IS
        a genuine local referent — 'reify:5181' in the reify graph really does
        mean local Task 5181, and extraction collapsing it onto 'Task 5181' is
        correct, not a bug. Reclassify it rather than dropping it, so a
        consumer needing the COMPLETE local referent set gets it. (The
        foreign-only filter in find_cross_project_task_refs reproduces the drop
        that module wants.)"""
        scan = scan_content(content, group_id=group_id)
        assert [(r.project_id, r.node_name) for r in scan.refs] == [('', 'Task 5181')]


class TestScanContentFindsCrossProjectReferents:
    """A qualifier naming a project other than the current group is a FOREIGN
    referent, rendered as the qualified entity name it should have."""

    def test_foreign_qualifier_yields_a_foreign_referent(self):
        scan = scan_content('see dark_factory:2500 for context', group_id='reify')
        assert [(r.project_id, r.number, r.node_name) for r in scan.refs] == [
            ('dark_factory', '2500', 'dark_factory:2500')
        ]

    def test_digits_are_preserved_verbatim(self):
        scan = scan_content('see dark_factory:0250', group_id='reify')
        assert [r.number for r in scan.refs] == ['0250']


class TestScanContentPrecision:
    """The scan drives destructive edge surgery, so colon-bearing noise that is
    not a label must never produce a referent. Each entry encodes a specific
    measured false positive; do not re-derive these narrowings."""

    @pytest.mark.parametrize(
        'content',
        [
            '',
            'no colons at all here',
            # Source locations: '.py' is too short AND preceded by a '.'.
            'see graphiti_client.py:2091 for the call site',
            'memory_service.py:1077 declares it',
            # Clock times: a qualifier must start with a letter.
            'at 12:30 the run started',
            # URL authorities.
            'see https://example.com:8080/path',
            # Too-short qualifiers (<3 chars) are noise, not project ids.
            'w6:2 is a cell reference',
            'py:3 is not a project',
            'a:1 is not a project',
            # A colon-chained token is not a qualifier.
            'foo:bar:3 is not a project ref',
            # A qualifier glued to a preceding word character.
            '2500dark_factory:2500',
            # A path-shaped qualifier is rejected by the lookbehind before
            # canonicalization ever sees it (RCA §4).
            'see /home/leo/src/dark-factory:2500',
            # Task vocabulary in qualifier position is rejected, AND the
            # word-glue lookbehind stops the 'task: N' substring inside it from
            # counting as a local mention either.
            'subtask: 2500',
            'sub-task: 2500',
            'subtasks: 2500',
        ],
    )
    def test_noise_yields_no_referents(self, content):
        scan = scan_content(content, group_id='reify')
        assert scan.refs == []
        assert scan.ambiguous == []

    @pytest.mark.parametrize(
        'content',
        [
            'dark_factory:2500 relates to subtask 2500',
            'dark_factory:2500 relates to multitask 2500',
            'dark_factory:2500 relates to reify-task 2500',
        ],
    )
    def test_word_glued_lookalikes_are_not_local_mentions(self, content):
        """'subtask 2500' is not a mention of task 2500, mirroring the anchored
        pattern's refusal to match 'subtask 5'."""
        scan = scan_content(content, group_id='reify')
        assert [r.node_name for r in scan.refs] == ['dark_factory:2500']

    def test_path_shaped_group_id_yields_an_empty_scan(self):
        """Without a trustworthy local project id, local and foreign referents
        cannot be told apart — report nothing rather than guess. No raise: the
        scanner sits on a write path."""
        scan = scan_content('see dark_factory:2500', group_id='-home-leo-src-reify')
        assert scan == LabelScan()


class TestScanContentOrderingAndDedup:
    """Positional first-seen order, de-duplicated on (kind, project_id, number)
    — so the result is deterministic and a consumer can rely on it."""

    def test_multiple_foreign_refs_preserve_first_seen_order(self):
        content = 'blocked on dark_factory:2500, then stepping:7, then dark_factory:11'
        scan = scan_content(content, group_id='reify')
        assert [r.node_name for r in scan.refs] == [
            'dark_factory:2500',
            'stepping:7',
            'dark_factory:11',
        ]

    def test_local_and_foreign_referents_interleave_by_position(self):
        """Order is by offset in the content, NOT by which pattern found it —
        the two passes are merged, not concatenated."""
        scan = scan_content('task 7 then dark_factory:11 then task 9', group_id='reify')
        assert [r.node_name for r in scan.refs] == ['Task 7', 'dark_factory:11', 'Task 9']

    def test_differently_spelled_qualifiers_dedupe_to_one_referent(self):
        content = 'dark_factory:2500 and Dark-Factory:2500 and DARK_FACTORY:2500'
        scan = scan_content(content, group_id='reify')
        assert [r.node_name for r in scan.refs] == ['dark_factory:2500']

    def test_repeated_scans_are_deterministic(self):
        content = 'dark_factory:2500, stepping:7, dark_factory:2500, stepping:7'
        first = scan_content(content, group_id='reify')
        second = scan_content(content, group_id='reify')
        assert [r.node_name for r in first.refs] == ['dark_factory:2500', 'stepping:7']
        assert [r.node_name for r in first.refs] == [r.node_name for r in second.refs]


class TestScanContentAllowlist:
    """An optional narrowing: when a registry of known project ids is supplied,
    FOREIGN referents naming an unknown project are dropped."""

    def test_unknown_projects_are_dropped_when_allowlist_supplied(self):
        content = 'blocked on dark_factory:2500, stepping:3 and unknown_proj:7'
        scan = scan_content(content, group_id='reify', known_project_ids={'dark_factory'})
        assert [r.node_name for r in scan.refs] == ['dark_factory:2500']

    def test_allowlist_entries_are_canonicalized_before_comparison(self):
        scan = scan_content(
            'see dark_factory:2500', group_id='reify', known_project_ids={'Dark-Factory'}
        )
        assert [r.node_name for r in scan.refs] == ['dark_factory:2500']

    def test_allowlist_accepts_a_mapping_of_project_id_to_root(self):
        """The caller's registry is MemoryService._known_projects, a
        {project_id: project_root} dict — iterating it yields its keys."""
        scan = scan_content(
            'blocked on dark_factory:2500 and stepping:3',
            group_id='reify',
            known_project_ids={'dark_factory': '/some/root', 'reify': '/other/root'},
        )
        assert [r.node_name for r in scan.refs] == ['dark_factory:2500']

    @pytest.mark.parametrize('allowlist', [None, set(), frozenset(), []])
    def test_empty_or_missing_allowlist_is_permissive(self, allowlist):
        """Mirrors validate_known_project_id's documented empty-registry mode:
        an unavailable registry must NOT silently disable the protection."""
        content = 'blocked on dark_factory:2500 and stepping:3'
        scan = scan_content(content, group_id='reify', known_project_ids=allowlist)
        assert [r.node_name for r in scan.refs] == ['dark_factory:2500', 'stepping:3']

    def test_path_shaped_allowlist_entry_is_skipped_not_fatal(self):
        scan = scan_content(
            'see dark_factory:2500',
            group_id='reify',
            known_project_ids={'dark_factory', '-home-leo-src-oops'},
        )
        assert [r.node_name for r in scan.refs] == ['dark_factory:2500']

    def test_own_project_referents_are_never_dropped_by_the_allowlist(self):
        """The allowlist narrows FOREIGN referents only, and is applied AFTER
        the self->local reclassification — so an allowlist that omits the
        current group still cannot suppress a local referent."""
        scan = scan_content(
            'task 7 and reify:8', group_id='reify', known_project_ids={'dark_factory'}
        )
        assert [r.node_name for r in scan.refs] == ['Task 7', 'Task 8']


class TestShapeValidProseMatchesByDesign:
    """Prose with the '<word>: <number>' shape DOES match — the scanner is
    deliberately whitespace-tolerant, because that is how humans write a
    qualified reference. Final precision is delegated to the consumer's own
    guard (and to the allowlist, when a registry is wired)."""

    def test_prose_colon_number_matches_shape_only(self):
        scan = scan_content('Total: 42 items', group_id='reify')
        assert [r.node_name for r in scan.refs] == ['total:42']

    def test_allowlist_removes_the_prose_match(self):
        scan = scan_content('Total: 42 items', group_id='reify', known_project_ids={'dark_factory'})
        assert scan.refs == []


class TestScanContentAmbiguityPartition:
    """A task number claimed by BOTH an own-project and a foreign-qualified
    mention in the same content is genuinely ambiguous about which task the
    facts belong to. EVERY referent carrying that number goes to .ambiguous
    and none to .refs — refuse rather than guess.

    The partition is SYMMETRIC on purpose. Marking only the foreign side would
    hand a consumer a confidently-wrong LOCAL referent derived from provably
    ambiguous prose; both sides are equally unusable as evidence.

    Ordering note: .ambiguous preserves POSITIONAL order, the same rule
    TestScanContentOrderingAndDedup pins for .refs. In the content below the
    qualified ref sits at offset 0 and the bare mention at offset 25 (measured),
    so 'dark_factory:2500' comes first.
    """

    def test_same_number_local_and_foreign_is_ambiguous(self):
        scan = scan_content('dark_factory:2500 blocks task 2500 here', group_id='reify')
        assert scan.refs == []
        assert [r.node_name for r in scan.ambiguous] == ['dark_factory:2500', 'Task 2500']

    @pytest.mark.parametrize(
        'bare_form',
        [
            'task 2500',
            'Task 2500',
            'TASK 2500',
            'tasks 2500',
            'Task: 2500',
            # NEW in task 3667: the '#' spelling was invisible to the pre-3667
            # mention patterns, so this content used to yield a CONFIDENT split.
            'task #2500',
        ],
    )
    def test_every_bare_spelling_triggers_ambiguity(self, bare_form):
        scan = scan_content(f'dark_factory:2500 relates to {bare_form}', group_id='reify')
        assert scan.refs == []
        assert [r.node_name for r in scan.ambiguous] == ['dark_factory:2500', 'Task 2500']

    def test_ambiguity_is_per_number_not_per_content(self):
        """One contested number must never suppress an unrelated clean ref."""
        content = 'dark_factory:2500 blocks task 2500; also see dark_factory:99'
        scan = scan_content(content, group_id='reify')
        assert [r.node_name for r in scan.refs] == ['dark_factory:99']
        assert [r.node_name for r in scan.ambiguous] == ['dark_factory:2500', 'Task 2500']

    def test_zero_padded_literals_are_distinct_so_no_false_ambiguity(self):
        """'250' is a different literal from '0250'. Digits are never
        int-normalized, so the two numbers never contest each other."""
        scan = scan_content('dark_factory:0250 relates to task 250', group_id='reify')
        assert [r.node_name for r in scan.refs] == ['dark_factory:0250', 'Task 250']
        assert scan.ambiguous == []

    @pytest.mark.parametrize(
        'lookalike',
        ['subtask 2500', 'multitask 2500', 'reify-task 2500', 'subtask: 2500'],
    )
    def test_word_glued_lookalikes_create_no_ambiguity(self, lookalike):
        """'subtask 2500' is not a mention of task 2500, so it must not make a
        real foreign ref ambiguous — the regression guard for the unified
        mention pattern's lookbehind."""
        scan = scan_content(f'dark_factory:2500 relates to {lookalike}', group_id='reify')
        assert [r.node_name for r in scan.refs] == ['dark_factory:2500']
        assert scan.ambiguous == []

    def test_bare_mention_of_a_different_number_does_not_suppress_the_ref(self):
        """The incident content that motivated the split hook: the bare mention
        names a DIFFERENT task number, so nothing is contested."""
        content = 'Reify task 5181 was cancelled; its work was rerouted to dark_factory:2500.'
        scan = scan_content(content, group_id='reify')
        assert [r.node_name for r in scan.refs] == ['Task 5181', 'dark_factory:2500']
        assert scan.ambiguous == []

    def test_self_qualified_plus_bare_mention_is_not_ambiguous(self):
        """Both spellings denote the SAME own-project referent, so the
        self->local reclassification (which runs BEFORE this partition) lets
        dedup collapse them into one unambiguous ref rather than a contest."""
        scan = scan_content('reify:5181 and task 5181', group_id='reify')
        assert [r.node_name for r in scan.refs] == ['Task 5181']
        assert scan.ambiguous == []

    def test_ambiguous_referents_are_deduplicated_too(self):
        content = 'dark_factory:2500 blocks task 2500; dark_factory:2500 again'
        scan = scan_content(content, group_id='reify')
        assert scan.refs == []
        assert [r.node_name for r in scan.ambiguous] == ['dark_factory:2500', 'Task 2500']


class TestUnregisteredKindIsRejectedLoudly:
    """The kind is registry-extensible ('escalation' is the next entry per the
    PRD) and today's registry holds only 'task'.

    Validating at construction makes the extension point concrete and testable,
    and keeps node_name from raising a bare KeyError deep inside a consumer
    (loud-over-silent-degradation / structured-facts-at-failure).
    """

    def test_unregistered_kind_raises_value_error_naming_the_kind(self):
        with pytest.raises(ValueError, match='escalation'):
            Referent(kind='escalation', number='7')

    def test_error_names_the_registered_kinds(self):
        with pytest.raises(ValueError, match='task'):
            Referent(kind='escalation', number='7')
