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

import dataclasses

import pytest

from fused_memory.utils import canonical_labels
from fused_memory.utils.canonical_labels import (
    LabelScan,
    Referent,
    _canonical_allowlist,
    parse_node_name,
    scan_content,
)

# Unicode decimal digits that ``str.isdigit()`` accepts but ``str.isascii()``
# does not — spelled by ESCAPE so the fixture survives transport through any
# tool or terminal that mangles non-ASCII glyphs, and so a reader sees the
# codepoint rather than a character that renders like an ASCII '3'.
ARABIC_INDIC_THREE = '\u0663'  # ARABIC-INDIC DIGIT THREE
FULLWIDTH_THREE = '\uff13'  # FULLWIDTH DIGIT THREE — the same hazard, a second block


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
        with pytest.raises(dataclasses.FrozenInstanceError):
            referent.project_id = 'other'  # type: ignore[misc]

    def test_number_is_immutable(self):
        referent = Referent(number='132')
        with pytest.raises(dataclasses.FrozenInstanceError):
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
    destructive graph surgery, not a mutable accumulator.

    ``frozen=True`` alone does NOT deliver that: it blocks attribute rebinding
    only, so list-typed fields would stay freely mutable in place. The fields
    are tuples, and these tests pin BOTH halves — no rebinding and no in-place
    mutation — so the class docstring's claim is actually enforced.
    """

    def test_both_fields_default_empty(self):
        scan = LabelScan()
        assert scan.refs == ()
        assert scan.ambiguous == ()

    def test_refs_cannot_be_rebound(self):
        scan = LabelScan()
        with pytest.raises(dataclasses.FrozenInstanceError):
            scan.refs = ()  # type: ignore[misc]

    @pytest.mark.parametrize('attr', ['refs', 'ambiguous'])
    def test_fields_cannot_be_mutated_in_place(self, attr):
        """The half ``frozen=True`` does not cover. With list fields
        ``scan.refs.append(...)`` silently succeeded, letting a consumer add a
        referent the scanner deliberately refused to infer."""
        scan = scan_content(
            'dark_factory:2500 blocks task 2500; also see dark_factory:99', group_id='reify'
        )
        assert getattr(scan, attr)  # both fields are populated, so this is a real test
        with pytest.raises(AttributeError):
            getattr(scan, attr).append(Referent(number='9'))

    def test_a_scan_is_hashable(self):
        """A frozen dataclass holding lists is silently unhashable despite its
        generated __hash__ — a trap for any consumer that sets/dict-keys a
        scan. Tuple fields make the generated __hash__ actually work."""
        assert hash(LabelScan()) == hash(LabelScan())
        scan = scan_content('see dark_factory:2500', group_id='reify')
        assert len({scan, scan_content('see dark_factory:2500', group_id='reify')}) == 1


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
        assert scan.refs == ()
        assert scan.ambiguous == ()

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

    @pytest.mark.parametrize(
        'content',
        [
            # A markdown heading after a paragraph break is not a task mention.
            'Completed the task\n\n# 1153 retrospective',
            'Completed the task\n# 1153 retrospective',
            # A numbered-list item after a 'tasks:' lead-in is not task 1.
            'The following tasks:\n\n1. fix the parser',
            'The following tasks:\n1. fix the parser',
        ],
    )
    def test_the_hash_and_colon_separators_never_span_a_line_break(self, content):
        """The '#'/':' separator branch is padded with '[ \\t]', not '\\s'.

        '\\s' matches '\\n', so the branch task 3667 added would otherwise read a
        markdown heading or a numbered-list item across a paragraph break as a
        bare task mention. Neither spelling could match before 3667 (the
        pre-existing mention pattern was 'tasks?\\s+(\\d+)', which cannot see a
        '#'), so this is a phantom the extraction introduced, not a narrowing of
        prior behaviour.
        """
        assert scan_content(content, group_id='reify').refs == ()

    def test_a_phantom_heading_mention_cannot_suppress_a_real_foreign_ref(self):
        """The live consequence, and why this is a correctness fix and not
        cosmetics: a phantom bare mention lands in bare_numbers and contests the
        matching foreign ref, moving a genuine cross-project repair into
        .ambiguous where the consumer refuses to act on it."""
        scan = scan_content(
            'dark_factory:1153 owns it\n\nthe task\n\n# 1153 notes', group_id='reify'
        )
        assert [r.node_name for r in scan.refs] == ['dark_factory:1153']
        assert scan.ambiguous == ()

    def test_the_hash_separator_still_matches_on_one_line(self):
        """Guard against over-tightening: the '#' form is the headline signal of
        task 3667 and must keep matching with any same-line padding."""
        for content in ('task #1153 here', 'task#1153 here', 'TASK # 1153 here', 'task\t#\t1153'):
            assert [r.node_name for r in scan_content(content, group_id='reify').refs] == [
                'Task 1153'
            ], content


class TestUnicodeDigitsAreNotTaskNumbers:
    """A task number is written in ASCII digits. Python's ``re`` matches Unicode
    decimal digits with ``\\d`` on str patterns, so 'task \\u0663' parsed as a task
    whose number was that character.

    This is a false POSITIVE against the contract :func:`scan_content` states for
    itself — PRECISION OVER RECALL, because its consumers perform destructive
    edge surgery — and it is the OPPOSITE direction from the KNOWN BLIND SPOTS
    that docstring enumerates (bare-digit names, title references, codename
    aliases), every one of which is a recall loss. So it is not covered by them:
    a blind spot silently misses a real referent, while this MINTS one that names
    nothing.

    The DECLARED path already refuses these — referent_resolution._is_task_number
    guards its bare-digit branch with ``text.isascii() and text.isdigit()`` (task
    3668), its docstring saying "a Unicode digit is not a task id". Until this
    narrowing the DERIVED path minted exactly what the declared path refused; that
    asymmetry is what these fixtures close, at the one normative site (INV-5)
    rather than per consumer.

    The standing ASCII regression guards are deliberately NOT copied into this
    class. TestParseNodeNameMatchesLocalForms (every local spelling) plus its
    test_digits_are_preserved_verbatim ('task 0132'),
    TestParseNodeNameMatchesQualifiedForms ('reify:132', 'Dark-Factory:2500')
    and TestQualifiedRefNeverSpansALineBreak.test_same_line_spellings_are_unaffected
    (the four colon-padding spellings) already run on every commit, and between
    them they are what proves this narrowing disturbed neither the separator
    alternation, the case-insensitivity, nor the colon padding — that padding
    being a SEPARATE axis, tracked as task 4235 (duplicate filing 4239) and
    scoped out here. A second copy of those lists inside this class would be
    the very lockstep duplication INV-5 exists to prevent, reproduced in the
    test suite: the next narrowing would touch one copy and the two would drift.
    """

    @pytest.mark.parametrize(
        'name',
        [
            # Measured RED before the fix: each returned a Referent whose
            # ``number`` was the Unicode character itself, rendering a
            # 'Task ٣' node name a destructive consumer would act on.
            'Task ' + ARABIC_INDIC_THREE,
            'task#' + ARABIC_INDIC_THREE,
            'tasks ' + ARABIC_INDIC_THREE,
            'Task: ' + ARABIC_INDIC_THREE,
            # A second Unicode block, so the fixture is not pinned to one script.
            'Task ' + FULLWIDTH_THREE,
        ],
    )
    def test_local_node_name_with_a_unicode_digit_is_not_a_task_label(self, name):
        assert parse_node_name(name) is None

    def test_a_unicode_digit_mention_yields_no_referent(self):
        content = 'blocked on task ' + ARABIC_INDIC_THREE + ' for now'
        scan = scan_content(content, group_id='dark_factory')
        assert scan.refs == ()
        assert scan.ambiguous == ()

    @pytest.mark.parametrize(
        'name',
        ['Task 12' + ARABIC_INDIC_THREE, 'task #12' + FULLWIDTH_THREE],
    )
    def test_a_mixed_run_yields_nothing_rather_than_the_truncated_prefix(self, name):
        """The ANCHORED half of the mixed-run guard. What it pins is the CAPTURE
        class, not the right-edge lookahead.

        Before the fix this parsed to Referent(number='12٣') — a mangled number
        naming no task. What refuses it now is purely '([0-9]+)': the two anchored
        patterns carry NO '(?!\\d)' clause at all — their right edge is '\\s*$', and a
        Unicode digit is not '\\s', so the match dies at the anchor either way.

        The lookahead's breadth, and the reason it deliberately stays BROAD while
        the capture narrows, is pinned by the UNANCHORED siblings instead:
        test_a_mixed_run_mention_yields_no_referent, the scan_content half of
        test_a_qualified_mixed_run_yields_nothing_rather_than_the_truncated_prefix,
        and test_cross_project_refs.py's 'see reify:12٣ now' case. Measured, not
        assumed: mutating both '([0-9]+)(?!\\d)' occurrences to '([0-9]+)(?![0-9])'
        fails exactly those three and NOT this one, minting Referent(number='12')
        and Referent(project_id='reify', number='12') — the TRUNCATED PREFIX that
        lookahead's own comment says it exists to prevent, and strictly worse than
        the bug being fixed, since 'Task ٣' is obvious junk while 'Task 12' names
        a REAL node a consumer will happily rename or re-attach edges on. So do
        not trim those three as duplicative of this one: this one survives the
        mutation they catch.
        """
        assert parse_node_name(name) is None

    def test_a_mixed_run_mention_yields_no_referent(self):
        """Same guard on the unanchored side, where the truncated prefix would
        land in ``refs`` and drive the surgery directly."""
        scan = scan_content('task 12' + ARABIC_INDIC_THREE, group_id='dark_factory')
        assert scan.refs == ()
        assert scan.ambiguous == ()

    @pytest.mark.parametrize(
        'name',
        [
            # The PROJECT-QUALIFIED half fails identically — measured RED:
            # parse_node_name('reify:\u0663') returned
            # Referent(project_id='reify', number='\u0663'). The qualifier
            # classes were already ASCII-explicit ([A-Za-z][A-Za-z0-9_-]{2,});
            # the digit capture was the last Unicode-permissive class left in
            # the vocabulary.
            'reify:' + ARABIC_INDIC_THREE,
            'Dark-Factory:' + ARABIC_INDIC_THREE,
            'reify:' + FULLWIDTH_THREE,
        ],
    )
    def test_qualified_node_name_with_a_unicode_digit_is_not_a_task_label(self, name):
        assert parse_node_name(name) is None

    def test_a_unicode_digit_qualified_ref_yields_no_referent(self):
        content = 'see reify:' + ARABIC_INDIC_THREE + ' now'
        scan = scan_content(content, group_id='dark_factory')
        assert scan.refs == ()
        assert scan.ambiguous == ()

    def test_a_qualified_mixed_run_yields_nothing_rather_than_the_truncated_prefix(self):
        """Where the qualified half is MOST dangerous, and why its right-edge
        lookahead stays broad too.

        Under the wrong fix — narrowing '(?!\\d)' alongside the capture class —
        'reify:12\u0663' yields Referent(project_id='reify', number='12'): not
        mangled junk but a fully well-formed FOREIGN referent pointing at the
        wrong task, which is exactly the misattribution cross_project_refs' split
        hook exists to prevent and would instead perform. The local truncation
        guard costs a rename; this one costs destructive edge surgery onto a real
        node in another project.
        """
        assert parse_node_name('reify:12' + ARABIC_INDIC_THREE) is None
        scan = scan_content('see reify:12' + ARABIC_INDIC_THREE, group_id='dark_factory')
        assert scan.refs == ()
        assert scan.ambiguous == ()


class TestQualifiedRefNeverSpansALineBreak:
    """The PROJECT-QUALIFIED pattern's colon is padded with '[ \\t]', not '\\s',
    for the same measured reason as the mention pattern's above: '\\s' matches
    '\\n', and no real project-qualified reference is written across a line
    break — while episode bodies routinely carry YAML/'key: value' blocks and
    hard-wrapped prose, which the '\\s' spelling read as qualified refs.

    Direction of safety: this narrowing REMOVES foreign refs, and for a
    consumer performing destructive edge surgery a false positive
    MISATTRIBUTES facts — so narrowing is the safe direction here, unlike the
    _LOCAL_MENTION_PATTERN whitespace branch, deliberately left as '\\s+'
    because narrowing THERE removes contests.
    """

    def test_colon_followed_by_newline_is_not_a_qualified_ref(self):
        """A 'key:' line lead-in followed by a number on the next line — the
        shape a YAML block or a hard-wrapped note produces — read as
        'notes:2500', a foreign project that does not exist."""
        assert scan_content('Notes:\n2500 items', group_id='reify').refs == ()

    def test_newline_before_the_colon_is_not_a_qualified_ref(self):
        """BOTH halves are narrowed: the '\\s*' preceding the colon spans a
        newline exactly as the trailing one does, so fixing only the trailing
        half would leave the stated invariant half-true."""
        assert scan_content('Notes\n: 2500', group_id='reify').refs == ()

    def test_blank_line_between_qualifier_and_number_is_not_a_qualified_ref(self):
        content = 'See the section on graphiti:\n\n2500 rows were affected'
        assert scan_content(content, group_id='reify').refs == ()

    def test_a_hard_wrapped_genuine_ref_is_a_DELIBERATE_miss(self):
        """The price of the narrowing, measured and accepted — not overlooked.

        The three cases above are all TRUE negatives ('Notes:', 'graphiti:' are
        not projects). This one is a FALSE negative: a real qualified ref that
        hard wrapping split, since wrapping breaks at the space after the colon
        (``textwrap.fill(..., 40)`` produces exactly this string). It is the
        same '\\s'-vs-'[ \\t]' decision seen from the losing side, and it is
        pinned here — and listed in scan_content's KNOWN BLIND SPOTS — so a
        future reader sees the recall loss was weighed rather than missed. It
        is accepted for the reason stated on the pattern: for a consumer doing
        destructive edge surgery a missed ref is recoverable and a
        misattributed one is not. Changing this to a match is a real design
        decision, not a bug fix.
        """
        content = 'please see the reference dark_factory:\n2500 for more detail'
        assert scan_content(content, group_id='reify').refs == ()

    def test_same_line_spellings_are_unaffected(self):
        """Regression guard, green before AND after: the padding still tolerates
        the spaces and tabs humans actually write around a colon. This is what
        proves the change narrows ONLY across line breaks and is not an
        undeclared tightening of human spacing."""
        for content in (
            'dark_factory:2500',
            'dark_factory: 2500',
            'dark_factory :2500',
            'dark_factory\t:\t2500',
        ):
            assert [r.node_name for r in scan_content(content, group_id='reify').refs] == [
                'dark_factory:2500'
            ], content


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

    def test_all_path_shaped_registry_is_permissive_not_fail_closed(self):
        """A registry whose entries are ALL path-shaped degrades to PERMISSIVE.

        The one-bad-entry case above is only half the contract
        ``_canonical_allowlist`` states — "one bad entry never disables the
        whole allowlist". When EVERY entry is skipped the surviving set is
        empty, and an empty frozenset ``is not None``, so the qualified-ref
        guard would read "allowlist of nothing" as "allow nothing" and drop
        every foreign ref — silently disabling the detection this scanner
        exists to provide. Mirrors ``validate_known_project_id``'s permissive
        falsy-registry mode rather than inventing a fail-closed one.
        """
        scan = scan_content(
            'see dark_factory:2500', group_id='reify', known_project_ids={'-home-leo-bad'}
        )
        assert [r.node_name for r in scan.refs] == ['dark_factory:2500']

    def test_canonical_allowlist_returns_none_when_every_entry_is_path_shaped(self):
        """``is None`` SPECIFICALLY, not merely falsy: an empty frozenset is
        falsy too, and ``allowlist is not None`` is the exact property the
        qualified-ref filter reads to decide whether to narrow at all."""
        assert _canonical_allowlist({'-home-leo-bad', '-home-leo-other'}) is None

    def test_a_blank_registry_key_also_yields_the_permissive_fallback(self):
        """The path-shaped case is only one of TWO ways the surviving allowlist
        becomes useless. ``canonicalize_project_id('')`` does NOT raise
        PathShapedProjectIdError — it returns '' — so a blank key survives into
        the set and makes it non-empty, reaching the caller as an "allowlist of
        nothing" and dropping every foreign ref. '' can never match a foreign
        referent anyway (the qualified-ref pattern requires a >=3-character
        qualifier), so it is skipped like a path-shaped key and feeds the same
        permissive fallback.
        """
        assert _canonical_allowlist({'', '-home-leo-bad'}) is None

    def test_blank_registry_key_does_not_drop_foreign_refs_at_the_scan_boundary(self):
        """The same contract at the boundary a caller actually uses."""
        scan = scan_content(
            'see dark_factory:2500', group_id='reify', known_project_ids={'', '-home-leo-bad'}
        )
        assert [r.node_name for r in scan.refs] == ['dark_factory:2500']

    def test_a_blank_key_beside_a_usable_one_still_narrows(self):
        """Regression guard for the blank-key skip, the twin of the mixed
        path-shaped case below: dropping '' must not degenerate into disabling
        an allowlist that still has a usable entry in it."""
        scan = scan_content(
            'see dark_factory:2500 and evil_proj:9',
            group_id='reify',
            known_project_ids={'dark_factory', ''},
        )
        assert [r.node_name for r in scan.refs] == ['dark_factory:2500']

    def test_a_non_sized_registry_does_not_crash_on_the_degraded_path(self):
        """The parameter is advertised as "any collection", which invites a
        generator. A generator is truthy (so it passes the empty-registry
        guard) but NOT Sized, so counting the entries with a trailing ``len()``
        would raise TypeError from a memory write path — on the one branch
        whose whole purpose is to degrade gracefully."""

        def registry():
            yield '-home-leo-bad'
            yield ''

        assert _canonical_allowlist(registry()) is None

    def test_mixed_registry_still_narrows_after_bad_entries_skipped(self):
        """Regression guard, green before AND after the permissive fallback:
        the fallback must fire only when NOTHING survived canonicalization, not
        degenerate into disabling the allowlist whenever an entry is skipped."""
        scan = scan_content(
            'see dark_factory:2500 and evil_proj:9',
            group_id='reify',
            known_project_ids={'dark_factory', '-home-leo-bad'},
        )
        assert [r.node_name for r in scan.refs] == ['dark_factory:2500']


class TestAllPathShapedRegistryIsLoud:
    """The permissive fallback for an all-path-shaped registry is a fail-SOFT
    path, so it must be audible — INV-4 ``storm-escape-required``
    (docs/legibility/design-invariants.md): every degradation carries an
    escalation, loud over silent.

    Warn-ONCE per process, mirroring ``validate_known_project_id``'s
    ``_empty_registry_warned`` 250 lines away in the same package; the tests
    below follow tests/test_validation.py::test_warn_once_on_empty_registry,
    including resetting the process-global flag, without which they would pass
    or fail depending on test ordering.
    """

    def test_all_path_shaped_registry_logs_a_warning(self, monkeypatch, caplog):
        """A registry that was deliberately supplied and turned out entirely
        unusable is an operator MISCONFIGURATION, and its consequence — the
        cross-project filter silently running unnarrowed — is invisible without
        this line."""
        monkeypatch.setattr(canonical_labels, '_all_path_shaped_warned', False)
        with caplog.at_level('WARNING', logger='fused_memory.utils.canonical_labels'):
            _canonical_allowlist({'-home-leo-bad'})
        warnings = [r for r in caplog.records if r.levelname == 'WARNING']
        assert len(warnings) == 1, (
            f'Expected exactly 1 WARNING for an all-path-shaped registry; '
            f'got {len(warnings)}: {[r.message for r in warnings]}'
        )
        # Assert MEANING, not wording: the message must name the consequence
        # (the filter is now permissive), not merely complain about an input.
        assert 'permissive' in warnings[0].getMessage().lower()

    def test_warning_is_emitted_only_once_per_process(self, monkeypatch, caplog):
        """Warn-once: a scanner on a write path must not turn one
        misconfiguration into a per-call log storm."""
        monkeypatch.setattr(canonical_labels, '_all_path_shaped_warned', False)
        with caplog.at_level('WARNING', logger='fused_memory.utils.canonical_labels'):
            _canonical_allowlist({'-home-leo-bad'})
            _canonical_allowlist({'-home-leo-other'})
        warnings = [r for r in caplog.records if r.levelname == 'WARNING']
        assert len(warnings) == 1, (
            f'Expected exactly 1 WARNING across two calls (warn-once); '
            f'got {len(warnings)}: {[r.message for r in warnings]}'
        )

    @pytest.mark.parametrize('registry', [None, set(), frozenset(), []])
    def test_empty_and_none_registry_stay_silent(self, monkeypatch, caplog, registry):
        """The load-bearing half. A missing/empty registry is pre-existing,
        documented, and expected on every deployment that never wires one — and
        ``validate_known_project_id`` already owns the warning for it. Warning
        here would add a new noise source on the COMMON path; only the
        misconfigured path is loud."""
        monkeypatch.setattr(canonical_labels, '_all_path_shaped_warned', False)
        with caplog.at_level('WARNING', logger='fused_memory.utils.canonical_labels'):
            _canonical_allowlist(registry)
        warnings = [r for r in caplog.records if r.levelname == 'WARNING']
        assert warnings == [], f'Empty/None registry must stay silent; got {[r.message for r in warnings]}'

    def test_fully_usable_registry_stays_silent(self, monkeypatch, caplog):
        """Only the DEGRADED path is loud — a working registry logs nothing."""
        monkeypatch.setattr(canonical_labels, '_all_path_shaped_warned', False)
        with caplog.at_level('WARNING', logger='fused_memory.utils.canonical_labels'):
            _canonical_allowlist({'dark_factory'})
        warnings = [r for r in caplog.records if r.levelname == 'WARNING']
        assert warnings == [], f'A usable registry must stay silent; got {[r.message for r in warnings]}'


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
        assert scan.refs == ()


class TestScanContentAmbiguityPartition:
    """A task number claimed by BOTH a BARE, unqualified own-project mention
    and a foreign-qualified reference in the same content is genuinely
    ambiguous about which task the facts belong to. EVERY referent carrying
    that number goes to .ambiguous and none to .refs — refuse rather than
    guess. (Only an UNQUALIFIED mention creates the contest;
    TestSelfQualifiedRefsNeverContestForeignRefs pins the other half.)

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
        assert scan.refs == ()
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
        assert scan.refs == ()
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
        assert scan.ambiguous == ()

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
        assert scan.ambiguous == ()

    def test_bare_mention_of_a_different_number_does_not_suppress_the_ref(self):
        """The incident content that motivated the split hook: the bare mention
        names a DIFFERENT task number, so nothing is contested."""
        content = 'Reify task 5181 was cancelled; its work was rerouted to dark_factory:2500.'
        scan = scan_content(content, group_id='reify')
        assert [r.node_name for r in scan.refs] == ['Task 5181', 'dark_factory:2500']
        assert scan.ambiguous == ()

    def test_self_qualified_plus_bare_mention_is_not_ambiguous(self):
        """Both spellings denote the SAME own-project referent, so the
        self->local reclassification (which runs BEFORE this partition) lets
        dedup collapse them into one ref. There is no contest either way: the
        bare mention has no FOREIGN referent of the same number to contest."""
        scan = scan_content('reify:5181 and task 5181', group_id='reify')
        assert [r.node_name for r in scan.refs] == ['Task 5181']
        assert scan.ambiguous == ()

    def test_ambiguous_referents_are_deduplicated_too(self):
        content = 'dark_factory:2500 blocks task 2500; dark_factory:2500 again'
        scan = scan_content(content, group_id='reify')
        assert scan.refs == ()
        assert [r.node_name for r in scan.ambiguous] == ['dark_factory:2500', 'Task 2500']


class TestSelfQualifiedRefsNeverContestForeignRefs:
    """A SELF-qualified reference ('reify:2500' scanned in group reify) is
    EXPLICITLY project-qualified, so it is not evidence of ambiguity about
    anything. Only an UNQUALIFIED bare mention ('task 2500') leaves the project
    in doubt.

    Ambiguity is a property of what the PROSE failed to say. Two qualified
    references — even when one of them names the local project — each already
    name their project, so they can never be mutually ambiguous.

    This closes a regression the extraction introduced by composing two
    individually-reasonable changes: scan_content RECLASSIFIES a self-qualified
    ref into an own-project referent instead of dropping it, and the partition
    contested any own-project referent. Measured on this branch, with a
    self-qualified ref plus a foreign ref and nothing bare anywhere:
    find_cross_project_task_refs('reify:2500 relates to dark_factory:2500',
    group_id='reify') returned refs=['dark_factory:2500'], ambiguous=[]
    pre-3667 (fe89df000d) and refs=[], ambiguous=['dark_factory:2500'] at
    d295f5402e — a real cross-project repair silently suppressed on a path that
    performs destructive edge surgery.
    """

    def test_self_qualified_ref_does_not_contest_a_foreign_ref(self):
        scan = scan_content('reify:2500 relates to dark_factory:2500', group_id='reify')
        assert [r.node_name for r in scan.refs] == ['Task 2500', 'dark_factory:2500']
        assert scan.ambiguous == ()

    def test_the_rule_is_order_independent(self):
        scan = scan_content('dark_factory:2500 relates to reify:2500', group_id='reify')
        assert [r.node_name for r in scan.refs] == ['dark_factory:2500', 'Task 2500']
        assert scan.ambiguous == ()

    def test_unrelated_self_qualified_refs_also_survive(self):
        scan = scan_content('reify:2500 and dark_factory:2500 and reify:99', group_id='reify')
        assert [r.node_name for r in scan.refs] == ['Task 2500', 'dark_factory:2500', 'Task 99']
        assert scan.ambiguous == ()

    def test_self_qualification_is_recognised_after_canonicalization(self):
        """Both sides are canonicalized before the local comparison, so a
        differently-spelled self-qualifier is still self-qualified — and so
        still never contests."""
        scan = scan_content('Reify-Factory:2500 and dark_factory:2500', group_id='reify_factory')
        assert [r.node_name for r in scan.refs] == ['Task 2500', 'dark_factory:2500']
        assert scan.ambiguous == ()

    def test_a_bare_mention_still_contests_even_when_self_qualified_wins_dedup(self):
        """The guard against over-correcting: bare-ness is STICKY per number.

        Here the self-qualified spelling appears FIRST, so it wins first-seen
        dedup and the surviving own-project referent is the one that arrived
        qualified — but the content DOES also say 'task 2500' bare, so the
        contest is real and both sides must still be refused.
        """
        content = 'reify:2500 and task 2500 and dark_factory:2500'
        scan = scan_content(content, group_id='reify')
        assert scan.refs == ()
        assert [r.node_name for r in scan.ambiguous] == ['Task 2500', 'dark_factory:2500']


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
