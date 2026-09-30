"""Unit tests for fused_memory.utils.task_naming.

The module answers one question — "which task-node family does this name belong
to?" — in two views over ONE acceptance rule: ``canonicalize_task_node_name``
returns the family's canonical NAME and ``task_node_referent`` returns its
structured ``Referent`` identity. Both are pure and dependency-free, and are
used by the post-add_episode node-name-normalization hook (MemoryService.
_normalize_task_node_names, task 2110) to detect and correct non-canonical
task-entity node names (e.g. 'task 132', 'tasks 153') minted by graphiti-core's
LLM entity extraction, without ever touching legitimate non-task-node names.
"""
from __future__ import annotations

import re

import pytest

from fused_memory.utils import task_naming
from fused_memory.utils.canonical_labels import Referent, parse_node_name
from fused_memory.utils.task_naming import (
    canonicalize_task_node_name,
    group_task_node_families,
    task_node_referent,
)

# A Unicode decimal digit that ``str.isdigit()`` accepts but ``str.isascii()``
# does not — spelled by ESCAPE so the fixture survives transport and a reader
# sees the codepoint rather than a glyph that renders like an ASCII '3'.
ARABIC_INDIC_THREE = '\u0663'  # ARABIC-INDIC DIGIT THREE

#: Every name the two views are asserted to agree on \u2014 matches and non-matches
#: in one table, because the equivalence has to hold in BOTH directions. Shared
#: by TestTaskNodeReferent's agreement test, which is what pins
#: canonicalize_task_node_name and task_node_referent to a SINGLE acceptance
#: rule: an edit that loosens or tightens either one alone fails here.
TWO_VIEW_AGREEMENT_NAMES = [
    'task 132',
    'tasks 153',
    'TASK 42',
    'Task  7',
    ' tasks 9 ',
    'task #1153',
    'task#1153',
    'TASK # 1153',
    'Task: 132',
    'Task 42',
    'task 0132',
    'Alice',
    '',
    'task',
    'subtask 5',
    'multitask 3',
    'taskforce 9',
    'Task 42 orchestrator',
    'reify task 12',
    'task132',
    'reify:132',
    'Task ' + ARABIC_INDIC_THREE,
    'Task 12' + ARABIC_INDIC_THREE,
]


class TestCanonicalizeTaskNodeNameMatches:
    """Bare 'task N' / 'tasks N' node names (any case, extra whitespace) canonicalize
    to 'Task N', preserving the matched digits verbatim."""

    @pytest.mark.parametrize(
        ('name', 'expected'),
        [
            ('task 132', 'Task 132'),
            ('tasks 153', 'Task 153'),
            ('TASK 42', 'Task 42'),
            ('Task  7', 'Task 7'),
            (' tasks 9 ', 'Task 9'),
            ('Task 42', 'Task 42'),  # idempotence: already-canonical maps to itself
        ],
    )
    def test_canonicalizes_to_expected(self, name, expected):
        assert canonicalize_task_node_name(name) == expected

    def test_idempotent_on_already_canonical_name(self):
        """Applying canonicalize_task_node_name to its own output is a no-op change
        (same string in, same string out) — required for the hook's canonical!=name
        guard to treat already-canonical nodes as untouched."""
        once = canonicalize_task_node_name('task 132')
        assert once is not None  # narrows str | None -> str for the chained call below
        twice = canonicalize_task_node_name(once)
        assert once == 'Task 132'
        assert twice == 'Task 132'


class TestCanonicalizeTaskNodeNameNonMatches:
    """Non-task-node names, and anything beyond a bare 'task(s) N', return None
    so the caller leaves the node name untouched — no false positives."""

    @pytest.mark.parametrize(
        'name',
        [
            'Alice',
            '',
            'task',  # no number
            'subtask 5',
            'multitask 3',
            'taskforce 9',
            'Task 42 orchestrator',
            'reify task 12',
            # A Unicode digit is not a task number. task_naming owns no pattern
            # of its own, so these assert the canonical_labels narrowing
            # propagates through this adapter. Measured RED before that fix:
            # 'Task \u0663' canonicalized to itself and 'Task 12\u0663' to
            # 'Task 12\u0663', either of which the normalization hook would have
            # written back onto a real graph node.
            'Task ' + ARABIC_INDIC_THREE,
            # The mixed run is the sharper case: it must yield None, never the
            # TRUNCATED prefix 'Task 12', which names a REAL node the hook would
            # happily rename. See TestUnicodeDigitsAreNotTaskNumbers in
            # tests/test_canonical_labels.py for the full rationale.
            'Task 12' + ARABIC_INDIC_THREE,
        ],
    )
    def test_returns_none(self, name):
        assert canonicalize_task_node_name(name) is None


class TestCanonicalizeTaskNodeNameVariantSpellings:
    """Variant spellings of the same task label now canonicalize (task 3667).

    Before the canonical_labels extraction this function owned its own
    compiled pattern, ``^\\s*tasks?\\s+(\\d+)\\s*$``. The ``\\s+`` between the
    word and the digits means it was STRUCTURALLY unable to see 'task #1153' —
    that name returned None and the node was left alone, keeping it split from
    the canonical 'Task 1153' node. It is one of the PRD's 53 measured
    task-node variant splits, and collapsing it is exactly what makes this
    extraction a fix rather than only a refactor: the single consumer,
    MemoryService._normalize_task_node_names, now renames such a node onto the
    canonical form.
    """

    @pytest.mark.parametrize(
        ('name', 'expected'),
        [
            ('task #1153', 'Task 1153'),
            ('Task #1153', 'Task 1153'),
            ('task#1153', 'Task 1153'),
            ('TASK # 1153', 'Task 1153'),
            # A task-vocabulary qualifier is local task vocabulary, never a
            # project id — the same rule cross_project_refs enforces.
            ('Task: 132', 'Task 132'),
        ],
    )
    def test_variant_spellings_canonicalize(self, name, expected):
        assert canonicalize_task_node_name(name) == expected

    def test_glued_digits_still_do_not_match(self):
        """Pins that the added '#'/':' alternation did not widen the pattern
        beyond those two separators. 'task132' was a non-match before the
        extraction and must stay one — the obvious spelling
        'tasks?\\s*#?\\s*(\\d+)' would have matched it, silently changing what
        the normalization hook renames."""
        assert canonicalize_task_node_name('task132') is None

    def test_variant_spelling_result_is_idempotent(self):
        """Required by the hook's ``canonical != name`` guard: re-applying the
        function to its own output must be a no-op, or the hook would rename
        the same node on every pass."""
        once = canonicalize_task_node_name('task #1153')
        assert once == 'Task 1153'
        assert once is not None  # narrows str | None -> str for the chained call
        assert canonicalize_task_node_name(once) == 'Task 1153'


class TestTaskNodeReferent:
    """``task_node_referent`` is the STRUCTURED view of the acceptance rule
    ``canonicalize_task_node_name`` answers as a string.

    It exists because the family-keyed normalizer needs BOTH halves of a task
    label: the canonical NAME to rename onto, and the bare DIGITS to probe the
    backend with (``find_entity_nodes_by_name_substring``). Recovering the
    digits by splitting 'Task 605' back apart would be an ad-hoc parser over a
    meaningful string; ``canonical_labels.Referent`` is already the frozen
    structured carrier of exactly that pair, so returning it costs no new type
    and cannot drift from the name it renders.
    """

    @pytest.mark.parametrize(
        ('name', 'expected_number'),
        [
            ('task 132', '132'),
            ('tasks 153', '153'),
            ('TASK 42', '42'),
            ('Task  7', '7'),
            (' tasks 9 ', '9'),
            ('task #1153', '1153'),
            ('task#1153', '1153'),
            ('TASK # 1153', '1153'),
            ('Task: 132', '132'),
            ('Task 42', '42'),  # idempotence: already-canonical still parses
        ],
    )
    def test_returns_own_project_referent_carrying_number_and_canonical_name(
        self, name, expected_number
    ):
        referent = task_node_referent(name)
        assert referent == Referent(kind='task', number=expected_number)
        assert referent is not None  # narrows Referent | None for the reads below
        assert referent.number == expected_number
        assert referent.node_name == f'Task {expected_number}'
        assert referent.project_id == ''

    def test_number_preserves_leading_zeros_verbatim(self):
        """'0132' is a DIFFERENT family from '132' and must stay one.

        The number is the substring the normalizer probes with, so
        int-normalizing it here would both invent a task number and make two
        distinct families share one key.
        """
        referent = task_node_referent('task 0132')
        assert referent == Referent(kind='task', number='0132')
        assert referent is not None  # narrows Referent | None for the reads below
        assert referent.number == '0132'
        assert referent.node_name == 'Task 0132'
        assert referent != Referent(kind='task', number='132')

    @pytest.mark.parametrize(
        'name',
        [
            'Alice',
            '',
            'task',  # no number
            'subtask 5',
            'multitask 3',
            'taskforce 9',
            'Task 42 orchestrator',
            'reify task 12',
            'task132',  # the separator is required
            'Task ' + ARABIC_INDIC_THREE,
            'Task 12' + ARABIC_INDIC_THREE,
        ],
    )
    def test_returns_none_for_every_non_task_node_name(self, name):
        assert task_node_referent(name) is None

    def test_project_qualified_name_is_never_a_local_family_key(self):
        """The load-bearing refusal: 'reify:132' PARSES but is not our family.

        ``parse_node_name`` resolves it to a foreign referent, and folding that
        into the local 'Task 132' family would have the normalization hook
        commit the very cross-project misattribution utils/cross_project_refs.py
        exists to detect — so the rejection lives in this adapter, at the one
        site both views read.
        """
        assert parse_node_name('reify:132') == Referent(
            kind='task', project_id='reify', number='132'
        )
        assert task_node_referent('reify:132') is None

    @pytest.mark.parametrize('name', TWO_VIEW_AGREEMENT_NAMES)
    def test_the_two_views_agree_by_construction(self, name):
        """One acceptance rule, two renderings — asserted, not asserted-about.

        ``canonicalize_task_node_name`` is re-expressed over
        ``task_node_referent``, so this can only fail if a future edit gives one
        of them a rule of its own.
        """
        referent = task_node_referent(name)
        canonical = canonicalize_task_node_name(name)
        if referent is None:
            assert canonical is None
        else:
            assert canonical == referent.node_name


class TestGroupTaskNodeFamilies:
    """Partitioning nodes by family is THE operation both consumers need.

    ``MemoryService._normalize_task_node_names`` uses it to filter the backend's
    substring candidates down to one family; ``maintenance/task_family_census``
    uses it to partition a whole graph. It is a named function rather than two
    inlined loops so the family rule stays at one site.
    """

    def test_every_spelling_of_one_number_collapses_into_one_family(self):
        """The keying the whole task turns on: four spellings, one key."""
        nodes = [
            {'name': 'Task 605'},
            {'name': 'task 605'},
            {'name': 'tasks 605'},
            {'name': 'task #605'},
        ]

        families = group_task_node_families(nodes)

        assert list(families) == [Referent(kind='task', number='605')]
        assert families[Referent(kind='task', number='605')] == nodes

    def test_distinct_numbers_are_distinct_families(self):
        nodes = [{'name': 'Task 605'}, {'name': 'task 700'}]

        families = group_task_node_families(nodes)

        assert families == {
            Referent(kind='task', number='605'): [nodes[0]],
            Referent(kind='task', number='700'): [nodes[1]],
        }

    def test_numbers_merely_containing_the_digits_do_not_join_the_family(self):
        """The precision the normalizer's substring prefilter relies on.

        A ``CONTAINS '605'`` probe hands back 'Task 6051' and 'Task 1605' too;
        this filter is what keeps them out of the 605 family, so the label
        vocabulary never has to be re-expressed inside a Cypher predicate.
        """
        nodes = [{'name': 'Task 605'}, {'name': 'Task 6051'}, {'name': 'Task 1605'}]

        families = group_task_node_families(nodes)

        assert families[Referent(kind='task', number='605')] == [nodes[0]]
        assert families[Referent(kind='task', number='6051')] == [nodes[1]]
        assert families[Referent(kind='task', number='1605')] == [nodes[2]]

    def test_non_task_and_project_qualified_names_are_dropped(self):
        """'reify:605' names a FOREIGN task and must never join the local family.

        It is exactly the kind of node a ``CONTAINS '605'`` probe returns, and
        folding it in would have the normalizer merge another project's node
        into ours.
        """
        nodes = [
            {'name': 'Alice'},
            {'name': 'deploy pipeline'},
            {'name': 'reify:605'},
            {'name': 'release 605 notes'},
        ]

        assert group_task_node_families(nodes) == {}

    def test_within_a_family_input_order_is_preserved_exactly(self):
        """Load-bearing, not incidental: the normalizer takes ``members[0]`` as
        the merge survivor, so the backend's survivor-first ordering has to
        survive the grouping untouched.
        """
        survivor = {'name': 'task 605', 'uuid': 'u-high-edges'}
        second = {'name': 'Task 605', 'uuid': 'u-mid'}
        third = {'name': 'tasks 605', 'uuid': 'u-low'}

        members = group_task_node_families([survivor, second, third])[
            Referent(kind='task', number='605')
        ]

        assert members[0] is survivor
        assert members[1] is second
        assert members[2] is third

    def test_leading_zeros_make_a_different_family(self):
        """Digits are compared verbatim, never int-normalized."""
        padded = {'name': 'Task 0605'}
        bare = {'name': 'Task 605'}

        families = group_task_node_families([padded, bare])

        assert families == {
            Referent(kind='task', number='0605'): [padded],
            Referent(kind='task', number='605'): [bare],
        }

    def test_empty_input_yields_an_empty_dict(self):
        assert group_task_node_families([]) == {}

    def test_a_node_with_a_missing_or_empty_name_is_skipped_not_raised(self):
        """This runs on a best-effort post-commit path, where raising on one
        malformed row would abandon every other family in the batch."""
        good = {'name': 'Task 605'}

        families = group_task_node_families([{'uuid': 'u-1'}, {'name': ''}, good])

        assert families == {Referent(kind='task', number='605'): [good]}


class TestNoSecondCopyOfTheLabelPattern:
    """INV-5 (no lockstep duplication), the invariant task 3667 exists to
    enforce: the task-label vocabulary lives ONLY in utils/canonical_labels.py.

    A vocabulary that exists in two places drifts, and the drift is invisible
    until a destructive consumer acts on the stale half — which is precisely
    how 'task #1153' came to be a task-node name to a human, to
    cross_project_refs' mention scanner, and not to this module.

    Asserted structurally over the imported module's attributes (not as a
    source grep and not as a docstring pin), so re-introducing a compiled copy
    fails the suite rather than merely contradicting a comment.
    """

    def test_module_exposes_no_compiled_pattern(self):
        compiled = [
            name for name, value in vars(task_naming).items() if isinstance(value, re.Pattern)
        ]
        assert compiled == []
