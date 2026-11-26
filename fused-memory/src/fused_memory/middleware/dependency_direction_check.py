"""Post-write check of extracted dependency-DIRECTION facts against Taskmaster (task 3770).

Graphiti's extraction names real, adjacent task ids but can get the direction of
the relation between them wrong. It FLATTENS two parallel tasks (siblings that
share a dependent or a dependency) into "A waits behind B", and it INVERTS a
transitive chain. Either fact reads as plausible, so it is checked against the
live dependency graph instead of being trusted.

Pure: no I/O. The caller parses an episode's edges once with
``extract_dependency_facts``. An empty result means the episode is out of scope
and no ground truth need be read. Otherwise the caller builds a
``DependencyIndex`` from Taskmaster's edge set and classifies the facts with
``check_dependency_direction``.

Every ambiguity resolves to NOT flagging: a false positive retires a true fact,
while a false negative only leaves a wrong one unflagged.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from fused_memory.reconciliation.task_filter import (
    STRICT_CLAUSE_BOUNDARY_RE,
    TASK_REF_RE,
)

# One alternation per direction feeds BOTH the cheap gate and the positional
# binder, so the gate can never refuse a fact the binder would accept.

#: The FIRST task reference is the dependent, the second its dependency.
_FORWARD_PHRASE_ALT = (
    r'(?:'
    r'waits?\s+behind'
    r'|waiting\s+behind'
    r'|needs?'
    r'|requires?'
    r'|depends?\s+(?:up)?on'
    r'|dependent\s+(?:up)?on'
    r'|blocked\s+(?:by|on)'
    r'|waits?\s+(?:on|for)'
    r'|waiting\s+(?:on|for)'
    r')'
)

#: The FIRST task reference is the dependency, the second its dependent.
_INVERSE_PHRASE_ALT = (
    r'(?:'
    r'blocks?'
    r'|blocking'
    r'|gates?'
    r'|gating'
    r'|unblocks?'
    r'|a\s+(?:hard\s+|blocking\s+)?dependency\s+(?:of|for)'
    r'|a\s+(?:hard\s+|blocking\s+)?prerequisite\s+(?:of|for)'
    r'|depended\s+(?:up)?on\s+by'
    r')'
)

#: Optional PRESENT-tense copula ("is blocked by"). Past tense is deliberately
#: absent: "was blocked by" is history, not a claim about the current graph.
_COPULA_ALT = r'(?:(?:is|are|remains?|stays?|sits?)\s+)?'

#: Adverbs that may sit before the phrase without changing its meaning.
_ADVERB_ALT = r'(?:(?:currently|still|now|already|directly|transitively)\s+)*'

_PHRASE_GATE_RE: re.Pattern[str] = re.compile(
    r'\b(?:' + _FORWARD_PHRASE_ALT + r'|' + _INVERSE_PHRASE_ALT + r')',
    re.IGNORECASE,
)

#: Matched against the WHOLE gap between two adjacent task references, so a
#: phrase binds only the pair it sits literally between.
_PHRASE_GAP_RE: re.Pattern[str] = re.compile(
    r'\A\s*' + _COPULA_ALT + _ADVERB_ALT
    + r'(?:(?P<forward>' + _FORWARD_PHRASE_ALT + r')'
    + r'|(?P<inverse>' + _INVERSE_PHRASE_ALT + r'))\s*\Z',
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DependencyAssertion:
    """A claim that ``dependent`` waits for ``dependency``.

    Always in that orientation, whichever way round the source phrase wrote it.
    ``phrase`` is the matched connective, verbatim.
    """

    dependent: int
    dependency: int
    phrase: str


def extract_dependency_assertions(fact: object) -> list[DependencyAssertion]:
    """Return the directional dependency claims *fact* makes, or ``[]``.

    A phrase binds only the two ADJACENT task references it sits between, and
    only within one clause, so "A waits behind B, C needs D" yields exactly two
    assertions. Clauses split on ``STRICT_CLAUSE_BOUNDARY_RE`` because an
    assertion can end in a retired edge, where a longer clause is the unsafe
    direction. Never raises: a non-string or blank fact yields ``[]``.
    """
    if not isinstance(fact, str) or not _PHRASE_GATE_RE.search(fact):
        return []

    assertions: list[DependencyAssertion] = []
    for clause in STRICT_CLAUSE_BOUNDARY_RE.split(fact):
        refs = [
            (int(m.group(1)), m.start(), m.end())
            for m in TASK_REF_RE.finditer(clause)
        ]
        for (first, _, first_end), (second, second_start, _) in zip(
            refs, refs[1:], strict=False
        ):
            match = _PHRASE_GAP_RE.match(clause[first_end:second_start])
            if match is None:
                continue
            if match.group('forward'):
                assertions.append(
                    DependencyAssertion(first, second, match.group('forward'))
                )
            else:
                assertions.append(
                    DependencyAssertion(second, first, match.group('inverse'))
                )
    return assertions


@dataclass(frozen=True)
class DependencyFact:
    """An extracted edge whose fact makes at least one dependency assertion."""

    edge_uuid: str
    fact: str
    assertions: tuple[DependencyAssertion, ...]


def _edge_field(edge: object, name: str) -> str:
    """Read *name* off an attribute- or dict-shaped edge; ``''`` if unreadable."""
    value = (
        edge.get(name) if isinstance(edge, Mapping) else getattr(edge, name, None)
    )
    return value if isinstance(value, str) else ''


def extract_dependency_facts(edges: Iterable[object]) -> list[DependencyFact]:
    """Parse an episode's edges once. This is the scope gate for the whole check.

    Returns one entry per edge that has a uuid and whose fact makes at least one
    assertion. ``[]`` means the episode is out of scope and ground truth need
    not be read. An unreadable edge is skipped, never raised on.
    """
    facts: list[DependencyFact] = []
    for edge in edges or ():
        edge_uuid = _edge_field(edge, 'uuid')
        fact = _edge_field(edge, 'fact')
        assertions = extract_dependency_assertions(fact)
        if edge_uuid and assertions:
            facts.append(DependencyFact(edge_uuid, fact, tuple(assertions)))
    return facts


@dataclass(frozen=True)
class DependencyIndex:
    """The ground-truth dependency graph.

    ``direct`` maps each task to its own dependencies and ``dependents`` is the
    reverse map. Every task in EITHER edge column is a key of both, so a leaf
    task is still a known node.
    """

    direct: Mapping[int, frozenset[int]]
    dependents: Mapping[int, frozenset[int]]

    def closure_of(self, task_id: int) -> frozenset[int]:
        """Every task *task_id* transitively waits for.

        Iterative, so neither chain depth nor a cycle can break it. A task on a
        cycle reaches itself.
        """
        reachable: set[int] = set()
        stack = list(self.direct.get(task_id, ()))
        while stack:
            node = stack.pop()
            if node not in reachable:
                reachable.add(node)
                stack.extend(self.direct.get(node, ()))
        return frozenset(reachable)


def build_dependency_index(edges: Mapping[int, Iterable[int]]) -> DependencyIndex:
    """Build a :class:`DependencyIndex` from a ``{task_id: [depends_on, ...]}`` map.

    Accepts the shape ``SqliteTaskBackend.get_dependency_edges`` returns, where a
    task with no dependencies is absent from the map. The caller's mapping is
    never mutated.
    """
    direct: dict[int, set[int]] = {}
    dependents: dict[int, set[int]] = {}
    for task_id, deps in (edges or {}).items():
        node = int(task_id)
        direct.setdefault(node, set())
        dependents.setdefault(node, set())
        for raw_dep in deps or ():
            dep = int(raw_dep)
            direct[node].add(dep)
            direct.setdefault(dep, set())
            dependents.setdefault(dep, set()).add(node)
    return DependencyIndex(
        direct={k: frozenset(v) for k, v in direct.items()},
        dependents={k: frozenset(v) for k, v in dependents.items()},
    )


#: Ground truth runs the other way: ``dependency`` waits for ``dependent``.
REVERSED = 'reversed'
#: Two parallel tasks, sharing a dependent or a dependency, stated as sequential.
SIBLING_SEQUENTIAL = 'sibling_sequential'
#: Both ids are known, no path runs either way, and they are not siblings.
UNSUPPORTED = 'unsupported'
#: The classifications that positively contradict ground truth, and so justify
#: retiring the edge. UNSUPPORTED is not one: "no relation in Taskmaster" is
#: also what a true ordering it does not model looks like, such as merge-queue
#: order or a dependency stated before it is wired.
CONTRADICTIONS: frozenset[str] = frozenset({REVERSED, SIBLING_SEQUENTIAL})


def classify_dependency_assertion(
    assertion: DependencyAssertion, index: DependencyIndex
) -> str | None:
    """Classify *assertion* against *index*; ``None`` means do not flag.

    The rules apply in this order, and the order is behaviour:

    1. A self-reference, or either id unknown to the index -> ``None``. Unknown
       means unknown, not wrong.
    2. ``dependency`` reachable from ``dependent`` -> ``None``: supported,
       directly or transitively. This must precede rule 4, because a real edge
       between two tasks that also share a neighbour is still supported.
    3. ``dependent`` reachable from ``dependency`` -> ``REVERSED``.
    4. The pair shares a dependent or a dependency -> ``SIBLING_SEQUENTIAL``.
    5. Otherwise -> ``UNSUPPORTED``.
    """
    dependent, dependency = assertion.dependent, assertion.dependency
    if dependent == dependency:
        return None
    if dependent not in index.direct or dependency not in index.direct:
        return None
    if dependency in index.closure_of(dependent):
        return None
    if dependent in index.closure_of(dependency):
        return REVERSED
    if (index.dependents[dependent] & index.dependents[dependency]) or (
        index.direct[dependent] & index.direct[dependency]
    ):
        return SIBLING_SEQUENTIAL
    return UNSUPPORTED


@dataclass(frozen=True)
class DependencyDirectionFinding:
    """One extracted fact whose direction ground truth does not support.

    ``fact`` is the extraction's own wording, verbatim. ``ground_truth`` carries
    enough of the graph to adjudicate the finding without re-reading it.
    """

    edge_uuid: str
    fact: str
    dependent: int
    dependency: int
    classification: str
    ground_truth: Mapping[str, object]

    @property
    def contradicts_ground_truth(self) -> bool:
        """Whether the classification is one of ``CONTRADICTIONS``."""
        return self.classification in CONTRADICTIONS

    def to_dict(self) -> dict[str, object]:
        """A JSON-safe record, with every set rendered as a sorted list."""
        return {
            'edge_uuid': self.edge_uuid,
            'fact': self.fact,
            'dependent': self.dependent,
            'dependency': self.dependency,
            'classification': self.classification,
            'contradicts_ground_truth': self.contradicts_ground_truth,
            'ground_truth': {
                key: sorted(value) if isinstance(value, (set, frozenset)) else value
                for key, value in dict(self.ground_truth).items()
            },
        }


def check_dependency_direction(
    facts: Iterable[DependencyFact], index: DependencyIndex
) -> list[DependencyDirectionFinding]:
    """Return one finding per assertion ``classify_dependency_assertion`` flags.

    Detection only: no fact is repaired or reworded.
    """
    findings: list[DependencyDirectionFinding] = []
    for fact in facts:
        for assertion in fact.assertions:
            classification = classify_dependency_assertion(assertion, index)
            if classification is None:
                continue
            findings.append(
                DependencyDirectionFinding(
                    edge_uuid=fact.edge_uuid,
                    fact=fact.fact,
                    dependent=assertion.dependent,
                    dependency=assertion.dependency,
                    classification=classification,
                    ground_truth=_ground_truth(assertion, index),
                )
            )
    return findings


def _ground_truth(
    assertion: DependencyAssertion, index: DependencyIndex
) -> dict[str, object]:
    dependent, dependency = assertion.dependent, assertion.dependency
    return {
        'phrase': assertion.phrase,
        'dependent_direct_dependencies': index.direct[dependent],
        'dependency_direct_dependencies': index.direct[dependency],
        'shared_dependents': index.dependents[dependent]
        & index.dependents[dependency],
        'shared_dependencies': index.direct[dependent] & index.direct[dependency],
    }
