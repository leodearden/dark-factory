"""AST conformance check: every store-mutating script under
``fused-memory/scripts/`` must call ``assert_store_mutation_allowed`` before
its first mutation (task 4280, landed by task 4848).

``fused_memory.utils.store_mutation_preflight`` documents the rule by hand --
a dated, hand-checked audit of which scripts call the guard -- and states
plainly that nothing enforces it mechanically. This module is that
mechanism: :func:`mutating_calls` and :func:`is_guarded` below are applied
over every script under ``fused-memory/scripts/`` to assert that every
CANDIDATE (a script whose AST contains a call this module recognises as
mutating) also calls the guard, imported from the real module. The only
escape hatch is
``fused_memory.utils.store_mutation_preflight.PREFLIGHT_EXEMPT_SCRIPTS``,
itself policed here for staleness so it cannot become a graveyard of dead
exemptions.

WHAT COUNTS AS A MUTATING CALL, AND WHY CALL SHAPE RATHER THAN SUBSTRATE CONTACT
---------------------------------------------------------------------------------
Constructing a ``MemoryService``, or importing a Qdrant/graph client, is a
READ capability too -- a dozen scripts in this repo do exactly that and
mutate nothing. A detector triggered by construction would false-positive on
every new read-only script, which is precisely the failure this check exists
to avoid. Mutation is instead recognised by CALL SHAPE, in two tiers:

  TIER A -- a distinctive mutating callee name (``delete_memory``,
  ``update_edge``, ``delete_collection``, ``upsert``, ...) is flagged
  regardless of receiver.

  TIER B -- a generic verb (``delete``, ``update``, ``add``, ``save``) is
  flagged only when its receiver's dotted name carries a substrate hint
  (``qdrant``, ``mem0``, ``graph``, ``driver``, ``backend``). This tier is
  load-bearing, not defensive padding: it is the only thing that catches
  ``qdrant_client.delete(...)`` and ``memory.mem0.update(...)`` -- the two
  real, currently-guarded spellings the production module's own docstring
  calls out as a mutation no pattern search finds, because ``.update(`` also
  matches every dict update in the repo. The receiver-hint set was narrowed
  by measurement: an early draft including ``'collection'``/``'store'``/
  ``'client'`` produced a real false positive on ``collections.update(...)``,
  a plain dict update in ``bake_off_storage_shape.py``; dropping those three
  removed it while still catching every genuine site.

AST, NOT GREP. A docstring or comment that merely names ``delete_memory`` or
``assert_store_mutation_allowed`` must not be flagged as a candidate, nor
read as guarded -- prose about the rule is not conformance to it. Guard
detection requires BOTH a call to ``assert_store_mutation_allowed`` AND that
the name be imported from ``fused_memory.utils.store_mutation_preflight``: a
script that defines its own no-op ``assert_store_mutation_allowed`` and
calls it would otherwise read as compliant, which is a real fail-open this
import check closes.

WHAT THIS CHECK PROVABLY DOES NOT COVER
----------------------------------------
  * PRESENCE, not PLACEMENT: ``is_guarded`` proves a call to
    ``assert_store_mutation_allowed`` exists somewhere in the module and is
    correctly imported, but never compares its line number against the
    first mutating call, and cannot tell a probe made once per run from one
    made once per RECORD inside a loop. Both of
    ``store_mutation_preflight.py``'s "Two placement rules" -- probe before
    the scan, and a refusal that RAISES rather than returning a
    report-shaped outcome -- remain entirely hand-enforced; this check would
    not have caught a misplaced probe on its own, which is the shape closest
    to the sweep_toolcall_xml_leak incident this whole module exists
    because of;
  * raw Cypher/SQL built inside a STRING LITERAL is invisible to a
    call-shaped AST check -- ``migrate_cross_graph_leak.py``'s graph
    ``DETACH DELETE`` is classified a non-candidate by this detector. It is
    guarded by hand, so greenness is unaffected, but this check would not
    have caught it if it were not;
  * ``cgl_eta_auto_apply_impl.py`` is a deliberate NON-candidate: its only
    mutation is ``migrate_cross_graph_leak.run()`` reached through
    ``importlib``, so it inherits migrate's guard and needs no allowlist
    entry. Being undetected and being exempt are different dispositions --
    conflating them would re-file it as a gap. The production module's
    GUARDED BY INHERITANCE column is the only place that records it;
  * writes ``MemoryService.initialize()`` performs before ``run()`` is ever
    called (Graphiti index creation, the W6-epsilon dup-uuid-edge
    scan-and-repair) are outside this check entirely, and remain tracked by
    tasks 4318 and 4350.

Hermetic by construction for the classifier's own unit tests below: tmp_path
+ synthetic sources only, mirroring test_store_mutation_preflight.py's
stated convention -- no dependence on the live ``fused-memory/scripts/``
tree, so a future script edit cannot silently change what the Tier A / Tier
B / decoy-import cases mean. The live tree is exercised separately, by the
repo-wide conformance test.

NOT integration-marked: this module only parses source, so it must run in
the default ``-m 'not integration'`` lane with no live services -- the
configuration least able to notice a regression here.
"""

from __future__ import annotations

import ast
import pathlib

import pytest
from _ast_guard import calls_named, imported_names_from, parse_python_module

from fused_memory.utils.store_mutation_preflight import PREFLIGHT_EXEMPT_SCRIPTS

# ---------------------------------------------------------------------------
# The classifier (task 4280 / 4848)
# ---------------------------------------------------------------------------
#
# Lives in this test module rather than in production source, mirroring the
# sibling AST guards (test_falkor_index_barrier_guard.py's
# _discover_live_index_modules, test_gather_idiom_helper_routing.py's
# _gather_calls_with_return_exceptions): the classifier has exactly one
# consumer -- this file's own assertions -- and promoting it to
# src/fused_memory/ would give it an audience it does not have.
# ---------------------------------------------------------------------------

#: The guard this whole module polices calls to.
GUARD_CALLABLE = 'assert_store_mutation_allowed'
#: Where it must be imported FROM to count -- see is_guarded().
GUARD_MODULE = 'fused_memory.utils.store_mutation_preflight'

#: Tier A -- distinctive mutating callee names, flagged regardless of
#: receiver. No plain read method collides with any of these spellings.
MUTATING_CALL_NAMES: frozenset[str] = frozenset({
    'delete_memory',
    'update_memory',
    'add_memory',
    'add_episode',
    'delete_episode',
    'delete_entity',
    'update_edge',
    'delete_edge',
    'delete_collection',
    'create_collection',
    'set_payload',
    'overwrite_payload',
    'delete_points',
    'upsert',
    'delete_all',
    'reset',
})

#: Tier B -- generic verbs, counted only when the receiver hints at a
#: substrate (see SUBSTRATE_RECEIVER_HINTS below). Load-bearing, not
#: defensive padding: this is the ONLY tier that catches
#: `qdrant_client.delete(...)` and `memory.mem0.update(...)` -- the two
#: spellings the production module's docstring calls a mutation no pattern
#: search finds, because `.update(` also matches every dict update in the
#: repo.
GENERIC_MUTATING_VERBS: frozenset[str] = frozenset({'delete', 'update', 'add', 'save'})

#: Tightened by MEASUREMENT, not taste: an early draft also included
#: 'collection', 'store' and 'client', and 'collection' produced a real
#: false positive on `collections.update(...)`, a plain dict update in
#: bake_off_storage_shape.py. Dropping those three removed it while still
#: catching every genuine site.
SUBSTRATE_RECEIVER_HINTS: tuple[str, ...] = ('qdrant', 'mem0', 'graph', 'driver', 'backend')


def _dotted_receiver(node: ast.Attribute) -> str:
    """Render the dotted receiver chain of an attribute access, e.g. ``memory.mem0``.

    Recurses through nested ``ast.Attribute``/``ast.Name`` nodes so a
    multi-hop receiver renders in full. Any other expression as the ultimate
    base (a call result, a subscript, ...) renders as ``'<expr>'`` rather
    than raising -- Tier B only needs to test *membership* of a hint
    substring, not a faithful source reprint.
    """
    if isinstance(node.value, ast.Name):
        base = node.value.id
    elif isinstance(node.value, ast.Attribute):
        base = _dotted_receiver(node.value)
    else:
        base = '<expr>'
    return f'{base}.{node.attr}'


def mutating_calls(tree: ast.Module) -> list[tuple[str, int]]:
    """Every Tier A / Tier B mutating call in *tree*, as ``(callee_name, lineno)``.

    Tier A hits are unconditional on the receiver. Tier B hits require the
    receiver's dotted name to contain a substrate hint -- see the module
    docstring for why both tiers exist and why the hint set is shaped the
    way it is. A module with zero hits is not a mutation CANDIDATE at all,
    which is what keeps a purely read-only script (one that only constructs
    a MemoryService and calls search/get/count methods) out of scope.
    """
    hits: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            name = func.id
            receiver = ''
        elif isinstance(func, ast.Attribute):
            name = func.attr
            if isinstance(func.value, ast.Name):
                receiver = func.value.id
            elif isinstance(func.value, ast.Attribute):
                receiver = _dotted_receiver(func.value)
            else:
                receiver = ''
        else:
            continue

        # Tier A is unconditional on the receiver; Tier B requires a
        # substrate-hinting one. Combined into one `or` (rather than kept as
        # two `if`/`elif` branches) so ruff's duplicate-body check does not
        # flag it -- both conditions lead to the identical append.
        if name in MUTATING_CALL_NAMES or (
            name in GENERIC_MUTATING_VERBS
            and any(hint in receiver for hint in SUBSTRATE_RECEIVER_HINTS)
        ):
            hits.append((name, node.lineno))
    return hits


def is_guarded(tree: ast.Module) -> bool:
    """True iff *tree* both calls the guard AND imports it from the real module.

    Both halves are required. A call alone could be satisfied by a
    locally-defined no-op decoy carrying the same name; an import alone
    proves nothing was ever invoked. Either half missing means the script is
    not actually protected against the half-completed-mutation failure mode
    this module exists to close.
    """
    if not calls_named(tree, GUARD_CALLABLE):
        return False
    return GUARD_CALLABLE in imported_names_from(tree, GUARD_MODULE)


#: Pinned census mirroring EXPECTED_EXEMPT_SCRIPTS further below: any
#: WIDENING of MUTATING_CALL_NAMES requires a deliberate edit here too.
#: Parametrizing the test below directly off MUTATING_CALL_NAMES already
#: catches a REMOVAL (the parametrize list, and so test coverage, shrinks
#: with it); this pin is what catches the other direction -- an ADDITION
#: would otherwise ship with a passing parametrized case and no reviewer
#: ever forced to look at it.
EXPECTED_TIER_A_NAMES = frozenset({
    'delete_memory',
    'update_memory',
    'add_memory',
    'add_episode',
    'delete_episode',
    'delete_entity',
    'update_edge',
    'delete_edge',
    'delete_collection',
    'create_collection',
    'set_payload',
    'overwrite_payload',
    'delete_points',
    'upsert',
    'delete_all',
    'reset',
})


class TestMutatingCallsTierA:
    """Tier A: a distinctive mutating callee name is flagged regardless of receiver."""

    def test_tier_a_census_matches_the_reviewed_list(self):
        assert MUTATING_CALL_NAMES == EXPECTED_TIER_A_NAMES, (
            'MUTATING_CALL_NAMES has drifted from the reviewed census.\n'
            f'  unexpected additions: '
            f'{sorted(MUTATING_CALL_NAMES - EXPECTED_TIER_A_NAMES)}\n'
            f'  missing entries:      '
            f'{sorted(EXPECTED_TIER_A_NAMES - MUTATING_CALL_NAMES)}\n'
            'Update EXPECTED_TIER_A_NAMES here to match -- this pin exists so a '
            'new Tier A name is always a deliberate, reviewed edit, exercised by '
            'the parametrized test below rather than shipping silently.'
        )

    @pytest.mark.parametrize('callee', sorted(MUTATING_CALL_NAMES))
    def test_distinctive_mutating_callee_is_flagged_with_name_and_lineno(
        self, tmp_path, callee
    ):
        source = tmp_path / 'candidate.py'
        source.write_text(f'def run():\n    memory.{callee}(x)\n')

        hits = mutating_calls(parse_python_module(source))

        assert (callee, 2) in hits, (
            f'{callee}(...) at line 2 was not flagged as a mutating call; got {hits}'
        )


class TestMutatingCallsTierB:
    """Tier B: a generic verb is flagged only when the receiver hints at a substrate."""

    def test_qdrant_client_delete_is_flagged(self, tmp_path):
        source = tmp_path / 'candidate.py'
        source.write_text('def run():\n    qdrant_client.delete(ids=[1])\n')

        hits = mutating_calls(parse_python_module(source))

        assert ('delete', 2) in hits, (
            f'qdrant_client.delete(...) must be flagged -- this is one of the two '
            f'real spellings the production module docstring calls unsweepable; '
            f'got {hits}'
        )

    def test_memory_mem0_update_is_flagged(self, tmp_path):
        source = tmp_path / 'candidate.py'
        source.write_text('def run():\n    memory.mem0.update(x)\n')

        hits = mutating_calls(parse_python_module(source))

        assert ('update', 2) in hits, (
            f'memory.mem0.update(...) must be flagged -- the other of the two '
            f'real spellings the production module docstring calls unsweepable; '
            f'got {hits}'
        )

    @pytest.mark.parametrize(
        'snippet',
        [
            "counts.update({'a': 1})",
            'results.update(other)',
            'collections.update(x)',
        ],
    )
    def test_generic_receiver_without_a_substrate_hint_is_not_flagged(
        self, tmp_path, snippet
    ):
        source = tmp_path / 'candidate.py'
        source.write_text(f'def run():\n    {snippet}\n')

        hits = mutating_calls(parse_python_module(source))

        assert hits == [], (
            f'{snippet!r} has no substrate-hinting receiver (qdrant/mem0/graph/'
            f'driver/backend) and must not be flagged -- this is the '
            f'false-positive guard that makes Tier B usable at all; got {hits}'
        )


class TestAstNotGrep:
    """Prose that merely names the tokens this module looks for must not
    satisfy or trip either the candidate check or the guard check."""

    def test_docstring_or_comment_mention_is_neither_flagged_nor_guarded(
        self, tmp_path
    ):
        source = tmp_path / 'candidate.py'
        source.write_text(
            '"""This module never calls delete_memory, and never calls\n'
            'assert_store_mutation_allowed either -- both names are only\n'
            'mentioned here, in prose."""\n'
            '# delete_memory and assert_store_mutation_allowed, again, in a comment.\n'
            'def run():\n'
            '    pass\n'
        )
        tree = parse_python_module(source)

        assert mutating_calls(tree) == [], (
            'a docstring/comment mentioning delete_memory must not be treated as '
            'a mutating call -- prose is not conformance'
        )
        assert is_guarded(tree) is False, (
            'a docstring/comment mentioning assert_store_mutation_allowed must '
            'not be treated as guarded -- prose is not conformance'
        )


class TestIsGuardedRequiresCallAndProvenance:
    """is_guarded requires BOTH a call AND that the name be imported from the
    real module -- a locally-defined decoy must not count."""

    def test_call_plus_correct_import_is_guarded(self, tmp_path):
        source = tmp_path / 'candidate.py'
        source.write_text(
            'from fused_memory.utils.store_mutation_preflight import '
            'assert_store_mutation_allowed\n\n'
            'def run():\n'
            "    assert_store_mutation_allowed(operation='run')\n"
        )

        assert is_guarded(parse_python_module(source)) is True

    def test_locally_defined_decoy_is_not_guarded(self, tmp_path):
        source = tmp_path / 'candidate.py'
        source.write_text(
            'def assert_store_mutation_allowed(*, operation):\n'
            '    pass\n\n'
            'def run():\n'
            "    assert_store_mutation_allowed(operation='run')\n"
        )

        assert is_guarded(parse_python_module(source)) is False, (
            'a locally-defined no-op assert_store_mutation_allowed must not read '
            'as guarded -- this is the exact fail-open the import-provenance '
            'check exists to close'
        )

    def test_call_imported_from_the_wrong_module_is_not_guarded(self, tmp_path):
        source = tmp_path / 'candidate.py'
        source.write_text(
            'from somewhere.other import assert_store_mutation_allowed\n\n'
            'def run():\n'
            "    assert_store_mutation_allowed(operation='run')\n"
        )

        assert is_guarded(parse_python_module(source)) is False

    def test_import_without_a_call_is_not_guarded(self, tmp_path):
        source = tmp_path / 'candidate.py'
        source.write_text(
            'from fused_memory.utils.store_mutation_preflight import '
            'assert_store_mutation_allowed\n\n'
            'def run():\n'
            '    pass\n'
        )

        assert is_guarded(parse_python_module(source)) is False


class TestReadOnlyModuleIsNotACandidate:
    """Constructing a MemoryService and calling only read methods must never
    make a module a mutation candidate -- construction is a read capability too."""

    def test_construction_and_read_only_calls_are_not_flagged(self, tmp_path):
        source = tmp_path / 'candidate.py'
        source.write_text(
            'def run():\n'
            '    memory = MemoryService(config)\n'
            "    memory.search(query='x')\n"
            "    memory.get_memories_by_metadata(project_id='p')\n"
            "    memory.count_memories_by_metadata(project_id='p')\n"
            '    return memory\n'
        )

        hits = mutating_calls(parse_python_module(source))

        assert hits == [], (
            'constructing a MemoryService and calling only read methods must '
            f'never make a module a mutation candidate; got {hits}'
        )


# ---------------------------------------------------------------------------
# Repo-wide conformance (task 4280 / 4848)
# ---------------------------------------------------------------------------
#
# Everything above pins the classifier's own semantics against synthetic
# sources. Everything below applies it to the real fused-memory/scripts/
# tree -- the actual enforcement this task exists to add.
# ---------------------------------------------------------------------------

SCRIPTS_ROOT = pathlib.Path(__file__).parents[1] / 'scripts'

#: Union of every token a qualifying ast.Call's callee name could possibly
#: be. Used only as a raw-text prefilter, never as the classification
#: itself -- see _discover_candidate_scripts.
_PREFILTER_TOKENS = MUTATING_CALL_NAMES | GENERIC_MUTATING_VERBS

#: Measured at plan time (task 4848) by running this same two-tier detector
#: over the repo's 43 scripts. A FLOOR, not a pin: discovery is free to find
#: MORE as new mutating scripts are added (and does -- the live count already
#: exceeds it, since two scripts landed guarded after the measurement was
#: taken). It must never find FEWER, which is what would mean the detection
#: criteria regressed.
CANDIDATE_FLOOR = 14


def _discover_candidate_scripts() -> list[pathlib.Path]:
    """Every script under SCRIPTS_ROOT whose AST contains a Tier A/B mutating call.

    A raw ``read_text()`` prefilter runs first: a file whose text contains
    NONE of the Tier A names nor any Tier B verb cannot possibly contain a
    qualifying ``ast.Call`` -- the identifier has to appear literally in the
    source for the call to exist -- so the prefilter is a strict superset of
    the AST criterion and cannot hide a real candidate. It exists purely so
    the majority of scripts that plainly never mutate are not parsed and
    memoised (via parse_python_module's session cache) for the rest of the
    run.
    """
    found: list[pathlib.Path] = []
    for path in sorted(SCRIPTS_ROOT.glob('*.py')):
        text = path.read_text()
        if not any(token in text for token in _PREFILTER_TOKENS):
            continue
        if mutating_calls(parse_python_module(path)):
            found.append(path)
    return found


CANDIDATE_SCRIPTS = _discover_candidate_scripts()


class TestCandidateDiscoveryIsNotVacuous:
    """Discovery must keep finding at least the measured candidate count.

    Without this floor, a detector whose criteria silently stopped matching
    (a rename of the Tier A/B tokens, a broken prefilter) would parametrize
    the conformance test below over an empty set and report green having
    checked nothing.
    """

    def test_candidate_discovery_is_not_vacuous(self):
        assert len(CANDIDATE_SCRIPTS) >= CANDIDATE_FLOOR, (
            f'candidate discovery under {SCRIPTS_ROOT} found only '
            f'{len(CANDIDATE_SCRIPTS)} mutation candidate(s), below the measured '
            f'floor of {CANDIDATE_FLOOR}. Fix the criteria; do NOT lower the '
            f'floor, or this guard silently checks nothing.'
        )


@pytest.mark.parametrize('path', CANDIDATE_SCRIPTS, ids=lambda p: p.name)
class TestEveryMutatingScriptCallsTheGuard:
    """Repo-wide conformance: every discovered candidate calls the guard,
    unless explicitly exempted in PREFLIGHT_EXEMPT_SCRIPTS."""

    def test_every_mutating_script_calls_the_guard(self, path):
        if path.name in PREFLIGHT_EXEMPT_SCRIPTS:
            return

        tree = parse_python_module(path)
        if is_guarded(tree):
            return

        hits = mutating_calls(tree)
        offenders = '; '.join(
            f'scripts/{path.name}:{lineno} calls {name}(...)' for name, lineno in hits
        )
        pytest.fail(
            f'{offenders} -- found: no call to {GUARD_CALLABLE} anywhere in this '
            f'module; expected: guarded. Remedy: call {GUARD_CALLABLE} from '
            f'{GUARD_MODULE} once per run before the scan, NOT per record. This '
            f'matters because a probe placed after the scan (or inside a '
            f'per-record loop) is exactly the shape that let '
            f'sweep_toolcall_xml_leak --apply destroy memory '
            f'7d073281-4c5d-4ba3-a01c-3a167f4460f4 -- a half-completed '
            f'delete-then-re-add split across the Qdrant/mem0 substrate '
            f"boundary. Do not add an allowlist entry unless this script's "
            f'blast radius is statically bounded to scratch substrate; write '
            f'the reasoning into the script itself first (see '
            f'bake_off_storage_shape.py / cleanup_test_collections.py for the '
            f'shape that reasoning takes).'
        )


# ---------------------------------------------------------------------------
# PREFLIGHT_EXEMPT_SCRIPTS anti-rot suite (task 4280 / 4848)
# ---------------------------------------------------------------------------
#
# The allowlist is the conformance check's ONLY escape hatch, and an
# allowlist with no staleness policing silently becomes a graveyard: a stale
# entry keeps suppressing the check for a path nothing occupies, and would
# grandfather a NEW script later created at that name. The closest in-repo
# precedent, DRAIN_ALLOWLIST in test_gather_convention_guard.py, has exactly
# this gap -- its check only fires on an OVER-count. The four checks below
# are adapted instead from test_check_bare_magicmock_config.py, the one
# module in the repo where entry-exists, still-a-live-violation,
# reason-non-empty and key-set-equality all exist together.
# ---------------------------------------------------------------------------

#: A reason shorter than this cannot be a real blast-radius argument -- it
#: catches an empty string or a placeholder like "safe" or "TODO" without
#: going anywhere near the length of a genuine paragraph (every real reason
#: in PREFLIGHT_EXEMPT_SCRIPTS today runs past 400 characters).
_MIN_SUBSTANTIVE_REASON_LENGTH = 40

#: The reviewed census this allowlist is expected to hold, pinned here so
#: any WIDENING of PREFLIGHT_EXEMPT_SCRIPTS requires a deliberate edit in
#: BOTH this file and fused_memory/utils/store_mutation_preflight.py. The
#: list is expected to SHRINK as scripts are migrated onto the guard
#: directly, never to grow -- a newly-written mutating script should call
#: assert_store_mutation_allowed, not earn an entry here.
EXPECTED_EXEMPT_SCRIPTS = frozenset({
    'bake_off_storage_shape.py',
    'cleanup_test_collections.py',
})


class TestPreflightExemptScriptsAntiRot:
    """Anti-rot suite for PREFLIGHT_EXEMPT_SCRIPTS, the conformance check's
    only escape hatch.

    Without these four checks an allowlist entry can silently outlive the
    condition that justified it: a renamed/deleted file leaves a blanket
    suppression nothing occupies; a script that starts calling the guard, or
    stops mutating altogether, stays quietly exempted forever; an empty or
    placeholder reason defeats the entire point of a str-valued allowlist;
    and with no key-set pin, a third exemption could be added alongside an
    unrelated change with no reviewer ever forced to look at it.
    """

    def test_every_exempt_entry_names_an_existing_script(self):
        """A deleted or renamed script must not leave a stale blanket exemption behind."""
        missing = [
            name for name in PREFLIGHT_EXEMPT_SCRIPTS if not (SCRIPTS_ROOT / name).is_file()
        ]
        assert missing == [], (
            f'PREFLIGHT_EXEMPT_SCRIPTS names script(s) that no longer exist: {missing}. '
            'A stale entry silently exempts a path nothing occupies -- and would '
            'grandfather a NEW file later created at that name. Remove the entry in '
            'fused_memory/utils/store_mutation_preflight.py.'
        )

    def test_no_exempt_entry_is_stale(self):
        """Every exempt key must still be a live, unguarded mutation candidate.

        If a script later calls the guard, or stops mutating altogether, its
        entry has outlived the condition that justified it and must be deleted
        -- not left behind to grandfather whatever that file becomes next.
        """
        stale: list[tuple[str, str]] = []
        for name in PREFLIGHT_EXEMPT_SCRIPTS:
            path = SCRIPTS_ROOT / name
            if not path.is_file():
                continue  # covered by test_every_exempt_entry_names_an_existing_script
            tree = parse_python_module(path)
            if is_guarded(tree):
                stale.append((name, 'it now calls the guard'))
            elif not mutating_calls(tree):
                stale.append((name, 'it no longer contains a mutating call'))
        assert stale == [], '\n'.join(
            f"remove PREFLIGHT_EXEMPT_SCRIPTS['{name}'] in "
            f'fused_memory/utils/store_mutation_preflight.py -- {reason}'
            for name, reason in stale
        ) + (
            '\nAn exemption that no longer matches a live, unguarded candidate is '
            'dead weight that would silently grandfather any future edit to that '
            'file.'
        )

    def test_every_exempt_entry_has_a_substantive_reason(self):
        """A blank or placeholder reason is a silent exemption."""
        weak = {
            name: reason
            for name, reason in PREFLIGHT_EXEMPT_SCRIPTS.items()
            if len(reason.strip()) < _MIN_SUBSTANTIVE_REASON_LENGTH
        }
        assert weak == {}, (
            f'PREFLIGHT_EXEMPT_SCRIPTS entries with no substantive reason: {weak!r}. '
            'A blank or placeholder reason is a silent exemption -- write the actual '
            'blast-radius argument (see bake_off_storage_shape.py / '
            'cleanup_test_collections.py for the shape it takes).'
        )

    def test_exempt_key_set_matches_the_reviewed_census(self):
        """The allowlist may only shrink; widening it is a deliberate two-file edit."""
        actual = set(PREFLIGHT_EXEMPT_SCRIPTS)
        assert actual == EXPECTED_EXEMPT_SCRIPTS, (
            'PREFLIGHT_EXEMPT_SCRIPTS has drifted from the reviewed census.\n'
            f'  unexpected additions: {sorted(actual - EXPECTED_EXEMPT_SCRIPTS)}\n'
            f'  missing entries:      {sorted(EXPECTED_EXEMPT_SCRIPTS - actual)}\n'
            'This list is expected to SHRINK, never grow: a newly-written mutating '
            'script should call assert_store_mutation_allowed directly rather than '
            'earn an allowlist entry. Widening it requires a deliberate edit both '
            'here and in fused_memory/utils/store_mutation_preflight.py.'
        )
