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

import pytest
from _ast_guard import parse_python_module


class TestMutatingCallsTierA:
    """Tier A: a distinctive mutating callee name is flagged regardless of receiver."""

    @pytest.mark.parametrize(
        'callee',
        [
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
        ],
    )
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
            'from somewhere.else import assert_store_mutation_allowed\n\n'
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
