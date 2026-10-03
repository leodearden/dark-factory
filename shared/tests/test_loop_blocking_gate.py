"""Gate: no ``async def`` may reach a blocking primitive without an offload hop.

INV-8 ("no blocking call on the event loop") has now been missed across three
consecutive census batches — task 3778, then task 4091, then task 4201 — each
time finding sites the previous batch had already claimed were clean.  INV-8's
own *Census seam* section names the remedy for exactly that pattern: "a slug
violated repeatedly across census batches is an enforcement gap: file a guard
task".  Task 4484 is that guard task, and this module is the gate.

Why a *caller-side* scanner (the whole point)
---------------------------------------------
Task 3778's census enumerated the sites where the blocking PRIMITIVE is
written — "7 sync ``subprocess.run`` sites ... across 3 modules" — and then
made a per-MODULE offload claim about one of them
(``recon_claim_verification_guard.py``: "already offloaded at its call
sites").  A single caller that wraps the helper in ``asyncio.to_thread``
therefore made the whole module read as clean, and every *other* caller of
that helper was invisible to the census.  That is a definition-side census,
and it cannot see the defect it is looking for.

The question this scanner asks instead is the caller-side one: *which
coroutines reach a blocking primitive without a hop?*  Findings are per CALL
SITE, never per helper — see :class:`TestCallerSideEnumeration`, whose first
test is a direct regression against 3778's methodology.

The triad
---------
Mirrors the established ``shared/tests`` house pattern so a reader who knows
one knows all three:

* ``loop_blocking_scan.py``      — the scanner (pure ``ast``, no I/O)
* ``loop_blocking_allowlist.py`` — the pure-data dispositioned baseline
* ``test_loop_blocking_gate.py`` — this file, the assertions

Same shape as ``silent_fallthrough_{scan,allowlist}`` +
``test_silent_fallthrough_gate`` and ``config_dir_archival_allowlist`` +
``test_config_dir_archival_gate``.  Nothing new is invented.

The synthetic-source tests below carry the real weight.  Once the baseline is
written the live tree is green by construction, so a detector that silently
stopped detecting would still pass the whole-tree gate; only the synthetic
fixtures prove it still fires.
"""

from __future__ import annotations

import ast
import re
import textwrap
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import pytest
from loop_blocking_allowlist import (
    ALLOWLIST_KEYS,
    AUDITED_SITES,
    DISPOSITIONS,
)
from loop_blocking_scan import (
    BUILTIN_PRIMITIVES,
    DOTTED_PRIMITIVES,
    METHOD_PRIMITIVES,
    LoopBlockingSite,
    find_loop_blocking_sites,
    find_loop_blocking_sites_in_trees,
    site_key,
)
from silent_fallthrough_scan import (
    ParsedFile,
    reconcile_against_allowlist,
)


def _src(text: str) -> str:
    """Dedent one triple-quoted chunk of a synthetic module source."""
    return textwrap.dedent(text).strip('\n')


def _module(*chunks: str) -> str:
    """Compose a synthetic module from independently-dedented chunks.

    Each chunk is dedented ON ITS OWN and only then joined.  Interpolating an
    unindented fragment into an indented f-string block instead would defeat
    ``textwrap.dedent`` (the common prefix collapses to ``''``), leaving the
    surrounding lines indented and the "module" an unparseable string -- which
    this scanner is fail-soft about, so every such fixture would pass
    VACUOUSLY.  These fixtures are the only thing proving the detector still
    detects, so a vacuous pass here is the same class of silent hole the gate
    exists to close.
    """
    return '\n\n\n'.join(_src(chunk) for chunk in chunks) + '\n'


# --------------------------------------------------------------------------- #
# The blocking helper used by the caller-side fixtures.  It is deliberately
# written the way the real ones are (task_curator.py's registry loaders): a
# plain sync function whose body reaches a filesystem primitive.
# --------------------------------------------------------------------------- #
_HELPER_DEF = """\
def load_registry(path):
    return path.read_text(encoding='utf-8')
"""


class TestCallerSideEnumeration:
    """The property task 3778's census lacked: findings are per CALL SITE."""

    def test_offloaded_caller_clean_inline_caller_flagged(self):
        """THE GAP REGRESSION (task 3778).

        One module, one blocking helper, two callers: ``a`` hops through
        ``asyncio.to_thread``, ``b`` calls it inline on the loop thread.

        Task 3778's census would return ZERO findings here.  It counted the
        helper's definition site and recorded the module as "already offloaded
        at its call sites" because *a* caller offloads -- which is true, and
        which is exactly why ``b`` stayed invisible.  A definition-side census
        cannot express the difference between these two callers; a caller-side
        one returns EXACTLY ONE finding, naming ``b``.

        This assertion is the reason task 4484 exists.  If it ever starts
        passing vacuously (zero findings), the scanner has regressed to the
        methodology that produced the three missed batches.
        """
        sources = {
            'pkg/mod.py': _module(
                'import asyncio',
                _HELPER_DEF,
                """
                async def a(path):
                    return await asyncio.to_thread(load_registry, path)
                """,
                """
                async def b(path):
                    return load_registry(path)
                """,
            )
        }

        findings = find_loop_blocking_sites(sources)

        assert len(findings) == 1, (
            'expected exactly one caller-side finding (b), got '
            f'{[(f.qualname, f.callee) for f in findings]}'
        )
        assert findings[0].qualname == 'b'
        assert findings[0].callee == 'load_registry'
        assert findings[0].filename == 'pkg/mod.py'

    def test_second_inline_caller_yields_second_finding(self):
        """Findings are per call site, never per helper -- a THIRD caller adds a row.

        A per-helper (or per-module) ledger would collapse ``b`` and ``c`` into
        one entry, and blessing that entry would silently bless every future
        caller.  Two findings here is what keeps the ratchet honest.
        """
        sources = {
            'pkg/mod.py': _module(
                'import asyncio',
                _HELPER_DEF,
                """
                async def a(path):
                    return await asyncio.to_thread(load_registry, path)
                """,
                """
                async def b(path):
                    return load_registry(path)
                """,
                """
                async def c(path):
                    return load_registry(path)
                """,
            )
        }

        findings = find_loop_blocking_sites(sources)

        assert sorted(f.qualname for f in findings) == ['b', 'c']

    def test_run_in_executor_is_an_offload_hop(self):
        """``loop.run_in_executor(None, fn, ...)`` offloads just as ``to_thread`` does."""
        sources = {
            'pkg/mod.py': _module(
                'import asyncio',
                _HELPER_DEF,
                """
                async def a(path):
                    loop = asyncio.get_running_loop()
                    return await loop.run_in_executor(None, load_registry, path)
                """,
            )
        }

        assert find_loop_blocking_sites(sources) == []

    def test_to_thread_on_an_unrelated_receiver_is_not_an_offload_hop(self):
        """``helper.to_thread(...)`` is NOT ``asyncio.to_thread(...)``.

        The offload match is the one place in this scanner where a name
        collision costs a MISS instead of the documented silence: matching on
        the attribute name alone lets any object with a ``to_thread`` method
        exempt its first argument from a merge-blocking whole-tree sweep.  The
        receiver is therefore resolved to ``asyncio.to_thread`` /
        ``anyio.to_thread.run_sync`` before the exemption applies.

        The fixture puts the CALL inside the offload argument because that is
        the only shape where the exemption is observable: a recognised hop
        skips the whole argument subtree (that is what makes
        ``to_thread(load_registry, path)`` clean), so a bare name there is
        never a finding either way.
        """
        def _sources(receiver: str) -> dict[str, str]:
            return {
                'pkg/mod.py': _module(
                    'import asyncio',
                    _HELPER_DEF,
                    f"""
                    async def b(helper, path):
                        return {receiver}.to_thread(load_registry(path))
                    """,
                )
            }

        unrelated = find_loop_blocking_sites(_sources('helper'))
        real = find_loop_blocking_sites(_sources('asyncio'))

        assert [(f.qualname, f.callee) for f in unrelated] == [('b', 'load_registry')], (
            "an unrelated receiver's .to_thread() must not exempt its argument "
            f'from the sweep, got {unrelated}'
        )
        assert real == [], (
            'the real asyncio.to_thread hop must still exempt its argument, '
            f'got {real}'
        )

    def test_aliased_and_from_imported_to_thread_are_offload_hops(self):
        """Every spelling of the real hop still offloads: alias and ``from`` import.

        Constraining the match to a RESOLVED dotted path (the test above) must
        not cost the legitimate spellings ``import asyncio as aio`` and ``from
        asyncio import to_thread``, or a correctly-fixed site turns red.  Same
        call-inside-the-argument fixture, for the same reason.
        """
        aliased = {
            'pkg/mod.py': _module(
                'import asyncio as aio',
                _HELPER_DEF,
                """
                async def a(path):
                    return await aio.to_thread(load_registry(path))
                """,
            )
        }
        from_imported = {
            'pkg/mod.py': _module(
                'from asyncio import to_thread',
                _HELPER_DEF,
                """
                async def a(path):
                    return await to_thread(load_registry(path))
                """,
            )
        }

        assert find_loop_blocking_sites(aliased) == []
        assert find_loop_blocking_sites(from_imported) == []

    def test_awaited_async_callee_that_offloads_is_not_a_finding(self):
        """An ``await``ed coroutine that itself hops is clean at both levels.

        ``outer`` awaits ``inner``; ``inner`` awaits ``to_thread``.  Neither is
        a finding: awaiting a coroutine yields to the loop, and the blocking
        work happens on a worker thread.  Flagging this shape would make the
        gate fire on correctly-fixed code, which is the fastest way to train
        reviewers to bless rows unread.
        """
        sources = {
            'pkg/mod.py': _module(
                'import asyncio',
                _HELPER_DEF,
                """
                async def inner(path):
                    return await asyncio.to_thread(load_registry, path)
                """,
                """
                async def outer(path):
                    return await inner(path)
                """,
            )
        }

        assert find_loop_blocking_sites(sources) == []

    def test_syntax_error_is_fail_soft(self):
        """A mid-edit file yields ``[]`` and never raises.

        A whole-tree gate that crashes on an unparseable file turns every
        in-progress edit into a red gate.  Fail-soft is the house idiom
        (``silent_fallthrough_scan.find_violations``).
        """
        assert find_loop_blocking_sites({'pkg/broken.py': 'def ( oops\n'}) == []

    def test_syntax_error_does_not_suppress_other_modules(self):
        """One unparseable module must not silence the rest of the sweep."""
        sources = {
            'pkg/broken.py': 'def ( oops\n',
            'pkg/mod.py': _module(
                _HELPER_DEF,
                """
                async def b(path):
                    return load_registry(path)
                """,
            ),
        }

        findings = find_loop_blocking_sites(sources)

        assert [f.qualname for f in findings] == ['b']

    def test_unresolvable_bare_name_is_silence_not_a_finding(self):
        """An unbound callee is NOT a finding -- unresolvable is silence, never a false RED.

        The throwaway script used to size task 4484's audit matched callees by
        bare name across the whole tree, so a common name like ``check`` or
        ``run`` could bind to an unrelated module's blocking function.  A false
        RED in a whole-tree gate is worse than a miss: it blocks every merge
        until someone blesses a non-defect.
        """
        sources = {
            'pkg/mod.py': _module(
                """
                async def b(path):
                    return mystery_helper(path)
                """,
            )
        }

        assert find_loop_blocking_sites(sources) == []

    def test_a_parameter_shadowing_a_helper_name_is_silence(self):
        """A callee that is a PARAMETER cannot be the module's def of that name.

        ``async def b(load_registry, path)`` calls whatever the caller passed
        in; the module-level ``def load_registry`` is not reachable through
        that name at all.  Resolving it anyway is a false RED, and the whole
        point of a merge-blocking gate is that it never manufactures one.
        """
        sources = {
            'pkg/mod.py': _module(
                _HELPER_DEF,
                """
                async def b(load_registry, path):
                    return load_registry(path)
                """,
            )
        }

        assert find_loop_blocking_sites(sources) == []

    def test_a_local_rebinding_of_a_helper_name_is_silence(self):
        """A local assignment shadows the module def for the rest of the scope."""
        sources = {
            'pkg/mod.py': _module(
                _HELPER_DEF,
                """
                async def b(path):
                    load_registry = lambda p: 1

                    return load_registry(path)
                """,
            )
        }

        assert find_loop_blocking_sites(sources) == []

    def test_a_class_method_is_not_reachable_as_a_bare_name(self):
        """``Loader.load`` is not what a bare ``load(...)`` elsewhere calls.

        Indexing class methods under their bare names would resolve any
        same-named parameter or local call to a method that is not in scope --
        the shape of a false RED, reported by review against the delivered
        scanner.
        """
        sources = {
            'pkg/mod.py': _module(
                """
                class Loader:
                    def load(self, path):
                        return path.read_text()
                """,
                """
                async def b(load, path):
                    return load(path)
                """,
            )
        }

        assert find_loop_blocking_sites(sources) == []

    def test_a_nested_def_is_not_visible_from_a_sibling_scope(self):
        """THE LIVE FALSE POSITIVE (task 4484 amendment pass).

        ``make_probe`` closes over a nested ``probe`` that shells out;
        ``verify`` takes a ``probe`` PARAMETER and calls it.  The two names are
        unrelated -- ``verify`` cannot see inside ``make_probe`` -- but a scanner
        that indexes defs at any nesting depth resolves one to the other and
        reports ``verify``'s callers as reaching ``subprocess.run``.

        This is not hypothetical.  The delivered scanner produced exactly this
        row against the live tree
        (``server/tools.py::create_mcp_server._completion_claim_gate ->
        verify_claims``, via ``services/completion_claim_gate.py``'s
        ``make_commit_probe.probe`` and ``_verify_task``'s ``probe``
        parameter), and it was blessed in the ledger as a real defect.  The
        genuine site next door -- ``_claim_commit_presence ->
        make_commit_probe`` -- was unaffected and still reported.
        """
        sources = {
            'pkg/mod.py': _module(
                'import subprocess',
                """
                def make_probe(root):
                    def probe(sha):
                        return subprocess.run(['git', 'cat-file', '-e', sha])

                    return probe
                """,
                """
                def verify(claim, probe):
                    return probe(claim)
                """,
                """
                async def handler(claim):
                    return verify(claim, lambda ref: None)
                """,
            )
        }

        assert find_loop_blocking_sites(sources) == [], (
            'a nested def must not resolve a same-named parameter in an '
            'unrelated scope'
        )

    def test_an_enclosing_scope_helper_still_resolves(self):
        """A closure IS visible to the coroutines defined beside it.

        The counterpart to the test above, and the live shape the ledger's
        largest cluster is made of: ``server/tools.py::create_mcp_server``
        defines a nested ``_normalize_project_root`` and 22 nested MCP handlers
        call it.  Restricting resolution to module level alone would silence
        all 22 -- correctness here is scope visibility, not nesting depth.
        """
        sources = {
            'pkg/mod.py': _module(
                _HELPER_DEF,
                """
                def create_server():
                    def _normalize(path):
                        return load_registry(path)

                    async def handler(path):
                        return _normalize(path)

                    return handler
                """,
            )
        }

        findings = find_loop_blocking_sites(sources)

        assert [(f.qualname, f.callee) for f in findings] == [
            ('create_server.handler', '_normalize')
        ], f'the enclosing-scope closure must still resolve, got {findings}'

    def test_cross_module_import_binding_resolves(self):
        """``from X import Y`` then an inline ``Y(...)`` -- the real task_curator shape.

        All four confirmed ``task_curator.py`` sites are written exactly this
        way (a function-local ``from fused_memory.middleware.<registry> import
        load_...`` followed by an inline call), so this is the resolution path
        that has to work for the whole-tree gate to find them.
        """
        sources = {
            'pkg/registry.py': _module(_HELPER_DEF),
            'pkg/caller.py': _module(
                """
                async def b(path):
                    from pkg.registry import load_registry

                    return load_registry(path)
                """,
            ),
        }

        findings = find_loop_blocking_sites(sources)

        assert len(findings) == 1, (
            f'expected the cross-module call to resolve, got {findings}'
        )
        assert findings[0].filename == 'pkg/caller.py'
        assert findings[0].qualname == 'b'
        assert findings[0].callee == 'load_registry'

    def test_cross_module_call_to_a_clean_helper_is_not_a_finding(self):
        """The cross-module resolver must not flag an imported helper that never blocks.

        Pairs with the test above: together they show the resolver is deciding
        on the callee's BODY, not merely on the fact that it was imported.
        """
        sources = {
            'pkg/registry.py': _module(
                """
                def parse_registry(text):
                    return text.split(',')
                """,
            ),
            'pkg/caller.py': _module(
                """
                async def b(text):
                    from pkg.registry import parse_registry

                    return parse_registry(text)
                """,
            ),
        }

        assert find_loop_blocking_sites(sources) == []

    def test_sync_caller_of_a_blocking_helper_is_not_a_finding(self):
        """Only coroutines can block the loop; a sync caller is out of scope."""
        sources = {
            'pkg/mod.py': _module(
                _HELPER_DEF,
                """
                def b(path):
                    return load_registry(path)
                """,
            )
        }

        assert find_loop_blocking_sites(sources) == []

    def test_self_method_call_resolves(self):
        """``self.<method>(...)`` resolves within the enclosing class.

        This is the ``reconciliation/harness.py::ReconciliationHarness._escalate``
        shape -- a sync method with ~10 async callers, all of them written
        ``self._escalate(...)``.  Without this resolution path that whole
        cluster is invisible to the sweep.
        """
        sources = {
            'pkg/mod.py': _module(
                """
                class Harness:
                    def _escalate(self, path):
                        return path.read_text()

                    async def run(self, path):
                        return self._escalate(path)
                """,
            )
        }

        findings = find_loop_blocking_sites(sources)

        assert [(f.qualname, f.callee) for f in findings] == [
            ('Harness.run', '_escalate')
        ]


class TestFindingIdentity:
    """Findings carry a drift-resistant key -- never a line number."""

    _SOURCES = {
        'pkg/mod.py': _module(
            _HELPER_DEF,
            """
            async def b(path):
                return load_registry(path)
            """,
        )
    }

    def test_site_key_is_relpath_qualname_content_hash(self):
        (finding,) = find_loop_blocking_sites(self._SOURCES)

        assert site_key(finding) == (
            finding.filename,
            finding.qualname,
            finding.content_hash,
        )

    def test_content_hash_is_twelve_hex_chars(self):
        (finding,) = find_loop_blocking_sites(self._SOURCES)

        assert len(finding.content_hash) == 12
        assert all(ch in '0123456789abcdef' for ch in finding.content_hash)

    def test_key_survives_pure_line_drift(self):
        """Inserting unrelated lines above the site must not invalidate its blessing.

        ``silent_fallthrough_scan.violation_key`` already learned this lesson:
        the key is ``(relpath, qualname, content_hash)`` with
        ``content_hash = sha256(ast.unparse(node))[:12]``, invariant under
        reindentation and edits above the site.  A ``lineno``-keyed baseline
        churns on every neighbouring change, and a baseline that churns is one
        reviewers re-bless without reading.
        """
        (before,) = find_loop_blocking_sites(self._SOURCES)

        shifted = {
            'pkg/mod.py': _module(
                '# a new comment\n# and another',
                _HELPER_DEF,
                """
                async def b(path):
                    return load_registry(path)
                """,
            )
        }
        (after,) = find_loop_blocking_sites(shifted)

        assert site_key(before) == site_key(after)
        assert before.lineno != after.lineno, (
            'fixture is not exercising line drift -- the site did not move'
        )

    def test_identity_names_the_callee(self):
        """``(relpath, qualname, callee)`` is the human-readable identity.

        The whole-tree known-site floor asserts against this triple, because
        ``content_hash`` changes whenever the site is edited but the defect
        identity ("this coroutine calls that blocking helper") does not.
        """
        (finding,) = find_loop_blocking_sites(self._SOURCES)

        assert (finding.filename, finding.qualname, finding.callee) == (
            'pkg/mod.py',
            'b',
            'load_registry',
        )


class TestScannerHygiene:
    """Cheap structural guarantees the whole-tree sweep depends on."""

    def test_empty_sources_yield_no_findings(self):
        assert find_loop_blocking_sites({}) == []

    def test_recursive_helper_does_not_hang(self):
        """Cycle-safe reachability: mutual recursion must terminate.

        Without a ``seen`` set the transitive walk would recurse forever on
        this pair and take the whole suite down with it.
        """
        sources = {
            'pkg/mod.py': _module(
                """
                def ping(path):
                    return pong(path)
                """,
                """
                def pong(path):
                    return ping(path)
                """,
                """
                async def b(path):
                    return ping(path)
                """,
            )
        }

        assert find_loop_blocking_sites(sources) == []

    def test_recursive_helper_that_blocks_is_still_found(self):
        """A cycle must not become an escape hatch that hides a real primitive."""
        sources = {
            'pkg/mod.py': _module(
                """
                def ping(path, depth=0):
                    if depth > 3:
                        return path.read_text()
                    return pong(path, depth + 1)
                """,
                """
                def pong(path, depth):
                    return ping(path, depth)
                """,
                """
                async def b(path):
                    return ping(path)
                """,
            )
        }

        findings = find_loop_blocking_sites(sources)

        assert [f.qualname for f in findings] == ['b']


def _trees(sources: dict[str, str]) -> dict[str, ast.Module]:
    return {relpath: ast.parse(source, filename=relpath) for relpath, source in sources.items()}


#: A caller in one module reaching a blocking helper defined in another.
_CROSS_MODULE_SOURCES = {
    'pkg/registry.py': _module(_HELPER_DEF),
    'pkg/caller.py': _module(
        """
        async def b(path):
            from pkg.registry import load_registry

            return load_registry(path)
        """,
    ),
}

_CLEAN_SOURCES = {
    'pkg/mod.py': _module(
        """
        async def a(text):
            return text.split(',')
        """,
    ),
}

#: Task 5099's async-consumption shape: an awaited read_text is an async API,
#: a bare one in a sibling coroutine is a finding.
_ASYNC_CONSUMPTION_SOURCES = {
    'pkg/mod.py': _module(
        """
        async def a(apath):
            return await apath.read_text()
        """,
        """
        async def d(path):
            return path.read_text()
        """,
    ),
}


class TestTreesEntryPoint:
    """``find_loop_blocking_sites_in_trees`` is what the gate calls on the shared ASTs.

    ``find_loop_blocking_sites`` parses and delegates to it, so the two must
    agree exactly on every mapping.
    """

    @pytest.mark.parametrize(
        ('sources', 'expected'),
        [
            (_CROSS_MODULE_SOURCES, [('pkg/caller.py', 'b', 'load_registry')]),
            (_CLEAN_SOURCES, []),
            (_ASYNC_CONSUMPTION_SOURCES, [('pkg/mod.py', 'd', 'read_text')]),
        ],
        ids=['cross-module', 'clean', 'async-consumption'],
    )
    def test_matches_the_source_entry_point_exactly(self, sources, expected):
        from_trees = find_loop_blocking_sites_in_trees(_trees(sources))
        assert from_trees == find_loop_blocking_sites(sources)
        assert [(f.filename, f.qualname, f.callee) for f in from_trees] == expected

    def test_an_unparseable_module_left_out_of_the_trees_contributes_nothing(self):
        """Mirrors the source entry point's SyntaxError skip."""
        sources = {**_CROSS_MODULE_SOURCES, 'pkg/broken.py': 'def ( oops\n'}
        assert find_loop_blocking_sites_in_trees(_trees(_CROSS_MODULE_SOURCES)) == (
            find_loop_blocking_sites(sources)
        )


# --------------------------------------------------------------------------- #
# The primitive table (task 4484 gap B)
# --------------------------------------------------------------------------- #
#
# Task 3778's census enumerated `subprocess.run` and nothing else, while INV-8's
# own Rule text names "subprocess, network, filesystem, lock, sleep".  Both of
# task 4201's missed sites and one of task 4091's are FILESYSTEM, not
# subprocess, so this narrowing alone accounts for them even if the census had
# been caller-side.  One parameter per primitive, so a regression names the
# offender rather than reporting "the table shrank".

_PRIMITIVE_CASES = [
    # (case id, helper body line, expected LoopBlockingSite.primitive)
    ('subprocess.run', "return subprocess.run(['git', 'status'])", 'subprocess.run'),
    ('subprocess.check_output', "return subprocess.check_output(['git'])",
     'subprocess.check_output'),
    ('subprocess.check_call', "return subprocess.check_call(['git'])",
     'subprocess.check_call'),
    ('subprocess.Popen', "return subprocess.Popen(['git'])", 'subprocess.Popen'),
    ('yaml.safe_load', 'return yaml.safe_load(payload)', 'yaml.safe_load'),
    ('yaml.safe_dump', 'return yaml.safe_dump(payload)', 'yaml.safe_dump'),
    ('Path.read_text', 'return payload.read_text()', 'read_text'),
    ('Path.write_text', "return payload.write_text('x')", 'write_text'),
    ('Path.read_bytes', 'return payload.read_bytes()', 'read_bytes'),
    ('Path.write_bytes', "return payload.write_bytes(b'x')", 'write_bytes'),
    ('fcntl.flock', 'return fcntl.flock(payload, 2)', 'fcntl.flock'),
    ('time.sleep', 'return time.sleep(0.1)', 'time.sleep'),
    ('socket.create_connection', "return socket.create_connection(('h', 1))",
     'socket.create_connection'),
    ('os.system', "return os.system('ls')", 'os.system'),
    # -- task 5099: the directory-walk / metadata methods, matched by name
    ('Path.mkdir', 'return payload.mkdir()', 'mkdir'),
    ('Path.rmdir', 'return payload.rmdir()', 'rmdir'),
    ('Path.touch', 'return payload.touch()', 'touch'),
    ('Path.unlink', 'return payload.unlink()', 'unlink'),
    ('Path.rename', "return payload.rename('x')", 'rename'),
    ('Path.exists', 'return payload.exists()', 'exists'),
    ('Path.is_file', 'return payload.is_file()', 'is_file'),
    ('Path.is_dir', 'return payload.is_dir()', 'is_dir'),
    ('Path.stat', 'return payload.stat()', 'stat'),
    ('Path.iterdir', 'return list(payload.iterdir())', 'iterdir'),
    ('Path.glob', "return list(payload.glob('*'))", 'glob'),
    ('Path.rglob', "return list(payload.rglob('*'))", 'rglob'),
    ('Path.open', 'return payload.open()', 'open'),
    # -- task 5099: their os / glob spellings, receiver-pinned.  A dotted match
    # is checked before the method match, so os.stat reports 'os.stat'.
    ('os.makedirs', 'return os.makedirs(payload)', 'os.makedirs'),
    ('os.mkdir', 'return os.mkdir(payload)', 'os.mkdir'),
    ('os.rmdir', 'return os.rmdir(payload)', 'os.rmdir'),
    ('os.remove', 'return os.remove(payload)', 'os.remove'),
    ('os.unlink', 'return os.unlink(payload)', 'os.unlink'),
    ('os.rename', "return os.rename(payload, 'x')", 'os.rename'),
    ('os.replace', "return os.replace(payload, 'x')", 'os.replace'),
    ('os.stat', 'return os.stat(payload)', 'os.stat'),
    ('os.scandir', 'return list(os.scandir(payload))', 'os.scandir'),
    ('os.path.exists', 'return os.path.exists(payload)', 'os.path.exists'),
    ('os.path.isfile', 'return os.path.isfile(payload)', 'os.path.isfile'),
    ('os.path.isdir', 'return os.path.isdir(payload)', 'os.path.isdir'),
    ('glob.glob', 'return glob.glob(payload)', 'glob.glob'),
]

_PRIMITIVE_CASE_IMPORTS = (
    'import fcntl\nimport glob\nimport os\nimport socket\nimport subprocess\n'
    'import time\n\nimport yaml'
)


class TestPrimitiveTable:
    """Gap B: the census vocabulary must be the WHOLE INV-8 vocabulary."""

    @pytest.mark.parametrize(
        ('body', 'expected'),
        [pytest.param(body, expected, id=case_id)
         for case_id, body, expected in _PRIMITIVE_CASES],
    )
    def test_primitive_is_detected_through_a_sync_helper(self, body, expected):
        """Each INV-8 primitive is blocking when a coroutine reaches it without a hop."""
        sources = {
            'pkg/mod.py': _module(
                _PRIMITIVE_CASE_IMPORTS,
                f'def helper(payload):\n    {body}',
                """
                async def caller(payload):
                    return helper(payload)
                """,
            )
        }

        findings = find_loop_blocking_sites(sources)

        assert len(findings) == 1, (
            f'{expected} not detected through a sync helper; '
            f'got {[(f.callee, f.primitive) for f in findings]}'
        )
        assert findings[0].primitive == expected
        assert findings[0].qualname == 'caller'
        assert findings[0].callee == 'helper'

    @pytest.mark.parametrize(
        ('body', 'expected'),
        [pytest.param(body, expected, id=case_id)
         for case_id, body, expected in _PRIMITIVE_CASES],
    )
    def test_primitive_is_detected_directly_in_an_async_body(self, body, expected):
        """A primitive written INLINE in a coroutine needs no helper hop to count.

        This is the ``manifest_stamping::_stamp_capability_manifests_impl``
        shape: ``read_text`` + ``yaml.safe_load`` + ``write_text`` +
        ``yaml.safe_dump`` all in one coroutine body, with no helper anywhere
        for a definition-side census to point at.
        """
        sources = {
            'pkg/mod.py': _module(
                _PRIMITIVE_CASE_IMPORTS,
                f'async def caller(payload):\n    {body}',
            )
        }

        findings = find_loop_blocking_sites(sources)

        assert len(findings) == 1, (
            f'{expected} not detected inline in a coroutine; '
            f'got {[(f.callee, f.primitive) for f in findings]}'
        )
        assert findings[0].primitive == expected
        assert findings[0].qualname == 'caller'

    def test_manifest_stamping_shape_yields_one_finding_per_primitive(self):
        """The four-primitive inline coroutine yields FOUR findings, not one.

        Per call site, never per function: blessing one row must not silently
        bless the other three.
        """
        sources = {
            'pkg/mod.py': _module(
                'import yaml',
                """
                async def stamp(path, out):
                    raw = path.read_text()
                    doc = yaml.safe_load(raw)
                    out.write_text(yaml.safe_dump(doc))
                    return doc
                """,
            )
        }

        findings = find_loop_blocking_sites(sources)

        assert sorted(f.primitive for f in findings) == [
            'read_text', 'write_text', 'yaml.safe_dump', 'yaml.safe_load'
        ]

    def test_builtin_open_is_a_finding_and_a_rebound_open_is_not(self):
        """``open(...)`` blocks, and it is the one primitive nobody imports.

        A bare builtin is invisible to both halves of the table -- it has no
        dotted path and no receiver -- so ``open(p).read()`` in a coroutine
        would have landed silently past this gate.  The rebinding half is what
        keeps the builtin match honest: a module with its own ``open`` helper
        is talking about that helper.
        """
        builtin = {
            'pkg/mod.py': _module(
                """
                async def b(path):
                    with open(path, encoding='utf-8') as fh:
                        return fh.read()
                """,
            )
        }
        rebound = {
            'pkg/mod.py': _module(
                """
                async def b(path, open):
                    return open(path)
                """,
            )
        }

        findings = find_loop_blocking_sites(builtin)

        assert [(f.qualname, f.primitive) for f in findings] == [('b', 'open')], (
            f'a bare builtin open() in a coroutine must be a finding, got {findings}'
        )
        assert find_loop_blocking_sites(rebound) == []

    def test_json_load_through_a_helper_is_a_finding(self):
        """The vocabulary is not YAML-only: a JSON registry blocks identically.

        Gap B (task 3778 enumerated ``subprocess.run`` alone) is a vocabulary
        failure, and a vocabulary that stops at the spellings the tree happens
        to use today re-opens it for the next author who picks a different one.
        """
        sources = {
            'pkg/mod.py': _module(
                'import json',
                """
                def load_registry(fh):
                    return json.load(fh)
                """,
                """
                async def b(fh):
                    return load_registry(fh)
                """,
            )
        }

        findings = find_loop_blocking_sites(sources)

        assert [(f.qualname, f.primitive) for f in findings] == [('b', 'json.load')]

    def test_from_import_binding_resolves_to_the_dotted_primitive(self):
        """``from subprocess import run`` then a bare ``run(...)`` still counts.

        Matching the bare name ``run`` on its own would be a false-RED
        generator; matching it only once the module's import bindings resolve
        it to ``subprocess.run`` is precise.
        """
        sources = {
            'pkg/mod.py': _module(
                'from subprocess import run',
                """
                async def caller(cmd):
                    return run(cmd)
                """,
            )
        }

        findings = find_loop_blocking_sites(sources)

        assert [f.primitive for f in findings] == ['subprocess.run']

    def test_asyncio_sleep_is_not_flagged(self):
        """``await asyncio.sleep(...)`` is the NON-blocking sibling of ``time.sleep``.

        A table that cannot tell them apart flags every correctly-written
        coroutine in the tree and is worthless.
        """
        sources = {
            'pkg/mod.py': _module(
                'import asyncio',
                """
                async def caller():
                    await asyncio.sleep(0.1)
                """,
            )
        }

        assert find_loop_blocking_sites(sources) == []

    def test_a_method_call_consumed_asynchronously_is_not_a_filesystem_primitive(self):
        """An awaited / async-with / async-for method call is an async API.

        The names here are placeholders: the rule is about the CONSTRUCT.  A
        method primitive is matched by attribute name alone, receiver
        unresolved, and a sync pathlib method never returns an awaitable, an
        async context manager or an async iterator -- so a call one of those
        constructs consumes is some other API (anyio.Path, aiofiles, an async
        store's ``open()``), which is exactly what an INV-8 fix switches to.
        """
        consumed = {
            'await operand': """
                async def a(apath):
                    return await apath.read_text()
                """,
            'async-with context expression': """
                async def b(remote):
                    async with remote.read_bytes() as fh:
                        return fh
                """,
            'async-for iterable': """
                async def c(remote):
                    async for chunk in remote.read_bytes():
                        return chunk
                """,
        }
        for shape, body in consumed.items():
            findings = find_loop_blocking_sites({'pkg/mod.py': _module(body)})
            assert findings == [], (
                f'a method call consumed as the {shape} is an async API, not a '
                f'sync filesystem primitive; got '
                f'{[(f.qualname, f.primitive) for f in findings]}'
            )

        bare = find_loop_blocking_sites({'pkg/mod.py': _module(
            """
            async def d(path):
                return path.read_text()
            """,
        )})
        assert [(f.qualname, f.primitive) for f in bare] == [('d', 'read_text')]

        argument_of_awaited = find_loop_blocking_sites({'pkg/mod.py': _module(
            """
            async def e(writer, path):
                await writer.send(path.read_text())
            """,
        )})
        assert [(f.qualname, f.primitive) for f in argument_of_awaited] == [
            ('e', 'read_text')
        ], (
            'only the call that IS the await operand is excluded; an argument '
            'of an awaited call still evaluates on the loop thread'
        )

    def test_names_shared_with_builtin_types_are_not_primitives(self):
        """``replace`` / ``remove`` / ``walk`` stay out of the method table.

        Matching by attribute name alone, they are ``str.replace``,
        ``list.remove`` and ``ast.walk`` far more often than a filesystem
        call, and every string edit in the tree would become a merge-blocking
        row.  Their filesystem spellings are matched only receiver-pinned:
        ``os.replace``, ``os.remove``, ``os.walk``.
        """
        sources = {
            'pkg/mod.py': _module(
                'import ast',
                """
                async def caller(name, items, x, tree):
                    name.replace('-', '_')
                    items.remove(x)
                    return list(ast.walk(tree))
                """,
            )
        }

        findings = find_loop_blocking_sites(sources)

        assert findings == [], (
            'str.replace / list.remove / ast.walk must not match a method '
            f'primitive; got {[(f.qualname, f.primitive) for f in findings]}'
        )

    def test_async_file_apis_sharing_a_primitive_name_are_not_findings(self):
        """The live shapes the async-consumption rule exists for, once 'open' exists.

        ``await cost_store.open()`` is server/main.py::_setup_curator_usage_gate's
        async CostStore open -- the false positive widening the table exposed.
        ``aiofiles.open`` and anyio's ``iterdir`` are the canonical INV-8 fix;
        flagging them would be a false RED on correctly fixed code.
        """
        async_apis = {
            'awaited store open': """
                async def a(cost_store):
                    await cost_store.open()
                """,
            'aiofiles open': """
                async def b(p):
                    async with aiofiles.open(p) as fh:
                        return await fh.read()
                """,
            'anyio iterdir': """
                async def c(apath):
                    async for child in apath.iterdir():
                        return child
                """,
        }
        for shape, body in async_apis.items():
            findings = find_loop_blocking_sites(
                {'pkg/mod.py': _module('import aiofiles', body)}
            )
            assert findings == [], (
                f'{shape} is an async API, not a sync filesystem call; got '
                f'{[(f.qualname, f.primitive) for f in findings]}'
            )

        control = find_loop_blocking_sites({'pkg/mod.py': _module(
            """
            async def d(store):
                return store.open()
            """,
        )})
        assert [(f.qualname, f.primitive) for f in control] == [('d', 'open')]

    def test_asyncio_siblings_are_excluded_wholesale(self):
        """Neither ``asyncio.*`` nor ``anyio.*`` may be read as blocking."""
        for dotted in DOTTED_PRIMITIVES:
            assert not dotted.startswith(('asyncio.', 'anyio.')), (
                f'{dotted} is an async sibling and must never be in the table'
            )

    def test_every_table_entry_carries_a_justification(self):
        """A primitive with no stated reason is one a future census can drop unchallenged."""
        for table in (DOTTED_PRIMITIVES, METHOD_PRIMITIVES, BUILTIN_PRIMITIVES):
            for name, justification in table.items():
                assert justification.strip(), f'{name} has an empty justification'

    def test_table_covers_every_inv8_limb(self):
        """subprocess / network / filesystem / lock / sleep -- all five, not just the first.

        INV-8's Rule text names all five.  Task 3778's census enumerated one.
        """
        dotted = set(DOTTED_PRIMITIVES)
        assert {'subprocess.run', 'subprocess.check_output', 'subprocess.check_call',
                'subprocess.call', 'subprocess.Popen', 'os.system',
                'os.popen'} <= dotted
        assert {'socket.create_connection', 'urllib.request.urlopen'} <= dotted
        assert 'fcntl.flock' in dotted
        assert 'time.sleep' in dotted
        assert {'yaml.safe_load', 'yaml.safe_dump', 'json.load', 'json.dump',
                'os.listdir', 'os.walk', 'shutil.rmtree'} <= dotted
        assert {'read_text', 'write_text', 'read_bytes', 'write_bytes'} <= set(
            METHOD_PRIMITIVES
        )
        assert 'open' in BUILTIN_PRIMITIVES, (
            'the plainest filesystem spelling of all is a bare open(); a table '
            'that only knows dotted paths and methods cannot see it'
        )


# --------------------------------------------------------------------------- #
# The whole-tree gate
# --------------------------------------------------------------------------- #

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Task 4484's charter is "re-run the enumeration caller-side across
# fused-memory", so the finding set is scoped to that package. The shared
# first-party tree spans every scope root in silent_fallthrough_scan's
# _SCOPE_ROOTS; orchestrator/src is also heavily async and would balloon the
# baseline past anything a reviewer can read, turning the ratchet into a merge
# blocker for unrelated work. Widening is a deliberate follow-on decision with
# merge-lane consequences, not a side effect of this audit.
#
# The scope is applied by FILTERING the shared first-party tree, never by
# handing the provider a narrower root: it validates repo_root against
# sentinel dirs ('shared/src', 'orchestrator/src') and RAISES rather than
# yielding a vacuously empty scan. Passing 'fused-memory/src' as the root
# would trip that sentinel, and relaxing the sentinel to accommodate us would
# delete the loud-failure property its existing consumers rely on.
_SCOPE_PREFIX = 'fused-memory/src/'

# How many out-of-scope first-party files the scope-filter test may scan before
# giving up. It stops at the FIRST file that yields a finding (the 4th, at the
# task 4484 amendment pass); the cap only bounds the pathological case, so the
# gate cannot become slow because the other scope roots got clean.
_OUT_OF_SCOPE_PROBE_CAP = 80


class _TreeScan(NamedTuple):
    """Cached result of one whole-tree sweep (session-scoped)."""

    scanned_files: int
    findings: list[LoopBlockingSite]


def _build_tree_scan(records: Sequence[ParsedFile]) -> _TreeScan:
    """Scan the ``_SCOPE_PREFIX`` records of the shared first-party tree.

    Every in-scope module is in the mapping handed to the scanner, so
    cross-module resolution sees the whole package. The trees are walked, never
    re-read or re-parsed.
    """
    in_scope = [record for record in records if record.relpath.startswith(_SCOPE_PREFIX)]
    trees = {record.relpath: record.tree for record in in_scope if record.tree is not None}
    return _TreeScan(
        scanned_files=len(in_scope), findings=find_loop_blocking_sites_in_trees(trees)
    )


@pytest.fixture(scope='session')
def tree_scan(first_party_tree: tuple[ParsedFile, ...]) -> _TreeScan:
    """Scan ``fused-memory/src`` once per test session, off the shared tree."""
    return _build_tree_scan(first_party_tree)


class TestSweepUsesTheSharedTree:
    """The sweep walks the session's shared ASTs and does no I/O of its own."""

    def test_the_sweep_reads_nothing_and_parses_nothing(self, first_party_tree, monkeypatch):
        """Scoped by ``monkeypatch.context()``: pytest's own failure report calls
        ``ast.parse``, so the patch must be gone before a failure is rendered."""

        def no_parse(*_args, **_kwargs):
            raise AssertionError('ast.parse called: the sweep re-parsed a file')

        def no_read(*_args, **_kwargs):
            raise AssertionError('pathlib.Path.read_text called: the sweep re-read a file')

        with monkeypatch.context() as patched:
            patched.setattr(ast, 'parse', no_parse)
            patched.setattr(Path, 'read_text', no_read)
            scan = _build_tree_scan(first_party_tree)
        assert scan.findings, 'the patched sweep walked nothing, so it proved nothing about I/O'


class TestSweepIsNotVacuous:
    """A gate that scans nothing passes trivially and protects nothing."""

    def test_scanned_file_floor(self, tree_scan):
        """Floor deliberately BELOW the measured 167 files at HEAD 6696f1ce0c."""
        assert tree_scan.scanned_files >= 100, (
            f'only {tree_scan.scanned_files} files scanned under {_SCOPE_PREFIX} '
            f'(167 at HEAD 6696f1ce0c) -- the SWEEP is broken, not the tree. '
            f'Every assertion below would pass vacuously. Is _REPO_ROOT right? '
            f'({_REPO_ROOT})'
        )

    def test_finding_floor(self, tree_scan):
        """Floor deliberately BELOW the measured 60 findings at HEAD 6696f1ce0c.

        Slack is the point.  The in-flight owners of ``filed`` rows
        legitimately REMOVE findings when they land; a floor set at the
        measured value would turn red on success.  The floor exists only to
        prove the detector still detects on the live tree, not to pin a count.
        """
        assert len(tree_scan.findings) >= 10, (
            f'only {len(tree_scan.findings)} findings (60 at HEAD 6696f1ce0c) -- '
            f'the detector has almost certainly stopped detecting. Check the '
            f'synthetic fixtures above before believing the tree got clean.'
        )

    def test_the_prefix_filter_is_what_keeps_the_ledger_scoped(
        self, tree_scan, first_party_tree
    ):
        """Out-of-scope first-party files DO yield findings; the filter excludes them.

        Asserting that no finding in ``tree_scan`` lies outside
        ``_SCOPE_PREFIX`` would test nothing: ``_build_tree_scan`` only keeps a
        record whose relpath starts with that prefix, so the property holds by
        construction one function above the assertion.

        This scans out-of-scope files INDEPENDENTLY (bounded, one file at a
        time, stopping at the first that yields anything -- ~4 files and well
        under a second in practice) and then checks those keys are absent from
        the ledger sweep.  Delete the prefix filter from ``_build_tree_scan``
        and this goes red, which is what the earlier version could not do.
        """
        probe: list[LoopBlockingSite] = []
        scanned = 0
        for record in first_party_tree:
            if record.relpath.startswith(_SCOPE_PREFIX) or record.tree is None:
                continue
            scanned += 1
            probe.extend(find_loop_blocking_sites_in_trees({record.relpath: record.tree}))
            if probe or scanned >= _OUT_OF_SCOPE_PROBE_CAP:
                break

        assert probe, (
            f'no out-of-scope finding in the first {scanned} first-party files '
            f'outside {_SCOPE_PREFIX} -- either the other scope roots got '
            f'clean (measured: 123 findings across 294 files at the task 4484 '
            f'amendment pass) or the sweep is broken. Without one, this test '
            f'cannot show the filter is doing anything.'
        )

        swept = {site_key(f) for f in tree_scan.findings}
        strays = sorted(site_key(f) for f in probe if site_key(f) in swept)
        assert strays == [], (
            f'out-of-scope sites reached the ledger sweep: {strays}. Widening '
            f'the gate past {_SCOPE_PREFIX} is a deliberate decision with '
            f'merge-lane consequences (see the _SCOPE_PREFIX comment), not a '
            f'drift.'
        )


class TestScanHelperStaysInGateScope:
    """The extracted archive scan must stay where this sweep can still see it.

    Task 5550 moved the escalation-archive walk out of
    ``reconciliation/harness.py`` into
    ``fused_memory/reconciliation/escalation_archive.py``, and the ten
    ``_escalate`` rows in the ledger survive only because the scanner can
    follow ``_escalate -> _finding_recently_resolved -> ... -> read_text``
    across that module boundary.  It can do so ONLY because the new module
    sits under ``_SCOPE_PREFIX``: cross-module resolution sees nothing outside
    the sweep's ``sources``.

    Relocating the helper into the ``escalation`` package -- its otherwise
    natural home, right next to ``iter_all_escalation_paths`` -- would
    therefore take that reach out of view entirely.  Nothing would break
    loudly: the ten rows would go stale, ``test_no_stale_blessings`` would
    demand their deletion as fixed, and the ledger would quietly lose coverage
    of a defect (task 5270's) that nobody had fixed.  Moving blocking code out
    of a guard's field of view is precisely the failure this guard exists to
    catch, so the constraint is asserted rather than left in a comment.

    Asserted on ``(qualname, callee)`` pairs -- never a hash or a count -- so
    that task 5270 legitimately removing these sites cannot turn the gate red
    on success.
    """

    _HARNESS = 'fused-memory/src/fused_memory/reconciliation/harness.py'

    def _escalate_sites(self, tree_scan):
        return {
            (f.qualname, f.callee)
            for f in tree_scan.findings
            if f.filename == self._HARNESS and f.callee == '_escalate'
        }

    def test_escalate_still_reaches_a_blocking_primitive(self, tree_scan):
        """At least one coroutine must still reach a primitive via ``_escalate``."""
        found = self._escalate_sites(tree_scan)

        assert found, (
            'no coroutine in reconciliation/harness.py is reported as reaching a '
            'blocking primitive through _escalate. TWO VERY DIFFERENT CAUSES: if '
            'task 5270 landed and offloaded _escalate, DELETE this floor entry '
            'with it. If the reach vanished for any OTHER reason, the archive '
            'scan helper has been moved out of _SCOPE_PREFIX (fused-memory/src/) '
            '-- most likely into the escalation package next to '
            'iter_all_escalation_paths -- the scanner can no longer follow into '
            "it, and the ledger's ten _escalate rows are now lying about a "
            'defect that is still live.'
        )

    def test_surviving_escalate_sites_are_still_dispositioned_filed(self, tree_scan):
        """Whichever ``_escalate`` sites remain must still be blessed ``filed``.

        Same falsifiable property as the 4201 floor above: a site an in-flight
        task owns must keep a row saying so, rather than being re-blessed as
        ``accepted`` -- a permanent waiver for a defect somebody is fixing.
        Task 5550 memoised their archive walk per run but left every miss
        blocking, so they stay ``filed`` and stay 5270's.
        """
        filed_keys = {
            (relpath, qualname, content_hash)
            for relpath, qualname, content_hash, disposition, _why in AUDITED_SITES
            if disposition == 'filed'
        }

        surviving = [
            f for f in tree_scan.findings
            if f.filename == self._HARNESS and f.callee == '_escalate'
        ]
        undispositioned = sorted(
            (f.qualname, f.callee) for f in surviving if site_key(f) not in filed_keys
        )

        assert undispositioned == [], (
            f'task 5270 owns {undispositioned}, but the ledger no longer carries '
            f'a "filed" row for them. If 5270 landed, DELETE the rows (and this '
            f'floor entry); do not downgrade them to "accepted", which turns a '
            f'defect someone is fixing into a permanent waiver.'
        )


class TestRatchet:
    """Every live finding is dispositioned, and every disposition is live."""

    def test_no_unblessed_findings(self, tree_scan):
        unblessed, _stale = reconcile_against_allowlist(
            tree_scan.findings, ALLOWLIST_KEYS
        )

        if unblessed:
            offenders = '\n'.join(
                f'  {f.filename}::{f.qualname} -> {f.callee}  [{f.primitive}] '
                f'~L{f.lineno}  hash={f.content_hash}'
                for f in unblessed
            )
            raise AssertionError(
                'Coroutine call sites reaching a blocking primitive with no '
                'recorded disposition:\n' + offenders + '\n\n'
                'Fix it (wrap the call in asyncio.to_thread) OR add\n'
                '  (relpath, qualname, content_hash, disposition, justification)\n'
                'to loop_blocking_allowlist.AUDITED_SITES **in this same '
                'change**. A justification of "existing" is a silent waiver and '
                'defeats the gate -- say what makes the cost acceptable, or name '
                'the task that will fix it.'
            )

    def test_no_stale_blessings(self, tree_scan):
        """A landed fix must DELETE its blessing, so the ledger self-corrects.

        This half is what stops the baseline becoming a comfortable lie: when
        task 4201 landed and offloaded its two sites, their rows went stale and
        this test named them.
        """
        _unblessed, stale = reconcile_against_allowlist(
            tree_scan.findings, ALLOWLIST_KEYS
        )

        assert stale == [], (
            'blessed sites that no longer exist -- delete these rows from '
            f'loop_blocking_allowlist.AUDITED_SITES: {stale}\n\n'
            'If you just landed a fix for one of these, THE DELETION IS PART '
            'OF THAT FIX: shared/tests is the first segment of this repo\'s '
            'test_command, so a stale row reds verify for every subsequent '
            'task until the row goes. Delete, never re-bless -- a blessing for '
            'a site that no longer exists cannot describe anything. See '
            'plans/inv8-caller-side-census-2026-09-03.md section 7.'
        )

    def test_ratchet_is_a_multiset_not_a_set(self):
        """A SECOND site inside an already-blessed function is still unblessed.

        Set membership would let it pass silently -- which is a per-function
        restatement of exactly the per-module blindness that made task 3778's
        census miss the ``task_curator.py`` callers.  ``reconcile_against_allowlist``
        uses Counter subtraction for this reason; the assertion pins it here so
        nobody "simplifies" it to a set.
        """
        def _site(content_hash):
            return LoopBlockingSite(
                filename='pkg/mod.py',
                qualname='Cls.handler',
                callee='load_registry',
                primitive='read_text',
                lineno=1,
                content_hash=content_hash,
                message='',
            )

        blessed = [('pkg/mod.py', 'Cls.handler', 'aaaaaaaaaaaa')]

        one_unblessed, stale = reconcile_against_allowlist(
            [_site('aaaaaaaaaaaa'), _site('aaaaaaaaaaaa')], blessed
        )

        assert len(one_unblessed) == 1, (
            'a duplicate site in an already-blessed function must remain '
            'unblessed -- the ratchet has been reduced to set membership'
        )
        assert stale == []


class TestAllowlistHygiene:
    """A blessing with no reason is a silent waiver."""

    def test_disposition_vocabulary_is_fixed(self):
        assert frozenset({'accepted', 'filed', 'to_file'}) == DISPOSITIONS

    def test_every_entry_has_a_known_disposition(self):
        for relpath, qualname, _hash, disposition, _why in AUDITED_SITES:
            assert disposition in DISPOSITIONS, (
                f'{relpath}::{qualname}: unknown disposition {disposition!r}'
            )

    def test_every_entry_carries_a_justification(self):
        """"existing" is not a reason.  An accepted row must say what makes the
        cost acceptable (cached, startup-only, measured cheap); a filed row must
        name its task."""
        for relpath, qualname, _hash, _disposition, why in AUDITED_SITES:
            assert why.strip(), f'{relpath}::{qualname} has an empty justification'
            assert len(why.strip()) > 40, (
                f'{relpath}::{qualname}: justification is too short to be a '
                f'reason -- {why!r}'
            )

    def test_filed_entries_name_their_task(self):
        """A ``filed`` row without a task id points nowhere and closes nothing."""
        for relpath, qualname, _hash, disposition, why in AUDITED_SITES:
            if disposition != 'filed':
                continue
            assert re.search(r'\b\d{3,5}\b', why), (
                f'{relpath}::{qualname}: disposition "filed" but the '
                f'justification names no task id -- {why!r}'
            )

    def test_content_hashes_are_well_formed(self):
        for relpath, qualname, content_hash, _disposition, _why in AUDITED_SITES:
            assert len(content_hash) == 12 and all(
                ch in '0123456789abcdef' for ch in content_hash
            ), f'{relpath}::{qualname}: malformed content_hash {content_hash!r}'

    def test_every_blessed_file_exists(self):
        """A row for a deleted file is a stale record the ratchet cannot catch."""
        missing = sorted({
            relpath for relpath, _q, _h, _d, _w in AUDITED_SITES
            if not (_REPO_ROOT / relpath).is_file()
        })
        assert missing == [], f'blessed rows naming files that do not exist: {missing}'

    def test_allowlist_keys_match_the_entries(self):
        """``ALLOWLIST_KEYS`` is derived, never hand-maintained alongside the rows."""
        assert [
            (relpath, qualname, content_hash)
            for relpath, qualname, content_hash, _d, _w in AUDITED_SITES
        ] == ALLOWLIST_KEYS


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(pytest.main([__file__, '-q']))
