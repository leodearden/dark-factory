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

import re
import textwrap
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
    site_key,
)
from silent_fallthrough_scan import (
    iter_first_party_files,
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
        make_commit_probe`` -- is unaffected and still reported.
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
]


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
                'import fcntl\nimport os\nimport socket\nimport subprocess\nimport time\n\nimport yaml',
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
                'import fcntl\nimport os\nimport socket\nimport subprocess\nimport time\n\nimport yaml',
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
# fused-memory", so the finding set is scoped to that package.
# iter_first_party_files yields all SEVEN scope roots; orchestrator/src is also
# heavily async and would balloon the baseline past anything a reviewer can
# read, turning the ratchet into a merge blocker for unrelated work. Widening
# is a deliberate follow-on decision with merge-lane consequences, not a
# side effect of this audit.
#
# The scope is applied by FILTERING that generator's output, never by handing
# it a narrower root: it validates repo_root against sentinel dirs
# ('shared/src', 'orchestrator/src') and RAISES rather than yielding a
# vacuously empty scan. Passing 'fused-memory/src' as the root would trip that
# sentinel, and relaxing the sentinel to accommodate us would delete the
# loud-failure property its existing consumers rely on.
_SCOPE_PREFIX = 'fused-memory/src/'

# How many out-of-scope first-party files the scope-filter test may read before
# giving up. It stops at the FIRST file that yields a finding (the 4th, at the
# task 4484 amendment pass); the cap only bounds the pathological case, so the
# gate cannot become slow because the other six scope roots got clean.
_OUT_OF_SCOPE_PROBE_CAP = 80


class _TreeScan(NamedTuple):
    """Cached result of one whole-tree sweep (session-scoped)."""

    scanned_files: int
    findings: list[LoopBlockingSite]


@pytest.fixture(scope='session')
def tree_scan() -> _TreeScan:
    """Enumerate, read and scan ``fused-memory/src`` once per test session."""
    sources: dict[str, str] = {}
    for path in iter_first_party_files(_REPO_ROOT):
        rel = path.relative_to(_REPO_ROOT).as_posix()
        if not rel.startswith(_SCOPE_PREFIX):
            continue
        sources[rel] = path.read_text(encoding='utf-8', errors='replace')
    return _TreeScan(scanned_files=len(sources), findings=find_loop_blocking_sites(sources))


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

    def test_the_prefix_filter_is_what_keeps_the_ledger_scoped(self, tree_scan):
        """Out-of-scope first-party files DO yield findings; the filter excludes them.

        Asserting that no finding in ``tree_scan`` lies outside
        ``_SCOPE_PREFIX`` would test nothing: the fixture only inserts a path
        into ``sources`` when it starts with that prefix, so the property holds
        by construction one function above the assertion.

        This scans out-of-scope files INDEPENDENTLY (bounded, one file at a
        time, stopping at the first that yields anything -- ~4 files and well
        under a second in practice) and then checks those keys are absent from
        the ledger sweep.  Delete the prefix filter from the fixture and this
        goes red, which is what the earlier version could not do.
        """
        probe: list[LoopBlockingSite] = []
        scanned = 0
        for path in iter_first_party_files(_REPO_ROOT):
            rel = path.relative_to(_REPO_ROOT).as_posix()
            if rel.startswith(_SCOPE_PREFIX):
                continue
            scanned += 1
            probe.extend(find_loop_blocking_sites(
                {rel: path.read_text(encoding='utf-8', errors='replace')}
            ))
            if probe or scanned >= _OUT_OF_SCOPE_PROBE_CAP:
                break

        assert probe, (
            f'no out-of-scope finding in the first {scanned} first-party files '
            f'outside {_SCOPE_PREFIX} -- either the other six scope roots got '
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


class TestKnownSiteFloor:
    """The four ``task_curator.py`` sites task 4484 confirmed by hand."""

    # (qualname, callee) pairs, all in
    # fused-memory/src/fused_memory/middleware/task_curator.py.
    _CURATOR = 'fused-memory/src/fused_memory/middleware/task_curator.py'

    # No in-flight task removes these two, so they are asserted unconditionally.
    # Both are `async def _maybe_*` guards lazily loading a YAML registry off
    # disk on the loop thread -- the same shape as their two siblings below,
    # and NEITHER had a task filed before task 4484 found them.
    _UNFILED = {
        ('TaskCurator._maybe_blocklist_drop', 'load_blocklist'),
        ('TaskCurator._maybe_route_deterministic', 'load_operational_registry'),
    }

    # Task 4201 owns these two; when it lands they disappear, which must NOT
    # turn this gate red.  Hence a subset check, never an equality.
    _FILED_4201 = {
        ('TaskCurator._maybe_premise_refuted_drop', 'load_premise_registry'),
        ('TaskCurator._maybe_premise_refuted_drop', 'premise_refuted_entry'),
    }

    def _curator_sites(self, tree_scan):
        return {
            (f.qualname, f.callee)
            for f in tree_scan.findings
            if f.filename == self._CURATOR
        }

    def test_unfiled_curator_sites_are_found(self, tree_scan):
        """The two previously-UNFILED sites must be found by the live sweep.

        These are the ones that make task 4484's case: task 3778's census read
        ``task_curator.py``'s neighbourhood as covered, task 4091 fixed one
        sibling and task 4201 filed two more, and these two still had no task
        at all.  A caller-side sweep names all four at once.
        """
        found = self._curator_sites(tree_scan)

        assert found >= self._UNFILED, (
            f'missing {sorted(self._UNFILED - found)} from {sorted(found)} -- '
            f'the caller-side property regressed, or the sites were fixed '
            f'without updating this floor.'
        )

    def test_surviving_4201_sites_are_still_dispositioned_filed(self, tree_scan):
        """Whichever of 4201's sites remain must still be blessed as ``filed``.

        A subset check is right -- when 4201 lands its sites disappear and this
        gate must not go red -- but ``remaining <= found`` on an intersection is
        a tautology and asserts nothing.  The falsifiable property is the one
        that actually matters while 4201 is in flight: a site 4201 owns must
        keep a ledger row saying so, rather than being quietly re-blessed as
        ``accepted`` (a permanent waiver for a defect somebody is fixing) or
        losing its row.
        """
        filed_keys = {
            (relpath, qualname, content_hash)
            for relpath, qualname, content_hash, disposition, _why in AUDITED_SITES
            if disposition == 'filed'
        }

        surviving = [
            f for f in tree_scan.findings
            if f.filename == self._CURATOR and (f.qualname, f.callee) in self._FILED_4201
        ]
        undispositioned = sorted(
            (f.qualname, f.callee) for f in surviving if site_key(f) not in filed_keys
        )

        assert undispositioned == [], (
            f'task 4201 owns {undispositioned}, but the ledger no longer carries '
            f'a "filed" row for them. If 4201 landed, DELETE the rows (and this '
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
