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

import textwrap

import pytest
from loop_blocking_scan import (
    DOTTED_PRIMITIVES,
    METHOD_PRIMITIVES,
    find_loop_blocking_sites,
    site_key,
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

    def test_yaml_safe_load_justification_cites_task_4201_measurement(self):
        """The table must carry WHY yaml parsing counts, with the measurement.

        Task 4201 measured ``yaml.safe_load`` at 8.15 ms for an 11 KB document
        -- the same order as a subprocess spawn.  Without that number recorded
        at the table, the next census re-reads it as "just parsing" and drops
        it back out, which is precisely how gap B happened the first time.
        """
        justification = DOTTED_PRIMITIVES['yaml.safe_load']

        assert '4201' in justification, justification
        assert '8.15' in justification, justification
        assert '11' in justification, justification

    def test_filesystem_justifications_cite_their_discovering_tasks(self):
        """Filesystem primitives carry the tasks that found them (4091 / 4201).

        These are the entries task 3778's ``subprocess.run``-only vocabulary
        omitted, so their justification is the record of why the vocabulary is
        wider now.
        """
        for name in ('read_text', 'write_text', 'read_bytes', 'write_bytes'):
            justification = METHOD_PRIMITIVES[name]
            assert '4091' in justification or '4201' in justification, (
                f'{name}: {justification}'
            )

    def test_every_table_entry_carries_a_justification(self):
        """A primitive with no stated reason is one a future census can drop unchallenged."""
        for table in (DOTTED_PRIMITIVES, METHOD_PRIMITIVES):
            for name, justification in table.items():
                assert justification.strip(), f'{name} has an empty justification'

    def test_table_covers_every_inv8_limb(self):
        """subprocess / network / filesystem / lock / sleep -- all five, not just the first.

        INV-8's Rule text names all five.  Task 3778's census enumerated one.
        """
        dotted = set(DOTTED_PRIMITIVES)
        assert {'subprocess.run', 'subprocess.check_output', 'subprocess.check_call',
                'subprocess.Popen', 'os.system'} <= dotted
        assert 'socket.create_connection' in dotted
        assert 'fcntl.flock' in dotted
        assert 'time.sleep' in dotted
        assert {'yaml.safe_load', 'yaml.safe_dump'} <= dotted
        assert {'read_text', 'write_text', 'read_bytes', 'write_bytes'} <= set(
            METHOD_PRIMITIVES
        )


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(pytest.main([__file__, '-q']))
