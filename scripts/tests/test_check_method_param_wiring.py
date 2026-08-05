"""Tests for scripts/check_method_param_wiring.py (task 3364).

Why the script exists at all — and why each conjunct is load-bearing — is
argued once, in that module's own docstring; this file does not restate it.

Imports the module under test bare (``import check_method_param_wiring``);
``scripts/tests/conftest.py`` already puts ``scripts/`` on ``sys.path``. No
first-party package is imported here — that is load-bearing for the fallback
verify chain (see the conftest docstring).
"""
import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

import check_method_param_wiring as cmpw

# scripts/tests -> scripts -> repo root. Same idiom as
# shared/tests/capability_manifest_corpus.py; correct inside a worktree too.
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / 'scripts' / 'check_method_param_wiring.py'
MEM0_CLIENT = 'fused-memory/src/fused_memory/backends/mem0_client.py'

# Mirrors mem0_client.py's real shape: a class with two async methods, each
# signature spread one parameter per line — the exact layout that defeats a
# single-line grep. `scroll_by_metadata` declares keyword-only
# `with_vectors: bool = False`; `get_point_by_id` declares no such parameter
# and its only `with_vectors=False` is a hardcoded literal on a different call.
REAL_SHAPE = '''
import asyncio


class Mem0Client:
    async def scroll_by_metadata(
        self,
        scope: Scope,
        filters: dict[str, Any],
        limit: int = 1000,
        *,
        with_vectors: bool = False,
    ) -> list[dict[str, Any]]:
        """Deterministic enumeration."""
        client = await self._get_async_qdrant()
        points, _next_offset = await asyncio.wait_for(
            client.scroll(
                collection_name=collection_name,
                scroll_filter=qdrant_filter,
                with_payload=True,
                with_vectors=with_vectors,
                limit=limit,
            ),
            timeout=self._read_timeout,
        )
        return points

    async def get_point_by_id(self, memory_id: str, scope: Scope) -> dict | None:
        """Direct point-fetch by id."""
        client = await self._get_async_qdrant()
        records = await asyncio.wait_for(
            client.retrieve(
                collection_name=collection_name,
                ids=[memory_id],
                with_payload=True,
                with_vectors=False,
            ),
            timeout=self._read_timeout,
        )
        return records[0] if records else None
'''


@pytest.fixture
def real_tree() -> ast.Module:
    return cmpw.parse_module(REAL_SHAPE)


def _resolved(tree: ast.Module, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    """Resolve *name*, failing loudly when the fixture does not declare it.

    Also narrows away `resolve_function`'s `| None` for pyright, which
    `scripts/orchestrator.yaml` runs over this directory.
    """
    fn = cmpw.resolve_function(tree, name)
    assert fn is not None, f'fixture module declares no function {name!r}'
    return fn


class TestResolveFunction:
    """`resolve_function` walks the whole module, not just its top level."""

    def test_finds_async_method_nested_in_class(self, real_tree):
        fn = cmpw.resolve_function(real_tree, 'scroll_by_metadata')
        assert isinstance(fn, ast.AsyncFunctionDef)
        assert fn.name == 'scroll_by_metadata'

    def test_finds_plain_def_at_module_level(self):
        tree = cmpw.parse_module('def helper(a, b):\n    return a + b\n')
        fn = cmpw.resolve_function(tree, 'helper')
        assert isinstance(fn, ast.FunctionDef)
        assert fn.name == 'helper'

    def test_absent_function_returns_none(self, real_tree):
        assert cmpw.resolve_function(real_tree, 'no_such_name') is None

    def test_ambiguity_is_explicit_not_silent_first_match(self):
        """Two same-named defs must NOT silently resolve to one of them.

        A check that picks arbitrarily between two candidates is not
        method-scoped; it is grep with extra steps.
        """
        tree = cmpw.parse_module(
            'class A:\n'
            '    def dup(self, with_vectors: bool = False):\n'
            '        pass\n'
            '\n'
            '\n'
            'class B:\n'
            '    def dup(self, other: int = 0):\n'
            '        pass\n'
        )
        with pytest.raises(cmpw.AmbiguousFunction) as excinfo:
            cmpw.resolve_function(tree, 'dup')
        assert 'dup' in str(excinfo.value)


class TestDeclaresParam:
    """`declares_param` inspects the resolved function's own signature."""

    def test_true_for_keyword_only_declaration(self, real_tree):
        fn = _resolved(real_tree, 'scroll_by_metadata')
        assert cmpw.declares_param(fn, 'with_vectors') is True

    def test_false_for_the_method_that_does_not_declare_it(self, real_tree):
        """`get_point_by_id` has no `with_vectors` parameter.

        Its `with_vectors=False` is a literal keyword on `client.retrieve`.
        The superseded whole-file grep could not tell these two apart.
        """
        fn = _resolved(real_tree, 'get_point_by_id')
        assert cmpw.declares_param(fn, 'with_vectors') is False

    def test_finds_positional_or_keyword_param(self):
        """Must not silently depend on the `*` in today's signature."""
        tree = cmpw.parse_module('def f(self, with_vectors: bool = False):\n    pass\n')
        fn = _resolved(tree, 'f')
        assert cmpw.declares_param(fn, 'with_vectors') is True

    def test_finds_positional_only_param(self):
        tree = cmpw.parse_module('def f(with_vectors: bool = False, /):\n    pass\n')
        fn = _resolved(tree, 'f')
        assert cmpw.declares_param(fn, 'with_vectors') is True


class TestDeclaresParamAnnotation:
    """Annotation matching preserves what the grep pattern asserted."""

    def test_matching_annotation_is_true(self, real_tree):
        fn = _resolved(real_tree, 'scroll_by_metadata')
        assert cmpw.declares_param(fn, 'with_vectors', annotation='bool') is True

    def test_mismatched_annotation_is_false(self, real_tree):
        fn = _resolved(real_tree, 'scroll_by_metadata')
        assert cmpw.declares_param(fn, 'with_vectors', annotation='int') is False

    def test_unannotated_param_fails_a_required_annotation(self):
        tree = cmpw.parse_module('def f(self, with_vectors=False):\n    pass\n')
        fn = _resolved(tree, 'f')
        assert cmpw.declares_param(fn, 'with_vectors', annotation='bool') is False

    def test_unannotated_param_passes_when_no_annotation_required(self):
        tree = cmpw.parse_module('def f(self, with_vectors=False):\n    pass\n')
        fn = _resolved(tree, 'f')
        assert cmpw.declares_param(fn, 'with_vectors', annotation=None) is True


class TestForwardsParamTo:
    """Declaration alone is NOT the capability (see the script's docstring)."""

    def test_true_for_the_real_forward(self, real_tree):
        """`client.scroll(..., with_vectors=with_vectors, ...)` on main."""
        fn = _resolved(real_tree, 'scroll_by_metadata')
        assert cmpw.forwards_param_to(fn, 'with_vectors', 'scroll') is True

    def test_false_for_a_hardcoded_literal(self, real_tree):
        """`client.retrieve(..., with_vectors=False)` is NOT a forward.

        This is exactly `get_point_by_id`'s shape on main (mem0_client.py:839)
        and is the discrimination the whole task turns on: an `ast.Constant`
        value means the parameter is not reaching the call.
        """
        fn = _resolved(real_tree, 'get_point_by_id')
        assert cmpw.forwards_param_to(fn, 'with_vectors', 'retrieve') is False

    def test_false_when_keyword_forwards_a_different_name(self):
        tree = cmpw.parse_module(
            'def f(with_vectors: bool = False):\n'
            '    return client.scroll(with_vectors=want_vectors)\n'
        )
        fn = _resolved(tree, 'f')
        assert cmpw.forwards_param_to(fn, 'with_vectors', 'scroll') is False

    def test_false_when_the_right_forward_goes_to_the_wrong_callee(self):
        tree = cmpw.parse_module(
            'def f(with_vectors: bool = False):\n'
            '    return client.query_points(with_vectors=with_vectors)\n'
        )
        fn = _resolved(tree, 'f')
        assert cmpw.forwards_param_to(fn, 'with_vectors', 'scroll') is False

    def test_matches_on_attribute_name_regardless_of_receiver(self):
        """`other.scroll(...)` still matches `callee='scroll'`.

        The check binds to the method NAME, not to the receiver expression —
        resolving the receiver would need type inference the AST cannot give.
        """
        tree = cmpw.parse_module(
            'def f(with_vectors: bool = False):\n'
            '    return other.scroll(with_vectors=with_vectors)\n'
        )
        fn = _resolved(tree, 'f')
        assert cmpw.forwards_param_to(fn, 'with_vectors', 'scroll') is True

    def test_true_for_a_bare_call_callee(self):
        """A plain `ast.Name` func, not just an `ast.Attribute`."""
        tree = cmpw.parse_module(
            'def f(with_vectors: bool = False):\n'
            '    return scroll(with_vectors=with_vectors)\n'
        )
        fn = _resolved(tree, 'f')
        assert cmpw.forwards_param_to(fn, 'with_vectors', 'scroll') is True

    def test_true_when_the_call_is_nested_in_another_expression(self):
        """On main the real call sits inside `await asyncio.wait_for(...)`.

        A body scan inspecting only top-level statements would miss it. The
        `real_tree` fixture carries that exact nesting; this pins the property
        in isolation.
        """
        tree = cmpw.parse_module(
            'async def f(with_vectors: bool = False):\n'
            '    points, _ = await asyncio.wait_for(\n'
            '        client.scroll(with_vectors=with_vectors),\n'
            '        timeout=30,\n'
            '    )\n'
            '    return points\n'
        )
        fn = _resolved(tree, 'f')
        assert cmpw.forwards_param_to(fn, 'with_vectors', 'scroll') is True

    def test_a_forward_in_a_different_function_does_not_count(self):
        """Walk `fn`, never the module — this is what keeps it method-scoped."""
        tree = cmpw.parse_module(
            'def target(with_vectors: bool = False):\n'
            '    return client.scroll(with_payload=True)\n'
            '\n'
            '\n'
            'def neighbour(with_vectors: bool = False):\n'
            '    return client.scroll(with_vectors=with_vectors)\n'
        )
        fn = _resolved(tree, 'target')
        assert cmpw.forwards_param_to(fn, 'with_vectors', 'scroll') is False

    def test_a_forward_from_a_nested_def_that_shadows_the_param_does_not_count(self):
        """Same scoping rule, one shadowing level IN rather than one out.

        `inner`'s `with_vectors` is a fresh binding, so its forward says
        nothing about the outer parameter — which this body in fact drops.
        Plain `ast.walk` descends into nested `def`s and would call this
        DELIVERED.
        """
        tree = cmpw.parse_module(
            'def target(with_vectors: bool = False):\n'
            '    def inner(with_vectors):\n'
            '        return client.scroll(with_vectors=with_vectors)\n'
            '    return client.scroll(with_payload=True)\n'
        )
        fn = _resolved(tree, 'target')
        assert cmpw.forwards_param_to(fn, 'with_vectors', 'scroll') is False

    def test_a_forward_from_a_shadowing_lambda_does_not_count(self):
        tree = cmpw.parse_module(
            'def target(with_vectors: bool = False):\n'
            '    fetch = lambda with_vectors: client.scroll(with_vectors=with_vectors)\n'
            '    return fetch(True)\n'
        )
        fn = _resolved(tree, 'target')
        assert cmpw.forwards_param_to(fn, 'with_vectors', 'scroll') is False

    def test_a_nested_scope_that_closes_over_the_param_still_counts(self):
        """Only a REBINDING nested scope is pruned.

        `inner` declares no `with_vectors`, so the name it forwards IS the
        outer parameter — a real forward, just deferred.
        """
        tree = cmpw.parse_module(
            'def target(with_vectors: bool = False):\n'
            '    def inner():\n'
            '        return client.scroll(with_vectors=with_vectors)\n'
            '    return inner()\n'
        )
        fn = _resolved(tree, 'target')
        assert cmpw.forwards_param_to(fn, 'with_vectors', 'scroll') is True


# The argv the manifest declares, minus --file (each test supplies its own).
_ARGV_TAIL = [
    '--function', 'scroll_by_metadata',
    '--param', 'with_vectors',
    '--annotation', 'bool',
    '--forwards-to', 'scroll',
]


class TestMainExitCodeContract:
    """rc==0 -> DELIVERED, ANY non-zero -> FAILED; no rc means "cannot tell".

    So every non-zero return must be one the script understands and must name
    which conjunct failed — argued in full in the script's docstring.
    """

    def _write(self, tmp_path, source: str, name: str = 'mod.py'):
        target = tmp_path / name
        target.write_text(source)
        return target

    def test_real_shape_returns_zero_and_is_silent(self, tmp_path, capsys):
        target = self._write(tmp_path, REAL_SHAPE)
        rc = cmpw.main(['--file', str(target), *_ARGV_TAIL])
        assert rc == 0
        assert capsys.readouterr().err == ''

    def test_unknown_function_is_distinguishable(self, tmp_path, capsys):
        target = self._write(tmp_path, REAL_SHAPE)
        rc = cmpw.main(['--file', str(target), '--function', 'nope',
                        '--param', 'with_vectors'])
        assert rc != 0
        err = capsys.readouterr().err
        assert 'not found' in err
        assert 'nope' in err and str(target) in err

    def test_ambiguous_function_is_distinguishable(self, tmp_path, capsys):
        target = self._write(
            tmp_path,
            'class A:\n'
            '    def dup(self, with_vectors: bool = False):\n'
            '        pass\n'
            '\n'
            '\n'
            'class B:\n'
            '    def dup(self, with_vectors: bool = False):\n'
            '        pass\n',
        )
        rc = cmpw.main(['--file', str(target), '--function', 'dup',
                        '--param', 'with_vectors'])
        assert rc != 0
        assert 'ambiguous' in capsys.readouterr().err

    def test_param_absent_is_distinguishable(self, tmp_path, capsys):
        target = self._write(tmp_path, REAL_SHAPE)
        rc = cmpw.main(['--file', str(target), '--function', 'get_point_by_id',
                        '--param', 'with_vectors'])
        assert rc != 0
        err = capsys.readouterr().err
        assert 'does not declare' in err
        assert 'get_point_by_id' in err and 'with_vectors' in err

    def test_annotation_mismatch_is_distinguishable(self, tmp_path, capsys):
        target = self._write(tmp_path, REAL_SHAPE)
        rc = cmpw.main(['--file', str(target), '--function', 'scroll_by_metadata',
                        '--param', 'with_vectors', '--annotation', 'int'])
        assert rc != 0
        err = capsys.readouterr().err
        assert 'annotation' in err
        assert 'int' in err

    def test_not_forwarded_is_distinguishable(self, tmp_path, capsys):
        """Declared but dropped — the hollow-DELIVERED case."""
        target = self._write(
            tmp_path,
            'def scroll_by_metadata(self, *, with_vectors: bool = False):\n'
            '    return client.scroll(with_payload=True)\n',
        )
        rc = cmpw.main(['--file', str(target), *_ARGV_TAIL])
        assert rc != 0
        err = capsys.readouterr().err
        assert 'does not forward' in err
        assert 'scroll' in err

    def test_missing_file_is_distinguishable(self, tmp_path, capsys):
        rc = cmpw.main(['--file', str(tmp_path / 'absent.py'), *_ARGV_TAIL])
        assert rc != 0
        err = capsys.readouterr().err
        assert 'file not found' in err
        assert 'absent.py' in err

    def test_unparseable_file_is_distinguishable(self, tmp_path, capsys):
        target = self._write(tmp_path, 'def broken(:\n    pass\n')
        rc = cmpw.main(['--file', str(target), *_ARGV_TAIL])
        assert rc != 0
        assert 'could not parse' in capsys.readouterr().err

    def test_undecodable_file_is_distinguishable(self, tmp_path, capsys):
        """Decoding is pinned to UTF-8, not to the ambient locale.

        `_run_script_check` execs this via `git_ops._run`, which forces
        LC_ALL=C/LANG=C in the child. A decode failure must still be a
        self-describing diagnostic, never a bare traceback.
        """
        target = tmp_path / 'latin1.py'
        target.write_bytes(b'# caf\xe9\ndef f(with_vectors: bool = False):\n    pass\n')
        rc = cmpw.main(['--file', str(target), '--function', 'f',
                        '--param', 'with_vectors'])
        assert rc != 0
        err = capsys.readouterr().err
        assert 'not valid UTF-8' in err
        assert 'check_method_param_wiring:' in err

    def test_non_utf8_source_is_decoded_under_a_c_locale(self, tmp_path):
        """The positive half: real non-ASCII content still parses.

        Run as a subprocess under a scrubbed C locale with PEP 540's UTF-8 mode
        explicitly DISABLED, so the assertion rests on this script's own
        `encoding='utf-8'` rather than on an interpreter default.
        """
        target = tmp_path / 'unicode_mod.py'
        target.write_text(
            '"""Docstring with a — em dash and a ✓ check."""\n'
            'def f(with_vectors: bool = False):\n'
            '    return client.scroll(with_vectors=with_vectors)\n',
            encoding='utf-8',
        )
        result = subprocess.run(
            [sys.executable, str(SCRIPT), '--file', str(target), '--function', 'f',
             '--param', 'with_vectors', '--annotation', 'bool', '--forwards-to', 'scroll'],
            capture_output=True,
            text=True,
            env={'PATH': os.environ.get('PATH', ''), 'LC_ALL': 'C', 'LANG': 'C',
                 'PYTHONUTF8': '0', 'PYTHONCOERCECLOCALE': '0'},
        )
        assert result.returncode == 0, result.stderr

    def test_unknown_flag_is_not_reported_as_a_regression(self, tmp_path, capsys):
        """A drifted manifest `args` list must not read as "capability gone".

        argparse would exit 2 on its own, which `_run_script_check` maps to
        FAILED — a definitive-absence claim about the capability.
        """
        target = self._write(tmp_path, REAL_SHAPE)
        rc = cmpw.main(['--file', str(target), *_ARGV_TAIL, '--no-such-flag', 'x'])
        assert rc != 0
        err = capsys.readouterr().err
        assert 'check_method_param_wiring: check arguments are invalid' in err

    def test_help_does_not_report_delivered_without_running_a_check(self, capsys):
        """`--help` exits 0 in argparse; 0 here would mean DELIVERED."""
        rc = cmpw.main(['--help'])
        assert rc != 0
        assert 'check arguments are invalid' in capsys.readouterr().err

    def test_every_diagnostic_is_distinct(self, tmp_path, capsys):
        """No two failure modes may print the same message.

        An operator triaging a DEP_CAPABILITY_NOT_DELIVERED escalation reads
        only this line.
        """
        good = self._write(tmp_path, REAL_SHAPE, 'good.py')
        dropped = self._write(
            tmp_path,
            'def scroll_by_metadata(self, *, with_vectors: bool = False):\n'
            '    return client.scroll(with_payload=True)\n',
            'dropped.py',
        )
        broken = self._write(tmp_path, 'def broken(:\n', 'broken.py')
        dup = self._write(
            tmp_path,
            'def dup(with_vectors: bool = False):\n    pass\n'
            '\n'
            '\n'
            'def dup(with_vectors: bool = False):\n    pass\n',
            'dup.py',
        )
        latin1 = tmp_path / 'latin1.py'
        latin1.write_bytes(b'# caf\xe9\ndef f(with_vectors: bool = False):\n    pass\n')
        cases = [
            ['--file', str(good), '--function', 'nope', '--param', 'with_vectors'],
            ['--file', str(dup), '--function', 'dup', '--param', 'with_vectors'],
            ['--file', str(good), '--function', 'get_point_by_id', '--param', 'with_vectors'],
            ['--file', str(good), '--function', 'scroll_by_metadata',
             '--param', 'with_vectors', '--annotation', 'int'],
            ['--file', str(dropped), *_ARGV_TAIL],
            ['--file', str(tmp_path / 'absent.py'), *_ARGV_TAIL],
            ['--file', str(broken), *_ARGV_TAIL],
            ['--file', str(latin1), '--function', 'f', '--param', 'with_vectors'],
            ['--file', str(good), *_ARGV_TAIL, '--no-such-flag'],
        ]
        messages = []
        for argv in cases:
            assert cmpw.main(argv) != 0
            messages.append(capsys.readouterr().err.strip())
        assert all(messages), 'every non-zero return must explain itself'
        assert len(set(messages)) == len(messages), messages


class TestMainOptionalConjuncts:
    def test_annotation_and_forwards_to_are_optional(self, tmp_path, capsys):
        """Omitting both, declaration alone suffices."""
        target = tmp_path / 'mod.py'
        target.write_text('def f(self, with_vectors=False):\n    pass\n')
        rc = cmpw.main(['--file', str(target), '--function', 'f',
                        '--param', 'with_vectors'])
        assert rc == 0
        assert capsys.readouterr().err == ''

    def test_file_is_resolved_relative_to_cwd(self, tmp_path, monkeypatch, capsys):
        """`_run_script_check` runs with cwd=project_root and a relative path."""
        pkg = tmp_path / 'nested'
        pkg.mkdir()
        (pkg / 'mod.py').write_text(REAL_SHAPE)
        monkeypatch.chdir(tmp_path)
        rc = cmpw.main(['--file', 'nested/mod.py', *_ARGV_TAIL])
        assert rc == 0
        assert capsys.readouterr().err == ''


MANIFEST = REPO_ROOT / 'docs' / 'prds' / 'memory-eval-program.capability-manifest.yaml'
CAPABILITY = 'qdrant-vector-access-for-ann'


@pytest.fixture
def delivered_check() -> dict:
    """The `qdrant-vector-access-for-ann` delivered_check, from the sidecar.

    Plain `yaml.safe_load`, NOT `shared.capability_manifest` — `conftest.py`
    documents "scripts/tests/ imports no first-party package" as load-bearing
    for the fallback verify chain, which folds this directory into a single
    `uv run --project shared pytest` invocation on the strength of it.
    """
    import yaml

    doc = yaml.safe_load(MANIFEST.read_text())
    for task in doc.get('tasks', []):
        for capability in task.get('capabilities', []):
            if capability.get('name') == CAPABILITY:
                return capability['delivered_check']
    pytest.fail(f'{CAPABILITY} not found in {MANIFEST}')


class TestEndToEndAgainstTheRealRepo:
    """Exec the script the way `_run_script_check` actually does.

    It builds ``argv = [str(project_root / meta.script), *meta.args]`` and execs
    it directly, so this is the only test that covers the shebang and the
    executable bit as well as the assertion. The argv comes from the SIDECAR,
    not from a second hardcoded copy — a manifest that drifts must fail here
    rather than pass against an argv nothing runs.
    """

    def _run(self, script: str, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [str(REPO_ROOT / script), *args], cwd=REPO_ROOT, capture_output=True, text=True
        )

    def test_manifest_argv_passes_on_the_real_module(self, delivered_check):
        """The positive case, run with exactly what the manifest declares."""
        result = self._run(delivered_check['script'], *delivered_check['args'])
        assert result.returncode == 0, result.stderr

    def test_the_mutation_the_whole_file_grep_would_have_missed_fails(self, delivered_check):
        """`get_point_by_id` declares no `with_vectors` parameter.

        Its `with_vectors=False` is a literal on `client.retrieve`. The
        superseded whole-file grep `'with_vectors: bool'` would have reported
        DELIVERED had the parameter landed here instead of on
        `scroll_by_metadata`; this is the regression proof that the replacement
        is method-scoped.

        Built by mutating the manifest's OWN argv, so it cannot drift into
        testing a shape the real check no longer has.
        """
        args = list(delivered_check['args'])
        args[args.index('--function') + 1] = 'get_point_by_id'
        args[args.index('--forwards-to') + 1] = 'retrieve'
        result = self._run(delivered_check['script'], *args)
        assert result.returncode != 0
        assert 'check_method_param_wiring:' in result.stderr


class TestManifestScriptCoherence:
    """A manifest naming a script that cannot be exec'd is ERRORED forever.

    Complements rather than duplicates
    `shared/tests/test_capability_manifest.py::TestCheckedInManifestCorpus`,
    whose docstring states it deliberately "Does NOT assert delivered_check
    `script:` targets exist on disk". That exclusion is exactly the silent
    un-gating this task exists to remove, so it is the one thing worth pinning.

    NOTE (task 3364 review): the exists+executable property is generic to EVERY
    `kind: script` check in every checked-in sidecar, and pinning it per
    capability leaves the others uncovered — the repo's only other one,
    `plans/agent-transcript-archival-prd.capability-manifest.yaml` ->
    `scripts/gc_agent_transcripts.py`, is committed at mode 100644 and would
    ERROR forever today. Generalizing this into the corpus sweep needs
    `shared/tests/`, which is outside this task's locks; filed as a follow-up.
    """

    def test_check_is_the_script_kind(self, delivered_check):
        assert delivered_check['kind'] == 'script'

    def test_declared_script_exists_and_is_executable(self, delivered_check):
        target = REPO_ROOT / delivered_check['script']
        assert target.is_file(), f'{target} does not exist'
        assert os.access(target, os.X_OK), (
            f'{target} is not executable; _run_script_check execs it as argv[0], '
            'so a non-executable target ERRORs forever'
        )

    def test_timeout_secs_is_set_and_positive(self, delivered_check):
        """The schema requires it for kind: script."""
        assert delivered_check.get('timeout_secs') is not None
        assert delivered_check['timeout_secs'] > 0

    def test_grep_only_fields_are_absent(self, delivered_check):
        """`_check_kind_conditional_fields` FORBIDS these for kind: script."""
        assert not delivered_check.get('pattern')
        assert not delivered_check.get('expect')
        assert not delivered_check.get('paths')

    def test_args_name_what_makes_the_check_method_scoped(self, delivered_check):
        args = delivered_check['args']
        assert MEM0_CLIENT in args
        assert 'scroll_by_metadata' in args
        assert 'with_vectors' in args

    def test_manifest_argv_is_exactly_the_argv_this_file_pins(self, delivered_check):
        """Sidecar and tests must not drift apart silently.

        The full list, not a subset: dropping `--forwards-to scroll` would
        reduce the gate to a declaration-only check — the hollow-DELIVERED mode
        the script exists to close — while every partial assertion stayed
        green. `_ARGV_TAIL` is what `TestMainExitCodeContract` exercises and
        `TestEndToEndAgainstTheRealRepo` execs, so equality here is what makes
        those two tests statements about the REAL check.
        """
        assert delivered_check['script'] == 'scripts/check_method_param_wiring.py'
        assert delivered_check['args'] == ['--file', MEM0_CLIENT, *_ARGV_TAIL]
