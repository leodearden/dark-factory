"""Tests for orchestrator.verify_cmd — structured VerifyCmd command model.

Task β of the verify-plan PRD (plans/verify-plan-prd.md §Contract·VerifyCmd).
Replaces verify.py's raw-shell-string find/replace-surgery command model with
``parse_config_command`` (raw string -> VerifyCmd), ``render`` (VerifyCmd ->
raw string), and a set of pure VerifyCmd -> VerifyCmd mutators.

No source stub exists yet — every test in this module is RED until
orchestrator/src/orchestrator/verify_cmd.py is created (step-2).
"""

from __future__ import annotations

import dataclasses
import shlex

import pytest

from orchestrator.verify_cmd import (
    ToolKind,
    VerifyCmd,
    cargo_scope,
    govern_cpu,
    parse_config_command,
    render,
    reproject,
    scope_to,
    serial_pytest,
    strip_cwd,
)


class TestToolKind:
    """ToolKind is a StrEnum whose members are the JSON-serialisable tool identities."""

    def test_members_present(self):
        names = {member.name for member in ToolKind}
        assert names == {
            'PYTEST', 'RUFF', 'PYRIGHT', 'CARGO_TEST', 'CARGO_CLIPPY', 'NPX', 'OPAQUE',
        }

    def test_str_is_byte_identical_member_value(self):
        """str(ToolKind.PYTEST) == 'pytest' — a StrEnum member IS its string value."""
        assert str(ToolKind.PYTEST) == 'pytest'
        assert ToolKind.PYTEST == 'pytest'


class TestVerifyCmdConstruction:
    """VerifyCmd is a frozen dataclass with 8 fields and sensible empty defaults."""

    def test_constructs_with_all_fields(self):
        cmd = VerifyCmd(
            tool=ToolKind.PYTEST,
            uv_project='shared',
            cwd_rel='fused-memory',
            base_flags=('-v',),
            targets=('tests/test_x.py',),
            env={'FOO': 'bar'},
            wrappers=('npx',),
            raw=None,
        )
        assert cmd.tool is ToolKind.PYTEST
        assert cmd.uv_project == 'shared'
        assert cmd.cwd_rel == 'fused-memory'
        assert cmd.base_flags == ('-v',)
        assert cmd.targets == ('tests/test_x.py',)
        assert cmd.env == {'FOO': 'bar'}
        assert cmd.wrappers == ('npx',)
        assert cmd.raw is None

    def test_empty_defaults(self):
        """Only `tool` is required; every other field defaults to empty/None."""
        cmd = VerifyCmd(tool=ToolKind.OPAQUE)
        assert cmd.uv_project is None
        assert cmd.cwd_rel is None
        assert cmd.base_flags == ()
        assert cmd.targets == ()
        assert cmd.env == {}
        assert cmd.wrappers == ()
        assert cmd.raw is None

    def test_is_frozen(self):
        cmd = VerifyCmd(tool=ToolKind.PYTEST)
        with pytest.raises(dataclasses.FrozenInstanceError):
            cmd.tool = ToolKind.RUFF

    def test_equal_instances_compare_equal(self):
        a = VerifyCmd(tool=ToolKind.PYTEST, targets=('x',))
        b = VerifyCmd(tool=ToolKind.PYTEST, targets=('x',))
        assert a == b


class TestParseConfigCommandHeadClassification:
    """parse_config_command classifies a single-segment command's head token."""

    def test_pytest(self):
        cmd = parse_config_command('pytest tests/test_x.py')
        assert cmd.tool is ToolKind.PYTEST
        assert cmd.targets == ('tests/test_x.py',)
        assert cmd.raw is None

    def test_ruff_check(self):
        cmd = parse_config_command('ruff check src/foo.py')
        assert cmd.tool is ToolKind.RUFF
        assert cmd.targets == ('src/foo.py',)

    def test_pyright(self):
        cmd = parse_config_command('pyright src/foo.py')
        assert cmd.tool is ToolKind.PYRIGHT
        assert cmd.targets == ('src/foo.py',)

    def test_cargo_test(self):
        cmd = parse_config_command('cargo test --workspace')
        assert cmd.tool is ToolKind.CARGO_TEST

    def test_cargo_clippy(self):
        cmd = parse_config_command('cargo clippy --workspace')
        assert cmd.tool is ToolKind.CARGO_CLIPPY

    def test_npx_bare(self):
        """A non-pyright npx subcommand classifies as the catch-all NPX kind."""
        cmd = parse_config_command('npx --version')
        assert cmd.tool is ToolKind.NPX


class TestParseConfigCommandUvWrapper:
    """parse_config_command recognises a `uv run [--project X|--directory X] <tool>` wrapper."""

    def test_uv_run_with_project_sets_uv_project(self):
        cmd = parse_config_command('uv run --project shared pytest tests/x.py')
        assert cmd.tool is ToolKind.PYTEST
        assert cmd.uv_project == 'shared'
        assert cmd.cwd_rel is None
        assert cmd.targets == ('tests/x.py',)

    def test_uv_run_with_directory_sets_cwd_rel(self):
        cmd = parse_config_command('uv run --directory foo ruff check x')
        assert cmd.tool is ToolKind.RUFF
        assert cmd.cwd_rel == 'foo'
        assert cmd.targets == ('x',)

    def test_bare_uv_run_sets_empty_string_uv_project(self):
        """A bare `uv run <tool>` (no --project/--directory) is uv-wrapped but
        projectless — tri-state uv_project='' (uv-wrapped, no project) is
        distinct from None (not uv-wrapped at all); see reproject()."""
        cmd = parse_config_command('uv run ruff check x')
        assert cmd.tool is ToolKind.RUFF
        assert cmd.uv_project == ''
        assert cmd.cwd_rel is None

    def test_no_uv_wrapper_leaves_uv_project_none(self):
        cmd = parse_config_command('pytest tests/x.py')
        assert cmd.uv_project is None


class TestParseConfigCommandLeadingCd:
    """parse_config_command recognises a leading `cd <dir> &&` segment."""

    def test_leading_cd_sets_cwd_rel(self):
        cmd = parse_config_command('cd fused-memory && npx pyright')
        assert cmd.tool is ToolKind.PYRIGHT
        assert cmd.cwd_rel == 'fused-memory'

    def test_no_leading_cd_leaves_cwd_rel_none(self):
        cmd = parse_config_command('pytest tests/x.py')
        assert cmd.cwd_rel is None


class TestParseConfigCommandOpaque:
    """Unparseable / empty / unrecognised-head commands classify OPAQUE, raw retained."""

    def test_unbalanced_quote_is_opaque(self):
        cmd = parse_config_command('pytest "unterminated')
        assert cmd.tool is ToolKind.OPAQUE
        assert cmd.raw == 'pytest "unterminated'

    def test_empty_command_is_opaque(self):
        cmd = parse_config_command('')
        assert cmd.tool is ToolKind.OPAQUE
        assert cmd.raw == ''

    def test_unrecognised_head_is_opaque(self):
        raw = 'mypy src/'
        cmd = parse_config_command(raw)
        assert cmd.tool is ToolKind.OPAQUE
        assert cmd.raw == raw

    def test_opaque_retains_raw_and_no_structured_fields(self):
        """raw == input verbatim; every structured field stays at its default (P1)."""
        raw = 'true'
        cmd = parse_config_command(raw)
        assert cmd.raw == raw
        assert cmd.uv_project is None
        assert cmd.cwd_rel is None
        assert cmd.base_flags == ()
        assert cmd.targets == ()
        assert cmd.wrappers == ()


class TestRenderRoundTrip:
    """render(parse(x)) is argv-equivalent to x for well-formed x (P2)."""

    @pytest.mark.parametrize(
        'raw',
        [
            'pytest tests/test_x.py',
            'ruff check src/foo.py',
            'pyright src/foo.py',
            'cargo test --workspace',
            'cargo clippy --workspace',
            'npx --version',
            'uv run --project shared pytest tests/x.py',
            'cd fused-memory && npx pyright',
        ],
        ids=[
            'pytest', 'ruff', 'pyright', 'cargo_test', 'cargo_clippy', 'npx',
            'uv_run_project', 'leading_cd_npx',
        ],
    )
    def test_round_trip_argv_equivalent(self, raw):
        assert shlex.split(render(parse_config_command(raw))) == shlex.split(raw)


class TestRenderOpaqueExact:
    """render(parse(OPAQUE_raw)) == OPAQUE_raw exactly — raw passes through verbatim."""

    @pytest.mark.parametrize(
        'raw',
        ['mypy src/', 'pytest "unterminated', ''],
        ids=['unrecognised_head', 'unbalanced_quote', 'empty'],
    )
    def test_opaque_renders_back_to_raw(self, raw):
        assert render(parse_config_command(raw)) == raw


class TestRenderCanonicalFieldOrder:
    """render reconstructs cwd_rel, uv_project (--project), base_flags and
    targets in a canonical order — pinned independent of parse_config_command."""

    def test_full_field_order(self):
        cmd = VerifyCmd(
            tool=ToolKind.PYTEST,
            uv_project='shared',
            cwd_rel='fused-memory',
            base_flags=('-v',),
            targets=('tests/test_x.py',),
        )
        assert render(cmd) == 'cd fused-memory && uv run --project shared pytest -v tests/test_x.py'


class TestScopeTo:
    """scope_to(cmd, files) replaces targets with *files*, worktree-root-relative."""

    def test_replaces_targets_preserving_tool_flags_uv_project(self):
        cmd = VerifyCmd(
            tool=ToolKind.PYTEST,
            uv_project='shared',
            base_flags=('-v',),
            targets=('tests/old.py',),
        )
        scoped = scope_to(cmd, ['tests/new_a.py', 'tests/new_b.py'])
        assert scoped.tool is ToolKind.PYTEST
        assert scoped.uv_project == 'shared'
        assert scoped.base_flags == ('-v',)
        assert scoped.targets == ('tests/new_a.py', 'tests/new_b.py')

    def test_dash_prefixed_flags_not_harvested_into_targets(self):
        """Migrates the _scope_command dash-token regression (verify.py's

        pre-VerifyCmd scoper harvested any dash-prefixed remainder token as a
        stray flag rather than leaving it out of the new target list; the
        structural equivalent here is that base_flags parsed from the
        original command must survive scope_to's targets-only replacement
        untouched — no flag ever leaks into (or out of) `targets`.
        """
        cmd = parse_config_command('pytest -v tests/old.py')
        assert cmd.base_flags == ('-v',)  # parsed once, not re-derived by scope_to
        scoped = scope_to(cmd, ['tests/new.py'])
        assert scoped.base_flags == ('-v',)
        assert scoped.targets == ('tests/new.py',)

    def test_noop_on_opaque(self):
        """scope_to on an OPAQUE VerifyCmd is a no-op returning the same OPAQUE (P1)."""
        cmd = parse_config_command('mypy src/')
        assert cmd.tool is ToolKind.OPAQUE
        assert scope_to(cmd, ['tests/new.py']) == cmd

    def test_noop_on_raw_retained_chain(self):
        """scope_to also no-ops on a recognised-but-unstructurable raw chain —

        targets is repurposed by cargo_scope for crate flags on such chains,
        so scope_to must not touch it.
        """
        raw = 'cargo test --workspace && cargo test --workspace'
        cmd = parse_config_command(raw)
        assert cmd.raw == raw
        assert scope_to(cmd, ['tests/new.py']) == cmd

    def test_empty_files_is_noop(self):
        cmd = parse_config_command('pytest tests/old.py')
        assert scope_to(cmd, []) == cmd

    def test_pyright_merge_gate_scoping(self):
        """Mirrors test_pyright_merge_gate.py: scoping a subproject's pyright

        command to the changed test file(s) alone must produce a runnable
        `pyright <changed files>` invocation.
        """
        cmd = parse_config_command('uv run --project fused-memory pyright fused-memory/')
        scoped = scope_to(cmd, ['fused-memory/tests/test_pyright_gate_probe.py'])
        assert render(scoped) == 'uv run --project fused-memory pyright fused-memory/tests/test_pyright_gate_probe.py'


class TestStripCwd:
    """strip_cwd(cmd) clears cwd_rel, unifying the leading-cd and --directory forms."""

    def test_clears_cwd_rel_from_leading_cd(self):
        cmd = parse_config_command('cd fused-memory && npx pyright')
        assert cmd.cwd_rel == 'fused-memory'
        assert strip_cwd(cmd).cwd_rel is None

    def test_clears_cwd_rel_from_uv_directory_flag(self):
        cmd = parse_config_command('uv run --directory fused-memory pytest tests/')
        assert cmd.cwd_rel == 'fused-memory'
        assert strip_cwd(cmd).cwd_rel is None

    def test_strip_cwd_then_scope_to_root_relative_runs_from_worktree_root(self):
        """Regression 4bb128496f: scoping a root-level file after stripping cwd

        must not re-resolve the path inside the (now-stripped) subproject
        directory — the rendered command must have no leading `cd` at all.
        """
        cmd = parse_config_command('cd fused-memory && uv run pytest tests/')
        fixed = strip_cwd(cmd)
        scoped = scope_to(fixed, ['tests/scripts/test_spawn_claude.py'])
        assert render(scoped) == 'uv run pytest tests/scripts/test_spawn_claude.py'

    def test_noop_on_opaque(self):
        cmd = parse_config_command('mypy src/')
        assert strip_cwd(cmd) == cmd

    def test_noop_on_raw_retained_chain(self):
        raw = 'cargo test --workspace && cargo test --workspace'
        cmd = parse_config_command(raw)
        assert strip_cwd(cmd) == cmd

    def test_noop_when_no_cwd_set(self):
        cmd = parse_config_command('pytest tests/x.py')
        assert cmd.cwd_rel is None
        assert strip_cwd(cmd) == cmd


class TestReproject:
    """reproject(cmd, project) sets uv_project on a bare `uv run <tool>`."""

    def test_bare_uv_run_reprojects(self):
        """Regression ef68777a17: a bare `uv run ruff check X` gains --project."""
        cmd = parse_config_command('uv run ruff check x')
        assert cmd.uv_project == ''
        reprojected = reproject(cmd, 'shared')
        assert reprojected.uv_project == 'shared'
        assert render(reprojected) == 'uv run --project shared ruff check x'

    def test_noop_when_project_already_set(self):
        """Structural equivalent of 05c2d87a72's clause-scoped guard: an

        explicit --project already present means don't second-guess it.
        """
        cmd = parse_config_command('uv run --project orchestrator ruff check x')
        assert reproject(cmd, 'shared') == cmd

    def test_noop_when_directory_already_set(self):
        """An explicit --directory already present is also an explicit uv

        context — don't second-guess it either.
        """
        cmd = parse_config_command('uv run --directory foo ruff check x')
        assert reproject(cmd, 'shared') == cmd

    def test_idempotent(self):
        cmd = parse_config_command('uv run ruff check x')
        once = reproject(cmd, 'shared')
        twice = reproject(once, 'shared')
        assert twice == once

    def test_noop_on_non_uv_command(self):
        cmd = parse_config_command('ruff check x')
        assert cmd.uv_project is None
        assert reproject(cmd, 'shared') == cmd

    def test_noop_on_opaque(self):
        cmd = parse_config_command('mypy src/')
        assert reproject(cmd, 'shared') == cmd

    def test_noop_on_raw_retained_chain(self):
        raw = 'cargo test --workspace && cargo test --workspace'
        cmd = parse_config_command(raw)
        assert reproject(cmd, 'shared') == cmd


class TestCargoScopeStructured:
    """cargo_scope(crates) on a single, structured cargo invocation.

    Migrates test_verify.py::TestScopeCargoWorkspaceRewrite's A1/A2/A3/A6
    cases (regression fd4758fcff: --exclude flags must be dropped after
    rewriting --workspace -> -p <crate>).
    """

    def test_a1_single_exclude_stripped(self):
        raw = 'cargo test --workspace --exclude foo -- --test-threads=1'
        cmd = parse_config_command(raw)
        result = render(cargo_scope(cmd, ['bar']))
        assert '-p bar' in result
        assert '--workspace' not in result
        assert '--exclude' not in result
        assert 'foo' not in result
        assert '-- --test-threads=1' in result

    def test_a2_multiple_excludes_all_stripped(self):
        raw = (
            'cargo test --workspace '
            '--exclude alpha --exclude beta --exclude gamma '
            '-- --test-threads=1'
        )
        cmd = parse_config_command(raw)
        result = render(cargo_scope(cmd, ['delta']))
        assert '-p delta' in result
        assert '--workspace' not in result
        assert '--exclude' not in result
        assert 'alpha' not in result
        assert 'beta' not in result
        assert 'gamma' not in result

    def test_a3_exclude_equals_form_stripped(self):
        raw = 'cargo test --workspace --exclude=foo -- --test-threads=1'
        cmd = parse_config_command(raw)
        result = render(cargo_scope(cmd, ['bar']))
        assert '-p bar' in result
        assert '--workspace' not in result
        assert '--exclude' not in result

    def test_a6_idempotent_on_already_scoped_command(self):
        """A command that's already -p-scoped (no --workspace) is a no-op."""
        raw = 'cargo test -p my-crate -- --test-threads=1'
        cmd = parse_config_command(raw)
        assert render(cargo_scope(cmd, ['my-crate'])) == raw

    def test_noop_empty_crates(self):
        cmd = parse_config_command('cargo test --workspace')
        assert cargo_scope(cmd, []) == cmd

    def test_noop_non_cargo_tool(self):
        cmd = parse_config_command('pytest tests/x.py')
        assert cargo_scope(cmd, ['bar']) == cmd

    def test_noop_on_opaque(self):
        cmd = parse_config_command('mypy src/')
        assert cargo_scope(cmd, ['bar']) == cmd


class TestCargoScopeRawRetainedChain:
    """cargo_scope on a recognised-but-unstructurable multi-segment cargo chain.

    Migrates test_verify.py::TestScopeCargoWorkspaceRewrite's A4/A5/A7 cases:
    the reify orchestrator.yaml 4-segment test_command (two gated wrapper
    segments with no --workspace, followed by two ungated --workspace
    segments).
    """

    _GATED_1 = (
        './scripts/cargo-test-occt-gated.sh cargo test '
        '-p reify-kernel-occt -p reify-eval -p reify-cli'
    )
    _GATED_2 = (
        './scripts/cargo-test-occt-gated.sh cargo test '
        '-p reify-kernel-occt-extra'
    )
    _UNGATED_EXCLUDES = (
        '--exclude reify-kernel-occt --exclude reify-eval '
        '--exclude reify-cli --exclude reify-kernel-occt-extra'
    )

    def test_a4_gated_segments_untouched_ungated_segments_rewritten(self):
        ungated = f'cargo test --workspace {self._UNGATED_EXCLUDES} -- --test-threads=1'
        raw = f'{self._GATED_1} && {self._GATED_2} && {ungated} && {ungated}'

        cmd = parse_config_command(raw)
        assert cmd.tool is ToolKind.CARGO_TEST
        assert cmd.raw == raw

        result = render(cargo_scope(cmd, ['reify-compiler']))
        assert self._GATED_1 in result, f'gated_1 missing: {result!r}'
        assert self._GATED_2 in result, f'gated_2 missing: {result!r}'
        assert '--workspace' not in result
        assert '-p reify-compiler' in result
        assert '-- --test-threads=1' in result

    def test_a5_non_cargo_exclude_in_chain_not_stripped(self):
        raw = (
            'cargo test --workspace --exclude some-crate -- --test-threads=1'
            ' && npm test --exclude foo'
        )
        cmd = parse_config_command(raw)
        result = render(cargo_scope(cmd, ['my-crate']))
        assert 'npm test --exclude foo' in result, (
            f'npm --exclude was incorrectly removed: {result!r}'
        )
        cargo_part = result.split('&&')[0]
        assert '--exclude' not in cargo_part

    def test_a7_no_exclude_token_in_rewritten_ungated_segment(self):
        """The rewritten (last) segment alone must be clean of --exclude/crate

        names — a whole-string check would be masked by the gated segment's
        own, legitimate `-p reify-kernel-occt` substring.
        """
        ungated = f'cargo test --workspace {self._UNGATED_EXCLUDES} -- --test-threads=1'
        raw = f'{self._GATED_1} && {ungated}'
        cmd = parse_config_command(raw)
        result = render(cargo_scope(cmd, ['reify-compiler']))
        segments = [s.strip() for s in result.split('&&')]
        rewritten = segments[-1]
        assert '--exclude' not in rewritten
        for excluded in ('reify-kernel-occt', 'reify-eval', 'reify-cli'):
            assert excluded not in rewritten

    def test_noop_no_workspace_in_chain(self):
        raw = 'cargo test -p already-scoped && cargo test -p other'
        cmd = parse_config_command(raw)
        assert cargo_scope(cmd, ['bar']) == cmd


class TestSerialPytest:
    """serial_pytest() appends the `-p no:xdist -o addopts=` serial-recovery flags.

    Migrates test_verify_env_transient.py::TestForceSerialPytest (step-5).
    """

    # The real multi-module test_command from orchestrator/config.yaml — six
    # chained `cd <module> && uv run pytest tests/` invocations.
    REAL_CONFIG_TEST_COMMAND = (
        'cd shared && uv run pytest tests/ && '
        'cd ../escalation && uv run pytest tests/ && '
        'cd ../orchestrator && uv run pytest tests/ && '
        'cd ../fused-memory && uv run pytest tests/ && '
        'cd ../dashboard && uv run pytest tests/'
    )

    def test_structured_single_invocation_appends_flags(self):
        cmd = parse_config_command('pytest tests/x.py')
        result = render(serial_pytest(cmd))
        assert result == 'pytest -p no:xdist -o addopts= tests/x.py'

    def test_rewrites_every_pytest_invocation_in_chained_command(self):
        """Each of the 5 chained `uv run pytest tests/` segments gains the flags."""
        cmd = parse_config_command(self.REAL_CONFIG_TEST_COMMAND)
        assert cmd.tool is ToolKind.PYTEST
        assert cmd.raw == self.REAL_CONFIG_TEST_COMMAND

        result = render(serial_pytest(cmd))
        assert result.count('pytest') == 5
        assert result.count("-p no:xdist -o addopts=''") == 5

    def test_non_pytest_command_returned_unchanged(self):
        cmd = parse_config_command('cargo test --workspace')
        assert serial_pytest(cmd) == cmd

    def test_noop_on_opaque(self):
        cmd = parse_config_command('mypy src/')
        assert serial_pytest(cmd) == cmd


class TestGovernCpu:
    """govern_cpu(exec_path) wraps the whole rendered command as an outer

    cpu-governed-exec.sh invocation. Migrates test_verify.py::
    TestMaybeGovernMergeCmd's merge-role-wrap and shell-operator-survival
    cases.
    """

    _EXEC = '/abs/scripts/cpu-governed-exec.sh'

    def test_wraps_rendered_command(self):
        cmd = parse_config_command('cargo test --workspace')
        result = render(govern_cpu(cmd, self._EXEC))
        expected = (
            f'{shlex.quote(self._EXEC)} --role merge -- '
            f'/bin/bash -c {shlex.quote(render(cmd))}'
        )
        assert result == expected

    def test_shell_operators_survive_inside_quoted_inner(self):
        """A raw-retained chain's operators (&&) survive intact inside the

        shlex.quote'd inner payload — mirrors _maybe_govern_merge_cmd's
        test_shell_operators_in_cmd_survive.
        """
        raw = 'cargo test && cargo clippy --all -- -D warnings'
        cmd = parse_config_command(raw)
        assert cmd.raw == raw  # a chain; render(cmd) == raw verbatim

        result = render(govern_cpu(cmd, self._EXEC))
        expected = f'{shlex.quote(self._EXEC)} --role merge -- /bin/bash -c {shlex.quote(raw)}'
        assert result == expected

    def test_noop_falsy_exec_path(self):
        cmd = parse_config_command('pytest tests/x.py')
        assert govern_cpu(cmd, '') == cmd
        assert govern_cpu(cmd, None) == cmd

    def test_noop_on_opaque(self):
        """govern_cpu on OPAQUE is a no-op (P1); OPAQUE still renders to its raw."""
        cmd = parse_config_command('mypy src/')
        assert govern_cpu(cmd, self._EXEC) == cmd
        assert render(cmd) == 'mypy src/'
