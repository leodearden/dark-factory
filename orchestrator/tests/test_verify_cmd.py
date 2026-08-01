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
    apply_pytest_numprocesses,
    cargo_scope,
    describe_dropped_clauses,
    govern_cpu,
    has_unpreserved_chain_clauses,
    parse_config_command,
    render,
    reproject,
    scope_to,
    serial_pytest,
    split_chain_tail,
    split_top_level_and,
    strip_cwd,
    with_junitxml,
)

# ---------------------------------------------------------------------------
# Real config command strings, verbatim from the repo's orchestrator configs.
# These are the corpus `split_chain_tail`'s gate must classify correctly: the
# lint chains are SIBLING-CHECKER chains (accept, preserve the tail); the root
# type/test chains are cwd-sequenced same-tool fan-outs (reject, keep today's
# truncation).
# ---------------------------------------------------------------------------

# fused-memory/orchestrator.yaml:11
_FM_LINT_COMMAND = (
    'uv run --project fused-memory --directory fused-memory ruff check src/ tests/'
    ' && python3 fused-memory/scripts/check_bare_magicmock_config.py fused-memory/tests'
    ' && python3 fused-memory/scripts/check_asyncmock_assertion_style.py fused-memory/tests'
)

# dark-factory-orchestrator.yaml:50
_ROOT_LINT_COMMAND = (
    'uv run ruff check shared escalation fused-memory orchestrator dashboard'
    ' && python3 fused-memory/scripts/check_bare_magicmock_config.py shared/tests'
    ' escalation/tests fused-memory/tests orchestrator/tests dashboard/tests'
)

# dark-factory-orchestrator.yaml:51
_ROOT_TYPE_CHECK_COMMAND = (
    'cd fused-memory && npx pyright && cd ../orchestrator && npx pyright'
    ' && cd ../dashboard && npx pyright'
)

# dark-factory-orchestrator.yaml:41
_ROOT_TEST_COMMAND = (
    'cd shared && uv run pytest tests/ --timeout=300'
    ' && cd ../escalation && uv run pytest tests/ --timeout=300'
    ' && cd ../orchestrator && uv run pytest tests/ --timeout=300'
    ' && cd ../fused-memory && uv run pytest tests/ --timeout=300'
    ' && cd ../dashboard && uv run pytest tests/ --timeout=300'
    ' && cd ../sampler && uv run pytest tests/ --timeout=300'
    ' && cd .. && ( [ -d cockpit ] || exit 0; cd cockpit && uv run pytest tests/ --timeout=300 )'
    ' && uv run --project shared pytest tests/scripts/ --timeout=300'
)

# Not a config in this repo (yet) — the shape task 3218 predicts and must not
# regress on: a pytest slot chaining a whole-directory sibling checker. Two
# spellings, because the SCRIPT PATH decides which degradation fires today:
#
# * `check_pytest_markers.py` NAMES the tool, so the pre-3218 substring test
#   at condition 7 sees 'pytest' in segment 1 and rejects — accidentally
#   protecting the pytest slot. Once step-6 replaces that substring test with
#   argv-head matching the accident disappears, and only the keyword
#   ALLOWLIST (condition 0) keeps the slot structured.
# * `check_markers.py` does NOT name the tool, so the tail is preserved TODAY
#   and the scoped command comes back raw-retained — on which `with_junitxml`
#   and `with_pytest_timeout` are documented no-ops. That is degradation 1
#   live, and it is what makes this class RED before the allowlist lands.
_SIBLING_CHECKER_TEST_COMMAND = (
    'uv run pytest tests/ && python3 scripts/check_pytest_markers.py tests'
)
_SIBLING_CHECKER_TEST_COMMAND_UNNAMED = (
    'uv run pytest tests/ && python3 scripts/check_markers.py tests'
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
            cmd.tool = ToolKind.RUFF  # type: ignore[misc]

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

    def test_uv_run_with_project_and_directory_sets_both(self):
        """--project and --directory can both appear on one `uv run` wrapper.

        Real per-subproject commands (orchestrator.yaml) carry both flags
        together, e.g. `uv run --project orchestrator --directory
        orchestrator pyright src/ tests/` — --project selects the venv,
        --directory shifts cwd. Both must be captured, not just the first
        one peeled.
        """
        cmd = parse_config_command(
            'uv run --project orchestrator --directory orchestrator pyright src/ tests/'
        )
        assert cmd.tool is ToolKind.PYRIGHT
        assert cmd.uv_project == 'orchestrator'
        assert cmd.cwd_rel == 'orchestrator'
        assert cmd.targets == ('src/', 'tests/')

    def test_uv_run_with_directory_and_project_reversed_order_sets_both(self):
        """The two flags are also recognised in the opposite order."""
        cmd = parse_config_command(
            'uv run --directory orchestrator --project orchestrator pyright src/'
        )
        assert cmd.uv_project == 'orchestrator'
        assert cmd.cwd_rel == 'orchestrator'
        assert cmd.targets == ('src/',)


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


class TestSeparateTokenValueFlagBinding:
    """A pytest separate-token value flag (-k/-m/-p/-o/-n/...) must bind to its

    following value at PARSE time, as an adjacent ``(flag, value)`` pair
    inside ``base_flags`` — not be severed by the naive dash-prefix split,
    which strands the value in ``targets``. Because every ``base_flags``
    mutator (``apply_pytest_numprocesses``, ``serial_pytest``,
    ``with_junitxml``) appends to the END of ``base_flags``, a bound
    contiguous pair can never have a later flag inserted between it and its
    value (task 2727).

    Mirrors TestSerialPytest's and
    test_verify_admission_pytest_n.py::TestApplyPytestNumprocesses's
    parse -> mutate -> render -> assert style.
    """

    @pytest.mark.parametrize(
        ('flag', 'value'),
        [
            ('-k', 'foo'),
            ('-m', 'slow'),
            ('-p', 'no:cacheprovider'),
            ('-o', 'addopts='),
            ('-n', '4'),
        ],
    )
    def test_value_flag_binds_to_following_token_at_parse_time(self, flag, value):
        cmd = parse_config_command(f'pytest {flag} {value} tests/')
        assert cmd.base_flags == (flag, value)
        assert cmd.targets == ('tests/',)

    def test_round_trip_preserved(self):
        """The pre-existing coincidental round-trip must still hold post-fix."""
        raw = 'pytest -k foo tests/'
        cmd = parse_config_command(raw)
        assert shlex.split(render(cmd)) == shlex.split(raw)

    @pytest.mark.parametrize('flag', ['-k', '-m', '-p'])
    def test_apply_pytest_numprocesses_does_not_split_bound_value_flag(self, flag):
        """Acceptance regression: appending `-n 16` must never land between a

        bound value flag and its value (on main this renders
        `pytest -k -n 16 foo tests/` — the flag consumes -n and the value
        becomes a misplaced positional).
        """
        cmd = parse_config_command(f'pytest {flag} VAL tests/')
        rendered = render(apply_pytest_numprocesses(cmd, '16'))
        tokens = shlex.split(rendered)
        assert tokens[tokens.index(flag) + 1] == 'VAL'
        assert tokens[tokens.index('-n') + 1] == '16'
        assert f'{flag} -n' not in rendered, f'flag/value split corruption in {rendered!r}'

    def test_serial_pytest_does_not_split_bound_value_flag(self):
        """Acceptance regression: serial_pytest's appended `-p no:xdist -o

        addopts=` must never land between `-k` and its value `foo`.
        """
        cmd = parse_config_command('pytest -k foo tests/')
        rendered = render(serial_pytest(cmd))
        tokens = shlex.split(rendered)
        assert tokens[tokens.index('-k') + 1] == 'foo'

    @pytest.mark.parametrize('flag', ['-k', '-m', '-p', '-o', '-n'])
    def test_trailing_value_flag_falls_back_to_bare_flag(self, flag):
        """A listed value flag with no following token (e.g. the expression

        omitted from `pytest tests/ -k`) must not index past the end of
        `rest` — _split_pytest_args's `i + 1 < n` guard falls it back to the
        bare-flag classification rather than raising IndexError (see
        _split_pytest_args's docstring). This is the one input shape the fix
        does NOT fully protect against a later base_flags append corrupting
        the command (e.g. a subsequent apply_pytest_numprocesses would still
        render `pytest -k -n 16`, where pytest's -k would swallow -n) — that
        is malformed input the fix does not claim to guard against.
        """
        cmd = parse_config_command(f'pytest tests/ {flag}')
        assert flag in cmd.base_flags
        assert cmd.targets == ('tests/',)

    @pytest.mark.parametrize(
        'token',
        [
            '--maxfail=2',
            '-n=4',
            '--override-ini=addopts=',
        ],
    )
    def test_attached_value_form_does_not_consume_following_token(self, token):
        """A single-token `--flag=value` form is a distinct grammar branch

        from a separate-token value flag: it is NOT a member of
        _PYTEST_VALUE_FLAGS (which lists only separate-token value flags),
        so it correctly falls to the `tok.startswith('-')` bare-flag arm and
        must not consume the following token as though it were a bound
        value-flag pair.
        """
        cmd = parse_config_command(f'pytest {token} tests/')
        assert cmd.base_flags == (token,)
        assert cmd.targets == ('tests/',)


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

    def test_opaque_command_still_wrapped(self):
        """govern_cpu applies to OPAQUE too — the sole deliberate P1 exemption.

        The historical _maybe_govern_merge_cmd bash-wrapped ANY non-None
        command for role=='merge' regardless of shape; OPAQUE (arbitrary/
        unparseable shell) is exactly the case that safe bash-wrap exists
        for. Dropping merge cpu-governance for OPAQUE would silently starve
        dark_factory's real lint_command/type_check_command chains (which
        parse OPAQUE — see TestParseChain) of the merge-weighted cgroup
        scope every other merge verify command gets.
        """
        cmd = parse_config_command('mypy src/')
        assert cmd.tool is ToolKind.OPAQUE

        result = render(govern_cpu(cmd, self._EXEC))
        expected = f'{shlex.quote(self._EXEC)} --role merge -- /bin/bash -c {shlex.quote("mypy src/")}'
        assert result == expected

    def test_chained_non_pytest_non_cargo_merge_command_still_governed(self):
        """A chained lint-style command (ruff-check && a follow-up script) is

        the real dark_factory.orchestrator.config.yaml lint_command/
        type_check_command shape: multi-clause, non-pytest/non-cargo, so
        parse_config_command classifies it OPAQUE (_parse_chain only
        recognises pytest/cargo chains). It must still be cpu-governed on
        merge, matching the historical unconditional bash-wrap.
        """
        raw = (
            'uv run ruff check shared escalation && '
            'python3 fused-memory/scripts/check_bare_magicmock_config.py shared/tests'
        )
        cmd = parse_config_command(raw)
        assert cmd.tool is ToolKind.OPAQUE
        assert cmd.raw == raw

        result = render(govern_cpu(cmd, self._EXEC))
        expected = f'{shlex.quote(self._EXEC)} --role merge -- /bin/bash -c {shlex.quote(raw)}'
        assert result == expected


class TestWithJunitxml:
    """with_junitxml(cmd, junit_path) appends a `--junitxml <path>` flag to a
    structured pytest command's base_flags (task μ, verify-scope-inversion-prd.md).

    Deliberately narrower than apply_pytest_numprocesses/serial_pytest: there
    is NO raw-chain regex-rewrite branch — a raw-retained pytest chain (a
    recognised-but-unstructurable `&&`-chain) is returned byte-identical
    rather than rewritten, and OPAQUE/non-pytest commands no-op (P1). This
    means a merge-role test_command that happens to be a multi-segment chain
    degrades gracefully to no junit collection for that run (B3) rather than
    risking a mis-scoped regex injection into an unstructured shell string.
    """

    def test_structured_single_invocation_appends_junitxml_flag(self):
        cmd = parse_config_command('pytest tests/x.py')
        result = render(with_junitxml(cmd, '/abs/attempt/junit.xml'))
        assert result == 'pytest --junitxml /abs/attempt/junit.xml tests/x.py'

    def test_base_flags_end_with_junitxml_pair(self):
        cmd = parse_config_command('pytest tests/x.py')
        mutated = with_junitxml(cmd, '/abs/attempt/junit.xml')
        assert mutated.base_flags[-2:] == ('--junitxml', '/abs/attempt/junit.xml')
        assert mutated.raw is None

    def test_raw_retained_chain_returned_unchanged(self):
        """A recognised-but-unstructurable pytest chain (raw is not None) is
        NEVER regex-rewritten — the sole deliberate divergence from
        apply_pytest_numprocesses/serial_pytest, which DO rewrite every
        invocation in such a chain. Byte-identical no-op here instead.
        """
        raw = (
            'cd shared && uv run pytest tests/ && '
            'cd ../orchestrator && uv run pytest tests/'
        )
        cmd = parse_config_command(raw)
        assert cmd.tool is ToolKind.PYTEST
        assert cmd.raw == raw

        result = with_junitxml(cmd, '/abs/attempt/junit.xml')
        assert result == cmd
        assert render(result) == raw

    def test_non_pytest_command_returned_unchanged(self):
        cmd = parse_config_command('ruff check .')
        assert with_junitxml(cmd, '/abs/attempt/junit.xml') == cmd

    def test_pyright_command_returned_unchanged(self):
        cmd = parse_config_command('pyright')
        assert with_junitxml(cmd, '/abs/attempt/junit.xml') == cmd

    def test_noop_on_opaque(self):
        cmd = parse_config_command('mypy src/')
        assert with_junitxml(cmd, '/abs/attempt/junit.xml') == cmd


class TestRenderInvariantAsserts:
    """render() defensively asserts P1 and P3 against a hand-constructed

    VerifyCmd that bypasses every mutator's raw-retained no-op guard —
    the mutators (scope_to/strip_cwd/reproject/cargo_scope/serial_pytest/
    govern_cpu) never themselves produce these states; the asserts document
    and enforce that a raw-retained command's cwd_rel/targets stay at their
    parse-time defaults, since render() ignores them (uses `raw` verbatim).
    """

    def test_opaque_with_forced_fields_raises(self):
        """P1: an OPAQUE VerifyCmd whose structural fields were forced

        non-default (uv_project/cwd_rel/targets/wrappers all set, bypassing
        every mutator's OPAQUE no-op guard) makes render() raise —
        OPAQUE must never have been mutated.
        """
        cmd = VerifyCmd(
            tool=ToolKind.OPAQUE,
            raw='mypy src/',
            uv_project='shared',
            cwd_rel='fused-memory',
            targets=('src/foo.py',),
            wrappers=('/abs/exec.sh',),
        )
        with pytest.raises(AssertionError):
            render(cmd)

    def test_raw_retained_chain_with_forced_cwd_and_targets_raises(self):
        """P3: a raw-retained (non-OPAQUE) chain hand-constructed with a

        forced cwd_rel + targets makes render() raise — neither field is
        legitimately settable on a raw-retained command (cargo_scope /
        serial_pytest rewrite `raw` itself instead), so non-empty `targets`
        alongside a non-None `cwd_rel` here can never be worktree-root-
        relative in a way render() should honour.
        """
        raw = 'cargo test && cargo clippy --all -- -D warnings'
        cmd = VerifyCmd(
            tool=ToolKind.CARGO_TEST,
            raw=raw,
            cwd_rel='some/subdir',
            targets=('some/subdir/x',),
        )
        with pytest.raises(AssertionError):
            render(cmd)

    def test_govern_wrapped_chain_is_the_legitimate_case_and_still_renders(self):
        """The contrasting legitimate case: govern_cpu is the only mutator

        that legitimately sets a field (wrappers) on a raw-retained chain;
        cwd_rel/targets stay at their untouched defaults, so render() does
        not raise.
        """
        raw = 'cargo test && cargo clippy --all -- -D warnings'
        cmd = govern_cpu(parse_config_command(raw), '/abs/scripts/cpu-governed-exec.sh')
        assert cmd.cwd_rel is None
        assert cmd.targets == ()
        render(cmd)  # must not raise


class TestSplitTopLevelAnd:
    """split_top_level_and(raw) splits on `&&` only at shell quote depth 0.

    Segments are returned VERBATIM — interior and boundary whitespace is
    untouched — so a caller can re-emit the tail byte-for-byte rather than
    re-rendering it. A quoted `&&` (single or double) is NOT a split point:
    it is an argument value (e.g. pytest's `-k 'a && b'`), not a shell chain
    operator, and splitting there would corrupt the expression.
    """

    @pytest.mark.parametrize(
        ('raw', 'expected'),
        [
            ('a && b && c', ['a ', ' b ', ' c']),
            ("pytest -k 'a && b' tests/", ["pytest -k 'a && b' tests/"]),
            ('ruff check "x && y"', ['ruff check "x && y"']),
            ('ruff check src/ --select E', ['ruff check src/ --select E']),
            ('', ['']),
            (
                'uv run ruff check f.py && python3 check.py',
                ['uv run ruff check f.py ', ' python3 check.py'],
            ),
        ],
        ids=[
            'three-segments-verbatim-whitespace',
            'single-quoted-and-is-not-a-split-point',
            'double-quoted-and-is-not-a-split-point',
            'no-and-single-segment',
            'empty-string',
            'two-segments',
        ],
    )
    def test_segments(self, raw, expected):
        assert split_top_level_and(raw) == expected

    @pytest.mark.parametrize(
        'raw',
        [
            'a && b && c',
            "pytest -k 'a && b' tests/",
            'ruff check "x && y"',
            'ruff check src/ --select E',
            '',
            _FM_LINT_COMMAND,
            _ROOT_LINT_COMMAND,
            _ROOT_TYPE_CHECK_COMMAND,
            _ROOT_TEST_COMMAND,
        ],
        ids=[
            'three-segments',
            'single-quoted-and',
            'double-quoted-and',
            'no-and',
            'empty-string',
            'fm-lint',
            'root-lint',
            'root-type-check',
            'root-test',
        ],
    )
    def test_round_trip_reconstructs_input(self, raw):
        """The segments are a lossless decomposition: re-joining on `&&` is exact.

        Equivalently, ``''.join(segments)`` is the input minus exactly its
        top-level `&&` separators — no other byte is consumed or rewritten,
        which is what lets a caller re-emit the tail verbatim.
        """
        segments = split_top_level_and(raw)
        assert '&&'.join(segments) == raw
        assert len(''.join(segments)) == len(raw) - 2 * (len(segments) - 1)


class TestSplitChainTail:
    """split_chain_tail(raw, keyword) -> (prefix, tail): the tail-preservation gate.

    ACCEPT (a sibling-checker chain) returns ``(segments[0], tail)`` where
    ``tail`` is every byte of *raw* after segment 0 — so it carries its own
    leading `&&` and ``prefix + tail == raw`` exactly.

    REJECT returns ``(raw, '')`` — deliberately the WHOLE original string, so
    the caller's existing truncate-at-keyword algorithm runs on an untouched
    input and its output stays byte-identical to today's by construction.
    Rejecting to ``(segments[0], '')`` would silently truncate, which is the
    very class of bug this gate exists to fix.
    """

    @pytest.mark.parametrize(
        ('raw', 'keyword'),
        [
            (_FM_LINT_COMMAND, 'ruff check'),
            (_ROOT_LINT_COMMAND, 'ruff check'),
            ('ruff check src/ --select E', 'ruff check'),
            (_ROOT_TYPE_CHECK_COMMAND, 'pyright'),
            (
                'uv run --project a ruff check src/ && uv run --project b ruff check src/',
                'ruff check',
            ),
            ('echo hi && ruff check src/ && python3 x.py', 'ruff check'),
            (_ROOT_TEST_COMMAND, 'pytest'),
            ('ruff check "unterminated && python3 x.py', 'ruff check'),
            ("ruff check -k 'a && b' src/ && python3 x.py", 'ruff check'),
        ],
        ids=[
            'accept-fm-lint',
            'accept-root-lint',
            'reject-no-and',
            'reject-cd-token',
            'reject-keyword-in-two-segments',
            'reject-keyword-absent-from-segment-0',
            'reject-non-and-chain-operator',
            'reject-unbalanced-quote',
            'quoted-and-inside-segment-0',
        ],
    )
    def test_prefix_plus_tail_is_always_the_original(self, raw, keyword):
        """Invariant holding on BOTH dispositions — the gate never loses a byte."""
        prefix, tail = split_chain_tail(raw, keyword)
        assert prefix + tail == raw

    def test_accepts_fused_memory_lint_chain_and_preserves_both_checkers(self):
        """The task's headline case: fused-memory/orchestrator.yaml:11.

        The ruff clause is segment 0 (the caller will scope it); both
        `python3 .../check_*.py` sibling clauses live in the preserved tail,
        byte-identical to their slice of the config string.
        """
        prefix, tail = split_chain_tail(_FM_LINT_COMMAND, 'ruff check')
        assert prefix == (
            'uv run --project fused-memory --directory fused-memory ruff check src/ tests/ '
        )
        assert tail == (
            '&& python3 fused-memory/scripts/check_bare_magicmock_config.py fused-memory/tests'
            ' && python3 fused-memory/scripts/check_asyncmock_assertion_style.py fused-memory/tests'
        )
        assert tail == _FM_LINT_COMMAND[len(prefix):]

    def test_accepts_root_lint_chain(self):
        """dark-factory-orchestrator.yaml:50 — one sibling checker clause."""
        prefix, tail = split_chain_tail(_ROOT_LINT_COMMAND, 'ruff check')
        assert prefix == 'uv run ruff check shared escalation fused-memory orchestrator dashboard '
        assert tail == (
            '&& python3 fused-memory/scripts/check_bare_magicmock_config.py shared/tests'
            ' escalation/tests fused-memory/tests orchestrator/tests dashboard/tests'
        )

    @pytest.mark.parametrize(
        ('raw', 'keyword'),
        [
            ('ruff check src/ --select E', 'ruff check'),
            (_ROOT_TYPE_CHECK_COMMAND, 'pyright'),
            (
                'uv run --project a ruff check src/ && uv run --project b ruff check src/',
                'ruff check',
            ),
            ('echo hi && ruff check src/ && python3 x.py', 'ruff check'),
            (_ROOT_TEST_COMMAND, 'pytest'),
            ('ruff check "unterminated && python3 x.py', 'ruff check'),
            ('mypy src/', 'ruff check'),
            ('true', 'ruff check'),
            ('ruff check $(git ls-files && echo x) && python3 y.py', 'ruff check'),
            ('ruff check `ls && echo x` && python3 y.py', 'ruff check'),
            ('(ruff check src/ && echo x) && python3 y.py', 'ruff check'),
            ('ruff check "$(ls && echo x)" && python3 y.py', 'ruff check'),
        ],
        ids=[
            'no-and-at-all',
            'cd-token-shell-cwd-sequencing',
            'keyword-in-more-than-one-segment',
            'keyword-absent-from-segment-0',
            'non-and-chain-operator',
            'unbalanced-quote-shlex-raises',
            'keyword-absent-entirely',
            'no-op-command',
            'command-substitution-dollar-paren',
            'command-substitution-backtick',
            'unspaced-subshell-parens',
            'substitution-nested-in-double-quotes',
        ],
    )
    def test_rejects_return_whole_raw_and_empty_tail(self, raw, keyword):
        """Every reject disposition is ``(raw, '')`` — never a truncated prefix."""
        assert split_chain_tail(raw, keyword) == (raw, '')

    @pytest.mark.parametrize(
        'raw',
        [
            'ruff check $(git ls-files && echo x) && python3 y.py',
            'ruff check `ls && echo x` && python3 y.py',
            '(ruff check src/ && echo x) && python3 y.py',
            'ruff check "$(ls && echo x)" && python3 y.py',
        ],
        ids=['dollar-paren', 'backtick', 'unspaced-subshell', 'dquoted-substitution'],
    )
    def test_nested_and_inside_a_shell_construct_is_never_a_split_point(self, raw):
        """An `&&` hiding inside `$(...)`, backticks or `(...)` must not be lifted.

        ``_NON_AND_CHAIN_TOKENS`` is token-EQUALITY based, so it only catches a
        paren ``shlex`` isolated as its own whitespace-separated token. These
        four inputs slip past it, yet ``split_top_level_and`` (quote state only)
        happily splits at the nested `&&` — and the shlex cross-check agrees
        with it on the count, so nothing downstream catches it either. Carrying
        a tail out of one truncates the head mid-construct and emits an
        unbalanced shell string (a stray `)` / an unpaired backtick), which is
        a bash syntax error: a spurious RED verify, strictly worse than the
        missed sibling checker this whole gate exists to fix. Reject is the
        only safe disposition — it restores the exact pre-gate output.
        """
        assert split_chain_tail(raw, 'ruff check') == (raw, '')

    def test_literal_paren_inside_double_quotes_is_not_a_shell_construct(self):
        """A quoted paren is inert text, so it must NOT trip the grouping gate.

        Guards the conservative character scan against over-rejection: only a
        substitution (``$(`` / backtick) is active inside double quotes, and a
        bare ``(`` there — a ``-k`` selector expression is the real case — is
        literal. This chain is a legitimate sibling-checker chain and must
        still ACCEPT.
        """
        raw = 'ruff check --config "lint(x)" src/ && python3 y.py'
        prefix, tail = split_chain_tail(raw, 'ruff check')
        assert prefix == 'ruff check --config "lint(x)" src/ '
        assert tail == '&& python3 y.py'
        assert prefix + tail == raw

    def test_quoted_and_in_segment_zero_is_never_corrupted(self):
        """A quoted `&&` inside the keyword segment must survive intact.

        The gate cross-checks the quote-aware splitter against
        ``shlex.split``'s `&&` token count precisely so a quoted `&&` can
        never be mistaken for a split point. Whatever disposition is taken,
        segment 0's quoted expression must come back byte-identical.
        """
        raw = "ruff check -k 'a && b' src/ && python3 x.py"
        prefix, tail = split_chain_tail(raw, 'ruff check')
        assert "-k 'a && b'" in prefix
        assert prefix + tail == raw
        # ACCEPT: only the unquoted `&&` is a split point.
        assert prefix == "ruff check -k 'a && b' src/ "
        assert tail == '&& python3 x.py'


class TestTailPreservationAllowlist:
    """Tail preservation is restricted to an ALLOWLIST of keywords (condition 0).

    Task 3218 part 1. A preserved tail makes the gate's caller return a
    RECOGNISED-BUT-UNSTRUCTURABLE ``VerifyCmd`` (``raw is not None``), and
    ``with_junitxml``/``with_pytest_timeout`` are documented no-ops on that
    shape. For the lint/type slots that costs nothing. For the PYTEST slot it
    silently drops the ``--junitxml`` report that drives
    ``_extract_failing_test_ids_from_junit``, flake confirmation and the
    per-test timeout floor — so ``'pytest'`` is deliberately absent from the
    allowlist and a pytest chain is always rejected to ``(raw, '')``.

    The default for an UNLISTED keyword is no preservation, i.e. exactly the
    pre-task-3061 behaviour: a future verify slot cannot silently acquire the
    degradation by being added, it must opt in explicitly.
    """

    @pytest.mark.parametrize(
        'raw',
        [_SIBLING_CHECKER_TEST_COMMAND, _SIBLING_CHECKER_TEST_COMMAND_UNNAMED],
        ids=['sibling-names-the-tool', 'sibling-does-not-name-the-tool'],
    )
    def test_pytest_chain_is_never_tail_preserved(self, raw):
        """Both spellings reject, and to the WHOLE original — never ``segments[0]``.

        The unnamed-sibling spelling is the one that is ACCEPTED before this
        change; the named-sibling spelling is accepted once step-6 replaces
        condition 7's substring test with argv-head matching. The allowlist
        is what makes the disposition independent of the sibling's filename.
        """
        assert split_chain_tail(raw, 'pytest') == (raw, '')

    @pytest.mark.parametrize(
        ('raw', 'keyword', 'expected_tail'),
        [
            (
                _FM_LINT_COMMAND,
                'ruff check',
                '&& python3 fused-memory/scripts/check_bare_magicmock_config.py'
                ' fused-memory/tests'
                ' && python3 fused-memory/scripts/check_asyncmock_assertion_style.py'
                ' fused-memory/tests',
            ),
            ('npx pyright && python3 y.py', 'pyright', '&& python3 y.py'),
            (
                _FM_LINT_COMMAND,
                'uv run',
                '&& python3 fused-memory/scripts/check_bare_magicmock_config.py'
                ' fused-memory/tests'
                ' && python3 fused-memory/scripts/check_asyncmock_assertion_style.py'
                ' fused-memory/tests',
            ),
        ],
        ids=['ruff-check', 'pyright', 'uv-run'],
    )
    def test_allowlisted_keywords_still_preserve(self, raw, keyword, expected_tail):
        """The three allowlisted keywords keep today's ACCEPT disposition exactly.

        ``'uv run'`` is on the list for ``verify._reproject_str``, whose tail
        preservation is load-bearing: without it a chained lint command
        re-parses OPAQUE and the ``--project`` injection is silently dropped,
        which the depless workspace-root project turns into an exit-127
        breakage (task 2036), not a cosmetic diff.
        """
        prefix, tail = split_chain_tail(raw, keyword)
        assert tail == expected_tail
        assert prefix + tail == raw

    @pytest.mark.parametrize(
        ('raw', 'keyword'),
        [
            (_SIBLING_CHECKER_TEST_COMMAND, 'pytest'),
            (_SIBLING_CHECKER_TEST_COMMAND_UNNAMED, 'pytest'),
            (_FM_LINT_COMMAND, 'ruff check'),
            (_FM_LINT_COMMAND, 'uv run'),
            ('npx pyright && python3 y.py', 'pyright'),
        ],
        ids=[
            'reject-pytest-named-sibling',
            'reject-pytest-unnamed-sibling',
            'accept-ruff-check',
            'accept-uv-run',
            'accept-pyright',
        ],
    )
    def test_prefix_plus_tail_is_always_the_original(self, raw, keyword):
        """The CONSTRAINT holds on the new reject path too — no byte is lost."""
        prefix, tail = split_chain_tail(raw, keyword)
        assert prefix + tail == raw

    def test_reject_returns_the_whole_original_not_segment_zero(self):
        """A REJECT must not silently truncate — that is the bug the gate exists to fix."""
        prefix, tail = split_chain_tail(_SIBLING_CHECKER_TEST_COMMAND_UNNAMED, 'pytest')
        assert prefix == _SIBLING_CHECKER_TEST_COMMAND_UNNAMED
        assert tail == ''
        assert prefix != split_top_level_and(_SIBLING_CHECKER_TEST_COMMAND_UNNAMED)[0]

    @pytest.mark.parametrize(
        'raw',
        [
            _SIBLING_CHECKER_TEST_COMMAND,
            _SIBLING_CHECKER_TEST_COMMAND_UNNAMED,
            'uv run --project orchestrator pytest tests/ && python3 x.py',
            'python3 -m pytest tests/ && python3 x.py',
        ],
        ids=[
            'sibling-names-the-tool',
            'sibling-does-not-name-the-tool',
            'uv-project-wrapper',
            'python-dash-m-wrapper',
        ],
    )
    def test_an_allowlisted_keyword_cannot_smuggle_a_tail_onto_a_pytest_clause(self, raw):
        """Condition 0b — the allowlist is keyed on the KEYWORD, the invariant
        it protects is a property of the SLOT, and ``'uv run'`` is where the
        two come apart.

        ``'uv run'`` is allowlisted for ``verify._reproject_str``, but it is a
        WRAPPER phrase: called with that keyword, every command here clears
        condition 0 even though segment 0 runs pytest. Nothing in the gate
        could stop it except the convention that ``_reproject_str`` is only
        ever handed a lint/type command — which is a comment, not a check, and
        would hand the pytest slot back the exact junitxml/timeout no-op task
        3218 closed. So the gate asks what segment 0 actually INVOKES and
        refuses, whatever keyword it was called with.

        Note each input clears the keyword-level allowlist for real — the
        assertion below is not vacuous — because ``'uv run'`` / ``'python3'``
        do occur in segment 0 and no later segment invokes them.
        """
        assert split_chain_tail(raw, 'uv run') == (raw, '')

    @pytest.mark.parametrize(
        ('raw', 'keyword'),
        [
            (_FM_LINT_COMMAND, 'uv run'),
            ('uv run ruff check src/ && python3 scripts/check_noqa.py src', 'uv run'),
            ('uv run --project orchestrator pyright src/ && python3 x.py', 'uv run'),
        ],
        ids=['fm-lint', 'uv-run-ruff', 'uv-run-pyright'],
    )
    def test_condition_0b_leaves_the_reproject_path_untouched(self, raw, keyword):
        """Condition 0b must cost ``_reproject_str`` nothing: its real inputs
        are lint/type commands, whose segment 0 invokes ruff/pyright — never
        pytest — so every one still preserves its tail.

        This is the load-bearing half. Losing preservation here would drop the
        ``--project`` injection and turn a clean tree RED at exit 127 (task
        2036), so the new condition has to be narrower than "no tails for
        ``'uv run'``".
        """
        prefix, tail = split_chain_tail(raw, keyword)
        assert tail, 'the reproject path must keep preserving its sibling checker'
        assert prefix + tail == raw


class TestGateMatchesToolAtArgvHead:
    """A later segment only counts as the same tool when it INVOKES it (task 3218 part 2).

    The pre-3218 test was ``keyword in segment`` — a plain substring. A
    sibling checker whose SCRIPT PATH happens to name the tool
    (``check_pyright_config.py``, ``check_pytest_markers.py``) therefore read
    as a same-tool fan-out and had its clause dropped, so a real check never
    ran: an over-rejection, and the possible-false-GREEN direction.

    Argv-head matching can only UNDER-reject, and only behind a wrapper the
    module does not recognise (``poetry run ruff check b/``) — the
    consequence there is that clause running UNSCOPED, a superset of the
    checks that would otherwise run, never a false GREEN. It also cannot
    misresolve relative paths, because condition 4 already rejects any chain
    containing a ``cd`` token. That asymmetry is what licenses replacing the
    over-conservative test with a precise one.
    """

    # --- ACCEPT: the tool name appears, but nothing invokes the tool --------

    def test_sibling_checker_script_named_after_the_tool_keeps_its_tail(self):
        """The task's headline part-2 case: `check_pyright_config.py`.

        Segment 1's argv head is ``python3`` (no ``-m``), so nothing there
        invokes pyright — the substring inside the script's filename is not
        an invocation.
        """
        raw = 'npx pyright && python3 scripts/check_pyright_config.py src'
        prefix, tail = split_chain_tail(raw, 'pyright')
        assert prefix == 'npx pyright '
        assert tail == '&& python3 scripts/check_pyright_config.py src'
        assert prefix + tail == raw

    def test_ruff_sibling_checker_keeps_its_tail(self):
        raw = 'uv run ruff check src/ && python3 scripts/check_ruff_noqa.py src'
        prefix, tail = split_chain_tail(raw, 'ruff check')
        assert prefix == 'uv run ruff check src/ '
        assert tail == '&& python3 scripts/check_ruff_noqa.py src'
        assert prefix + tail == raw

    def test_tool_name_as_a_flag_value_keeps_its_tail(self):
        """A quoted flag VALUE spelling the keyword is not an invocation either."""
        raw = 'ruff check src/ && python3 lint.py --tool "ruff check"'
        prefix, tail = split_chain_tail(raw, 'ruff check')
        assert prefix == 'ruff check src/ '
        assert tail == '&& python3 lint.py --tool "ruff check"'
        assert prefix + tail == raw

    # --- REJECT: a genuine same-tool fan-out must keep being truncated ------

    @pytest.mark.parametrize(
        ('raw', 'keyword'),
        [
            (
                'uv run --project a ruff check src/ && uv run --project b ruff check src/',
                'ruff check',
            ),
            ('npx pyright && npx pyright other/', 'pyright'),
            ('ruff check src/ && python3 -m ruff check other/', 'ruff check'),
            (
                'uv run --directory a ruff check src/ && uv run --directory b ruff check src/',
                'ruff check',
            ),
            ('ruff check src/ && python3 x.py "unterminated', 'ruff check'),
        ],
        ids=[
            'uv-run-project-peel',
            'npx-peel',
            'python-dash-m-peel',
            'uv-run-directory-peel',
            'unbalanced-quote-is-conservative',
        ],
    )
    def test_same_tool_fan_out_behind_a_known_wrapper_still_rejects(self, raw, keyword):
        """Each recognised wrapper prefix must still expose the tool at an argv head."""
        assert split_chain_tail(raw, keyword) == (raw, '')

    # --- The helper's own contract -----------------------------------------

    @pytest.mark.parametrize(
        ('segment', 'keyword', 'expected'),
        [
            ('ruff check src/', 'ruff check', True),
            ('uv run ruff check src/', 'ruff check', True),
            ('uv run --project a --directory a ruff check src/', 'ruff check', True),
            ('uv run --directory a --project a ruff check src/', 'ruff check', True),
            ('npx pyright src/', 'pyright', True),
            ('python3 -m ruff check other/', 'ruff check', True),
            ('python -m pytest tests/', 'pytest', True),
            ('uv run ruff check src/', 'uv run', True),
            ('python3 scripts/check_pyright_config.py src', 'pyright', False),
            ('python3 lint.py --tool "ruff check"', 'ruff check', False),
            ('poetry run ruff check b/', 'ruff check', False),
            ('python3 -m coverage run -m pytest', 'pytest', False),
            ('echo "unterminated', 'pytest', True),
        ],
        ids=[
            'bare-head',
            'uv-run-head',
            'uv-run-project-then-directory',
            'uv-run-directory-then-project',
            'npx-head',
            'python3-dash-m',
            'python-dash-m',
            'keyword-at-index-0-before-any-peel',
            'tool-named-in-a-script-path',
            'tool-named-in-a-flag-value',
            'unrecognised-wrapper-under-rejects',
            'not-at-a-head-position',
            'undecodable-segment-is-conservative',
        ],
    )
    def test_segment_invokes_tool(self, segment, keyword, expected):
        """``_segment_invokes_tool`` decides argv-head occupancy, nothing else.

        ``'poetry run ruff check b/'`` is the documented UNDER-rejection: an
        unrecognised wrapper, so the clause is preserved and runs unscoped —
        wasteful, never a false GREEN.

        ``'echo "unterminated'`` is the conservative direction: an
        undecodable segment counts as a MATCH, so the gate rejects and the
        pre-3218 disposition is restored.
        """
        from orchestrator.verify_cmd import _segment_invokes_tool

        assert _segment_invokes_tool(segment, keyword) is expected

    def test_index_zero_is_tested_before_any_wrapper_is_peeled(self):
        """The ``'uv run'`` keyword (verify._reproject_str) must match segment 0.

        Peeling ``uv run`` first and only then looking for the keyword would
        make ``'uv run'`` unmatchable — silently changing ``_reproject_str``'s
        gate disposition on every chained lint command.
        """
        from orchestrator.verify_cmd import _segment_invokes_tool

        assert _segment_invokes_tool('uv run --project a ruff check src/', 'uv run') is True

    # --- Corpus non-regression ---------------------------------------------

    @pytest.mark.parametrize(
        ('raw', 'keyword', 'preserves'),
        [
            (_FM_LINT_COMMAND, 'ruff check', True),
            (_ROOT_LINT_COMMAND, 'ruff check', True),
            (_ROOT_TYPE_CHECK_COMMAND, 'pyright', False),
            (_ROOT_TEST_COMMAND, 'pytest', False),
        ],
        ids=['fm-lint', 'root-lint', 'root-type-check', 'root-test'],
    )
    def test_real_config_corpus_keeps_its_exact_disposition(self, raw, keyword, preserves):
        """No command in this repo's configs changes disposition — the tightening
        is purely additive capability for the sibling spelling task 3218 predicts.
        """
        prefix, tail = split_chain_tail(raw, keyword)
        assert bool(tail) is preserves
        assert prefix + tail == raw
        if not preserves:
            assert prefix == raw, 'a REJECT returns the whole untouched original'


class TestHasUnpreservedChainClauses:
    """has_unpreserved_chain_clauses(raw, tail) — the DIAGNOSTIC-ONLY predicate.

    Task 3218 part 2b. ``split_chain_tail`` returns ``(raw, '')`` for BOTH
    "single-segment, nothing to preserve" and "multi-segment, gate rejected",
    so a caller cannot tell them apart and a dropped clause is invisible.
    This predicate distinguishes them, and gates a log line — nothing else.
    It deliberately feeds no control-flow decision, which is what makes
    best-effort acceptable: a miss on an exotic spelling costs a missing log
    record, never a behaviour change.
    """

    @pytest.mark.parametrize(
        ('raw', 'tail'),
        [
            (_FM_LINT_COMMAND, '&& python3 fused-memory/scripts/check_x.py fused-memory/tests'),
            ('ruff check src/ && python3 y.py', '&& python3 y.py'),
        ],
        ids=['fm-lint-chain', 'two-clause-chain'],
    )
    def test_false_whenever_a_tail_was_preserved(self, raw, tail):
        """Nothing was dropped — even though *raw* is plainly a multi-clause chain."""
        assert has_unpreserved_chain_clauses(raw, tail) is False

    @pytest.mark.parametrize(
        'raw', ['ruff check src/ --select E', ''], ids=['single-clause', 'empty'],
    )
    def test_false_for_a_single_clause_command(self, raw):
        """A REJECT with nothing to preserve must NOT be reported as a drop.

        This is the discrimination the predicate exists for: without it every
        unchained command would log a spurious "clauses dropped" record.
        """
        assert has_unpreserved_chain_clauses(raw, '') is False

    def test_true_for_a_gate_rejected_and_chain(self):
        """The root type-check fan-out — five clauses dropped, silently, today."""
        assert has_unpreserved_chain_clauses(_ROOT_TYPE_CHECK_COMMAND, '') is True

    def test_true_for_the_unspaced_and_form(self):
        """`a&&b` shlex-splits to one token, so token equality misses it.

        Caught by the ``split_top_level_and`` disjunct, which is quote-aware
        and does not care about whitespace around the operator.
        """
        assert has_unpreserved_chain_clauses('a&&b', '') is True
        assert shlex.split('a&&b') == ['a&&b'], 'premise: token equality cannot see this'

    @pytest.mark.parametrize(
        'raw',
        [
            'ruff check src/ || echo fail',
            'ruff check src/ ; echo done',
            'ruff check src/ | tee log',
        ],
        ids=['or', 'semicolon', 'pipe'],
    )
    def test_true_for_non_and_chain_operators(self, raw):
        """Caught by ``_CHAIN_OPERATOR_TOKENS``, which — unlike the gate's
        narrower ``_NON_AND_CHAIN_TOKENS`` — deliberately includes `&&` too.
        """
        assert has_unpreserved_chain_clauses(raw, '') is True

    def test_true_for_an_unbalanced_quote(self):
        """Undecodable: log loudly rather than stay silent.

        The predicate only gates a log record, so the loud direction is free.
        """
        assert has_unpreserved_chain_clauses('ruff check "x && y.py', '') is True

    @pytest.mark.parametrize(
        ('raw', 'keyword'),
        [
            (_ROOT_TYPE_CHECK_COMMAND, 'pyright'),
            (_ROOT_TEST_COMMAND, 'pytest'),
            (_SIBLING_CHECKER_TEST_COMMAND, 'pytest'),
            (_SIBLING_CHECKER_TEST_COMMAND_UNNAMED, 'pytest'),
            (_FM_LINT_COMMAND, 'ruff check'),
            ('ruff check src/ --select E', 'ruff check'),
        ],
        ids=[
            'root-type-check-rejected',
            'root-test-rejected',
            'pytest-named-sibling-rejected',
            'pytest-unnamed-sibling-rejected',
            'fm-lint-accepted',
            'single-clause',
        ],
    )
    def test_agrees_with_the_gate_on_the_real_corpus(self, raw, keyword):
        """Composed against ``split_chain_tail``: True exactly when the gate
        rejected something that HAD clauses to drop.
        """
        prefix, tail = split_chain_tail(raw, keyword)
        dropped = has_unpreserved_chain_clauses(prefix, tail)
        expected = not tail and len(split_top_level_and(raw)) > 1
        assert dropped is expected


# ---------------------------------------------------------------------------
# What a gate REJECT actually dropped — the clauses, and whether they re-invoke
# ---------------------------------------------------------------------------


class TestDescribeDroppedClauses:
    """describe_dropped_clauses(raw, retained, keyword) — the DIAGNOSTIC-ONLY companion.

    ``has_unpreserved_chain_clauses`` answers "was anything dropped at all";
    this answers "WHAT was dropped, and is it the same tool again". Both are
    pure, which is what keeps this module logging-free — the caller
    (``verify_plan.log_dropped_chain_clauses``) turns the answer into a record.

    Two properties are pinned here because the first version of that record
    got both wrong (task 3218, review findings 1 and 2):

    * the COUNT is the top-level `&&` SEGMENT DELTA across *retained*, not
      ``len(split_top_level_and(raw)) - 1``. The caller's truncation point is
      ``head[: idx + len(keyword)]``, which for every ``cd X && <tool>`` config
      in this repo retains segments 0 AND 1 — so counting every clause in the
      whole original over-reports by one. Nor can it be the re-split of the
      dropped TEXT ``raw[len(retained):]``: *retained* normally ends
      MID-segment (``'uv run pytest'`` of ``'uv run pytest tests/ && ...'``),
      so the leftover ``tests/`` would be counted as a second clause when it is
      a truncated ARGUMENT, not a clause;
    * the FAN-OUT flag comes from the dropped clauses themselves, via the same
      ``_segment_invokes_tool`` predicate gate condition 7 uses. Keying it on
      ``keyword == 'pytest'`` instead mislabelled this repo's own root
      ``test_command`` — a pure pytest fan-out with no sibling checker anywhere
      — as a dropped sibling check.
    """

    def test_root_type_check_fan_out(self):
        """The live root ``type_check_command``: 6 segments, 2 retained, 4 dropped.

        The record used to say 5 — every clause in the original — because the
        keyword sits in segment 1, not segment 0.
        """
        dropped, fan_out = describe_dropped_clauses(
            _ROOT_TYPE_CHECK_COMMAND, 'cd fused-memory && npx pyright', 'pyright',
        )
        assert len(dropped) == 4
        assert fan_out is True
        assert dropped[0] == 'cd ../orchestrator'
        assert dropped[-1] == 'npx pyright'

    def test_root_test_command_fan_out(self):
        """The live root ``test_command``: 16 segments, 2 retained, 14 dropped.

        This is the regression case for review finding 2. Every dropped
        ``uv run pytest tests/ --timeout=300`` clause invokes pytest at an
        argv head, so this is a SAME-TOOL FAN-OUT — never a sibling check —
        and it is the highest-frequency pytest-slot drop in this repo.
        """
        dropped, fan_out = describe_dropped_clauses(
            _ROOT_TEST_COMMAND, 'cd shared && uv run pytest', 'pytest',
        )
        assert len(dropped) == 14
        assert fan_out is True

    @pytest.mark.parametrize(
        ('raw', 'expected_clause'),
        [
            (_SIBLING_CHECKER_TEST_COMMAND, 'python3 scripts/check_pytest_markers.py tests'),
            (_SIBLING_CHECKER_TEST_COMMAND_UNNAMED, 'python3 scripts/check_markers.py tests'),
        ],
        ids=['named-sibling', 'unnamed-sibling'],
    )
    def test_pytest_sibling_checker_is_one_clause_and_not_a_fan_out(self, raw, expected_clause):
        """*retained* ends MID-segment-0 here — ``'uv run pytest'`` drops the
        ``tests/`` argument — so the count MUST come from the segment delta.
        Re-splitting ``raw[len(retained):]`` would report 2.
        """
        dropped, fan_out = describe_dropped_clauses(raw, 'uv run pytest', 'pytest')
        assert dropped == (expected_clause,)
        assert fan_out is False

    def test_nothing_dropped_for_an_unchained_command(self):
        """A single-clause command has no clause past the truncation point.

        ``'src/'`` is a leftover ARGUMENT, not a dropped clause: the fallback
        below must not mistake it for one.
        """
        assert describe_dropped_clauses('ruff check src/', 'ruff check', 'ruff check') == ((), False)

    @pytest.mark.parametrize(
        ('raw', 'retained', 'keyword'),
        [
            ('ruff check src/ || python3 x.py', 'ruff check', 'ruff check'),
            ('npx pyright ; python3 y.py', 'npx pyright', 'pyright'),
            ('uv run pytest tests/ | tee log', 'uv run pytest', 'pytest'),
        ],
        ids=['or', 'semicolon', 'pipe'],
    )
    def test_non_and_operator_chain_reports_one_clause(self, raw, retained, keyword):
        """The segment view cannot see a `||` / `;` / `|` chain — one `&&` segment.

        Without the remainder fallback the record would read "dropped 0
        trailing chain clause(s)" on a path only reached because
        ``has_unpreserved_chain_clauses`` reported a REAL drop — a record that
        contradicts itself.
        """
        dropped, fan_out = describe_dropped_clauses(raw, retained, keyword)
        assert len(dropped) == 1
        assert fan_out is False

    def test_undecodable_dropped_clause_counts_as_a_fan_out(self):
        """Mirrors ``_segment_invokes_tool``'s ValueError->True.

        An unbalanced quote makes the clause undecodable, so it cannot be
        shown NOT to re-invoke the tool. Treat it as the quiet fan-out case
        rather than shouting a sibling-check claim that may be false.
        """
        dropped, fan_out = describe_dropped_clauses(
            'ruff check src/ && python3 "x.py', 'ruff check', 'ruff check',
        )
        assert dropped == ('python3 "x.py',)
        assert fan_out is True
