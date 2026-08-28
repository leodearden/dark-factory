"""Tests for the fileless-misfile soft scope signals.

These signals exist for the ~50% of real misfiles that declare NO files at
all, which every existing path-scope guard (all of which classify by
DECLARED file paths) is structurally blind to.

Test layout (mirrors the plan):
- step-1: the TITLE-CONVENTION signal (a) and its alias derivation
- step-3: the ABSOLUTE-FOREIGN-ROOT signal (b1)
- step-5: the WEAK project-NAME signal (b2), the aggregator, the enforce flag
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fused_memory.middleware.path_scope_guard import find_paths
from fused_memory.middleware.project_prefix_registry import ProjectPrefixRegistry
from fused_memory.middleware.soft_scope_signals import (
    SoftScopeFinding,
    SoftScopeSignal,
    collect_soft_scope_signals,
    find_absolute_foreign_roots,
    find_foreign_project_names,
    find_title_project_prefix,
    project_name_aliases,
    soft_scope_enforced,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mkproj(parent: Path, name: str, dirs: list[str]) -> Path:
    root = parent / name
    root.mkdir()
    for d in dirs:
        (root / d).mkdir()
    return root


def _two_project_registry(tmp_path: Path) -> ProjectPrefixRegistry:
    """Reify (crates/, gui/) + dark-factory (fused-memory/, orchestrator/).

    ``from_roots`` derives project_ids by basename canonicalisation, so the
    registry keys are ``reify`` and ``dark_factory`` while the roots keep
    their on-disk ``dark-factory`` spelling — which is exactly the gap
    :func:`project_name_aliases` bridges.
    """
    a = _mkproj(tmp_path, 'reify', ['crates', 'gui'])
    b = _mkproj(tmp_path, 'dark-factory', ['fused-memory', 'orchestrator'])
    return ProjectPrefixRegistry.from_roots([str(a), str(b)])


# ---------------------------------------------------------------------------
# step-1: alias derivation
# ---------------------------------------------------------------------------


class TestProjectNameAliases:
    def test_yields_underscore_and_hyphen_spellings(self):
        aliases = project_name_aliases('dark_factory', '/home/leo/src/dark-factory')
        assert 'dark_factory' in aliases
        assert 'dark-factory' in aliases

    def test_all_aliases_are_lowercased(self):
        aliases = project_name_aliases('Dark_Factory', '/home/leo/src/Dark-Factory')
        assert all(a == a.lower() for a in aliases)
        assert 'dark_factory' in aliases
        assert 'dark-factory' in aliases

    def test_deduplicated_and_stable(self):
        # project_id already equals the root basename -> one alias, not three.
        aliases = project_name_aliases('reify', '/home/leo/src/reify')
        assert list(aliases) == ['reify']
        assert list(project_name_aliases('reify', '/home/leo/src/reify')) == list(
            aliases
        )

    def test_no_root_still_yields_id_spellings(self):
        aliases = project_name_aliases('dark_factory', None)
        assert 'dark_factory' in aliases
        assert 'dark-factory' in aliases

    def test_trailing_slash_on_root_does_not_leak_empty_alias(self):
        aliases = project_name_aliases('dark_factory', '/home/leo/src/dark-factory/')
        assert '' not in aliases
        assert 'dark-factory' in aliases


# ---------------------------------------------------------------------------
# step-1: signal (a) — the leading "<project>:" title convention
# ---------------------------------------------------------------------------


class TestFindTitleProjectPrefix:
    def test_canonical_fileless_case_reify_5575(self, tmp_path):
        registry = _two_project_registry(tmp_path)
        signal = find_title_project_prefix(
            'dark-factory: wire the recurring timer + request consumer for X',
            project_id='reify',
            registry=registry,
        )
        assert isinstance(signal, SoftScopeSignal)
        assert signal.kind == 'title_project_prefix'
        # The IMPLICATED foreign project, not the filer.
        assert signal.project_id == 'dark_factory'
        assert signal.strength == 'strong'
        assert signal.evidence == 'dark-factory:'

    def test_cross_repo_parenthetical_shape_reify_4851(self, tmp_path):
        registry = _two_project_registry(tmp_path)
        signal = find_title_project_prefix(
            'CROSS-REPO (dark-factory merge_queue): teach the queue to ...',
            project_id='reify',
            registry=registry,
        )
        assert signal is not None
        assert signal.kind == 'title_project_prefix'
        assert signal.project_id == 'dark_factory'
        assert signal.strength == 'strong'
        assert signal.evidence == 'CROSS-REPO (dark-factory merge_queue):'

    def test_third_measured_true_positive_reify_5638(self, tmp_path):
        registry = _two_project_registry(tmp_path)
        signal = find_title_project_prefix(
            'dark-factory: give the _merge-verify step a real signal',
            project_id='reify',
            registry=registry,
        )
        assert signal is not None
        assert signal.project_id == 'dark_factory'
        assert signal.evidence == 'dark-factory:'

    def test_self_reference_is_not_a_misfile(self, tmp_path):
        """A reify-titled task filed under reify announces its own scope."""
        registry = _two_project_registry(tmp_path)
        assert (
            find_title_project_prefix(
                'reify: wire the recurring timer for X',
                project_id='reify',
                registry=registry,
            )
            is None
        )

    def test_self_reference_via_hyphen_alias_is_not_a_misfile(self, tmp_path):
        registry = _two_project_registry(tmp_path)
        assert (
            find_title_project_prefix(
                'dark-factory: wire the recurring timer for X',
                project_id='dark_factory',
                registry=registry,
            )
            is None
        )

    def test_mid_title_mention_is_not_the_announcement_shape(self, tmp_path):
        registry = _two_project_registry(tmp_path)
        assert (
            find_title_project_prefix(
                'teach dark-factory: about X',
                project_id='reify',
                registry=registry,
            )
            is None
        )

    def test_no_colon_at_all_does_not_match(self, tmp_path):
        registry = _two_project_registry(tmp_path)
        assert (
            find_title_project_prefix(
                'dark-factory wire the recurring timer for X',
                project_id='reify',
                registry=registry,
            )
            is None
        )

    def test_colon_beyond_the_forty_char_window_does_not_match(self, tmp_path):
        registry = _two_project_registry(tmp_path)
        title = 'dark-factory ' + ('a' * 60) + ': tail'
        assert (
            find_title_project_prefix(title, project_id='reify', registry=registry)
            is None
        )

    def test_leading_whitespace_is_tolerated(self, tmp_path):
        registry = _two_project_registry(tmp_path)
        signal = find_title_project_prefix(
            '   dark-factory: wire the timer',
            project_id='reify',
            registry=registry,
        )
        assert signal is not None
        assert signal.project_id == 'dark_factory'

    def test_empty_title_returns_none(self, tmp_path):
        registry = _two_project_registry(tmp_path)
        assert find_title_project_prefix('', 'reify', registry) is None
        assert find_title_project_prefix('   ', 'reify', registry) is None
        assert find_title_project_prefix(None, 'reify', registry) is None

    def test_empty_registry_returns_none(self):
        empty = ProjectPrefixRegistry.from_roots([])
        assert (
            find_title_project_prefix(
                'dark-factory: wire the timer', 'reify', empty
            )
            is None
        )
        assert (
            find_title_project_prefix('dark-factory: wire the timer', 'reify', None)
            is None
        )

    def test_known_false_positive_stays_visible(self, tmp_path):
        """PINNED: the measured 1-of-4 false positive (75% precision, n=4).

        'Reify first census: human gate — decide X' is a dark_factory task
        ABOUT reify, not a misfiled reify task — but it wears the
        announcement shape, so the signal fires.  Asserted here so the
        precision cost stays visible in the suite instead of implied clean;
        it is exactly why this is a trigger for a confirmation step and
        never a stamp.
        """
        registry = _two_project_registry(tmp_path)
        signal = find_title_project_prefix(
            'Reify first census: human gate — decide X',
            project_id='dark_factory',
            registry=registry,
        )
        assert signal is not None
        assert signal.project_id == 'reify'
        assert signal.strength == 'strong'
        assert signal.evidence == 'Reify first census:'


# ---------------------------------------------------------------------------
# step-3: signal (b1) — an ABSOLUTE path under a foreign project root
# ---------------------------------------------------------------------------


class TestFindPathsIsBlindToAbsolutePaths:
    """PREMISE PIN: the structural gap signal (b1) exists to close.

    ``find_paths``' left-boundary class (path_scope_guard.py, the
    ``[^A-Za-z0-9_\\-/.]`` class) excludes ``/`` deliberately, so that
    ``vendor/corpus/expr.txt`` does not match the bare prefix ``corpus/``.
    Absolute paths are collateral damage of that exclusion: a prefix
    preceded by ``/`` can never match.  Asserted here rather than asserted
    ABOUT, so the day someone widens that class this pin says so.
    """

    def test_absolute_spelling_is_invisible(self, tmp_path):
        registry = _two_project_registry(tmp_path)
        df_root = registry.root_for_project('dark_factory')
        assert (
            find_paths(
                f'Modify {df_root}/orchestrator/scheduler.py',
                registry.all_prefixes(),
            )
            == []
        )

    def test_repo_relative_spelling_is_visible(self, tmp_path):
        registry = _two_project_registry(tmp_path)
        assert find_paths(
            'Modify orchestrator/scheduler.py', registry.all_prefixes()
        ) == ['orchestrator/']


class TestFindAbsoluteForeignRoots:
    def test_fires_on_foreign_root_with_path_segment(self, tmp_path):
        registry = _two_project_registry(tmp_path)
        df_root = registry.root_for_project('dark_factory')
        signals = find_absolute_foreign_roots(
            f'ALL of the asked work is in {df_root}/orchestrator/scheduler.py',
            project_id='reify',
            registry=registry,
        )
        assert len(signals) == 1
        assert signals[0].kind == 'absolute_foreign_root'
        assert signals[0].project_id == 'dark_factory'
        assert signals[0].strength == 'strong'
        assert signals[0].evidence == df_root

    def test_fires_on_bare_root_with_no_trailing_segment(self, tmp_path):
        """An absolute root ALONE is already unambiguous ownership evidence.

        Unlike a bare relative prefix (which ``_RIGHT_CONTEXT`` deliberately
        refuses without a following path segment), nothing else spells a
        project's absolute root.
        """
        registry = _two_project_registry(tmp_path)
        df_root = registry.root_for_project('dark_factory')
        signals = find_absolute_foreign_roots(
            f'all of this lives under {df_root}',
            project_id='reify',
            registry=registry,
        )
        assert [s.project_id for s in signals] == ['dark_factory']

    def test_sibling_root_near_miss_does_not_fire(self, tmp_path):
        """Component boundary — the rule ``_owner_for_absolute_path`` applies."""
        registry = _two_project_registry(tmp_path)
        df_root = registry.root_for_project('dark_factory')
        assert (
            find_absolute_foreign_roots(
                f'see {df_root}-old/x.py', project_id='reify', registry=registry
            )
            == []
        )
        assert (
            find_absolute_foreign_roots(
                f'see {df_root}ish', project_id='reify', registry=registry
            )
            == []
        )

    def test_dotted_suffix_near_miss_does_not_fire(self, tmp_path):
        registry = _two_project_registry(tmp_path)
        df_root = registry.root_for_project('dark_factory')
        assert (
            find_absolute_foreign_roots(
                f'see {df_root}.bak/x.py', project_id='reify', registry=registry
            )
            == []
        )

    def test_filing_projects_own_root_does_not_fire(self, tmp_path):
        registry = _two_project_registry(tmp_path)
        reify_root = registry.root_for_project('reify')
        assert (
            find_absolute_foreign_roots(
                f'work in {reify_root}/crates/foo.rs',
                project_id='reify',
                registry=registry,
            )
            == []
        )

    def test_repeated_mentions_deduplicate_to_one_signal(self, tmp_path):
        registry = _two_project_registry(tmp_path)
        df_root = registry.root_for_project('dark_factory')
        signals = find_absolute_foreign_roots(
            f'{df_root}/a.py and {df_root}/b.py and also {df_root}',
            project_id='reify',
            registry=registry,
        )
        assert len(signals) == 1
        assert signals[0].project_id == 'dark_factory'

    def test_empty_text_and_empty_registry_return_empty(self, tmp_path):
        registry = _two_project_registry(tmp_path)
        assert find_absolute_foreign_roots('', 'reify', registry) == []
        assert find_absolute_foreign_roots(None, 'reify', registry) == []
        empty = ProjectPrefixRegistry.from_roots([])
        assert find_absolute_foreign_roots('/home/leo/src/x/y.py', 'reify', empty) == []
        assert find_absolute_foreign_roots('/home/leo/src/x/y.py', 'reify', None) == []


# ---------------------------------------------------------------------------
# step-5: signal (b2) — the WEAK bare project NAME in prose
# ---------------------------------------------------------------------------


class TestFindForeignProjectNames:
    def test_fires_on_hyphen_spelling(self, tmp_path):
        registry = _two_project_registry(tmp_path)
        signals = find_foreign_project_names(
            'ALL of the asked work is dark-factory-side',
            project_id='reify',
            registry=registry,
        )
        assert len(signals) == 1
        assert signals[0].kind == 'foreign_project_name'
        assert signals[0].project_id == 'dark_factory'
        assert signals[0].strength == 'weak'

    def test_fires_on_underscore_spelling(self, tmp_path):
        registry = _two_project_registry(tmp_path)
        signals = find_foreign_project_names(
            'dark_factory owns this', project_id='reify', registry=registry
        )
        assert [s.project_id for s in signals] == ['dark_factory']
        assert signals[0].strength == 'weak'

    def test_filing_projects_own_name_does_not_fire(self, tmp_path):
        registry = _two_project_registry(tmp_path)
        assert (
            find_foreign_project_names(
                'this is reify work, all of it',
                project_id='reify',
                registry=registry,
            )
            == []
        )

    def test_name_embedded_in_a_longer_word_does_not_fire(self, tmp_path):
        registry = _two_project_registry(tmp_path)
        assert (
            find_foreign_project_names(
                'the dark-factoryish shim and the darkfactory typo',
                project_id='reify',
                registry=registry,
            )
            == []
        )

    def test_repeated_mentions_deduplicate_to_one_signal(self, tmp_path):
        registry = _two_project_registry(tmp_path)
        signals = find_foreign_project_names(
            'dark-factory here, dark_factory there, dark-factory everywhere',
            project_id='reify',
            registry=registry,
        )
        assert len(signals) == 1

    def test_empty_text_and_empty_registry_return_empty(self, tmp_path):
        registry = _two_project_registry(tmp_path)
        assert find_foreign_project_names('', 'reify', registry) == []
        assert find_foreign_project_names(None, 'reify', registry) == []
        empty = ProjectPrefixRegistry.from_roots([])
        assert find_foreign_project_names('dark-factory owns this', 'reify', empty) == []
        assert find_foreign_project_names('dark-factory owns this', 'reify', None) == []


# ---------------------------------------------------------------------------
# step-5: the aggregator
# ---------------------------------------------------------------------------


class TestCollectSoftScopeSignals:
    def test_weak_only_finding_does_not_trigger_adjudication(self, tmp_path):
        """LOAD-BEARING: the bare-name rule is measured at 20.6% / 3.2%.

        Triggering a ~$0.105 LLM confirmation call at that rate is not
        affordable, so a weak signal is CONTEXT — carried in ``.signals``
        and in the census line — and never a trigger on its own.
        """
        registry = _two_project_registry(tmp_path)
        finding = collect_soft_scope_signals(
            title='wire the recurring timer for X',
            description='this is dark-factory-adjacent but the work is here',
            details='',
            project_id='reify',
            registry=registry,
        )
        assert isinstance(finding, SoftScopeFinding)
        assert finding.signals
        assert all(s.strength == 'weak' for s in finding.signals)
        assert finding.should_adjudicate is False

    def test_title_convention_alone_triggers_adjudication(self, tmp_path):
        registry = _two_project_registry(tmp_path)
        finding = collect_soft_scope_signals(
            title='dark-factory: wire the recurring timer',
            description='',
            details='',
            project_id='reify',
            registry=registry,
        )
        assert finding.should_adjudicate is True
        assert finding.suggested_project == 'dark_factory'
        assert any(s.kind == 'title_project_prefix' for s in finding.signals)

    def test_absolute_foreign_root_alone_triggers_adjudication(self, tmp_path):
        registry = _two_project_registry(tmp_path)
        df_root = registry.root_for_project('dark_factory')
        finding = collect_soft_scope_signals(
            title='wire the recurring timer',
            description='',
            details=f'the file is {df_root}/orchestrator/scheduler.py',
            project_id='reify',
            registry=registry,
        )
        assert finding.should_adjudicate is True
        assert finding.suggested_project == 'dark_factory'
        assert any(s.kind == 'absolute_foreign_root' for s in finding.signals)

    def test_title_rule_is_anchored_to_the_title_alone(self, tmp_path):
        """The rule is start-anchored, so it must not see a joined blob."""
        registry = _two_project_registry(tmp_path)
        finding = collect_soft_scope_signals(
            title='wire the recurring timer',
            description='dark-factory: wire the recurring timer',
            details='',
            project_id='reify',
            registry=registry,
        )
        assert not any(s.kind == 'title_project_prefix' for s in finding.signals)

    def test_strong_signals_are_ordered_before_weak_ones(self, tmp_path):
        registry = _two_project_registry(tmp_path)
        finding = collect_soft_scope_signals(
            title='dark-factory: wire the recurring timer',
            description='all of this is dark-factory work',
            details='',
            project_id='reify',
            registry=registry,
        )
        strengths = [s.strength for s in finding.signals]
        assert strengths == sorted(strengths, key=lambda s: s != 'strong')
        assert strengths[0] == 'strong'

    def test_disagreeing_signals_yield_no_suggested_project(self, tmp_path):
        """Mirrors _aggregate_owner_mismatches' multi-owner rule."""
        a = _mkproj(tmp_path, 'reify', ['crates'])
        b = _mkproj(tmp_path, 'dark-factory', ['orchestrator'])
        c = _mkproj(tmp_path, 'cockpit', ['ui'])
        registry = ProjectPrefixRegistry.from_roots([str(a), str(b), str(c)])
        finding = collect_soft_scope_signals(
            title='dark-factory: wire the timer',
            description=f'but see also {registry.root_for_project("cockpit")}/ui/x.ts',
            details='',
            project_id='reify',
            registry=registry,
        )
        assert finding.should_adjudicate is True
        assert finding.suggested_project is None

    def test_weak_disagreement_does_not_dilute_a_strong_suggestion(self, tmp_path):
        """LOAD-BEARING: agreement is computed over STRONG signals ONLY.

        The module's contract is that weak signals are "context and census
        detail only" — ``should_adjudicate`` deliberately ignores them. This
        pins the same rule for ``suggested_project``, which is the ONLY
        field of the ``possible_scope_mismatch`` stamp any landed consumer
        reads (``orchestrator/src/orchestrator/cross_repo_gate.py::
        _resolve_owner``) and is also handed to the paid confirmation step.

        Shape: ONE strong signal names ``dark_factory`` unambiguously, and a
        bare mention of ``cockpit`` in prose fires the weak name rule. Were
        agreement computed over ALL signals, that 20.6%-fire-rate rule would
        null out a suggestion unambiguous strong evidence had established —
        measured at 10 of 111 strong firings (9.0%) across the dark_factory
        corpus (tasks 1542, 1543, 1858, 2101, 3168, 3641, 3642, 4264, 4505,
        4710).
        """
        a = _mkproj(tmp_path, 'reify', ['crates'])
        b = _mkproj(tmp_path, 'dark-factory', ['orchestrator'])
        c = _mkproj(tmp_path, 'cockpit', ['ui'])
        registry = ProjectPrefixRegistry.from_roots([str(a), str(b), str(c)])
        finding = collect_soft_scope_signals(
            title='dark-factory: wire the timer',
            description='similar in spirit to what cockpit does for its panes',
            details='',
            project_id='reify',
            registry=registry,
        )

        strong = [s for s in finding.signals if s.strength == 'strong']
        weak = [s for s in finding.signals if s.strength == 'weak']
        # Premise checks: exactly the shape described above, so a green
        # result cannot come from the weak rule silently failing to fire.
        assert {s.project_id for s in strong} == {'dark_factory'}
        assert 'cockpit' in {s.project_id for s in weak}

        assert finding.should_adjudicate is True
        assert finding.suggested_project == 'dark_factory'

    def test_empty_finding(self, tmp_path):
        registry = _two_project_registry(tmp_path)
        finding = collect_soft_scope_signals(
            title='wire the recurring timer for X',
            description='ordinary local work',
            details='',
            project_id='reify',
            registry=registry,
        )
        assert finding.signals == ()
        assert finding.should_adjudicate is False
        assert finding.suggested_project is None

    def test_canonical_reify_5575_shape_collects_all_three_kinds(self, tmp_path):
        """The fileless case this whole task exists to serve."""
        registry = _two_project_registry(tmp_path)
        df_root = registry.root_for_project('dark_factory')
        finding = collect_soft_scope_signals(
            title='dark-factory: wire the recurring timer + request consumer',
            description=(
                f'ALL of the asked work is in {df_root}. '
                'ASKED WORK (dark-factory side): the timer and the consumer.'
            ),
            details='',
            project_id='reify',
            registry=registry,
        )
        kinds = {s.kind for s in finding.signals}
        assert kinds == {
            'title_project_prefix',
            'absolute_foreign_root',
            'foreign_project_name',
        }
        assert finding.should_adjudicate is True
        assert finding.suggested_project == 'dark_factory'


# ---------------------------------------------------------------------------
# step-5: the enforce flag
# ---------------------------------------------------------------------------


class TestSoftScopeEnforced:
    @pytest.mark.parametrize(
        'value', ['1', 'true', 'TRUE', 'True', 'yes', 'YES', 'on', 'ON', '  on  ']
    )
    def test_truthy_values_enable_enforcement(self, monkeypatch, value):
        monkeypatch.setenv('FUSED_SOFT_SCOPE_ENFORCE', value)
        assert soft_scope_enforced() is True

    @pytest.mark.parametrize('value', ['', '   ', '0', 'false', 'no', 'off', 'maybe'])
    def test_unrecognised_values_stay_warn_only(self, monkeypatch, value):
        monkeypatch.setenv('FUSED_SOFT_SCOPE_ENFORCE', value)
        assert soft_scope_enforced() is False

    def test_unset_stays_warn_only(self, monkeypatch):
        monkeypatch.delenv('FUSED_SOFT_SCOPE_ENFORCE', raising=False)
        assert soft_scope_enforced() is False


class TestAbsoluteForeignRootSentenceBoundary:
    """The trailing-'.' distinction, pinned in both directions.

    Sentence-final punctuation after a cited root is the COMMON prose
    spelling of the fileless case this signal exists to catch, so it must
    fire; a dotted directory suffix names a different directory, so it must
    not.  Same character, opposite verdicts — hence the two-character
    lookahead in ``_ROOT_SUFFIX_RE`` rather than a flat path-name class.
    """

    def test_sentence_final_period_still_fires(self, tmp_path):
        registry = _two_project_registry(tmp_path)
        df_root = registry.root_for_project('dark_factory')
        signals = find_absolute_foreign_roots(
            f'ALL of the asked work is in {df_root}. Nothing here.',
            project_id='reify',
            registry=registry,
        )
        assert [s.project_id for s in signals] == ['dark_factory']

    def test_trailing_comma_and_end_of_text_still_fire(self, tmp_path):
        registry = _two_project_registry(tmp_path)
        df_root = registry.root_for_project('dark_factory')
        assert find_absolute_foreign_roots(
            f'see {df_root}, then stop', 'reify', registry
        )
        assert find_absolute_foreign_roots(f'see {df_root}', 'reify', registry)
