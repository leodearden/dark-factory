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

from fused_memory.middleware.project_prefix_registry import ProjectPrefixRegistry
from fused_memory.middleware.soft_scope_signals import (
    SoftScopeSignal,
    find_title_project_prefix,
    project_name_aliases,
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
