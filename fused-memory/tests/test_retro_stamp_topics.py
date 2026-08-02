"""Tests for scripts/retro_stamp_topics.py — PRD leaf θ (task 3201).

The bounded retro topic/canonical stamping sweep.  Loaded via importlib so
the script (``scripts/`` is not a package and is not on PYTHONPATH) can be
tested without sys.path pollution — the same idiom as
``test_sweep_orphan_flag_markers.py`` / ``test_cleanup_count_snapshots.py``.

Every derivation function in the script is pure, so almost everything here
is a plain unit test; the single I/O boundary (an injected
``memory_service``) is exercised with ``AsyncMock``.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

from fused_memory import topic_slug as topic_slug_module

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'retro_stamp_topics.py'


def _load_module() -> types.ModuleType:
    """Load retro_stamp_topics.py from its file path.

    The module is registered in sys.modules under its name so that
    reflection-based decorators work correctly.
    """
    mod_name = 'retro_stamp_topics'
    spec = importlib.util.spec_from_file_location(mod_name, SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {SCRIPT_PATH}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return module


_mod = _load_module()


# ===========================================================================
# derive_topic_slug — the fold from a raw value to ε's slug shape
# ===========================================================================

class TestDeriveTopicSlug:
    """``derive_topic_slug(value) -> str | None``.

    Folds a raw topic value into the shape
    :mod:`fused_memory.topic_slug` defines, or returns ``None`` when no
    honest fold exists.  Never guesses: a value that cannot conform is
    reported, not repaired into something plausible.
    """

    def test_conforming_slug_round_trips_unchanged(self):
        """A value already in ε's shape is returned byte-identical.

        This is what makes the sweep idempotent at the derivation layer:
        run two folds the target topic to itself, so ``compute_patch``
        sees no change to write.
        """
        assert _mod.derive_topic_slug('docs-prd-landing') == 'docs-prd-landing'
        assert _mod.derive_topic_slug('a') == 'a'
        assert _mod.derive_topic_slug('x1-2y') == 'x1-2y'

    @pytest.mark.parametrize(
        ('raw', 'expected'),
        [
            # The single live dark_factory `canonical: true` record's topic.
            (
                'eval_worktree_plan_tools_missing',
                'eval-worktree-plan-tools-missing',
            ),
            # One of the five live reify `canonical: true` topics.
            (
                'merge_request_bare_task_id_branch_arg',
                'merge-request-bare-task-id-branch-arg',
            ),
        ],
    )
    def test_measured_live_snake_case_topics_fold_to_hyphens(
        self, raw: str, expected: str
    ):
        """The two measured snake_case live topics are exactly θ's job.

        ``eval_worktree_plan_tools_missing`` is the snake_case twin of the
        seeded cluster id ``eval-worktree-plan-tools-missing``; ε's
        enforcement note names this normalization as the precondition for
        flipping ``memory_metadata.enforce``.  These are not invented
        examples — they are the values the live corpus carries.
        """
        assert _mod.derive_topic_slug(raw) == expected

    @pytest.mark.parametrize(
        ('raw', 'expected'),
        [
            ('Docs-PRD-Landing', 'docs-prd-landing'),
            ('  docs-prd-landing  ', 'docs-prd-landing'),
            ('\tDocs PRD Landing\n', 'docs-prd-landing'),
            ('docs/prd/landing', 'docs-prd-landing'),
            ('docs -- prd  landing', 'docs-prd-landing'),
            ('--docs-prd-landing--', 'docs-prd-landing'),
            ('Docs_PRD__Landing!!!', 'docs-prd-landing'),
        ],
    )
    def test_case_whitespace_and_punctuation_runs_collapse(
        self, raw: str, expected: str
    ):
        """Case folds, runs collapse, edges strip — never a doubled hyphen."""
        got = _mod.derive_topic_slug(raw)
        assert got == expected
        assert '--' not in got
        assert not got.startswith('-')
        assert not got.endswith('-')

    @pytest.mark.parametrize(
        ('raw', 'why'),
        [
            ('', 'empty'),
            ('   ', 'whitespace only'),
            ('\n\t', 'whitespace only, non-space'),
            ('!!!', 'all punctuation — nothing survives the fold'),
            ('---', 'all separators'),
            ('a' * 120, 'exceeds TOPIC_SLUG_MAX_LEN'),
            (None, 'not a string'),
            (12345, 'not a string'),
            (['docs-prd-landing'], 'not a string'),
        ],
    )
    def test_unfoldable_values_return_none_rather_than_a_guess(
        self, raw: object, why: str
    ):
        """No honest fold exists -> ``None``, so the caller can report it.

        Truncating the over-long value or inventing a slug for ``'!!!'``
        would silently file a record under a topic no human chose.  The
        loud-over-silent read is to refuse and surface it.
        """
        assert _mod.derive_topic_slug(raw) is None, why

    def test_over_length_boundary_is_the_shared_cap_not_a_local_number(self):
        """Exactly at the cap passes; one over fails — via ε's constant."""
        cap = topic_slug_module.TOPIC_SLUG_MAX_LEN
        assert _mod.derive_topic_slug('a' * cap) == 'a' * cap
        assert _mod.derive_topic_slug('a' * (cap + 1)) is None

    @pytest.mark.parametrize(
        'raw',
        [
            'docs-prd-landing',
            'eval_worktree_plan_tools_missing',
            'Docs PRD Landing',
            'docs/prd/landing',
            'a',
            'x1-2y',
            '--docs--prd--landing--',
        ],
    )
    def test_every_non_none_return_satisfies_the_shared_predicate(
        self, raw: str
    ):
        """The output contract: whatever comes back is a valid slug.

        Checked against :func:`fused_memory.topic_slug.is_valid_topic_slug`
        directly (not the script's re-export) so the property holds against
        the normative home even if the re-export were ever broken.
        """
        got = _mod.derive_topic_slug(raw)
        assert got is not None
        assert topic_slug_module.is_valid_topic_slug(got)


class TestTopicSlugNamespaceIsShared:
    """INV-5: the script gets ε's rule by import, never by copy.

    ``tests/test_topic_slug_namespace.py`` pins the same identity for the
    metadata registry and the config schema.  A second copy of the regex or
    the cap anywhere in the tree is a bug; these ``is`` assertions make a
    copy fail mechanically rather than by review.
    """

    def test_predicate_is_the_same_object(self):
        assert _mod.is_valid_topic_slug is topic_slug_module.is_valid_topic_slug

    def test_cap_is_the_same_object(self):
        assert _mod.TOPIC_SLUG_MAX_LEN is topic_slug_module.TOPIC_SLUG_MAX_LEN

    def test_script_does_not_define_its_own_slug_pattern(self):
        """No local ``re.Pattern`` re-expressing the slug shape.

        The fold itself needs a character-class pattern, so this does not
        forbid ``re`` outright — it forbids a *second anchored slug
        validator*, which is the copy that would silently diverge from ε.
        """
        source = SCRIPT_PATH.read_text()
        assert '[a-z0-9]+(?:-[a-z0-9]+)*' not in source
