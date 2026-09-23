"""Tests for escalation/canonical.py — the ONE canonicalisation transform.

This module pins three things that are easy to lose and expensive to lose
silently:

1. The **policy** (`punctuation='separator'` for root causes): punctuation is a
   SEPARATOR, never removable decoration.  69% of the live queue's distinct
   root_cause keys carry a digit adjacent to a separator (task ids, SHA
   prefixes, dates), and those digits ARE the identity-bearing segments — so
   deleting the separator would glue `risk:3184` onto `risk:318:4` and silently
   absorb two distinct incidents into one L2.
2. The **legacy policy** (`punctuation='strip'`) stays reachable and unchanged,
   because `dedupe.compute_content_fingerprint`'s sha256 digests are already
   persisted across the live recon corpus.
3. **Leaf-ness** — the structural property that lets `queue.py` import this
   module at all (`dedupe.py` imports `escalation.queue` at module level, so
   `queue.py` importing `dedupe.py` would be a hard cycle).  Asserted
   BEHAVIOURALLY via a fresh-interpreter subprocess probe, not by grepping the
   source: a substring check is defeated by relative imports,
   ``importlib.import_module`` and line-broken import statements, and is falsely
   tripped by a docstring that merely mentions the word.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

# Real keys drawn from the live queue (data/escalations, root + archive), kept
# verbatim so the idempotence table exercises the shapes that actually occur:
# category-prefixed task ids, hyphenated slugs, dates, and an underscore form.
_REAL_ROOT_CAUSE_KEYS = [
    'infra_issue:4087:dashboard-unit-stale-on-host-setup-host-not-propagated',
    'design-concern:landlock-exec-claude-home-not-narrowed-to-fleet',
    'merge-resource-leak:unregistered-worktree-ledger',
    'scope_violation:3916:orphaned-review-followup-parent-cancelled',
    'dependency_discovered:3898:subsumes-half-of-3244-other-half-more-urgent',
    'review_issues:4104:review-cycles-exhausted-with-blocking-issue',
    'mcp-markup-write-storm-active-leak:dark-factory',
    'risk:3815:bare-file-line-drift-200-sites-in-comments',
    'curator-failure:account-cap-2026-08-05',
    'bad-merge-to-main:65b011ed8c-worktree-reset',
]


class TestCanonicalRootCause:
    """canonical_root_cause() — the separator-policy alias used at the match site."""

    def test_case_whitespace_and_trailing_punctuation_fold(self):
        """PRD boundary row B4: case, doubled space and a trailing '.' all fold."""
        from escalation.canonical import canonical_root_cause

        assert canonical_root_cause('Watcher lease stolen.') == canonical_root_cause(
            'watcher  lease STOLEN'
        )

    def test_edges_are_trimmed(self):
        """Leading/trailing whitespace and punctuation never change the key."""
        from escalation.canonical import canonical_root_cause

        assert canonical_root_cause('  watcher lease stolen  ') == canonical_root_cause(
            'watcher lease stolen'
        )

    def test_punctuation_is_a_separator_not_a_deletion(self):
        """A delimiter-dense key folds onto its space-separated spelling."""
        from escalation.canonical import canonical_root_cause

        assert canonical_root_cause(
            'starvation:2370:persistent-lock-contention'
        ) == canonical_root_cause('Starvation 2370 persistent lock contention')

    def test_distinct_identity_segments_never_glue(self):
        """The CONSERVATIVE direction — this is what separator semantics buys.

        Under DELETION semantics (``a.b`` -> ``ab``) both of these pairs collapse
        to one key, silently merging two distinct incidents into one L2.  Under
        SEPARATOR semantics they stay distinct.  An under-fold leaves a noisy but
        safe duplicate; an over-fold is silent and unrecoverable — hence this pin.
        """
        from escalation.canonical import canonical_root_cause

        assert canonical_root_cause('risk:3184') != canonical_root_cause('risk:318:4')
        assert canonical_root_cause(
            'curator-failure:account-cap-2026-08-05'
        ) != canonical_root_cause('curator-failure:account-cap-2026-0-805')

    @pytest.mark.parametrize('key', _REAL_ROOT_CAUSE_KEYS)
    def test_idempotent_on_real_keys(self, key):
        """f(f(x)) == f(x) for keys drawn from the live queue.

        Load-bearing: both sides of the match are canonicalised at compare time,
        and a stored root_cause may itself already be a canonical form (nothing
        forbids an agent filing one), so a non-idempotent transform would make
        matching depend on how many times a key had been round-tripped.
        """
        from escalation.canonical import canonical_root_cause

        once = canonical_root_cause(key)
        assert canonical_root_cause(once) == once
        assert once  # no real key canonicalises away to nothing

    @pytest.mark.parametrize('text', ['::', '--', '  :  ', '', '   ', '!!! ???'])
    def test_punctuation_only_input_canonicalises_to_empty(self, text):
        """The falsy-key contract the match site's early return depends on.

        `'::'` is non-empty after `.strip()` but carries no identity at all, so
        it must canonicalise to `''` and be refused as a dedup key rather than
        minting a permanently unfoldable L2.
        """
        from escalation.canonical import canonical_root_cause

        assert canonical_root_cause(text) == ''

    def test_unicode_word_characters_survive(self):
        r"""A CJK key stays non-empty — `\w` is Unicode-aware.

        This is what makes the "reject a root_cause that canonicalises to empty"
        rule safe: only a purely punctuation/symbol key is refused, never a
        non-Latin one.
        """
        from escalation.canonical import canonical_root_cause

        assert canonical_root_cause('ロック競合:2370') != ''
        assert canonical_root_cause('Ünïcödé-cause') != ''


class TestCanonicalTextPolicies:
    """canonical_text() — one transform, two explicitly-named punctuation policies."""

    def test_both_policies_are_reachable_and_differ(self):
        """The legacy DELETION policy stays available under its own name.

        `dedupe._normalize_description` pins ``strip`` because
        `compute_content_fingerprint`'s digests are already on disk; the
        root-cause match site pins ``separator``.  One implementation, two
        call-site-pinned policies.
        """
        from escalation.canonical import canonical_text

        assert canonical_text('a.b', punctuation='strip') == 'ab'
        assert canonical_text('a.b', punctuation='separator') == 'a b'

    def test_separator_is_the_default(self):
        """The safe (conservative) policy is what you get without asking."""
        from escalation.canonical import canonical_text

        assert canonical_text('a.b') == canonical_text('a.b', punctuation='separator')

    def test_root_cause_alias_pins_the_separator_policy(self):
        """canonical_root_cause is exactly canonical_text(..., 'separator')."""
        from escalation.canonical import canonical_root_cause, canonical_text

        for key in _REAL_ROOT_CAUSE_KEYS + ['A.B c', '::', 'ロック競合:2370']:
            assert canonical_root_cause(key) == canonical_text(
                key, punctuation='separator'
            )

    def test_strip_policy_reproduces_dedupes_current_pipeline(self):
        """Characterisation of the lifted transform under the legacy policy."""
        from escalation.canonical import canonical_text

        assert canonical_text('Fused-memory  CONNECTION timeout!', punctuation='strip') == (
            'fusedmemory connection timeout'
        )
        assert canonical_text('cpu+memory leak', punctuation='strip') == 'cpumemory leak'
        # Underscore is a word character (Unicode Pc) and is PRESERVED by both
        # policies — the same deliberate divergence dedupe.summary_dedupe_key
        # already documents.
        assert canonical_text('fused_memory', punctuation='strip') == 'fused_memory'
        assert canonical_text('fused_memory', punctuation='separator') == 'fused_memory'

    def test_unknown_policy_is_rejected_loudly(self):
        """A typo'd policy must fail, never silently pick a default.

        Picking a default here would silently apply the WRONG matching rule at a
        call site that thought it had pinned one — the exact silent-degradation
        this module exists to prevent.
        """
        from escalation.canonical import canonical_text

        with pytest.raises(ValueError, match='punctuation'):
            canonical_text('a.b', punctuation='delete')  # type: ignore[arg-type]


class TestCanonicalModuleIsALeaf:
    """The structural property that lets queue.py import this module at all."""

    def test_importing_canonical_pulls_in_no_other_escalation_module(self):
        """Fresh-interpreter probe: the sys.modules delta is exactly this module.

        WHY BEHAVIOURAL, NOT A GREP: leaf-ness is a runtime property.  A source
        scan for ``import escalation`` is defeated by relative imports,
        ``importlib.import_module`` and line-broken import statements, and is
        falsely tripped by prose.  This probe tests the actual property.

        WHY IT MATTERS: ``dedupe.py`` imports ``escalation.queue`` at module
        level, so the moment this module acquires ANY intra-package import it
        risks completing a cycle — and the failure then surfaces as an
        ImportError far from the edit that caused it.

        The technique is the house pattern already used for
        ``fused_memory/reconciliation/recon_pool_map.py``, which exists to break
        an analogous cycle.
        """
        src_dir = Path(__file__).resolve().parent.parent / 'src'
        probe = (
            'import sys\n'
            f'sys.path.insert(0, {str(src_dir)!r})\n'
            'import escalation\n'
            "before = {m for m in sys.modules if m.startswith('escalation')}\n"
            'import escalation.canonical\n'
            "after = {m for m in sys.modules if m.startswith('escalation')}\n"
            "print(repr(sorted(after - before)))\n"
        )
        proc = subprocess.run(
            [sys.executable, '-c', probe],
            capture_output=True,
            text=True,
            timeout=60,
        )
        # Assert the probe SURVIVED first: a probe that dies during import
        # reports an empty delta and would otherwise pass vacuously.
        assert proc.returncode == 0, (
            f'leaf-ness probe failed to run (rc={proc.returncode}).\n'
            f'stdout: {proc.stdout}\nstderr: {proc.stderr}'
        )
        delta = eval(proc.stdout.strip())  # noqa: S307 — our own repr(sorted(...))
        assert delta == ['escalation.canonical'], (
            'escalation.canonical must stay a LEAF (zero intra-package imports) '
            'so queue.py can import it without completing the dedupe -> queue '
            f'cycle. It now also pulls in: {sorted(set(delta) - {"escalation.canonical"})}'
        )
