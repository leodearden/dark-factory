"""Tests for capability_manifest_twins: a sidecar's task labels must all appear in its .md twin."""

from __future__ import annotations

from pathlib import Path

import pytest
from capability_manifest_twins import md_twin_path, unmentioned_labels


class TestMdTwinPath:
    def test_plans_sidecar_twin_is_same_stem_md(self):
        sidecar = Path('plans/x-prd.capability-manifest.yaml')
        assert md_twin_path(sidecar) == Path('plans/x-prd.capability-manifest.md')

    def test_docs_prds_sidecar_twin_keeps_its_directory(self):
        sidecar = Path('docs/prds/y.capability-manifest.yaml')
        assert md_twin_path(sidecar) == Path('docs/prds/y.capability-manifest.md')


class TestUnmentionedLabels:
    def test_every_label_present_verbatim_leaves_nothing_unmentioned(self):
        text = '## α — a\n| β | b |\n### γ1 — c\n'
        assert unmentioned_labels(['α', 'β', 'γ1'], text) == ()

    def test_absent_labels_are_returned_in_input_order(self):
        assert unmentioned_labels(['β', 'α', 'γ'], '## β — only this one') == ('α', 'γ')

    def test_a_bare_parent_letter_never_satisfies_its_numbered_split(self):
        text = '### γ — register on four servers\n\n| γ | tool-registered | PASS |\n'
        assert unmentioned_labels(['γ1', 'γ2', 'γ3'], text) == ('γ1', 'γ2', 'γ3')

    @pytest.mark.parametrize(
        ('label', 'text'),
        [
            ('γ1', 'see γ10 for details'),
            ('δ', 'see δ′ for details'),
            ('contested-heal', 'see contested-heal-full for details'),
            ('alpha', 'the alphabet'),
        ],
    )
    def test_a_longer_token_containing_the_label_does_not_satisfy_it(self, label, text):
        assert unmentioned_labels([label], text) == (label,)

    @pytest.mark.parametrize(
        'text',
        ['**γ1**', '| γ1 |', '`γ1`', '## γ1 — x', '(γ1←β←α)', 'then γ1.'],
    )
    def test_markdown_punctuation_around_a_label_still_counts(self, text):
        assert unmentioned_labels(['γ1'], text) == ()

    def test_a_primed_label_is_satisfied_by_its_verbatim_spelling(self):
        assert unmentioned_labels(['δ′'], '## δ′ — x') == ()

    @pytest.mark.parametrize(
        ('label', 'text'),
        [
            ('alpha', '## α — x'),
            ('gamma1', '| Γ1 | x |'),
            ('gamma1', '| γ1 | x |'),
            ('pi2', '## π2 — x'),
            ('psi', '## ψ — x'),
            ('theta', '## θ — x'),
            ('alpha', '## alpha — x'),
        ],
    )
    def test_an_ascii_greek_name_is_satisfied_by_its_letter_in_either_case(self, label, text):
        assert unmentioned_labels([label], text) == ()

    @pytest.mark.parametrize('text', ['## γ — x', '## γ2 — x'])
    def test_an_ascii_greek_name_needs_its_exact_letter_and_digits(self, text):
        assert unmentioned_labels(['gamma1'], text) == ('gamma1',)

    def test_a_non_greek_ascii_label_is_not_satisfied_by_a_symbol_rendering(self):
        assert unmentioned_labels(['Cprime'], '## C′ — x') == ('Cprime',)

    def test_a_non_greek_ascii_label_is_satisfied_by_its_verbatim_token(self):
        assert unmentioned_labels(['Cprime'], '## C′ (`Cprime`) — x') == ()

    def test_a_greek_letter_label_is_not_satisfied_by_its_ascii_name(self):
        assert unmentioned_labels(['α'], '## alpha — x') == ('α',)
