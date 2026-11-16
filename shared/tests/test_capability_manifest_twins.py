"""Tests for capability_manifest_twins: a sidecar's task labels must all appear in its .md twin.

The sweep over the committed corpus is tests/repo_guards/test_checked_in_capability_manifest_twins.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from capability_manifest_twins import check_md_twin, md_twin_path, unmentioned_labels
from pydantic import ValidationError

from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)


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


def _sidecar_yaml(*labels: str, extra_capability_key: str = '') -> str:
    blocks = ''.join(
        f'  - label: {label}\n'
        '    capabilities:\n'
        f'      - name: cap-{index}\n'
        '        binding: a binding\n'
        '        verdict: PASS\n'
        f'{extra_capability_key}'
        for index, label in enumerate(labels)
    )
    return f'prd: plans/x-prd.md\nschema_version: 1\ntasks:\n{blocks}'


def _write_pair(repo_root: Path, sidecar_text: str, md_text: str | None) -> Path:
    sidecar = repo_root / 'plans' / 'x-prd.capability-manifest.yaml'
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text(sidecar_text, encoding='utf-8')
    if md_text is not None:
        sidecar.with_suffix('.md').write_text(md_text, encoding='utf-8')
    return sidecar


class TestCheckMdTwin:
    def test_a_twin_mentioning_every_label_passes(self, tmp_path):
        md = '## α — first\n\n| Γ1 | cap-1 | PASS |\n'
        sidecar = _write_pair(tmp_path, _sidecar_yaml('α', 'gamma1'), md)
        assert check_md_twin(sidecar, tmp_path) is None

    def test_a_missing_twin_names_both_repo_relative_paths(self, tmp_path):
        sidecar = _write_pair(tmp_path, _sidecar_yaml('α'), None)
        message = check_md_twin(sidecar, tmp_path)
        assert message is not None
        assert 'plans/x-prd.capability-manifest.yaml' in message
        assert 'plans/x-prd.capability-manifest.md' in message
        assert str(tmp_path) not in message

    def test_drift_names_the_twin_and_only_the_unmentioned_labels(self, tmp_path):
        md = '## α — first\n\n### γ — register on four servers\n'
        sidecar = _write_pair(tmp_path, _sidecar_yaml('α', 'γ1', 'γ2', 'γ3'), md)
        message = check_md_twin(sidecar, tmp_path)
        assert message is not None
        assert 'plans/x-prd.capability-manifest.md' in message
        for label in ('γ1', 'γ2', 'γ3'):
            assert label in message
        assert "'α'" not in message

    @pytest.mark.parametrize('md_text', [None, '## α — first\n\nno γ1 here\n'])
    def test_a_failure_message_is_one_line(self, tmp_path, md_text):
        sidecar = _write_pair(tmp_path, _sidecar_yaml('α', 'γ2'), md_text)
        message = check_md_twin(sidecar, tmp_path)
        assert message is not None
        assert '\n' not in message

    def test_a_schema_invalid_sidecar_propagates_instead_of_reporting_twin_drift(self, tmp_path):
        invalid = _sidecar_yaml('α', extra_capability_key='        verdcit: PASS\n')
        sidecar = _write_pair(tmp_path, invalid, '## α — first\n')
        with pytest.raises(ValidationError):
            check_md_twin(sidecar, tmp_path)

