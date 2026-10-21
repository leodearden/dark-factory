"""A PRD's title and excerpt, read from git objects at a main commit, never from the working tree."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from _git_repo import Repo, make_repo
from client_feed.sources.prds import PrdsRead, PrdText, read_prds, summarize_prd

T0 = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)

A_PRD = """# Plan A

Status: draft, owner leo.

## 1. Goal

The goal of plan A.

## 2. Scope

Not this.
"""

B_PRD = '# Plan B\n\n## Goal\n\nThe goal of plan B.\n'


def summary(markdown: str, prd_path: str = 'plans/x-prd.md') -> tuple[str, str]:
    text = summarize_prd(prd_path, markdown)
    assert text.prd_path == prd_path
    return text.title, text.excerpt


class TestSummarize:
    def test_the_title_is_the_first_h1_and_the_excerpt_follows_the_first_h2(self) -> None:
        assert summary(A_PRD) == ('Plan A', 'The goal of plan A.')

    def test_subheadings_before_the_first_paragraph_are_skipped(self) -> None:
        markdown = '# T\n\n## 1. Goal\n\n### 1.1 Detail\n\nThe paragraph.\n'

        assert summary(markdown) == ('T', 'The paragraph.')

    def test_a_multi_line_paragraph_is_joined_with_single_spaces(self) -> None:
        markdown = '#   Spaced title  \n\n## Goal\n\nFirst line\n  second line\nthird.\n\nNext.\n'

        assert summary(markdown) == ('Spaced title', 'First line second line third.')

    @pytest.mark.parametrize('fence', ['```', '~~~'])
    def test_fenced_lines_are_neither_title_nor_excerpt(self, fence: str) -> None:
        markdown = (
            f'{fence}\n# not a title\n{fence}\n\n# Real title\n\n## Goal\n\n'
            f'{fence}python\nnot = "the excerpt"\n\n## not a heading\n{fence}\n\nThe excerpt.\n'
        )

        assert summary(markdown) == ('Real title', 'The excerpt.')

    def test_without_an_h1_the_title_is_the_path(self) -> None:
        assert summary('## Goal\n\nText.\n', 'plans/q-prd.md') == ('plans/q-prd.md', 'Text.')

    @pytest.mark.parametrize(
        'markdown', ['# T\n\nOnly a status line.\n', '# T\n\n## Goal\n', '# T\n\n## Goal\n\n### Sub\n']
    )
    def test_no_paragraph_after_an_h2_is_an_empty_excerpt(self, markdown: str) -> None:
        assert summary(markdown) == ('T', '')

    def test_paths_are_redacted_in_both(self) -> None:
        markdown = '# Plan for /home/leo/secret/x\n\n## Goal\n\nRead ~/notes/y first.\n'

        assert summary(markdown) == ('Plan for [path]', 'Read [path] first.')


@pytest.fixture
def repo(tmp_path: Path) -> Repo:
    repo = make_repo(tmp_path / 'root')
    repo.write('plans/a-prd.md', A_PRD)
    repo.write('plans/b-prd.md', B_PRD)
    repo.symlink('plans/link-prd.md', 'a-prd.md')
    return repo


class TestReadPrds:
    def test_the_committed_text_is_read_not_the_working_tree(self, repo: Repo) -> None:
        head = repo.commit('prds', when=T0)
        repo.write('plans/a-prd.md', '# Uncommitted title\n\n## Goal\n\nUncommitted.\n')

        read = read_prds(repo.root, commit=head, prd_paths=['plans/a-prd.md'])

        assert read == PrdsRead(
            found=(PrdText(prd_path='plans/a-prd.md', title='Plan A', excerpt='The goal of plan A.'),),
            absent=(),
        )

    def test_paths_that_are_not_regular_files_at_the_commit_are_absent(self, repo: Repo) -> None:
        head = repo.commit('prds', when=T0)

        read = read_prds(
            repo.root,
            commit=head,
            prd_paths=['plans/z-prd.md', 'plans/link-prd.md', 'plans', 'plans/b-prd.md'],
        )

        assert [text.prd_path for text in read.found] == ['plans/b-prd.md']
        assert read.absent == ('plans', 'plans/link-prd.md', 'plans/z-prd.md')

    def test_duplicates_are_read_once_and_results_are_sorted(self, repo: Repo) -> None:
        head = repo.commit('prds', when=T0)

        read = read_prds(
            repo.root,
            commit=head,
            prd_paths=['plans/b-prd.md', 'plans/a-prd.md', 'plans/b-prd.md', 'x', 'x'],
        )

        assert [text.prd_path for text in read.found] == ['plans/a-prd.md', 'plans/b-prd.md']
        assert read.absent == ('x',)

    def test_bytes_that_are_not_utf8_decode_with_replacement(self, repo: Repo) -> None:
        repo.write('plans/c-prd.md', b'# Caf\xe9 plan\n\n## Goal\n\nOk.\n')
        head = repo.commit('prds', when=T0)

        (text,) = read_prds(repo.root, commit=head, prd_paths=['plans/c-prd.md']).found

        assert (text.title, text.excerpt) == ('Caf� plan', 'Ok.')

    def test_no_paths_is_an_empty_read(self, repo: Repo) -> None:
        head = repo.commit('prds', when=T0)

        assert read_prds(repo.root, commit=head, prd_paths=[]) == PrdsRead(found=(), absent=())
