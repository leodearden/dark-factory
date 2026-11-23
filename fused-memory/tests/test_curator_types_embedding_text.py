"""Task 11429: every curator embedding input is composed by embedding_text and bounded by clip_for_embedding."""

from __future__ import annotations

from fused_memory.middleware.curator_types import (
    EMBEDDING_INPUT_MAX_BYTES,
    clip_for_embedding,
    embedding_text,
)

_TEXT_EMBEDDING_3_WINDOW_TOKENS = 8192
_OVER_THE_BOUND = 60_000


def _utf8_size(text: str) -> int:
    return len(text.encode('utf-8'))


class TestEmbeddingText:
    """`embedding_text` is the single owner of the curator corpus embedding text."""

    def test_title_alone(self):
        assert embedding_text('Fix the bug', '', []) == 'Fix the bug'

    def test_title_and_description_join_with_blank_line(self):
        assert embedding_text('T', 'D', []) == 'T\n\nD'

    def test_files_are_their_own_newline_block(self):
        assert embedding_text('T', 'D', ['a.py', 'b.py']) == 'T\n\nD\n\na.py\nb.py'

    def test_empty_description_contributes_nothing(self):
        assert embedding_text('T', '', ['a.py']) == 'T\n\na.py'

    def test_an_over_long_description_is_cut_to_the_bound_keeping_the_title(self):
        title = 'Sweep transcript secrets'

        text = embedding_text(title, 'x' * _OVER_THE_BOUND, [])

        assert _utf8_size(text) <= EMBEDDING_INPUT_MAX_BYTES
        assert text.startswith(title + '\n\n')

    def test_an_over_long_multibyte_description_is_cut_on_a_character_boundary(self):
        for char in ('€', '\U0001f600'):
            text = embedding_text('T', char * _OVER_THE_BOUND, [])

            assert _utf8_size(text) <= EMBEDDING_INPUT_MAX_BYTES
            assert text.encode('utf-8', 'strict').decode('utf-8', 'strict') == text


class TestClipForEmbedding:
    def test_the_bound_fits_the_text_embedding_3_window(self):
        assert EMBEDDING_INPUT_MAX_BYTES < _TEXT_EMBEDDING_3_WINDOW_TOKENS

    def test_a_lone_surrogate_does_not_raise_and_the_result_is_bounded(self):
        text = clip_for_embedding('\ud800' + 'x' * _OVER_THE_BOUND)

        assert len(text.encode('utf-8', 'surrogatepass')) <= EMBEDDING_INPUT_MAX_BYTES

    def test_a_short_text_is_returned_unchanged(self):
        assert clip_for_embedding('Fix the bug') == 'Fix the bug'

    def test_clipping_is_idempotent(self):
        once = clip_for_embedding('€' * _OVER_THE_BOUND)

        assert clip_for_embedding(once) == once
