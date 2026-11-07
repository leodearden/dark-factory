"""Warning lines picked out of a reify-style script's tagged stderr."""

from __future__ import annotations

from orchestrator.script_warnings import tagged_warning_lines

SEED_WARN = '\x1b[1;33m[warn]\x1b[0m  hooks-armed-guard.sh arm failed (exit 9) — seed continues'
GUARD_WARN = '[warn] hooks-armed-guard: NOT armed in /x: reason'
GUARD_ERROR = '[error] hooks-armed-guard: setup-main-gate-worktree-config.sh failed to pin /x'


class TestTaggedWarningLines:
    def test_coloured_seed_warn_keeps_tag_and_drops_colour(self):
        lines = tagged_warning_lines(SEED_WARN + '\n')

        assert lines == ('[warn]  hooks-armed-guard.sh arm failed (exit 9) — seed continues',)
        assert not any('\x1b' in line for line in lines)

    def test_uncoloured_warn_and_error_lines_kept_in_order(self):
        stderr = f'{GUARD_WARN}\n{GUARD_ERROR}\n'

        assert tagged_warning_lines(stderr) == (GUARD_WARN, GUARD_ERROR)

    def test_info_ok_untagged_and_embedded_tags_excluded(self):
        stderr = '\n'.join(
            [
                '\x1b[1;34m[info]\x1b[0m  seeding lane',
                '[info] plain info',
                '\x1b[1;32m[ok]\x1b[0m  done',
                '[ok] plain ok',
                'an untagged line',
                '[info] previously saw [warn] x',
            ]
        )

        assert tagged_warning_lines(stderr) == ()

    def test_leading_whitespace_tagged_line_included_stripped(self):
        assert tagged_warning_lines('   [warn] indented\n') == ('[warn] indented',)

    def test_empty_and_whitespace_only_stderr(self):
        assert tagged_warning_lines('') == ()
        assert tagged_warning_lines('  \n\t\n   ') == ()

    def test_mixed_multiline_input_returns_only_warn_lines_in_order(self):
        continuation = '\x1b[1;33m[warn]\x1b[0m    run scripts/hooks-armed-guard.sh check /lane'
        stderr = '\n'.join(
            [
                '\x1b[1;34m[info]\x1b[0m  seeding /lane',
                SEED_WARN,
                '\x1b[1;32m[ok]\x1b[0m  target seeded',
                continuation,
            ]
        )

        result = tagged_warning_lines(stderr)

        assert isinstance(result, tuple)
        assert result == (
            '[warn]  hooks-armed-guard.sh arm failed (exit 9) — seed continues',
            '[warn]    run scripts/hooks-armed-guard.sh check /lane',
        )
