"""Role-prompt guidance: a search pattern is parsed by the shell, then by the regex engine.

`orchestrator/src/orchestrator/agents/roles.py::_BASH_CAPABLE_ROLE_PREAMBLE`
splices `GREP_PATTERN_ESCAPING_GUIDANCE` into every role holding unqualified
`Bash`. It is a leaf module, imported downward by roles.py, only so that
already oversized file does not grow further.

A character meant literally can be syntax to the shell (backticks and `$`
inside double quotes) or to the regex engine (an unescaped paren), and either
layer fails before anything is searched.

`orchestrator/tests/test_roles_grep_pattern_escaping.py` checks the block's
shape and its splice (carrier roles, count, order after
`GREP_LOOKAROUND_GUIDANCE`), never its wording.
"""

GREP_PATTERN_ESCAPING_GUIDANCE = """
## A search pattern is parsed twice: by the shell, then by the regex engine

A `grep` or `rg` pattern in a `Bash` call is parsed twice before anything is
searched: first by the shell, which runs your command text through `eval`,
then by the regex engine. A character you meant literally can be syntax to
either layer, and each failure is easy to misread as a search result. A `Grep`
tool pattern skips the shell: it reaches ripgrep as a parameter.

THE SHELL. Inside DOUBLE quotes the shell still expands backticks and `$`. An
unpaired backtick (three of them, say, to match a markdown code fence) is a
shell syntax error: "/bin/bash: eval: line 1: unexpected EOF while looking for
matching", with exit 2. The shell rejected your quoting, so grep never ran: the
error says nothing about your pattern, your matches or your file. A `$name`
inside double quotes fails SILENTLY instead: it expands, usually to nothing,
the pattern quietly changes, and you get a wrong count with no error. Fix both
by single-quoting every pattern in a `Bash` call: inside single quotes the
shell changes nothing. If the pattern itself holds a single quote, use the
`Grep` tool.

THE REGEX ENGINE. Parentheses, square brackets, `.`, `*`, `+`, `?`, `|`, `^`,
`$` and backslash are regex syntax. A literal paren in an alternation, such as
"def foo(" and "class Bar(" in one pattern, is an unclosed group, and the
search never runs. `Grep` reports that ripgrep rejected the pattern, with
"regex parse error" and "unclosed group"; `grep -E` says "Unmatched ( or \\(",
and `grep -P` says "missing closing parenthesis". In `Bash` that is exit 2, the
same code as a missing file, while a clean no-match is exit 1, so read the
message, not the code. Nor is it the look-around rejection earlier in this
prompt: that pattern was well-formed but unsupported, this one is malformed,
and `grep -P` rejects it too.

THE FIX for a literal metacharacter. A bracket class, `[(]`, is literal in
every flavor, so it is the safe default. A backslash is flavor-dependent: `\\(`
is literal in `Grep`, `rg`, `grep -E` and `grep -P`, but in plain `grep` a
bare `(` is already literal and `\\(` OPENS a group, which reproduces the
error. For a purely literal search in `Bash`, `grep -F` or `rg -F` with one
`-e` per alternative turns the regex engine off. `Grep` has no fixed-string
mode, so use the bracket class there.
"""
