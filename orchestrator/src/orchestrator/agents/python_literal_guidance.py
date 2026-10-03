"""Role-prompt guidance: source text pasted into a Python string literal is rewritten by Python.

`orchestrator/src/orchestrator/agents/roles.py::_BASH_CAPABLE_ROLE_PREAMBLE`
splices `PASTED_TEXT_PYTHON_LITERAL_GUIDANCE` into every role holding
unqualified `Bash`. It is a leaf module, imported downward by roles.py, only so
that already oversized file does not grow further.

A plain literal decodes the backslash sequences in pasted code: an incomplete
one rejects the whole script at compile time, and a complete one is silently
turned into the character it stands for. Task 5964 refiles reify
legibility-census candidate #7910. The hazard needs no host hook, since it
reproduces with the script's bytes delivered intact, so it is unlike the
skim-hook flattening block retired with that hook (commit c735478936).

`orchestrator/tests/test_roles_pasted_text_python_literal.py` checks the
block's shape and its splice (carrier roles, count, order after
`GREP_LOOKAROUND_GUIDANCE`), never its wording.
"""

PASTED_TEXT_PYTHON_LITERAL_GUIDANCE = r"""
## Source text pasted into a Python string literal is rewritten by Python

THE RULE: text copied from a file, above all source code in another language,
enters a Python script only inside a RAW literal: `r'''...'''`. When the job is
replacing text in a file and `Edit` is in your tool list, prefer `Edit`: no
Python literal is involved, so none of the rewriting below applies.

WHY. A plain Python literal decodes backslash sequences, and pasted code is
full of them: `\n` and `\t` inside string literals, the `\u` escapes of Rust
and JavaScript, regexes, a Windows path such as `C:\users`. It fails in two
ways, and the quiet one costs more.

- LOUD. An incomplete escape, such as a `\u` not followed by four hex digits
  or a malformed `\x`, `\U` or `\N`, rejects the WHOLE script at compile time:
  "SyntaxError: (unicode error) 'unicodeescape' codec can't decode bytes in
  position 649-650: truncated \uXXXX escape". Nothing in the script ran, so
  there is no partial edit to undo. The line it reports is where the offending
  literal STARTS, and the position counts from that literal's opening quotes,
  not from the start of the line. Search the pasted text for a backslash
  rather than counting.
- SILENT. A complete escape decodes without any warning: `\n`, `\t`, a doubled
  backslash, an escaped quote, or `\u` plus four hex digits. Your old text then
  no longer matches the file, so a search finds nothing or a replace changes
  nothing. Your new text lands with those escapes already turned into the
  characters they stand for. Before writing, check that the old text occurs
  exactly once; that check catches this case.

WHAT A RAW LITERAL DOES NOT FIX. It still ends at the first matching triple
quote, so a pasted excerpt that reaches a docstring's own closing quotes ends
your literal early: switch quote style or stop the excerpt short of it. It also
cannot end in an odd number of backslashes.

THE SHELL IS A SECOND LAYER. Quote the heredoc delimiter: `<<'PY'`, not
`<<PY`. Unquoted, the shell collapses doubled backslashes and expands `$name`
before Python sees a byte. A `python3 -c "..."` argument gets the same
double-quote processing, so send a script that carries pasted text on stdin
through a quoted heredoc.
"""
