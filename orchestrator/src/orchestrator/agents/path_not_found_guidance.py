"""Role-prompt guidance: a not-found error on an expected path falsifies the path.

`orchestrator/src/orchestrator/agents/roles.py::_BASH_CAPABLE_ROLE_PREAMBLE`
and `orchestrator/src/orchestrator/agents/roles.py::JUDGE` splice
`PATH_NOT_FOUND_GUIDANCE` into every role with a literal system prompt. It is a
leaf module, imported downward by roles.py, only so that already oversized file
does not grow further.

`orchestrator/src/orchestrator/agents/partial_failure_guidance.py::MULTI_PATH_PARTIAL_FAILURE_GUIDANCE`
covers SPOTTING a missing path in multi-path output; this block covers
RECOVERING once a path the agent expected to exist has come back not-found.

`orchestrator/tests/test_roles_path_not_found.py` checks the block's shape and
its splice (carrier roles, count, order after the Grep block), never its
wording.
"""

PATH_NOT_FOUND_GUIDANCE = """
## A not-found error falsifies the path, whichever tool reported it

This is about a path you EXPECTED to exist. If you were asking whether a path
exists, not-found is simply the answer and nothing below applies.

A "No such file or directory", or a tool's own not-found error, is about the
PATH, not the tool that reported it. Handing the same path to another tool --
`wc`, then `cat`, `sed`, `Read` or `Grep` -- tests the same guess again and
fails the same way. Switching tools is not a new probe.

THE RULE. Once an expected path has come back not-found, the next call that
names that file must SEARCH for it, not use the path again. A second not-found
on the same path means that step was skipped: stop and re-derive the path
before doing anything else.

How to re-derive it: search for what you actually know, which is the file's
name. Run `Glob` with the pattern `**/<basename>` from the checkout root, and
use the absolute path it returns verbatim. A relative path can fail because it
resolved against a different directory than you assumed; an absolute path from
the search settles that cause too, so there is no need to work out which cause
it was.

An EMPTY search result means no file by that name exists where you looked. The
premise that it exists is what was wrong. Report that as a finding about the
task; it is not a cue to guess another directory.

Carry the correction forward. Every other path you built from the same assumed
directory -- sibling files, entries in a plan or file list, paths you mean to
commit -- rests on the same falsified guess. Re-derive them now rather than
finding them one not-found at a time.

Catalogued sighting: a test file assumed to live under a `harness_cli/`
subdirectory was probed with `wc` and then with `Grep`. Both answered
not-found, and the agent kept working from that directory for many more turns
before it re-derived the path.
"""
