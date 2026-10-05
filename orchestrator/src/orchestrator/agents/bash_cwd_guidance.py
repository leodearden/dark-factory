"""Role-prompt guidance: an earlier `cd` can move every later relative path.

`orchestrator/src/orchestrator/agents/roles.py::_BASH_CAPABLE_ROLE_PREAMBLE`
splices `BASH_CWD_ANCHOR_GUIDANCE` into every role holding unqualified `Bash`.
It is a leaf module, imported downward by roles.py, only so that already
oversized file does not grow further. Provenance: reify #7924 via task 5971;
dark_factory codebook entry-cand-20260729-4.

The mechanical half is
`orchestrator/src/orchestrator/agents/invoke.py::apply_bash_cwd_reset_env`.
This prose is written to hold whether or not the shell is reset: non-Claude
backends ignore that env var, an operator may opt out, and its live effect has
not yet been exercised end-to-end.

`orchestrator/src/orchestrator/agents/path_not_found_guidance.py::PATH_NOT_FOUND_GUIDANCE`
owns RECOVERING a not-found path by search; this block owns the CWD cause and
its prevention.

`orchestrator/tests/test_roles_bash_cwd_anchor.py` checks the block's shape
and its splice, never its wording.
"""

BASH_CWD_ANCHOR_GUIDANCE = """
## An earlier `cd` can move every later relative path

You are dispatched with your working directory at the ROOT of your checkout.
Whether a `cd` outlives the `Bash` call that ran it depends on the session: in
some, the shell stays in that directory for every later call -- including
after a `cd` buried in a compound command many turns ago; in others it is put
back at the root, often silently. A relative or omitted `path` given to `Grep`
or `Glob` resolves against that same shell directory. A repo-root-relative
path used after a lingering `cd` then fails as "No such file or directory" or
"Path does not exist" for a file that is really there, and the shell's error
names neither the directory you assumed nor the one you got. Rely on neither
behaviour:

- Keep a `cd` in the command that needs it -- `cd crates/foo && cargo test` --
  never in an earlier call.
- Anchor any command that uses repo-root-relative paths:
  `cd "$(git rev-parse --show-toplevel)" && <command>`. That query answers with
  the checkout root from every directory inside it.
- Give `Read`, `Grep` and `Glob` an ABSOLUTE path. Leaving `path` out does not
  help: an omitted path is the same shell directory.
- When a repo-root-relative path you expected comes back not-found, check
  where the shell is before anything else: run `pwd` and
  `git rev-parse --show-toplevel`. A drifted directory breaks EVERY relative
  path you use next, not only this one, so fix the directory once. Searching
  for the file by name recovers that one path; fixing the directory recovers
  all of them. A `Grep` "Path does not exist" error already names your current
  working directory -- read it.
- A `Bash` result ending "Shell cwd was reset to <dir>" means the harness put
  you back at <dir>, and the `cd` in that call did not carry over. Its absence
  proves nothing: the reset can also be silent.
"""
