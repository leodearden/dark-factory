"""Role-prompt guidance: locate a file by name through git's index, not a tree walk.

`orchestrator/src/orchestrator/agents/roles.py::_BASH_CAPABLE_ROLE_PREAMBLE`
and `orchestrator/src/orchestrator/agents/roles.py::JUDGE` splice
`FILE_LOOKUP_GUIDANCE` into every role with a literal system prompt. It is a
leaf module, imported downward by roles.py, only so that already oversized file
does not grow further. Provenance: reify #7936 via task 5975; reify codebook
entry-cand-20260827-17.

`orchestrator/src/orchestrator/agents/path_not_found_guidance.py::PATH_NOT_FOUND_GUIDANCE`
owns RECOVERING a not-found path by searching for it; this block owns WHERE a
by-name search is safe to run, and which lookup cannot walk into worktree
copies.

`orchestrator/tests/test_roles_file_lookup.py` checks the block's shape and its
splice, never its wording. The measurements behind it live in the commit that
introduced it, not here.
"""

FILE_LOOKUP_GUIDANCE = """
## Locating a file by name: ask git's index before walking a tree

A task checkout holds ONE copy of its project's tree, so a by-name `Glob` or
`find` from your own checkout root is fine. Two kinds of root are different:

- A project's MAIN checkout, the directory that holds its task worktrees,
  carries a full copy of the tree for every one of them: under `.worktrees/`,
  and in dark-factory's own root also under `.worktrees-orphaned/`,
  `.eval-worktrees/` and `.claude/worktrees/`. You land in such a root when you
  look in your project's main checkout, or in another project's root.
- Build trees (`target/`, `.venv`, `node_modules`) can dwarf the source, even
  inside a single checkout.

A recursive `find` from such a root visits every copy. It prints the same file
once per copy and runs into the `Bash` timeout before it finishes. Pruning one
directory leaves all the others, and raising the timeout is not the fix.

The `Glob` tool is no way round it. Unlike `Grep`, it reads no ignore files,
so from that root it walks the same copies and fails with a ripgrep timeout
that tells you to search a more specific path. `Grep` does honour `.gitignore`.

THE RECOURSE, for a file git tracks:

    git -C <root> ls-files -- '*<name>'

It answers from git's index, never enters an ignored directory, and returns in
well under a second. Its paths are relative to <root>, so prefix <root> to get
the absolute path `Read` needs. Add `--others --exclude-standard` to include
new untracked files as well; git still skips ignored directories. `git -C`
needs no `cd`, so it runs under a git-only `Bash` grant and leaves the shell's
directory where it was.

For a file that is itself gitignored, such as a `.task/` file or a build
artifact, git's index has no entry. Point `Glob` at the directory that should
hold it, never at a root holding copies.

Catalogued sighting: from another project's root, a `find` that pruned only
`node_modules` went looking for one test file. It printed that file once per
worktree copy and was still walking when its timeout killed it.
"""
