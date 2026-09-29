#!/usr/bin/env bash
# scripts/lib_task_citation.sh — the task-citation grammar, SPOT.
#
# "Does this commit message cite task N?" is a NORMATIVE question across the
# reify/dark-factory seam: dark-factory's reconciliation decides whether a
# branch landed by asking it, and reify's classifiers must answer it the same
# way or the two repos disagree about which branches are phantom-done. This
# file is the ONLY copy of reify's answer.
#
# Consumers (both source this file; neither may re-inline the EREs):
#   scripts/warm-lane-degenerate-ref-check.sh    — landed-vs-degenerate ref
#                                                  classification
#   scripts/task-branch-contamination-sweep.sh   — peer-commit census
#
# DO NOT re-inline any ERE in a consumer. A second copy is the drift this file
# exists to prevent, and tests/infra/test_lib_task_citation.sh fails if one
# reappears.
#
# The grammar — a message cites <id> iff ANY of:
#   * a line of it is `Merge <prefix><id> into …` (grep's `^` is a LINE anchor;
#     see Block B of the test suite for why that is kept);
#   * its SUBJECT is a conventional-commit head citing <id>: `<kind>(<id>)` /
#     `<kind>(<id>:`, or — for a non-empty prefix — `<kind>` followed later in
#     the subject by `<prefix><id>`, where <kind> is dark-factory's closed list.
#     SUBJECT means what git's `%s` means: the first paragraph, its lines
#     joined by single spaces. A caller holding `%B` and a caller holding `%s`
#     therefore get the same verdict;
#   * it carries a `#<id>` reference anywhere,
# each with boundary safety in both directions: task/1 must not match
# "Merge task/10 into main", impl(5) must not match impl(50), and #45 must not
# match #4588.
#
# Relationship to dark-factory — measured, not assumed. dark-factory's
# normative pattern is orchestrator/git_ops.py DEFAULT_COMMIT_CITATION_PATTERN,
# applied to the SUBJECT only (its task 2675). Arm by arm:
#   * merge-subject arm — agrees with DF's `^Merge task/{tid} into `
#     alternative, and with find_merge_marker's unanchored search.
#   * conventional-commit arm — mirrors DF's first alternative, kind list and
#     `[):]` terminator included, for the default prefix `task/` (DF hardcodes
#     `task/` there; this arm takes the caller's prefix, and drops the
#     `…<prefix><id>` alternative for an EMPTY prefix, where it would accept any
#     bare number). It is subject-only for the same reason DF's is: a squash or
#     merge body that LISTS `impl(<id>): …` lines is not a citation. This is the
#     form reify's own commits actually use, so without it the grammar missed
#     ~98% of task commits.
#   * `#<id>` arm — reify-only; DF has no counterpart. Kept because
#     warm-lane-degenerate-ref-check.sh's `landed` verdict rests on it for tips
#     like "feat(selectors): … (#4857, Option B)".
#   * DF's two UNANCHORED paren alternatives, `(#?<id>)` and `(task <id>)`, are
#     deliberately NOT mirrored. DF's own comment on them (task 2870) accepts
#     their collision risk because its landing attribution also requires the
#     citing commit's effect to be present at main HEAD; neither consumer here
#     has such a check, and on reify's history they attribute orchestrator
#     subjects such as "chore: save WIP before warm-lane reclaim (task 1933)"
#     to 1933, and enumerations like "(2)" to task 2.
#
# Purity contract — every function here takes all of its input as explicit
# parameters. No git invocation, no caller global (REPO_DIR, BRANCH_PREFIX_RE
# and friends are passed in, never read), no argv parsing, no `set -e` /
# `pipefail` side effect on a sourcing script, no stdout beyond the documented
# result. That is what makes this file safe to `source` from anywhere.
#
# The converse of that contract is the reason for the feeding idiom below: this
# file does not set `pipefail`, but every consumer does, so a pipeline's status
# here is the caller's shell option applied to OUR process list. NEVER make a
# `grep -q` the READER of a pipe. `grep -q` exits at its first match, and a
# writer still writing when it goes away dies of SIGPIPE — which `pipefail`
# then promotes into the pipeline's status, turning a MATCH into a non-match.
# The arbiter therefore feeds grep by REDIRECTION (`<<<`), so grep's own
# match/no-match is the verdict and no second process can overrule it. The
# harvest's pipes are exempt for a checkable reason, not a size guess: their
# readers (the candidate loop, `sort`) consume to EOF, so no reader ever
# departs early and there is no SIGPIPE window to lose a citation through.
# Block G of tests/infra/test_lib_task_citation.sh pins this.
#
# Exports exactly three functions:
#   task_citation_regex_escape  <string>                       -> escaped (stdout)
#   task_citation_message_cites <message> <id> <escaped_prefix> -> exit 0/1
#   task_citation_peer_ids      <message> <escaped_prefix>      -> ids (stdout)

# Source guard — prevent double-sourcing when two consumers meet in one shell.
if [ "${_REIFY_LIB_TASK_CITATION_SH_SOURCED:-}" = "1" ]; then
    return 0 2>/dev/null || true
fi
_REIFY_LIB_TASK_CITATION_SH_SOURCED=1

# dark-factory DEFAULT_COMMIT_CITATION_PATTERN's conventional-commit kinds,
# verbatim. A closed list, like DF's: an open `[a-z]+` would also accept kinds
# DF does not (verify, review, step-N — ~2% of reify's id-headed subjects),
# and the two repos would then disagree on exactly those commits.
_TASK_CITATION_KINDS='(merge|impl|amend|fix|test|feat|chore|docs|refactor|style|build)'

# task_citation_regex_escape <string>
# Escapes ERE metacharacters in <string> (anything outside [a-zA-Z0-9_]) so it
# can be safely interpolated into a `grep -E` pattern as a literal. Guards the
# merge-subject check below against a caller-supplied branch prefix containing
# regex metacharacters (e.g. "." or "+"), which would otherwise be interpreted
# as regex syntax instead of matched literally. Prints the escaped string with
# no trailing newline.
task_citation_regex_escape() {
    printf '%s' "$1" | sed -e 's/[^a-zA-Z0-9_]/\\&/g'
}

# task_citation_message_cites <message> <id> <escaped_prefix>
# True iff <message> cites task <id> under the grammar documented above.
# <escaped_prefix> must already have been through task_citation_regex_escape.
#
# This function is the single ARBITER of the grammar: task_citation_peer_ids
# below offers it a provably complete candidate set and defers every verdict
# here, so the two can never disagree about what "cites" means. Block F of
# tests/infra/test_lib_task_citation.sh enforces that as set equality, in both
# directions, over a corpus of branch prefixes.
#
# The conventional-commit grep runs under LC_ALL=C: in a UTF-8 locale glibc's
# `[A-Za-z]` also matches letters such as 'é', so the id boundary would
# otherwise depend on the caller's environment.
task_citation_message_cites() {
    local msg="$1" id="$2" prefix_re="$3" line subject="" kind_ref
    if grep -qE "^Merge ${prefix_re}${id} into " <<<"$msg"; then
        return 0
    fi
    while IFS= read -r line; do
        if [ -z "${line//[[:space:]]/}" ]; then
            [ -z "$subject" ] || break
            continue
        fi
        subject="${subject:+$subject }$line"
    done <<<"$msg"
    kind_ref="\(${id}[):]"
    [ -z "$prefix_re" ] \
        || kind_ref="$kind_ref|.*[^A-Za-z0-9_]${prefix_re}${id}([^A-Za-z0-9_]|\$)"
    if LC_ALL=C grep -qE "^${_TASK_CITATION_KINDS}(${kind_ref})" <<<"$subject"; then
        return 0
    fi
    if grep -qE "(^|[^0-9])#${id}([^0-9]|\$)" <<<"$msg"; then
        return 0
    fi
    return 1
}

# task_citation_peer_ids <message> <escaped_prefix>
# Prints every task id <message> cites, one per line, numerically sorted and
# de-duplicated; prints nothing when it cites none. No id-width restriction —
# the grammar is digit-boundary-delimited, not width-delimited.
#
# Two stages, deliberately: enumerate CANDIDATES, then let
# task_citation_message_cites adjudicate each one. The candidates are every
# SUFFIX of every maximal digit run in the message, which is complete by
# construction: every arm requires a non-digit immediately after the id, so an
# accepted id always ends where its digit run ends. Nothing about a sigil or
# the prefix is consulted to find them, and that is the point — an earlier
# harvest that scanned for `#`, `(` or the prefix lost ids whenever one sigil's
# match swallowed the start of another's (prefix `1/` over "see (1/200)"
# yielded nothing), and before that it fused a digit-bearing prefix onto the
# id. The enumeration cannot do either.
task_citation_peer_ids() {
    local msg="$1" prefix_re="$2" run i id
    local -A seen=()
    {
        while IFS= read -r run; do
            for ((i = 0; i < ${#run}; i++)); do
                id="${run:i}"
                [ -z "${seen[$id]:-}" ] || continue
                seen[$id]=1
                if task_citation_message_cites "$msg" "$id" "$prefix_re"; then
                    printf '%s\n' "$id"
                fi
            done
        done < <(grep -oE '[0-9]+' <<<"$msg" || true)
    } | sort -nu
}
