"""Role-prompt guidance: a check refutes a claim only in the claim's own execution context.

`orchestrator/src/orchestrator/agents/roles.py::_BASH_CAPABLE_ROLE_PREAMBLE`
splices `PREMISE_REFUTATION_GUIDANCE` into every role holding unqualified
`Bash`. It is a leaf module, imported downward by roles.py, only so that
already oversized file does not grow further. Provenance: reify #7937 via task
5976 (plans/confusion-reduction-prd.md §6 decision 4); reify codebook
entry-cand-20260819-24, where a brief declared a boot-time timer's PATH
premise false from a post-login interactive shell, and the premise was right.

THIS CONSTANT IS THE SINGLE NORMATIVE STATEMENT OF THE RULE. The skills
escalation-watcher-auto, escalation-watcher, recon-escalation-watcher, unblock
and spawn carry a one-line headline and point here. Correct the rule here.

`orchestrator/tests/test_roles_premise_refutation.py` checks the block's shape
and its splice, never its wording.
"""

PREMISE_REFUTATION_GUIDANCE = """
## A check refutes a claim only in the claim's own execution context

THE RULE. When you test a claim someone else made (an escalation's premise, a
task's stated failure, a verify or CI result, another agent's diagnosis), a
check that does NOT reproduce it is a NON-REPRODUCTION, not a refutation,
unless it ran in the execution context the claim is about. That context has
four parts:

- the process, and the environment it inherited from whatever launched it: a
  systemd unit or timer, the verify subprocess, a headless agent;
- the moment: a boot-time run before any login, the failing run's own time;
- the tree: the worktree, the commit and the virtualenv;
- the host and the service instance.

YOUR SESSION IS THE WRONG PLACE. Your own session is almost never that
context: it is already initialised, it inherited someone else's environment,
and it runs after the fact. Three tells:

- an interactive `which` says nothing about the PATH a boot-time timer had;
- a test passing in your shell says nothing about the verify subprocess,
  which drops your inherited virtualenv and resolves the tree's own;
- an absent log line refutes nothing unless the claimed code path writes to
  that log.

THE TWO VALID ROUTES. Re-run the check IN the claimed context (the same unit,
runner, tree and environment), or read an artefact that context produced at
the time: the unit's journal at the failure timestamp, the failing run's own
output or state file, or the environment of a process that context launched.

THE FALLBACK. If neither route is open, record the result as "not reproduced
in <the context you used>" and leave the premise standing. Never close,
dismiss or brief a claim as disproved on that evidence, and never tell a later
reader not to re-derive it. Whenever you cite a check as evidence, name the
context it ran in.
"""
