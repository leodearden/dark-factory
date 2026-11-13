"""One configurable stand-in for
``escalation/src/escalation/sha_citations.py::CitationResolver``: no git, every
answer chosen by the test.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence


class StubCitationResolver:
    """Each lookup answers from *answers*. *commits* names the tokens that name
    a commit (every token when None); a token in *faults* raises its exception
    on lookup, and *classify_fault* makes classification itself raise.

    ``classifications`` records each ``commits_named`` call's tokens, and
    ``lookups`` each ``resolve_rebased_citation`` call's sha, in call order.
    """

    def __init__(
        self,
        answers: Mapping[str, Mapping[str, object]] | None = None,
        *,
        commits: Collection[str] | None = None,
        faults: Mapping[str, Exception] | None = None,
        classify_fault: Exception | None = None,
    ) -> None:
        self._answers = answers or {}
        self._commits = commits
        self._faults = faults or {}
        self._classify_fault = classify_fault
        self.classifications: list[list[str]] = []
        self.lookups: list[str] = []

    async def commits_named(self, tokens: Sequence[str]) -> Mapping[str, str]:
        self.classifications.append(list(tokens))
        if self._classify_fault is not None:
            raise self._classify_fault
        return {t: t for t in tokens if self._commits is None or t in self._commits}

    async def resolve_rebased_citation(self, sha: str) -> Mapping[str, object] | None:
        self.lookups.append(sha)
        if sha in self._faults:
            raise self._faults[sha]
        return self._answers.get(sha)
