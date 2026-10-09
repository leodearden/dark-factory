"""D7's multiset key for an unowned inline suppression.

An unowned site contributes ``(kind, sorted codes, a 12-hex sha256 of its
stripped physical line)`` to the ratchet's multiset — and no path.  A rename or
a file split therefore invents no key and needs no baseline diff, while editing
the line a marker rides on does invent one, which is how an ordinary edit
converts a grandfathered suppression into one somebody has to own.  The accepted
cost — two identical lines share one key, which is what ``slack`` measures — is
PRD D7's.  :func:`key_params` names what two baselines must agree on for their
keys to mean the same thing.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from inline_suppression_kinds import Kind, Site

#: How many hex characters of the sha256 the key keeps.  Twelve is the PRD's,
#: and it is part of the ratchet's ``params`` block: changing it makes two
#: baselines not two measurements of the same thing, which is exactly what
#: ``shared.ratchet.ParamsMismatch`` exists to refuse.
_DIGEST_HEX = 12


@dataclass(frozen=True)
class SuppressionKey:
    """D7's multiset key: ``(kind, sorted codes, digest of the stripped line)``.

    No path, for the reason this module's docstring gives.

    Attributes:
        kind: Which suppression this is.
        codes: The rule codes, SORTED at construction — ``[b, a]`` and
            ``[a, b]`` are the same suppression, and a multiset that
            disagreed would ratchet on the author's typing order.
        digest: ``sha256`` of the stripped physical line, truncated.  Digesting
            the LINE rather than the comment is what makes an edit anywhere on
            it a touch, which is D7's whole conversion mechanism.
    """

    kind: Kind
    codes: tuple[str, ...]
    digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, 'codes', tuple(sorted(self.codes)))

    def render(self) -> str:
        """The key as one string, for JSON and for a violation line.

        A RENDERING EXISTS; AN INVERSE DOES NOT, and that asymmetry is the
        point.  JSON object keys are strings by the format's definition, so
        something has to render — ``shared.ratchet``'s docstring makes the same
        argument for the same reason.  What must not exist is a reader that
        parses one back: ``shared.ratchet`` treats every key as an opaque
        identity token, so a parser would be the ad-hoc parser of an internal
        value heuristic 12 forbids, and it would silently promote this spelling
        to a wire format nobody could ever change.

        The codes group is omitted entirely when there are none, because
        ``pragma: no cover[]`` reads as a missing code rather than as a kind
        that never has one.  The digest is separated by a SPACE, never by
        ``@`` — ``@`` is ``scripts/inline_suppressions.py::SuppressionClass.render``'s
        separator, and two
        key spellings a reader could confuse is the one thing worth spending a
        character to avoid.
        """
        codes = f'[{",".join(self.codes)}]' if self.codes else ''
        return f'{self.kind.value}{codes} {self.digest}'


def key_for(site: Site) -> SuppressionKey:
    """The D7 key *site* contributes to the multiset."""
    return SuppressionKey(
        kind=site.kind,
        codes=site.codes,
        digest=hashlib.sha256(site.text.encode('utf-8')).hexdigest()[:_DIGEST_HEX],
    )


#: The key scheme, as one opaque token in the ratchet's ``params`` block.  It is
#: a NAME rather than a description: params are compared for equality, so its
#: only job is to differ when the keys mean something different.
_KEY_SCHEME = 'kind+codes+sha256-of-stripped-line'


def key_params() -> dict[str, object]:
    """This scan's measurement parameters, for the baseline's ``params`` block.

    ONLY WHAT CHANGES THE MEANING OF THE MULTISET.  A params mismatch is exit 2
    and the only way out of it is re-seeding, which is itself the widening move
    D7 built the verbs to prevent — so a field belongs here exactly when a
    change to it makes two baselines two measurements of DIFFERENT things.  The
    scanned kinds and the key scheme qualify: add a kind, or change the hash, and
    every count is about something else.

    The resolved ruff ``select``/``ignore`` lists deliberately do NOT qualify,
    and that omission is the decision worth recording.  Putting them here would
    convert an ordinary, reviewable ``pyproject.toml`` edit into a forced
    baseline regeneration — punishing a legitimate config change by demanding
    the one operation nobody should perform casually.  They are published in
    ``--json`` instead, so the consumer model stays auditable without arming
    that tripwire.
    """
    return {
        'kinds': [kind.value for kind in Kind],
        'key_scheme': _KEY_SCHEME,
        'digest_hex': _DIGEST_HEX,
    }
