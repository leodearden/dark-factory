# Tool-call parser over-consumption and silent parameter drop

## Summary

When a model closes a string-valued tool-call parameter with the wrong
closing tag, the parser does not raise an error. Instead it keeps scanning
forward past the intended value until it finds some other closing tag it
does recognize, and treats everything in between — every other parameter the
call was going to pass — as part of the first parameter's value. Those
intervening parameters are never delivered to the tool: not defaulted, not
logged, not surfaced as an error, simply discarded. We can reproduce this
deterministically, we have measured it happening in roughly 1 out of every
400 tool calls in our own transcripts, and for the large majority of the
corrupted calls we can deterministically reconstruct exactly which arguments
were dropped and what they contained — which is itself the strongest
evidence that the parser could raise an error here instead of silently
guessing.

> **Notation.** Every angle bracket below that would form part of a markup
> tag is written as the HTML entity `&#60;` instead of a literal `<`. This is
> not a style choice: the defect described in this report is triggered by a
> literal `<` appearing inside a string argument of a tool call, including
> the tool call that would be used to author this very report. Writing the
> raw character here would reproduce the bug in the act of reporting it —
> the authoring call would itself terminate early and silently drop its own
> remaining arguments, corrupting this document before it could even be
> saved. That this report cannot safely be written with literal angle
> brackets is itself part of the evidence for how the defect behaves.

## Over-consumption

Tool calls are serialized with each parameter's value wrapped in a pair of
tags: an opening tag naming the parameter, e.g. `&#60;parameter
name="description">`, and a matching closing tag, `&#60;/parameter>`. Models
occasionally close a parameter with the wrong tag. Two variants of this are
common: echoing the parameter's own name instead of using the generic
closer (`&#60;/description>` instead of `&#60;/parameter>`), and, having
drifted into that dialect, continuing to open and close the *following*
parameters the same way (`&#60;priority>medium&#60;/priority>` instead of
the canonical form). Both are a model-side formatting mistake, and this
report is not about fixing the model — see "What we're asking for" below.

What is under the parser's control is what happens next. When the parser
does not find the closing tag it expects, it does not treat this as a syntax
error. It **over-consumes**: it keeps scanning forward through the text
until it finds *some* closing tag it does recognize — a later well-formed
`&#60;/parameter>`, or, failing that, the tool call's own closing
`&#60;/invoke>` — and it treats everything between the original mis-close
and that terminator as though it were still part of the first parameter's
value.

The practical effect: a corrupted parameter's value ends up containing the
literal, still-tagged text of one or more *other* parameters, appended after
whatever the model actually intended that first parameter to contain.

## Silent parameter drop

Over-consumption alone would just produce an ugly value. What makes it a
serious defect is what happens to the parameters it swallows: every one of
them is **silently discarded**. Not defaulted to a documented fallback, not
logged, not surfaced to the caller or the model as an error of any kind —
the tool call simply proceeds as though those arguments were never supplied.

This is a substantially worse failure mode than a parse error would be, and
the difference is the entire point of this report. A parse error is loud: it
fails at the call site, with the offending text in hand, and whoever is
driving the call finds out immediately. A silently dropped argument is
quiet: the call *succeeds* — frequently with some other, unrelated piece of
code substituting a plausible-looking default for the now-missing parameter
— and the actual consequence surfaces, if it ever does, arbitrarily far
downstream, disconnected from its cause. We have repeatedly had to
reconstruct, well after the fact, which of several plausible default values
was silently substituted for an argument the model had actually supplied.
