"""Generic concept pair dataclass for concept localization experiments."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ConceptPair:
    """A matched pair of prompts that differ by exactly 1–2 tokens.

    The 'pos' prompt is an instance where the target concept is present;
    the 'neg' prompt is the contrast where it is absent.  Everything else
    is held constant so that the residual-stream delta isolates the concept.

    template: surface-form variant name (e.g. 'T0', 'T1', 'T2') used to
    check that the concept direction is consistent across phrasings.
    """

    prompt_pos: str
    prompt_neg: str
    label_pos: str = ""
    label_neg: str = ""
    template: str = "T0"
    meta: dict = field(default_factory=dict)
