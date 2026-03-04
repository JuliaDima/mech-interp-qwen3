"""Prompt suite for the Anthropic addition case study reproduction.

Provides all prompt constants used across the experiment.

Anthropic used:
  - Template: "calc: {a}+{b}=" over all a,b in {0,...,99}
  - Focus:    "calc: 36+59="
  - NL variant: "Answer in one word. What is 36+59?"
  - Generalization: prompts where addition is an intermediate step
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from mechinterp_qwen3.dataset_generation.generate_add_dataset import (
    DatasetConfig,
    SamplingStrategy,
    TemplateID,
    build_prompt,
    generate_pairs,
)

# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------

CALC_TEMPLATE: str = "calc: {a}+{b}="


# ---------------------------------------------------------------------------
# A. Full grid — calc: a+b= for all a,b in {0,...,99}
# ---------------------------------------------------------------------------


class CalcEntry(TypedDict):
    a: int
    b: int
    prompt: str
    answer: str


def _build_calc_grid() -> list[CalcEntry]:
    """Build all 10,000 calc: a+b= prompts for a,b in {0,...,99} using shared logic."""
    # We use a dummy config to get the grid pairs
    config = DatasetConfig(
        model_name="dummy",
        output_path=Path("dummy.jsonl"),
        templates=[TemplateID.T0],
        sampling_strategy=SamplingStrategy.GRID,
        max_value=99,
    )
    pairs = generate_pairs(config)

    grid: list[CalcEntry] = []
    for a, b in pairs:
        grid.append(
            CalcEntry(
                a=a,
                b=b,
                prompt=build_prompt(TemplateID.T0, a, b),
                answer=str(a + b),
            )
        )
    return grid


CALC_GRID: list[CalcEntry] = _build_calc_grid()


# ---------------------------------------------------------------------------
# B. Focus prompt — primary attribution target (Anthropic's 36+59=)
# ---------------------------------------------------------------------------

FOCUS_A: int = 36
FOCUS_B: int = 59
FOCUS_ANSWER: str = str(FOCUS_A + FOCUS_B)  # "95"
FOCUS_PROMPT: str = CALC_TEMPLATE.format(a=FOCUS_A, b=FOCUS_B)  # "calc: 36+59="


# ---------------------------------------------------------------------------
# C. Natural-language variant
#
# Anthropic's version: "Answer in one word. What is 36+59?\nAssistant:"
# We attribute from the *first* output token of the correct answer, not the
# explanation.  A follow-up string is also provided for the chain-of-thought
# comparison (see README.md for the exact experiment design).
# ---------------------------------------------------------------------------

NL_VARIANT: str = "Answer in one word. What is 36+59?\nAssistant:"

# Follow-up used to elicit an explanation (Anthropic's faithfulness check).
# Attribution for the case-study is run on NL_VARIANT, *not* on this.
NL_FOLLOWUP: str = (
    "Answer in one word. What is 36+59?\nAssistant: 95\nUser: Briefly, how did you get that?"
    "\nAssistant:"
)

# Qwen3-style chat-template variant (for exact fidelity with Qwen3's tokenizer).
# The plain NL_VARIANT is fine for most experiments; use this when you need the
# model to "see" a proper chat boundary.
NL_VARIANT_CHAT: str = (
    "<|im_start|>user\nAnswer in one word. What is 36+59?<|im_end|>\n" "<|im_start|>assistant\n"
)

NL_FOLLOWUP_CHAT: str = (
    "<|im_start|>user\nAnswer in one word. What is 36+59?<|im_end|>\n"
    "<|im_start|>assistant\n95<|im_end|>\n"
    "<|im_start|>user\nBriefly, how did you get that?<|im_end|>\n"
    "<|im_start|>assistant\n"
)


# ---------------------------------------------------------------------------
# D. Generalization prompts — addition as an intermediate step
# ---------------------------------------------------------------------------


class GeneralizationEntry(TypedDict):
    """A generalization prompt where addition is an intermediate step."""

    label: str
    prompt: str
    answer: str


GENERALIZATION_PROMPTS: list[GeneralizationEntry] = [
    # Anthropic's exact example (Python-style assertion)
    GeneralizationEntry(
        label="python_assert_product",
        prompt="assert (4 + 5) * 3 ==",
        answer="27",
    ),
    # Parenthesized addition before comparison
    GeneralizationEntry(
        label="python_assert_gte",
        prompt='assert (17 + 25) >= 40, "should be True"\n# (17 + 25) =',
        answer="42",
    ),
    # Math word-problem format
    GeneralizationEntry(
        label="word_problem",
        prompt="If Alice has 13 apples and Bob gives her 29 more, Alice has",
        answer="42",
    ),
    # Tabular / columnar
    GeneralizationEntry(
        label="table_sum",
        prompt="| Item | Count |\n| A | 47 |\n| B | 36 |\n| Total |",
        answer="83",
    ),
    # Obfuscated spacing — tests robustness outside calc: template
    GeneralizationEntry(
        label="spaced_equals",
        prompt="Evaluate: 28 + 55 =",
        answer="83",
    ),
    # Roman-numeral decode + addition (extra reasoning step)
    GeneralizationEntry(
        label="verbal_sum",
        prompt="Twenty-two plus nineteen equals",
        answer="41",
    ),
]
