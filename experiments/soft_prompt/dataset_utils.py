"""Dataset loading utilities for the soft-prompt experiment.

Pairs are generated on-the-fly (deterministic, seeded) by reusing the
_load_concept registry already defined in run_concept.py.  No pre-stored
file is needed or created.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("soft_prompt.dataset_utils")


def _get_pairs(concept: str, template: str, n_per_template: int, seed: int):
    """Generate ConceptPairs via the existing registry in run_concept.py."""
    from experiments.concept_localization.pipeline.run_concept import _load_concept

    mod_name = f"experiments.concept_localization.concept_datasets.{concept}_dataset"
    import importlib
    mod = importlib.import_module(mod_name)
    if template not in mod.TEMPLATES:
        raise ValueError(
            f"Template '{template}' not available for concept '{concept}'. "
            f"Available: {list(mod.TEMPLATES)}"
        )
    pairs = _load_concept(concept, n_per_template, seed)
    # _load_concept returns all templates; keep only the requested one
    pairs = [p for p in pairs if p.template == template]
    log.info(
        "Loaded %d pairs for concept=%s template=%s", len(pairs), concept, template
    )
    return pairs


def load_concept_pairs(
    concept: str,
    template: str = "T0",
    n_per_template: int = 500,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Return flat list of dicts (no tokenizer needed — for visualization)."""
    pairs = _get_pairs(concept, template, n_per_template, seed)
    samples: list[dict[str, Any]] = []
    for p in pairs:
        for prompt_str, predict_str, label in [
            (p.prompt_pos, p.predict_pos, p.label_pos),
            (p.prompt_neg, p.predict_neg, p.label_neg),
        ]:
            entry: dict[str, Any] = {
                "prompt": prompt_str,
                "true_answer_str": predict_str,
                "label": label,
                "template": template,
            }
            entry.update(p.meta)
            samples.append(entry)
    return samples


def load_concept_dataset(
    concept: str,
    tokenizer,
    template: str = "T0",
    n_per_template: int = 500,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Return tokenised samples (prompt_token_ids, answer_token_ids) for training/eval.

    Each ConceptPair contributes two samples (pos + neg prompt).
    Tokenisation uses the HF tokenizer with add_special_tokens=False.
    """
    pairs = _get_pairs(concept, template, n_per_template, seed)
    samples: list[dict[str, Any]] = []
    for p in pairs:
        for prompt_str, predict_str, label in [
            (p.prompt_pos, p.predict_pos, p.label_pos),
            (p.prompt_neg, p.predict_neg, p.label_neg),
        ]:
            prompt_ids = tokenizer(prompt_str, add_special_tokens=False).input_ids
            answer_ids = tokenizer(predict_str, add_special_tokens=False).input_ids
            entry: dict[str, Any] = {
                "prompt_token_ids": prompt_ids,
                "answer_token_ids": answer_ids,
                "prompt": prompt_str,
                "true_answer_str": predict_str,
                "label": label,
                "template": template,
            }
            entry.update(p.meta)
            samples.append(entry)
    log.info("Total tokenised samples: %d", len(samples))
    return samples
