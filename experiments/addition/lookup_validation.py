#!/usr/bin/env python3
"""Lookup table feature identification and causal validation.

Step 4 of the Anthropic addition case study reproduction on Qwen3-4B.

Given the 100x100 operand activation matrices from Step 2, this script
identifies transcoder features that act as lookup tables for specific
ones-digit pairs (e.g. ones(a)=6, ones(b)=9 → output ones digit 5),
then validates their causal role via two complementary interventions:

  Inhibition   — drives the feature below zero using negative alpha scaling;
                 the downstream prediction for the target ones digit should
                 weaken as alpha becomes more negative.

  Substitution — patches MLP inputs from a different ones-digit context
                 (calc: 39+59= instead of calc: 36+59=) into all layers
                 before the lookup feature's layer, without any feature
                 suppression.  If the lookup circuit is causal, the output
                 should shift from predicting ones digit 5 to ones digit 8.

Usage:
  python experiments/addition/lookup_validation.py \\
      --operand_plots_dir runs/addition/YYYY-MM-DD_HHMM/operand_plots \\
      --transcoder_set mwhanna/qwen3-4b-transcoders \\
      --out_dir runs/addition/YYYY-MM-DD_HHMM/lookup_validation

  # Override the auto-detected top candidate:
  python experiments/addition/lookup_validation.py ... \\
      --feature_layer 14 --feature_idx 3072

Outputs written to out_dir/:
  candidates.json           — top-k features ranked by lookup specificity
  inhibition_results.json   — alpha sweep: Δlogit / Δprob per alpha value
  substitution_results.json — context-swap experiment results
  summary.json              — one-line verdict for each test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature scoring
# ---------------------------------------------------------------------------


def score_lookup_feature(
    matrix: np.ndarray,
    ones_a: int = 6,
    ones_b: int = 9,
) -> float:
    """Score how specifically an activation matrix encodes an exact ones-digit pair.

    A lookup table feature for the pair (ones_a, ones_b) should concentrate
    its activation at the 200 cells where a%10==ones_a and b%10==ones_b, or
    their commutative counterpart (a%10==ones_b and b%10==ones_a).  Those 200
    cells account for 2% of the 10,000-cell matrix; a perfect lookup feature
    would score near 1.0, while a uniformly active feature scores near 0.02.

    Args:
        matrix: (100, 100) float32 array — matrix[a, b] = activation at '=' token.
        ones_a: Ones digit of the first operand.
        ones_b: Ones digit of the second operand.

    Returns:
        float in [0, 1].  Higher means stronger lookup specificity for this pair.
    """
    aa, bb = np.meshgrid(np.arange(100), np.arange(100), indexing="ij")
    mask = ((aa % 10 == ones_a) & (bb % 10 == ones_b)) | ((aa % 10 == ones_b) & (bb % 10 == ones_a))
    total = float(matrix.sum())
    if total < 1e-9:
        return 0.0
    return float(matrix[mask].sum() / total)


def identify_lookup_features(
    matrices: dict[tuple[int, int], np.ndarray],
    ones_a: int = 6,
    ones_b: int = 9,
    min_max_activation: float = 0.05,
    top_k: int = 10,
) -> list[dict]:
    """Rank all transcoder features by their lookup-table specificity.

    Features whose peak activation across the grid is below min_max_activation
    are skipped to avoid spurious high scores from near-zero matrices.

    Args:
        matrices: Dict from (layer, feat_idx) to (100, 100) activation array.
        ones_a: Target ones digit for the first operand.
        ones_b: Target ones digit for the second operand.
        min_max_activation: Skip features with max activation below this value.
        top_k: Number of top candidates to return.

    Returns:
        List of dicts sorted by descending score, each containing:
        layer, feat_idx, score, max_activation.
    """
    candidates = []
    for (layer, feat_idx), mat in matrices.items():
        peak = float(mat.max())
        if peak < min_max_activation:
            continue
        score = score_lookup_feature(mat, ones_a, ones_b)
        candidates.append(
            {
                "layer": layer,
                "feat_idx": feat_idx,
                "score": round(score, 4),
                "max_activation": round(peak, 4),
            }
        )

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:top_k]


def load_operand_matrices(operand_plots_dir: Path) -> dict[tuple[int, int], np.ndarray]:
    """Load all .npy activation matrices from an operand plots directory.

    Expects files named L{layer:02d}_F{feat_idx:06d}.npy, as written by
    experiments/addition/operand_plots.py.

    Args:
        operand_plots_dir: Directory containing the .npy files.

    Returns:
        Dict mapping (layer, feat_idx) to a (100, 100) float32 array.
    """
    matrices: dict[tuple[int, int], np.ndarray] = {}
    for npy_path in sorted(operand_plots_dir.glob("L*_F*.npy")):
        stem = npy_path.stem
        parts = stem.split("_")
        layer = int(parts[0][1:])
        feat_idx = int(parts[1][1:])
        matrices[(layer, feat_idx)] = np.load(str(npy_path)).astype(np.float32)
    return matrices


# ---------------------------------------------------------------------------
# Inhibition validation
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_inhibition_validation(
    model,
    tokens: torch.Tensor,
    feature_layer: int,
    feature_idx: int,
    target_token_id: int,
    alphas: list[float] | None = None,
) -> list[dict]:
    """Sweep a feature through negative alpha values and record logit shifts.

    The inhibit hook multiplies a feature's activation by alpha before
    decoding it back into residual space.  Setting alpha < 0 drives an
    originally positive activation below zero, disrupting the downstream
    signal it carries.  A causal lookup feature should produce a monotonic
    drop in the target token's logit as alpha decreases.

    Args:
        model: AttributionModel instance.
        tokens: Tokenized clean prompt, shape (1, n_pos).
        feature_layer: Layer index of the lookup feature.
        feature_idx: Feature index within the transcoder at that layer.
        target_token_id: Vocabulary id of the token to track (e.g. first token of "95").
        alphas: Scale factors to sweep.  Default: [1.0, 0.5, 0.0, -1.0, -5.0, -10.0].

    Returns:
        List of dicts, one per alpha, with keys:
        alpha, logit_clean, logit_inhibited, delta_logit, delta_prob.
    """
    from mechinterp_qwen3.interventions import compute_logit_diff, inhibit_features

    if alphas is None:
        alphas = [1.0, 0.5, 0.0, -1.0, -5.0, -10.0]

    baseline_logits = model(tokens)
    baseline_logit = float(baseline_logits[0, -1, target_token_id].item())

    results = []
    for alpha in alphas:
        intervened = inhibit_features(model, tokens, {feature_layer: [feature_idx]}, alpha=alpha)
        d_logit, d_prob = compute_logit_diff(baseline_logits, intervened, target_token_id)
        results.append(
            {
                "alpha": alpha,
                "logit_clean": round(baseline_logit, 4),
                "logit_inhibited": round(baseline_logit + d_logit, 4),
                "delta_logit": round(d_logit, 4),
                "delta_prob": round(d_prob, 6),
            }
        )
        log.info("  alpha=%+5.1f  Δlogit=%+.3f  Δp=%+.6f", alpha, d_logit, d_prob)

    return results


# ---------------------------------------------------------------------------
# Substitution validation
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_substitution_validation(
    model,
    tokens_clean: torch.Tensor,
    tokens_sub: torch.Tensor,
    feature_layer: int,
    target_token_ids: dict[str, int],
) -> dict:
    """Patch ones-digit context from a substitution prompt into the clean run.

    All MLP inputs at layers strictly before feature_layer are replaced with
    the activations cached from the substitution run, while the rest of the
    network executes normally on the clean prompt.  No feature suppression is
    applied.  If the lookup circuit is causally responsible for the ones-digit
    prediction, this context swap should shift the model's output from the
    clean answer to the substitution answer without any explicit intervention
    at the feature level.

    Args:
        model: AttributionModel instance.
        tokens_clean: Tokenized clean prompt (calc: 36+59=), shape (1, n_pos).
        tokens_sub: Tokenized substitution prompt (calc: 39+59=), shape (1, n_pos).
        feature_layer: Clamping boundary; layers [0, feature_layer) are patched.
        target_token_ids: Dict mapping a human-readable label to a vocabulary id.

    Returns:
        Dict with fields:
          feature_layer, clean_top1_token_id, sub_top1_token_id,
          patched_top1_token_id, prediction_shifted, per_token.
    """
    from mechinterp_qwen3.interventions import collect_mlp_inputs, compute_logit_diff

    baseline_logits = model(tokens_clean)
    sub_baseline_logits = model(tokens_sub)

    sub_acts = collect_mlp_inputs(model, tokens_sub)

    clamp_hooks = []
    for layer in range(feature_layer):
        if layer not in sub_acts:
            continue
        clamp_val = sub_acts[layer].to(model.cfg.device)

        def _clamp(acts: torch.Tensor, hook, *, _v: torch.Tensor = clamp_val) -> torch.Tensor:
            return _v.to(acts.dtype)

        clamp_hooks.append((f"blocks.{layer}.{model.feature_input_hook}", _clamp))

    patched_logits = model.run_with_hooks(tokens_clean, fwd_hooks=clamp_hooks)

    top_clean = int(baseline_logits[0, -1].argmax().item())
    top_sub = int(sub_baseline_logits[0, -1].argmax().item())
    top_patched = int(patched_logits[0, -1].argmax().item())

    per_token = {}
    for label, tok_id in target_token_ids.items():
        d_logit, d_prob = compute_logit_diff(baseline_logits, patched_logits, tok_id)
        per_token[label] = {
            "token_id": tok_id,
            "delta_logit": round(d_logit, 4),
            "delta_prob": round(d_prob, 6),
        }
        log.info("  [%s] tok_id=%d  Δlogit=%+.3f  Δp=%+.6f", label, tok_id, d_logit, d_prob)

    return {
        "feature_layer": feature_layer,
        "clean_top1_token_id": top_clean,
        "sub_top1_token_id": top_sub,
        "patched_top1_token_id": top_patched,
        "prediction_shifted": top_patched == top_sub,
        "per_token": per_token,
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_lookup_validation(
    operand_plots_dir: Path,
    transcoder_set: str,
    out_dir: Path,
    *,
    model_name: str | None = None,
    dtype: str = "float32",
    ones_a: int = 6,
    ones_b: int = 9,
    top_k_candidates: int = 10,
    feature_layer: int | None = None,
    feature_idx: int | None = None,
    alphas: list[float] | None = None,
) -> dict:
    """Run the full lookup feature identification and validation pipeline.

    Steps:
      1. Load .npy operand matrices from operand_plots_dir.
      2. Score every feature for specificity to the (ones_a, ones_b) pair.
      3. Load the model and tokenize the clean and substitution prompts.
      4. Run the inhibition sweep on the top candidate (or the overridden feature).
      5. Run the substitution experiment at the feature's layer boundary.
      6. Write all results to out_dir/ as JSON files.

    Args:
        operand_plots_dir: Directory with L{layer}_F{feat}.npy files from Step 2.
        transcoder_set: HuggingFace repo id for the transcoder set.
        out_dir: Output directory for results JSON files.
        model_name: HuggingFace model identifier.
        dtype: Model weight dtype string ("float32", "bfloat16", "float16").
        ones_a: Ones digit of the first operand for the target lookup pair.
        ones_b: Ones digit of the second operand for the target lookup pair.
        top_k_candidates: How many candidates to surface in candidates.json.
        feature_layer: Override auto-detected layer (None = use top candidate).
        feature_idx: Override auto-detected feature index (None = use top candidate).
        alphas: Alpha values for the inhibition sweep.

    Returns:
        Dict with keys "candidates", "inhibition", "substitution".
    """
    _REPO_ROOT = Path(__file__).resolve().parent.parent.parent
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    from experiments.addition.dataset_generation.generate_dataset_with_predictions import (
        TemplateID,
        build_prompt,
    )
    from experiments.addition.expected_answers import get_target_tokens
    from experiments.addition.prompts import FOCUS_A, FOCUS_B, FOCUS_PROMPT
    from mechinterp_qwen3.attribution_model import AttributionModel
    from mechinterp_qwen3.utils.hf_utils import load_transcoder_from_hub
    from mechinterp_qwen3.utils.token_utils import tokenize_qwen_input

    out_dir.mkdir(parents=True, exist_ok=True)

    # --- step 1: load operand matrices ---
    log.info("Loading operand matrices from %s", operand_plots_dir)
    matrices = load_operand_matrices(operand_plots_dir)
    if not matrices:
        raise FileNotFoundError(
            f"No .npy files found in {operand_plots_dir}. "
            "Run --operand-plots first to generate them."
        )
    log.info("  %d matrices loaded", len(matrices))

    # --- step 2: rank candidates ---
    log.info("Scoring features for ones_a=%d, ones_b=%d", ones_a, ones_b)
    candidates = identify_lookup_features(
        matrices, ones_a=ones_a, ones_b=ones_b, top_k=top_k_candidates
    )
    if not candidates:
        raise RuntimeError(
            "No features passed the min_max_activation threshold. "
            "Try lowering --min_max_activation or check the operand plots directory."
        )

    best = candidates[0]
    log.info(
        "  top candidate: L%d F%d  score=%.3f  peak=%.3f",
        best["layer"],
        best["feat_idx"],
        best["score"],
        best["max_activation"],
    )

    with open(out_dir / "candidates.json", "w") as f:
        json.dump({"ones_a": ones_a, "ones_b": ones_b, "candidates": candidates}, f, indent=2)
    log.info("candidates.json written")

    # Resolve which feature to validate
    target_layer = feature_layer if feature_layer is not None else best["layer"]
    target_feat = feature_idx if feature_idx is not None else best["feat_idx"]
    log.info("Validating L%d F%d", target_layer, target_feat)

    # --- step 3: load model ---
    dtype_obj = getattr(torch, dtype)
    log.info("Loading transcoders from %r", transcoder_set)
    transcoder, config = load_transcoder_from_hub(
        transcoder_set, dtype=dtype_obj, lazy_decoder=True
    )
    resolved_model = model_name or config.get("model_name", default_model())
    log.info("Loading model %r  dtype=%s", resolved_model, dtype)
    model = AttributionModel.from_pretrained_and_transcoders(
        resolved_model, transcoder, dtype=dtype_obj
    )
    model.eval()

    device = model.cfg.device

    # --- tokenize prompts ---
    tokens_clean = tokenize_qwen_input(FOCUS_PROMPT, model.tokenizer, device).unsqueeze(0)

    # Substitution prompt: keep tens digit of a, change ones digit to ones_b.
    # E.g. for FOCUS_A=36, ones_b=9: sub_a = 39.
    sub_a = (FOCUS_A // 10) * 10 + ones_b
    sub_prompt = build_prompt(TemplateID.T0, sub_a, FOCUS_B)
    tokens_sub = tokenize_qwen_input(sub_prompt, model.tokenizer, device).unsqueeze(0)
    log.info("Clean prompt: %r", FOCUS_PROMPT)
    log.info("Sub prompt:   %r  (ones digits now %d+%d)", sub_prompt, ones_b, FOCUS_B % 10)

    # --- resolve token ids ---
    clean_answer_toks = get_target_tokens(FOCUS_PROMPT, model)
    target_token_id = clean_answer_toks[0]

    sub_answer = str(sub_a + FOCUS_B)
    sub_answer_toks = get_target_tokens(sub_prompt, model, span=sub_answer)
    target_token_ids = {
        f"answer_{FOCUS_A + FOCUS_B}": clean_answer_toks[0],
        f"answer_{sub_a + FOCUS_B}": sub_answer_toks[0],
    }
    log.info("Tracking tokens: %s", target_token_ids)

    # --- step 4: inhibition sweep ---
    log.info("Running inhibition sweep on L%d F%d ...", target_layer, target_feat)
    inhibition_results = run_inhibition_validation(
        model, tokens_clean, target_layer, target_feat, target_token_id, alphas
    )
    with open(out_dir / "inhibition_results.json", "w") as f:
        json.dump(
            {
                "feature_layer": target_layer,
                "feature_idx": target_feat,
                "target_token_id": target_token_id,
                "results": inhibition_results,
            },
            f,
            indent=2,
        )
    log.info("inhibition_results.json written")

    # --- step 5: substitution experiment ---
    log.info("Running substitution experiment (clamping layers 0..%d) ...", target_layer - 1)
    substitution_results = run_substitution_validation(
        model, tokens_clean, tokens_sub, target_layer, target_token_ids
    )
    with open(out_dir / "substitution_results.json", "w") as f:
        json.dump(
            {
                "clean_prompt": FOCUS_PROMPT,
                "sub_prompt": sub_prompt,
                "results": substitution_results,
            },
            f,
            indent=2,
        )
    log.info("substitution_results.json written")

    # --- summary ---
    inh_neg5 = next((r for r in inhibition_results if r["alpha"] == -5.0), None)
    summary = {
        "target_ones_pair": [ones_a, ones_b],
        "validated_feature": {"layer": target_layer, "feat_idx": target_feat},
        "top_candidate_score": best["score"],
        "inhibition_at_neg5": inh_neg5,
        "substitution_shifted": substitution_results.get("prediction_shifted"),
        "clean_top1": substitution_results.get("clean_top1_token_id"),
        "sub_top1": substitution_results.get("sub_top1_token_id"),
        "patched_top1": substitution_results.get("patched_top1_token_id"),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓  lookup validation complete → {out_dir}")
    print(f"   Top candidate:  L{target_layer} F{target_feat}  (score={best['score']:.3f})")
    if inh_neg5:
        print(
            f"   Inhibition (α=-5):  Δlogit={inh_neg5['delta_logit']:+.3f}"
            f"  Δp={inh_neg5['delta_prob']:+.6f}"
        )
    print(f"   Substitution shifted prediction:  {substitution_results.get('prediction_shifted')}")

    return {
        "candidates": candidates,
        "inhibition": inhibition_results,
        "substitution": substitution_results,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lookup_validation.py",
        description=(
            "Identify and validate lookup-table features for ones-digit addition. "
            "Requires operand plot .npy files from Step 2 (--operand-plots)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--operand_plots_dir",
        required=True,
        help="Directory containing L{layer}_F{feat}.npy matrices from Step 2.",
    )
    p.add_argument(
        "-t",
        "--transcoder_set",
        default=default_transcoder_set(),
        help="HuggingFace repo id for the transcoder set.",
    )
    p.add_argument(
        "-m",
        "--model",
        default=default_model(),
        help="HuggingFace model name.",
    )
    p.add_argument(
        "-o",
        "--out_dir",
        required=True,
        help="Output directory for validation results.",
    )
    p.add_argument(
        "--ones_a",
        type=int,
        default=6,
        help="Ones digit of first operand for the target lookup pair.",
    )
    p.add_argument(
        "--ones_b",
        type=int,
        default=9,
        help="Ones digit of second operand for the target lookup pair.",
    )
    p.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of feature candidates to surface in candidates.json.",
    )
    p.add_argument(
        "--feature_layer",
        type=int,
        default=None,
        help="Override auto-detected feature layer (skips auto-detection).",
    )
    p.add_argument(
        "--feature_idx",
        type=int,
        default=None,
        help="Override auto-detected feature index (skips auto-detection).",
    )
    p.add_argument(
        "--dtype",
        choices=["float32", "bfloat16", "float16"],
        default="float32",
        help="Model weight dtype.",
    )
    p.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=None,
        help=("Alpha values for the inhibition sweep. Default: 1.0 0.5 0.0 -1.0 -5.0 -10.0"),
    )
    return p


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    p = build_parser()
    args = p.parse_args()

    run_lookup_validation(
        operand_plots_dir=Path(args.operand_plots_dir),
        transcoder_set=args.transcoder_set,
        out_dir=Path(args.out_dir),
        model_name=args.model,
        dtype=args.dtype,
        ones_a=args.ones_a,
        ones_b=args.ones_b,
        top_k_candidates=args.top_k,
        feature_layer=args.feature_layer,
        feature_idx=args.feature_idx,
        alphas=args.alphas,
    )


if __name__ == "__main__":
    main()
