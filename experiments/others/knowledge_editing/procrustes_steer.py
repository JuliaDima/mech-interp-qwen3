"""Procrustes-alignment steering experiment.

Derives a steering vector for Qwen3-4B's carry circuit by:

  1. Collecting paired activations at the '=' token for N samples:
       A_small ∈ ℝ^{N × d_small}  — small addition model, resid_post at last layer
       A_large ∈ ℝ^{N × d_large}  — large model, resid_post at inject_layer

  2. Fitting a linear map  W: ℝ^{d_small} → ℝ^{d_large}  via one of:
       "lstsq"     — unconstrained least-squares  (A_small @ W.T ≈ A_large)
       "procrustes" — orthogonality-constrained via SVD of  A_large.T @ A_small

  3. Extracting the carry direction in the small model's space:
       carry_dir_small = mean(A_small | carry=1) − mean(A_small | carry=0)

  4. Translating to the large model's space:
       sv = W @ carry_dir_small  ∈ ℝ^{d_large}

  5. Sweeping  alpha * sv  at (inject_layer, eq_pos)  — same protocol as steer.py.

  6. Sanity check: cos(sv, contrastive_sv_from_steer)  and top SAE features of sv.

Pipeline stages:

  --setup    Collect activations, fit W, compute sv, save.
  --steer    Sweep alpha × sv; report accuracy delta.
  --analyze  Cosine alignment with contrastive sv + SAE feature decomposition.
  --all      setup → steer → analyze

Usage:

    python experiments/knowledge_editing/procrustes_steer.py --all
    python experiments/knowledge_editing/procrustes_steer.py --setup --method procrustes
    python experiments/knowledge_editing/procrustes_steer.py --steer --alphas 0.5 1 2 5 10
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in [str(_REPO_ROOT), str(_REPO_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mechinterp_qwen3.attribution_model import AttributionModel  # noqa: E402
from mechinterp_qwen3.probe.label_utils import compute_carry_label  # noqa: E402
from mechinterp_qwen3.utils.hf_utils import load_transcoder_from_hub  # noqa: E402
from mechinterp_qwen3.utils.model_utils import get_default_device, parse_dtype  # noqa: E402
from scripts.model_config import default_model, default_transcoder_set
from mechinterp_qwen3.utils.token_utils import tokenize_qwen_input  # noqa: E402

_OTHERS_DIR = str(Path(__file__).parent.parent)
if _OTHERS_DIR not in sys.path:
    sys.path.insert(0, _OTHERS_DIR)

from common.small_addition import (  # noqa: E402
    _load_small_model,
    get_small_model_tokenizer,
    load_addition_dataset,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("procrustes_steer")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_eq_pos_large(token_ids: torch.Tensor, tokenizer) -> int | None:
    eq_ids = tokenizer("=", add_special_tokens=False).input_ids
    if not eq_ids:
        return None
    eq_id = eq_ids[-1]
    positions = (token_ids == eq_id).nonzero(as_tuple=True)[0]
    return int(positions[-1]) if len(positions) > 0 else None


def _eq_pos_small(prompt: str) -> int | None:
    idx = prompt.rfind("=")
    return idx if idx >= 0 else None


# ---------------------------------------------------------------------------
# Stage: setup — collect activations + fit W + compute sv
# ---------------------------------------------------------------------------


def run_setup(args: argparse.Namespace, device: torch.device) -> None:
    out_dir = Path(args.out_root) / "setup"
    out_dir.mkdir(parents=True, exist_ok=True)

    sv_path = out_dir / f"sv_{args.method}.pt"
    W_path = out_dir / f"W_{args.method}.pt"
    meta_path = out_dir / "setup_meta.json"

    if sv_path.exists() and not args.force:
        log.info("Setup already done (use --force to redo). sv at %s", sv_path)
        return

    log.info("=== Stage: setup [%s] ===", args.method)
    dtype = parse_dtype(args.dtype)

    # Load large model
    log.info("Loading large model %s", args.model)
    transcoder_set, _ = load_transcoder_from_hub(
        args.transcoder_set, dtype=dtype, lazy_encoder=False, lazy_decoder=False
    )
    large_model = AttributionModel.from_pretrained_and_transcoders(
        args.model, transcoder_set, dtype=dtype, device=device
    )
    large_model.eval()
    for p in large_model.parameters():
        p.requires_grad_(False)

    tokenizer = large_model.tokenizer

    # Load small model
    log.info("Loading small model from %s", args.small_model_path)
    small_model = _load_small_model(args, Path(args.small_model_path), device)
    small_model.model.eval()
    for p in small_model.model.parameters():
        p.requires_grad_(False)

    small_tokenize = get_small_model_tokenizer(small_model)
    sae_layer = small_model.n_layers - 1
    d_small = small_model.model.cfg.d_model

    # Dataset
    samples = load_addition_dataset(args.dataset_path, max_samples=args.max_samples)
    log.info("Collecting activations from %d samples", len(samples))

    small_acts_list: list[torch.Tensor] = []  # (d_small,) each
    large_acts_list: list[torch.Tensor] = []  # (d_large,) each
    carry_labels: list[int] = []

    with torch.no_grad():
        for sample in tqdm(samples, desc="Collecting activations"):
            a, b = sample.get("a", 0), sample.get("b", 0)
            answer_str = sample.get("answer", sample.get("true_answer_str", str(a + b)))
            small_prompt = f"{a}+{b}={answer_str}"
            eq_pos_s = _eq_pos_small(small_prompt)
            if eq_pos_s is None:
                continue

            # --- Small model ---
            small_ids = torch.tensor(
                [small_tokenize(small_prompt)], device=device, dtype=torch.long
            )
            small_cache: dict[str, torch.Tensor] = {}

            def _hook_small(act, hook, _pos=eq_pos_s, cache=small_cache):
                cache["resid"] = act[:, _pos, :].detach().clone()
                return act

            small_model.model.run_with_hooks(
                small_ids,
                fwd_hooks=[(f"blocks.{sae_layer}.hook_resid_post", _hook_small)],
            )
            if "resid" not in small_cache:
                continue
            v_small = small_cache["resid"].squeeze(0).float()  # (d_small,)

            # --- Large model ---
            prompt_str = sample.get("prompt_str", sample.get("prompt", ""))
            large_ids = tokenize_qwen_input(prompt_str, tokenizer, device=device).unsqueeze(0)
            eq_pos_l = _find_eq_pos_large(large_ids.squeeze(0), tokenizer)
            if eq_pos_l is None:
                continue

            large_cache: dict[str, torch.Tensor] = {}

            def _hook_large(act, hook, _pos=eq_pos_l, cache=large_cache):
                cache["resid"] = act[:, _pos, :].detach().clone()
                return act

            large_model.run_with_hooks(
                large_ids,
                fwd_hooks=[(f"blocks.{args.inject_layer}.hook_resid_post", _hook_large)],
            )
            if "resid" not in large_cache:
                continue
            v_large = large_cache["resid"].squeeze(0).float()  # (d_large,)

            small_acts_list.append(v_small.cpu())
            large_acts_list.append(v_large.cpu())
            carry_labels.append(compute_carry_label(a, b))

    N = len(small_acts_list)
    log.info("Collected %d paired activation samples", N)

    A_small = torch.stack(small_acts_list)  # (N, d_small)
    A_large = torch.stack(large_acts_list)  # (N, d_large)
    carry_t = torch.tensor(carry_labels, dtype=torch.float32)

    # --- Fit W: d_small → d_large ---
    log.info("Fitting alignment map W via method='%s'", args.method)
    W = _fit_alignment(A_small, A_large, method=args.method)  # (d_large, d_small)

    # --- Carry direction in small model space ---
    carry_mask = carry_t.bool()
    mean_carry = A_small[carry_mask].mean(0)  # (d_small,)
    mean_no_carry = A_small[~carry_mask].mean(0)  # (d_small,)
    carry_dir_small = mean_carry - mean_no_carry  # (d_small,)

    n_carry = carry_mask.sum().item()
    n_no_carry = (~carry_mask).sum().item()
    log.info(
        "Carry direction computed: %d carry / %d no-carry samples  |dir|=%.4f",
        n_carry,
        n_no_carry,
        carry_dir_small.norm().item(),
    )

    # --- Translate ---
    sv = (W @ carry_dir_small.to(W.device)).float()  # (d_large,)
    log.info("Steering vector computed  |sv|=%.4f", sv.norm().item())

    torch.save(sv, sv_path)
    torch.save(W, W_path)

    d_large = A_large.shape[1]
    meta = {
        "method": args.method,
        "n_samples": N,
        "n_carry": int(n_carry),
        "n_no_carry": int(n_no_carry),
        "d_small": d_small,
        "d_large": d_large,
        "inject_layer": args.inject_layer,
        "sv_norm": sv.norm().item(),
        "carry_dir_small_norm": carry_dir_small.norm().item(),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    del large_model, small_model, transcoder_set, A_small, A_large
    torch.cuda.empty_cache()
    log.info("Setup complete — outputs in %s", out_dir)


def _fit_alignment(
    A_small: torch.Tensor,  # (N, d_small)
    A_large: torch.Tensor,  # (N, d_large)
    method: str = "lstsq",
) -> torch.Tensor:
    """Fit W: ℝ^{d_small} → ℝ^{d_large}.

    Returns W of shape (d_large, d_small).
    Apply as: v_large = W @ v_small
    """
    # Center both matrices
    A_s = A_small - A_small.mean(0, keepdim=True)
    A_l = A_large - A_large.mean(0, keepdim=True)

    if method == "lstsq":
        # Solve A_s @ W.T ≈ A_l  →  W.T = lstsq(A_s, A_l)
        # lstsq returns X of shape (d_small, d_large) s.t. A_s @ X ≈ A_l
        result = torch.linalg.lstsq(A_s, A_l, rcond=None)
        W_T = result.solution  # (d_small, d_large)
        W = W_T.T  # (d_large, d_small)

    elif method == "procrustes":
        # Orthogonal Procrustes: minimise ||A_l − A_s @ W.T||_F  s.t. W orthogonal
        # (rectangular: d_large × d_small, d_large >= d_small typically not true here)
        # M = A_l.T @ A_s  shape (d_large, d_small)
        # SVD(M) = U S Vt  →  W = U @ Vt  (d_large × d_small)
        M = A_l.T @ A_s  # (d_large, d_small)
        U, S, Vt = torch.linalg.svd(M, full_matrices=False)
        W = U @ Vt  # (d_large, d_small) — semi-orthogonal

    else:
        raise ValueError(f"Unknown method: {method!r}")

    log.info("W fitted — shape %s  |W|_F=%.4f", tuple(W.shape), W.norm().item())
    return W


# ---------------------------------------------------------------------------
# Stage: steer — sweep alpha × sv
# ---------------------------------------------------------------------------


def run_steer(args: argparse.Namespace, device: torch.device) -> None:
    out_dir = Path(args.out_root)
    setup_dir = out_dir / "setup"
    sv_path = setup_dir / f"sv_{args.method}.pt"

    log.info("=== Stage: steer ===")
    dtype = parse_dtype(args.dtype)

    sv: torch.Tensor = torch.load(sv_path, weights_only=True).to(device=device, dtype=dtype)
    sv_unit = F.normalize(sv.unsqueeze(0)).squeeze(0)  # unit norm for stable scaling

    with open(setup_dir / "setup_meta.json") as f:
        meta = json.load(f)
    inject_layer: int = meta["inject_layer"]

    log.info("Loading large model %s", args.model)
    transcoder_set, _ = load_transcoder_from_hub(
        args.transcoder_set, dtype=dtype, lazy_encoder=True, lazy_decoder=True
    )
    model = AttributionModel.from_pretrained_and_transcoders(
        args.model, transcoder_set, dtype=dtype, device=device
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    tokenizer = model.tokenizer

    samples = load_addition_dataset(args.dataset_path, max_samples=args.max_samples)
    eval_samples = samples[int(0.9 * len(samples)) :]
    log.info("Evaluating on %d samples", len(eval_samples))

    # Baseline
    n_base_correct = 0
    n_total = 0
    with torch.no_grad():
        for sample in tqdm(eval_samples, desc="Baseline"):
            ids, target_id, eq_pos = _get_sample_tensors(sample, tokenizer, device)
            if ids is None:
                continue
            logits = model(ids.unsqueeze(0))
            pred = int(logits[0, -1, :].argmax())
            n_base_correct += int(pred == target_id)
            n_total += 1

    acc_base = 100.0 * n_base_correct / max(n_total, 1)
    log.info("Baseline accuracy: %.2f%%", acc_base)

    results = []
    for alpha in args.alphas:
        n_correct = 0
        with torch.no_grad():
            for sample in tqdm(eval_samples, desc=f"Steer α={alpha}", leave=False):
                ids, target_id, eq_pos = _get_sample_tensors(sample, tokenizer, device)
                if ids is None:
                    continue

                _sv = sv_unit * alpha

                def _steer(act, hook, _sv=_sv, _pos=eq_pos):
                    act = act.clone()
                    act[0, _pos, :] = act[0, _pos, :] + _sv
                    return act

                logits = model.run_with_hooks(
                    ids.unsqueeze(0),
                    fwd_hooks=[(f"blocks.{inject_layer}.hook_resid_post", _steer)],
                )
                pred = int(logits[0, -1, :].argmax())
                n_correct += int(pred == target_id)

        acc = 100.0 * n_correct / max(n_total, 1)
        delta = acc - acc_base
        log.info("α=%5.1f  acc=%.2f%%  Δ=%+.2f%%", alpha, acc, delta)
        results.append({"alpha": alpha, "acc": acc, "delta_acc": delta})

    results_path = out_dir / f"steer_results_{args.method}.json"
    with open(results_path, "w") as f:
        json.dump({"baseline_acc": acc_base, "results": results}, f, indent=2)
    log.info("Results saved to %s", results_path)

    del model, transcoder_set, sv, sv_unit
    torch.cuda.empty_cache()


def _get_sample_tensors(
    sample: dict,
    tokenizer,
    device: torch.device,
) -> tuple[torch.Tensor | None, int | None, int | None]:
    """Return (token_ids_1d, target_id, eq_pos) or (None, None, None) on failure."""
    if "prompt_token_ids" in sample:
        ids = torch.tensor(sample["prompt_token_ids"], dtype=torch.long, device=device)
        answer_ids = sample.get("answer_token_ids", [])
    else:
        prompt_str = sample.get("prompt_str", sample.get("prompt", ""))
        ids = tokenize_qwen_input(prompt_str, tokenizer, device=device)
        ans = sample.get("true_answer_str", sample.get("answer", ""))
        answer_ids = tokenizer(ans, add_special_tokens=False).input_ids

    if not answer_ids:
        return None, None, None
    eq_pos = _find_eq_pos_large(ids, tokenizer)
    if eq_pos is None:
        return None, None, None
    return ids, answer_ids[0], eq_pos


# ---------------------------------------------------------------------------
# Stage: analyze — cosine alignment + SAE features
# ---------------------------------------------------------------------------


def run_analyze(args: argparse.Namespace, device: torch.device) -> None:
    out_dir = Path(args.out_root)
    setup_dir = out_dir / "setup"
    sv_path = setup_dir / f"sv_{args.method}.pt"

    log.info("=== Stage: analyze ===")
    dtype = parse_dtype(args.dtype)

    sv: torch.Tensor = torch.load(sv_path, weights_only=True).to(device=device, dtype=dtype)

    with open(setup_dir / "setup_meta.json") as f:
        meta = json.load(f)
    inject_layer: int = meta["inject_layer"]

    log.info("Loading large model for analysis")
    transcoder_set, _ = load_transcoder_from_hub(
        args.transcoder_set, dtype=dtype, lazy_encoder=False, lazy_decoder=False
    )
    model = AttributionModel.from_pretrained_and_transcoders(
        args.model, transcoder_set, dtype=dtype, device=device
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    tokenizer = model.tokenizer

    # ------------------------------------------------------------------
    # 1. Contrastive steering vector (correct − wrong) from large model
    #    Same method as steer.py — use as reference direction
    # ------------------------------------------------------------------
    samples = load_addition_dataset(args.dataset_path, max_samples=args.max_samples)
    eval_samples = samples[int(0.9 * len(samples)) :][: args.analyze_n]

    correct_acts: list[torch.Tensor] = []
    wrong_acts: list[torch.Tensor] = []

    log.info("Collecting contrastive activations from %d samples", len(eval_samples))
    with torch.no_grad():
        for sample in tqdm(eval_samples, desc="Contrastive"):
            ids, target_id, eq_pos = _get_sample_tensors(sample, tokenizer, device)
            if ids is None:
                continue

            cache: dict = {}

            def _cap(act, hook, _pos=eq_pos, cache=cache):
                cache["h"] = act[0, _pos, :].detach().clone()
                return act

            logits = model.run_with_hooks(
                ids.unsqueeze(0),
                fwd_hooks=[(f"blocks.{inject_layer}.hook_resid_post", _cap)],
            )
            if "h" not in cache:
                continue

            pred = int(logits[0, -1, :].argmax())
            bucket = correct_acts if pred == target_id else wrong_acts
            bucket.append(cache["h"].float().cpu())

    n_c, n_w = len(correct_acts), len(wrong_acts)
    log.info("Contrastive: %d correct / %d wrong", n_c, n_w)

    analysis: dict[str, Any] = {"method": args.method, "inject_layer": inject_layer}

    if n_c > 0 and n_w > 0:
        n = min(n_c, n_w)
        sv_contrastive = (
            torch.stack(correct_acts[:n]).mean(0) - torch.stack(wrong_acts[:n]).mean(0)
        ).to(device=device, dtype=dtype)

        cos_sv = F.cosine_similarity(sv.unsqueeze(0), sv_contrastive.unsqueeze(0)).item()
        log.info(
            "cos(procrustes_sv, contrastive_sv) = %.4f  "
            "|procrustes_sv|=%.4f  |contrastive_sv|=%.4f",
            cos_sv,
            sv.norm().item(),
            sv_contrastive.norm().item(),
        )
        analysis["cos_procrustes_vs_contrastive"] = cos_sv
        analysis["sv_norm"] = sv.norm().item()
        analysis["sv_contrastive_norm"] = sv_contrastive.norm().item()
    else:
        log.warning("Not enough samples for contrastive sv")

    # ------------------------------------------------------------------
    # 2. SAE feature decomposition of sv
    # ------------------------------------------------------------------
    log.info("SAE feature decomposition of steering vector at layer %d", inject_layer)
    transcoder = transcoder_set.transcoders[inject_layer]

    feat_acts = transcoder.encode(sv.unsqueeze(0).unsqueeze(0))  # (1, 1, d_tc)
    feat_acts = feat_acts.squeeze()  # (d_tc,)

    topk_vals, topk_idx = feat_acts.abs().topk(args.top_k_features)
    log.info("Top-%d SAE features in procrustes steering vector:", args.top_k_features)
    top_features = []
    for rank, (idx, val) in enumerate(zip(topk_idx.tolist(), topk_vals.tolist(), strict=False), 1):
        act_val = feat_acts[idx].item()
        log.info("  #%2d  feature %6d  |act|=%.4f  act=%.4f", rank, idx, val, act_val)
        top_features.append({"rank": rank, "feature_idx": int(idx), "activation": float(act_val)})

    analysis["top_sae_features"] = top_features

    out_path = out_dir / f"analysis_{args.method}.json"
    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2)
    log.info("Analysis saved to %s", out_path)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Procrustes-alignment steering for carry circuit",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--setup", action="store_true")
    p.add_argument("--steer", action="store_true")
    p.add_argument("--analyze", action="store_true")
    p.add_argument("--all", action="store_true", help="setup → steer → analyze")
    p.add_argument("--force", action="store_true", help="Redo setup even if cached")

    # Alignment method
    p.add_argument(
        "--method",
        default="lstsq",
        choices=["lstsq", "procrustes"],
        help="Linear map fitting method",
    )

    # Models
    p.add_argument("--model", default=default_model())
    p.add_argument("--transcoder_set", default=default_transcoder_set())
    p.add_argument(
        "--small_model_path",
        default="models/small_addition_model.pt",
        help="Path to small addition model checkpoint",
    )
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument(
        "--inject_layer", type=int, default=14, help="Large model layer to inject/capture at"
    )

    # Small model architecture (forwarded to _load_small_model)
    p.add_argument("--small_model_type", default="quanta", choices=["quanta", "scratch"])
    p.add_argument("--small_n_layers", type=int, default=2)
    p.add_argument("--small_d_model", type=int, default=128)
    p.add_argument("--small_n_heads", type=int, default=4)

    # Data
    p.add_argument("--dataset_path", default="data/addition_3digit.jsonl")
    p.add_argument("--max_samples", type=int, default=None)

    # Steer
    p.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0])

    # Analyze
    p.add_argument("--analyze_n", type=int, default=300)
    p.add_argument("--top_k_features", type=int, default=20)

    # Output
    p.add_argument("--out_root", default="runs/procrustes_steer")

    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = build_parser().parse_args()

    if not any([args.setup, args.steer, args.analyze, args.all]):
        build_parser().print_help()
        return

    if args.all:
        args.setup = args.steer = args.analyze = True

    device = get_default_device()
    log.info("Device: %s  dtype: %s  method: %s", device, args.dtype, args.method)

    Path(args.out_root).mkdir(parents=True, exist_ok=True)

    if args.setup:
        run_setup(args, device)
    if args.steer:
        run_steer(args, device)
    if args.analyze:
        run_analyze(args, device)


if __name__ == "__main__":
    main()
