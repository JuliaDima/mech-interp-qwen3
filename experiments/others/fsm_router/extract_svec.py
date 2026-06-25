"""Extract per-primitive steering vectors using the trained FSM router.

Strategy
--------
For each primitive (addition, subtraction, multiplication, modular):

1. Generate controlled single-primitive prompts  "a OP b ="  where the
   model sometimes fails (this ensures that both correct and wrong buckets are populated).
2. Run the model live → binary correctness label (first-token correct).
3. Tokenize the prompt with the Qwen tokenizer → token strings.
4. Run FSM router on the predicate sequence → A_{t,k} ∈ [0,1].
5. Find the firing token position: last token of the predicate group where
   A_{t,k} is maximal (i.e. FSM completes). This is the anchor for both extraction and injection.
6. Capture h[anchor_pos, layer] for each sweep layer.
7. sv[layer] = mean(h | correct) − mean(h | wrong).
8. Save to runs/fsm_router/svecs/{primitive_name}.pt
   Format: {"layer": Tensor(d_model), "meta": {...}}

Why this anchor?
  The anchor (FSM-firing position) is where the model has just processed
  the complete primitive expression.  Injecting there at inference time puts
  the vector in the same distributional neighbourhood as where it was
  extracted, which is necessary for local steering across multiple primitives.

On p(correct):
  Binary bucketing is used (same as the successful steer.py).  The script
  also saves raw p(correct) scores so soft-weighting can be explored later
  without re-running extraction.

Run:
    python -m experiments.fsm_router.extract_svec --primitive addition
    python -m experiments.fsm_router.extract_svec --primitive all
    python -m experiments.fsm_router.extract_svec --primitive multiplication \\
        --layers 16 20 24 28 --collect_n 200 --n_samples 2000
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.fsm_router.fsm import PrimitiveRouter  # noqa: E402
from experiments.fsm_router.predicates import (  # noqa: E402
    N_PREDICATES,
    token_groups_from_strings,
)
from experiments.fsm_router.primitives import FSM_SPECS, PRIMITIVE_DEFS  # noqa: E402
from mechinterp_qwen3.attribution_model import AttributionModel  # noqa: E402
from mechinterp_qwen3.utils.hf_utils import load_transcoder_from_hub  # noqa: E402
from mechinterp_qwen3.utils.model_utils import get_default_device, parse_dtype  # noqa: E402
from scripts.model_config import default_model, default_transcoder_set
from mechinterp_qwen3.utils.token_utils import tokenize_qwen_input  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("extract_svec")

_MODEL = default_model()
_TRANSCODER_SET = default_transcoder_set()
_ROUTER_PATH = "runs/fsm_router/router.pt"
_OUT_DIR = "runs/fsm_router/svecs"
_SWEEP_LAYERS = list(range(36))  # all layers — extraction is cheap, let evaluation pick the best
_COLLECT_N = 200  # samples per bucket (correct/wrong)
_N_SAMPLES = 3000  # total generated prompts per primitive
_FSM_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Primitive configurations
# ---------------------------------------------------------------------------
# Each config defines how to generate single-primitive prompts and the
# expected answer.

_PRIM_CONFIGS: dict[str, dict] = {
    "addition": {
        "template": "calc: {a}+{b}= ",
        "answer_fn": lambda a, b: a + b,
        "a_range": (
            1,
            999,
        ),  # spans 1-,2-,3-digit — forces vector to capture operation not magnitude
        "b_range": (1, 999),
        "ensure_a_gt_b": False,
        "prim_idx": next(i for i, p in enumerate(PRIMITIVE_DEFS) if p.name == "addition"),
    },
    "subtraction": {
        "template": "calc: {a}-{b}= ",
        "answer_fn": lambda a, b: a - b,  # a always >= b (enforced in generate_samples)
        "a_range": (100, 999),
        "b_range": (100, 999),
        "ensure_a_gt_b": True,  # prevents negative answers
        "prim_idx": next(i for i, p in enumerate(PRIMITIVE_DEFS) if p.name == "subtraction"),
    },
    "multiplication": {
        "template": "calc: {a}*{b}= ",
        "answer_fn": lambda a, b: a * b,
        "a_range": (2, 99),
        "b_range": (2, 99),
        "ensure_a_gt_b": False,
        "prim_idx": next(i for i, p in enumerate(PRIMITIVE_DEFS) if p.name == "multiplication"),
    },
    "modular": {
        "template": "calc: {a}%{b}= ",
        "answer_fn": lambda a, b: a % b,
        "a_range": (100, 999),
        "b_range": (2, 13),
        "ensure_a_gt_b": False,
        "prim_idx": next(i for i, p in enumerate(PRIMITIVE_DEFS) if p.name == "modular"),
    },
}


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------


def generate_samples(
    config: dict,
    n_samples: int,
    tokenizer,
    seed: int = 42,
) -> list[dict]:
    """Generate plain-text single-primitive prompts and tokenize them."""
    rng = random.Random(seed)
    template = config["template"]
    answer_fn = config["answer_fn"]
    a_lo, a_hi = config["a_range"]
    b_lo, b_hi = config["b_range"]

    samples = []
    seen: set[tuple[int, int]] = set()
    attempts = 0

    ensure_a_gt_b = config.get("ensure_a_gt_b", False)

    while len(samples) < n_samples and attempts < n_samples * 10:
        attempts += 1
        a = rng.randint(a_lo, a_hi)
        b = rng.randint(b_lo, b_hi)
        if ensure_a_gt_b and a < b:
            a, b = b, a
        if (a, b) in seen:
            continue
        seen.add((a, b))

        prompt_str = template.format(a=a, b=b)
        answer_val = answer_fn(a, b)
        answer_str = str(answer_val)

        prompt_ids = tokenizer(prompt_str, add_special_tokens=False).input_ids
        answer_ids = tokenizer(answer_str, add_special_tokens=False).input_ids

        samples.append(
            {
                "a": a,
                "b": b,
                "prompt_str": prompt_str,
                "answer_str": answer_str,
                "prompt_token_ids": prompt_ids,
                "answer_token_ids": answer_ids,
            }
        )

    return samples


# ---------------------------------------------------------------------------
# Baseline correctness
# ---------------------------------------------------------------------------


def _digit_accuracy(pred_ids: list[int], target_ids: list[int], tokenizer) -> float:
    """Fraction of digits in the predicted string that match the target string."""
    pred_str = tokenizer.decode(pred_ids).strip()
    tgt_str = tokenizer.decode(target_ids).strip()
    if not tgt_str:
        return 0.0
    matches = sum(p == t for p, t in zip(pred_str, tgt_str, strict=False))
    return matches / len(tgt_str)


def _run_greedy(
    model: AttributionModel,
    prompt_ids: list[int],
    answer_ids: list[int],
    device: torch.device,
    fwd_hooks=None,
) -> dict:
    """Greedy decode; return correctness, log-prob, log-prob margin, digit accuracy."""
    generated = []
    true_first_logp = float("-inf")
    true_first_prob = 0.0
    logp_margin = float("-inf")  # log P(correct) - log P(best wrong)

    with torch.no_grad():
        for pos, target_id in enumerate(answer_ids):
            input_ids = tokenize_qwen_input(
                prompt_ids + generated, model.tokenizer, device
            ).unsqueeze(0)
            if fwd_hooks:
                logits = model.run_with_hooks(input_ids, fwd_hooks=fwd_hooks)[0, -1, :]
            else:
                logits = model(input_ids)[0, -1, :]
            log_probs = F.log_softmax(logits.float(), dim=-1)
            pred_id = int(logits.argmax())
            if pos == 0:
                true_first_logp = log_probs[target_id].item()
                true_first_prob = log_probs[target_id].exp().item()
                # margin: how much the model prefers the correct token over the best alternative
                lp = log_probs.clone()
                lp[target_id] = float("-inf")
                best_wrong_logp = lp.max().item()
                logp_margin = true_first_logp - best_wrong_logp
            generated.append(pred_id)

    digit_acc = _digit_accuracy(generated, answer_ids, model.tokenizer)

    return {
        "first_token_correct": generated[0] == answer_ids[0],
        "full_correct": generated == answer_ids,
        "true_first_prob": true_first_prob,
        "true_first_logp": true_first_logp,
        "logp_margin": logp_margin,
        "digit_acc": digit_acc,
    }


def compute_baseline(
    model: AttributionModel,
    samples: list[dict],
    device: torch.device,
) -> list[dict]:
    model.eval()
    results = []
    with torch.no_grad():
        for s in tqdm(samples, desc="Baseline"):
            r = _run_greedy(model, s["prompt_token_ids"], s["answer_token_ids"], device)
            results.append(r)
    return results


# ---------------------------------------------------------------------------
# FSM firing position
# ---------------------------------------------------------------------------


def find_firing_token_pos(
    tok_strings: list[str],
    router: PrimitiveRouter,
    prim_idx: int,
    device: torch.device,
    threshold: float = _FSM_THRESHOLD,
) -> int | None:
    """Return the token index where primitive prim_idx most strongly fires.

    Maps predicate-space argmax(A_{t, prim_idx}) back to the last token
    index in that predicate group (so the full number/operator has been seen).
    Returns None if the router never exceeds threshold.
    """
    groups = token_groups_from_strings(tok_strings)
    if not groups:
        return None

    pred_ids = torch.tensor(
        [[g[0] for g in groups]],
        dtype=torch.long,  # router always on CPU
    )  # (1, T_pred)

    with torch.no_grad():
        A = router(pred_ids)  # (1, T_pred, K)

    a_prim = A[0, :, prim_idx]  # (T_pred,)

    if a_prim.max().item() < threshold:
        return None

    best_pred_t = int(a_prim.argmax())
    # last_tok_idx of that predicate group
    return groups[best_pred_t][2]


# ---------------------------------------------------------------------------
# Collect activations and build steering vectors
# ---------------------------------------------------------------------------


def collect_svecs(
    model: AttributionModel,
    router: PrimitiveRouter,
    samples: list[dict],
    baseline: list[dict],
    prim_idx: int,
    layers: list[int],
    collect_n: int,
    device: torch.device,
    dtype: torch.dtype,
    bucket_metric: str = "logp_margin",
) -> dict[int, torch.Tensor]:
    """Return {layer: steering_vec} for the primitive.

    bucket_metric controls how activations are split into correct/wrong:
      "first_token_correct" — binary bucket (original approach)
      "logp_margin"         — continuous: top-25% margin vs bottom-25% margin.
                              Uses ALL samples; no data starvation for hard primitives.
      "true_first_logp"     — same but using raw log-prob instead of margin.
    """
    use_continuous = bucket_metric in ("logp_margin", "true_first_logp")

    if use_continuous:
        scores = [b[bucket_metric] for b in baseline]
        sorted_scores = sorted(scores)
        n_total = len(sorted_scores)
        lo_thresh = sorted_scores[n_total // 4]  # bottom 25%
        hi_thresh = sorted_scores[3 * n_total // 4]  # top 25%

        def _bucket(base: dict) -> str | None:
            s = base[bucket_metric]
            if s >= hi_thresh:
                return "correct"
            if s <= lo_thresh:
                return "wrong"
            return None  # middle 50% discarded — cleaner contrast
    else:

        def _bucket(base: dict) -> str | None:
            return "correct" if bool(base[bucket_metric]) else "wrong"

    acts: dict[int, dict[str, list[torch.Tensor]]] = {
        layer: {"correct": [], "wrong": []} for layer in layers
    }

    model.eval()
    with torch.no_grad():
        for sample, base in tqdm(
            zip(samples, baseline, strict=False),
            desc="Collecting",
            total=len(samples),
        ):
            bucket = _bucket(base)
            if bucket is None:
                continue

            if all(len(acts[l][bucket]) >= collect_n for l in layers):
                continue

            tok_strings = model.tokenizer.convert_ids_to_tokens(sample["prompt_token_ids"])
            anchor_pos = find_firing_token_pos(tok_strings, router, prim_idx, device)
            if anchor_pos is None:
                continue

            input_ids = torch.tensor([sample["prompt_token_ids"]], dtype=torch.long, device=device)
            cache: dict[int, torch.Tensor] = {}

            hooks = [
                (
                    f"blocks.{layer}.hook_resid_post",
                    lambda act, hook, _l=layer, _pos=anchor_pos: (
                        cache.update({_l: act[0, _pos, :].detach().clone()}) or act
                    ),
                )
                for layer in layers
                if len(acts[layer][bucket]) < collect_n
            ]
            if not hooks:
                continue

            model.run_with_hooks(input_ids, fwd_hooks=hooks)

            for layer in layers:
                if layer in cache and len(acts[layer][bucket]) < collect_n:
                    acts[layer][bucket].append(cache[layer].to(dtype=dtype))

    svecs: dict[int, torch.Tensor] = {}
    for layer in layers:
        cor = acts[layer]["correct"]
        wrg = acts[layer]["wrong"]
        if not cor or not wrg:
            log.warning("Layer %d: correct=%d wrong=%d — skipping", layer, len(cor), len(wrg))
            continue
        n = min(len(cor), len(wrg))
        sv = torch.stack(cor[:n]).mean(0) - torch.stack(wrg[:n]).mean(0)
        svecs[layer] = sv
        log.info(
            "Layer %2d  sv_norm=%.3f  (n=%d per bucket, %s)",
            layer,
            sv.norm().item(),
            n,
            bucket_metric,
        )

    return svecs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def extract_one(
    prim_name: str,
    args: argparse.Namespace,
    model: AttributionModel,
    router: PrimitiveRouter,
    device: torch.device,
    dtype: torch.dtype,
    out_dir: Path,
) -> None:
    cfg = _PRIM_CONFIGS[prim_name]
    prim_idx = cfg["prim_idx"]

    log.info("=== Primitive: %s (FSM index %d) ===", prim_name, prim_idx)

    samples = generate_samples(cfg, args.n_samples, model.tokenizer, seed=args.seed)
    log.info("Generated %d samples", len(samples))

    baseline = compute_baseline(model, samples, device)
    first_token_acc = 100.0 * sum(b["first_token_correct"] for b in baseline) / len(baseline)
    full_acc = 100.0 * sum(b["full_correct"] for b in baseline) / len(baseline)
    log.info("Baseline  first-token=%.1f%%  full=%.1f%%", first_token_acc, full_acc)

    svecs = collect_svecs(
        model,
        router,
        samples,
        baseline,
        prim_idx=prim_idx,
        layers=args.layers,
        collect_n=args.collect_n,
        device=device,
        dtype=dtype,
        bucket_metric=args.bucket_metric,
    )

    if not svecs:
        log.error("No steering vectors produced for %s", prim_name)
        return

    save_path = out_dir / f"{prim_name}.pt"
    torch.save(
        {
            "svecs": svecs,
            "meta": {
                "primitive": prim_name,
                "prim_idx": prim_idx,
                "layers": args.layers,
                "collect_n": args.collect_n,
                "n_samples": args.n_samples,
                "baseline_first_token_acc": first_token_acc,
                "baseline_full_acc": full_acc,
            },
        },
        save_path,
    )
    log.info("Saved %s → %s", prim_name, save_path)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--primitive",
        default="all",
        choices=list(_PRIM_CONFIGS) + ["all"],
        help="Which primitive to extract SVec for.",
    )
    parser.add_argument("--model", default=_MODEL)
    parser.add_argument("--transcoder_set", default=_TRANSCODER_SET)
    parser.add_argument("--router_path", default=_ROUTER_PATH)
    parser.add_argument("--out_dir", default=_OUT_DIR)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--layers", type=int, nargs="+", default=_SWEEP_LAYERS)
    parser.add_argument("--collect_n", type=int, default=_COLLECT_N)
    parser.add_argument("--n_samples", type=int, default=_N_SAMPLES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--bucket_metric",
        default="logp_margin",
        choices=["logp_margin", "true_first_logp", "first_token_correct"],
        help="Label used to split activations. logp_margin (default) uses top/bottom "
        "25%% of log P(correct)-log P(best wrong), avoiding binary data starvation.",
    )
    args = parser.parse_args()

    device = get_default_device()
    dtype = parse_dtype(args.dtype)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading model %s", args.model)
    transcoder_set, _ = load_transcoder_from_hub(
        args.transcoder_set, dtype=dtype, lazy_encoder=True, lazy_decoder=True
    )
    model = AttributionModel.from_pretrained_and_transcoders(
        args.model, transcoder_set, dtype=dtype, device=device
    )
    model.eval()

    log.info("Loading router from %s", args.router_path)
    router = PrimitiveRouter(FSM_SPECS, N_PREDICATES)
    router.load_state_dict(torch.load(args.router_path, map_location="cpu"))
    router.eval()

    primitives = list(_PRIM_CONFIGS) if args.primitive == "all" else [args.primitive]
    for prim_name in primitives:
        extract_one(prim_name, args, model, router, device, dtype, out_dir)

    log.info("Done. SVecs saved to %s", out_dir)


if __name__ == "__main__":
    main()
