"""Apply per-primitive steering vectors using the trained FSM router.

Two modes
---------
--local  (default, recommended)
    h_{l,t} += alpha * sum_k  A_{t,k} * sv_{k,l}

    A_{t,k} is the FSM router output for primitive k at token position t
    (in [0,1]).  Steering is proportional to how strongly the router fires
    at each position, so only the relevant span is perturbed.  Multiple
    primitives can coexist in the same prompt without interference.

--no-local  (global baseline)
    h_{l,t} += alpha * sum_k  sv_{k,l}    (all t, all k)

    Equivalent to applying all primitive directions uniformly.  Useful for
    ablation: does local gating actually matter?

Workflow
--------
1. Pre-compute predicate groups + router output A ∈ R^{T_pred × K} from
   the *prompt* token strings.  A is expanded to token space (each
   predicate group maps to all its member tokens).
2. Register per-layer forward hooks that add the weighted steering at
   every forward pass.  Because hooks fire at every token during autoregressive
   generation, prompt positions are steered on every decode step; generated
   positions receive zero contribution (they were not in the original prompt
   predicate computation).
3. Evaluate first-token and full-answer correctness vs un-steered baseline.

Run:
    # Evaluate addition with local steering, sweep alpha and layers:
    python -m experiments.fsm_router.steer_with_router \\
        --primitives addition --layers 20 24 28 --alphas 1 2 5

    # All primitives, local steering, in-distribution only:
    python -m experiments.fsm_router.steer_with_router --primitives all --splits in

    # OOD eval only (4-digit operands), separate job:
    python -m experiments.fsm_router.steer_with_router --primitives all \\
        --splits ood --ood_range 1000 9999 --out runs/fsm_router/steer_results_ood.json

    # Global steering ablation:
    python -m experiments.fsm_router.steer_with_router --no-local --primitives addition
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.fsm_router.extract_svec import (  # noqa: E402
    _PRIM_CONFIGS,
    compute_baseline,
    generate_samples,
)
from experiments.fsm_router.fsm import PrimitiveRouter  # noqa: E402
from experiments.fsm_router.predicates import (  # noqa: E402
    N_PREDICATES,
    token_groups_from_strings,
)
from experiments.fsm_router.primitives import FSM_SPECS  # noqa: E402
from mechinterp_qwen3.attribution_model import AttributionModel  # noqa: E402
from mechinterp_qwen3.utils.hf_utils import load_transcoder_from_hub  # noqa: E402
from mechinterp_qwen3.utils.model_utils import get_default_device, parse_dtype  # noqa: E402
from scripts.model_config import default_model, default_transcoder_set

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("steer_with_router")

_MODEL = default_model()
_TRANSCODER_SET = default_transcoder_set()
_ROUTER_PATH = "runs/fsm_router/router.pt"
_SVEC_DIR = "runs/fsm_router/svecs"
_SWEEP_LAYERS = None  # None = use all layers present in the saved svecs
_SWEEP_ALPHAS = [0.5, 1.0, 2.0, 5.0]
_EVAL_N = 300


# ---------------------------------------------------------------------------
# Router A-matrix expansion: predicate space → token space
# ---------------------------------------------------------------------------


def build_A_tok(
    tok_strings: list[str],
    router: PrimitiveRouter,
    device: torch.device,
) -> torch.Tensor:
    """Run the router and expand A from predicate space to token space.

    Returns A_tok ∈ R^{T_tok × K} where A_tok[t, k] = A[pred_group(t), k].
    Each token inherits the router score of its predicate group.
    """
    groups = token_groups_from_strings(tok_strings)
    K = router.n_primitives

    if not groups:
        return torch.zeros(len(tok_strings), K, device=device)

    pred_ids = torch.tensor(
        [[g[0] for g in groups]],
        dtype=torch.long,  # router always on CPU
    )  # (1, T_pred)

    with torch.no_grad():
        A = router(pred_ids)  # (1, T_pred, K)

    A_tok = torch.zeros(len(tok_strings), K, device=device)
    for pred_t, (_, first_idx, last_idx) in enumerate(groups):
        A_tok[first_idx : last_idx + 1] = A[0, pred_t]

    return A_tok  # (T_tok, K)


# ---------------------------------------------------------------------------
# Hook builder
# ---------------------------------------------------------------------------


def make_steer_hooks(
    tok_strings: list[str],
    router: PrimitiveRouter,
    svecs: dict[str, dict[int, torch.Tensor]],  # primitive → layer → sv
    primitive_names: list[str],
    layers: list[int],
    alpha: float,
    local: bool,
    device: torch.device,
) -> list[tuple[str, callable]]:
    """Build forward hooks for one prompt.

    Args:
        tok_strings:     Qwen token strings for the *prompt* (not generated part).
        router:          Trained FSM router.
        svecs:           {primitive_name: {layer_int: Tensor(d_model)}}.
        primitive_names: Subset of svecs to apply (must all appear in svecs).
        layers:          Layers to hook.
        alpha:           Global scale on the steering vectors.
        local:           If True, weight each token by A_{t,k}; else constant.
        device:          Torch device.

    Returns:
        List of (hook_name, hook_fn) pairs ready for model.run_with_hooks().
    """
    T_prompt = len(tok_strings)
    A_tok = build_A_tok(tok_strings, router, device)  # (T_prompt, K)

    # Index map: primitive name → column index in A_tok (order matches FSM_SPECS / router.fsms)
    prim_to_k = {name: idx for idx, (name, _) in enumerate(FSM_SPECS)}

    hooks = []
    for layer in layers:
        # Pre-gather svecs for this layer across primitives
        layer_svecs: list[tuple[int, torch.Tensor]] = []  # (k, sv)
        for prim_name in primitive_names:
            if prim_name not in svecs:
                continue
            if layer not in svecs[prim_name]:
                continue
            k = prim_to_k.get(prim_name)
            if k is None:
                continue
            layer_svecs.append((k, svecs[prim_name][layer].to(device)))

        if not layer_svecs:
            continue

        def _hook(
            act,
            hook,
            _layer_svecs=layer_svecs,
            _A_tok=A_tok,
            _T_prompt=T_prompt,
            _alpha=alpha,
            _local=local,
        ):
            act = act.clone()
            B, T, D = act.shape
            T_eff = min(T, _T_prompt)  # only steer prompt positions

            # Normalise total injection budget per token so compound expressions
            # don't receive more total perturbation than single-primitive ones.
            # clamp(min=1) leaves single-primitive sums (~0.93) unchanged.
            norm = _A_tok[:T_eff, :].sum(dim=-1).clamp(min=1.0)  # (T_eff,)

            for k, sv in _layer_svecs:
                if _local:
                    # (T_eff,) scale × (D,) sv → (T_eff, D)
                    scale = _A_tok[:T_eff, k] / norm  # (T_eff,)
                    delta = _alpha * scale.unsqueeze(-1) * sv  # (T_eff, D)
                else:
                    delta = (_alpha * sv).unsqueeze(0).expand(T_eff, -1)

                act[:, :T_eff, :] = act[:, :T_eff, :] + delta.unsqueeze(0)

            return act

        hooks.append((f"blocks.{layer}.hook_resid_post", _hook))

    return hooks


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _digit_accuracy(pred_ids: list[int], target_ids: list[int], tokenizer) -> float:
    pred_str = tokenizer.decode(pred_ids).strip()
    tgt_str = tokenizer.decode(target_ids).strip()
    if not tgt_str:
        return 0.0
    return sum(p == t for p, t in zip(pred_str, tgt_str, strict=False)) / len(tgt_str)


def evaluate_steered(
    model: AttributionModel,
    router: PrimitiveRouter,
    svecs: dict[str, dict[int, torch.Tensor]],
    samples: list[dict],
    baseline: list[dict],
    primitive_names: list[str],
    layers_to_eval: list[int],
    alphas: list[float],
    local: bool,
    device: torch.device,
) -> list[dict]:
    base_first_acc = 100 * sum(b["first_token_correct"] for b in baseline) / len(baseline)
    base_full_acc = 100 * sum(b["full_correct"] for b in baseline) / len(baseline)
    base_digit_acc = 100 * sum(b["digit_acc"] for b in baseline) / len(baseline)
    log.info(
        "Baseline  first-token=%.1f%%  full=%.1f%%  digit=%.1f%%  (n=%d)",
        base_first_acc,
        base_full_acc,
        base_digit_acc,
        len(baseline),
    )

    results = []
    model.eval()

    for layer in layers_to_eval:
        for alpha in alphas:
            n_first = n_full = n_eval = 0
            sum_digit = 0.0

            for sample in tqdm(samples, desc=f"Eval L={layer} α={alpha}", leave=False):
                tok_strings = model.tokenizer.convert_ids_to_tokens(sample["prompt_token_ids"])
                hooks = make_steer_hooks(
                    tok_strings,
                    router,
                    svecs,
                    primitive_names,
                    layers=[layer],
                    alpha=alpha,
                    local=local,
                    device=device,
                )

                with torch.no_grad():
                    res = _run_greedy_hooked(
                        model,
                        sample["prompt_token_ids"],
                        sample["answer_token_ids"],
                        device,
                        hooks,
                        tokenizer=model.tokenizer,
                    )

                n_eval += 1
                n_first += int(res["first_token_correct"])
                n_full += int(res["full_correct"])
                sum_digit += res["digit_acc"]

            first_acc = 100 * n_first / n_eval
            full_acc = 100 * n_full / n_eval
            digit_acc = 100 * sum_digit / n_eval
            mode = "local" if local else "global"
            log.info(
                "L=%2d  α=%5.2f  [%s]  first-token=%5.1f%% (%+.1f%%)  "
                "full=%5.1f%% (%+.1f%%)  digit=%5.1f%% (%+.1f%%)",
                layer,
                alpha,
                mode,
                first_acc,
                first_acc - base_first_acc,
                full_acc,
                full_acc - base_full_acc,
                digit_acc,
                digit_acc - base_digit_acc,
            )
            results.append(
                {
                    "layer": layer,
                    "alpha": alpha,
                    "local": local,
                    "primitives": primitive_names,
                    "n_eval": n_eval,
                    "first_token_acc": first_acc,
                    "delta_first_token_acc": first_acc - base_first_acc,
                    "full_acc": full_acc,
                    "delta_full_acc": full_acc - base_full_acc,
                    "digit_acc": digit_acc,
                    "delta_digit_acc": digit_acc - base_digit_acc,
                }
            )

    return results


def _run_greedy_hooked(
    model: AttributionModel,
    prompt_ids: list[int],
    answer_ids: list[int],
    device: torch.device,
    hooks,
    tokenizer=None,
) -> dict:
    generated = []
    with torch.no_grad():
        for pos, target_id in enumerate(answer_ids):
            input_ids = torch.tensor([prompt_ids + generated], dtype=torch.long, device=device)
            logits = model.run_with_hooks(input_ids, fwd_hooks=hooks)[0, -1, :]
            pred_id = int(logits.argmax())
            generated.append(pred_id)
    digit_acc = _digit_accuracy(generated, answer_ids, tokenizer) if tokenizer else 0.0
    return {
        "first_token_correct": generated[0] == answer_ids[0],
        "full_correct": generated == answer_ids,
        "digit_acc": digit_acc,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def load_svecs(
    svec_dir: Path,
    primitives: list[str],
    dtype: torch.dtype,
) -> dict[str, dict[int, torch.Tensor]]:
    svecs: dict[str, dict[int, torch.Tensor]] = {}
    for prim in primitives:
        path = svec_dir / f"{prim}.pt"
        if not path.exists():
            log.warning("SVec file not found for %s: %s", prim, path)
            continue
        data = torch.load(path, map_location="cpu")
        svecs[prim] = {int(layer): sv.to(dtype=dtype) for layer, sv in data["svecs"].items()}
        log.info("Loaded svec for %s  layers=%s", prim, sorted(svecs[prim]))
    return svecs


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--primitives", nargs="+", default=["addition"], choices=list(_PRIM_CONFIGS) + ["all"]
    )
    parser.add_argument("--model", default=_MODEL)
    parser.add_argument("--transcoder_set", default=_TRANSCODER_SET)
    parser.add_argument("--router_path", default=_ROUTER_PATH)
    parser.add_argument("--svec_dir", default=_SVEC_DIR)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--layers", type=int, nargs="+", default=_SWEEP_LAYERS)
    parser.add_argument("--alphas", type=float, nargs="+", default=_SWEEP_ALPHAS)
    parser.add_argument("--eval_n", type=int, default=_EVAL_N)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument(
        "--local",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use local (A_{t,k}-weighted) steering.  --no-local for global baseline.",
    )
    parser.add_argument(
        "--ood_range",
        type=int,
        nargs=2,
        default=None,
        metavar=("LO", "HI"),
        help="If set, also evaluate on OOD samples drawn from [LO, HI] for both operands. "
        "Example: --ood_range 1000 9999 tests 4-digit generalisation.",
    )
    parser.add_argument(
        "--splits",
        default="all",
        choices=["in", "ood", "all"],
        help="Which evaluation splits to run. 'in' = in-distribution only, "
        "'ood' = OOD only (requires --ood_range), 'all' = both.",
    )
    parser.add_argument("--out", default="runs/fsm_router/steer_results.json")
    args = parser.parse_args()

    device = get_default_device()
    dtype = parse_dtype(args.dtype)
    svec_dir = Path(args.svec_dir)

    primitives = list(_PRIM_CONFIGS) if "all" in args.primitives else args.primitives

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

    svecs = load_svecs(svec_dir, primitives, dtype)
    if not svecs:
        log.error("No svecs loaded — run extract_svec.py first")
        return

    # Determine which layers to evaluate: explicit --layers, or all layers in saved svecs
    if args.layers is not None:
        layers_to_eval = args.layers
    else:
        layers_to_eval = sorted({layer for prim_svecs in svecs.values() for layer in prim_svecs})
    log.info("Evaluating layers: %s", layers_to_eval)

    def _make_eval_samples(range_override: tuple[int, int] | None = None) -> list[dict]:
        samples = []
        for prim in primitives:
            if prim not in _PRIM_CONFIGS:
                continue
            cfg = _PRIM_CONFIGS[prim].copy()
            if range_override is not None:
                cfg["a_range"] = range_override
                cfg["b_range"] = range_override
            s = generate_samples(cfg, args.eval_n, model.tokenizer, seed=args.seed)
            samples.extend(s[: args.eval_n // len(primitives)])
        return samples

    if args.splits == "ood" and args.ood_range is None:
        log.error("--splits ood requires --ood_range LO HI")
        return

    results = []

    # In-distribution eval
    if args.splits in ("in", "all"):
        eval_samples = _make_eval_samples()
        log.info("In-distribution eval: %d samples", len(eval_samples))
        baseline = compute_baseline(model, eval_samples, device)
        in_results = evaluate_steered(
            model,
            router,
            svecs,
            eval_samples,
            baseline,
            primitive_names=primitives,
            layers_to_eval=layers_to_eval,
            alphas=args.alphas,
            local=args.local,
            device=device,
        )
        for r in in_results:
            r["split"] = "in_distribution"
        results.extend(in_results)

    # OOD eval
    if args.splits in ("ood", "all") and args.ood_range is not None:
        ood_range = tuple(args.ood_range)
        log.info("=== OOD eval: operand range %s ===", ood_range)
        ood_samples = _make_eval_samples(range_override=ood_range)
        ood_baseline = compute_baseline(model, ood_samples, device)
        ood_results = evaluate_steered(
            model,
            router,
            svecs,
            ood_samples,
            ood_baseline,
            primitive_names=primitives,
            layers_to_eval=layers_to_eval,
            alphas=args.alphas,
            local=args.local,
            device=device,
        )
        for r in ood_results:
            r["split"] = f"ood_{ood_range[0]}_{ood_range[1]}"
        results.extend(ood_results)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Results → %s", args.out)

    best = max(results, key=lambda r: r["delta_full_acc"])
    log.info(
        "Best Δfull-acc: layer=%d  α=%.2f  [%s]  Δ=%+.1f%%",
        best["layer"],
        best["alpha"],
        "local" if best["local"] else "global",
        best["delta_full_acc"],
    )


if __name__ == "__main__":
    main()
