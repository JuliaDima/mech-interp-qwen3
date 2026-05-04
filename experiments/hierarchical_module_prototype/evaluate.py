"""Evaluation for the trained hierarchical module prototype.

Generates four outputs in --out_dir/:

1. accuracy_vs_digits.png   — Baseline vs module-augmented accuracy by digit count
2. gate_profile.png         — Mean g_t per token position averaged across problems
3. read_attention.png       — Learned α_l (read attention) bar chart over layer index
4. write_attention.png      — Learned α_l^w (write attention) bar chart over layer index

With --attribution:
    Runs the attribution graph pipeline (run_attribution.attribute) on 10 held-out
    problems under both baseline and module-augmented conditions.  Saves graph .pt
    files and a side-by-side summary JSON to --out_dir/attribution/.

Run:
    python -m experiments.hierarchical_module_prototype.evaluate \
        --checkpoint_dir runs/hierarchical_module/stage2

    python -m experiments.hierarchical_module_prototype.evaluate \
        --checkpoint_dir runs/hierarchical_module/stage2 \
        --attribution
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.addition.dataset_generation.generate_dataset_with_predictions import (  # noqa: E402
    TEMPLATES,
    TemplateID,
)
from experiments.hierarchical_module_prototype.model import PrototypeModule  # noqa: E402
from experiments.hierarchical_module_prototype.utils import (  # noqa: E402
    collect_residuals,
    generate_hard_regime_pairs,
    greedy_decode_with_module,
)
from mechinterp_qwen3.utils.config_utils import (  # noqa: E402
    add_config_args,
    load_config,
    print_config,
    set_parser_defaults_from_config,
)
from mechinterp_qwen3.utils.model_utils import get_default_device, parse_dtype  # noqa: E402
from mechinterp_qwen3.utils_seed import seed_everything  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("hmp.evaluate")

TEMPLATE_STR = TEMPLATES[TemplateID.T0]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate trained hierarchical module prototype",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_config_args(p)

    p.add_argument(
        "--checkpoint_dir",
        required=True,
        help="Directory produced by train_write.py (contains module.pt, meta.json)",
    )
    p.add_argument("--model", default=None, help="Override model name from meta.json")
    p.add_argument("--dtype", default=None, choices=["float32", "bfloat16", "float16"])

    # Evaluation dataset
    p.add_argument(
        "--n_digits_min", type=int, default=2, help="Min digit count for the accuracy sweep"
    )
    p.add_argument(
        "--n_digits_max", type=int, default=6, help="Max digit count for the accuracy sweep"
    )
    p.add_argument(
        "--n_test", type=int, default=100, help="Number of held-out test problems per digit count"
    )
    p.add_argument(
        "--max_answer_tokens", type=int, default=8, help="Max tokens to generate per problem"
    )

    # Attribution
    p.add_argument(
        "--attribution",
        action="store_true",
        help="Run attribution graph pipeline on 10 held-out problems",
    )
    p.add_argument(
        "--attr_n_problems",
        type=int,
        default=10,
        help="Problems to run attribution on (--attribution only)",
    )
    p.add_argument(
        "--transcoder_set",
        default="mwhanna/qwen3-4b-transcoders",
        help="Transcoder hub repo for attribution (--attribution only)",
    )
    p.add_argument("--attr_max_feature_nodes", type=int, default=4000)

    p.add_argument("--out_dir", default="runs/hierarchical_module/eval")
    p.add_argument("--seed", type=int, default=0)

    return p


# ---------------------------------------------------------------------------
# Accuracy evaluation (greedy generation)
# ---------------------------------------------------------------------------


@torch.no_grad()
def eval_accuracy(
    qwen,
    module: PrototypeModule,
    n_digits: int,
    n_test: int,
    seed: int,
    max_answer_tokens: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[float, float]:
    """Return (baseline_accuracy, module_accuracy) on n_test held-out problems.

    Uses greedy decoding (argmax at each step) to generate answers, then
    checks if the numeric portion of the completion matches str(a+b).
    """
    pairs = generate_hard_regime_pairs(
        n_digits,
        n_test * 2,
        seed=seed,
        held_out=True,
        held_out_fraction=0.5,
    )[:n_test]

    baseline_correct = 0
    module_correct = 0

    for a, b in pairs:
        expected = str(a + b)
        prompt_str = TEMPLATE_STR.format(a=a, b=b)

        prompt_ids = qwen.tokenizer(prompt_str, return_tensors=None, add_special_tokens=False)[
            "input_ids"
        ]
        prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)

        # Baseline: plain Qwen greedy decode
        gen_ids = qwen.generate(
            prompt_tensor,
            max_new_tokens=max_answer_tokens,
            do_sample=False,
            verbose=False,
            prepend_bos=False,
        )
        new_ids = gen_ids[0, len(prompt_ids) :].tolist()
        baseline_str = qwen.tokenizer.decode(new_ids, skip_special_tokens=True)
        # Extract leading digits
        baseline_digits = ""
        for ch in baseline_str:
            if ch.isdigit():
                baseline_digits += ch
            elif baseline_digits:
                break
        if baseline_digits == expected:
            baseline_correct += 1

        # Module-augmented greedy decode
        module_ids = greedy_decode_with_module(
            qwen, module, prompt_ids, max_answer_tokens, device, dtype
        )
        module_str = qwen.tokenizer.decode(module_ids, skip_special_tokens=True)
        module_digits = ""
        for ch in module_str:
            if ch.isdigit():
                module_digits += ch
            elif module_digits:
                break
        if module_digits == expected:
            module_correct += 1

    n = max(1, len(pairs))
    return baseline_correct / n, module_correct / n


# ---------------------------------------------------------------------------
# Gate activation profile
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_gate_profile(
    qwen,
    module: PrototypeModule,
    pairs: list[tuple[int, int]],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[list[str], list[float]]:
    """Return mean g_t per token position averaged over all problems.

    Returns:
        token_labels: decoded token strings at each position (from first problem)
        mean_gates:   mean gate value per position
    """
    all_gates: list[torch.Tensor] = []
    token_labels: list[str] = []

    for a, b in pairs:
        prompt_str = TEMPLATE_STR.format(a=a, b=b)
        prompt_ids = qwen.tokenizer(prompt_str, return_tensors=None, add_special_tokens=False)[
            "input_ids"
        ]
        tokens = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)

        residuals = collect_residuals(qwen, tokens).to(device=device, dtype=dtype)
        x = module.read(residuals)
        f = module.primitive(x)
        g = module.gate(f)  # (1, seq, 1)

        all_gates.append(g.squeeze(0).squeeze(-1).cpu())  # (seq,)

        if not token_labels:
            token_labels = [qwen.tokenizer.decode([tid]) for tid in prompt_ids]

    if not all_gates:
        return [], []

    # Pad to same length
    max_len = max(g.shape[0] for g in all_gates)
    padded = torch.zeros(len(all_gates), max_len)
    for i, g in enumerate(all_gates):
        padded[i, : g.shape[0]] = g

    mean_gates = padded.mean(0).tolist()
    return token_labels, mean_gates


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_accuracy(
    digit_counts: list[int],
    baseline_accs: list[float],
    module_accs: list[float],
    out_path: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available — skipping accuracy plot")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(digit_counts, [a * 100 for a in baseline_accs], "o-", label="Baseline Qwen")
    ax.plot(digit_counts, [a * 100 for a in module_accs], "s-", label="Module-augmented Qwen")
    ax.axhline(70, color="gray", linestyle="--", alpha=0.5, label="70% threshold")
    ax.set_xlabel("Operand digit count")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Addition accuracy: baseline vs module-augmented Qwen3-4B")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Accuracy plot saved to %s", out_path)


def plot_gate_profile(
    token_labels: list[str],
    mean_gates: list[float],
    out_path: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available — skipping gate profile plot")
        return

    fig, ax = plt.subplots(figsize=(max(6, len(token_labels) * 0.5), 4))
    x = list(range(len(mean_gates)))
    ax.bar(x, mean_gates, color="steelblue", alpha=0.8)
    if token_labels:
        ax.set_xticks(x[: len(token_labels)])
        ax.set_xticklabels(
            [repr(t) for t in token_labels[: len(mean_gates)]],
            rotation=45,
            ha="right",
            fontsize=8,
        )
    ax.set_ylabel("Mean gate g_t")
    ax.set_title("Gate activation profile — mean over test problems")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Gate profile plot saved to %s", out_path)


def plot_attention_weights(
    weights: list[float],
    title: str,
    out_path: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available — skipping attention plot")
        return

    fig, ax = plt.subplots(figsize=(max(6, len(weights) * 0.3), 4))
    ax.bar(range(len(weights)), weights, color="coral", alpha=0.8)
    ax.set_xlabel("Qwen layer index")
    ax.set_ylabel("Attention weight α_l")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Attention plot saved to %s", out_path)


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------


def run_attribution_eval(
    qwen_for_attr,  # AttributionModel
    module: PrototypeModule,
    pairs: list[tuple[int, int]],
    out_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    max_feature_nodes: int,
) -> None:
    """Run attribution graphs for baseline and module-augmented Qwen.

    For module-augmented attribution, write hooks are registered as permanent
    on the AttributionModel so they survive the run_with_hooks() resets that
    attribute() issues internally.  They are surgically removed from
    hook_resid_post hook points after each attribute() call, leaving the
    attribution model's own permanent hooks (on attention, LayerNorm, embed)
    intact.

    Args:
        qwen_for_attr: AttributionModel with transcoders (required by run_attribution.attribute).
        module: Trained PrototypeModule in eval mode.
        pairs: List of (a, b) pairs to attribute.
        out_dir: Output directory for graph .pt files and summary.
        device: Compute device.
        dtype: Module compute dtype.
        max_feature_nodes: Passed to attribute().
    """
    from mechinterp_qwen3.run_attribution import attribute

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    n_layers = qwen_for_attr.cfg.n_layers

    for idx, (a, b) in enumerate(pairs):
        prompt = TEMPLATE_STR.format(a=a, b=b)
        log.info("Attribution %d/%d: a=%d b=%d expected=%s", idx + 1, len(pairs), a, b, a + b)

        # ── Baseline attribution ──────────────────────────────────────────
        baseline_graph = attribute(
            prompt,
            qwen_for_attr,
            max_feature_nodes=max_feature_nodes,
            verbose=False,
        )
        baseline_path = out_dir / f"baseline_{idx:03d}.pt"
        baseline_graph.to_pt(baseline_path)

        # ── Module-augmented attribution ─────────────────────────────────
        # Precompute module deltas from the prompt's residual streams.
        prompt_ids = qwen_for_attr.tokenizer(prompt, return_tensors=None, add_special_tokens=False)[
            "input_ids"
        ]
        tokens = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)

        with torch.no_grad():
            residuals = collect_residuals(qwen_for_attr, tokens).to(device=device, dtype=dtype)
            deltas, _f, g = module(residuals)  # (n_layers, 1, seq, d_model)

        # Build per-layer hook functions and tag them for later removal.
        # They must be permanent (is_permanent=True) so they survive the
        # run_with_hooks() reset that attribute() calls internally.
        write_hook_fns: list[object] = []
        for layer_idx in range(n_layers):
            delta_l = deltas[layer_idx].detach()

            def _make_write_hook(d: torch.Tensor):
                def _hook(resid: torch.Tensor, hook) -> torch.Tensor:
                    return resid + d.to(device=resid.device, dtype=resid.dtype)

                return _hook

            fn = _make_write_hook(delta_l)
            fn.is_permanent = True  # type: ignore[attr-defined]
            write_hook_fns.append(fn)
            qwen_for_attr.add_hook(f"blocks.{layer_idx}.hook_resid_post", fn, is_permanent=True)

        try:
            module_graph = attribute(
                prompt,
                qwen_for_attr,
                max_feature_nodes=max_feature_nodes,
                verbose=False,
            )
        finally:
            # Remove exactly our write hooks from hook_resid_post, leaving all
            # other permanent hooks (attention, LayerNorm, embed) untouched.
            for layer_idx, fn in enumerate(write_hook_fns):
                hook_name = f"blocks.{layer_idx}.hook_resid_post"
                hp = qwen_for_attr.hook_dict[hook_name]
                hp.fwd_hooks = [h for h in hp.fwd_hooks if h is not fn]

        module_path = out_dir / f"module_{idx:03d}.pt"
        module_graph.to_pt(module_path)

        summary.append(
            {
                "idx": idx,
                "a": a,
                "b": b,
                "prompt": prompt,
                "expected": str(a + b),
                "mean_gate": float(g.mean().item()),
                "baseline_graph": str(baseline_path),
                "module_graph": str(module_path),
            }
        )
        log.info(
            "  Saved baseline=%s  module=%s  mean_g=%.4f",
            baseline_path.name,
            module_path.name,
            g.mean().item(),
        )

    with open(out_dir / "attribution_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Attribution summary saved to %s/attribution_summary.json", out_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = build_parser()
    pre, _ = parser.parse_known_args()
    config = load_config(pre.config)
    set_parser_defaults_from_config(parser, config)
    args = parser.parse_args()
    print_config(args, title="Evaluation Configuration")

    seed_everything(args.seed)
    device = get_default_device()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint metadata
    ckpt_dir = Path(args.checkpoint_dir)
    with open(ckpt_dir / "meta.json") as f:
        meta = json.load(f)

    model_name = args.model or meta["model"]
    dtype_str = args.dtype or meta["dtype"]
    dtype = parse_dtype(dtype_str)
    n_layers = meta["n_layers"]
    d_model = meta["d_model"]
    d_small = meta["d_small"]
    trained_n_digits = meta.get("stage2", {}).get("n_digits", meta.get("n_digits", 4))

    log.info("Loading frozen Qwen: %s", model_name)
    from transformer_lens import HookedTransformer

    qwen = HookedTransformer.from_pretrained(
        model_name,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        dtype=dtype,
    )
    qwen.requires_grad_(False)
    qwen.eval()

    # Load full trained module
    module = PrototypeModule(n_layers, d_model, d_small).to(device=device, dtype=dtype)
    module.load_state_dict(torch.load(ckpt_dir / "module.pt", map_location=device))
    module.requires_grad_(False)
    module.eval()
    log.info("Module loaded from %s", ckpt_dir / "module.pt")

    # ── 1. Read/write attention weights (static — just from module params) ──
    with torch.no_grad():
        read_alpha = module.read.attention_weights().cpu().tolist()
        write_alpha = module.write.attention_weights().cpu().tolist()

    plot_attention_weights(
        read_alpha, "Read attention α_l over Qwen layers", out_dir / "read_attention.png"
    )
    plot_attention_weights(
        write_alpha, "Write attention α_l^w over Qwen layers", out_dir / "write_attention.png"
    )

    with open(out_dir / "attention_weights.json", "w") as f:
        json.dump({"read_alpha": read_alpha, "write_alpha": write_alpha}, f, indent=2)
    log.info("Read α_l  (top-3 layers): %s", sorted(enumerate(read_alpha), key=lambda x: -x[1])[:3])
    log.info(
        "Write α_l (top-3 layers): %s", sorted(enumerate(write_alpha), key=lambda x: -x[1])[:3]
    )

    # ── 2. Gate activation profile (use trained n_digits for representative prompts) ──
    gate_pairs = generate_hard_regime_pairs(
        trained_n_digits, 100, seed=args.seed, held_out=True, held_out_fraction=0.5
    )[:50]

    token_labels, mean_gates = compute_gate_profile(qwen, module, gate_pairs, device, dtype)
    plot_gate_profile(token_labels, mean_gates, out_dir / "gate_profile.png")

    with open(out_dir / "gate_profile.json", "w") as f:
        json.dump({"token_labels": token_labels, "mean_gates": mean_gates}, f, indent=2)

    # ── 3. Accuracy vs digit count ─────────────────────────────────────────
    digit_counts = list(range(args.n_digits_min, args.n_digits_max + 1))
    baseline_accs: list[float] = []
    module_accs: list[float] = []

    for n_d in digit_counts:
        log.info("Evaluating accuracy for %d-digit operands (%d problems)...", n_d, args.n_test)
        b_acc, m_acc = eval_accuracy(
            qwen,
            module,
            n_d,
            args.n_test,
            args.seed,
            args.max_answer_tokens,
            device,
            dtype,
        )
        baseline_accs.append(b_acc)
        module_accs.append(m_acc)
        log.info(
            "  n_digits=%d  baseline=%.1f%%  module=%.1f%%  Δ=%+.1f%%",
            n_d,
            b_acc * 100,
            m_acc * 100,
            (m_acc - b_acc) * 100,
        )

    plot_accuracy(digit_counts, baseline_accs, module_accs, out_dir / "accuracy_vs_digits.png")

    accuracy_results = {
        "digit_counts": digit_counts,
        "baseline_accuracy": baseline_accs,
        "module_accuracy": module_accs,
        "delta": [m - b for m, b in zip(module_accs, baseline_accs, strict=False)],
    }
    with open(out_dir / "accuracy_results.json", "w") as f:
        json.dump(accuracy_results, f, indent=2)

    # ── 4. Attribution (optional) ──────────────────────────────────────────
    if args.attribution:
        log.info("Loading AttributionModel for attribution analysis...")
        from mechinterp_qwen3.attribution_model import AttributionModel
        from mechinterp_qwen3.utils.hf_utils import load_transcoder_from_hub

        transcoder, _cfg = load_transcoder_from_hub(
            args.transcoder_set, dtype=dtype, lazy_encoder=False, lazy_decoder=True
        )
        attr_model = AttributionModel.from_pretrained_and_transcoders(
            model_name,
            transcoder,
            dtype=dtype,
            low_cpu_mem_usage=True,
        )

        attr_pairs = generate_hard_regime_pairs(
            trained_n_digits,
            args.attr_n_problems * 4,
            seed=args.seed,
            held_out=True,
            held_out_fraction=0.5,
        )[: args.attr_n_problems]

        run_attribution_eval(
            attr_model,
            module,
            attr_pairs,
            out_dir / "attribution",
            device,
            dtype,
            args.attr_max_feature_nodes,
        )

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Output directory: {out_dir}")
    print("\nAccuracy summary:")
    for n_d, b, m in zip(digit_counts, baseline_accs, module_accs, strict=False):
        marker = "◀ trained" if n_d == trained_n_digits else ""
        print(f"  {n_d}-digit:  baseline={b:.1%}  module={m:.1%}  Δ={m-b:+.1%}  {marker}")
    print(f"\nTop read layers:  {sorted(enumerate(read_alpha), key=lambda x: -x[1])[:3]}")
    print(f"Top write layers: {sorted(enumerate(write_alpha), key=lambda x: -x[1])[:3]}")
    if mean_gates:
        print(
            f"Gate profile max position: {mean_gates.index(max(mean_gates))} "
            f"(token: {repr(token_labels[mean_gates.index(max(mean_gates))]) if token_labels else 'N/A'})"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
