"""Transcoder-feature modulation on a concept dataset.

Scales selected features by alpha and measures the effect on teacher-forced accuracy:
  alpha = 0.0   → full ablation (feature zeroed)
  alpha = 0.5   → partial suppression
  alpha = 1.0   → no-op (sanity check; should give zero delta)
  alpha > 1.0   → amplification
  alpha < 0.0   → sign inversion / over-suppression

When no features are specified the script runs in eval-only mode: it reports
raw-model concept accuracy on the chosen dataset and template, then exits.

Examples:
    # Eval-only — just report baseline accuracy on T0 carry
    python -m experiments.concept_localization.run_feature_modulation \\
        --concept carry

    # Ablate (alpha=0)
    python -m experiments.concept_localization.run_feature_modulation \\
        --concept carry --features L13_F56616 L16_F34883 --alpha 0.0

    # Amplify (alpha=2)
    python -m experiments.concept_localization.run_feature_modulation \\
        --concept carry --features L13_F56616 L16_F34883 --alpha 2.0

    # Joint modulation
    python -m experiments.concept_localization.run_feature_modulation \\
        --concept carry --joint --features L13_F56616 L16_F34883 --alpha 2.0
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.concept_localization.run_concept import (
    CONCEPTS,
    _MODEL,
    _TRANSCODER_SET,
    _load_concept,
)
from mechinterp_qwen3.attribution_model import AttributionModel
from mechinterp_qwen3.interventions import inhibit_features
from mechinterp_qwen3.utils.hf_utils import load_transcoder_from_hub
from mechinterp_qwen3.utils.model_utils import get_default_device, parse_dtype
from mechinterp_qwen3.utils.token_utils import tokenize_qwen_input

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("feature_modulation")


# ---------------------------------------------------------------------------
# data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    layer: int
    feature_id: int

    @property
    def key(self) -> str:
        return f"L{self.layer}_F{self.feature_id}"


@dataclass
class SplitMetrics:
    n: int
    accuracy: float
    mean_correct_prob: float


@dataclass
class EvalMetrics:
    all: SplitMetrics
    pos: SplitMetrics
    neg: SplitMetrics
    skipped: int


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def parse_feature_name(name: str) -> FeatureSpec:
    nums = [int(x) for x in re.findall(r"\d+", name)]
    if len(nums) != 2:
        raise ValueError(
            f"Feature {name!r} must contain exactly two integers: layer and feature id"
        )
    return FeatureSpec(name=name, layer=nums[0], feature_id=nums[1])


def _dedupe_preserving_order(names: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        key = parse_feature_name(name).key
        if key in seen:
            log.info("Skipping duplicate feature %s", name)
            continue
        seen.add(key)
        out.append(name)
    return out


def load_feature_names(args: argparse.Namespace) -> list[str]:
    names = list(args.features or [])
    if args.features_file:
        path = Path(args.features_file)
        text = path.read_text().strip()
        if path.suffix == ".json":
            data = json.loads(text)
            if not isinstance(data, list):
                raise ValueError("--features_file JSON must contain a list")
            names.extend(str(x) for x in data)
        else:
            names.extend(line.strip() for line in text.splitlines() if line.strip())
    return _dedupe_preserving_order(names)


def select_pairs(pairs: list, sample_per_class: int, seed: int) -> list:
    if len(pairs) < sample_per_class:
        raise ValueError(
            f"Need at least {sample_per_class} pairs, found {len(pairs)}"
        )
    return random.Random(seed).sample(pairs, sample_per_class)


def _split_metrics(rows: list[dict], split: str) -> SplitMetrics:
    items = rows if split == "all" else [r for r in rows if r["split"] == split]
    if not items:
        return SplitMetrics(n=0, accuracy=0.0, mean_correct_prob=0.0)
    return SplitMetrics(
        n=len(items),
        accuracy=sum(1.0 for r in items if r["correct"]) / len(items),
        mean_correct_prob=sum(float(r["correct_prob"]) for r in items) / len(items),
    )


def _metrics(rows: list[dict], skipped: int) -> EvalMetrics:
    return EvalMetrics(
        all=_split_metrics(rows, "all"),
        pos=_split_metrics(rows, "pos"),
        neg=_split_metrics(rows, "neg"),
        skipped=skipped,
    )


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    model,
    pairs: list,
    feature_map: dict[int, list[int]] | None = None,
    alpha: float = 0.0,
    batch_size: int = 8,
    desc: str | None = None,
) -> EvalMetrics:
    """Teacher-forced full-answer accuracy.

    feature_map=None       → plain model (no transcoder)
    feature_map={L: []}    → transcoder reconstruction at L, no features scaled
    feature_map={L: [fid]} → transcoder at L, feature fid scaled by alpha
    """
    rows: list[dict] = []
    skipped = 0

    if desc is None:
        if feature_map is None:
            desc = "raw-model"
        else:
            parts = []
            for layer in sorted(feature_map):
                fids = feature_map[layer]
                parts.append(
                    f"L{layer}"
                    + (f"_F{fids[0]}" if len(fids) == 1 else f"[{len(fids)}feat]" if fids else "_tc")
                )
            desc = "+".join(parts) or "tc-baseline"

    examples_by_len: dict[int, list] = {}
    for pair in pairs:
        pred_pos = pair.predict_pos if pair.predict_pos else pair.label_pos
        pred_neg = pair.predict_neg if pair.predict_neg else pair.label_neg
        for split, prompt, answer_str in [
            ("pos", pair.prompt_pos, pred_pos),
            ("neg", pair.prompt_neg, pred_neg),
        ]:
            prompt_ids = model.tokenizer(prompt, add_special_tokens=False).input_ids
            answer_ids = model.tokenizer(answer_str, add_special_tokens=False).input_ids
            if not answer_ids:
                skipped += 1
                continue
            examples_by_len.setdefault(len(prompt_ids) + len(answer_ids), []).append(
                (split, prompt_ids, answer_ids)
            )

    total = sum(len(v) for v in examples_by_len.values())
    with tqdm(total=total, desc=desc) as pbar:
        for items in examples_by_len.values():
            for start in range(0, len(items), max(1, batch_size)):
                batch = items[start : start + batch_size]
                tokens = torch.stack(
                    [
                        tokenize_qwen_input(
                            prompt_ids + answer_ids, model.tokenizer, model.cfg.device
                        )
                        for _, prompt_ids, answer_ids in batch
                    ],
                    dim=0,
                )
                if feature_map is None:
                    logits = model(tokens)
                else:
                    logits = inhibit_features(model, tokens, feature_map, alpha=alpha)

                for i, (split, prompt_ids, answer_ids) in enumerate(batch):
                    n = len(prompt_ids)
                    correct_prob = torch.softmax(logits[i, n], dim=-1)[answer_ids[0]].item()
                    all_correct = all(
                        int(logits[i, n + j].argmax()) == tok_id
                        for j, tok_id in enumerate(answer_ids)
                    )
                    rows.append({"split": split, "correct": all_correct, "correct_prob": correct_prob})
                pbar.update(len(batch))

    return _metrics(rows, skipped)


def build_cascade_map(
    start_layer: int,
    n_layers: int,
    feature_ids_by_layer: dict[int, list[int]] | None = None,
) -> dict[int, list[int]]:
    """Transcoder at every layer from start_layer to end; features scaled only at specified layers."""
    m: dict[int, list[int]] = {L: [] for L in range(start_layer, n_layers)}
    if feature_ids_by_layer:
        for layer, fids in feature_ids_by_layer.items():
            if layer in m:
                m[layer] = list(fids)
    return m


def diff_metrics(baseline: EvalMetrics, modulated: EvalMetrics) -> dict:
    out = {"baseline": asdict(baseline), "modulated": asdict(modulated), "change": {}}
    for split in ("all", "pos", "neg"):
        base = getattr(baseline, split)
        mod = getattr(modulated, split)
        out["change"][split] = {
            "accuracy": mod.accuracy - base.accuracy,
            "mean_correct_prob": mod.mean_correct_prob - base.mean_correct_prob,
        }
    return out


def _mean_std(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    mean = sum(values) / len(values)
    if len(values) == 1:
        return {"mean": mean, "std": 0.0}
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return {"mean": mean, "std": var**0.5}


def summarize_runs(metric_runs: list[dict]) -> dict:
    summary: dict = {"repeats": len(metric_runs)}
    for section in ("baseline", "modulated"):
        summary[section] = {}
        for split in ("all", "pos", "neg"):
            rows = [run[section][split] for run in metric_runs]
            summary[section][split] = {
                "n": int(round(sum(row["n"] for row in rows) / max(1, len(rows)))),
                "accuracy": _mean_std([row["accuracy"] for row in rows]),
                "mean_correct_prob": _mean_std([row["mean_correct_prob"] for row in rows]),
            }
    summary["change"] = {}
    for split in ("all", "pos", "neg"):
        rows = [run["change"][split] for run in metric_runs]
        summary["change"][split] = {
            "accuracy": _mean_std([row["accuracy"] for row in rows]),
            "mean_correct_prob": _mean_std([row["mean_correct_prob"] for row in rows]),
        }
    return summary


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def print_eval_report(raw: EvalMetrics) -> None:
    print("\n=== Concept Accuracy (raw model) ===")
    print(f"{'split':<6} {'n':>5} {'accuracy':>10} {'mean_p(correct)':>16}")
    print("-" * 42)
    for split in ("all", "pos", "neg"):
        m = getattr(raw, split)
        print(f"{split:<6} {m.n:>5} {m.accuracy:>10.1%} {m.mean_correct_prob:>16.4f}")


def print_modulation_report(results: dict) -> None:
    alpha = results["config"]["alpha"]
    mode = "ablation" if alpha == 0.0 else ("amplification" if alpha > 1.0 else f"alpha={alpha}")
    print(f"\n=== Feature modulation report ({mode}) ===")
    print(
        "feature,split,n,repeats,"
        "baseline_acc_mean,baseline_acc_std,"
        "modulated_acc_mean,modulated_acc_std,"
        "delta_acc_mean,delta_acc_std,"
        "baseline_p_mean,baseline_p_std,"
        "modulated_p_mean,modulated_p_std,"
        "delta_p_mean,delta_p_std"
    )
    for row in results["features"]:
        summary = row["summary"]
        for split in ("all", "pos", "neg"):
            base = summary["baseline"][split]
            mod = summary["modulated"][split]
            chg = summary["change"][split]
            print(
                f"{row['feature']},{split},{base['n']},{summary['repeats']},"
                f"{base['accuracy']['mean']:.4f},{base['accuracy']['std']:.4f},"
                f"{mod['accuracy']['mean']:.4f},{mod['accuracy']['std']:.4f},"
                f"{chg['accuracy']['mean']:+.4f},{chg['accuracy']['std']:.4f},"
                f"{base['mean_correct_prob']['mean']:.6f},{base['mean_correct_prob']['std']:.6f},"
                f"{mod['mean_correct_prob']['mean']:.6f},{mod['mean_correct_prob']['std']:.6f},"
                f"{chg['mean_correct_prob']['mean']:+.6f},{chg['mean_correct_prob']['std']:.6f}"
            )


def plot_results(results: dict, out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import experiments.plot_style as ps

    rows = sorted(
        results["features"],
        key=lambda row: row["summary"]["change"]["all"]["accuracy"]["mean"],
    )
    if not rows:
        return []

    ps.apply()
    labels = [row["feature"] for row in rows]
    y = list(range(len(rows)))
    offsets = {"all": -0.24, "pos": 0.0, "neg": 0.24}
    colors = {"all": ps.NAVY, "pos": ps.VIOLET, "neg": ps.TEAL}

    alpha = results["config"]["alpha"]
    mode = "ablation" if alpha == 0.0 else ("amplification" if alpha > 1.0 else f"α={alpha}")

    fig_h = max(5.0, 0.36 * len(rows) + 1.8)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, fig_h), sharey=True)

    for ax, metric, title, xlabel in [
        (axes[0], "accuracy", "Accuracy change", "modulated − baseline accuracy"),
        (axes[1], "mean_correct_prob", "Correct-probability change", "modulated − baseline mean p(correct)"),
    ]:
        ax.axvline(0.0, color=ps.GRAY, lw=1.0, ls="--", alpha=0.8)
        for split in ("all", "pos", "neg"):
            xs = [row["summary"]["change"][split][metric]["mean"] for row in rows]
            xerr = [row["summary"]["change"][split][metric]["std"] for row in rows]
            ys = [v + offsets[split] for v in y]
            ax.errorbar(xs, ys, xerr=xerr, fmt="o", ms=4.5, capsize=2.5, elinewidth=1.0,
                        color=colors[split], label=split, alpha=0.95)
        ax.set_title(title, fontsize=11, pad=6)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.grid(axis="x", color="#E0E0E0", lw=0.5)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="x", labelsize=8)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].set_ylabel("modulated feature", fontsize=9)
    axes[1].tick_params(axis="y", labelleft=False)
    axes[1].legend(title="split", fontsize=8, title_fontsize=8, loc="best")

    config = results.get("config", {})
    fig.suptitle(
        f"{config.get('concept', 'concept')} feature modulation ({mode}, "
        f"n={config.get('sample_per_class', '?')}/split, repeats={config.get('repeats', 1)})",
        fontsize=12, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))

    paths = [out_dir / "feature_modulation_summary.png", out_dir / "feature_modulation_summary.pdf"]
    for path in paths:
        fig.savefig(path)
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--concept", required=True, choices=CONCEPTS)
    parser.add_argument(
        "--features", nargs="+", default=None,
        help="Features to modulate, e.g. L13_F56616 L16_F34883. "
             "Omit to run eval-only mode (report baseline accuracy and exit).",
    )
    parser.add_argument("--features_file", default=None, help="Text or JSON list of feature names")
    parser.add_argument("--model", default=_MODEL)
    parser.add_argument("--transcoder_set", default=_TRANSCODER_SET)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--n", type=int, default=100, help="Pairs per template to generate")
    parser.add_argument("--sample_per_class", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--template", default="T0",
                        help="Template filter (T0/T1/T2/None). Default T0.")
    parser.add_argument(
        "--alpha", type=float, default=0.0,
        help="Feature scale factor: 0=ablate, 0<α<1=suppress, α=1=no-op, α>1=amplify, α<0=invert",
    )
    parser.add_argument("--joint", action="store_true",
                        help="Modulate all features simultaneously")
    parser.add_argument(
        "--cascade", action="store_true",
        help="Run transcoder at all layers from min(feature layers) to end. "
             "Baseline and modulated both use this cascade, so downstream layers "
             "always see transcoder representations rather than raw MLP outputs.",
    )
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    feature_names = load_feature_names(args)
    eval_only = not feature_names

    log.info("Generating %d pairs/template for concept '%s'", args.n, args.concept)
    all_pairs = _load_concept(args.concept, args.n, args.seed)
    if args.template and args.template.lower() != "none":
        all_pairs = [p for p in all_pairs if p.template == args.template]
        log.info("Filtered to template '%s': %d pairs", args.template, len(all_pairs))

    device = get_default_device()
    dtype = parse_dtype(args.dtype)
    log.info("Loading model %s", args.model)
    transcoder_set, _ = load_transcoder_from_hub(
        args.transcoder_set, dtype=dtype, lazy_encoder=True, lazy_decoder=True
    )
    model = AttributionModel.from_pretrained_and_transcoders(
        args.model, transcoder_set, dtype=dtype, device=device
    )
    model.eval()

    repeats = max(1, args.repeats)
    pairs = select_pairs(all_pairs, args.sample_per_class, args.seed)

    # --- eval-only mode ---
    if eval_only:
        log.info("No features specified — running eval-only mode")
        raw = evaluate(model, pairs, feature_map=None,
                       batch_size=args.batch_size, desc="raw-model")
        print_eval_report(raw)

        out_dir = Path(args.out_dir or f"runs/concept_localization/{args.concept}/eval_only")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "concept_accuracy.json"
        out_path.write_text(json.dumps({
            "config": {
                "concept": args.concept,
                "template": args.template,
                "sample_per_class": args.sample_per_class,
                "seed": args.seed,
            },
            "raw_model": {split: asdict(getattr(raw, split)) for split in ("all", "pos", "neg")},
        }, indent=2))
        log.info("Saved → %s", out_path)
        return

    # --- modulation mode ---
    features = [parse_feature_name(n) for n in feature_names]

    for feature in features:
        if feature.layer < 0 or feature.layer >= model.cfg.n_layers:
            raise ValueError(f"{feature.name!r} has invalid layer {feature.layer}")
        n_features = model.transcoders[feature.layer].W_enc.shape[0]
        if feature.feature_id < 0 or feature.feature_id >= n_features:
            raise ValueError(
                f"{feature.name!r}: feature id {feature.feature_id} out of range "
                f"(layer {feature.layer} has {n_features} features)"
            )

    alpha = args.alpha
    mode_str = (
        "ablation" if alpha == 0.0
        else "amplification" if alpha > 1.0
        else f"suppression (α={alpha})" if 0.0 < alpha < 1.0
        else f"inversion (α={alpha})"
    )
    log.info("Mode: %s | features: %s", mode_str, [f.key for f in features])

    if args.joint:
        joint_feature_map: dict[int, list[int]] = {}
        for f in features:
            joint_feature_map.setdefault(f.layer, []).append(f.feature_id)
        joint_key = "+".join(f.key for f in features)
        min_layer = min(f.layer for f in features)
        log.info("Joint modulation of %d features: %s", len(features), joint_key)

        if args.cascade:
            log.info("Cascade mode: transcoder at layers %d–%d", min_layer, model.cfg.n_layers - 1)
            joint_baseline_map = build_cascade_map(min_layer, model.cfg.n_layers)
            joint_modulated_map = build_cascade_map(min_layer, model.cfg.n_layers, joint_feature_map)
        else:
            joint_baseline_map = {l: [] for l in joint_feature_map}
            joint_modulated_map = joint_feature_map

        joint_row: dict = {"feature": joint_key, "features": [f.key for f in features], "runs": []}

        for repeat_idx in range(repeats):
            sample_seed = args.seed + repeat_idx
            pairs_r = select_pairs(all_pairs, args.sample_per_class, sample_seed)
            baseline = evaluate(model, pairs_r, feature_map=joint_baseline_map,
                                batch_size=args.batch_size, desc="joint-baseline")
            modulated = evaluate(model, pairs_r, feature_map=joint_modulated_map,
                                 alpha=alpha, batch_size=args.batch_size, desc="joint-modulated")
            joint_row["runs"].append({"sample_seed": sample_seed,
                                      "metrics": diff_metrics(baseline, modulated)})

        joint_row["summary"] = summarize_runs([r["metrics"] for r in joint_row["runs"]])
        rows = [joint_row]

    else:
        rows = [
            {
                "feature": feature.key,
                "input_name": feature.name,
                "layer": feature.layer,
                "feature_id": feature.feature_id,
                "runs": [],
            }
            for feature in features
        ]

        for repeat_idx in range(repeats):
            sample_seed = args.seed + repeat_idx
            pairs_r = select_pairs(all_pairs, args.sample_per_class, sample_seed)
            log.info("Repeat %d/%d seed=%d", repeat_idx + 1, repeats, sample_seed)

            layer_baselines: dict[int, EvalMetrics] = {}
            for feature in features:
                if feature.layer not in layer_baselines:
                    if args.cascade:
                        baseline_map = build_cascade_map(feature.layer, model.cfg.n_layers)
                    else:
                        baseline_map = {feature.layer: []}
                    layer_baselines[feature.layer] = evaluate(
                        model, pairs_r, feature_map=baseline_map,
                        batch_size=args.batch_size, desc=f"L{feature.layer}_tc-baseline",
                    )
            for feature, row in zip(features, rows, strict=False):
                baseline = layer_baselines[feature.layer]
                if args.cascade:
                    modulated_map = build_cascade_map(
                        feature.layer, model.cfg.n_layers,
                        {feature.layer: [feature.feature_id]},
                    )
                else:
                    modulated_map = {feature.layer: [feature.feature_id]}
                modulated = evaluate(
                    model, pairs_r,
                    feature_map=modulated_map,
                    alpha=alpha, batch_size=args.batch_size,
                )
                row["runs"].append({"sample_seed": sample_seed,
                                    "metrics": diff_metrics(baseline, modulated)})

        for row in rows:
            row["summary"] = summarize_runs([run["metrics"] for run in row["runs"]])
            if repeats == 1:
                row["metrics"] = row["runs"][0]["metrics"]

    alpha_tag = f"alpha{alpha}".replace(".", "p").replace("-", "neg")
    cascade_tag = "_cascade" if args.cascade else ""
    default_out = f"runs/concept_localization/{args.concept}/modulation_{alpha_tag}{cascade_tag}"
    out_dir = Path(args.out_dir or default_out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "config": {
            "concept": args.concept,
            "model": args.model,
            "transcoder_set": args.transcoder_set,
            "dtype": args.dtype,
            "n_per_template": args.n,
            "sample_per_class": args.sample_per_class,
            "repeats": repeats,
            "sample_seeds": [args.seed + i for i in range(repeats)],
            "batch_size": args.batch_size,
            "seed": args.seed,
            "template": args.template,
            "alpha": alpha,
            "joint": args.joint,
            "cascade": args.cascade,
            "cascade_min_layer": min(f.layer for f in features) if args.cascade else None,
            "features_file": args.features_file,
        },
        "features": rows,
    }
    out_path = out_dir / "feature_modulation.json"
    out_path.write_text(json.dumps(results, indent=2))
    log.info("Saved → %s", out_path)

    plot_paths = plot_results(results, out_dir)
    for p in plot_paths:
        log.info("Saved plot → %s", p)

    print_modulation_report(results)


if __name__ == "__main__":
    main()
