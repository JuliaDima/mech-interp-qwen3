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
import pickle
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
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
from mechinterp_qwen3.interventions import ablate_feature_directions, inhibit_features
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


@torch.no_grad()
def _show_examples(model, pairs: list, n: int = 2, max_new_tokens: int = 50) -> None:
    """Print greedy-decoded outputs for the first n pairs (pos + neg each)."""
    tok = model.tokenizer
    device = model.cfg.device
    for pair in pairs[:n]:
        for label, prompt, expected in [
            ("pos", pair.prompt_pos, pair.predict_pos or pair.label_pos),
            ("neg", pair.prompt_neg, pair.predict_neg or pair.label_neg),
        ]:
            ids = tokenize_qwen_input(
                tok(prompt, add_special_tokens=False).input_ids, tok, device
            ).unsqueeze(0)
            out_ids = model.generate(
                ids, max_new_tokens=max_new_tokens,
                do_sample=False, prepend_bos=False, verbose=False,
            )
            new_tokens = out_ids[0, ids.shape[1]:]
            raw = tok.decode(new_tokens, skip_special_tokens=True)
            got = raw.strip().split()[0] if raw.strip() else ""
            marker = "✓" if got.rstrip(".,!?").lower() == expected.strip().lower() else "✗"
            print(f"  [{label}] {marker} expect={expected!r:8s}  got={got!r:8s}  -> {raw[:80]!r}")


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
            if isinstance(data, list):
                names.extend(str(x) for x in data)
            elif isinstance(data, dict) and ("pos" in data or "neg" in data):
                # edec_features.json format: {"pos": [...], "neg": [...]}
                direction = getattr(args, "features_direction", "pos")
                rows = data.get(direction, [])
                names.extend(r["feature"] for r in rows)
            else:
                raise ValueError(
                    "--features_file JSON must be a list or edec_features.json dict with 'pos'/'neg' keys"
                )
        else:
            names.extend(line.strip() for line in text.splitlines() if line.strip())
    return _dedupe_preserving_order(names)



def _filter_active_prompts(
    feature_keys: list[str],
    npz,
    examples: list[dict],
) -> list[dict]:
    """Return individual prompts where any feature in feature_keys has activation > 0.

    Filters at the prompt level (pos/neg independently), not the pair level.
    Returns a flat list of {"prompt", "target", "split", "pair_idx"} dicts.
    Targets are predict_pos/predict_neg — the true expected answers (e.g. str(a+b) for addition),
    not label_pos/label_neg which are binary concept labels (e.g. "carry"/"no_carry").
    """
    # Track which (pair_idx, side) has any feature firing. side: 0=pos, 1=neg.
    active_sides: set[tuple[int, int]] = set()
    missing: list[str] = []
    for key in feature_keys:
        if key not in npz.files:
            missing.append(key)
            continue
        acts = npz[key]  # shape (2*n_pairs,): interleaved [pos_0, neg_0, pos_1, neg_1, ...]
        for ex in examples:
            i = ex["pair_idx"]
            if acts[2 * i] > 0:
                active_sides.add((i, 0))
            if acts[2 * i + 1] > 0:
                active_sides.add((i, 1))

    if missing:
        log.warning("Features not in sweep_activations.npz (skipped for filter): %s", missing)

    prompts: list[dict] = []
    for ex in examples:
        i = ex["pair_idx"]
        if (i, 0) in active_sides:
            prompts.append({
                "prompt": ex["prompt_pos"],
                "target": ex["predict_pos"],
                "split": "pos",
                "pair_idx": i,
            })
        if (i, 1) in active_sides:
            prompts.append({
                "prompt": ex["prompt_neg"],
                "target": ex["predict_neg"],
                "split": "neg",
                "pair_idx": i,
            })

    n_pos = sum(1 for p in prompts if p["split"] == "pos")
    n_neg = sum(1 for p in prompts if p["split"] == "neg")
    log.info("Active prompts: %d pos + %d neg = %d total", n_pos, n_neg, len(prompts))
    return prompts


def _pairs_to_prompts(pairs: list) -> list[dict]:
    """Flatten ConceptPair list into individual prompt dicts for evaluate()."""
    prompts = []
    for pair in pairs:
        prompts.append({
            "prompt": pair.prompt_pos,
            "target": pair.predict_pos,
            "split": "pos",
        })
        prompts.append({
            "prompt": pair.prompt_neg,
            "target": pair.predict_neg,
            "split": "neg",
        })
    return prompts


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
    prompts: list[dict],
    feature_map: dict[int, list[int]] | None = None,
    alpha: float = 0.0,
    batch_size: int = 8,
    desc: str | None = None,
    subtract_mode: bool = False,
) -> EvalMetrics:
    """Teacher-forced full-answer accuracy over a flat list of prompt dicts.

    Each prompt dict: {"prompt": str, "target": str, "split": "pos"|"neg"}.

    feature_map=None                        → plain model (no intervention)
    feature_map={L: [fid]}, subtract_mode  → subtract act_f * W_dec[f] from real MLP output
    feature_map={L: [fid]}                 → transcoder reconstruction at L, feature fid scaled by alpha
    feature_map={L: []}                    → transcoder reconstruction at L, no features scaled
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
    for p in prompts:
        prompt_ids = model.tokenizer(p["prompt"], add_special_tokens=False).input_ids
        answer_ids = model.tokenizer(p["target"], add_special_tokens=False).input_ids
        if not answer_ids:
            skipped += 1
            continue
        examples_by_len.setdefault(len(prompt_ids) + len(answer_ids), []).append(
            (p["split"], prompt_ids, answer_ids)
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
                elif subtract_mode:
                    logits = ablate_feature_directions(model, tokens, feature_map)
                else:
                    logits = inhibit_features(model, tokens, feature_map, alpha=alpha)

                for i, (split, prompt_ids, answer_ids) in enumerate(batch):
                    n = len(prompt_ids)
                    correct_prob = torch.softmax(logits[i, n], dim=-1)[answer_ids[0]].item()
                    pred_first = int(logits[i, n].argmax())
                    all_correct = all(
                        int(logits[i, n + j].argmax()) == tok_id
                        for j, tok_id in enumerate(answer_ids)
                    )
                    rows.append({"split": split, "correct": all_correct, "correct_prob": correct_prob, "pred_first": pred_first})
                pbar.update(len(batch))

    return _metrics(rows, skipped), rows


def diff_metrics(
    baseline: EvalMetrics, modulated: EvalMetrics,
    base_rows: list[dict], mod_rows: list[dict],
) -> dict:
    out = {"baseline": asdict(baseline), "modulated": asdict(modulated), "change": {}}
    for split in ("all", "pos", "neg"):
        base = getattr(baseline, split)
        mod = getattr(modulated, split)
        br = base_rows if split == "all" else [r for r in base_rows if r["split"] == split]
        mr = mod_rows if split == "all" else [r for r in mod_rows if r["split"] == split]
        improved_frac = (
            sum(1 for b, m in zip(br, mr) if m["correct_prob"] > b["correct_prob"]) / len(br)
            if br else 0.0
        )
        rel_changes = [
            (m["correct_prob"] - b["correct_prob"]) / b["correct_prob"] * 100.0
            for b, m in zip(br, mr) if b["correct_prob"] > 1e-12
        ]
        mean_rel_p_change = sum(rel_changes) / len(rel_changes) if rel_changes else float("nan")
        tok_change_rate = (
            sum(1 for b, m in zip(br, mr) if m.get("pred_first") != b.get("pred_first")) / len(br)
            if br else 0.0
        )
        out["change"][split] = {
            "accuracy": mod.accuracy - base.accuracy,
            "mean_correct_prob": mod.mean_correct_prob - base.mean_correct_prob,
            "improved_frac": improved_frac,
            "mean_rel_p_change": mean_rel_p_change,
            "tok_change_rate": tok_change_rate,
        }
    return out




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
        "feature,split,n,"
        "baseline_acc,modulated_acc,delta_acc,"
        "baseline_p,modulated_p,delta_p,"
        "rel_delta_p_pct,improved_frac,tok_change_rate"
    )
    for row in results["features"]:
        m = row["metrics"]
        for split in ("all", "pos", "neg"):
            base = m["baseline"][split]
            mod  = m["modulated"][split]
            chg  = m["change"][split]
            rel  = chg.get("mean_rel_p_change", float("nan"))
            tok  = chg.get("tok_change_rate", float("nan"))
            print(
                f"{row['feature']},{split},{base['n']},"
                f"{base['accuracy']:.4f},{mod['accuracy']:.4f},{chg['accuracy']:+.4f},"
                f"{base['mean_correct_prob']:.6f},{mod['mean_correct_prob']:.6f},"
                f"{chg['mean_correct_prob']:+.6f},"
                f"{rel:+.2f},{chg['improved_frac']:.4f},{tok:.4f}"
            )


def plot_results(results: dict, out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import experiments.plot_style as ps

    rows = sorted(
        results["features"],
        key=lambda row: row["metrics"]["change"]["all"]["accuracy"],
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
            xs = [row["metrics"]["change"][split][metric] for row in rows]
            ys = [v + offsets[split] for v in y]
            ax.plot(xs, ys, "o", ms=4.5, color=colors[split], label=split, alpha=0.95)
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
        f"{config.get('concept', 'concept')} feature modulation ({mode})",
        fontsize=12, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))

    paths = [out_dir / "feature_modulation_summary.pdf"]
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
    parser.add_argument("--features_file", default=None, help="Text, JSON list, or edec_features.json dict")
    parser.add_argument("--features_direction", default="pos", choices=["pos", "neg"],
                        help="Which direction to load from edec_features.json ('pos'=positive, 'neg'=negative)")
    parser.add_argument("--model", default=_MODEL)
    parser.add_argument("--transcoder_set", default=_TRANSCODER_SET)
    parser.add_argument("--dtype", default="bfloat16")
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
        "--sweep_dir", default=None, type=Path,
        help="Path to sweep dir containing sweep_activations.npz and sweep_examples.pkl. "
             "Required in modulation mode: each feature is evaluated only on prompts where it fires.",
    )
    parser.add_argument(
        "--subtract", action="store_true", default=True,
        help="Subtract act_f * W_dec[f] from the real MLP output instead of using "
             "transcoder reconstruction. Baseline is the raw model; no reconstruction "
             "error is introduced. Pass --no-subtract to use the old transcoder-reconstruction mode.",
    )
    parser.add_argument("--no-subtract", dest="subtract", action="store_false")
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    feature_names = load_feature_names(args)
    eval_only = not feature_names

    if not eval_only and args.sweep_dir is None:
        parser.error("--sweep_dir is required in modulation mode")

    if eval_only:
        log.info("Generating pairs for concept '%s'", args.concept)
        all_pairs = _load_concept(args.concept, 200, args.seed)
        if args.template and args.template.lower() != "none":
            all_pairs = [p for p in all_pairs if p.template == args.template]
        log.info("Loaded %d pairs for concept '%s' template '%s'", len(all_pairs), args.concept, args.template)
    else:
        log.info("sweep_dir=%s — each feature evaluated only on prompts where it fires", args.sweep_dir)

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

    # --- eval-only mode ---
    if eval_only:
        log.info("No features specified — running eval-only mode")
        print("\n--- sample outputs (raw model) ---")
        _show_examples(model, all_pairs, n=2, max_new_tokens=50)
        print()
        raw, _ = evaluate(model, _pairs_to_prompts(all_pairs), feature_map=None,
                          batch_size=args.batch_size, desc="raw-model")
        print_eval_report(raw)

        tmpl_suffix = f"_{args.template}" if args.template and args.template.lower() != "none" else ""
        out_dir = Path(args.out_dir or f"runs/concept_localization/{args.concept}/eval_only")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"concept_accuracy{tmpl_suffix}.json"
        out_path.write_text(json.dumps({
            "config": {
                "concept": args.concept,
                "template": args.template,

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

    # Load sweep files once — shared across all features
    sweep_npz = np.load(args.sweep_dir / "sweep_activations.npz")
    with open(args.sweep_dir / "sweep_examples.pkl", "rb") as fh:
        sweep_examples = pickle.load(fh)

    if args.joint:
        joint_feature_map: dict[int, list[int]] = {}
        for f in features:
            joint_feature_map.setdefault(f.layer, []).append(f.feature_id)
        joint_key = "+".join(f.key for f in features)
        log.info("Joint modulation of %d features: %s", len(features), joint_key)

        feature_keys_joint = [f.key for f in features]
        prompts_r = _filter_active_prompts(feature_keys_joint, sweep_npz, sweep_examples)
        if not prompts_r:
            raise ValueError("No prompts where any joint feature fires — check sweep_dir")
        log.info("Joint: %d active prompts (any feature fires)", len(prompts_r))

        if args.subtract:
            baseline, base_rows = evaluate(model, prompts_r, feature_map=None,
                                           batch_size=args.batch_size, desc="joint-baseline")
            modulated, mod_rows = evaluate(model, prompts_r, feature_map=joint_feature_map,
                                           batch_size=args.batch_size, desc="joint-modulated",
                                           subtract_mode=True)
        else:
            joint_baseline_map = {l: [] for l in joint_feature_map}
            baseline, base_rows = evaluate(model, prompts_r, feature_map=joint_baseline_map,
                                           batch_size=args.batch_size, desc="joint-baseline")
            modulated, mod_rows = evaluate(model, prompts_r, feature_map=joint_feature_map,
                                           alpha=alpha, batch_size=args.batch_size, desc="joint-modulated")

        rows = [{"feature": joint_key, "features": [f.key for f in features],
                 "metrics": diff_metrics(baseline, modulated, base_rows, mod_rows)}]

    else:
        rows = []
        for feature in features:
            prompts_r = _filter_active_prompts([feature.key], sweep_npz, sweep_examples)
            if not prompts_r:
                log.warning("Feature %s fires on no prompts — skipping", feature.key)
                continue
            log.info("Feature %s: %d active prompts (%d pos, %d neg)",
                     feature.key, len(prompts_r),
                     sum(1 for p in prompts_r if p["split"] == "pos"),
                     sum(1 for p in prompts_r if p["split"] == "neg"))

            # Baseline
            if args.subtract:
                baseline, base_rows = evaluate(
                    model, prompts_r, feature_map=None,
                    batch_size=args.batch_size, desc=f"{feature.key}_base",
                )
            else:
                baseline, base_rows = evaluate(
                    model, prompts_r, feature_map={feature.layer: []},
                    batch_size=args.batch_size, desc=f"{feature.key}_base",
                )

            # Modulated
            modulated_map = {feature.layer: [feature.feature_id]}
            if args.subtract:
                modulated, mod_rows = evaluate(
                    model, prompts_r, feature_map=modulated_map,
                    batch_size=args.batch_size, subtract_mode=True,
                )
            else:
                modulated, mod_rows = evaluate(
                    model, prompts_r, feature_map=modulated_map,
                    alpha=alpha, batch_size=args.batch_size,
                )

            rows.append({
                "feature": feature.key,
                "input_name": feature.name,
                "layer": feature.layer,
                "feature_id": feature.feature_id,
                "metrics": diff_metrics(baseline, modulated, base_rows, mod_rows),
            })

    alpha_tag = f"alpha{alpha}".replace(".", "p").replace("-", "neg")
    subtract_tag = "_subtract" if args.subtract else ""
    default_out = f"runs/concept_localization/{args.concept}/modulation_{alpha_tag}{subtract_tag}"
    out_dir = Path(args.out_dir or default_out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "config": {
            "concept": args.concept,
            "model": args.model,
            "transcoder_set": args.transcoder_set,
            "dtype": args.dtype,
            "batch_size": args.batch_size,
            "template": args.template,
            "alpha": alpha,
            "joint": args.joint,
            "subtract": args.subtract,
            "features_file": args.features_file,
            "sweep_dir": str(args.sweep_dir),
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
