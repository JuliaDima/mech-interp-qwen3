"""Run individual transcoder-feature ablations on a concept dataset.

For each requested feature, this script evaluates teacher-forced binary
accuracy before and after direct feature inhibition.  It reports aggregate,
positive-example, and negative-example accuracy/probability changes.

Example:
    python -m experiments.concept_localization.run_feature_ablation \\
        --concept carry --sample_per_class 50
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

from experiments.concept_localization.causal_analysis import _resolve_target
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
log = logging.getLogger("feature_ablation")

_DEFAULT_CARRY_FEATURES_FILE = (
    _REPO_ROOT / "runs" / "concept_localization" / "carry" / "features_list_to_plot.json"
)


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


def parse_feature_name(name: str) -> FeatureSpec:
    """Parse existing feature-name styles such as L12_F128110 or 12_128110."""
    nums = [int(x) for x in re.findall(r"\d+", name)]
    if len(nums) != 2:
        raise ValueError(
            f"Feature {name!r} must contain exactly two integers: layer and feature id"
        )
    return FeatureSpec(name=name, layer=nums[0], feature_id=nums[1])


def _dedupe_preserving_order(names: list[str]) -> list[str]:
    seen = set()
    out = []
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
    features_file = args.features_file
    if features_file is None and args.concept == "carry" and not names:
        features_file = str(_DEFAULT_CARRY_FEATURES_FILE)

    if features_file:
        path = Path(features_file)
        text = path.read_text().strip()
        if path.suffix == ".json":
            data = json.loads(text)
            if not isinstance(data, list):
                raise ValueError("--features_file JSON must contain a list")
            names.extend(str(x) for x in data)
        else:
            names.extend(line.strip() for line in text.splitlines() if line.strip())
    if not names:
        raise ValueError("Pass at least one feature with --features or --features_file")
    return _dedupe_preserving_order(names)


def select_pairs(pairs: list, sample_per_class: int, seed: int) -> list:
    """Sample paired examples; N pairs give N positive and N negative prompts."""
    if len(pairs) < sample_per_class:
        raise ValueError(
            f"Need at least {sample_per_class} pairs for a {sample_per_class}/"
            f"{sample_per_class} pos/neg sample, found {len(pairs)}"
        )
    rng = random.Random(seed)
    return rng.sample(pairs, sample_per_class)


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


@torch.no_grad()
def evaluate(
    model,
    pairs: list,
    feature: FeatureSpec | None = None,
    alpha: float = 0.0,
    batch_size: int = 8,
) -> EvalMetrics:
    rows: list[dict] = []
    skipped = 0
    feature_map = None if feature is None else {feature.layer: [feature.feature_id]}

    examples_by_len: dict[int, list[tuple[str, list[int], int, int]]] = {}
    for pair in pairs:
        prefix_ids, pos_target_id, neg_target_id = _resolve_target(model.tokenizer, pair)
        if pos_target_id is None or neg_target_id is None:
            skipped += 2
            continue

        examples = [
            ("pos", pair.prompt_pos, pos_target_id, neg_target_id),
            ("neg", pair.prompt_neg, neg_target_id, pos_target_id),
        ]
        for split, prompt, correct_id, foil_id in examples:
            prompt_ids = model.tokenizer(prompt, add_special_tokens=False).input_ids
            input_ids = prompt_ids + prefix_ids
            examples_by_len.setdefault(len(input_ids), []).append(
                (split, input_ids, correct_id, foil_id)
            )

    desc = "Baseline" if feature is None else feature.key
    total = sum(len(items) for items in examples_by_len.values())
    batch_size = max(1, batch_size)
    with tqdm(total=total, desc=desc) as pbar:
        for items in examples_by_len.values():
            for start in range(0, len(items), batch_size):
                batch = items[start : start + batch_size]
                tokens = torch.stack(
                    [
                        tokenize_qwen_input(input_ids, model.tokenizer, model.cfg.device)
                        for _, input_ids, _, _ in batch
                    ],
                    dim=0,
                )

                if feature_map is None:
                    logits = model(tokens)
                else:
                    logits = inhibit_features(model, tokens, feature_map, alpha=alpha)

                probs = torch.softmax(logits[:, -1], dim=-1)
                correct_ids = torch.tensor(
                    [correct_id for _, _, correct_id, _ in batch],
                    device=probs.device,
                    dtype=torch.long,
                )
                foil_ids = torch.tensor(
                    [foil_id for _, _, _, foil_id in batch],
                    device=probs.device,
                    dtype=torch.long,
                )
                batch_idx = torch.arange(len(batch), device=probs.device)
                correct_probs = probs[batch_idx, correct_ids].float().cpu().tolist()
                foil_probs = probs[batch_idx, foil_ids].float().cpu().tolist()

                for (split, _, _, _), correct_prob, foil_prob in zip(
                    batch, correct_probs, foil_probs, strict=False
                ):
                    rows.append(
                        {
                            "split": split,
                            "correct": correct_prob > foil_prob,
                            "correct_prob": correct_prob,
                            "foil_prob": foil_prob,
                        }
                    )
                pbar.update(len(batch))

    return _metrics(rows, skipped)


def diff_metrics(baseline: EvalMetrics, ablated: EvalMetrics) -> dict:
    out = {"baseline": asdict(baseline), "ablated": asdict(ablated), "change": {}}
    for split in ("all", "pos", "neg"):
        base = getattr(baseline, split)
        abl = getattr(ablated, split)
        out["change"][split] = {
            "accuracy": abl.accuracy - base.accuracy,
            "mean_correct_prob": abl.mean_correct_prob - base.mean_correct_prob,
        }
    return out


def print_report(results: dict) -> None:
    print("feature,split,n,baseline_acc,ablated_acc,delta_acc,baseline_p,ablated_p,delta_p")
    for row in results["features"]:
        for split in ("all", "pos", "neg"):
            base = row["metrics"]["baseline"][split]
            abl = row["metrics"]["ablated"][split]
            chg = row["metrics"]["change"][split]
            print(
                f"{row['feature']},{split},{base['n']},"
                f"{base['accuracy']:.4f},{abl['accuracy']:.4f},{chg['accuracy']:+.4f},"
                f"{base['mean_correct_prob']:.6f},{abl['mean_correct_prob']:.6f},"
                f"{chg['mean_correct_prob']:+.6f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--concept", required=True, choices=CONCEPTS)
    parser.add_argument(
        "--features",
        nargs="+",
        default=None,
        help=(
            "Feature names to ablate. For --concept carry, defaults to "
            "runs/concept_localization/carry/features_list_to_plot.json when omitted."
        ),
    )
    parser.add_argument("--features_file", default=None, help="Text or JSON list of feature names")
    parser.add_argument("--model", default=_MODEL)
    parser.add_argument("--transcoder_set", default=_TRANSCODER_SET)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--n", type=int, default=100, help="Pairs per template to generate")
    parser.add_argument("--sample_per_class", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8, help="Evaluation batch size within equal-length prompts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--template", default=None, help="Optional template filter, e.g. T0")
    parser.add_argument("--alpha", type=float, default=0.0, help="Feature scale; 0 fully ablates")
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory (default: runs/concept_localization/<concept>/ablation)",
    )
    args = parser.parse_args()

    features = [parse_feature_name(name) for name in load_feature_names(args)]

    log.info("Generating %d pairs/template for concept '%s'", args.n, args.concept)
    pairs = _load_concept(args.concept, args.n, args.seed)
    if args.template:
        pairs = [p for p in pairs if p.template == args.template]
        log.info("Filtered to template '%s': %d pairs", args.template, len(pairs))
    pairs = select_pairs(pairs, args.sample_per_class, args.seed)
    log.info("Sampled %d positive and %d negative examples", len(pairs), len(pairs))

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

    for feature in features:
        if feature.layer < 0 or feature.layer >= model.cfg.n_layers:
            raise ValueError(f"{feature.name!r} has invalid layer {feature.layer}")
        n_features = model.transcoders[feature.layer].W_enc.shape[0]
        if feature.feature_id < 0 or feature.feature_id >= n_features:
            raise ValueError(
                f"{feature.name!r} has invalid feature id {feature.feature_id} "
                f"for layer {feature.layer} with {n_features} features"
            )

    baseline = evaluate(model, pairs, batch_size=args.batch_size)
    rows = []
    for feature in features:
        ablated = evaluate(model, pairs, feature=feature, alpha=args.alpha, batch_size=args.batch_size)
        rows.append(
            {
                "feature": feature.key,
                "input_name": feature.name,
                "layer": feature.layer,
                "feature_id": feature.feature_id,
                "metrics": diff_metrics(baseline, ablated),
            }
        )

    out_dir = Path(args.out_dir or f"runs/concept_localization/{args.concept}/ablation")
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "config": {
            "concept": args.concept,
            "model": args.model,
            "transcoder_set": args.transcoder_set,
            "dtype": args.dtype,
            "n_per_template": args.n,
            "sample_per_class": args.sample_per_class,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "template": args.template,
            "alpha": args.alpha,
            "features_file": (
                args.features_file
                or (
                    str(_DEFAULT_CARRY_FEATURES_FILE)
                    if args.concept == "carry" and args.features is None
                    else None
                )
            ),
        },
        "features": rows,
    }
    out_path = out_dir / "feature_ablation.json"
    out_path.write_text(json.dumps(results, indent=2))
    log.info("Saved results → %s", out_path)
    print_report(results)


if __name__ == "__main__":
    main()
