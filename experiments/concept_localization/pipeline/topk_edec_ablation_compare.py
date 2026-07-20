"""Compare two top-K feature-selection strategies for joint ablation, across all
anchors of carry/gcd/residue_class/prime. Ranking and anchor bookkeeping both
live here; run_feature_modulation.py only ever receives explicit features.

  default_encdec — the existing dec+enc_dec candidate pool (delta_feature_projections.py's
                   edec_features.json), truncated to the top --top_k by |score|
                   (rank_top_encdec_features).
  cohens_d       — exhaustive scan of every transcoder feature in every layer cached in
                   the anchor's sweep, ranked by standardised effect size, top --top_k
                   (rank_by_cohens_d).

For every anchor: joint-ablate each config (shared baseline pass, features passed
in as an explicit feature_map — same interface run_feature_modulation.py's CLI
takes via --features), record the accuracy/probability deltas, and dump top-K
token distributions — baseline + both ablated configs, for both prompt_pos and
prompt_neg — for every pair either config is active on. Unlike
run_feature_modulation.py's --dump_top_tokens_if_flat, this always dumps (not
gated on delta_acc).

Model is loaded once and reused across all anchors/concepts.

Usage:
    python -m experiments.concept_localization.pipeline.topk_edec_ablation_compare
    python -m experiments.concept_localization.pipeline.topk_edec_ablation_compare \
        --concepts gcd --top_k 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from experiments.concept_localization.pipeline.run_concept import _MODEL, _TRANSCODER_SET, _load_concept
from experiments.concept_localization.pipeline.run_feature_modulation import (
    _ensure_sweep_activations,
    _load_sweep_examples,
    _filter_active_prompts,
    dump_top_tokens_compare,
    diff_metrics,
    evaluate,
    parse_feature_name,
)
from experiments.concept_localization.sweep_utils import apply_transcoder_all
from mechinterp_qwen3.attribution_model import AttributionModel
from mechinterp_qwen3.utils.hf_utils import load_transcoder_from_hub
from mechinterp_qwen3.utils.model_utils import get_default_device, parse_dtype

ANCHORS: dict[str, list[str]] = {
    "carry": ["anchor_rank1_pos5", "anchor_rank2_pos9", "anchor_rank3_pos6",
              "anchor_rank4_pos10", "anchor_rank5_pos7", "anchor_rank6_pos8"],
    "gcd": ["anchor_rank1_pos4", "anchor_rank2_pos5", "anchor_rank3_pos6",
            "anchor_rank4_pos7", "anchor_rank5_pos9", "anchor_rank6_pos10"],
    "residue_class": ["anchor_rank1_pos3", "anchor_rank2_pos4", "anchor_rank3_pos5",
                       "anchor_rank4_pos6", "anchor_rank5_pos7", "anchor_rank6_pos8"],
    "prime": ["anchor_rank1_pos2", "anchor_rank2_pos3", "anchor_rank3_pos4",
              "anchor_rank4_pos5", "anchor_rank5_pos6", "anchor_rank6_pos7"],
}


def _feature_map(feature_keys: list[str]) -> dict[int, list[int]]:
    fm: dict[int, list[int]] = {}
    for key in feature_keys:
        spec = parse_feature_name(key)
        fm.setdefault(spec.layer, []).append(spec.feature_id)
    return fm


def _active_counts(prompts: list[dict]) -> dict[str, int]:
    pos = sum(1 for p in prompts if p["split"] == "pos")
    neg = sum(1 for p in prompts if p["split"] == "neg")
    return {"pos": pos, "neg": neg, "total": pos + neg}


def _summary_rows(concept: str, anchor: str, cfg_name: str, diff: dict) -> list[dict]:
    rows = []
    for split in ("all", "pos", "neg"):
        b = diff["baseline"][split]
        m = diff["modulated"][split]
        c = diff["change"][split]
        rows.append({
            "concept": concept, "anchor": anchor, "config": cfg_name, "split": split,
            "n": b["n"],
            "baseline_acc": round(b["accuracy"], 4),
            "ablated_acc": round(m["accuracy"], 4),
            "delta_acc": round(c["accuracy"], 4),
            "baseline_p": round(b["mean_correct_prob"], 6),
            "ablated_p": round(m["mean_correct_prob"], 6),
            "delta_p": round(c["mean_correct_prob"], 6),
        })
    return rows


def rank_top_encdec_features(anchor_dir: Path, top_k: int) -> list[str]:
    """Top-K features from the existing dec+enc_dec candidate pool (edec_features.json,
    written by delta_feature_projections.py), ranked by |score| and truncated.

    dec + enc_dec rows are pooled and deduped by feature (enc_dec's combined
    dec_cos+enc_cos score wins over dec-only when a feature appears in both, since
    it's read second and overwrites).
    """
    rows_by_feat: dict[str, dict] = {}
    for mode in ("dec", "enc_dec"):
        p = anchor_dir / "sweep" / f"delta_feature_projections_{mode}" / "edec_features.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        for side in ("pos", "neg"):
            for row in d.get(side, []):
                rows_by_feat[row["feature"]] = row
    ranked = sorted(rows_by_feat.values(), key=lambda r: -abs(r["score"]))
    return [r["feature"] for r in ranked[:top_k]]


def rank_by_cohens_d(model, sweep_dir: Path, top_k: int) -> list[str]:
    """Top-K transcoder features (across every layer cached in the anchor's sweep)
    by standardised effect size |mu_pos - mu_neg| / pooled_std, computed from the
    raw sweep residuals (not restricted to the pre-selected dec/enc_dec candidate
    pool that rank_top_encdec_features draws from)."""
    residuals = np.load(sweep_dir / "sweep_residuals.npz")
    pos_mask = residuals["pos_mask"].astype(bool)
    all_layers = sorted(int(k[3:]) for k in residuals.files if k.startswith("H_L"))
    ranked: list[tuple[float, str]] = []
    for layer in all_layers:
        H = residuals[f"H_L{layer}"]
        acts = apply_transcoder_all(model, layer, H)
        pos_acts = acts[pos_mask]
        neg_acts = acts[~pos_mask]
        pooled_std = np.sqrt(0.5 * (pos_acts.var(axis=0) + neg_acts.var(axis=0))) + 1e-8
        score = np.abs((pos_acts.mean(axis=0) - neg_acts.mean(axis=0)) / pooled_std)
        for fid, s in enumerate(score):
            ranked.append((float(s), f"L{layer}_F{fid}"))
    ranked.sort(key=lambda x: x[0], reverse=True)
    return [name for _, name in ranked[:top_k]]


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--concepts", nargs="+", default=list(ANCHORS), choices=list(ANCHORS))
    ap.add_argument("--top_k", type=int, default=10, help="Features per config")
    ap.add_argument("--dump_top_k", type=int, default=100, help="Top-K tokens saved per prompt/config")
    ap.add_argument("--model", default=_MODEL)
    ap.add_argument("--transcoder_set", default=_TRANSCODER_SET)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--template", default="T0")
    ap.add_argument("--out_dir", default="runs/concept_localization/topk_edec_compare")
    ap.add_argument("--force", action="store_true",
                     help="Recompute anchors even if modulation_topk10_compare/ already has output "
                          "(default: skip already-completed anchors, e.g. to resume after a timeout)")
    args = ap.parse_args()

    device = get_default_device()
    dtype = parse_dtype(args.dtype)
    print(f"Loading model {args.model}")
    transcoder_set, _ = load_transcoder_from_hub(
        args.transcoder_set, dtype=dtype, lazy_encoder=True, lazy_decoder=True
    )
    model = AttributionModel.from_pretrained_and_transcoders(
        args.model, transcoder_set, dtype=dtype, device=device
    )
    model.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_token_records: list[dict] = []
    summary_rows: list[dict] = []

    for concept in args.concepts:
        base = Path(f"runs/concept_localization/{concept}/{concept}_T0")
        all_pairs = _load_concept(concept, 200, args.seed)
        if args.template and args.template.lower() != "none":
            all_pairs = [p for p in all_pairs if p.template == args.template]

        for anchor in ANCHORS[concept]:
            anchor_dir = base / anchor
            sweep_dir = anchor_dir / "sweep"
            if not (sweep_dir / "sweep_residuals.npz").exists():
                print(f"[skip] {concept}/{anchor}: no sweep_residuals.npz")
                continue

            anchor_out = anchor_dir / "modulation_topk10_compare"
            dump_path = anchor_out / "generated_tokens.jsonl"
            compare_path = anchor_out / "feature_modulation_compare.json"
            if not args.force and dump_path.exists() and compare_path.exists():
                print(f"\n=== {concept} / {anchor} === [skip: already completed]")
                cached = json.loads(compare_path.read_text())
                summary_rows.extend(_summary_rows(concept, anchor, "default_encdec", cached["default_encdec"]))
                summary_rows.extend(_summary_rows(concept, anchor, "cohens_d", cached["cohens_d"]))
                all_token_records.extend(json.loads(line) for line in dump_path.read_text().splitlines())
                continue

            print(f"\n=== {concept} / {anchor} ===")
            default_feats = rank_top_encdec_features(anchor_dir, top_k=args.top_k)
            if not default_feats:
                print("  no edec features found -- skipping")
                continue
            cohens_feats = rank_by_cohens_d(model, sweep_dir, top_k=args.top_k)

            print(f"  default_encdec (top {len(default_feats)}): {default_feats}")
            print(f"  cohens_d       (top {len(cohens_feats)}): {cohens_feats}")

            fm_default = _feature_map(default_feats)
            fm_cohens = _feature_map(cohens_feats)

            feature_keys_all = sorted(set(default_feats) | set(cohens_feats))
            sweep_npz = _ensure_sweep_activations(sweep_dir, feature_keys_all, model)
            sweep_examples = _load_sweep_examples(sweep_dir)

            prompts_default = _filter_active_prompts(default_feats, sweep_npz, sweep_examples, all_pairs)
            prompts_cohens = _filter_active_prompts(cohens_feats, sweep_npz, sweep_examples, all_pairs)

            seen: set[tuple[int, str]] = set()
            prompts_union: list[dict] = []
            for p in prompts_default + prompts_cohens:
                key = (p["pair_idx"], p["split"])
                if key not in seen:
                    seen.add(key)
                    prompts_union.append(p)
            if not prompts_union:
                print("  no active prompts under either config -- skipping")
                continue
            print(f"  union active prompts: {len(prompts_union)}")

            baseline, base_rows = evaluate(
                model, prompts_union, feature_map=None,
                batch_size=args.batch_size, desc=f"{anchor}-baseline",
            )
            mod_default, rows_default = evaluate(
                model, prompts_union, feature_map=fm_default,
                batch_size=args.batch_size, desc=f"{anchor}-default_encdec",
            )
            mod_cohens, rows_cohens = evaluate(
                model, prompts_union, feature_map=fm_cohens,
                batch_size=args.batch_size, desc=f"{anchor}-cohens_d",
            )

            diff_default = diff_metrics(baseline, mod_default, base_rows, rows_default)
            diff_cohens = diff_metrics(baseline, mod_cohens, base_rows, rows_cohens)

            summary_rows.extend(_summary_rows(concept, anchor, "default_encdec", diff_default))
            summary_rows.extend(_summary_rows(concept, anchor, "cohens_d", diff_cohens))

            anchor_out.mkdir(parents=True, exist_ok=True)
            (anchor_out / "feature_modulation_compare.json").write_text(json.dumps({
                "config": {
                    "concept": concept, "anchor": anchor, "top_k": args.top_k,
                    "default_features": default_feats, "cohens_d_features": cohens_feats,
                    "default_active": _active_counts(prompts_default),
                    "cohens_d_active": _active_counts(prompts_cohens),
                    "union_active": _active_counts(prompts_union),
                },
                "default_encdec": diff_default,
                "cohens_d": diff_cohens,
            }, indent=2))

            pair_indices = sorted({p["pair_idx"] for p in prompts_union})
            dump_path = anchor_out / "generated_tokens.jsonl"
            dump_top_tokens_compare(
                model, concept, anchor, all_pairs, pair_indices,
                configs={"baseline": None, "default_encdec": fm_default, "cohens_d": fm_cohens},
                top_k=args.dump_top_k, out_path=dump_path,
            )
            all_token_records.extend(
                json.loads(line) for line in dump_path.read_text().splitlines()
            )

    combined_path = out_dir / "generated_tokens_all.jsonl"
    with combined_path.open("w") as f:
        for r in all_token_records:
            f.write(json.dumps(r) + "\n")
    print(f"\nSaved combined token dataset ({len(all_token_records)} pairs) -> {combined_path}")

    df = pd.DataFrame(summary_rows)
    print("\n=== Summary: default_encdec vs cohens_d ablation, all anchors ===")
    print(df.to_string(index=False))
    summary_path = out_dir / "summary.csv"
    df.to_csv(summary_path, index=False)
    print(f"\nSaved -> {summary_path}")


if __name__ == "__main__":
    main()
