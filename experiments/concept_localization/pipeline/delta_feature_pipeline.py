
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from numbers import Integral
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
matplotlib.use("Agg")

from experiments.concept_localization.sweep_utils import apply_transcoder_all
from experiments.concept_localization.pipeline.run_concept_sweep import _load_concept
from experiments.concept_localization.attr_survival import load_survival_set
from experiments.concept_localization.analyze import (
    FeatureMatch,
    collect_layer_residuals_batched as collect_layer_residuals,
)
from experiments.concept_localization.extract_deltas_generic import _resolve_anchor
from experiments.concept_localization.plots.visualize import plot_feature_heatmap_grid
import experiments.plot_style as ps
from mechinterp_qwen3.attribution_model import AttributionModel
from mechinterp_qwen3.utils.hf_utils import load_transcoder_from_hub
from mechinterp_qwen3.utils.model_utils import get_default_device, parse_dtype
from scripts.model_config import add_model_config_arg, default_model, default_transcoder_set, resolve_model_args

_MODEL = default_model()
_TRANSCODER_SET = default_transcoder_set()
_DEFAULT_SCRATCH_BASE = Path(
    os.environ.get("MIQ_SCRATCH_BASE", f"/rds/user/{os.environ.get('USER', '$USER')}/hpc-work/p28")
)


def _build_inputs_and_examples(model, pairs, anchor_mode):
    """Pair (token_ids, anchor) with aligned example metadata; skips unequal-length pairs."""
    inputs, examples = [], []
    skipped = 0
    for pair in pairs:
        ids_pos = model.tokenizer(pair.prompt_pos, add_special_tokens=False).input_ids
        ids_neg = model.tokenizer(pair.prompt_neg, add_special_tokens=False).input_ids
        if len(ids_pos) != len(ids_neg):
            skipped += 1
            continue
        anchor = _resolve_anchor(ids_pos, model.tokenizer, anchor_mode, None, None)
        inputs.append((ids_pos, anchor))
        inputs.append((ids_neg, anchor))
        examples.append({
            "pair_idx": len(examples),
            "template": pair.template,
            "meta": pair.meta,
            "label_pos": pair.label_pos,
        })
    if skipped:
        print(f"  skipped {skipped}/{len(pairs)} pairs with tokenization length mismatch; {len(examples)} pairs remain")
    return inputs, examples


def _build_inputs_from_saved_examples(model, records: list[dict]):
    """Build inputs from sweep_dataset_examples.pkl records with saved prompts/anchors."""
    if not isinstance(records, list) or not records:
        raise ValueError("Saved sweep examples must be a non-empty list.")

    inputs, examples = [], []
    required = {"prompt_pos", "prompt_neg", "anchor", "meta"}
    for index, rec in enumerate(records):
        if not isinstance(rec, dict):
            raise ValueError(f"Saved sweep example {index} must be a dictionary.")
        missing = sorted(required - rec.keys())
        if missing:
            raise ValueError(f"Saved sweep example {index} is missing required fields: {missing}")
        if not isinstance(rec["prompt_pos"], str) or not isinstance(rec["prompt_neg"], str):
            raise ValueError(f"Saved sweep example {index} prompts must be strings.")
        if not isinstance(rec["meta"], dict):
            raise ValueError(f"Saved sweep example {index} metadata must be a dictionary.")
        ids_pos = model.tokenizer(rec["prompt_pos"], add_special_tokens=False).input_ids
        ids_neg = model.tokenizer(rec["prompt_neg"], add_special_tokens=False).input_ids
        if len(ids_pos) != len(ids_neg):
            raise ValueError(
                f"Saved sweep example {index} has unequal tokenized prompt lengths: "
                f"positive={len(ids_pos)}, negative={len(ids_neg)}. Regenerate the sweep cache."
            )
        if isinstance(rec["anchor"], bool) or not isinstance(rec["anchor"], Integral):
            raise ValueError(f"Saved sweep example {index} has an invalid integer anchor: {rec['anchor']!r}")
        anchor = int(rec["anchor"])
        if not 0 <= anchor < len(ids_pos):
            raise ValueError(
                f"Saved sweep example {index} anchor {anchor} is outside its "
                f"{len(ids_pos)}-token prompts."
            )
        inputs.append((ids_pos, anchor))
        inputs.append((ids_neg, anchor))
        examples.append(dict(rec))
    return inputs, examples


def _load_anchor_inputs_and_examples(
    anchor_dir: Path,
    model,
    concept: str,
    anchor_mode: str,
):
    saved_examples = anchor_dir / "sweep" / "sweep_dataset_examples.pkl"
    if not saved_examples.exists():
        raise FileNotFoundError(
            f"Required sweep examples file is missing: {saved_examples}\n"
            "Run the sweep stage before feature projection, for example:\n"
            "  python -m experiments.concept_localization.pipeline.run_concept_sweep "
            f"--concept {concept} --anchor {anchor_mode} --layers all --out_dir {anchor_dir / 'sweep'}\n"
            "Do not regenerate examples implicitly; rerun anchors/sweeps after relevant code or config changes."
        )

    with saved_examples.open("rb") as f:
        records = pickle.load(f)
    inputs, examples = _build_inputs_from_saved_examples(model, records)
    print(f"Loaded {len(examples)} anchor dataset examples from {saved_examples}")
    return inputs, examples


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _stable_hash(payload: dict) -> str:
    encoded = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_sweep_cache_metadata(
    anchor_dir: Path,
    inputs: list,
    examples: list[dict],
    npz,
    expected: dict | None = None,
    expected_layers: list[int] | None = None,
) -> dict:
    metadata_path = anchor_dir / "sweep" / "sweep_residuals.meta.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Residual cache metadata is missing: {metadata_path}\n"
            "Regenerate the sweep cache with run_concept_sweep so the residuals can be validated."
        )
    metadata = json.loads(metadata_path.read_text())
    payload = metadata.get("payload")
    expected_hash = metadata.get("hash")
    if not isinstance(payload, dict) or not expected_hash:
        raise ValueError(f"Invalid residual cache metadata in {metadata_path}")
    actual_hash = _stable_hash(payload)
    if actual_hash != expected_hash:
        raise ValueError(
            f"Residual cache metadata hash mismatch for {metadata_path}: "
            f"expected {expected_hash}, recomputed {actual_hash}. Regenerate the sweep cache."
        )

    if expected:
        mismatches = []
        for key, value in expected.items():
            if payload.get(key) != value:
                mismatches.append(f"{key}: cache={payload.get(key)!r}, current={value!r}")
        if mismatches:
            raise ValueError(
                "Residual cache metadata does not match the current run context: "
                + "; ".join(mismatches)
                + ". Regenerate the sweep cache."
            )

    expected_prompts = payload.get("prompts")
    expected_examples = payload.get("examples")
    if not isinstance(expected_prompts, list) or not isinstance(expected_examples, list):
        raise ValueError(f"Cache metadata in {metadata_path} lacks prompt/example lists.")

    saved_prompts = [prompt for ex in examples for prompt in (ex["prompt_pos"], ex["prompt_neg"])]
    if saved_prompts != expected_prompts or _jsonable(examples) != expected_examples:
        raise ValueError(
            "Saved sweep examples do not exactly match the residual cache metadata. "
            "Regenerate sweep_residuals.npz and sweep_dataset_examples.pkl together."
        )
    cached_prompts = npz["prompts"].tolist() if "prompts" in npz else None
    if cached_prompts != expected_prompts:
        raise ValueError(
            "Residual cache prompts do not match sweep metadata. "
            "Regenerate sweep_residuals.npz and sweep_dataset_examples.pkl together."
        )
    if len(expected_prompts) != len(inputs):
        raise ValueError(
            f"Residual cache prompt count mismatch: metadata has {len(expected_prompts)}, "
            f"saved examples build {len(inputs)} inputs. Regenerate the sweep cache."
        )
    if len(expected_examples) != len(examples):
        raise ValueError(
            f"Residual cache example count mismatch: metadata has {len(expected_examples)}, "
            f"saved examples has {len(examples)}. Regenerate the sweep cache."
        )

    expected_pos_mask = np.tile(np.array([True, False], dtype=bool), len(examples))
    cached_pos_mask = np.asarray(npz["pos_mask"]) if "pos_mask" in npz else None
    if cached_pos_mask is None or not np.array_equal(cached_pos_mask, expected_pos_mask):
        raise ValueError("Residual cache pos_mask is missing or is not ordered [pos, neg] per example.")

    if "layers" not in npz:
        raise ValueError("Residual cache is missing its layers array.")
    cached_layers = [int(layer) for layer in npz["layers"].tolist()]
    if payload.get("layers") != cached_layers:
        raise ValueError(
            f"Residual cache layers do not match cache metadata: "
            f"cache={cached_layers}, metadata={payload.get('layers')!r}."
        )
    if len(cached_layers) != len(set(cached_layers)):
        raise ValueError("Residual cache contains duplicate layer identifiers.")
    if expected_layers is not None and cached_layers != expected_layers:
        raise ValueError(
            f"Residual cache layers do not match the model: cache={cached_layers}, "
            f"required={expected_layers}. Regenerate the sweep cache with --layers all."
        )
    for layer in cached_layers:
        key = f"H_L{layer}"
        if key not in npz:
            raise ValueError(f"Residual cache declares layer {layer} but is missing {key}.")
        rows = npz[key].shape[0] if npz[key].ndim >= 1 else 0
        if rows != len(expected_prompts):
            raise ValueError(
                f"Residual cache {key} has {rows} rows; "
                f"expected {len(expected_prompts)} prompts."
            )
    return metadata



_WITHHELD = (
    "{name} implements the transcoder feature-projection method described in the "
    "author's thesis and is withheld from the public repository pending "
    "examination/publication. Contact eid23@cam.ac.uk for access."
)


def _resolve_top_k(
    anchor_dir: Path,
    model,
    top_k: int,
    active_features: dict[int, set[int]],
    survival_set: set[tuple[int, int]] | None,
    score_mode: str,
    prelim_k: int | None = None,
) -> tuple[list[FeatureMatch], list[FeatureMatch]]:
    """Return top-k by absolute score, split into positive and negative pools.

    Implementation withheld; see _WITHHELD.
    """
    raise NotImplementedError(_WITHHELD.format(name="_resolve_top_k"))


def _match_components(m: FeatureMatch, score_mode: str) -> tuple[float, float, float]:
    """Return dec_cos, enc_cos, and signed cosine score.

    Implementation withheld; see _WITHHELD.
    """
    raise NotImplementedError(_WITHHELD.format(name="_match_components"))


def _match_score(m: FeatureMatch, score_mode: str) -> float:
    """Implementation withheld; see _WITHHELD."""
    raise NotImplementedError(_WITHHELD.format(name="_match_score"))


def _dz_weighted_score(row: dict, score_mode: str) -> float:
    """Δz-weighted feature score.

    Implementation withheld; see _WITHHELD.
    """
    raise NotImplementedError(_WITHHELD.format(name="_dz_weighted_score"))


def _unique_feature_order(*groups: list[FeatureMatch]) -> list[tuple[int, int]]:
    seen: set[tuple[int, int]] = set()
    features: list[tuple[int, int]] = []
    for group in groups:
        for m in group:
            key = (m.layer, m.feature_id)
            if key in seen:
                continue
            seen.add(key)
            features.append(key)
    return features


def _rows_for_matches(
    matches: list[FeatureMatch],
    score_mode: str,
    acts_1d: dict[str, np.ndarray],
    n_pairs: int,
) -> list[dict]:
    rows = []
    for m in matches:
        layer, fid = m.layer, m.feature_id
        dec_cos, enc_cos, score = _match_components(m, score_mode)
        key = f"L{layer}_F{fid}"
        row: dict = {
            "feature": key,
            "layer": layer,
            "feature_id": fid,
            "dec_cos": round(dec_cos, 5),
            "enc_cos": round(enc_cos, 5),
            "score": round(score, 5),
        }
        if key in acts_1d:
            arr = acts_1d[key]
            pos_arr = arr[0::2][:n_pairs]
            neg_arr = arr[1::2][:n_pairs]
            row.update({
                "mean_pos": round(float(pos_arr.mean()), 6),
                "mean_neg": round(float(neg_arr.mean()), 6),
                "std_pos": round(float(pos_arr.std()), 6),
                "std_neg": round(float(neg_arr.std()), 6),
            })
        rows.append(row)
    return rows


def _feature_match_from_row(row: dict) -> FeatureMatch:
    return FeatureMatch(
        feature_id=row["feature_id"],
        projection=row["score"],
        cos_sim=row["dec_cos"],
        layer=row["layer"],
        enc_cos_sim=row.get("enc_cos", 0.0),
    )


def _resolve_survival_set(args: argparse.Namespace) -> set[tuple[int, int]] | None:
    if args.no_attr_filter:
        print("  [attr-survival] filter disabled via --no_attr_filter")
        return None

    survival_file = args.attr_survival_file
    if survival_file is None:
        survival_file = (
            _REPO_ROOT
            / "runs"
            / "concept_localization"
            / args.concept
            / "feature_survival"
            / "survival_stats.json"
        )

    if not survival_file.exists():
        raise FileNotFoundError(
            f"Attribution-graph survival file not found for concept '{args.concept}':\n"
            f"  {survival_file}\n"
            "Generate it explicitly before feature projection, for example:\n"
            "  python -m experiments.concept_localization.attribution_feature_survival "
            f"--concept {args.concept} --graphs_dir <path/to/graphs> "
            f"--pattern {args.concept}_{args.template} --min_survival {args.attr_min_survival} "
            f"--out_dir {survival_file.parent}\n"
            "Or pass --no_attr_filter for an explicitly unfiltered exploratory run."
        )

    return load_survival_set(
        concept=args.concept,
        min_survival=args.attr_min_survival,
        survival_file=survival_file,
        required=True,
    )


def _bin_to_heatmap(acts_col: np.ndarray, examples: list[dict]) -> np.ndarray:
    """Bin activations into a 10×10 grid by (a%10, b%10) across all prompts (pos+neg)."""
    sums   = np.zeros((10, 10), dtype=np.float64)
    counts = np.zeros((10, 10), dtype=np.int64)
    for i, ex in enumerate(examples):
        meta = ex["meta"]
        for j, key_a, key_b in ((2 * i, "a_pos", "b_pos"), (2 * i + 1, "a_neg", "b_neg")):
            if j < len(acts_col) and key_a in meta and key_b in meta:
                a, b = int(meta[key_a]) % 10, int(meta[key_b]) % 10
                sums[a, b]   += float(acts_col[j])
                counts[a, b] += 1
    grid = np.full((10, 10), np.nan)
    grid[counts > 0] = sums[counts > 0] / counts[counts > 0]
    return grid.astype(np.float32)


def _metadata_int(meta: dict, key: str, example_index: int) -> int:
    value = meta.get(key)
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(
            f"Plot metadata field {key!r} in example {example_index} must be an integer; "
            f"got {value!r}."
        )
    return int(value)


def _classify_plot_metadata(examples: list[dict]) -> str:
    """Validate every example and return the supported plot schema: ``1d`` or ``2d``."""
    if not examples:
        raise ValueError("Cannot plot EDEC features without examples.")

    first_meta = examples[0].get("meta")
    if not isinstance(first_meta, dict):
        raise ValueError("Plot example 0 is missing dictionary metadata.")
    has_a = {"a_pos", "a_neg"}.issubset(first_meta)
    has_b = {"b_pos", "b_neg"}.issubset(first_meta)
    if has_a and has_b:
        schema = "2d"
        required = {"a_pos", "a_neg", "b_pos", "b_neg"}
    elif has_a and not ({"b_pos", "b_neg"} & first_meta.keys()):
        schema = "1d"
        required = {"a_pos", "a_neg"}
    else:
        return "sequential"

    modulus: int | None = None
    for index, ex in enumerate(examples):
        meta = ex.get("meta")
        if not isinstance(meta, dict):
            raise ValueError(f"Plot example {index} is missing dictionary metadata.")
        missing = sorted(required - meta.keys())
        if missing:
            raise ValueError(
                f"Plot example {index} does not match the {schema.upper()} schema; "
                f"missing fields: {missing}."
            )
        for key in required:
            _metadata_int(meta, key, index)

        b_fields = {"b_pos", "b_neg"} & meta.keys()
        if schema == "1d" and b_fields:
            raise ValueError(
                f"Plot example {index} mixes 1D and 2D metadata fields: {sorted(b_fields)}."
            )
        if schema == "1d":
            modulus_keys = [key for key in ("m", "g") if key in meta]
            if not modulus_keys:
                raise ValueError(
                    f"Plot example {index} requires an integer modulus field 'm' or 'g'."
                )
            values = {_metadata_int(meta, key, index) for key in modulus_keys}
            if len(values) != 1:
                raise ValueError(f"Plot example {index} has conflicting modulus fields: {values}.")
            current_modulus = values.pop()
            if current_modulus <= 0:
                raise ValueError(f"Plot example {index} modulus must be positive, got {current_modulus}.")
            if modulus is None:
                modulus = current_modulus
            elif current_modulus != modulus:
                raise ValueError(
                    f"Plot examples use inconsistent moduli: expected {modulus}, "
                    f"example {index} has {current_modulus}."
                )
    return schema


def _get_modulus(examples: list[dict]) -> int:
    """Extract the validated, consistent modulus from 1D example metadata."""
    if _classify_plot_metadata(examples) != "1d":
        raise ValueError("A modulus is only defined for 1D EDEC plot metadata.")
    meta = examples[0]["meta"]
    return int(meta["m"] if "m" in meta else meta["g"])




def _bin_to_1d_bar(acts_col: np.ndarray, examples: list[dict], modulus: int) -> tuple[np.ndarray, np.ndarray]:
    """Bin activations by a mod modulus, returning (pos_bar, neg_bar) of shape (modulus,)."""
    sums_pos   = np.zeros(modulus, dtype=np.float64)
    counts_pos = np.zeros(modulus, dtype=np.int64)
    sums_neg   = np.zeros(modulus, dtype=np.float64)
    counts_neg = np.zeros(modulus, dtype=np.int64)
    for i, ex in enumerate(examples):
        meta = ex["meta"]
        a_pos = int(meta["a_pos"]) % modulus
        a_neg = int(meta["a_neg"]) % modulus
        if 2 * i < len(acts_col):
            sums_pos[a_pos]   += float(acts_col[2 * i])
            counts_pos[a_pos] += 1
        if 2 * i + 1 < len(acts_col):
            sums_neg[a_neg]   += float(acts_col[2 * i + 1])
            counts_neg[a_neg] += 1
    pos = np.divide(sums_pos, counts_pos, out=np.zeros_like(sums_pos), where=counts_pos > 0).astype(np.float32)
    neg = np.divide(sums_neg, counts_neg, out=np.zeros_like(sums_neg), where=counts_neg > 0).astype(np.float32)
    return pos, neg


def _plot_1d_grid(
    rows: list[dict],
    acts_1d: dict[str, np.ndarray],
    examples: list[dict],
    out_path: Path,
    concept: str,
    anchor_label: str,
    ncols: int = 5,
    show_title: bool = True,
    dpi: int | None = None,
) -> None:
    import matplotlib.pyplot as plt
    ps.apply()
    modulus = _get_modulus(examples)
    xs = np.arange(modulus)
    width = 0.4

    valid = [(r["layer"], r["feature_id"], r["feature"],
              r.get("dec_cos", r.get("score", 0.0)), r.get("enc_cos", 0.0))
             for r in rows if r["feature"] in acts_1d]
    if not valid:
        raise RuntimeError("No selected feature activations are available for the 1D EDEC plot.")

    nrows = (len(valid) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2.5 * nrows), squeeze=False)
    if show_title:
        fig.suptitle(f"{concept} — {anchor_label} (a mod {modulus})", fontsize=9)

    for idx, (layer, fid, feat_key, dec_cos, enc_cos) in enumerate(valid):
        ax = axes[idx // ncols][idx % ncols]
        acts_col = acts_1d[feat_key]
        pos_bar, neg_bar = _bin_to_1d_bar(acts_col, examples, modulus)
        # Colors/alpha are load-bearing: plot_concept_report_figure.py's
        # _extract_1d_profile re-derives bar heights from these exact rendered
        # (alpha-blended) RGB values via _BAR_BLUE/_BAR_ORANGE. Do not change them.
        ax.bar(xs - width / 2, pos_bar, width, color="#4c72b0", alpha=0.75, label="pos")
        ax.bar(xs + width / 2, neg_bar, width, color="#dd8452", alpha=0.75, label="neg")

        if show_title:
            ax.set_title(
                f"$L^{{{layer}}}_{{{fid}}}$  dec={dec_cos:+.3f} enc={enc_cos:+.3f}",
                fontsize=6,
            )
        ax.set_xlabel(f"a mod {modulus}", fontsize=6)
        ax.set_xticks(xs)
        ax.tick_params(labelsize=5)
        ax.grid(False)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if idx == 0:
            ax.legend(fontsize=4, frameon=False)

    for idx in range(len(valid), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", **({"dpi": dpi} if dpi else {}))
    plt.close(fig)
    print(f"  Saved 1D bar grid → {out_path}")


def _plot_sequential_grid(
    rows: list[dict],
    acts_1d: dict[str, np.ndarray],
    n_pairs: int,
    out_path: Path,
    concept: str,
    anchor_label: str,
    ncols: int = 5,
    show_title: bool = True,
    dpi: int | None = None,
) -> None:
    import matplotlib.pyplot as plt
    ps.apply()

    valid = [(r["layer"], r["feature_id"], r["feature"],
              r.get("dec_cos", r.get("score", 0.0)), r.get("enc_cos", 0.0))
             for r in rows if r["feature"] in acts_1d]
    if not valid:
        raise RuntimeError("No selected feature activations are available for the sequential EDEC plot.")

    xs = np.arange(n_pairs)
    nrows = (len(valid) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2.5 * nrows), squeeze=False)
    if show_title:
        fig.suptitle(f"{concept} — {anchor_label}", fontsize=9)

    for idx, (layer, fid, feat_key, dec_cos, enc_cos) in enumerate(valid):
        ax = axes[idx // ncols][idx % ncols]
        col = acts_1d[feat_key]
        pos_vals = col[0::2][:n_pairs].astype(np.float32)
        neg_vals = col[1::2][:n_pairs].astype(np.float32)
        # Points, not lines: pair index is an arbitrary sample ordering, not a
        # continuous axis, so connecting them would imply a trend between
        # samples that isn't there.
        ax.scatter(xs, pos_vals, color="#4c72b0", alpha=0.8, s=8, label="pos")
        ax.scatter(xs, neg_vals, color="#dd8452", alpha=0.8, s=8, label="neg")
        if show_title:
            ax.set_title(f"$L^{{{layer}}}_{{{fid}}}$  dec={dec_cos:+.3f} enc={enc_cos:+.3f}", fontsize=6)
        ax.set_xlabel("pair index", fontsize=6)
        ax.tick_params(labelsize=5)
        ax.grid(False)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if idx == 0:
            ax.legend(fontsize=5, frameon=False)

    for idx in range(len(valid), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", **({"dpi": dpi} if dpi else {}))
    plt.close(fig)
    print(f"  Saved sequential grid → {out_path}")


def run_one_mode(
    anchor_dir: Path,
    model,
    inputs: list,
    examples: list[dict],
    active_features: dict[int, set[int]],
    survival_set: set[tuple[int, int]] | None,
    score_mode: str,
    top_k: int,
    concept: str,
    H_cached: dict[int, np.ndarray],
    rank_by: str = "score",
    display_inputs: list | None = None,
    display_examples: list[dict] | None = None,
) -> None:
    if not examples or len(inputs) != 2 * len(examples):
        raise ValueError(
            f"Projection dataset is empty or misaligned: {len(inputs)} inputs for "
            f"{len(examples)} paired examples."
        )
    if (display_inputs is None) != (display_examples is None):
        raise ValueError("display_inputs and display_examples must either both be provided or both be omitted.")
    if display_inputs is None:
        grid_examples = examples
    else:
        if not display_examples or not display_inputs:
            raise ValueError("The explicitly requested display dataset is empty; cannot produce an EDEC plot.")
        if len(display_inputs) != 2 * len(display_examples):
            raise ValueError(
                f"Display dataset is misaligned: {len(display_inputs)} inputs for "
                f"{len(display_examples)} paired examples."
            )
        grid_examples = display_examples
    schema = _classify_plot_metadata(grid_examples)

    mode_suffix = score_mode.replace("+", "_")
    out_dir = anchor_dir / "sweep" / f"delta_feature_projections_{mode_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    use_dz = rank_by == "dz"
    prelim_k = top_k * 5 if use_dz else None
    matches_pos, matches_neg = _resolve_top_k(
        anchor_dir, model, top_k, active_features, survival_set, score_mode,
        prelim_k=prelim_k,
    )
    features = _unique_feature_order(matches_pos, matches_neg)
    if not features:
        raise RuntimeError(
            f"No active/surviving features for mode={score_mode} in {anchor_dir.name}; "
            "cannot produce the requested EDEC plot."
        )

    layers = sorted({feature_layer for feature_layer, _ in features})
    labels_pos = [f"L{m.layer}_F{m.feature_id}" for m in matches_pos]
    labels_neg = [f"L{m.layer}_F{m.feature_id}" for m in matches_neg]
    print(f"Top-{top_k} positive active features ({score_mode}): {labels_pos}")
    print(f"Top-{top_k} negative active features ({score_mode}): {labels_neg}")
    print(f"Concept: {concept}  layers: {layers}")

    missing_layers = sorted(set(layers) - H_cached.keys())
    if missing_layers:
        raise ValueError(f"Residual cache is missing selected feature layers: {missing_layers}")
    H_all = {layer: H_cached[layer] for layer in layers}

    acts_1d: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for layer in layers:
            if H_all[layer].shape[0] != len(inputs):
                raise ValueError(
                    f"Layer {layer} residuals have {H_all[layer].shape[0]} rows; "
                    f"expected {len(inputs)} inputs."
                )
            acts = apply_transcoder_all(model, layer, H_all[layer])
            for feature_layer, fid in features:
                if feature_layer == layer:
                    acts_1d[f"L{feature_layer}_F{fid}"] = acts[:, fid].astype(np.float32)

    n_pairs = len(examples)
    rows_pos = _rows_for_matches(matches_pos, score_mode, acts_1d, n_pairs)
    rows_neg = _rows_for_matches(matches_neg, score_mode, acts_1d, n_pairs)

    if use_dz:
        all_rows = rows_pos + rows_neg
        for r in all_rows:
            r["score"] = round(_dz_weighted_score(r, score_mode), 5)
        all_rows.sort(key=lambda r: abs(r["score"]), reverse=True)
        all_rows = all_rows[:top_k]
        rows_pos = [r for r in all_rows if r["score"] >= 0]
        rows_neg = [r for r in all_rows if r["score"] < 0]

    ranked_by = f"dz_weighted_{score_mode} (prelim: abs({score_mode}))" if use_dz else f"abs({score_mode})"
    edec_json = {
        "config": {"score_mode": score_mode, "top_k": top_k, "ranked_by": ranked_by},
        "pos": rows_pos,
        "neg": rows_neg,
    }
    (out_dir / "edec_features.json").write_text(json.dumps(edec_json, indent=2))
    print(f"Saved edec features -> {out_dir / 'edec_features.json'}")

    with open(out_dir / "selected_feature_examples.pkl", "wb") as f:
        pickle.dump(examples, f)
    print(f"Saved selected feature metadata -> {out_dir}")

    if display_inputs is None:
        grid_acts_1d = acts_1d
    else:
        grid_inputs = display_inputs
        print(f"  Computing display activations over {len(grid_inputs)} dense inputs...")
        H_display = collect_layer_residuals(model, grid_inputs, layers)
        grid_acts_1d = {}
        with torch.no_grad():
            for layer in layers:
                if layer not in H_display or H_display[layer].shape[0] != len(grid_inputs):
                    raise ValueError(f"Display residuals for layer {layer} are missing or misaligned.")
                acts = apply_transcoder_all(model, layer, H_display[layer])
                for feature_layer, fid in features:
                    if feature_layer == layer:
                        grid_acts_1d[f"L{feature_layer}_F{fid}"] = acts[:, fid].astype(np.float32)
    projections: dict[int, list[FeatureMatch]] = {}
    for row in rows_pos + rows_neg:
        projections.setdefault(row["layer"], []).append(_feature_match_from_row(row))

    plot_path = out_dir / "edec_topk_grid.pdf"
    plot_path.unlink(missing_ok=True)
    if schema == "sequential":
        _plot_sequential_grid(
            rows=rows_pos + rows_neg,
            acts_1d=grid_acts_1d,
            n_pairs=len(grid_examples),
            out_path=plot_path,
            concept=concept,
            anchor_label=f"{anchor_dir.name}; +topk then -topk",
        )
    elif schema == "1d":
        _plot_1d_grid(
            rows=rows_pos + rows_neg,
            acts_1d=grid_acts_1d,
            examples=grid_examples,
            out_path=plot_path,
            concept=concept,
            anchor_label=f"{anchor_dir.name}; +topk then -topk",
        )
    else:
        coverage = np.zeros((10, 10), dtype=bool)
        for ex in grid_examples:
            meta = ex["meta"]
            coverage[int(meta["a_pos"]) % 10, int(meta["b_pos"]) % 10] = True
            coverage[int(meta["a_neg"]) % 10, int(meta["b_neg"]) % 10] = True

        matrices: dict[tuple[int, int], np.ndarray] = {}
        for row in rows_pos + rows_neg:
            layer, fid = row["layer"], row["feature_id"]
            if row["feature"] in grid_acts_1d:
                matrices[(layer, fid)] = _bin_to_heatmap(grid_acts_1d[row["feature"]], grid_examples)
        if not matrices:
            raise RuntimeError("No selected feature activations are available for the 2D EDEC plot.")
        plot_feature_heatmap_grid(
            matrices=matrices,
            projections=projections,
            out_path=plot_path,
            concept=concept,
            anchor_label=f"{anchor_dir.name}; +topk then -topk",
            xlabel="$a_0$",
            ylabel="$b_0$",
            ncols=5,
            coverage=coverage,
        )
    if not plot_path.is_file() or plot_path.stat().st_size == 0:
        raise RuntimeError(f"EDEC plotting completed without writing a non-empty PDF: {plot_path}")

def run_for_anchor(anchor_dir: Path, model, args) -> None:
    """Run delta feature projections for a pre-loaded model.

    Callable from run_anchor_pipeline to avoid reloading the model.
    args must have: concept, score_mode, top_k, seed, rank_by,
                    display_n_pairs, no_attr_filter, attr_min_survival,
                    attr_survival_file, graphs_dir, template.
    """
    anchor_dir = Path(anchor_dir)
    survival_set = _resolve_survival_set(args)

    cfg = json.loads((anchor_dir / "results.json").read_text())["config"]
    if "anchor_mode" in cfg:
        anchor_mode = str(cfg["anchor_mode"])
    elif "anchor_pos" in cfg:
        anchor_mode = str(cfg["anchor_pos"])
    else:
        raise ValueError(
            f"{anchor_dir / 'results.json'} config must define anchor_mode or anchor_pos."
        )
    inputs, examples = _load_anchor_inputs_and_examples(
        anchor_dir, model, args.concept, anchor_mode,
    )

    sweep_residuals_path = anchor_dir / "sweep" / "sweep_residuals.npz"
    if not sweep_residuals_path.exists():
        raise FileNotFoundError(
            f"Required residual cache is missing: {sweep_residuals_path}\n"
            "Run experiments.concept_localization.pipeline.run_concept_sweep first."
        )
    print(f"Loading cached residuals from {sweep_residuals_path}")
    npz = np.load(str(sweep_residuals_path), allow_pickle=True)
    all_layers = list(range(len(model.transcoders)))
    _validate_sweep_cache_metadata(
        anchor_dir, inputs, examples, npz,
        expected={
            "concept": args.concept,
            "anchor": anchor_mode,
            "model": getattr(args, "model", _MODEL),
            "transcoder_set": getattr(args, "transcoder_set", _TRANSCODER_SET),
            "dtype": getattr(args, "dtype", "bfloat16"),
        },
        expected_layers=all_layers,
    )
    H_cached = {
        int(layer): npz[f"H_L{layer}"].astype(np.float32)
        for layer in npz["layers"]
    }
    print(f"  Loaded validated cache: {len(H_cached)} layers, {next(iter(H_cached.values())).shape[0]} examples")

    active_features: dict[int, set[int]] = {}
    for layer in all_layers:
        acts = apply_transcoder_all(model, layer, H_cached[layer])
        active_ids = set(np.where(acts.max(axis=0) > 0)[0].tolist())
        active_features[layer] = active_ids
        print(f"  Layer {layer}: {len(active_ids)} active features / {acts.shape[1]}")

    if survival_set is not None:
        print(
            f"  [attr-survival] will additionally filter by attribution graph membership "
            f"(min_survival={args.attr_min_survival})"
        )

    display_inputs: list | None = None
    display_examples: list[dict] | None = None
    if getattr(args, "display_n_pairs", None) is not None:
        if args.display_n_pairs <= 0:
            raise ValueError("--display_n_pairs must be a positive integer.")
        print(f"\nGenerating {args.display_n_pairs} dense display pairs for grid visualization...")
        display_pairs = _load_concept(args.concept, args.display_n_pairs, getattr(args, "seed", 42))
        cfg2 = json.loads((anchor_dir / "results.json").read_text()).get("config", {})
        templates = set(cfg2.get("templates", []))
        if templates:
            display_pairs = [p for p in display_pairs if p.template in templates]
        display_inputs, display_examples = _build_inputs_and_examples(
            model, display_pairs, anchor_mode
        )
        if not display_inputs or not display_examples:
            raise ValueError(
                "--display_n_pairs produced no valid examples after template filtering and tokenization."
            )
        print(f"  {len(display_examples)} display pairs across {len(set(p.template for p in display_pairs))} template(s)")

    for mode in args.score_mode:
        print(f"\n{'='*50}")
        print(f"Score mode: {mode}")
        print(f"{'='*50}")
        run_one_mode(
            anchor_dir=anchor_dir,
            model=model,
            inputs=inputs,
            examples=examples,
            active_features=active_features,
            survival_set=survival_set,
            score_mode=mode,
            top_k=args.top_k,
            concept=args.concept,
            H_cached=H_cached,
            rank_by=getattr(args, "rank_by", "score"),
            display_inputs=display_inputs,
            display_examples=display_examples,
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__
    )
    ap.add_argument("--anchor_dir", required=True,
                    help="Anchor run dir containing deltas.pt and results.json")
    ap.add_argument("--concept", default="carry")
    add_model_config_arg(ap)
    ap.add_argument("--model", default=None)
    ap.add_argument("--transcoder_set", default=None)
    ap.add_argument("--score_mode", nargs="+", default=["enc+dec"],
                    choices=["dec", "enc", "enc+dec"],
                    help="Score mode(s) to run; each saves to its own delta_feature_projections_{mode}/ subdir")
    ap.add_argument("--top_k", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--rank_by", default="dz", choices=["score", "dz"],
                    help="Final ranking criterion: 'dz' (default) = Δz × dec_cos [+ enc_cos] "
                         "with 5× cosine pre-selection; 'score' = abs(cosine) only")
    ap.add_argument("--display_n_pairs", type=int, default=None,
                    help="If set, generate this many pairs (dense grid) for the heatmap display "
                         "only. JSON stats still use the sweep pairs. E.g. 990 = ~10 per cell.")
    ap.add_argument("--no_attr_filter", action="store_true",
                    help="Disable attribution-graph survival pre-filter")
    ap.add_argument("--attr_min_survival", type=float, default=0.05,
                    help="Min fraction of graphs a feature must survive to pass the filter")
    ap.add_argument("--attr_survival_file", type=Path, default=None)
    ap.add_argument("--graphs_dir", type=Path, default=None,
                    help="Deprecated here: generate survival_stats.json explicitly with attribution_feature_survival.py")
    ap.add_argument("--template", default="T0",
                    help="Prompt template key used in attribution graph directory names")
    args = ap.parse_args()
    resolve_model_args(args)

    device, dtype = get_default_device(), parse_dtype(args.dtype)
    tc_set, _ = load_transcoder_from_hub(
        args.transcoder_set, dtype=dtype, lazy_encoder=True, lazy_decoder=True
    )
    model = AttributionModel.from_pretrained_and_transcoders(
        args.model, tc_set, dtype=dtype, device=device
    )
    model.eval()

    run_for_anchor(Path(args.anchor_dir), model, args)


if __name__ == "__main__":
    main()
