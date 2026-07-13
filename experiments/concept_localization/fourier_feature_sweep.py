"""
Batch Fourier analysis of transcoder features identified by delta_feature_projections.

For each selected feature, compute the mean transcoder activation binned by operand
residue class, apply a DFT over the N residue bins, and report the dominant harmonic
(k*), its phase, the energy fraction in each harmonic, and the R² from k* alone.

Feature-selection modes:
  default     reads edec_features.json produced by delta_feature_projections
  --top_k N   re-derives top N features from deltas.pt (N//2 pos + N//2 neg),
              no model load required

Residue binning modes:
  default      uses existing sweep_residuals.npz (imbalanced: r=0 has 100 samples,
               r=1..6 have ~15 each)
  --balanced_n N  generates N fresh single prompts per residue class, runs the
               full model to capture residual stream, applies transcoder.  Requires
               loading AttributionModel + transcoders — submit via sbatch for large N.

Usage
-----
    python -m experiments.concept_localization.fourier_feature_sweep \
        --anchor_dir runs/concept_localization/residue_class/residue_class_T0/anchor_rank1_pos3 \
        --top_k 30

    # balanced binning, 30 prompts per class:
    sbatch scripts/sbatch_run.sh \
        python -m experiments.concept_localization.fourier_feature_sweep \
        --anchor_dir runs/concept_localization/residue_class/residue_class_T0/anchor_rank1_pos3 \
        --top_k 30 --balanced_n 30
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_TC_SNAPSHOT = (
    "/rds/user/eid23/hpc-work/p28/cache/hf/hub"
    "/models--mwhanna--qwen3-4b-transcoders"
    "/snapshots/94d176260ac39ce2f882b8b09aba8c118df29bb3"
)


# ── transcoder I/O ────────────────────────────────────────────────────────────

def _tc_path(layer: int) -> str:
    return os.path.join(_TC_SNAPSHOT, f"layer_{layer}.safetensors")


def _load_enc(layer: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    from safetensors import safe_open
    with safe_open(_tc_path(layer), framework="pt") as f:
        W = f.get_tensor("W_enc").float().to(device)
        b = f.get_tensor("b_enc").float().to(device)
    return W, b


def _load_enc_dec(layer: int, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from safetensors import safe_open
    with safe_open(_tc_path(layer), framework="pt") as f:
        W_enc = f.get_tensor("W_enc").float().to(device)
        b_enc = f.get_tensor("b_enc").float().to(device)
        W_dec = f.get_tensor("W_dec").float().to(device)
        b_dec = f.get_tensor("b_dec").float().to(device)
    return W_enc, b_enc, W_dec, b_dec


def _encode(H: np.ndarray, W_enc: torch.Tensor, b_enc: torch.Tensor) -> np.ndarray:
    """ReLU transcoder encoding. H: (N, d_model) → (N, d_tc) float32."""
    H_t = torch.from_numpy(H).to(W_enc.device, W_enc.dtype)
    acts = torch.relu(H_t @ W_enc.T + b_enc)
    return acts.cpu().numpy()


# ── balanced prompt generation + model inference ─────────────────────────────

def _generate_balanced_inputs(
    n_per_class: int,
    modulus: int,
    anchor: int,
    tokenizer,
    seed: int = 42,
) -> tuple[list[tuple[list[int], int]], list[int]]:
    """
    Generate n_per_class individual T0 prompts for each residue class 0..modulus-1.

    Returns (prompts_and_anchors, residue_labels) where prompts_and_anchors is
    a list of (token_ids, anchor) ready for collect_layer_residuals_batched, and
    residue_labels[i] gives the residue class of prompt i.
    """
    template = "calc: {a}%7= "
    rng = random.Random(seed)
    inputs: list[tuple[list[int], int]] = []
    labels: list[int] = []

    for r in range(modulus):
        # 3-digit numbers a ≡ r (mod modulus)
        k_min = max(1, (100 - r + modulus - 1) // modulus)
        k_max = (999 - r) // modulus
        pool = list(range(k_min, k_max + 1))
        rng.shuffle(pool)
        count = 0
        for k in pool:
            if count >= n_per_class:
                break
            a = modulus * k + r
            if not (100 <= a <= 999):
                continue
            prompt = template.format(a=a)
            ids = tokenizer(prompt, add_special_tokens=False).input_ids
            inputs.append((ids, anchor))
            labels.append(r)
            count += 1
        if count < n_per_class:
            raise RuntimeError(
                f"Could not generate {n_per_class} 3-digit numbers for residue {r} mod {modulus}; "
                f"got {count}. Reduce --balanced_n."
            )

    return inputs, labels


def _collect_balanced_acts(
    model,
    inputs: list[tuple[list[int], int]],
    layers: list[int],
    features_by_layer: dict[int, list[int]],
    batch_size: int = 32,
) -> dict[str, np.ndarray]:
    """
    Run model on balanced inputs, apply transcoder at each layer,
    return dict feature_label -> (N,) float32 activation array.
    """
    from experiments.concept_localization.analyze import collect_layer_residuals_batched
    from experiments.concept_localization.sweep_utils import apply_transcoder_all

    H = collect_layer_residuals_batched(model, inputs, layers, batch_size=batch_size)

    acts_out: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for layer in layers:
            if layer not in H:
                continue
            acts = apply_transcoder_all(model, layer, H[layer])  # (N, d_tc)
            for fid in features_by_layer.get(layer, []):
                acts_out[f"L{layer}_F{fid}"] = acts[:, fid].astype(np.float32)
    return acts_out


def _residue_response_balanced(
    feat_acts: np.ndarray,
    labels: list[int],
    modulus: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean activation per residue class from a balanced labelled array."""
    sums   = np.zeros(modulus, dtype=np.float64)
    counts = np.zeros(modulus, dtype=np.int64)
    for act, r in zip(feat_acts, labels):
        sums[r]   += float(act)
        counts[r] += 1
    v = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    return v.astype(np.float32), counts


# ── inline feature ranking (no model) ────────────────────────────────────────

def _cosine_scores(W: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Signed cosine of each row of W with delta. Returns (d_tc,)."""
    delta_norm = delta.norm().clamp(min=1e-8)
    row_norms  = W.norm(dim=1).clamp(min=1e-8)
    return (W @ delta) / (row_norms * delta_norm)


def _rank_features(
    deltas_path: Path,
    residuals_path: Path,
    n_each: int,
    device: str,
    candidates_per_layer: int = 2000,
) -> tuple[list[dict], list[dict]]:
    """
    Return (pos_features, neg_features), each a list of n_each dicts with keys:
      feature, layer, feature_id, dec_cos, enc_cos, score, side

    Strategy:
      1. For each layer, load W_enc + W_dec, compute enc+dec cosine with delta[layer].
      2. Keep top candidates_per_layer from each direction per layer.
      3. Merge all layers globally, deduplicate, sort by |score|.
      4. Check activity from cached residuals; filter dead features.
      5. Take top n_each pos and top n_each neg.
    """
    raw    = torch.load(str(deltas_path), map_location="cpu", weights_only=False)
    deltas = raw["all"]                         # {layer: (2560,) bfloat16}
    n_layers = len(deltas)

    npz = np.load(str(residuals_path), allow_pickle=True)

    # pass 1: per-layer candidate scores (cheap: two matrix-vector products)
    candidates: list[dict] = []
    print(f"  Pass 1: scoring {n_layers} layers...")
    for layer in sorted(deltas.keys()):
        key = f"H_L{layer}"
        if key not in npz:
            continue
        delta = deltas[layer].float().to(device)
        if delta.norm() < 1e-8:
            continue

        W_enc, b_enc, W_dec, b_dec = _load_enc_dec(layer, device)
        enc_cos = _cosine_scores(W_enc, delta)
        dec_cos = _cosine_scores(W_dec, delta)
        score   = enc_cos + dec_cos

        # top candidates_per_layer positive
        top_pos = score.topk(min(candidates_per_layer, score.numel()), largest=True)
        for fid, s in zip(top_pos.indices.tolist(), top_pos.values.tolist()):
            if s <= 0:
                break
            candidates.append({
                "layer": layer, "feature_id": fid,
                "dec_cos": float(dec_cos[fid]), "enc_cos": float(enc_cos[fid]),
                "score": s,
            })

        # top candidates_per_layer negative
        top_neg = score.topk(min(candidates_per_layer, score.numel()), largest=False)
        for fid, s in zip(top_neg.indices.tolist(), top_neg.values.tolist()):
            if s >= 0:
                break
            candidates.append({
                "layer": layer, "feature_id": fid,
                "dec_cos": float(dec_cos[fid]), "enc_cos": float(enc_cos[fid]),
                "score": s,
            })

        del W_enc, b_enc, W_dec, b_dec, enc_cos, dec_cos, score
        print(f"    L{layer}: {len([c for c in candidates if c['layer']==layer])} candidates")

    # deduplicate and sort by |score|
    seen: set[tuple[int,int]] = set()
    deduped: list[dict] = []
    for c in sorted(candidates, key=lambda x: abs(x["score"]), reverse=True):
        k = (c["layer"], c["feature_id"])
        if k not in seen:
            seen.add(k)
            deduped.append(c)

    print(f"  Pass 1 done: {len(deduped)} unique candidates")

    # pass 2: activity filter — only keep layers that appear in top 4×n_each candidates
    needed_pool = deduped[: 4 * (n_each * 2)]
    needed_layers = sorted({c["layer"] for c in needed_pool})
    print(f"  Pass 2: activity filter on {len(needed_layers)} layers...")

    active: dict[int, set[int]] = {}
    for layer in needed_layers:
        key = f"H_L{layer}"
        H = npz[key].astype(np.float32)
        W_enc, b_enc = _load_enc(layer, device)
        acts = _encode(H, W_enc, b_enc)
        del W_enc, b_enc
        active[layer] = set(int(i) for i in np.where(acts.max(axis=0) > 0)[0])
        print(f"    L{layer}: {len(active[layer])} active features")

    filtered = [
        c for c in deduped
        if c["layer"] in active and c["feature_id"] in active[c["layer"]]
    ]
    print(f"  After activity filter: {len(filtered)} candidates")

    pos_pool = [c for c in filtered if c["score"] >= 0]
    neg_pool = [c for c in filtered if c["score"] <  0]

    def _make_rows(pool: list[dict], side: str, n: int) -> list[dict]:
        rows = []
        for c in pool[:n]:
            label = f"L{c['layer']}_F{c['feature_id']}"
            rows.append({
                "feature":    label,
                "layer":      c["layer"],
                "feature_id": c["feature_id"],
                "dec_cos":    round(c["dec_cos"], 5),
                "enc_cos":    round(c["enc_cos"], 5),
                "score":      round(c["score"], 5),
                "side":       side,
            })
        return rows

    return _make_rows(pos_pool, "pos", n_each), _make_rows(neg_pool, "neg", n_each)


# ── residue binning ───────────────────────────────────────────────────────────

def _residue_response(
    feat_acts: np.ndarray,
    examples: list[dict],
    modulus: int,
) -> tuple[np.ndarray, np.ndarray]:
    """v[r] = mean activation over all examples where a % modulus == r."""
    sums   = np.zeros(modulus, dtype=np.float64)
    counts = np.zeros(modulus, dtype=np.int64)
    for i, ex in enumerate(examples):
        meta  = ex["meta"]
        r_pos = int(meta["a_pos"]) % modulus
        r_neg = int(meta["a_neg"]) % modulus
        if 2 * i < len(feat_acts):
            sums[r_pos]   += float(feat_acts[2 * i])
            counts[r_pos] += 1
        if 2 * i + 1 < len(feat_acts):
            sums[r_neg]   += float(feat_acts[2 * i + 1])
            counts[r_neg] += 1
    v = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    return v.astype(np.float32), counts


# ── DFT analysis ─────────────────────────────────────────────────────────────

def _dft_analysis(v: np.ndarray) -> dict:
    N     = len(v)
    F     = np.fft.fft(v)
    power = np.abs(F) ** 2
    total = power.sum()
    total_ac = total - power[0]

    harmonics = {}
    for k in range(1, N // 2 + 1):
        harmonic_power = 2.0 * power[k] if (k != N - k) else power[k]
        pct_ac = float(harmonic_power / total_ac) if total_ac > 1e-12 else 0.0
        amp    = (2.0 / N) * float(np.abs(F[k]))
        phase  = float(np.degrees(np.angle(F[k])))
        harmonics[k] = {
            "power_frac_ac": round(pct_ac, 4),
            "amplitude":     round(amp, 5),
            "phase_deg":     round(phase, 2),
        }

    dominant_k = max(harmonics, key=lambda k: harmonics[k]["power_frac_ac"])

    v_mean = float(v.mean())
    dc     = float(F[0].real) / N
    r_idx  = np.arange(N)
    v_hat  = dc + (2.0 / N) * np.real(
        F[dominant_k] * np.exp(2j * np.pi * dominant_k * r_idx / N)
    )
    ss_tot = float(np.sum((v - v_mean) ** 2))
    ss_res = float(np.sum((v - v_hat) ** 2))
    r2     = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

    return {
        "dc":          round(dc, 5),
        "harmonics":   harmonics,
        "dominant_k":  dominant_k,
        "dominant_r2": round(r2, 4),
        "v":           [round(float(x), 5) for x in v],
    }


# ── family plot ──────────────────────────────────────────────────────────────

def _plot_families(results: list[dict], out_path: Path) -> None:
    """One panel per dominant-k family; top-scoring representative each."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import experiments.plot_style as ps
    ps.apply()

    from collections import defaultdict

    # group by dominant_k; pick highest |score| per group
    by_k: dict[int, dict] = {}
    for r in results:
        k = r["dominant_k"]
        if k not in by_k or abs(r.get("score") or 0) > abs(by_k[k].get("score") or 0):
            by_k[k] = r

    ks = sorted(by_k)
    N  = results[0]["modulus"]
    xs = np.arange(N)
    r_cont = np.linspace(0, N - 1, 300)

    fig, axes = plt.subplots(1, len(ks), figsize=(3.5 * len(ks), 3.2), sharey=False)
    if len(ks) == 1:
        axes = [axes]

    k_color = {k: c for k, c in zip(ks, [ps.NAVY, ps.TEAL, ps.VIOLET, ps.MAUVE])}

    for ax, k in zip(axes, ks):
        rec = by_k[k]
        v   = np.array(rec["v"])
        dc  = rec["dc"]
        h   = rec["harmonics"][k]
        amp = h["amplitude"]
        phi = np.radians(h["phase_deg"])

        # continuous theoretical curve: DC + amplitude * cos(2π k r / N + φ)
        theory = dc + amp * np.cos(2 * np.pi * k * r_cont / N + phi)

        color = k_color[k]
        ax.bar(xs, v, color=color, alpha=0.55, width=0.6, label="observed", zorder=2)
        ax.plot(r_cont, theory, color=color, lw=1.8, label=f"k={k} theory", zorder=3)
        ax.axhline(dc, color=ps.GRAY, lw=0.9, ls="--", alpha=0.7, label="DC")

        side  = rec.get("side", "")
        score = rec.get("score") or 0.0
        r2    = rec["dominant_r2"]
        ax.set_title(
            f"{rec['feature']}  ({side}, score {score:+.3f})\n"
            f"dom. k={k}  R²={r2:.3f}",
            fontsize=8,
        )
        ax.set_xlabel("residue class $r$", fontsize=8)
        ax.set_ylabel("mean activation", fontsize=8)
        ax.set_xticks(xs)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, loc="upper right")

    fig.suptitle(f"Fourier families (mod {N}): representative features", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--anchor_dir", type=Path, required=True)
    ap.add_argument(
        "--score_mode", default="enc+dec",
        help="Score mode used in edec_features.json subdirectory name",
    )
    ap.add_argument(
        "--edec_json", type=Path, default=None,
        help="Override path to edec_features.json; ignored when --top_k is set",
    )
    ap.add_argument(
        "--top_k", type=int, default=None,
        help="If set, re-derive the top top_k features from deltas.pt (top_k/2 pos + "
             "top_k/2 neg) without loading the full model; ignores edec_features.json",
    )
    ap.add_argument("--out_json", type=Path, default=None)
    ap.add_argument("--plot", action="store_true",
                    help="Save a per-family representative plot alongside the JSON")
    ap.add_argument(
        "--balanced_npz", type=Path, default=None,
        help="Path to a pre-computed balanced mean-acts npz (from balanced_residue_sweep.py). "
             "Keys mean_L{i} of shape (modulus, d_tc). No model load required.",
    )
    ap.add_argument(
        "--balanced_n", type=int, default=None,
        metavar="N",
        help="If set, generate N fresh single prompts per residue class (balanced), "
             "run the full model, and use those activations instead of sweep_residuals.npz. "
             "Requires loading AttributionModel; submit via sbatch for large N.",
    )
    ap.add_argument("--model", default=None, help="HF model id (default: from config.yaml)")
    ap.add_argument("--transcoder_set", default=None, help="Transcoder id (default: from config.yaml)")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default=None, help="Device (default: auto-detect)")
    args = ap.parse_args()

    suffix         = args.score_mode.replace("+", "_")
    sweep_dir      = args.anchor_dir / "sweep"
    proj_dir       = sweep_dir / f"delta_feature_projections_{suffix}"
    examples_path  = sweep_dir / "sweep_dataset_examples.pkl"
    residuals_path = sweep_dir / "sweep_residuals.npz"

    # resolve device
    device = args.device
    if device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── feature selection ─────────────────────────────────────────────────────
    if args.top_k is not None:
        n_each = args.top_k // 2
        print(f"Inline ranking: top {n_each} pos + {n_each} neg features (no model load)")
        deltas_path = args.anchor_dir / "deltas.pt"
        if not deltas_path.exists():
            raise FileNotFoundError(deltas_path)
        pos_rows, neg_rows = _rank_features(
            deltas_path, residuals_path, n_each, device
        )
        all_features = pos_rows + neg_rows
        out_suffix   = f"top{args.top_k}"
    else:
        edec_json_path = args.edec_json or (proj_dir / "edec_features.json")
        print(f"Loading {edec_json_path}")
        with edec_json_path.open() as f:
            edec = json.load(f)
        pos_rows = [{**r, "side": "pos"} for r in edec.get("pos", [])]
        neg_rows = [{**r, "side": "neg"} for r in edec.get("neg", [])]
        all_features = pos_rows + neg_rows
        out_suffix   = "json"

    print(f"  {len(pos_rows)} pos + {len(neg_rows)} neg = {len(all_features)} features")

    # modulus from sweep examples (always needed for DFT)
    with examples_path.open("rb") as f:
        examples: list[dict] = pickle.load(f)
    moduli = {int(ex["meta"]["m"]) for ex in examples if "m" in ex["meta"]}
    moduli |= {int(ex["meta"]["g"]) for ex in examples if "g" in ex["meta"]}
    if len(moduli) != 1:
        raise ValueError(f"Inconsistent moduli: {moduli}")
    modulus = moduli.pop()

    # anchor position (for balanced path)
    anchor_cfg = json.loads((args.anchor_dir / "results.json").read_text())["config"]
    anchor_pos = int(anchor_cfg.get("anchor_pos", anchor_cfg.get("anchor_mode", 3)))

    # group features by layer
    by_layer: dict[int, list[dict]] = {}
    for feat in all_features:
        by_layer.setdefault(feat["layer"], []).append(feat)
    layers = sorted(by_layer)

    # ── activation computation ────────────────────────────────────────────────
    if args.balanced_npz is not None:
        # fastest balanced path: read pre-computed per-class means, no model load
        print(f"Loading pre-computed balanced activations from {args.balanced_npz}")
        bal = np.load(str(args.balanced_npz), allow_pickle=True)
        modulus = int(bal["modulus"])
        n_per_class = int(bal["n_per_class"])
        print(f"  modulus={modulus}, n_per_class={n_per_class}")
        out_suffix += f"_bal{n_per_class}"

        results = []
        for layer in layers:
            key = f"mean_L{layer}"
            if key not in bal:
                print(f"  WARN: {key} missing; skipping")
                continue
            mean_acts = bal[key]  # (modulus, d_tc)
            for feat in by_layer[layer]:
                fid = feat["feature_id"]
                if fid >= mean_acts.shape[1]:
                    print(f"  WARN: fid={fid} out of range for layer {layer}; skipping")
                    continue
                v = mean_acts[:, fid].astype(np.float32)
                analysis = _dft_analysis(v)
                results.append({
                    "feature":     feat["feature"],
                    "layer":       layer,
                    "feature_id":  fid,
                    "side":        feat.get("side", "?"),
                    "dec_cos":     feat.get("dec_cos"),
                    "enc_cos":     feat.get("enc_cos"),
                    "score":       feat.get("score"),
                    "frac_active": round(float(np.mean(v > 0)), 4),
                    "mean_act":    round(float(v.mean()), 5),
                    "modulus":     modulus,
                    "n_per_class": n_per_class,
                    "counts":      [n_per_class] * modulus,
                    **analysis,
                })

    elif args.balanced_n is not None:
        # balanced path: load model, generate N prompts per class, run inference
        print(f"\nBalanced mode: {args.balanced_n} prompts per residue class "
              f"({args.balanced_n * modulus} total), loading model...")
        from scripts.model_config import (
            add_model_config_arg, default_model, default_transcoder_set, resolve_model_args,
        )
        from mechinterp_qwen3.attribution_model import AttributionModel
        from mechinterp_qwen3.utils.hf_utils import load_transcoder_from_hub
        from mechinterp_qwen3.utils.model_utils import parse_dtype

        dtype = parse_dtype(args.dtype)
        model_id = args.model or default_model()
        tc_id    = args.transcoder_set or default_transcoder_set()
        tc_set, _ = load_transcoder_from_hub(tc_id, dtype=dtype, lazy_encoder=True, lazy_decoder=True)
        amodel = AttributionModel.from_pretrained_and_transcoders(
            model_id, tc_set, dtype=dtype, device=device
        )
        amodel.eval()

        print(f"  Generating balanced inputs (anchor={anchor_pos})...")
        bal_inputs, bal_labels = _generate_balanced_inputs(
            args.balanced_n, modulus, anchor_pos, amodel.tokenizer
        )
        print(f"  {len(bal_inputs)} prompts across {modulus} classes")

        features_by_layer = {layer: [f["feature_id"] for f in feats]
                             for layer, feats in by_layer.items()}
        print(f"  Running inference over {len(layers)} layers...")
        bal_acts = _collect_balanced_acts(amodel, bal_inputs, layers, features_by_layer)

        out_suffix += f"_bal{args.balanced_n}"
        print(f"  modulus={modulus}, n_per_class={args.balanced_n}")

        results = []
        for layer in layers:
            for feat in by_layer[layer]:
                fid  = feat["feature_id"]
                key  = f"L{layer}_F{fid}"
                if key not in bal_acts:
                    print(f"  WARN: {key} not in balanced acts; skipping")
                    continue
                feat_acts = bal_acts[key]
                v, counts = _residue_response_balanced(feat_acts, bal_labels, modulus)
                analysis  = _dft_analysis(v)
                results.append({
                    "feature":    key,
                    "layer":      layer,
                    "feature_id": fid,
                    "side":       feat.get("side", "?"),
                    "dec_cos":    feat.get("dec_cos"),
                    "enc_cos":    feat.get("enc_cos"),
                    "score":      feat.get("score"),
                    "frac_active": round(float(np.mean(feat_acts > 0)), 4),
                    "mean_act":   round(float(feat_acts.mean()), 5),
                    "modulus":    modulus,
                    "counts":     counts.tolist(),
                    "balanced_n": args.balanced_n,
                    **analysis,
                })

    else:
        # default path: reuse cached sweep residuals, apply transcoder via safetensors
        print(f"  modulus={modulus}, n_examples={len(examples)} (imbalanced)")
        npz = np.load(str(residuals_path), allow_pickle=True)

        results = []
        for layer in layers:
            feats = by_layer[layer]
            key   = f"H_L{layer}"
            if key not in npz:
                print(f"  WARN: {key} missing; skipping")
                continue
            H = npz[key].astype(np.float32)
            print(f"  L{layer}: {H.shape[0]} samples, {len(feats)} features")
            W_enc, b_enc = _load_enc(layer, device)
            acts = _encode(H, W_enc, b_enc)
            del W_enc, b_enc

            for feat in feats:
                fid       = feat["feature_id"]
                feat_acts = acts[:, fid]
                v, counts = _residue_response(feat_acts, examples, modulus)
                analysis  = _dft_analysis(v)
                results.append({
                    "feature":    feat["feature"],
                    "layer":      layer,
                    "feature_id": fid,
                    "side":       feat.get("side", "?"),
                    "dec_cos":    feat.get("dec_cos"),
                    "enc_cos":    feat.get("enc_cos"),
                    "score":      feat.get("score"),
                    "frac_active": round(float(np.mean(feat_acts > 0)), 4),
                    "mean_act":   round(float(feat_acts.mean()), 5),
                    "modulus":    modulus,
                    "counts":     counts.tolist(),
                    **analysis,
                })

    results.sort(key=lambda r: abs(r.get("score") or 0), reverse=True)

    # ── print table ───────────────────────────────────────────────────────────
    max_k = modulus // 2
    k_header = "  ".join(f"k={k}%" for k in range(1, max_k + 1))
    print()
    print(f"{'Feature':<16} {'side':>4} {'score':>7} {'dom_k':>5} "
          f"{'phase':>8} {'R²':>6}   {k_header}")
    print("-" * (60 + 8 * max_k))
    for r in results:
        h   = r["harmonics"]
        k_pcts = "  ".join(
            f"{100*h.get(k,{}).get('power_frac_ac',0):>5.1f}" for k in range(1, max_k + 1)
        )
        phase = h.get(r["dominant_k"], {}).get("phase_deg", 0.0)
        print(
            f"{r['feature']:<16} {r['side']:>4} {r['score']:>+7.4f} "
            f"{r['dominant_k']:>5}  {phase:>+7.1f}°  {r['dominant_r2']:>5.3f}   {k_pcts}"
        )

    # ── harmonic agreement summary ────────────────────────────────────────────
    print()
    k_counts = Counter(r["dominant_k"] for r in results)
    print("Dominant harmonic distribution:")
    for k, cnt in sorted(k_counts.items()):
        phase_for_k = [r["harmonics"][k]["phase_deg"] for r in results if r["dominant_k"] == k]
        print(f"  k={k}: {cnt:>2} features   phase [{min(phase_for_k):+.1f}°, {max(phase_for_k):+.1f}°]"
              f"  range={max(phase_for_k)-min(phase_for_k):.1f}°")

    print()
    print("Per-side breakdown:")
    for side in ("pos", "neg"):
        sub = [r for r in results if r["side"] == side]
        if not sub:
            continue
        sub_k = Counter(r["dominant_k"] for r in sub)
        print(f"  {side}: " + "  ".join(f"k={k}×{n}" for k, n in sorted(sub_k.items())))

    # ── save ──────────────────────────────────────────────────────────────────
    out_path = args.out_json or (proj_dir / f"fourier_sweep_{out_suffix}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "config": {"modulus": modulus, "n_features": len(results), "top_k": args.top_k},
        "features": results,
    }
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved → {out_path}")

    if args.plot:
        plot_path = out_path.with_suffix(".pdf")
        _plot_families(results, plot_path)


if __name__ == "__main__":
    main()
