"""Check which transcoder features separate N mod M == 0 vs != 0, for M in [2, 100],
at the "ones_a" anchor (last digit of N) for gcd and residue_class.

Default mode reuses the ~20 features already identified by delta_feature_pipeline
(edec_features.json, dec + enc+dec) — cheap, reads only those feature rows.

--all_features scans every one of the 163840 transcoder features per layer (not just
the previously-known ones): encodes the cached residual stream through the layer's
full W_enc, computes Cohen's d between the N%M==0 / N%M!=0 groups for every feature,
and keeps the top-K per M. Heavier (36 layers x 163840 features x 99 M values); run via
sbatch.

Usage:
    python -m experiments.concept_localization.mod_m_feature_scan
    python -m experiments.concept_localization.mod_m_feature_scan --all_features --top_k 10
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from safetensors import safe_open

_TC_SNAPSHOT = (
    "/rds/user/eid23/hpc-work/p28/cache/hf/hub"
    "/models--mwhanna--qwen3-4b-transcoders"
    "/snapshots/94d176260ac39ce2f882b8b09aba8c118df29bb3"
)

M_RANGE = range(2, 101)
TOP_N_PER_FEATURE = 5

CONFIGS = [
    ("gcd", Path("runs/concept_localization/gcd/gcd_T0/anchor_rank3_pos6")),
    ("residue_class", Path("runs/concept_localization/residue_class/residue_class_T0/anchor_rank3_pos5")),
    ("prime", Path("runs/concept_localization/prime/prime_T0/anchor_rank3_pos4")),
    ("number_grid", Path("runs/concept_localization/mod_m_scan/number_cache/number_residuals.npz")),
]

# meta dict key names vary by concept dataset (a_pos/a_neg vs n_pos/n_neg)
_N_META_KEYS = [("a_pos", "a_neg"), ("n_pos", "n_neg")]


def _load_feature_rows(layer: int, fids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    with safe_open(f"{_TC_SNAPSHOT}/layer_{layer}.safetensors", framework="pt") as f:
        W = f.get_slice("W_enc")[fids, :].float().numpy()
        b = f.get_slice("b_enc")[fids].float().numpy()
    return W, b


def _load_enc_full(layer: int) -> tuple[np.ndarray, np.ndarray]:
    with safe_open(f"{_TC_SNAPSHOT}/layer_{layer}.safetensors", framework="pt") as f:
        W = f.get_tensor("W_enc").float().numpy()
        b = f.get_tensor("b_enc").float().numpy()
    return W, b


def _build_n(anchor_dir: Path) -> tuple[np.ndarray, np.lib.npyio.NpzFile, Path]:
    # Raw number-grid cache (build_number_residual_cache.py): a flat npz with "n" and
    # H_L{layer} for every layer, no concept-pair structure — just use it directly.
    if anchor_dir.is_file() and anchor_dir.suffix == ".npz":
        npz = np.load(anchor_dir)
        return npz["n"].astype(np.int64), npz, anchor_dir.parent

    sweep_dir = anchor_dir / "sweep"
    examples = pickle.loads((sweep_dir / "sweep_dataset_examples.pkl").read_bytes())
    npz = np.load(sweep_dir / "sweep_residuals.npz")

    key_pos, key_neg = next(
        (kp, kn) for kp, kn in _N_META_KEYS if kp in examples[0]["meta"]
    )
    n = np.empty(2 * len(examples), dtype=np.int64)
    for i, ex in enumerate(examples):
        n[2 * i] = ex["meta"][key_pos]
        n[2 * i + 1] = ex["meta"][key_neg]
    return n, npz, sweep_dir


def scan_all_features(anchor_dir: Path, concept: str, top_k: int = 5) -> None:
    """Exhaustive scan: every transcoder feature at every cached layer, vs every M."""
    n, npz, _ = _build_n(anchor_dir)
    layers = sorted(int(k[3:]) for k in npz.files if k.startswith("H_L"))

    # best[m] = list of (abs_d, d, layer, fid, mu0, mu1), kept trimmed to top_k after each layer
    best: dict[int, list[tuple[float, float, int, int, float, float]]] = {m: [] for m in M_RANGE}

    for layer in layers:
        H = npz[f"H_L{layer}"].astype(np.float32)
        W, b = _load_enc_full(layer)
        acts = np.maximum(H @ W.T + b, 0.0)  # (n_rows, 163840)
        del W, b

        for m in M_RANGE:
            mask = (n % m == 0)
            if mask.sum() < 5 or (~mask).sum() < 5:
                continue
            mu0 = acts[mask].mean(axis=0)
            mu1 = acts[~mask].mean(axis=0)
            pooled = np.sqrt(0.5 * (acts[mask].var(axis=0) + acts[~mask].var(axis=0))) + 1e-8
            d = (mu0 - mu1) / pooled
            abs_d = np.abs(d)
            k = min(top_k, abs_d.size)
            idx = np.argpartition(-abs_d, k - 1)[:k]
            for fid in idx:
                fid = int(fid)
                best[m].append((float(abs_d[fid]), float(d[fid]), layer, fid, float(mu0[fid]), float(mu1[fid])))
            best[m] = sorted(best[m], key=lambda r: -r[0])[:top_k]
        del acts
        print(f"  layer {layer} done ({len(layers)} total)")

    print(f"\n=== {concept}  anchor={anchor_dir.name} (ones_a: last digit of N)  "
          f"ALL 163840 features x {len(layers)} layers, M in [{M_RANGE.start},{M_RANGE.stop - 1}] ===")
    for m in M_RANGE:
        rows = best[m]
        if not rows:
            continue
        summary = "  ".join(
            f"L{layer}_F{fid}:d={d:+.2f}(mu0={mu0:.2f},mu1={mu1:.2f})"
            for _, d, layer, fid, mu0, mu1 in rows
        )
        print(f"M={m:<3d} {summary}")


# Recurring candidates from the --all_features scan: features that kept showing up
# across several multiples of the same modulus (see mod_m_scan_all_31227108 log).
# M=7 included as the established baseline for comparison.
CANDIDATE_SETS: dict[int, list[str]] = {
    3: ["L10_F7949", "L22_F127578", "L19_F76703"],
    7: ["L22_F24461", "L22_F88709", "L10_F135288", "L23_F136813", "L26_F58277"],
    11: ["L10_F118303", "L11_F163754", "L13_F53611", "L12_F72744", "L21_F84263"],
    13: ["L10_F90921", "L11_F125018", "L9_F71524"],
}

PLOT_OUT_DIR = Path("runs/concept_localization/mod_m_scan")


def _activations_for_features(anchor_dir: Path, feature_names: list[str]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    by_layer: dict[int, list[int]] = defaultdict(list)
    for f in feature_names:
        layer_s, fid_s = f.split("_")
        by_layer[int(layer_s[1:])].append(int(fid_s[1:]))

    n, npz, _ = _build_n(anchor_dir)
    acts: dict[str, np.ndarray] = {}
    for layer, fids in by_layer.items():
        H = npz[f"H_L{layer}"].astype(np.float32)
        W, b = _load_feature_rows(layer, fids)
        a = np.maximum(H @ W.T + b, 0.0)
        for j, fid in enumerate(fids):
            acts[f"L{layer}_F{fid}"] = a[:, j]
    return acts, n


def plot_residue_profile(concept: str, anchor_dir: Path, modulus: int, feature_names: list[str]) -> None:
    """Bar plot of mean activation vs residue class r=0..modulus-1, over all cached prompts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import experiments.plot_style as ps
    ps.apply()

    acts, n = _activations_for_features(anchor_dir, feature_names)
    r = n % modulus

    fig, axes = plt.subplots(1, len(feature_names), figsize=(3.2 * len(feature_names), 3.0), sharex=True)
    if len(feature_names) == 1:
        axes = [axes]

    for ax, name in zip(axes, feature_names):
        a = acts[name]
        v = np.zeros(modulus, dtype=np.float64)
        counts = np.zeros(modulus, dtype=np.int64)
        for i in range(modulus):
            mask = r == i
            counts[i] = mask.sum()
            v[i] = a[mask].mean() if counts[i] > 0 else 0.0
        ax.bar(np.arange(modulus), v, color=ps.NAVY, alpha=0.75, width=0.7)
        ax.set_title(f"{name}\n(n per bin: {counts.min()}-{counts.max()})", fontsize=8)
        ax.set_xlabel(f"N mod {modulus}", fontsize=8)
        ax.set_ylabel("mean activation", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_xticks(np.arange(modulus))

    fig.suptitle(f"{concept} — {anchor_dir.name} (ones_a) — mean activation by N mod {modulus}", fontsize=9)
    fig.tight_layout()
    PLOT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOT_OUT_DIR / f"{concept}_M{modulus}_residue_profile.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path}")


def _collect_features(anchor_dir: Path) -> list[str]:
    feats: set[str] = set()
    for mode in ("enc_dec", "dec"):
        p = anchor_dir / "sweep" / f"delta_feature_projections_{mode}" / "edec_features.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        for side in ("pos", "neg"):
            for row in d.get(side, []):
                feats.add(row["feature"])
    return sorted(feats, key=lambda s: (int(s.split("_")[0][1:]), int(s.split("_")[1][1:])))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all_features", action="store_true",
                    help="Scan all 163840 features per layer instead of just the ~20 already-known ones")
    ap.add_argument("--top_k", type=int, default=5, help="Top-K features to report per M (--all_features mode)")
    ap.add_argument("--plot", action="store_true",
                    help="Plot mean-activation-by-residue bar charts for CANDIDATE_SETS, both concepts")
    ap.add_argument("--concepts", nargs="+", default=None,
                    help="Restrict to these concept names (default: all of CONFIGS)")
    args = ap.parse_args()

    configs = CONFIGS if not args.concepts else [c for c in CONFIGS if c[0] in args.concepts]

    if args.plot:
        for concept, anchor_dir in configs:
            for modulus, feature_names in CANDIDATE_SETS.items():
                plot_residue_profile(concept, anchor_dir, modulus, feature_names)
        return

    if args.all_features:
        for concept, anchor_dir in configs:
            scan_all_features(anchor_dir, concept, top_k=args.top_k)
        return

    for concept, anchor_dir in configs:
        feats = _collect_features(anchor_dir)
        by_layer: dict[int, list[int]] = defaultdict(list)
        for f in feats:
            layer_s, fid_s = f.split("_")
            by_layer[int(layer_s[1:])].append(int(fid_s[1:]))

        n, npz, sweep_dir = _build_n(anchor_dir)

        acts: dict[str, np.ndarray] = {}
        for layer, fids in by_layer.items():
            H = npz[f"H_L{layer}"].astype(np.float32)
            W, b = _load_feature_rows(layer, fids)
            a = np.maximum(H @ W.T + b, 0.0)
            for j, fid in enumerate(fids):
                acts[f"L{layer}_F{fid}"] = a[:, j]

        print(f"\n=== {concept}  anchor={anchor_dir.name} (ones_a: last digit of N)  "
              f"{len(feats)} features, N in [100,999], scanning M in [{M_RANGE.start},{M_RANGE.stop - 1}] ===")

        for name in feats:
            a = acts[name]
            rows = []
            for m in M_RANGE:
                mask = (n % m == 0)
                if mask.sum() < 5 or (~mask).sum() < 5:
                    continue
                mu0, mu1 = a[mask].mean(), a[~mask].mean()
                pooled = np.sqrt(0.5 * (a[mask].var() + a[~mask].var())) + 1e-8
                d = (mu0 - mu1) / pooled
                rows.append((m, d, mu0, mu1))
            rows.sort(key=lambda r: -abs(r[1]))
            top = rows[:TOP_N_PER_FEATURE]
            summary = "  ".join(f"M={m}:d={d:+.2f}(mu0={mu0:.2f},mu1={mu1:.2f})" for m, d, mu0, mu1 in top)
            print(f"{name:<14} {summary}")


if __name__ == "__main__":
    main()
