"""
Inter-layer delta geometry check: does the SAE projection profile explain C?

For a saved anchor directory (containing deltas.pt), computes:

  C[l,m]  = cos(δ_l, δ_m)                        -- true concept-direction similarity
  D[l,m]  = cos(p_l^dec, p_m^dec)                 -- decoder-profile similarity
  E[l,m]  = cos(p_l^enc, p_m^enc)                 -- encoder-profile similarity

where p_l^dec[f] = cos(δ_l, W_dec[f]) and p_l^enc[f] = cos(δ_l, W_enc[f]).

Reports:
  corr(upper(C), upper(D))  and  corr(upper(C), upper(E))

Residuals:
  R_dec = C - D
  R_enc = C - E

Saves a PDF with five heatmaps and a .pt with all matrices + correlation scalars.

Usage (CPU; fast enough without GPU):
    python -m experiments.concept_localization.profile_geometry_check \\
        --anchor_dir runs/concept_localization/carry/carry_T0/anchor_rank1_pos5

Or via sbatch for GPU access (not required):
    sbatch scripts/sbatch_run.sh python -m experiments.concept_localization.profile_geometry_check \\
        --anchor_dir runs/concept_localization/carry/carry_T0/anchor_rank1_pos5
"""

import argparse
import logging
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif"})
import torch
import torch.nn.functional as F

from mechinterp_qwen3.utils.hf_utils import load_transcoder_from_hub
from mechinterp_qwen3.utils.model_utils import parse_dtype

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s")

_TRANSCODER_SET = "mwhanna/qwen3-4b-transcoders"


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_deltas(anchor_dir: pathlib.Path) -> dict[int, torch.Tensor]:
    """Load layer→delta dict from deltas.pt. Uses the 'all' key (all templates pooled)."""
    d = torch.load(anchor_dir / "deltas.pt", map_location="cpu", weights_only=False)
    obj = d.get("all") or next(iter(d.values()))
    if hasattr(obj, "delta"):
        return obj.delta  # LayerDeltas dataclass
    return obj  # plain dict[int, tensor]


def _cosim_matrix(vecs: list[torch.Tensor]) -> torch.Tensor:
    """Return L×L cosine similarity matrix from a list of L vectors."""
    M = torch.stack([v.float() for v in vecs], dim=0)
    M = F.normalize(M, dim=1)
    return M @ M.T


def _profile_matrix(
    deltas: dict[int, torch.Tensor],
    transcoder_set,
    mode: str,
) -> torch.Tensor:
    """
    Build L×L profile-similarity matrix.

    mode='dec': p_l[f] = cos(δ_l, W_dec[f])
    mode='enc': p_l[f] = cos(δ_l, W_enc[f])
    """
    layers = sorted(deltas.keys())
    profiles: list[torch.Tensor] = []
    for l in layers:
        tc = transcoder_set[l]
        delta = deltas[l].float()
        delta_norm = delta.norm().clamp(min=1e-8)

        if mode == "dec":
            W = tc.W_dec.detach().float()  # (F, d_model)
        else:
            W = tc.W_enc.detach().float()  # (F, d_model)

        cos_f = (W @ delta) / (W.norm(dim=1).clamp(min=1e-8) * delta_norm)  # (F,)
        profiles.append(cos_f)

    return _cosim_matrix(profiles)


def _upper_triu(M: torch.Tensor) -> torch.Tensor:
    """Flatten strictly upper-triangular entries (offset=1, excluding diagonal)."""
    idx = torch.triu_indices(M.shape[0], M.shape[1], offset=1)
    return M[idx[0], idx[1]]


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float()
    b = b.float()
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if denom < 1e-10:
        return float("nan")
    return float((a * b).sum() / denom)


# ── plotting ──────────────────────────────────────────────────────────────────

def _plot_five(C, D, E, R_dec, R_enc, corr_D: float, corr_E: float, out_path: pathlib.Path):
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.8))

    panels = [
        (C,     r"$C$ (true $\delta$ cosine)",       -1, 1,    "RdBu_r"),
        (D,     r"$D$ (decoder profile)",             -1, 1,    "RdBu_r"),
        (E,     r"$E$ (encoder profile)",             -1, 1,    "RdBu_r"),
        (R_dec, r"$R^{\mathrm{dec}} = C - D$",       -0.5, 0.5, "RdBu_r"),
        (R_enc, r"$R^{\mathrm{enc}} = C - E$",       -0.5, 0.5, "RdBu_r"),
    ]

    for ax, (M, title, vmin, vmax, cmap) in zip(axes, panels):
        im = ax.imshow(
            M.detach().float().cpu().numpy(),
            origin="lower", aspect="equal",
            cmap=cmap, vmin=vmin, vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("layer $m$", fontsize=8)
        ax.set_ylabel("layer $l$", fontsize=8)
        ax.tick_params(labelsize=7)
        L = M.shape[0]
        ax.set_xticks(range(0, L, 5))
        ax.set_yticks(range(0, L, 5))
        for sp in ax.spines.values():
            sp.set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=7)
        cb.outline.set_visible(False)

    fig.suptitle(
        rf"corr$(C,D)={corr_D:.3f}$ (decoder)   corr$(C,E)={corr_E:.3f}$ (encoder)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    log.info("Saved figure → %s", out_path)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--anchor_dir", required=True,
        help="Path to anchor directory containing deltas.pt",
    )
    parser.add_argument("--transcoder_set", default=_TRANSCODER_SET)
    parser.add_argument("--dtype", default="float32",
                        help="Dtype for transcoder weights (float32 recommended for profiles)")
    parser.add_argument(
        "--out_dir", default=None,
        help="Output directory (default: <anchor_dir>/profile_geometry)",
    )
    args = parser.parse_args()

    anchor_dir = pathlib.Path(args.anchor_dir)
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else anchor_dir / "profile_geometry"
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype = parse_dtype(args.dtype)

    # ── load transcoder weights (no LM needed) ────────────────────────────────
    log.info("Loading transcoders from %s", args.transcoder_set)
    transcoder_set, _ = load_transcoder_from_hub(
        args.transcoder_set, dtype=dtype, lazy_encoder=False, lazy_decoder=False
    )
    n_layers = transcoder_set.n_layers
    log.info("Transcoders loaded: %d layers, d_tc=%d", n_layers, transcoder_set.d_transcoder)

    # ── load deltas ───────────────────────────────────────────────────────────
    log.info("Loading deltas from %s", anchor_dir / "deltas.pt")
    deltas = _load_deltas(anchor_dir)
    layers = sorted(deltas.keys())
    log.info("Deltas loaded: %d layers, delta shape %s", len(layers), deltas[layers[0]].shape)

    # ── compute matrices ──────────────────────────────────────────────────────
    log.info("Computing C (true delta cosine matrix)...")
    C = _cosim_matrix([deltas[l] for l in layers])

    log.info("Computing D (decoder profile similarity matrix)...")
    D = _profile_matrix(deltas, transcoder_set, mode="dec")

    log.info("Computing E (encoder profile similarity matrix)...")
    E = _profile_matrix(deltas, transcoder_set, mode="enc")

    R_dec = C - D
    R_enc = C - E

    # ── correlations (upper triangular, off-diagonal) ─────────────────────────
    c_upper = _upper_triu(C)
    d_upper = _upper_triu(D)
    e_upper = _upper_triu(E)

    corr_D = _pearson(c_upper, d_upper)
    corr_E = _pearson(c_upper, e_upper)

    log.info("corr(upper(C), upper(D)) = %.4f  [decoder profiles]", corr_D)
    log.info("corr(upper(C), upper(E)) = %.4f  [encoder profiles]", corr_E)

    # ── save matrices ─────────────────────────────────────────────────────────
    pt_path = out_dir / "profile_geometry.pt"
    torch.save({
        "C": C, "D": D, "E": E, "R_dec": R_dec, "R_enc": R_enc,
        "corr_D": corr_D, "corr_E": corr_E,
        "layers": layers,
        "anchor_dir": str(anchor_dir),
    }, pt_path)
    log.info("Saved matrices → %s", pt_path)

    # ── plot ──────────────────────────────────────────────────────────────────
    pdf_path = out_dir / "profile_geometry.pdf"
    _plot_five(C, D, E, R_dec, R_enc, corr_D, corr_E, pdf_path)

    print(f"\ncorr(C, D) = {corr_D:.4f}  [decoder profiles explain inter-layer delta geometry]")
    print(f"corr(C, E) = {corr_E:.4f}  [encoder profiles explain inter-layer delta geometry]")
    print(f"Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
