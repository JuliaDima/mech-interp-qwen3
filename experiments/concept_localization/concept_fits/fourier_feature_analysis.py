"""
fourier_feature_analysis.py

Analyze a 10x10 digit-feature matrix with a compact 2D Fourier approximation.

Input:
  - CSV file containing a 10x10 matrix.
  - Rows should correspond to b=0..9, columns to a=0..9.
  - The CSV may have row/column labels; the loader tries to handle both.

Outputs:
  - Side-by-side plot: original matrix, top-K Fourier approximation, Fourier spectrum.
  - Optional CSV of the Fourier approximation.
  - Printed dominant modes and explicit approximate Fourier formula.

Example:
  python experiments/concept_localization/concept_fits/fourier_feature_analysis.py scripts/sweeps/reconstructed_10x10_matrix.csv --K 8 --out feature_fourier.png --approx-csv feature_fourier_approx.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
import experiments.plot_style as ps


def load_matrix_csv(path: str) -> np.ndarray:
    """Load a 10x10 CSV as internal grid X[a,b].

    The project CSV convention is rows b=0..9 and columns a=0..9, often
    with labels like an empty top-left cell, a=0..a=9 columns, and b=0..b=9
    rows.  Fourier code below uses axis 0 for a and axis 1 for b, matching
    the sweep/all_feature_grid convention, so CSV values are transposed here.
    """
    # Plain numeric CSV: rows are b and columns are a.
    try:
        raw = pd.read_csv(path, header=None).values.astype(float)
        if raw.shape == (10, 10):
            return np.nan_to_num(raw.T, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception:
        pass

    # Labeled CSV: first column is row labels b=..., columns are a=... .
    df = pd.read_csv(path, index_col=0)
    raw = df.values.astype(float)

    if raw.shape != (10, 10):
        raise ValueError(f"Expected a 10x10 matrix, got shape {raw.shape}")

    return np.nan_to_num(raw.T, nan=0.0, posinf=0.0, neginf=0.0)


def signed_freq(k: int, N: int) -> int:
    """Convert FFT index to signed frequency."""
    return k if k <= N // 2 else k - N


def fourier_decompose_matrix(
    X: np.ndarray,
    K: int = 8,
    subtract_mean: bool = True,
):
    """
    Decompose X[a,b] into top-K unique real cosine Fourier modes.

    We use:
      C = FFT2(X - mean) / N^2

    and reconstruct as:
      Xhat[a,b] = mean + sum_j A_j cos(2π(u_j a + v_j b)/N + phase_j)

    where:
      A_j = 2 |C[u_j, v_j]|
      phase_j = arg C[u_j, v_j]
    """
    X = np.asarray(X, dtype=float)
    N, M = X.shape

    if N != M:
        raise ValueError("Expected a square matrix.")

    mu = X.mean() if subtract_mean else 0.0
    Xc = X - mu

    C = np.fft.fft2(Xc) / (N * N)

    modes = []
    for u in range(N):      # a/x frequency, axis 0
        for v in range(N):  # b/y frequency, axis 1
            if u == 0 and v == 0:
                continue

            amp = 2 * np.abs(C[u, v])
            phase = np.angle(C[u, v])
            su = signed_freq(u, N)
            sv = signed_freq(v, N)

            modes.append(
                {
                    "amp": float(amp),
                    "u": int(su),
                    "v": int(sv),
                    "phase": float(phase),
                    "fft_u": int(u),
                    "fft_v": int(v),
                }
            )

    modes.sort(key=lambda m: m["amp"], reverse=True)

    # Keep unique conjugate-pair cosine modes.
    kept = []
    used = set()

    for m in modes:
        u_idx = m["fft_u"]
        v_idx = m["fft_v"]
        conj = ((-u_idx) % N, (-v_idx) % N)
        key = tuple(sorted([(u_idx, v_idx), conj]))

        if key in used:
            continue

        used.add(key)
        kept.append(m)

        if len(kept) >= K:
            break

    return mu, kept, C


def fourier_reconstruct(mu: float, modes: list[dict], N: int = 10) -> np.ndarray:
    """Reconstruct an internal grid Xhat[a,b] from cosine Fourier modes."""
    a_grid, b_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    Xhat = np.full((N, N), mu, dtype=float)

    for m in modes:
        Xhat += m["amp"] * np.cos(
            2 * np.pi * (m["u"] * a_grid + m["v"] * b_grid) / N
            + m["phase"]
        )

    return Xhat


def fourier_r2(X: np.ndarray, Xhat: np.ndarray) -> float:
    """R² of Fourier approximation vs original."""
    ss_res = float(np.sum((X - Xhat) ** 2))
    ss_tot = float(np.sum((X - X.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")


def find_min_k(
    X: np.ndarray,
    r2_target: float = 0.95,
    k_max: int = 8,
    subtract_mean: bool = True,
) -> tuple:
    """Find the smallest K in 1..k_max such that R² >= r2_target.

    Returns (k_used, r2, modes, mu, C, Xhat).
    If the target is never reached, returns the k_max result.
    """
    N = X.shape[0]
    result = None
    for k in range(1, k_max + 1):
        mu, modes, C = fourier_decompose_matrix(X, K=k, subtract_mean=subtract_mean)
        Xhat = fourier_reconstruct(mu, modes, N=N)
        r2 = fourier_r2(X, Xhat)
        result = (k, r2, modes, mu, C, Xhat)
        if not np.isnan(r2) and r2 >= r2_target:
            break
    return result


def classify_mode(u: int, v: int, N: int = 10) -> str:
    """Heuristic interpretation of a Fourier mode."""
    if u == 0 and v == 0:
        return "mean / DC"
    if u == 0:
        return "row-only / b-only"
    if v == 0:
        return "column-only / a-only"
    if u == v:
        return "iso-sum / a+b"
    if u == -v:
        return "iso-difference / b-a"
    if abs(u) == N // 2 and abs(v) == N // 2:
        return "parity / (-1)^(a+b)"
    return "mixed"


def formula_string(mu: float, modes: list[dict], N: int = 10, digits: int = 3,
                   amp_thresh: float = 0.05) -> str:
    """Return a readable Fourier approximation formula.

    Drops modes with amplitude < amp_thresh * max_amplitude and renders
    phase signs cleanly (no '+ -x' artifacts).
    """
    import math as _math

    def _fmt(x: float) -> str:
        s = f"{x:.{digits}f}".rstrip("0").rstrip(".")
        return s or "0"

    def _freq_arg(u: int, v: int) -> str:
        parts = []
        if u != 0:
            parts.append(f"{u}a" if u != 1 else "a")
        if v != 0:
            if parts:
                if v > 0:
                    parts.append(f"+ {v}b" if v != 1 else "+ b")
                else:
                    parts.append(f"- {abs(v)}b" if v != -1 else "- b")
            else:
                parts.append(f"{v}b" if v != 1 else "b")
        return " ".join(parts) or "0"

    max_amp = max((m["amp"] for m in modes), default=1.0)
    active = [m for m in modes if m["amp"] >= amp_thresh * max_amp]

    terms = [_fmt(float(mu))]
    for m in active:
        amp = _fmt(float(m["amp"]))
        phase = float(m["phase"])
        freq = _freq_arg(int(m["u"]), int(m["v"]))
        phase_abs = abs(phase)
        phase_s = _fmt(phase_abs)
        if phase_abs < 1e-3:
            phase_part = ""
        elif phase > 0:
            phase_part = f" + {phase_s}"
        else:
            phase_part = f" - {phase_s}"
        terms.append(f"+ {amp}·cos(2π({freq})/{N}{phase_part})")

    n = len(terms)
    per_line = max(1, _math.ceil(n / 2))
    lines = ["  ".join(terms[i : i + per_line]) for i in range(0, n, per_line)]
    return "f(a,b) ≈ " + "\n    ".join(lines)


def print_report(mu: float, modes: list[dict], N: int = 10, digits: int = 3) -> None:
    print("\nDominant Fourier modes:")
    print(f"mean = {mu:.{digits}f}\n")

    for m in modes:
        typ = classify_mode(m["u"], m["v"], N=N)
        print(
            f"{m['amp']:.{digits}f} * "
            f"cos(2π({m['u']}a + {m['v']}b)/{N} + {m['phase']:.{digits}f})"
            f"    [{typ}]"
        )

    print("\nExplicit approximate function:")
    print(formula_string(mu, modes, N=N, digits=digits))


def plot_fourier_analysis(
    X: np.ndarray,
    Xhat: np.ndarray,
    C: np.ndarray,
    modes: list[dict] | None = None,
    mu: float = 0.0,
    out_path: str | None = None,
    title: str = "Fourier feature analysis",
    cmap: str = "Reds",   # kept for API compat but ignored; uses project style
) -> None:
    ps.apply()
    N = X.shape[0]

    cmap_seq = LinearSegmentedColormap.from_list("white_violet", ["white", ps.VIOLET])
    cmap_seq.set_bad("white")

    Xhat_clipped = np.clip(Xhat, 0.0, 1.0)

    # Layout: 2 heatmaps + formula text box
    fig = plt.figure(figsize=(13, 4.8))
    gs = fig.add_gridspec(1, 3, left=0.06, right=0.97, top=0.84, bottom=0.12,
                          wspace=0.22)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax_txt = fig.add_subplot(gs[0, 2])

    def _draw_grid(ax, data, ylabel=False):
        ax.imshow(data.T, origin="lower", aspect="equal", cmap=cmap_seq, vmin=0, vmax=1)
        ax.set_xticks(range(N)); ax.set_yticks(range(N))
        ax.set_xticks(np.arange(-0.5, N, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, N, 1), minor=True)
        ax.tick_params(which="both", length=0)
        ax.grid(which="minor", color="#DDDDDD", linewidth=0.3)
        ax.grid(which="major", visible=False)
        ax.set_axisbelow(False)
        for spine in ax.spines.values():
            spine.set_color(ps.GRAY)
        ax.tick_params(axis="both", colors="black", length=0)
        ax.set_xlabel("a mod 10", labelpad=6)
        if ylabel:
            ax.set_ylabel("b mod 10", labelpad=6)

    _draw_grid(ax0, X, ylabel=True)
    ax0.set_title("Original matrix")

    _draw_grid(ax1, Xhat_clipped)
    ax1.set_title("Top-K Fourier approximation")

    ax_txt.axis("off")
    if modes is not None:
        formula = formula_string(mu, modes, N=N, digits=3)
    else:
        formula = "(no modes available)"
    ax_txt.text(0.5, 0.95, formula, transform=ax_txt.transAxes,
                ha="center", va="top", fontsize=7.5, fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8F8F8",
                          edgecolor=ps.GRAY, linewidth=0.6))
    ax_txt.set_title("Fourier formula")

    fig.suptitle(title, fontsize=12)

    if out_path:
        out_p = Path(out_path).with_suffix(".pdf")
        plt.savefig(out_p, bbox_inches="tight")
        print(f"Saved plot to: {out_p}")

    plt.close(fig)


def save_modes_csv(modes: list[dict], path: str, N: int = 10) -> None:
    rows = []
    for m in modes:
        rows.append({
            "amplitude": m["amp"],
            "u": m["u"],
            "v": m["v"],
            "phase": m["phase"],
            "type": classify_mode(m["u"], m["v"], N=N),
            "formula_term": (
                f"{m['amp']:.6f}*cos(2π({m['u']}a + {m['v']}b)/{N} + {m['phase']:.6f})"
            ),
        })
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Saved modes CSV to: {path}")


def _resolve_sweep_dir(path: Path, key: str | None = None,
                       anchor: str | None = None) -> Path | None:
    """Return the sweep directory containing sweep_activations.npz.

    If ``path`` directly contains the file, return it.  Otherwise search
    recursively. Pass ``anchor`` (e.g. ``'rank5_pos9'``) to filter candidates
    by directory name substring before selecting.
    """
    if (path / "sweep_activations.npz").exists():
        return path
    candidates = sorted(path.rglob("sweep_activations.npz"))
    if not candidates:
        return None
    if anchor:
        filtered = [c for c in candidates if anchor in str(c)]
        if filtered:
            candidates = filtered
    if key is None:
        return candidates[0].parent
    for c in candidates:
        if key in set(np.load(c).files):
            return c.parent
    return candidates[0].parent


def _build_grid_from_sweep(sweep_dir: Path, key: str,
                           anchor: str | None = None) -> np.ndarray | None:
    """Build a 10x10 mean-activation grid from sweep_activations.npz for a feature key."""
    import pickle as _pickle

    resolved = _resolve_sweep_dir(Path(sweep_dir), key, anchor=anchor)
    if resolved is None:
        print(f"  [skip] no sweep_activations.npz found under {sweep_dir}")
        return None
    sweep_dir = resolved
    try:
        npz = np.load(sweep_dir / "sweep_activations.npz")
        with open(sweep_dir / "sweep_examples.pkl", "rb") as f:
            examples = _pickle.load(f)
    except FileNotFoundError as e:
        print(f"  [skip] {e}")
        return None

    if key not in npz:
        print(f"  [skip] {key} not in sweep_activations.npz")
        return None

    acts = np.asarray(npz[key], dtype=np.float64)
    sums = np.zeros((10, 10)); counts = np.zeros((10, 10), dtype=int)
    for pair_i, ex in enumerate(examples):
        meta = ex["meta"] if hasattr(ex, "__getitem__") else ex.meta
        for use_pos in (True, False):
            act_i = 2 * pair_i if use_pos else 2 * pair_i + 1
            if act_i >= len(acts):
                continue
            suffix = "_pos" if use_pos else "_neg"
            a = meta.get("a" + suffix, meta.get("a_pos" if use_pos else "a_neg", None))
            b = meta.get("b" + suffix, meta.get("b_pos" if use_pos else "b_neg", None))
            if a is None or b is None:
                continue
            da, db = int(a) % 10, int(b) % 10
            sums[da, db] += float(acts[act_i]); counts[da, db] += 1

    grid = np.full((10, 10), np.nan)
    mask = counts > 0
    grid[mask] = sums[mask] / counts[mask]
    lo, hi = np.nanmin(grid), np.nanmax(grid)
    if hi - lo > 1e-12:
        grid = (grid - lo) / (hi - lo)
    return np.nan_to_num(grid, nan=0.0)


def _canonical_key(name: str) -> str:
    name = name.strip().upper()
    if "_F" in name and name.startswith("L"):
        layer_s, feat_s = name.split("_F", 1)
        return f"{layer_s}_F{int(feat_s)}"
    if "_" in name:
        layer_s, feat_s = name.split("_", 1)
        return f"L{int(layer_s[1:]) if layer_s.startswith('L') else int(layer_s)}_F{int(feat_s)}"
    raise ValueError(f"Cannot parse feature name: {name!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="?", default=None, help="Input 10x10 matrix CSV.")
    parser.add_argument("--features_json", default=None,
                        help="JSON list of feature keys; requires --sweep_dir")
    parser.add_argument("--sweep_dir", default=None,
                        help="Sweep directory (sweep_activations.npz + sweep_examples.pkl) for --features_json mode")
    parser.add_argument("--anchor", default=None,
                        help="Filter sweep candidates by directory name substring, e.g. rank5_pos9")
    parser.add_argument("--out_dir", default=None,
                        help="Output directory for --features_json batch mode")
    parser.add_argument("--K", type=int, default=8, help="Maximum number of cosine modes (actual K chosen to hit --fourier_r2_target).")
    parser.add_argument("--fourier_r2_target", type=float, default=0.95,
                        help="Minimum R² for Fourier approximation; K is increased until reached.")
    parser.add_argument("--out", default=None, help="Output plot path (single-CSV mode).")
    parser.add_argument("--approx-csv", default=None, help="Optional CSV path for reconstructed Fourier approximation.")
    parser.add_argument("--modes-csv", default=None, help="Optional CSV path for top Fourier modes.")
    parser.add_argument("--title", default=None, help="Plot title.")
    parser.add_argument("--cmap", default="Reds", help="Matplotlib colormap for matrix plots.")
    parser.add_argument("--no-mean-subtract", action="store_true", help="Do not subtract the mean before FFT.")
    args = parser.parse_args()

    subtract_mean = not args.no_mean_subtract

    # ── Batch mode: features_json + sweep_dir ────────────────────────────────
    if args.features_json:
        if not args.sweep_dir:
            parser.error("--features_json requires --sweep_dir")
        keys = [_canonical_key(k) for k in json.loads(Path(args.features_json).read_text())]
        seen = set(); keys = [k for k in keys if not (k in seen or seen.add(k))]
        out_dir = Path(args.out_dir) if args.out_dir else Path(args.features_json).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Fourier batch: {len(keys)} features → {out_dir}")
        for key in keys:
            X = _build_grid_from_sweep(args.sweep_dir, key, anchor=args.anchor)
            if X is None:
                continue
            k_used, r2_val, modes, mu, C, Xhat = find_min_k(
                X, r2_target=args.fourier_r2_target, k_max=args.K, subtract_mean=subtract_mean)
            print(f"  {key}: K={k_used}, R²={r2_val:.3f}")
            print_report(mu, modes)
            out_path = str(out_dir / f"fourier_{key}.pdf")
            plot_fourier_analysis(X, Xhat, C, modes=modes, mu=mu, out_path=out_path,
                                  title=f"Fourier analysis — {key}  (K={k_used}, R²={r2_val:.3f})")
        return

    # ── Single CSV mode ───────────────────────────────────────────────────────
    if not args.csv:
        parser.error("Provide a CSV file or use --features_json + --sweep_dir")

    X = load_matrix_csv(args.csv)
    k_used, r2_val, modes, mu, C, Xhat = find_min_k(
        X, r2_target=args.fourier_r2_target, k_max=args.K, subtract_mean=subtract_mean)
    print(f"K={k_used}, R²={r2_val:.3f}")
    print_report(mu, modes)

    if args.approx_csv:
        pd.DataFrame(
            Xhat.T,
            index=[f"b={i}" for i in range(N)],
            columns=[f"a={i}" for i in range(N)],
        ).to_csv(args.approx_csv)
        print(f"Saved Fourier approximation CSV to: {args.approx_csv}")

    if args.modes_csv:
        save_modes_csv(modes, args.modes_csv, N=N)

    out_path = args.out or (str(Path(args.csv).with_suffix("")) + "_fourier.pdf")
    title = args.title or f"Fourier feature analysis — {Path(args.csv).name}  (K={k_used}, R²={r2_val:.3f})"
    plot_fourier_analysis(X, Xhat, C, modes=modes, mu=mu, out_path=out_path, title=title)


if __name__ == "__main__":
    main()
