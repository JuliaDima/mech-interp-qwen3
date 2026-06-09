"""
plot_feature_vs_fourier.py

For each feature in features_list_to_plot.json, generate a two-panel PDF:
  left  — actual normalised activation grid (10×10)
  right — Fourier approximation (10×10)
plus a LaTeX-rendered formula showing the dominant modes.

The dominant arithmetic pattern is classified and labelled from the mode
structure (iso-sum, iso-difference, parity, carry, etc.).

Output: runs/concept_localization/carry/feature_vs_fourier/

Usage
-----
    python experiments/concept_localization/concept_fits/plot_feature_vs_fourier.py \
        --features_json runs/concept_localization/carry/features_list_to_plot.json \
        --sweep_dir     runs/concept_localization/carry/carry_T0 \
        --out_dir       runs/concept_localization/carry/feature_vs_fourier
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import experiments.plot_style as ps
from experiments.concept_localization.concept_fits.fourier_feature_analysis import (
    _build_grid_from_sweep,
    find_min_k,
    classify_mode,
)


# ── Pattern classification ────────────────────────────────────────────────────

_PARITY_FREQ = 5  # N//2 for N=10


def _is_parity_mode(u: int, v: int, N: int = 10) -> bool:
    half = N // 2
    return abs(u) in (0, half) and abs(v) in (0, half) and not (u == 0 and v == 0)


def _classify_pattern(modes: list[dict], N: int = 10) -> str:
    """Human-readable label for the dominant arithmetic pattern."""
    if not modes:
        return "unknown"
    top = modes[0]
    u, v = int(top["u"]), int(top["v"])
    base = classify_mode(u, v, N=N)

    if base == "parity / (-1)^(a+b)":
        return "parity"
    if base == "iso-sum / a+b":
        k = abs(u)
        return f"iso-sum  (harmonic {k})" if k > 1 else "iso-sum"
    if base == "iso-difference / b-a":
        k = abs(u)
        return f"iso-difference  (harmonic {k})" if k > 1 else "iso-difference"
    if base == "row-only / b-only":
        # Check if this is actually a parity-family feature: top modes are all at freq N//2
        max_amp = max(m["amp"] for m in modes)
        parity_frac = sum(
            m["amp"] for m in modes[:4]
            if _is_parity_mode(int(m["u"]), int(m["v"]), N)
        ) / (max_amp * min(4, len(modes)))
        if parity_frac > 0.5:
            return "parity"
        return "b-only"
    if base == "column-only / a-only":
        return "a-only"

    # Mixed: check if all dominant modes are iso-sum → likely carry-like
    iso_sum_count = sum(
        1 for m in modes[:4] if int(m["u"]) == int(m["v"])
    )
    if iso_sum_count >= 2:
        return "carry  (iso-sum harmonics)"

    return "mixed"


_PARITY_FORMULA_LATEX = (
    r"f(a,b) = (a + b) \bmod 2 \;=\; (-1)^{a+b}"
)


# ── LaTeX helpers (adapted from fit_pysr_sweep.py) ────────────────────────────

def _fmt_num(x: float, digits: int = 3) -> str:
    s = f"{x:.{digits}f}".rstrip("0").rstrip(".")
    return s or "0"


def _signed_coeff_var(coeff: int, var: str) -> str | None:
    if coeff == 0:
        return None
    if coeff == 1:
        return var
    if coeff == -1:
        return rf"-{var}"
    return rf"{coeff}{var}"


def _freq_numerator(u: int, v: int) -> str:
    t_u = _signed_coeff_var(u, "a")
    t_v = _signed_coeff_var(v, "b")
    if t_u is None and t_v is None:
        return "0"
    if t_u is None:
        return t_v
    if t_v is None:
        return t_u
    return f"{t_u} {t_v}" if t_v.startswith("-") else f"{t_u} + {t_v}"


def _mode_term_latex(m: dict, N: int = 10, digits: int = 3) -> str:
    amp = _fmt_num(float(m["amp"]), digits)
    phase = float(m["phase"])
    u, v = int(m["u"]), int(m["v"])
    num = _freq_numerator(u, v)
    freq = rf"\frac{{{num}}}{{{N}}}"
    phase_abs = abs(phase)
    phase_s = _fmt_num(phase_abs, digits)
    if phase_abs < 1e-3:
        arg = rf"2\pi {freq}"
    elif phase > 0:
        arg = rf"2\pi {freq} + {phase_s}"
    else:
        arg = rf"2\pi {freq} - {phase_s}"
    return rf"{amp}\cos\!\left({arg}\right)"


def _fourier_formula_latex(
    mu: float,
    modes: list[dict],
    N: int = 10,
    digits: int = 3,
    amp_thresh: float = 0.05,
    max_terms: int = 3,
) -> str:
    """Single-line LaTeX for the Fourier formula, at most max_terms cosine terms."""
    max_amp = max((m["amp"] for m in modes), default=1.0)
    active = [m for m in modes if m["amp"] >= amp_thresh * max_amp][:max_terms]

    terms = [_fmt_num(float(mu), digits)]
    for m in active:
        terms.append(r"+ " + _mode_term_latex(m, N, digits))

    return " ".join(terms)




# ── Panels figure (matplotlib) ────────────────────────────────────────────────

def _draw_panels(
    grid: np.ndarray,
    fourier_approx: np.ndarray,
    key: str,
    fourier_r2: float,
    fourier_K: int,
    panels_path: Path,
) -> None:
    """Two-panel heatmap figure saved as PDF (no formula text — lives in TeX)."""
    ps.apply()
    plt.rcParams.update({"mathtext.fontset": "stix", "font.family": "serif"})
    cmap = LinearSegmentedColormap.from_list("white_violet", ["white", ps.VIOLET])
    cmap.set_bad("white")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))

    def _draw(ax, data, title, ylabel=False):
        ax.imshow(data.T, origin="lower", aspect="equal", cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, pad=5)
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.set_xticks(np.arange(-0.5, 10, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 10, 1), minor=True)
        ax.tick_params(which="both", length=0)
        ax.grid(which="minor", color="#DDDDDD", linewidth=0.3)
        ax.grid(which="major", visible=False)
        ax.set_axisbelow(False)
        for spine in ax.spines.values():
            spine.set_color(ps.GRAY)
        ax.set_xlabel("a mod 10", labelpad=5)
        if ylabel:
            ax.set_ylabel("b mod 10", labelpad=5)

    m = re.fullmatch(r"L(\d+)_F(\d+)", key)
    key_tex = rf"$L^{{{m.group(1)}}}_{{{m.group(2)}}}$" if m else key

    _draw(axes[0], grid, f"Activation matrix of {key_tex}", ylabel=True)
    _draw(axes[1], np.clip(fourier_approx, 0.0, 1.0), rf"$\mathbf{{F}}_{{\mathrm{{Fourier}}}}(a,b),\ K={fourier_K}$", ylabel=True)

    fig.subplots_adjust(left=0.09, right=0.88, top=0.88, bottom=0.15, wspace=0.15)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cax = fig.add_axes([0.91, 0.15, 0.018, 0.73])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("normalised activation", labelpad=6)
    cbar.outline.set_edgecolor(ps.GRAY)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.ax.tick_params(length=0)

    plt.savefig(panels_path, bbox_inches="tight")
    plt.close(fig)


# ── TeX wrapper ───────────────────────────────────────────────────────────────

def _write_tex(
    tex_path: Path,
    panels_path: Path,
    key: str,
    pattern: str,
    fourier_r2: float,
    fourier_K: int,
    mu: float,
    modes: list[dict],
    N: int = 10,
) -> None:
    fig_name = panels_path.name

    # Build per-term strings (2 decimal places, single line)
    max_amp = max((m["amp"] for m in modes), default=1.0)
    active = [m for m in modes if m["amp"] >= 0.05 * max_amp][:3]
    mean_str = _fmt_num(float(mu), 2)
    cos_terms = [_mode_term_latex(m, N=N, digits=2) for m in active]

    parts = [mean_str] + ["+ " + t for t in cos_terms]
    approx_body = " ".join(parts)

    equiv_line = ""
    if pattern == "parity":
        equiv_line = r"\\ &{\color{qgreen}\equiv (a + b) \bmod 2 \;=\; (-1)^{a+b}}"

    formula_content = rf"&\approx {approx_body}{equiv_line}"

    tex = rf"""\documentclass[11pt]{{article}}
\usepackage[margin=0.5in]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{amsmath,amssymb}}
\usepackage{{microtype}}
\usepackage[dvipsnames]{{xcolor}}
\pagestyle{{empty}}
\setlength{{\parindent}}{{0pt}}
\definecolor{{qgreen}}{{HTML}}{{2D6A4F}}

\begin{{document}}
\begin{{center}}

\begin{{align*}}
\mathbf{{F}}_\mathbf{{Fourier}}(a,b) {formula_content}
\end{{align*}}

\vspace{{0.6em}}

\includegraphics[width=0.96\linewidth]{{{fig_name}}}

\end{{center}}
\end{{document}}
"""
    tex_path.write_text(tex)


def _compile_tex(tex_path: Path) -> Path | None:
    if shutil.which("pdflatex") is None:
        print(f"  [no pdflatex] wrote {tex_path}")
        return None
    try:
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            cwd=tex_path.parent, check=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError:
        print(f"  [pdflatex failed] see {tex_path.with_suffix('.log')}")
        return None
    for ext in (".aux", ".log", ".out", ".fls", ".fdb_latexmk"):
        tex_path.with_suffix(ext).unlink(missing_ok=True)
    return tex_path.with_suffix(".pdf")


# ── Fallback grid loader from all_feature_grids ──────────────────────────────

def _build_grid_from_all_feature_grids(sweep_dir: Path, key: str) -> np.ndarray | None:
    """Fallback: load grid from all_feature_grids npz when absent from sweep_activations.npz."""
    m = re.fullmatch(r"L(\d+)_F(\d+)", key, re.IGNORECASE)
    if not m:
        return None
    layer, feat_id = int(m.group(1)), int(m.group(2))
    fg_dir = sweep_dir / "all_feature_grids"
    npz_path = fg_dir / f"layer_{layer:02d}_all_feature_grids.npz"
    if not npz_path.exists():
        return None
    d = np.load(npz_path, allow_pickle=True)
    feat_ids = list(d["feat_ids"])
    if feat_id not in feat_ids:
        return None
    idx = feat_ids.index(feat_id)
    grid = d["grids"][idx].astype(np.float64)
    # fill single missing cells with 0 before normalising
    grid = np.nan_to_num(grid, nan=0.0)
    lo, hi = grid.min(), grid.max()
    if hi - lo > 1e-12:
        grid = (grid - lo) / (hi - lo)
    return grid


# ── Canonical key normalisation ───────────────────────────────────────────────

def _canonical_key(name: str) -> str:
    name = name.strip()
    # Already correct: L13_F107956
    if re.fullmatch(r"L\d+_F\d+", name, re.IGNORECASE):
        parts = name.upper().split("_F")
        return f"L{int(parts[0][1:])}_F{int(parts[1])}"
    # Missing F: L13_107956
    m = re.fullmatch(r"(L\d+)_(\d+)", name, re.IGNORECASE)
    if m:
        return f"L{int(m.group(1)[1:])}_F{int(m.group(2))}"
    raise ValueError(f"Cannot parse feature name: {name!r}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--features_json", required=True)
    parser.add_argument("--sweep_dir", required=True)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--fourier_K", type=int, default=8)
    parser.add_argument("--fourier_r2_target", type=float, default=0.95)
    parser.add_argument("--anchor", default=None,
                        help="Filter sweep candidates by directory name substring")
    args = parser.parse_args()

    raw_keys = json.loads(Path(args.features_json).read_text())
    # Deduplicate while preserving order
    seen: set[str] = set()
    keys: list[str] = []
    for k in raw_keys:
        try:
            ck = _canonical_key(k)
        except ValueError as e:
            print(f"  [skip] {e}")
            continue
        if ck not in seen:
            seen.add(ck)
            keys.append(ck)

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.features_json).parent / "feature_vs_fourier"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output → {out_dir}")

    N = 10
    for key in keys:
        print(f"\n{key}")
        grid = _build_grid_from_sweep(args.sweep_dir, key, anchor=args.anchor)
        if grid is None:
            grid = _build_grid_from_all_feature_grids(Path(args.sweep_dir), key)
            if grid is not None:
                print(f"  [fallback] loaded from all_feature_grids (NaN cells filled with 0)")
            else:
                print(f"  [skip] no grid data found")
                continue

        k_used, r2_val, modes, mu, _C, fourier_approx = find_min_k(
            grid, r2_target=args.fourier_r2_target, k_max=args.fourier_K,
            subtract_mean=True,
        )
        pattern = _classify_pattern(modes, N=N)
        print(f"  K={k_used}, R²={r2_val:.3f}, pattern={pattern}")

        panels_path = out_dir / f"panels_{key}.pdf"
        tex_path = out_dir / f"feature_vs_fourier_{key}.tex"

        _draw_panels(grid, fourier_approx, key, r2_val, k_used, panels_path)
        _write_tex(tex_path, panels_path, key, pattern, r2_val, k_used, mu, modes, N=N)
        compiled = _compile_tex(tex_path)
        if compiled:
            print(f"  → {compiled.name}")
        else:
            print(f"  → {tex_path.name}  (compile manually)")

    combined = _write_combined_report(out_dir, keys)
    if combined:
        print(f"Combined report → {combined.name}")

    print(f"\nDone. {len(keys)} features processed → {out_dir}")


# ── Combined report ───────────────────────────────────────────────────────────

_INTRO_TEX = r"""
\section*{Transcoder feature activations on the carry task}

\paragraph{Notation.}
The label $L^x_Y$ identifies transcoder feature $Y$ at transformer layer $x$.

\paragraph{Model.}
All activations are extracted from Qwen2.5-3B-Instruct, a decoder-only transformer
with 36 layers and a hidden dimension of 2048.

\paragraph{Transcoders.}
Sparse autoencoders trained on each MLP sublayer are drawn from the
\texttt{mwhanna/qwen3-4b-transcoders} collection on HuggingFace.
Each transcoder decomposes the MLP output at its layer into a sparse sum of
learned feature directions, allowing individual features to be identified and
their activations measured.

\paragraph{Dataset and prompt template.}
The addition dataset consists of prompts of the form \texttt{calc: a+b=},
where $a, b \in \{0, \ldots, 999\}$ are three-digit integers.
Activations are recorded at the position of the equals sign, immediately before
the first output digit is produced.
The carry concept is probed at column 0 of the addition, so the relevant
operand digits are $a \bmod 10$ and $b \bmod 10$.

\paragraph{Activations.}
Each left panel shows a $10 \times 10$ grid of mean feature activations indexed
by $(a \bmod 10,\, b \bmod 10)$.
Activations are drawn from the \texttt{anchor\_rank5\_pos9} contrastive sweep,
in which positive examples satisfy $a \bmod 10 + b \bmod 10 \geq 10$ (carry occurs)
and negative examples are matched pairs without carry.
All values are normalised to $[0, 1]$ per feature.

\paragraph{Fourier approximation.}
The activation grid is decomposed into real two-dimensional cosine modes on the
digit torus via the discrete Fourier transform.
The approximation $\mathbf{F}_\mathrm{Fourier}(a,b)$ retains the top-$K$ modes by
amplitude, where each mode takes the form
$A\cos\!\bigl(2\pi(ua + vb)/10 + \phi\bigr)$,
with $u, v$ integer frequencies along the $a$ and $b$ digit axes respectively.
$K$ is chosen as the smallest value that captures the dominant structure.

\paragraph{Equivalence.}
Where the $\equiv$ line appears below a Fourier formula, it gives the theoretical
closed-form expression for the arithmetic concept the feature appears to encode.
This expression is not fitted to data; it is the exact mathematical function
whose structure the Fourier decomposition approximates.
"""


def _write_combined_report(out_dir: Path, keys: list[str]) -> Path | None:
    """Single PDF: intro page followed by one feature per page."""
    pdfs = [out_dir / f"feature_vs_fourier_{k}.pdf" for k in keys
            if (out_dir / f"feature_vs_fourier_{k}.pdf").exists()]
    if not pdfs:
        return None

    includes = "\n".join(
        rf"\includepdf[pages=-]{{{p.name}}}" for p in pdfs
    )

    tex = rf"""\documentclass[11pt]{{article}}
\PassOptionsToPackage{{dvipsnames}}{{xcolor}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{amsmath,amssymb}}
\usepackage{{microtype}}
\usepackage{{pdfpages}}
\usepackage{{xcolor}}
\usepackage{{parskip}}
\setlength{{\parindent}}{{0pt}}
\definecolor{{qgreen}}{{HTML}}{{2D6A4F}}

\begin{{document}}

{_INTRO_TEX}

\clearpage

{includes}

\end{{document}}
"""
    tex_path = out_dir / "combined_feature_vs_fourier.tex"
    tex_path.write_text(tex)
    compiled = _compile_tex(tex_path)
    return compiled or tex_path


if __name__ == "__main__":
    main()
