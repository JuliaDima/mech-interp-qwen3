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
  python -m experiments.concept_localization.concept_fits.fourier_feature_analysis <path_to_matrix.csv> --K 8 --out feature_fourier.png --approx-csv feature_fourier_approx.csv
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
    # Nyquist checks must precede the generic u==0/v==0 checks to be reachable
    if abs(u) == N // 2 and abs(v) == N // 2:
        return "parity / (-1)^(a+b)"
    if abs(u) == N // 2 and v == 0:
        return "row-parity / (-1)^a"
    if u == 0 and abs(v) == N // 2:
        return "col-parity / (-1)^b"
    if u == 0:
        return "row-only / b-only"
    if v == 0:
        return "column-only / a-only"
    if u == v:
        return "iso-sum / a+b"
    if u == -v:
        return "iso-difference / b-a"
    return "mixed"


def mode_energy_breakdown(C: np.ndarray) -> dict[str, float]:
    """Fraction of AC (non-DC) signal energy by Fourier mode category.

    Categories: parity, sum, diff, row, col, mixed.
    Self-conjugate modes (only possible at Nyquist) are counted once; all
    other conjugate pairs are counted once (energy = 2*|C|^2).
    """
    N = C.shape[0]
    half = N // 2

    cats: dict[str, float] = {
        "diff": 0.0, "sum": 0.0, "row": 0.0, "col": 0.0,
        "parity": 0.0, "row_parity": 0.0, "col_parity": 0.0, "mixed": 0.0,
    }
    total = 0.0
    seen: set[tuple] = set()

    for u in range(N):
        for v in range(N):
            if u == 0 and v == 0:
                continue
            conj = ((-u) % N, (-v) % N)
            key = tuple(sorted([(u, v), conj]))
            if key in seen:
                continue
            seen.add(key)

            # Self-conjugate modes are real; count once, no *2
            self_conj = (u, v) == conj
            energy = abs(C[u, v]) ** 2 * (1 if self_conj else 2)
            total += energy

            su = u if u <= half else u - N
            sv = v if v <= half else v - N

            if u == 0:
                if abs(sv) == half:
                    cats["col_parity"] += energy
                else:
                    cats["col"] += energy
            elif v == 0:
                if abs(su) == half:
                    cats["row_parity"] += energy
                else:
                    cats["row"] += energy
            elif abs(su) == half and abs(sv) == half:
                cats["parity"] += energy
            elif su == sv:
                cats["sum"] += energy
            elif su == -sv:
                cats["diff"] += energy
            else:
                cats["mixed"] += energy

    if total > 1e-12:
        return {k: v / total for k, v in cats.items()}
    return {k: 0.0 for k in cats}


def dominant_mode_direction(breakdown: dict[str, float]) -> str:
    """Return a one-line summary of the dominant Fourier direction."""
    top = max(breakdown, key=breakdown.__getitem__)
    frac = breakdown[top]
    label = {
        "diff": "iso-difference (b-a)",
        "sum": "iso-sum (a+b)",
        "row": "row-only (a)",
        "col": "col-only (b)",
        "parity": "parity (-1)^(a+b)",
        "row_parity": "row-parity (-1)^a",
        "col_parity": "col-parity (-1)^b",
        "mixed": "mixed",
    }.get(top, top)
    others = sorted(
        [(k, v) for k, v in breakdown.items() if k != top and v > 0.05],
        key=lambda x: -x[1],
    )
    secondary = ", ".join(f"{k}={v:.0%}" for k, v in others[:2])
    s = f"{label} ({frac:.0%})"
    if secondary:
        s += f";  also {secondary}"
    return s


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


def print_report(
    mu: float, modes: list[dict], C: np.ndarray | None = None, N: int = 10, digits: int = 3
) -> None:
    print("\nDominant Fourier modes:")
    print(f"mean = {mu:.{digits}f}\n")

    for m in modes:
        typ = classify_mode(m["u"], m["v"], N=N)
        print(
            f"{m['amp']:.{digits}f} * "
            f"cos(2π({m['u']}a + {m['v']}b)/{N} + {m['phase']:.{digits}f})"
            f"    [{typ}]"
        )

    if C is not None:
        bd = mode_energy_breakdown(C)
        print("\nAC energy by mode direction:")
        for k, v in sorted(bd.items(), key=lambda x: -x[1]):
            bar = "#" * int(round(v * 30))
            print(f"  {k:8s}  {v:5.1%}  {bar}")
        print(f"  → dominant: {dominant_mode_direction(bd)}")

    print("\nExplicit approximate function:")
    print(formula_string(mu, modes, N=N, digits=digits))


def _save_grids_png(X: np.ndarray, Xhat: np.ndarray, path: Path, N: int = 10) -> None:
    """Save original + K-mode approximation grids side-by-side as a PNG."""
    ps.apply()
    cmap_seq = LinearSegmentedColormap.from_list("white_violet", ["white", ps.VIOLET])
    cmap_seq.set_bad("white")
    Xhat_clipped = np.clip(Xhat, 0.0, 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.2))
    for ax, data, label, do_ylabel in [
        (axes[0], X, "Original", True),
        (axes[1], Xhat_clipped, r"$K$-mode approximation", False),
    ]:
        ax.imshow(data.T, origin="lower", aspect="equal", cmap=cmap_seq, vmin=0, vmax=1)
        ax.set_xticks(range(N)); ax.set_yticks(range(N))
        ax.set_xticks(np.arange(-0.5, N, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, N, 1), minor=True)
        ax.tick_params(which="both", length=0, labelsize=7)
        ax.grid(which="minor", color="#DDDDDD", linewidth=0.3)
        ax.grid(which="major", visible=False)
        ax.set_axisbelow(False)
        for spine in ax.spines.values():
            spine.set_color(ps.GRAY)
        ax.set_xlabel("a mod 10", labelpad=4, fontsize=8)
        if do_ylabel:
            ax.set_ylabel("b mod 10", labelpad=4, fontsize=8)
        ax.set_title(label, fontsize=9)

    fig.tight_layout(pad=0.5)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ── LaTeX output helpers ──────────────────────────────────────────────────────

def _tex_escape(s: str) -> str:
    """Escape characters that are special in LaTeX text mode."""
    for ch, rep in [("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"),
                    ("$", r"\$"), ("&", r"\&"), ("#", r"\#"), ("^", r"\^{}"),
                    ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"),
                    ("²", r"$^2$"), ("→", r"$\to$")]:
        s = s.replace(ch, rep)
    return s


def _phase_latex(phase: float, digits: int = 3) -> str:
    """Return LaTeX phase string, e.g. '{}', '- \\pi', '+ 1.23'.

    Returns empty string when phase ≈ 0.
    """
    import math
    pi = math.pi
    tol = 0.04
    if abs(phase) < tol:
        return ""
    specials = [
        (pi,       r"\pi"),
        (pi / 2,   r"\tfrac{\pi}{2}"),
        (2 * pi / 3, r"\tfrac{2\pi}{3}"),
        (pi / 3,   r"\tfrac{\pi}{3}"),
        (3 * pi / 4, r"\tfrac{3\pi}{4}"),
        (pi / 4,   r"\tfrac{\pi}{4}"),
    ]
    for val, sym in specials:
        if abs(abs(phase) - val) < tol:
            return rf"- {sym}" if phase < 0 else rf"+ {sym}"
    sign = "-" if phase < 0 else "+"
    val = f"{abs(phase):.{digits}f}".rstrip("0").rstrip(".")
    return f"{sign} {val}"


def _freq_latex(u: int, v: int) -> str:
    """Return LaTeX string for the frequency argument $ua + vb$."""
    parts: list[str] = []
    if u != 0:
        coeff = "" if abs(u) == 1 else str(abs(u))
        parts.append(f"-{coeff}a" if u < 0 else f"{coeff}a")
    if v != 0:
        coeff = "" if abs(v) == 1 else str(abs(v))
        if parts:
            parts.append(f"- {coeff}b" if v < 0 else f"+ {coeff}b")
        else:
            parts.append(f"-{coeff}b" if v < 0 else f"{coeff}b")
    return " ".join(parts) or "0"


def formula_latex(
    mu: float,
    modes: list[dict],
    N: int = 10,
    digits: int = 3,
    amp_thresh: float = 0.05,
) -> str:
    """Return a LaTeX align* block for the Fourier approximation."""
    max_amp = max((m["amp"] for m in modes), default=1.0)
    active = [m for m in modes if m["amp"] >= amp_thresh * max_amp]

    mu_s = f"{mu:.{digits}f}".rstrip("0").rstrip(".") or "0"
    lines: list[str] = [rf"f(a,b) &\approx {mu_s}"]
    for m in active:
        amp = f"{m['amp']:.{digits}f}".rstrip("0").rstrip(".")
        freq = _freq_latex(int(m["u"]), int(m["v"]))
        ph = _phase_latex(float(m["phase"]), digits=digits)
        ph_str = f" {ph}" if ph else ""
        lines.append(
            rf"    &\quad + {amp}\,\cos\!\left(\frac{{2\pi({freq})}}{{{N}}}{ph_str}\right)"
        )

    inner = " \\\\\n".join(lines)
    return "\\begin{align*}\n" + inner + "\n\\end{align*}"


def _energy_table_latex(breakdown: dict[str, float]) -> str:
    """Return a LaTeX tabular for the AC energy breakdown."""
    desc = {
        "diff":    r"$\cos(2\pi k(b-a)/N)$",
        "sum":     r"$\cos(2\pi k(a+b)/N)$",
        "parity":  r"$(-1)^{a+b}$",
        "row":     r"function of $a$ only",
        "col":     r"function of $b$ only",
        "mixed":   r"mixed / cross-frequency",
    }
    rows: list[str] = []
    for k, v in sorted(breakdown.items(), key=lambda x: -x[1]):
        bar = r"\rule{" + f"{v * 5:.2f}" + r"cm}{4pt}"
        rows.append(
            rf"  {_tex_escape(k):12s} & {desc.get(k, '')} & {int(round(v*100))}\% & {bar} \\"
        )
    return (
        "\\begin{tabular}{llrl}\n"
        "\\toprule\n"
        "Direction & Structure & Fraction & \\\\\n"
        "\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}"
    )


def _modes_table_latex(modes: list[dict], N: int = 10, digits: int = 3) -> str:
    """Return a LaTeX tabular listing the top Fourier modes."""
    rows: list[str] = []
    for m in modes:
        typ = classify_mode(m["u"], m["v"], N=N)
        ph = _phase_latex(float(m["phase"]), digits=digits)
        ph_cell = f"${ph}$" if ph else "---"
        rows.append(
            rf"  {m['amp']:.{digits}f} & ${m['u']:+d}$ & ${m['v']:+d}$ "
            rf"& {ph_cell} & {_tex_escape(typ)} \\"
        )
    return (
        "\\begin{tabular}{rccll}\n"
        "\\toprule\n"
        "Amplitude & $u$ & $v$ & Phase & Type \\\\\n"
        "\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}"
    )


def write_latex_pdf(
    X: np.ndarray,
    Xhat: np.ndarray,
    C: np.ndarray,
    modes: list[dict],
    mu: float,
    out_path: str | Path,
    title: str,
    N: int = 10,
    k_used: int | None = None,
    r2: float | None = None,
    digits: int = 3,
) -> None:
    """Compile a LaTeX PDF: formula + energy table + mode table + grids figure.

    Cleans up all LaTeX build artefacts; only the PDF is kept.
    """
    import subprocess, shutil, tempfile

    out_path = Path(out_path).with_suffix(".pdf")
    bd = mode_energy_breakdown(C)
    dominant = dominant_mode_direction(bd)

    subtitle_parts: list[str] = []
    if k_used is not None:
        subtitle_parts.append(f"$K = {k_used}$")
    if r2 is not None:
        subtitle_parts.append(f"$R^2 = {r2:.3f}$")
    subtitle = r",\quad ".join(subtitle_parts)

    formula_block = formula_latex(mu, modes, N=N, digits=digits)
    energy_table = _energy_table_latex(bd)
    modes_table = _modes_table_latex(modes, N=N, digits=digits)

    tex_title = _tex_escape(title)

    doc = rf"""\documentclass[11pt,a4paper]{{article}}
\usepackage{{amsmath}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage[margin=2.2cm]{{geometry}}
\usepackage{{microtype}}
\parindent=0pt
\parskip=5pt

\begin{{document}}

\begin{{center}}
{{\LARGE\bfseries {tex_title}}}\\[5pt]
{subtitle}\\[3pt]
\emph{{Dominant Fourier direction: {_tex_escape(dominant)}}}
\end{{center}}

\medskip\hrule\medskip

\section*{{Formula ($N={N}$)}}

{formula_block}

\section*{{AC energy by mode direction}}

\begin{{center}}
{energy_table}
\end{{center}}

\section*{{Top {len(modes)} Fourier modes}}

\begin{{center}}
{modes_table}
\end{{center}}

\section*{{Activation grids}}

\begin{{figure}}[h]
\centering
\includegraphics[width=0.78\textwidth]{{grids.png}}
\caption{{Left: original activation grid (normalised). Right: $K$-mode Fourier approximation.}}
\end{{figure}}

\end{{document}}
"""

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _save_grids_png(X, Xhat, tmp / "grids.png", N=N)
        (tmp / "doc.tex").write_text(doc, encoding="utf-8")
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "doc.tex"],
            cwd=tmp,
            capture_output=True,
            text=True,
        )
        pdf_tmp = tmp / "doc.pdf"
        if not pdf_tmp.exists():
            log = (tmp / "doc.log").read_text(errors="replace") if (tmp / "doc.log").exists() else result.stdout + result.stderr
            raise RuntimeError(f"pdflatex failed:\n{log[-3000:]}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pdf_tmp, out_path)

    print(f"Saved PDF to: {out_path}")


# ── Scan helpers ─────────────────────────────────────────────────────────────

def _parse_layer_sel(layer_sel: str, available: list[int]) -> list[int]:
    s = layer_sel.strip().lower()
    if s == "all":
        return available
    avail = set(available)
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return [x for x in out if x in avail]


def _aggregate_grids(acts: np.ndarray, a_mod: np.ndarray, b_mod: np.ndarray, N: int = 10) -> np.ndarray:
    """Mean activation per feature per (a%N, b%N) cell. Returns (n_feats, N, N)."""
    n = min(len(acts), len(a_mod), len(b_mod))
    acts = acts[:n].astype(np.float32, copy=False)
    cell_ids = a_mod[:n] * N + b_mod[:n]
    one_hot = np.zeros((n, N * N), dtype=np.float32)
    one_hot[np.arange(n), cell_ids] = 1.0
    counts = one_hot.sum(axis=0)
    sums = acts.T @ one_hot
    means = np.where(counts > 0, sums / np.where(counts > 0, counts, 1.0), np.nan)
    return means.reshape(acts.shape[1], N, N)


def _build_scan_masks(N: int = 10) -> dict[str, np.ndarray]:
    """Precompute boolean masks (N×N) for each energy category — called once."""
    half = N // 2
    u_idx = np.arange(N)
    v_idx = np.arange(N)
    U, V = np.meshgrid(u_idx, v_idx, indexing="ij")
    SU = np.where(U <= half, U, U - N)
    SV = np.where(V <= half, V, V - N)
    dc = (U == 0) & (V == 0)
    _col_all  = (U == 0) & ~dc                                        # u=0, v≠0
    _row_all  = (V == 0) & ~dc                                        # v=0, u≠0
    col_parity = _col_all & (np.abs(SV) == half)                      # u=0, |v|=N/2  → (-1)^b
    row_parity = _row_all & (np.abs(SU) == half)                      # |u|=N/2, v=0  → (-1)^a
    col     = _col_all & ~col_parity                                   # u=0, v≠0, not Nyquist
    row     = _row_all & ~row_parity                                   # v=0, u≠0, not Nyquist
    parity  = (np.abs(SU) == half) & (np.abs(SV) == half) & ~dc & ~row & ~col & ~row_parity & ~col_parity
    sum_    = (SU == SV)  & ~dc & ~row & ~col & ~parity & ~row_parity & ~col_parity
    diff    = (SU == -SV) & ~dc & ~row & ~col & ~parity & ~row_parity & ~col_parity
    mixed   = ~dc & ~row & ~col & ~parity & ~sum_ & ~diff & ~row_parity & ~col_parity
    # canonical half for top-mode detection (deduplicate conjugate pairs)
    canonical = (U < half) | ((U == 0) & (V <= half)) | ((U == half))
    canonical &= ~dc
    return dict(col=col, row=row, parity=parity, sum=sum_, diff=diff, mixed=mixed,
                row_parity=row_parity, col_parity=col_parity,
                dc=dc, canonical=canonical, SU=SU, SV=SV)

_SCAN_MASKS_10 = _build_scan_masks(10)


def _score_grids_batch(grids: np.ndarray, N: int = 10, subtract_mean: bool = True) -> dict:
    """Vectorised FFT energy breakdown + top mode for a batch of (n, N, N) grids.

    Returns dict of arrays of length n.
    """
    masks = _SCAN_MASKS_10 if N == 10 else _build_scan_masks(N)
    X = np.nan_to_num(grids.astype(np.float64), nan=0.0)
    if subtract_mean:
        X = X - X.mean(axis=(1, 2), keepdims=True)
    C = np.fft.fft2(X) / (N * N)          # (n, N, N) complex
    P = np.abs(C) ** 2                     # power spectrum

    # Energy fractions (sum over category indices / total AC power)
    ac_total = P[:, ~masks["dc"]].sum(axis=1) + 1e-30
    cat_energy: dict[str, np.ndarray] = {
        k: P[:, masks[k]].sum(axis=1) / ac_total
        for k in ("col", "row", "parity", "sum", "diff", "mixed", "row_parity", "col_parity")
    }

    # Top mode amplitude and type (canonical half only)
    amps = 2.0 * np.sqrt(P)
    amps[:, ~masks["canonical"]] = 0.0
    flat = amps.reshape(len(grids), -1)
    best = flat.argmax(axis=1)
    top_u_raw = best // N
    top_v_raw = best % N
    top_su = np.where(top_u_raw <= N // 2, top_u_raw, top_u_raw - N).astype(int)
    top_sv = np.where(top_v_raw <= N // 2, top_v_raw, top_v_raw - N).astype(int)
    top_amp = flat[np.arange(len(grids)), best]
    top_type = [classify_mode(int(su), int(sv), N=N) for su, sv in zip(top_su, top_sv)]

    return {
        **cat_energy,
        "structured_energy": cat_energy["diff"] + cat_energy["sum"] + cat_energy["parity"] + cat_energy["row_parity"] + cat_energy["col_parity"],
        "top_mode_amp": top_amp,
        "top_mode_type": top_type,
        "_C_batch": C,
    }


def load_transcoder_model(device: str | None, dtype: str, transcoder_set: str | None = None):
    import torch
    from types import SimpleNamespace
    from mechinterp_qwen3.utils.hf_utils import load_transcoder_from_hub
    from mechinterp_qwen3.utils.model_utils import get_default_device, parse_dtype
    from scripts.model_config import default_transcoder_set

    torch_device = torch.device(device) if device is not None else get_default_device()
    torch_dtype = parse_dtype(dtype)
    transcoders, _ = load_transcoder_from_hub(
        transcoder_set or default_transcoder_set(),
        device=torch_device,
        dtype=torch_dtype,
        lazy_encoder=False,
        lazy_decoder=True,
    )
    return SimpleNamespace(transcoders=transcoders, cfg=SimpleNamespace(device=torch_device))


def load_examples_ab(examples_path: Path) -> tuple[np.ndarray, np.ndarray]:
    import pickle as _pickle
    with examples_path.open("rb") as f:
        records = _pickle.load(f)
    a_vals: list[int] = []
    b_vals: list[int] = []
    for rec in records:
        meta = rec.get("meta", {})
        for a_key, b_key in (("a_pos", "b_pos"), ("a_neg", "b_neg")):
            if a_key not in meta or b_key not in meta:
                raise KeyError(f"{examples_path} missing {a_key}/{b_key} in record: {rec}")
            a_vals.append(int(meta[a_key]))
            b_vals.append(int(meta[b_key]))
    return np.asarray(a_vals, dtype=np.int64), np.asarray(b_vals, dtype=np.int64)


def scan_from_residuals(
    residuals_path: Path,
    layer_sel: str = "all",
    subtract_mean: bool = True,
    device: str | None = None,
    dtype: str = "bfloat16",
    top_k_grids: int = 0,
    transcoder_set: str | None = None,
) -> tuple[list[dict], list[tuple[float, str, np.ndarray]]]:
    """Score every transcoder feature in sweep_residuals.npz by Fourier structure.

    Returns (rows, top_grids) where rows is sorted by structured_energy and
    top_grids holds (structured_energy, feat_key, grid) for the top_k_grids hits.
    """
    import heapq
    from experiments.concept_localization.sweep_utils import apply_transcoder_all

    residuals_path = Path(residuals_path)
    npz = np.load(str(residuals_path))
    available = [int(x) for x in npz["layers"].tolist()]
    layers = _parse_layer_sel(layer_sel, available)
    print(f"Scanning {len(layers)} layers from {residuals_path.name}")

    model = load_transcoder_model(device, dtype, transcoder_set=transcoder_set)
    examples_path = residuals_path.parent / "sweep_dataset_examples.pkl"
    a_vals, b_vals = load_examples_ab(examples_path)
    a_mod = (a_vals % 10).astype(np.int64)
    b_mod = (b_vals % 10).astype(np.int64)

    rows: list[dict] = []
    heap: list[tuple[float, int, str, np.ndarray]] = []  # min-heap on energy
    counter = 0

    for layer in layers:
        h_key = f"H_L{layer}"
        if h_key not in npz:
            continue
        H_l = npz[h_key].astype(np.float32)
        print(f"  L{layer}: {H_l.shape} → activations …", end=" ", flush=True)
        acts = apply_transcoder_all(model, layer, H_l)
        grids_raw = _aggregate_grids(acts, a_mod, b_mod)
        del acts
        n_feats = grids_raw.shape[0]

        # Drop features that never fired on any prompt (flat grids add noise to ranking)
        peak = np.nanmax(np.abs(grids_raw), axis=(1, 2))  # (n_feats,)
        active = np.where(peak > 1e-8)[0]
        n_dead = n_feats - len(active)
        if n_dead:
            print(f"({n_dead} dead features skipped) ", end="", flush=True)

        # Normalise active grids, score in one vectorised FFT pass
        grids_active = grids_raw[active]
        lo = np.nanmin(grids_active, axis=(1, 2), keepdims=True)
        hi = np.nanmax(grids_active, axis=(1, 2), keepdims=True)
        span = np.where(hi - lo > 1e-12, hi - lo, 1.0)
        grids_norm = np.nan_to_num((grids_active - lo) / span, nan=0.0).astype(np.float32)

        scores = _score_grids_batch(grids_norm, subtract_mean=subtract_mean)

        for i, feat_id in enumerate(active):
            feat_key = f"L{layer:02d}_F{feat_id}"
            se = float(scores["structured_energy"][i])
            rows.append({
                "feat_key":          feat_key,
                "layer":             layer,
                "feat_id":           feat_id,
                "diff_energy":       float(scores["diff"][i]),
                "sum_energy":        float(scores["sum"][i]),
                "parity_energy":     float(scores["parity"][i]),
                "row_parity_energy": float(scores["row_parity"][i]),
                "col_parity_energy": float(scores["col_parity"][i]),
                "structured_energy": se,
                "top_mode_amp":      float(scores["top_mode_amp"][i]),
                "top_mode_type":     scores["top_mode_type"][i],
            })
            if top_k_grids > 0:
                entry = (se, counter, feat_key, grids_norm[i])
                counter += 1
                if len(heap) < top_k_grids:
                    heapq.heappush(heap, entry)
                elif heap[0][0] < se:
                    heapq.heapreplace(heap, entry)

        print(f"{n_feats} features")
        del grids_raw

    rows.sort(key=lambda r: r["structured_energy"], reverse=True)
    top_grids = sorted(heap, key=lambda x: -x[0])
    return rows, [(e, k, g) for e, _, k, g in top_grids]


# ── Compact multi-feature LaTeX PDF ──────────────────────────────────────────

def formula_latex_compact(
    mu: float,
    modes: list[dict],
    N: int = 10,
    digits: int = 3,
    amp_thresh: float = 0.05,
    terms_per_line: int = 3,
) -> str:
    """Fourier formula with multiple cosine terms per line (multline* env)."""
    max_amp = max((m["amp"] for m in modes), default=1.0)
    active = [m for m in modes if m["amp"] >= amp_thresh * max_amp]
    mu_s = f"{mu:.{digits}f}".rstrip("0").rstrip(".") or "0"

    cos_strs: list[str] = []
    for m in active:
        amp = f"{m['amp']:.{digits}f}".rstrip("0").rstrip(".")
        freq = _freq_latex(int(m["u"]), int(m["v"]))
        ph = _phase_latex(float(m["phase"]), digits=digits)
        cos_strs.append(
            rf"{amp}\,\cos\!\left(\tfrac{{2\pi({freq})}}{{{N}}}{' ' + ph if ph else ''}\right)"
        )

    all_parts = [mu_s] + [f"+ {t}" for t in cos_strs]
    lines: list[str] = []
    for i in range(0, len(all_parts), terms_per_line):
        chunk = " ".join(all_parts[i : i + terms_per_line])
        lines.append(rf"f(a,b) \approx {chunk}" if i == 0 else chunk)

    if len(lines) <= 1:
        return "\\[\n" + (lines[0] if lines else "") + "\n\\]"
    return "\\begin{multline*}\n" + " \\\\\n".join(lines) + "\n\\end{multline*}"


_TYPE_DESC_SHORT = {
    "diff":    r"$\cos(2\pi k(b{-}a)/N)$",
    "sum":     r"$\cos(2\pi k(a{+}b)/N)$",
    "parity":  r"$(-1)^{a+b}$",
    "row":     r"$f(a)$",
    "col":     r"$f(b)$",
    "mixed":   "---",
}


def _feature_tex_page(
    idx: int,
    feat_key: str,
    k_used: int | None,
    r2: float | None,
    X: np.ndarray,
    Xhat: np.ndarray,
    C: np.ndarray,
    modes: list[dict],
    mu: float,
    N: int = 10,
    digits: int = 3,
) -> str:
    bd = mode_energy_breakdown(C)

    stats: list[str] = []
    if k_used is not None:
        stats.append(f"$K={k_used}$")
    if r2 is not None:
        stats.append(f"$R^2={r2:.3f}$")
    stats_str = r",~".join(stats)

    bd_summary = "; ".join(
        f"{k}: {int(round(v*100))}%" for k, v in sorted(bd.items(), key=lambda x: -x[1]) if v > 0.02
    )

    header = (
        rf"{{\large\textbf{{\textit{{{_tex_escape(feat_key)}}}}}}}"
        rf"\quad {stats_str}"
        rf"\qquad {{\small\textit{{{_tex_escape(bd_summary)}}}}}"
    )

    formula = formula_latex_compact(mu, modes, N=N, digits=digits, terms_per_line=3)

    # Energy table (no bar, compact)
    energy_rows = "".join(
        rf"  {k} & {_TYPE_DESC_SHORT.get(k, '---')} & {int(round(v*100))}\% \\" + "\n"
        for k, v in sorted(bd.items(), key=lambda x: -x[1])
        if v > 0.01
    )
    energy_table = (
        r"\begin{tabular}{llr}" + "\n"
        r"\toprule" + "\n"
        r"Dir & Structure & Frac \\" + "\n"
        r"\midrule" + "\n"
        + energy_rows
        + r"\bottomrule" + "\n"
        r"\end{tabular}"
    )

    img_name = f"grids_{idx}.png"

    return rf"""
{header}

\vspace{{2pt}}
{{\footnotesize
{formula}
}}

\vspace{{4pt}}
\noindent
\begin{{minipage}}[c]{{0.44\textwidth}}\centering\footnotesize
{energy_table}
\end{{minipage}}%
\hfill
\begin{{minipage}}[c]{{0.50\textwidth}}\centering
\includegraphics[width=\textwidth]{{{img_name}}}
{{\footnotesize Left: original.\quad Right: $K$-mode approx.}}
\end{{minipage}}
"""


def write_latex_pdf_multi(
    features: list[dict],
    out_path: Path,
) -> None:
    """Compile a single PDF with one page per feature (compact layout).

    Each dict in features must have: feat_key, X, Xhat, C, modes, mu.
    Optional keys: k_used, r2, N (default 10).
    """
    import subprocess, shutil, tempfile

    out_path = Path(out_path).with_suffix(".pdf")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        pages: list[str] = []
        for i, feat in enumerate(features):
            N = feat.get("N", 10)
            _save_grids_png(feat["X"], feat["Xhat"], tmp / f"grids_{i}.png", N=N)
            pages.append(_feature_tex_page(
                i, feat["feat_key"], feat.get("k_used"), feat.get("r2"),
                feat["X"], feat["Xhat"], feat["C"], feat["modes"], feat["mu"], N=N,
            ))

        doc = (
            r"\documentclass[10pt,a4paper]{article}" + "\n"
            r"\usepackage{amsmath}" + "\n"
            r"\usepackage{graphicx}" + "\n"
            r"\usepackage{booktabs}" + "\n"
            r"\usepackage[margin=1.5cm]{geometry}" + "\n"
            r"\usepackage{microtype}" + "\n"
            r"\parindent=0pt" + "\n"
            r"\parskip=3pt" + "\n"
            r"\begin{document}" + "\n"
            + "\n\\medskip\\hrule\\medskip\n".join(pages)
            + "\n" + r"\end{document}" + "\n"
        )
        (tmp / "doc.tex").write_text(doc, encoding="utf-8")
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "doc.tex"],
            cwd=tmp, capture_output=True, text=True,
        )
        pdf_tmp = tmp / "doc.pdf"
        if not pdf_tmp.exists():
            log = (tmp / "doc.log").read_text(errors="replace") if (tmp / "doc.log").exists() else result.stdout
            raise RuntimeError(f"pdflatex failed:\n{log[-3000:]}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pdf_tmp, out_path)

    print(f"Saved multi-feature PDF to: {out_path}")


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


def _parse_feat_key(key: str) -> tuple[int, int]:
    """Parse 'L13_F107956' → (layer=13, feat_id=107956)."""
    key = key.strip().upper()
    if not (key.startswith("L") and "_F" in key):
        raise ValueError(f"Expected format L<layer>_F<feat_id>, got {key!r}")
    layer_s, feat_s = key[1:].split("_F", 1)
    return int(layer_s), int(feat_s)


def load_from_npz_dir(npz_dir: Path, layer: int, feat_id: int) -> np.ndarray:
    """Load a normalised 10×10 grid from closest_feature_grids/ npz files."""
    candidates = (
        list(npz_dir.glob(f"layer_{layer:02d}_all_feature_grids.npz"))
        or list(npz_dir.glob(f"layer_{layer}_all_feature_grids.npz"))
    )
    if not candidates:
        raise FileNotFoundError(f"No npz for layer {layer} in {npz_dir}")
    data = np.load(candidates[0])
    feat_ids = data["feat_ids"]
    matches = np.where(feat_ids == feat_id)[0]
    if not len(matches):
        raise KeyError(f"Feature {feat_id} not found in {candidates[0]}")
    grid = data["grids"][int(matches[0])].astype(np.float64)
    lo, hi = grid.min(), grid.max()
    if hi - lo > 1e-12:
        grid = (grid - lo) / (hi - lo)
    return np.nan_to_num(grid, nan=0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="?", default=None, help="Input 10x10 matrix CSV.")
    parser.add_argument("--feat_key", default=None,
                        help="Feature key like L13_F107956; requires --npz_dir")
    parser.add_argument("--npz_dir", default=None, type=Path,
                        help="Directory containing layer_NN_all_feature_grids.npz files")
    parser.add_argument("--K", type=int, default=8, help="Maximum number of cosine modes (actual K chosen to hit --fourier_r2_target).")
    parser.add_argument("--fourier_r2_target", type=float, default=0.95,
                        help="Minimum R² for Fourier approximation; K is increased until reached.")
    parser.add_argument("--out", default=None, help="Output plot path.")
    parser.add_argument("--approx-csv", default=None, help="Optional CSV path for reconstructed Fourier approximation.")
    parser.add_argument("--modes-csv", default=None, help="Optional CSV path for top Fourier modes.")
    parser.add_argument("--title", default=None, help="Plot title.")
    parser.add_argument("--cmap", default="Reds", help="Matplotlib colormap for matrix plots.")
    parser.add_argument("--no-mean-subtract", action="store_true", help="Do not subtract the mean before FFT.")
    # scan mode
    parser.add_argument("--scan_residuals", default=None, type=Path,
                        help="sweep_residuals.npz: score all transcoder features for Fourier structure")
    parser.add_argument("--scan_layers", default="all",
                        help="Layers to scan, e.g. 'all', '10-20', '11,13'")
    parser.add_argument("--scan_out_csv", default=None, type=Path,
                        help="Output CSV (default: <residuals_dir>/fourier_scan.csv)")
    parser.add_argument("--scan_top_k_pdf", type=int, default=3,
                        help="Generate a combined PDF for the top-k features; 0 to disable")
    parser.add_argument("--scan_pdf_out", default=None, type=Path,
                        help="Output path for the top-k PDF")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--transcoder_set", default=None)
    args = parser.parse_args()

    subtract_mean = not args.no_mean_subtract

    # ── Scan mode: score all features from residuals ──────────────────────────
    if args.scan_residuals:
        rows, top_grids = scan_from_residuals(
            args.scan_residuals,
            layer_sel=args.scan_layers,
            subtract_mean=subtract_mean,
            device=args.device,
            dtype=args.dtype,
            top_k_grids=args.scan_top_k_pdf,
            transcoder_set=args.transcoder_set,
        )
        csv_path = args.scan_out_csv or args.scan_residuals.parent / "fourier_scan.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"Saved scan CSV ({len(rows)} features): {csv_path}")
        print(f"Top 10 by structured_energy:")
        for r in rows[:10]:
            print(f"  {r['feat_key']:18s}  structured={r['structured_energy']:.3f}"
                  f"  diff={r['diff_energy']:.3f}  sum={r['sum_energy']:.3f}"
                  f"  par={r['parity_energy']:.3f}  top={r['top_mode_type'][:18]}")

        if args.scan_top_k_pdf > 0 and top_grids:
            pdf_features = []
            for energy, feat_key, grid in top_grids:
                k_used, r2_val, modes, mu, C, Xhat = find_min_k(
                    grid, r2_target=args.fourier_r2_target, k_max=args.K,
                    subtract_mean=subtract_mean)
                pdf_features.append(dict(
                    feat_key=feat_key, X=grid, Xhat=Xhat,
                    C=C, modes=modes, mu=mu, k_used=k_used, r2=r2_val,
                ))
            pdf_path = args.scan_pdf_out or args.scan_residuals.parent / "fourier_scan_top.pdf"
            write_latex_pdf_multi(pdf_features, pdf_path)
        return

    # ── Single feature from npz_dir ───────────────────────────────────────────
    if args.feat_key:
        if not args.npz_dir:
            parser.error("--feat_key requires --npz_dir")
        layer, feat_id = _parse_feat_key(args.feat_key)
        X = load_from_npz_dir(Path(args.npz_dir), layer, feat_id)
        k_used, r2_val, modes, mu, C, Xhat = find_min_k(
            X, r2_target=args.fourier_r2_target, k_max=args.K, subtract_mean=subtract_mean)
        print(f"{args.feat_key}: K={k_used}, R²={r2_val:.3f}")
        print_report(mu, modes, C=C)
        out_path = args.out or f"fourier_{args.feat_key}.pdf"
        title = args.title or f"Fourier analysis --- {args.feat_key}"
        write_latex_pdf(X, Xhat, C, modes, mu, out_path, title,
                        k_used=k_used, r2=r2_val)
        return

    # ── Single CSV mode ───────────────────────────────────────────────────────
    if not args.csv:
        parser.error("Provide a CSV file, --feat_key + --npz_dir, or --scan_residuals")

    X = load_matrix_csv(args.csv)
    k_used, r2_val, modes, mu, C, Xhat = find_min_k(
        X, r2_target=args.fourier_r2_target, k_max=args.K, subtract_mean=subtract_mean)
    N = X.shape[0]
    print(f"K={k_used}, R²={r2_val:.3f}")
    print_report(mu, modes, C=C)

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
    title = args.title or f"Fourier analysis --- {Path(args.csv).name}"
    write_latex_pdf(X, Xhat, C, modes, mu, out_path, title, k_used=k_used, r2=r2_val)


if __name__ == "__main__":
    main()
