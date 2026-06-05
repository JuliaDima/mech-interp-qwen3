"""Run PySR on selected features from a sweep directory.

Two fitting modes are selected automatically from the sweep metadata:

  carry-grid   — concept has a_pos/b_pos fields; fits over a 10×10 units-digit
                 grid of mean activations (da, db and engineered carry features
                 as inputs). Produces a 3-panel heatmap comparison plot.

  generic-meta — all other concepts with at least one numeric meta field; stacks
                 pos and neg examples and uses numeric meta variables as inputs,
                 activation as target. Produces an actual-vs-predicted scatter.

Concepts whose metadata contains only non-numeric fields (e.g. causal_direction
with string cause/effect) are skipped with a clear message.

Can be driven by a cluster_features.json from analyze_sweep_clusters.py
(--cluster_features_json) or by explicit --features.

Usage
-----
    python scripts/sweeps/fit_pysr_sweep.py \
        --sweep_dir runs/concept_localization/carry/anchor_rank5_pos9/sweep \
        --features L19_F97965

    python scripts/sweeps/fit_pysr_sweep.py \\
        --sweep_dir runs/concept_localization/gcd/anchor_rank1_pos6/sweep \\
        --cluster_features_json .../cluster_analysis_T0/cluster_features.json \\
        --out_dir .../cluster_analysis_T0
"""

from __future__ import annotations

import argparse
import json
import pickle
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
import sympy as sp

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import experiments.plot_style as ps


# ── Meta introspection ────────────────────────────────────────────────────────

def _is_numeric(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _meta_mode(examples: list[dict]) -> str:
    """Return 'carry', 'generic', or 'skip'."""
    if not examples:
        return "skip"
    meta = examples[0].get("meta", {})
    if "a_pos" in meta and "b_pos" in meta and _is_numeric(meta["a_pos"]):
        return "carry"
    numeric_keys = [k for k, v in meta.items() if _is_numeric(v)]
    return "generic" if numeric_keys else "skip"


def _canonical_key(name: str) -> str:
    name = name.strip().upper()
    if "_F" in name and name.startswith("L"):
        layer_s, feat_s = name.split("_F", 1)
        return f"{layer_s}_F{int(feat_s)}"
    if "_" in name:
        layer_s, feat_s = name.split("_", 1)
        return f"L{int(layer_s.lstrip('L'))}_F{int(feat_s)}"
    raise ValueError(f"Cannot parse feature name: {name!r}")


# ── Carry mode: 10×10 digit grid ─────────────────────────────────────────────

def _build_digit_grid(acts: np.ndarray, examples: list[dict], use_pos: bool) -> np.ndarray:
    sums   = np.zeros((10, 10), dtype=np.float64)
    counts = np.zeros((10, 10), dtype=np.int64)
    for pair_i, ex in enumerate(examples):
        meta  = ex["meta"]
        a, b  = (meta["a_pos"], meta["b_pos"]) if use_pos else (meta["a_neg"], meta["b_neg"])
        act_i = 2 * pair_i if use_pos else 2 * pair_i + 1
        if act_i >= len(acts):
            continue
        da, db = a % 10, b % 10
        sums[da, db]   += float(acts[act_i])
        counts[da, db] += 1
    grid = np.full((10, 10), np.nan, dtype=np.float64)
    mask = counts > 0
    grid[mask] = sums[mask] / counts[mask]
    return grid


def _carry_table(key: str, npz, examples: list[dict], basis: str = "default"):
    acts     = np.asarray(npz[key], dtype=np.float64)
    pos_grid = _build_digit_grid(acts, examples, use_pos=True)
    neg_grid = _build_digit_grid(acts, examples, use_pos=False)
    grid     = np.where(np.isnan(pos_grid), neg_grid, pos_grid)

    # Fit the full digit torus. Some sampled sweeps miss one or more cells;
    # those cells are genuine unobserved activations for this grid and are set
    # to zero so PySR/Fourier see a consistent 10x10 target.
    grid = np.nan_to_num(grid, nan=0.0)
    da_idx, db_idx = np.indices((10, 10))
    da_idx, db_idx = da_idx.ravel(), db_idx.ravel()
    y = grid[da_idx, db_idx].astype(np.float64)
    lo, hi = y.min(), y.max()
    if hi - lo > 1e-12:
        y    = (y - lo) / (hi - lo)
        grid = (grid - lo) / (hi - lo)

    da, db = da_idx.astype(float), db_idx.astype(float)
    s  = da + db
    if basis == "trig":
        X = np.column_stack([2 * np.pi * da / 10.0, 2 * np.pi * db / 10.0])
        names = ["theta_a", "theta_b"]
    else:
        X  = np.column_stack([
            da, db, s, da - db, np.abs(da - db), da * db,
            np.mod(s, 10.0), np.mod(db - da, 10.0), s - 9.5, np.abs(s - 10.0),
            (np.round(s) == 10).astype(float),
        ])
        names = ["da", "db", "sum_ab", "delta_ab", "abs_delta",
                 "mul_ab", "sum_mod10", "diff_mod10", "carry_margin", "dist_sum10",
                 "sum_eq10"]
    return X, y, grid, names


# ── Generic mode: numeric meta fields ────────────────────────────────────────

def _generic_table(key: str, npz, examples: list[dict], basis: str = "default"):
    acts = np.asarray(npz[key], dtype=np.float64)
    meta0 = examples[0].get("meta", {})

    # Shared fields (no _pos/_neg suffix, numeric)
    shared_keys = sorted(k for k in meta0
                         if not k.endswith("_pos") and not k.endswith("_neg")
                         and _is_numeric(meta0[k]))
    # Base names that have _pos/_neg variants
    base_keys = sorted(set(
        k[:-4] for k in meta0
        if (k.endswith("_pos") or k.endswith("_neg")) and _is_numeric(meta0[k])
    ))

    all_var_names = shared_keys + base_keys
    if not all_var_names:
        return None, None, None, None

    rows_X, rows_y = [], []
    for pair_i, ex in enumerate(examples):
        meta = ex.get("meta", {})
        for use_pos in (True, False):
            act_i = 2 * pair_i if use_pos else 2 * pair_i + 1
            if act_i >= len(acts):
                continue
            suffix = "_pos" if use_pos else "_neg"
            row = [float(meta.get(k, 0)) for k in shared_keys]
            for k in base_keys:
                row.append(float(meta.get(k + suffix, meta.get(k, 0))))
            rows_X.append(row)
            rows_y.append(float(acts[act_i]))

    if not rows_X:
        return None, None, None, None

    X = np.array(rows_X, dtype=np.float64)
    y = np.array(rows_y, dtype=np.float64)
    lo, hi = y.min(), y.max()
    if hi - lo > 1e-12:
        y = (y - lo) / (hi - lo)

    trig_M = None
    if basis == "trig" and "a" in all_var_names:
        a_col = all_var_names.index("a")
        mod_var = "g" if "g" in all_var_names else ("m" if "m" in all_var_names else None)
        if mod_var is not None:
            M_vals = X[:, all_var_names.index(mod_var)]
            M = float(np.median(M_vals))
            if M > 1 and np.allclose(M_vals, M):
                trig_M = int(round(M))
                X = np.column_stack([2 * np.pi * X[:, a_col] / M])
                all_var_names = ["theta_a"]
    return X, y, None, all_var_names, trig_M   # no grid for generic mode


# ── PySR fit ──────────────────────────────────────────────────────────────────

def _fit_pysr(X: np.ndarray, y: np.ndarray, variable_names: list[str],
              niterations: int, seed: int,
              populations: int = 12, population_size: int = 64, maxsize: int = 10,
              ncycles_per_iteration: int = 650, parsimony: float = 0.015,
              pysr_basis: str = "default"):
    from pysr import PySRRegressor
    if pysr_basis == "trig":
        binary_operators = ["+", "-", "*"]
        unary_operators = ["sin", "cos"]
        extra_sympy_mappings = {}
        constraints = {}
        nested_constraints = {"sin": {"sin": 0, "cos": 0}, "cos": {"sin": 0, "cos": 0}}
        complexity_of_operators = {"sin": 2, "cos": 2}
    else:
        binary_operators = ["+", "-", "*", "/"]
        unary_operators = [
            "square",
            "relu(x) = max(x, 0.0f0)",
            "step10(x) = x >= 10.0f0 ? 1.0f0 : 0.0f0",
            "abs",
        ]
        extra_sympy_mappings = {
            "relu":   lambda x: sp.Piecewise((x, x > 0), (0, True)),
            "step10": lambda x: sp.Piecewise((1, x >= 10), (0, True)),
            "parity": lambda x: sp.Mod(x, 2),
        }
        constraints = {"/": (-1, 3)}
        nested_constraints = {}
        complexity_of_operators = {"/": 3, "square": 2, "relu": 3, "step10": 2, "abs": 2}
    model = PySRRegressor(
        niterations=niterations,
        populations=populations,
        population_size=population_size,
        maxsize=maxsize,
        ncycles_per_iteration=ncycles_per_iteration,
        parsimony=parsimony,
        model_selection="accuracy",
        binary_operators=binary_operators,
        unary_operators=unary_operators,
        extra_sympy_mappings=extra_sympy_mappings,
        constraints=constraints,
        nested_constraints=nested_constraints,
        complexity_of_operators=complexity_of_operators,
        complexity_of_constants=3,
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        tournament_selection_n=9,
        deterministic=True,
        random_state=seed,
        parallelism="serial",
        verbosity=0,
        progress=False,
    )
    model.fit(X, y, variable_names=variable_names)
    return model


# ── Plots ─────────────────────────────────────────────────────────────────────

# Map each PySR input-variable symbol to its human-readable LaTeX form.
# sum_mod10 and carry_margin are themselves engineered features, rendered as
# explicit function expressions so nesting (e.g. mod(mod(a+b,10), …)) is clear.
_VAR_LATEX = {
    "mul_{ab}":       r"a \cdot b",
    "sum_{ab}":       r"\left(a + b\right)",
    "delta_{ab}":     r"\left(a - b\right)",
    "abs_{delta}":    r"\left|a - b\right|",
    "sum_{mod10}":    r"\operatorname{mod}\!\left(a + b,\, 10\right)",
    "diff_{mod10}":   r"\operatorname{mod}\!\left(b - a,\, 10\right)",
    "carry_{margin}": r"\left(a + b - 9.5\right)",
    "dist_{sum10}":   r"\left|a + b - 10\right|",
    "sum_{eq10}":     r"\mathbf{1}_{a+b=10}",
    "da":             r"a",
    "db":             r"b",
}


def _round_floats(s: str, decimals: int = 2) -> str:
    def _fmt(m: re.Match) -> str:
        v = round(float(m.group(0)), decimals)
        r = f"{v:.{decimals}f}".rstrip("0").rstrip(".")
        return r
    return re.sub(r"\d+\.\d{3,}", _fmt, s)


def _expr_to_latex(expr) -> str:
    """LaTeX for a sympy expr with mod printed in function form (mod(x, y)) and
    multiplication shown with an explicit dot — avoids ambiguous infix \\bmod
    and the fragile manual \\cdot insertion."""
    from sympy.printing.latex import LatexPrinter

    class _P(LatexPrinter):
        def _print_Mod(self, e, exp=None):
            a, b = e.args
            base = r"\operatorname{mod}\!\left(%s,\, %s\right)" % (
                self._print(a), self._print(b))
            # sympy passes exp when the Mod is itself raised to a power
            return rf"\left({base}\right)^{{{exp}}}" if exp is not None else base

    return _P({"mul_symbol": "dot"}).doprint(expr)


def _mathtext_ok(s: str) -> bool:
    """True if the $...$ math string can be rendered by matplotlib mathtext.

    Piecewise fits (relu/step10) produce \\begin{cases} LaTeX which mathtext
    cannot parse; this lets the caller fall back to plain text instead of
    crashing at savefig time.
    """
    from matplotlib.mathtext import MathTextParser
    try:
        MathTextParser("agg").parse(s)
        return True
    except Exception:
        return False


def _clean_latex(s: str) -> str:
    s = _round_floats(s)
    s = s.replace(r"\theta_{a}", r"\theta_a")
    s = s.replace(r"\theta_{b}", r"\theta_b")
    for sym, tex in _VAR_LATEX.items():
        s = s.replace(sym, tex)
    # word-boundary fallbacks for no-brace variants (e.g. from model.latex())
    s = re.sub(r"\bmul_ab\b",    r"a \\cdot b",   s)
    s = re.sub(r"\bsum_ab\b",    r"(a + b)",      s)
    s = re.sub(r"\bdelta_ab\b",  r"(a - b)",      s)
    s = re.sub(r"\babs_delta\b", r"\\left|a - b\\right|", s)
    s = re.sub(r"\bsum_mod10\b", r"\\operatorname{mod}\\!\\left(a + b,\\, 10\\right)", s)
    s = re.sub(r"\bdiff_mod10\b", r"\\operatorname{mod}\\!\\left(b - a,\\, 10\\right)", s)
    s = re.sub(r"\bdist_sum10\b", r"\\left|a + b - 10\\right|", s)
    s = re.sub(r"\bsum_eq10\b", r"\\mathbf{1}_{a+b=10}", s)
    s = re.sub(r"(?<!\\)\btheta_a\b", lambda _: r"\theta_a", s)
    s = re.sub(r"(?<!\\)\btheta_b\b", lambda _: r"\theta_b", s)
    s = re.sub(r"\bda\b", "a", s)
    s = re.sub(r"\bdb\b", "b", s)
    return s


_VAR_NAMES = ["da", "db", "sum_ab", "delta_ab", "abs_delta",
              "mul_ab", "sum_mod10", "diff_mod10", "carry_margin", "dist_sum10",
              "sum_eq10"]

# Substitute the engineered PySR input symbols with their meaning as sympy
# expressions *before* printing — avoids all fragile LaTeX string surgery.
_A, _B = sp.Symbol("a"), sp.Symbol("b")
_VAR_SUBS = {
    sp.Symbol("da"):          _A,
    sp.Symbol("db"):          _B,
    sp.Symbol("sum_ab"):      _A + _B,
    sp.Symbol("delta_ab"):    _A - _B,
    sp.Symbol("abs_delta"):   sp.Abs(_A - _B),
    sp.Symbol("mul_ab"):      _A * _B,
    sp.Symbol("sum_mod10"):   sp.Mod(_A + _B, 10),
    sp.Symbol("diff_mod10"):  sp.Mod(_B - _A, 10),
    sp.Symbol("carry_margin"): _A + _B - sp.Rational(19, 2),
    sp.Symbol("dist_sum10"): sp.Abs(_A + _B - 10),
    sp.Symbol("sum_eq10"): sp.Piecewise((1, sp.Eq(_A + _B, 10)), (0, True)),
}


def _disp_latex(expr) -> str:
    """Display LaTeX for a sympy expr: substitute engineered symbols for their
    a,b meaning, collapse integer-valued floats (1.0·a → a), round, and print
    with function-form mod and dot multiplication."""
    disp = expr.subs(_VAR_SUBS).replace(
        lambda e: e.is_Float and float(e) == int(float(e)),
        lambda e: sp.Integer(int(float(e))))
    return _clean_latex(_round_floats(_expr_to_latex(disp)))


def _model_latex(model):
    """Clean display LaTeX for a fitted PySR model (simplify → symbol subs →
    function-form mod). Shared by carry and generic plots. None if unrenderable."""
    candidates = []
    try:
        candidates.append(_simplify_eq(model.get_best()["sympy_format"]))
    except Exception:
        pass
    candidates.append(_get_sympy(model))
    for c in candidates:
        if c is None:
            continue
        try:
            return _disp_latex(c)
        except Exception:
            continue
    try:
        return _clean_latex(model.latex())
    except Exception:
        return None


def _draw_formula_header(
    fig,
    key,
    latex_eq,
    raw_eq,
    r2,
    args="",
    fit_points: str | None = None,
):
    """Shared title/formula header with fixed rows to avoid text overlap."""
    plt.rcParams.update({"mathtext.fontset": "stix"})
    r2_str = f"$R^2={r2:.3f}$" if not np.isnan(r2) else "$R^2=\mathrm{n/a}$"
    m = re.fullmatch(r"L(\d+)_F(\d+)", key)
    key_latex = (rf"Feature $L^{{{m.group(1)}}}_{{{m.group(2)}}}$" if m else key)
    fig.text(0.5, 0.985, f"{key_latex}    {r2_str}",
             ha="center", va="top", fontsize=14, fontfamily="serif")

    if latex_eq:
        lhs = rf"\mathbf{{F}}_{{\mathbf{{PySR}}}}\mathbf{{{args} = }}"
        body = f"${lhs}$" + f"${latex_eq}$"
        if not _mathtext_ok(body):
            body = f"F_PySR{args} = {raw_eq}"
        body_len = len(body)
        fontsize = 17 if body_len < 80 else 14 if body_len < 120 else 11
        fig.text(0.5, 0.905, body, ha="center", va="top", fontsize=fontsize,
                 fontfamily="serif", color="#2D6A4F", transform=fig.transFigure,
                 linespacing=0.9)

    if fit_points:
        fig.text(0.5, 0.690, fit_points, ha="center", va="top",
                 fontsize=10, color=ps.GRAY, transform=fig.transFigure)


def _predict_grid_and_latex(model, X_full: np.ndarray, variable_names: list[str]):
    """Return (pred_grid 10×10, display LaTeX) for a fitted PySR model."""
    pred_grid = model.predict(X_full).reshape(10, 10)
    try:
        simp = _simplify_eq(model.get_best()["sympy_format"])
        fn = sp.lambdify([sp.Symbol(n) for n in variable_names], simp, modules="numpy")
        pred_flat = np.real(np.asarray(fn(*X_full.T), dtype=float))
        pred_grid = np.broadcast_to(pred_flat, (100,)).reshape(10, 10)
    except Exception:
        pass
    return pred_grid, _model_latex(model)




def _carry_full_X(variable_names: list[str]) -> np.ndarray:
    da_flat = np.repeat(np.arange(10), 10).astype(float)
    db_flat = np.tile(np.arange(10), 10).astype(float)
    s = da_flat + db_flat
    if variable_names == ["theta_a", "theta_b"]:
        return np.column_stack([2 * np.pi * da_flat / 10.0, 2 * np.pi * db_flat / 10.0])
    return np.column_stack([
        da_flat, db_flat, s, da_flat - db_flat,
        np.abs(da_flat - db_flat), da_flat * db_flat,
        np.mod(s, 10.0), np.mod(db_flat - da_flat, 10.0),
        s - 9.5, np.abs(s - 10.0),
        (np.round(s) == 10).astype(float),
    ])

def _get_sympy(model):
    try:
        return model.get_best()["sympy_format"]
    except Exception:
        return None


def _load_known_constants() -> dict:
    path = _REPO_ROOT / "experiments/concept_localization/known_maths_constants.json"
    try:
        import json as _json
        raw = _json.load(open(path))
        return {name: sp.parse_expr(expr) for name, expr in raw.items()}
    except Exception:
        return {}


def _simplify_eq(sympy_expr, tol: float = 1e-2):
    """Replace float constants close to simple/known constants with clean symbols."""
    known = list(_load_known_constants().values())
    simple = [
        sp.Integer(i) for i in range(-20, 21)
    ] + [
        sp.Rational(n, d)
        for d in (2, 3, 4, 5, 10)
        for n in range(-40, 41)
    ]
    constants = known + simple

    def _replace(x):
        if not x.is_Float:
            return x
        best_err, best_const = float("inf"), None
        for c in constants:
            try:
                err = abs(float(x) - float(c.evalf()))
                if err < best_err:
                    best_err, best_const = err, c
            except Exception:
                pass
        return best_const if best_err < tol else x

    simplified = sympy_expr.replace(lambda x: x.is_Float, _replace)
    try:
        simplified = sp.simplify(simplified)
    except Exception:
        pass
    return simplified


def _tex_escape_text(s: str) -> str:
    return (s.replace("\\", r"\textbackslash{}")
             .replace("&", r"\&")
             .replace("%", r"\%")
             .replace("$", r"\$")
             .replace("#", r"\#")
             .replace("_", r"\_")
             .replace("{", r"\{")
             .replace("}", r"\}")
             .replace("~", r"\textasciitilde{}")
             .replace("^", r"\textasciicircum{}"))


def _latex_path_for_tex(path: Path) -> str:
    return str(path).replace("\\", "/")


def _fmt_latex_num(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}".rstrip("0").rstrip(".")


def _fourier_formula_latex(mu: float, modes: list[dict], N: int = 10, digits: int = 3,
                           amp_thresh: float = 0.05, max_lines: int = 4,
                           max_terms: int = 3) -> str:
    """LaTeX aligned Fourier formula, simplified and split into at most max_lines rows.

    Near-zero amplitude modes (< amp_thresh * max_amplitude) are dropped.
    Phase signs are rendered cleanly (no '+ -x' artifacts).
    """
    import math as _math

    def _signed_term(coeff: int, var: str) -> str | None:
        if coeff == 0:
            return None
        if coeff == 1:
            return var
        if coeff == -1:
            return rf"-{var}"
        return rf"{coeff}{var}"

    def _freq_numerator(u: int, v: int) -> str:
        t_u = _signed_term(u, "a")
        t_v = _signed_term(v, "b")
        if t_u is None and t_v is None:
            return "0"
        if t_u is None:
            return t_v
        if t_v is None:
            return t_u
        return (f"{t_u} {t_v}" if t_v.startswith("-") else f"{t_u} + {t_v}")

    def _mode_term(m: dict) -> str:
        amp = _fmt_latex_num(float(m["amp"]), digits)
        phase = float(m["phase"])
        u, v = int(m["u"]), int(m["v"])
        num = _freq_numerator(u, v)
        freq = rf"\frac{{{num}}}{{{N}}}"
        phase_abs = abs(phase)
        phase_s = _fmt_latex_num(phase_abs, digits)
        if phase_abs < 1e-3:
            arg = rf"2\pi {freq}"
        elif phase > 0:
            arg = rf"2\pi {freq} + {phase_s}"
        else:
            arg = rf"2\pi {freq} - {phase_s}"
        return rf"{amp}\cos\!\left({arg}\right)"

    max_amp = max((m["amp"] for m in modes), default=1.0)
    active = [m for m in modes if m["amp"] >= amp_thresh * max_amp][:max_terms]

    terms = [_fmt_latex_num(float(mu), digits)]
    for m in active:
        terms.append(r"+ " + _mode_term(m))

    n = len(terms)
    per_line = max(1, _math.ceil(n / max_lines))
    lines = [" ".join(terms[i : i + per_line]) for i in range(0, n, per_line)]
    return (r" \\" + "\n&\quad ").join(lines)


def _write_carry_tex_report(
    tex_path: Path,
    figure_path: Path,
    key: str,
    pysr_latex: str | None,
    raw_eq: str,
    r2: float,
    n_pos: int,
    n_neg: int,
    mu: float,
    modes: list[dict],
    fourier_K: int,
) -> None:
    """Write a standalone LaTeX document around the heatmap figure."""
    m = re.fullmatch(r"L(\d+)_F(\d+)", key)
    feature_tex = (rf"Feature $L^{{{m.group(1)}}}_{{{m.group(2)}}}$" if m else _tex_escape_text(key))
    pysr_body = pysr_latex or _tex_escape_text(raw_eq)
    fourier_body = _fourier_formula_latex(mu, modes, N=10, digits=3, max_lines=1)
    fig_rel = _latex_path_for_tex(figure_path.name)
    tex = rf"""\documentclass[11pt]{{article}}
\usepackage[margin=0.55in]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{amsmath,amssymb}}
\usepackage{{microtype}}
\usepackage[dvipsnames]{{xcolor}}
\pagestyle{{empty}}
\setlength{{\parindent}}{{0pt}}
\definecolor{{qgreen}}{{HTML}}{{2D6A4F}}

\begin{{document}}
\begin{{center}}
\begin{{minipage}}{{0.94\linewidth}}
\centering
{{\color{{qgreen}}\large
\[
\mathbf{{F}}_\mathbf{{PySR}}(a,b) = {pysr_body}
\]
}}
\end{{minipage}}

\vspace{{0.2em}}
\begin{{minipage}}{{0.94\linewidth}}
\centering
\[
\mathbf{{F}}_\mathbf{{Fourier}}(a,b) = {fourier_body}
\]
\end{{minipage}}

\vspace{{0.6em}}
\includegraphics[width=0.96\linewidth]{{{fig_rel}}}
\end{{center}}

\end{{document}}
"""
    tex_path.write_text(tex)


def _compile_tex_if_available(tex_path: Path) -> Path | None:
    """Compile a generated LaTeX report if pdflatex exists; otherwise leave .tex."""
    if shutil.which("pdflatex") is None:
        print(f"    wrote {tex_path} (pdflatex not found; compile manually if needed)")
        return None
    cmd = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        tex_path.name,
    ]
    try:
        subprocess.run(cmd, cwd=tex_path.parent, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError:
        print(f"    wrote {tex_path} (pdflatex failed; see .log next to it)")
        return None
    for ext in (".aux", ".log", ".out", ".fls", ".fdb_latexmk"):
        tex_path.with_suffix(ext).unlink(missing_ok=True)
    return tex_path.with_suffix(".pdf")




def _write_combined_pdf_report(out_dir: Path, feature_keys: list[str], name: str = "pysr_combined") -> Path | None:
    """Combine individual generated PySR report PDFs, two per page."""
    pdfs: list[Path] = []
    for key in feature_keys:
        pdf = out_dir / f"pysr_{key}.pdf"
        tex = out_dir / f"pysr_{key}.tex"
        if not pdf.exists() and tex.exists():
            _compile_tex_if_available(tex)
        if pdf.exists():
            pdfs.append(pdf)
    if not pdfs:
        return None

    tex_path = out_dir / f"{name}.tex"
    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=0.35in]{geometry}",
        r"\usepackage{graphicx}",
        r"\pagestyle{empty}",
        r"\setlength{\parindent}{0pt}",
        r"\begin{document}",
    ]
    for idx, pdf in enumerate(pdfs):
        if idx % 2 == 0:
            lines.append(r"\begin{center}")
        lines.append(
            r"\includegraphics[width=0.98\linewidth,height=0.46\textheight,keepaspectratio]{"
            + _latex_path_for_tex(pdf.name)
            + r"}"
        )
        if idx % 2 == 0 and idx + 1 < len(pdfs):
            lines.append(r"\vfill")
        if idx % 2 == 1 or idx == len(pdfs) - 1:
            lines.append(r"\end{center}")
            if idx != len(pdfs) - 1:
                lines.append(r"\newpage")
    lines.append(r"\end{document}")
    tex_path.write_text("\n".join(lines) + "\n")
    compiled = _compile_tex_if_available(tex_path)
    return compiled or tex_path

def _plot_carry_combined(
    grid: np.ndarray,
    model,
    key: str,
    out_path: Path,
    variable_names: list[str],
    r2_threshold: float = 0.0,
    n_pos: int = 0,
    n_neg: int = 0,
    fourier_K: int = 8,
    fourier_r2_target: float = 0.95,
) -> float:
    """PySR + Fourier report. Returns R².

    The figure panels are saved separately and a LaTeX wrapper owns all title and
    equation text. This keeps long PySR/Fourier formulas readable and avoids
    Matplotlib mathtext overlap.
    """
    from scripts.sweeps.fourier_feature_analysis import find_min_k

    ps.apply()
    X_full = _carry_full_X(variable_names)
    pred_grid, latex_eq = _predict_grid_and_latex(model, X_full, variable_names)

    valid = ~np.isnan(grid)
    ss_res = float(np.sum((grid[valid] - pred_grid[valid]) ** 2))
    ss_tot = float(np.sum((grid[valid] - grid[valid].mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")

    if np.isnan(r2) or r2 < r2_threshold:
        print(f"    [plot anyway] {key}: PySR R²={r2:.3f} < threshold {r2_threshold}; keeping Fourier report")

    # Fourier decomposition on the same normalised grid (missing cells already 0)
    grid_clean = np.asarray(grid, dtype=float)
    fourier_k_used, fourier_r2_val, modes, mu, _, fourier_approx = find_min_k(
        grid_clean, r2_target=fourier_r2_target, k_max=fourier_K, subtract_mean=True)
    print(f"    Fourier: K={fourier_k_used}, R²={fourier_r2_val:.3f}")
    fourier_norm = np.clip(fourier_approx, 0.0, 1.0)

    cmap_seq = LinearSegmentedColormap.from_list("white_violet", ["white", ps.VIOLET])
    cmap_seq.set_bad("white")

    panels_path = out_path.with_name(f"{out_path.stem}_panels.pdf")
    tex_path = out_path.with_suffix(".tex")

    # Panels only. All equations/title text live in the generated TeX wrapper.
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.35))

    def _draw(ax, data, title, ylabel=False):
        ax.imshow(data.T, origin="lower", aspect="equal", cmap=cmap_seq, vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_xticks(range(10)); ax.set_yticks(range(10))
        ax.set_xticks(np.arange(-0.5, 10, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 10, 1), minor=True)
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

    _draw(axes[0], grid, f"Actual activations ({key})", ylabel=True)
    _draw(axes[1], pred_grid, f"PySR best fit  $R^2={r2:.3f}$")
    _draw(axes[2], fourier_norm, f"Fourier approx  $K={fourier_k_used},\\ R^2={fourier_r2_val:.2f}$")
    fig.subplots_adjust(left=0.065, right=0.985, top=0.86, bottom=0.16, wspace=0.18)
    plt.savefig(panels_path, bbox_inches="tight")
    plt.close(fig)

    raw_eq = str(model.get_best()["equation"])
    _write_carry_tex_report(
        tex_path=tex_path,
        figure_path=panels_path,
        key=key,
        pysr_latex=latex_eq,
        raw_eq=raw_eq,
        r2=r2,
        n_pos=n_pos,
        n_neg=n_neg,
        mu=mu,
        modes=modes,
        fourier_K=fourier_k_used,
    )
    compiled = _compile_tex_if_available(tex_path)
    if compiled:
        print(f"    wrote {compiled} and {tex_path}")
    return r2

def _plot_carry(grid: np.ndarray, model, key: str, out_path: Path,
                variable_names: list[str],
                r2_threshold: float = 0.0,
                n_pos: int = 0,
                n_neg: int = 0) -> float:
    """Plot the actual / fit / residual grids. Returns R²; skips saving when
    R² < r2_threshold (the fit does not explain enough structure to plot)."""
    ps.apply()
    # da = row index (ones_a), db = col index (ones_b) — row-major so da varies slowly
    X_full = _carry_full_X(variable_names)
    pred_grid, latex_eq = _predict_grid_and_latex(model, X_full, variable_names)

    valid = ~np.isnan(grid)
    ss_res = float(np.sum((grid[valid] - pred_grid[valid]) ** 2))
    ss_tot = float(np.sum((grid[valid] - grid[valid].mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")

    if np.isnan(r2) or r2 < r2_threshold:
        print(f"    [skip plot] {key}: R²={r2:.3f} < threshold {r2_threshold}")
        return r2

    resid = np.abs(grid - pred_grid)

    cmap_seq = LinearSegmentedColormap.from_list("white_violet", ["white", ps.VIOLET])
    cmap_seq.set_bad("white")
    cmap_blue = LinearSegmentedColormap.from_list("white_navy", ["white", ps.NAVY])
    cmap_blue.set_bad("white")

    fig, axes = plt.subplots(1, 3, figsize=(10, 4.8))
    for idx, (ax, (data, cmap, title)) in enumerate(zip(axes, [
        (grid,      cmap_seq,  "Actual activations"),
        (pred_grid, cmap_seq,  "PySR best fit"),
        (resid,     cmap_blue, "|Residual|"),
    ])):
        ax.imshow(data.T, origin="lower", aspect="equal", cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.set_xticks(np.arange(-0.5, 10, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 10, 1), minor=True)
        ax.tick_params(which='both', length=0)   # no tick marks on either major or minor
        ax.grid(which='minor', color='#DDDDDD', linewidth=0.3)
        ax.grid(which='major', visible=False)
        ax.set_axisbelow(False)
        for spine in ax.spines.values():
            spine.set_color(ps.GRAY)
        ax.tick_params(axis='both', colors='black', length=0)
        # Axis labels only on the leftmost panel; tick numbers kept on all
        if idx == 0:
            ax.set_xlabel("a mod 10", labelpad=6)
            ax.set_ylabel("b mod 10", labelpad=6)

    # One shared colorbar — placed in a dedicated axes to avoid overlap
    sm = plt.cm.ScalarMappable(
        cmap=LinearSegmentedColormap.from_list("cb", ["white", ps.VIOLET]),
        norm=plt.Normalize(vmin=0, vmax=1),
    )
    sm.set_array([])
    fig.subplots_adjust(left=0.07, right=0.85, top=0.54, bottom=0.12, wspace=0.10)
    cax = fig.add_axes([0.875, 0.12, 0.010, 0.42])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("normalised activation", labelpad=6)
    cbar.outline.set_edgecolor(ps.GRAY)
    cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.ax.tick_params(length=0)

    _draw_formula_header(
        fig,
        key,
        latex_eq,
        str(model.get_best()["equation"]),
        r2,
        args="(a,\,b)",
        fit_points=f"PySR fit points: {n_pos} positive / {n_neg} negative",
    )

    plt.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    return r2


def _polar_residue(ax, residue, M, values, color, title, rmax, faint=True, pos_mask=None):
    """Draw a closed per-residue mean loop around the circle (one value per spoke),
    with faint individual points for context. If pos_mask is given, positive and
    negative examples are drawn in separate colours."""
    spokes = np.arange(M)
    theta_r = 2 * np.pi * spokes / M
    means = np.array([values[residue == r].mean() if (residue == r).any() else np.nan
                      for r in spokes])
    if faint:
        rng = np.random.default_rng(0)
        theta_pts = 2 * np.pi * residue / M
        jitter = (rng.random(len(values)) - 0.5) * 0.02
        if pos_mask is not None:
            ax.scatter(theta_pts[pos_mask], values[pos_mask] + jitter[pos_mask],
                       s=8, alpha=0.35, color=ps.NAVY, label="positive")
            ax.scatter(theta_pts[~pos_mask], values[~pos_mask] + jitter[~pos_mask],
                       s=8, alpha=0.25, color=ps.GRAY, label="negative")
        else:
            ax.scatter(theta_pts, values + jitter, s=6, alpha=0.12, color=ps.GRAY)
    tc = np.append(theta_r, theta_r[0])
    ax.plot(tc, np.append(means, means[0]), "-o", color=color, lw=1.8, ms=5)
    ax.set_xticks(theta_r)
    ax.set_xticklabels([str(i) for i in spokes], fontsize=8)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(90)
    ax.set_ylim(0, rmax)
    ax.set_title(title, pad=14)


def _modular_residue(X: np.ndarray, var_names: list[str]):
    """If the concept is modular (has a divisor 'g' or modulus 'm' plus operand
    'a'), return (residue = a mod M, M); else None. Enables a circular plot."""
    mod_var = "g" if "g" in var_names else ("m" if "m" in var_names else None)
    if mod_var is None or "a" not in var_names:
        return None
    mod_col = X[:, var_names.index(mod_var)]
    M = int(round(float(np.median(mod_col))))
    if M < 2 or M > 60 or not np.allclose(mod_col, M):
        return None  # only the constant-modulus case has clean fixed spokes
    a = np.round(X[:, var_names.index("a")]).astype(np.int64)
    return np.mod(a, M), M


def _plot_generic(X: np.ndarray, y: np.ndarray, model, key: str,
                  var_names: list[str], out_path: Path,
                  r2_threshold: float = 0.0,
                  n_pos: int = 0,
                  n_neg: int = 0,
                  trig_M: int | None = None) -> float:
    """Scatter + residual plot for generic-meta fits. Returns R²; skips saving
    when R² < r2_threshold. Modular concepts (gcd, residue) are plotted against
    the arithmetic variable a with positive and negative examples separated."""
    ps.apply()
    y_pred = model.predict(X)
    best_eq = str(model.get_best()["equation"])

    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    if np.isnan(r2) or r2 < r2_threshold:
        print(f"    [skip plot] {key}: R²={r2:.3f} < threshold {r2_threshold}")
        return r2

    idx = np.arange(len(y))
    is_pos = (idx % 2 == 0)
    a_col = var_names.index("a") if "a" in var_names else 0
    x_vals = X[:, a_col]
    sort_idx = np.argsort(x_vals)

    x_plot = x_vals[sort_idx]
    y_sorted = y[sort_idx]
    y_pred_sorted = y_pred[sort_idx]
    resid_sorted = np.abs(y_sorted - y_pred_sorted)
    is_pos_sorted = is_pos[sort_idx]

    is_trig_modular = "theta_a" in var_names
    mod_var = "g" if "g" in var_names else ("m" if "m" in var_names else None)
    is_modular = (mod_var is not None and "a" in var_names) or is_trig_modular
    if is_modular:
        if is_trig_modular:
            theta_vals = X[:, var_names.index("theta_a")]
            # M comes from _generic_table (the actual GCD/modulus value)
            M = trig_M if trig_M is not None else 2
            theta_wrapped = theta_vals % (2 * np.pi)
            residue = np.round(theta_wrapped * M / (2 * np.pi)).astype(int) % M
            pos_residue = 0  # GCD: fires when a is divisible by g
            concept_label = f"a mod {M} = 0"
        else:
            M = int(round(float(np.median(X[:, var_names.index(mod_var)]))))
            residue = np.mod(np.round(x_vals).astype(np.int64), M)
            pos_residue = int(np.bincount(residue[is_pos]).argmax()) if np.any(is_pos) else 0
            concept_label = (f"a mod {M} = {pos_residue}"
                             if mod_var == "m" else f"a mod {M} = 0")

        rmax = 1.0
        spokes = np.arange(M)
        theta_spokes = 2 * np.pi * spokes / M

        fig = plt.figure(figsize=(15, 5.2))
        ax0 = fig.add_subplot(1, 3, 1, projection="polar")
        _polar_residue(ax0, residue, M, y, ps.NAVY, f"Actual activations ({key})", rmax=rmax,
                       pos_mask=is_pos)

        ax1 = fig.add_subplot(1, 3, 2, projection="polar")
        _polar_residue(ax1, residue, M, y_pred, ps.TEAL, f"PySR fit  $R^2={r2:.3f}$", rmax=rmax)

        # Theoretical: all M spokes visible; active spoke highlighted
        ax2 = fig.add_subplot(1, 3, 3, projection="polar")
        for r, theta in zip(spokes, theta_spokes):
            if r == pos_residue:
                ax2.plot([theta, theta], [0, rmax], color=ps.VIOLET, lw=2.5)
                ax2.plot(theta, rmax, "o", color=ps.VIOLET, ms=6)
            else:
                ax2.plot([theta, theta], [0, rmax * 0.35], color=ps.GRAY, lw=1.2, alpha=0.55)
        ax2.set_xticks(theta_spokes)
        ax2.set_xticklabels([str(i) for i in spokes], fontsize=8)
        ax2.set_theta_zero_location("N")
        ax2.set_theta_direction(-1)
        ax2.set_rlabel_position(90)
        ax2.set_ylim(0, rmax)
        ax2.set_title(f"Theoretical: {concept_label}", pad=14)
    else:
        fig = plt.figure(figsize=(15, 4.8))
        ax0 = fig.add_subplot(1, 3, 1)
        ax0.plot(x_plot, y_sorted, "-o", ms=3, lw=1.2, alpha=0.9, color=ps.NAVY)
        ax0.set_xlabel(var_names[a_col])
        ax0.set_ylabel("activation (normalised)")
        ax0.set_title("Actual activations")

        ax1 = fig.add_subplot(1, 3, 2)
        ax1.plot(x_plot, y_pred_sorted, "-o", ms=3, lw=1.2, alpha=0.9, color=ps.TEAL)
        ax1.set_xlabel(var_names[a_col])
        ax1.set_ylabel("predicted activation")
        ax1.set_title(f"PySR fit  $R^2={r2:.3f}$")

        ax2 = fig.add_subplot(1, 3, 3)
        ax2.plot(x_plot, resid_sorted, "-o", ms=3, lw=1.2, alpha=0.9, color=ps.RED)
        ax2.set_xlabel(var_names[a_col])
        ax2.set_ylabel("|error|")
        ax2.set_title("Residuals")

    latex_eq = _model_latex(model)
    fig.subplots_adjust(left=0.06, right=0.97, top=0.54, bottom=0.15, wspace=0.26)
    _draw_formula_header(
        fig,
        key,
        latex_eq,
        best_eq,
        r2,
        args="",
        fit_points=f"PySR fit points: {n_pos} positive / {n_neg} negative",
    )

    plt.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    return r2


# ── Per-feature entry point ───────────────────────────────────────────────────

def fit_feature(
    key: str,
    npz,
    examples: list[dict],
    mode: str,
    out_dir: Path,
    niterations: int = 40,
    seed: int = 0,
    r2_threshold: float = 0.0,
    pysr_kwargs: dict | None = None,
    with_fourier: bool = False,
    fourier_K: int = 8,
    fourier_r2_target: float = 0.95,
    pysr_basis: str = "default",
) -> dict | None:
    if key not in npz:
        print(f"  [skip] {key} not in sweep_activations.npz")
        return None

    trig_M = None
    if mode == "carry":
        X, y, grid, names = _carry_table(key, npz, examples, basis=pysr_basis)
    else:
        result = _generic_table(key, npz, examples, basis=pysr_basis)
        if result[0] is None:
            print(f"  [skip] {key}: no numeric meta fields")
            return None
        X, y, grid, names, trig_M = result

    print(f"  Fitting {key}  mode={mode}  ({len(y)} pts)  y=[{y.min():.3g}, {y.max():.3g}]")
    prefix = out_dir / f"pysr_{key}"
    kw = pysr_kwargs or {}

    def _r2_or_none(v):
        return None if np.isnan(v) else round(float(v), 4)

    model = _fit_pysr(X, y, names, niterations=niterations, seed=seed, pysr_basis=pysr_basis, **kw)
    best_eq = str(model.get_best()["equation"])
    print(f"    best: {best_eq}")

    # Count pos/neg points separately
    n_pos = int((len(y) + 1) // 2)
    n_neg = int(len(y) // 2)

    if mode == "carry":
        plot_fn = _plot_carry_combined if with_fourier else _plot_carry
        plot_kw = {"fourier_K": fourier_K, "fourier_r2_target": fourier_r2_target} if with_fourier else {}
        r2 = plot_fn(
            grid,
            model,
            key,
            prefix.with_suffix(".pdf"),
            names,
            r2_threshold,
            n_pos=n_pos,
            n_neg=n_neg,
            **plot_kw,
        )
    else:
        r2 = _plot_generic(
            X,
            y,
            model,
            key,
            names,
            prefix.with_suffix(".pdf"),
            r2_threshold,
            n_pos=n_pos,
            n_neg=n_neg,
            trig_M=trig_M,
        )

    # Convert PySR equations table into JSON-safe plain Python objects.
    equations_table = []
    for row in model.equations_.to_dict(orient="records"):
        clean_row = {}
        for k, v in row.items():
            try:
                import numpy as _np
                import sympy as _sp

                if isinstance(v, (_np.floating, float)):
                    clean_row[k] = float(v)
                elif isinstance(v, (_np.integer, int)):
                    clean_row[k] = int(v)
                elif isinstance(v, (_sp.Basic,)):
                    clean_row[k] = str(v)
                else:
                    clean_row[k] = str(v) if not isinstance(v, (str, bool, type(None))) else v
            except Exception:
                clean_row[k] = str(v)

        equations_table.append(clean_row)

    record = {
        "feature": key,
        "mode": mode,
        "variables": names,
        "pysr_basis": pysr_basis,
        "n_points": int(len(y)),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "best_equation": str(best_eq),
        "r2": _r2_or_none(r2),
        "plotted": bool(not np.isnan(r2) and r2 >= r2_threshold),
        "equations_table": equations_table,
    }

    return record


# ── Sweep discovery ──────────────────────────────────────────────────────────

def _resolve_sweep_dir(path: Path, features: list[str] | None = None,
                       anchor: str | None = None) -> Path:
    """Return the sweep directory that contains sweep_activations.npz.

    If ``path`` already contains the file, return it directly.  Otherwise
    search recursively for all sweep_activations.npz files under ``path``.
    Pass ``anchor`` (e.g. ``'rank5_pos9'``) to filter candidates by directory
    name substring before scoring.
    """
    if (path / "sweep_activations.npz").exists():
        return path
    candidates = sorted(path.rglob("sweep_activations.npz"))
    if not candidates:
        raise FileNotFoundError(f"No sweep_activations.npz found under {path}")
    if anchor:
        filtered = [c for c in candidates if anchor in str(c)]
        if filtered:
            candidates = filtered
        else:
            print(f"  [sweep] warning: no candidate matches anchor={anchor!r}, ignoring filter")
    if not features:
        chosen = candidates[0].parent
        print(f"  [sweep] using: {chosen}")
        return chosen
    for c in candidates:
        keys = set(np.load(c).files)
        if all(k in keys for k in features):
            print(f"  [sweep] using: {c.parent}")
            return c.parent
    best = max(candidates, key=lambda c: sum(k in set(np.load(c).files) for k in features))
    print(f"  [sweep] using (partial match): {best.parent}")
    return best.parent


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--sweep_dir", required=True,
                        help="Directory containing sweep_activations.npz and sweep_examples.pkl")
    parser.add_argument("--features", nargs="*", default=None,
                        help="Explicit feature keys, e.g. L23_F91721 L22_F12345")
    parser.add_argument("--features_json", default=None,
                        help="JSON file containing a list of feature keys (e.g. features_list_to_plot.json)")
    parser.add_argument("--cluster_features_json", default=None,
                        help="cluster_features.json from analyze_sweep_clusters; "
                             "fits top-3 per cluster when --features is omitted")
    parser.add_argument("--out_dir", default=None,
                        help="Output directory (default: cluster_features_json dir or sweep_dir)")
    parser.add_argument("--niterations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--r2_threshold", type=float, default=0.0,
                        help="Only plot a feature's PySR fit when its R² exceeds this")
    parser.add_argument("--maxsize", type=int, default=10,
                        help="Max equation complexity (larger = more expressive fits)")
    parser.add_argument("--parsimony", type=float, default=0.015,
                        help="Complexity penalty for PySR equation selection; raise for simpler fits")
    parser.add_argument("--pysr_basis", choices=["default", "trig"], default="default",
                        help="PySR search space: tuned engineered variables/operators, or trig-only Fourier-like basis")
    parser.add_argument("--populations", type=int, default=12)
    parser.add_argument("--population_size", type=int, default=64)
    parser.add_argument("--ncycles_per_iteration", type=int, default=650)
    parser.add_argument("--anchor", default=None,
                        help="Filter sweep candidates by directory name substring, e.g. rank5_pos9")
    parser.add_argument("--with_fourier", action="store_true",
                        help="Add Fourier analysis row to each carry-mode plot")
    parser.add_argument("--fourier_K", type=int, default=8,
                        help="Maximum Fourier modes; actual K chosen to hit --fourier_r2_target")
    parser.add_argument("--fourier_r2_target", type=float, default=0.95,
                        help="Minimum Fourier R² target; K is increased until reached (up to fourier_K)")
    parser.add_argument("--no_combined_pdf", action="store_true",
                        help="Do not write the combined two-reports-per-page PySR PDF")
    parser.add_argument("--combined_pdf_name", default="pysr_combined",
                        help="Base filename for the combined PySR PDF")
    args = parser.parse_args()

    pysr_kwargs = {
        "populations": args.populations,
        "population_size": args.population_size,
        "maxsize": args.maxsize,
        "ncycles_per_iteration": args.ncycles_per_iteration,
        "parsimony": args.parsimony,
    }

    # Resolve feature list first so sweep discovery can match against them
    features: list[str] = []
    if args.features:
        features = [_canonical_key(k) for k in args.features]
    elif args.features_json:
        raw = json.loads(Path(args.features_json).read_text())
        features = [_canonical_key(k) for k in raw]
    elif args.cluster_features_json:
        cf = json.loads(Path(args.cluster_features_json).read_text())
        for feat_list in cf.values():
            features.extend(_canonical_key(k) for k in feat_list)
    else:
        parser.error("Provide --features, --features_json, or --cluster_features_json")

    seen: set[str] = set()
    unique_features = [k for k in features if not (k in seen or seen.add(k))]

    sweep_dir = _resolve_sweep_dir(Path(args.sweep_dir), unique_features, anchor=args.anchor)
    npz = np.load(sweep_dir / "sweep_activations.npz")
    with open(sweep_dir / "sweep_examples.pkl", "rb") as f:
        examples = pickle.load(f)

    mode = _meta_mode(examples)
    if mode == "skip":
        print(f"  [skip] no numeric meta fields in {sweep_dir} — PySR skipped")
        return

    out_dir = Path(args.out_dir) if args.out_dir else (
        Path(args.features_json).parent if args.features_json else
        Path(args.cluster_features_json).parent if args.cluster_features_json else sweep_dir
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"PySR mode: {mode}  ({len(unique_features)} features)")
    results = []
    for key in unique_features:
        r = fit_feature(key, npz, examples, mode, out_dir,
                        niterations=args.niterations, seed=args.seed,
                        r2_threshold=args.r2_threshold, pysr_kwargs=pysr_kwargs,
                        with_fourier=args.with_fourier, fourier_K=args.fourier_K,
                        fourier_r2_target=args.fourier_r2_target,
                        pysr_basis=args.pysr_basis)
        if r:
            results.append(r)

    summary_path = out_dir / "pysr_summary.json"
    with summary_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"PySR summary → {summary_path}")
    if not args.no_combined_pdf:
        combined = _write_combined_pdf_report(
            out_dir,
            [r["feature"] for r in results],
            name=args.combined_pdf_name,
        )
        if combined:
            print(f"Combined PySR report → {combined}")
        else:
            print("Combined PySR report skipped: no individual PySR PDFs found")


if __name__ == "__main__":
    main()
