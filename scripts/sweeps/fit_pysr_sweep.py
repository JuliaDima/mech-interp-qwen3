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
        return f"L{int(layer_s)}_F{int(feat_s)}"
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


def _carry_table(key: str, npz, examples: list[dict]):
    acts     = np.asarray(npz[key], dtype=np.float64)
    pos_grid = _build_digit_grid(acts, examples, use_pos=True)
    neg_grid = _build_digit_grid(acts, examples, use_pos=False)
    grid     = np.where(np.isnan(pos_grid), neg_grid, pos_grid)

    da_idx, db_idx = np.where(~np.isnan(grid))
    y = grid[da_idx, db_idx].astype(np.float64)
    lo, hi = y.min(), y.max()
    if hi - lo > 1e-12:
        y    = (y - lo) / (hi - lo)
        grid = (grid - lo) / (hi - lo)

    da, db = da_idx.astype(float), db_idx.astype(float)
    s  = da + db
    X  = np.column_stack([da, db, s, da - db, np.abs(da - db),
                           da * db, np.mod(s, 10.0), s - 9.5])
    names = ["da", "db", "sum_ab", "delta_ab", "abs_delta",
             "mul_ab", "sum_mod10", "carry_margin"]
    return X, y, grid, names


# ── Generic mode: numeric meta fields ────────────────────────────────────────

def _generic_table(key: str, npz, examples: list[dict]):
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
    return X, y, None, all_var_names   # no grid for generic mode


# ── PySR fit ──────────────────────────────────────────────────────────────────

def _fit_pysr(X: np.ndarray, y: np.ndarray, variable_names: list[str],
              niterations: int, seed: int,
              populations: int = 8, population_size: int = 24, maxsize: int = 12,
              ncycles_per_iteration: int = 550):
    from pysr import PySRRegressor
    model = PySRRegressor(
        niterations=niterations,
        populations=populations,
        population_size=population_size,
        maxsize=maxsize,
        ncycles_per_iteration=ncycles_per_iteration,
        binary_operators=["+", "-", "*", "/", "mod", "pow"],
        unary_operators=[
            "sin", "cos", "square",
            "relu(x) = max(x, 0.0f0)",
            "step10(x) = x >= 10.0f0 ? 1.0f0 : 0.0f0",
            "parity(x) = mod(x, 2.0f0)",   # 0 for even, 1 for odd integers
            "abs",
        ],
        extra_sympy_mappings={
            "relu":   lambda x: sp.Piecewise((x, x > 0), (0, True)),
            "step10": lambda x: sp.Piecewise((1, x >= 10), (0, True)),
            "parity": lambda x: sp.Mod(x, 2),
        },
        constraints={"/": (-1, 5), "mod": (-1, 3), "pow": (-1, 2)},
        nested_constraints={"sin": {"sin": 0, "cos": 0}, "cos": {"sin": 0, "cos": 0}},
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
    "carry_{margin}": r"\left(a + b - 9.5\right)",
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
    for sym, tex in _VAR_LATEX.items():
        s = s.replace(sym, tex)
    # word-boundary fallbacks for no-brace variants (e.g. from model.latex())
    s = re.sub(r"\bmul_ab\b",    r"a \\cdot b",   s)
    s = re.sub(r"\bsum_ab\b",    r"(a + b)",      s)
    s = re.sub(r"\bdelta_ab\b",  r"(a - b)",      s)
    s = re.sub(r"\babs_delta\b", r"\\left|a - b\\right|", s)
    s = re.sub(r"\bsum_mod10\b", r"\\operatorname{mod}\\!\\left(a + b,\\, 10\\right)", s)
    s = re.sub(r"\bda\b", "a", s)
    s = re.sub(r"\bdb\b", "b", s)
    return s


_VAR_NAMES = ["da", "db", "sum_ab", "delta_ab", "abs_delta",
              "mul_ab", "sum_mod10", "carry_margin"]

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
    sp.Symbol("carry_margin"): _A + _B - sp.Rational(19, 2),
}


def _disp_latex(expr) -> str:
    """Display LaTeX for a sympy expr: substitute engineered symbols for their
    a,b meaning, collapse integer-valued floats (1.0·a → a), round, and print
    with function-form mod and dot multiplication."""
    disp = expr.subs(_VAR_SUBS).replace(
        lambda e: e.is_Float and float(e) == int(float(e)),
        lambda e: sp.Integer(int(float(e))))
    return _round_floats(_expr_to_latex(disp))


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


def _draw_formula_header(fig, key, latex_eq, raw_eq, r2, args="", y=0.88):
    """Shared title + green F_PySR formula header for carry and generic plots."""
    plt.rcParams.update({"mathtext.fontset": "stix"})
    r2_str = f"$R^2={r2:.3f}$" if not np.isnan(r2) else "$R^2=\\mathrm{n/a}$"
    m = re.fullmatch(r"L(\d+)_F(\d+)", key)
    key_latex = (rf"Feature $L^{{{m.group(1)}}}_{{{m.group(2)}}}$" if m else key)
    fig.suptitle(f"{key_latex}    {r2_str}", fontsize=16, y=0.99)
    if not latex_eq:
        return
    lhs = rf"\mathbf{{F}}_{{\mathbf{{PySR}}}}\mathbf{{{args} = }}"
    body = f"${lhs}$" + f"${latex_eq}$"
    if not _mathtext_ok(body):
        body = f"F_PySR{args} = {raw_eq}"
    fig.text(0.5, y, body, ha="center", va="top", fontsize=22,
             fontfamily="serif", color="#2D6A4F", transform=fig.transFigure)


def _predict_grid_and_latex(model, X_full: np.ndarray):
    """Return (pred_grid 10×10, display LaTeX) for a fitted PySR model."""
    pred_grid = model.predict(X_full).reshape(10, 10)
    try:
        simp = _simplify_eq(model.get_best()["sympy_format"])
        fn = sp.lambdify([sp.Symbol(n) for n in _VAR_NAMES], simp, modules="numpy")
        pred_flat = np.real(np.asarray(fn(*X_full.T), dtype=float))
        pred_grid = np.broadcast_to(pred_flat, (100,)).reshape(10, 10)
    except Exception:
        pass
    return pred_grid, _model_latex(model)


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
    """Replace float constants close to known constants (π, e, √2 …) with their symbols."""
    known = list(_load_known_constants().values())
    if not known:
        return sympy_expr

    def _replace(x):
        if not x.is_Float:
            return x
        best_err, best_const = float("inf"), None
        for c in known:
            try:
                err = abs(float(x) - float(c.evalf()))
                if err < best_err:
                    best_err, best_const = err, c
            except Exception:
                pass
        return best_const if best_err < tol else x

    return sympy_expr.replace(lambda x: x.is_Float, _replace)


def _plot_carry(grid: np.ndarray, model, key: str, out_path: Path,
                r2_threshold: float = 0.0) -> float:
    """Plot the actual / fit / residual grids. Returns R²; skips saving when
    R² < r2_threshold (the fit does not explain enough structure to plot)."""
    ps.apply()
    # da = row index (ones_a), db = col index (ones_b) — row-major so da varies slowly
    da_flat = np.repeat(np.arange(10), 10).astype(float)
    db_flat = np.tile(np.arange(10), 10).astype(float)
    s = da_flat + db_flat
    X_full = np.column_stack([da_flat, db_flat, s, da_flat - db_flat,
                               np.abs(da_flat - db_flat), da_flat * db_flat,
                               np.mod(s, 10.0), s - 9.5])

    pred_grid, latex_eq = _predict_grid_and_latex(model, X_full)

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

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
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
    fig.subplots_adjust(left=0.07, right=0.85, top=0.67, bottom=0.13, wspace=0.08)
    cax = fig.add_axes([0.875, 0.13, 0.010, 0.54])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("normalised activation", labelpad=6)
    cbar.outline.set_edgecolor(ps.GRAY)
    cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.ax.tick_params(length=0)

    _draw_formula_header(fig, key, latex_eq, str(model.get_best()["equation"]), r2,
                         args="(a,\\,b)", y=0.88)

    plt.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    return r2


def _polar_residue(ax, residue, M, values, color, title, rmax, faint=True):
    """Draw a closed per-residue mean loop around the circle (one value per spoke),
    with faint individual points for context."""
    spokes = np.arange(M)
    theta_r = 2 * np.pi * spokes / M
    means = np.array([values[residue == r].mean() if (residue == r).any() else np.nan
                      for r in spokes])
    if faint:
        rng = np.random.default_rng(0)
        ax.scatter(2 * np.pi * residue / M, values + (rng.random(len(values)) - 0.5) * 0.02,
                   s=6, alpha=0.12, color=ps.GRAY)
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
                  r2_threshold: float = 0.0) -> float:
    """Scatter + residual plot for generic-meta fits. Returns R²; skips saving
    when R² < r2_threshold. Modular concepts (gcd, residue) get a circular
    activation-by-residue panel instead of the linear pair-order one."""
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
    mod_info = _modular_residue(X, var_names)

    fig = plt.figure(figsize=(15, 4))
    ax1 = fig.add_subplot(1, 3, 2)
    ax2 = fig.add_subplot(1, 3, 3)

    # Panel 0: circular for modular concepts, else linear pair-order view.
    if mod_info is not None:
        residue, M = mod_info
        ax0 = fig.add_subplot(1, 3, 1, projection="polar")
        spokes = np.arange(M)
        theta_r = 2 * np.pi * spokes / M
        # Per-residue means (collapses 400 piled points into one value per spoke)
        mean_act = np.array([y[residue == r].mean() if (residue == r).any() else np.nan
                             for r in spokes])
        mean_pred = np.array([y_pred[residue == r].mean() if (residue == r).any() else np.nan
                              for r in spokes])
        # Faint individual activations for context (radial jitter so they're visible)
        rng = np.random.default_rng(0)
        ax0.scatter(2 * np.pi * residue / M, y + (rng.random(len(y)) - 0.5) * 0.02,
                    s=6, alpha=0.12, color=ps.GRAY)
        # Closed loops: actual mean and PySR-fit mean as circles
        tc = np.append(theta_r, theta_r[0])
        ax0.plot(tc, np.append(mean_act, mean_act[0]), "-o", color=ps.NAVY,
                 lw=1.6, ms=5, label="actual (mean)")
        ax0.plot(tc, np.append(mean_pred, mean_pred[0]), "--s", color=ps.TEAL,
                 lw=1.6, ms=4, label="PySR fit (mean)")
        ax0.set_xticks(theta_r)
        ax0.set_xticklabels([str(i) for i in spokes], fontsize=8)
        ax0.set_theta_zero_location("N")
        ax0.set_theta_direction(-1)
        ax0.set_rlabel_position(90)
        ax0.set_title(f"Activation by residue  (a mod {M})", pad=14)
        ax0.legend(fontsize=7, loc="lower right", bbox_to_anchor=(1.18, -0.05))
    else:
        ax0 = fig.add_subplot(1, 3, 1)
        ax0.scatter(idx[is_pos], y[is_pos], s=12, alpha=0.6, color=ps.NAVY, label="pos (actual)")
        ax0.scatter(idx[~is_pos], y[~is_pos], s=12, alpha=0.6, color=ps.RED, label="neg (actual)")
        ax0.scatter(idx, y_pred, s=6, alpha=0.5, color=ps.TEAL, marker="x", label="PySR fit")
        ax0.set_xlabel("example index (pair order)")
        ax0.set_ylabel("activation (normalised)")
        ax0.set_title("Activation by example")
        ax0.legend(fontsize=7, loc="upper right")

    ax = ax1
    ax.scatter(y, y_pred, s=8, alpha=0.4, color=ps.NAVY)
    lim = [min(y.min(), y_pred.min()) - 0.05, max(y.max(), y_pred.max()) + 0.05]
    ax.plot(lim, lim, color=ps.RED, lw=1.0, ls="--", alpha=0.7)
    ax.set_xlabel("actual activation (normalised)")
    ax.set_ylabel("predicted")
    ax.set_title("Actual vs predicted")

    resid = y - y_pred
    ax2.hist(resid, bins=30, color=ps.TEAL, alpha=0.75, edgecolor="white")
    ax2.axvline(0, color=ps.RED, lw=1.0, ls="--")
    ax2.set_xlabel("residual")
    ax2.set_ylabel("count")
    ax2.set_title(f"Residuals  (n={len(y)})")

    latex_eq = _model_latex(model)
    fig.subplots_adjust(left=0.06, right=0.97, top=0.72, bottom=0.15, wspace=0.28)
    _draw_formula_header(fig, key, latex_eq, best_eq, r2, args="", y=0.90)

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
) -> dict | None:
    if key not in npz:
        print(f"  [skip] {key} not in sweep_activations.npz")
        return None

    if mode == "carry":
        X, y, grid, names = _carry_table(key, npz, examples)
    else:
        result = _generic_table(key, npz, examples)
        if result[0] is None:
            print(f"  [skip] {key}: no numeric meta fields")
            return None
        X, y, grid, names = result

    print(f"  Fitting {key}  mode={mode}  ({len(y)} pts)  y=[{y.min():.3g}, {y.max():.3g}]")
    prefix = out_dir / f"pysr_{key}"
    kw = pysr_kwargs or {}

    def _r2_or_none(v):
        return None if np.isnan(v) else round(float(v), 4)

    model = _fit_pysr(X, y, names, niterations=niterations, seed=seed, **kw)
    best_eq = str(model.get_best()["equation"])
    print(f"    best: {best_eq}")
    model.equations_.to_csv(prefix.with_suffix(".csv"), index=False)

    if mode == "carry":
        r2 = _plot_carry(grid, model, key, prefix.with_suffix(".png"), r2_threshold)
    else:
        r2 = _plot_generic(X, y, model, key, names, prefix.with_suffix(".png"), r2_threshold)

    record = {"feature": key, "mode": mode, "variables": names, "n_points": int(len(y)),
              "best_equation": best_eq, "r2": _r2_or_none(r2),
              "plotted": bool(not np.isnan(r2) and r2 >= r2_threshold)}

    with prefix.with_suffix(".json").open("w") as f:
        json.dump(record, f, indent=2)
    return record


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--sweep_dir", required=True,
                        help="Directory containing sweep_activations.npz and sweep_examples.pkl")
    parser.add_argument("--features", nargs="*", default=None,
                        help="Explicit feature keys, e.g. L23_F91721 L22_F12345")
    parser.add_argument("--cluster_features_json", default=None,
                        help="cluster_features.json from analyze_sweep_clusters; "
                             "fits top-3 per cluster when --features is omitted")
    parser.add_argument("--out_dir", default=None,
                        help="Output directory (default: cluster_features_json dir or sweep_dir)")
    parser.add_argument("--niterations", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--r2_threshold", type=float, default=0.5,
                        help="Only plot a feature's PySR fit when its R² exceeds this")
    parser.add_argument("--maxsize", type=int, default=12,
                        help="Max equation complexity (larger = more expressive fits)")
    parser.add_argument("--populations", type=int, default=8)
    parser.add_argument("--population_size", type=int, default=24)
    parser.add_argument("--ncycles_per_iteration", type=int, default=550)
    args = parser.parse_args()

    pysr_kwargs = {
        "populations": args.populations,
        "population_size": args.population_size,
        "maxsize": args.maxsize,
        "ncycles_per_iteration": args.ncycles_per_iteration,
    }

    sweep_dir = Path(args.sweep_dir)
    npz = np.load(sweep_dir / "sweep_activations.npz")
    with open(sweep_dir / "sweep_examples.pkl", "rb") as f:
        examples = pickle.load(f)

    mode = _meta_mode(examples)
    if mode == "skip":
        print(f"  [skip] no numeric meta fields in {sweep_dir} — PySR skipped")
        return

    # Resolve feature list
    features: list[str] = []
    if args.features:
        features = [_canonical_key(k) for k in args.features]
    elif args.cluster_features_json:
        cf = json.loads(Path(args.cluster_features_json).read_text())
        for feat_list in cf.values():
            features.extend(_canonical_key(k) for k in feat_list)
    else:
        parser.error("Provide --features or --cluster_features_json")

    seen: set[str] = set()
    unique_features = [k for k in features if not (k in seen or seen.add(k))]

    out_dir = Path(args.out_dir) if args.out_dir else (
        Path(args.cluster_features_json).parent if args.cluster_features_json else sweep_dir
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"PySR mode: {mode}  ({len(unique_features)} features)")
    results = []
    for key in unique_features:
        r = fit_feature(key, npz, examples, mode, out_dir,
                        niterations=args.niterations, seed=args.seed,
                        r2_threshold=args.r2_threshold, pysr_kwargs=pysr_kwargs)
        if r:
            results.append(r)

    summary_path = out_dir / "pysr_summary.json"
    with summary_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"PySR summary → {summary_path}")


if __name__ == "__main__":
    main()
