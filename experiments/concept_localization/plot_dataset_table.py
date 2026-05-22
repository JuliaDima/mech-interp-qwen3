"""Generate a LaTeX longtable illustrating all concept datasets.

One row per concept: a randomly rotated template (T0/T1/T2) and the
shortest representative pos/neg pair for that template.
Differing tokens are wrapped in \\textbf{}.

Output: runs/concept_localization/dataset_table.tex

LaTeX preamble requirements:
    \\usepackage{booktabs, array, longtable}
"""

from __future__ import annotations

import difflib
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ── LaTeX helpers ─────────────────────────────────────────────────────────────

# Characters that need escaping in LaTeX text mode
_LATEX_ESC: dict[str, str] = {
    "%": r"\%",
    "&": r"\&",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "\\": r"\textbackslash{}",
    "$": r"\$",
}

# Unicode math symbols → LaTeX commands safe inside \texttt{}
_UNICODE_MATH: dict[str, str] = {
    "·": r"\ensuremath{\cdot}",
    "⊂": r"\ensuremath{\subset}",
    "⊃": r"\ensuremath{\supset}",
    "→": r"\ensuremath{\to}",
    "←": r"\ensuremath{\leftarrow}",
    "≥": r"\ensuremath{\geq}",
    "≤": r"\ensuremath{\leq}",
    "≠": r"\ensuremath{\neq}",
    "×": r"\ensuremath{\times}",
    "∈": r"\ensuremath{\in}",
}


def _esc(s: str) -> str:
    """Escape LaTeX special chars and substitute Unicode math symbols."""
    out: list[str] = []
    for ch in s:
        if ch in _UNICODE_MATH:
            out.append(_UNICODE_MATH[ch])
        elif ch in _LATEX_ESC:
            out.append(_LATEX_ESC[ch])
        else:
            out.append(ch)
    return "".join(out)


def _apply_opcodes(seq_pos, seq_neg, opcodes, join_with=" ") -> tuple[str, str]:
    pos_parts: list[str] = []
    neg_parts: list[str] = []
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            chunk = join_with.join(_esc(x) for x in seq_pos[i1:i2])
            pos_parts.append(chunk)
            neg_parts.append(chunk)
        elif tag == "replace":
            pos_parts.append(r"\textbf{" + join_with.join(_esc(x) for x in seq_pos[i1:i2]) + "}")
            neg_parts.append(r"\textbf{" + join_with.join(_esc(x) for x in seq_neg[j1:j2]) + "}")
        elif tag == "delete":
            pos_parts.append(r"\textbf{" + join_with.join(_esc(x) for x in seq_pos[i1:i2]) + "}")
        elif tag == "insert":
            neg_parts.append(r"\textbf{" + join_with.join(_esc(x) for x in seq_neg[j1:j2]) + "}")
    return join_with.join(pos_parts), join_with.join(neg_parts)


def diff_plain(pos: str, neg: str) -> tuple[str, str]:
    """Return (pos_tex, neg_tex) with \\textbf{} on differing spans.

    Word-level diff when spaces are present; character-level otherwise.
    """
    pos = pos.rstrip()
    neg = neg.rstrip()
    if " " in pos or " " in neg:
        pw, nw = pos.split(" "), neg.split(" ")
        m = difflib.SequenceMatcher(None, pw, nw, autojunk=False)
        return _apply_opcodes(pw, nw, m.get_opcodes(), " ")
    else:
        m = difflib.SequenceMatcher(None, list(pos), list(neg), autojunk=False)
        return _apply_opcodes(list(pos), list(neg), m.get_opcodes(), "")


# ── Dataset registry ──────────────────────────────────────────────────────────
# (display_name_for_latex, module, function, kwargs, preferred_max_pos_len)
# display names use spaces; no LaTeX-escaped underscores needed.

CONCEPTS: list[tuple[str, str, str, dict, int]] = [
    (
        "carry",
        "data.concept_datasets.carry_dataset",
        "generate_carry_pairs",
        {"n_per_template": 300},
        20,
    ),
    ("gcd", "data.concept_datasets.gcd_dataset", "generate_gcd_pairs", {"n_per_template": 300}, 25),
    (
        "residue class",
        "data.concept_datasets.residue_class_dataset",
        "generate_residue_pairs",
        {"n_per_template": 300},
        35,
    ),
    (
        "trans. order.",
        "data.concept_datasets.transitive_ordering_dataset",
        "generate_ordering_pairs",
        {"n_per_template": 200},
        50,
    ),
    (
        "triangle ineq.",
        "data.concept_datasets.triangle_inequality_dataset",
        "generate_triangle_pairs",
        {"n_per_template": 100},
        45,
    ),
    (
        "perf. square",
        "data.concept_datasets.perfect_square_dataset",
        "generate_perfect_square_pairs",
        {"n_per_template": 100},
        30,
    ),
    (
        "dec. term.",
        "data.concept_datasets.decimal_termination_dataset",
        "generate_decimal_pairs",
        {"n_per_template": 80},
        42,
    ),
    (
        "geom. series",
        "data.concept_datasets.geometric_series_dataset",
        "generate_geometric_pairs",
        {"n_per_template": 80},
        55,
    ),
    (
        "dot product",
        "data.concept_datasets.dot_product_sign_dataset",
        "generate_dot_pairs",
        {"n_per_template": 100},
        55,
    ),
    (
        "conservation",
        "data.concept_datasets.conservation_dataset",
        "generate_conservation_pairs",
        {"n_per_template": 200},
        60,
    ),
    (
        "momentum",
        "data.concept_datasets.momentum_conservation_dataset",
        "generate_momentum_pairs",
        {"n_per_template": 100},
        70,
    ),
    (
        "doppler",
        "data.concept_datasets.doppler_shift_dataset",
        "generate_doppler_pairs",
        {"n_per_template": 60},
        50,
    ),
    (
        "wave interf.",
        "data.concept_datasets.wave_interference_dataset",
        "generate_wave_pairs",
        {"n_per_template": 100},
        50,
    ),
    (
        "causal dir.",
        "data.concept_datasets.causal_direction_dataset",
        "generate_causal_pairs",
        {"n_per_template": 200},
        35,
    ),
    (
        "neg. scope",
        "data.concept_datasets.negation_scope_dataset",
        "generate_negation_pairs",
        {"n_per_template": 200},
        45,
    ),
    (
        "syllogism",
        "data.concept_datasets.syllogism_dataset",
        "generate_syllogism_pairs",
        {"n_per_template": 45},
        65,
    ),
    (
        "bal. paren.",
        "data.concept_datasets.balanced_parentheses_dataset",
        "generate_parentheses_pairs",
        {"n_per_template": 100},
        45,
    ),
]


def _pick_one(
    concept_idx: int, mod: str, fn: str, kw: dict, max_len: int, seed: int = 42
) -> tuple[str, str, str]:
    """Return (pos, neg, template_raw) for one representative pair.

    Template type rotates T0→T1→T2 across concept index; falls back to
    adjacent templates if the shortest pair still exceeds max_len.
    """
    m = __import__(mod, fromlist=[fn, "TEMPLATES"])
    pairs = getattr(m, fn)(**kw, seed=seed)
    templates_dict = m.TEMPLATES
    templates = [t for t in ("T0", "T1", "T2") if t in templates_dict]

    for k in range(len(templates)):
        t = templates[(concept_idx + k) % len(templates)]
        candidates = [p for p in pairs if p.template == t]
        if not candidates:
            continue
        candidates.sort(key=lambda p: len(p.prompt_pos))
        p = candidates[0]
        if len(p.prompt_pos.rstrip()) <= max_len:
            return p.prompt_pos.rstrip(), p.prompt_neg.rstrip(), templates_dict[t][0].rstrip()

    # Last resort: globally shortest
    pairs.sort(key=lambda p: len(p.prompt_pos))
    p = pairs[0]
    return p.prompt_pos.rstrip(), p.prompt_neg.rstrip(), templates_dict[p.template][0].rstrip()


# ── LaTeX table ───────────────────────────────────────────────────────────────

_PREAMBLE = r"""% Auto-generated by plot_dataset_table.py
% Requires: \usepackage{booktabs, array, longtable} in preamble
\small
\setlength{\tabcolsep}{5pt}
\renewcommand{\arraystretch}{1.3}
\begin{longtable}{@{} l p{4.2cm} p{7.2cm} @{}}
  \caption{%
    One representative contrastive pair per concept.
    The \textbf{Template} column shows the raw prompt pattern with placeholder
    variables; the \textbf{Pairs} column shows the positive instance (top line)
    and the negative instance (bottom line), with \textbf{bold} marking the
    tokens that differ between them.
    Template forms are varied across rows: T0 uses compact symbolic notation,
    T1 a natural-language description, and T2 an interrogative phrasing.
  }
  \label{tab:concept_datasets} \\
  \toprule
  \textbf{Concept} & \textbf{Template} & \textbf{Pairs (pos / neg)} \\
  \midrule
  \endfirsthead
  \multicolumn{3}{l}{\small\itshape \tablename~\thetable{} (continued)} \\[2pt]
  \toprule
  \textbf{Concept} & \textbf{Template} & \textbf{Pairs (pos / neg)} \\
  \midrule
  \endhead
  \midrule
  \multicolumn{3}{r}{\small\itshape continued on next page} \\
  \endfoot
  \bottomrule
  \endlastfoot
"""

_POSTAMBLE = r"""\end{longtable}
"""


def generate_table(out_path: Path, seed: int = 42) -> None:
    lines: list[str] = [_PREAMBLE]

    for idx, (display_name, mod, fn, kw, max_len) in enumerate(CONCEPTS):
        pos_raw, neg_raw, tmpl_raw = _pick_one(idx, mod, fn, kw, max_len, seed)
        pos_tex, neg_tex = diff_plain(pos_raw, neg_raw)

        tmpl_tex = r"\texttt{" + _esc(tmpl_raw) + "}"
        pairs_cell = pos_tex + r"\newline " + neg_tex
        name_cell = rf"\textsc{{{_esc(display_name)}}}"

        line = f"  {name_cell} & {tmpl_tex} & {pairs_cell} \\\\"
        lines.append(line + "\n")

        if idx < len(CONCEPTS) - 1:
            lines.append("  \\addlinespace[4pt]\n")

    lines.append(_POSTAMBLE)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(lines))
    print(f"Saved → {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="runs/concept_localization")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = _REPO_ROOT / args.out_dir / "dataset_table.tex"
    generate_table(out, seed=args.seed)
