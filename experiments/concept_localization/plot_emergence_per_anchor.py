"""Per-anchor emergence plot for any concept with an emergence.npy file.

For each token-position anchor recorded by make_gif.py, plots raw ||delta|| (left axis)
and the double-normalised curve — act-norm further divided by its per-anchor peak — (right
axis).  The top-3 anchors by non-monotonicity (window behaviour) are highlighted in red.

Consecutive zero-signal anchors at the start of the sequence are collapsed into a single
summary subplot labelled with the joined prefix string.

Usage
-----
    python experiments/concept_localization/plot_emergence_per_anchor.py --concept doppler_shift
    python experiments/concept_localization/plot_emergence_per_anchor.py --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import experiments.plot_style as ps

BASE = _REPO_ROOT / "runs" / "concept_localization"


def _non_monotonicity(curve: np.ndarray) -> float:
    """Prominence of the most prominent local peak.

    For each interior position i, computes min(rise to i, fall from i), where
    rise = c_i - min(c[:i]) and fall = c_i - min(c[i+1:]).  Returns the maximum
    across all positions.  A monotone curve scores near zero; a sharp bump that
    both climbs and descends significantly scores high.
    """
    best = 0.0
    for i in range(1, len(curve) - 1):
        rise = curve[i] - curve[:i].min()
        fall = curve[i] - curve[i + 1:].min()
        if rise > 0 and fall > 0:
            best = max(best, min(rise, fall))
    return float(best)


def _build_slots(norms_raw, labels):
    """Return a list of plot slots.

    Each slot is either:
      ("prefix", prefix_label, zero_indices)  — collapsed zero-signal prefix
      ("anchor", idx)                          — individual active anchor
    """
    n = norms_raw.shape[0]
    # Find the leading run of zero-signal anchors
    first_active = next((i for i in range(n) if norms_raw[i].max() > 1e-8), n)

    slots = []
    if first_active > 0:
        prefix_label = "".join(labels[i] for i in range(first_active))
        slots.append(("prefix", prefix_label, list(range(first_active))))
    for i in range(first_active, n):
        slots.append(("anchor", i))
    return slots


def _draw_anchor_subplot(ax, layers, norms_raw_i, act_normed_i, title, highlight=False):
    l1, = ax.plot(layers, norms_raw_i, color=ps.VIOLET, lw=1.6, label="raw ‖δ‖")
    ax.set_ylabel("raw ‖δ‖", fontsize=7, color=ps.VIOLET)
    ax.tick_params(axis="y", labelcolor=ps.VIOLET, labelsize=7)
    ax.set_ylim(bottom=0)

    ax2 = ax.twinx()
    l3, = ax2.plot(layers, act_normed_i, color=ps.TEAL, lw=1.4, ls=":", label="double-norm")
    ax2.set_ylim(bottom=0)
    ax2.tick_params(axis="y", labelcolor=ps.TEAL, labelsize=7)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_edgecolor(ps.TEAL)
    ax2.set_ylabel("double-norm", fontsize=7, color=ps.TEAL)

    title_kw = dict(fontsize=9, pad=4)
    if highlight:
        title_kw.update(color=ps.RED, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_edgecolor(ps.RED)
            spine.set_linewidth(1.8)
        for spine in ax2.spines.values():
            spine.set_edgecolor(ps.RED)
            spine.set_linewidth(1.8)
    ax.set_title(title, **title_kw)
    return l1, l3


def plot_emergence_per_anchor(concept: str) -> Path | None:
    path = BASE / concept / "emergence.npy"
    if not path.exists():
        print(f"  [{concept}] emergence.npy not found — skipping")
        return None

    d           = np.load(path, allow_pickle=True).item()
    norms_raw   = d["norms_raw"]
    act_raw     = d["act_norms_raw"]
    layers      = np.array(d["layers"])
    labels      = d.get("token_labels_pos", [str(i) for i in range(norms_raw.shape[0])])
    labels_neg  = d.get("token_labels_neg", labels)
    n_anchors   = norms_raw.shape[0]
    template_key = d.get("template", "T0")

    # Try to load the pos template string from the dataset module
    template_str: str | None = None
    try:
        import importlib
        mod = importlib.import_module(f"data.concept_datasets.{concept}_dataset")
        templates = getattr(mod, "TEMPLATES", {})
        if template_key in templates:
            template_str = templates[template_key][0]
    except Exception:
        pass

    if template_str is None:
        template_str = "".join(str(t) for t in labels)

    # Find which tokens correspond to each template variable {var} by regex-matching
    # the template against the actual prompt (joined token labels).
    import re as _re
    actual_prompt = "".join(labels)
    _parts = _re.split(r'\{(\w+)\}', template_str)  # alternating literal / var_name

    # Use unique internal group names (_v0, _v1, …) to avoid re's duplicate-group error
    _group_map: list[str] = []  # group index → variable name
    _pattern_parts: list[str] = []
    for i, p in enumerate(_parts):
        if i % 2 == 0:
            _pattern_parts.append(_re.escape(p))
        else:
            _pattern_parts.append(f"(?P<_v{len(_group_map)}>.+?)")
            _group_map.append(p)
    _pattern = "".join(_pattern_parts) + "$"
    _m = _re.match(_pattern, actual_prompt, _re.DOTALL)

    var_token_positions: dict[str, list[int]] = {}
    if _m:
        char_pos = 0
        tok_char_ends = []
        for tok in labels:
            char_pos += len(tok)
            tok_char_ends.append(char_pos)
        tok_char_starts = [0] + tok_char_ends[:-1]

        for g_idx, var_name in enumerate(_group_map):
            try:
                cs, ce = _m.start(f"_v{g_idx}"), _m.end(f"_v{g_idx}")
                idxs = [i for i, (s, e) in enumerate(zip(tok_char_starts, tok_char_ends))
                        if s < ce and e > cs]
                var_token_positions.setdefault(var_name, [])
                var_token_positions[var_name].extend(i for i in idxs if i not in var_token_positions[var_name])
            except IndexError:
                pass

    all_var_tok_idxs = {i for idxs in var_token_positions.values() for i in idxs}
    # Build label mapping: tok_idx → var_name (first match)
    tok_to_var = {i: vn for vn, idxs in var_token_positions.items() for i in idxs}

    prompt_annotated = "".join(
        f"{{{tok}}}" if i in all_var_tok_idxs else tok
        for i, tok in enumerate(labels)
    )

    row_max    = act_raw.max(axis=1, keepdims=True).clip(min=1e-8)
    act_normed = act_raw / row_max

    active    = [i for i in range(n_anchors) if norms_raw[i].max() > 1e-8]
    non_mono  = {i: _non_monotonicity(act_normed[i]) for i in active}
    top3_idx  = sorted(active, key=lambda i: non_mono[i], reverse=True)[:3]
    ranks     = {idx: rank + 1 for rank, idx in enumerate(top3_idx)}

    slots = _build_slots(norms_raw, labels)
    n_slots = len(slots)

    NCOLS = 4
    NROWS = (n_slots + NCOLS - 1) // NCOLS

    ps.apply()
    fig, axes = plt.subplots(NROWS, NCOLS, figsize=(NCOLS * 3.2, NROWS * 2.6), sharex=True)
    if NROWS * NCOLS == 1:
        axes = np.array([[axes]])
    axes_flat = axes.flat

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=ps.VIOLET, lw=1.6, label="raw ‖δ‖"),
        Line2D([0], [0], color=ps.TEAL,   lw=1.4, ls=":", label="double-norm"),
    ]

    for slot_idx, slot in enumerate(slots):
        ax = axes_flat[slot_idx]
        ax.set_xlim(layers[0], layers[-1])
        if slot_idx >= (NROWS - 1) * NCOLS:
            ax.set_xlabel("layer", fontsize=8)

        if slot[0] == "prefix":
            _, prefix_label, _ = slot
            ax.plot(layers, np.zeros_like(layers, dtype=float), color=ps.GRAY, lw=1.0, ls="--")
            ax.set_ylim(0, 1)
            ax.set_title(f"prefix  '{prefix_label}'", fontsize=8, pad=4, color=ps.GRAY)
            ax.tick_params(axis="y", labelsize=7)
            ax2 = ax.twinx()
            ax2.set_ylim(0, 1)
            ax2.tick_params(axis="y", labelcolor=ps.TEAL, labelsize=7)
            ax2.spines["right"].set_visible(True)
            ax2.spines["right"].set_edgecolor(ps.TEAL)
            ax2.set_ylabel("double-norm", fontsize=7, color=ps.TEAL)
        else:
            _, idx = slot
            label = repr(labels[idx]) if idx < len(labels) else str(idx)
            is_top = idx in ranks
            nm = non_mono.get(idx, 0.0)
            title = (
                f"#{ranks[idx]}  pos {idx}  {label}  nm={nm:.2f}"
                if is_top else f"pos {idx}  {label}  nm={nm:.2f}"
            )
            _draw_anchor_subplot(ax, layers, norms_raw[idx], act_normed[idx], title, highlight=is_top)

        if slot_idx == 0:
            ax.legend(handles=legend_handles, fontsize=7, loc="upper left")

    for slot_idx in range(n_slots, NROWS * NCOLS):
        axes_flat[slot_idx].set_visible(False)

    # Reserve top margin for three header lines; tight_layout fills the rest
    fig.tight_layout(rect=[0, 0, 1, 0.88])

    fig.text(0.5, 0.995, f"{concept} — delta norm per anchor  (top-3 by non-monotonicity highlighted)",
             ha="center", va="top", fontsize=10, fontweight="bold", transform=fig.transFigure)
    fig.text(0.5, 0.965, f"template {template_key}:  {template_str}",
             ha="center", va="top", fontsize=8, color=ps.GRAY, style="italic",
             transform=fig.transFigure)
    fig.text(0.5, 0.938, f"prompt:  {prompt_annotated}  — bracketed tokens are variables",
             ha="center", va="top", fontsize=8, color=ps.NAVY,
             transform=fig.transFigure)

    out = BASE / concept / "emergence_per_anchor.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [{concept}] saved → {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--concept", help="Single concept name")
    group.add_argument("--all", action="store_true", help="Run for every concept with emergence.npy")
    args = parser.parse_args()

    if args.all:
        concepts = sorted(p.parent.name for p in BASE.glob("*/emergence.npy"))
        print(f"Found {len(concepts)} concepts with emergence.npy")
        for concept in concepts:
            plot_emergence_per_anchor(concept)
    else:
        plot_emergence_per_anchor(args.concept)


if __name__ == "__main__":
    main()
