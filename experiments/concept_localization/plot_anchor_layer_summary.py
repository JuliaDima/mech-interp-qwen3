"""Combined per-anchor layer summary plot.

Reads an anchor run directory containing results.json, deltas.pt, and optionally
null/null_permutation.json, then writes a single vertically stacked figure:

  1. E_dec feature projection scatter
  2. Raw and activation-normalised delta trajectory
  3. Cross-layer delta cosine heatmap
  4. Null permutation comparison
  5. Causal overlay
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import experiments.plot_style as ps


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _load_deltas(anchor_dir: Path, template: str) -> dict[int, torch.Tensor]:
    raw = torch.load(anchor_dir / "deltas.pt", map_location="cpu", weights_only=False)
    if template in raw:
        return raw[template]
    return raw.get("all", {})


def _layer_ticks(layers: list[int]) -> list[int]:
    if not layers:
        return []
    return list(range(min(layers), max(layers) + 1, 2))


def _peak_norm(vals: list[float]) -> list[float]:
    peak = max(abs(v) for v in vals) if vals else 1.0
    return [v / peak if peak > 1e-12 else 0.0 for v in vals]


def _plot_feature_projection(ax, results: dict, layers: list[int], top_k: int) -> None:
    xs, ys, cs = [], [], []
    for layer_s, rows in results.get("top_features_by_layer", {}).items():
        layer = int(layer_s)
        for row in rows[:top_k]:
            xs.append(layer)
            ys.append(int(row["feature_id"]))
            cs.append(float(row.get("cos_sim", 0.0)))

    if xs:
        vmax = max(abs(v) for v in cs) or 1.0
        sc = ax.scatter(
            xs,
            ys,
            c=cs,
            cmap=ps.CMAP_DIV,
            vmin=-vmax,
            vmax=vmax,
            s=28,
            alpha=0.85,
        )
        cb = plt.colorbar(sc, ax=ax, pad=0.01, fraction=0.025)
        cb.set_label("E_dec cos", fontsize=8)
        cb.ax.tick_params(labelsize=7, length=0)
    else:
        ax.text(0.5, 0.5, "No feature projections", ha="center", va="center", transform=ax.transAxes)

    ax.set_ylabel("feature id")
    ax.set_title("Feature projection scatter: top E_dec-aligned features per layer", fontsize=10, pad=4)
    ax.set_xlim(min(layers) - 0.5, max(layers) + 0.5)


def _plot_delta_trajectory(ax, deltas: dict[int, torch.Tensor], results: dict, layers: list[int]) -> None:
    raw = [float(deltas[l].norm().item()) if l in deltas else 0.0 for l in layers]
    ax.plot(layers, _peak_norm(raw), color=ps.NAVY, lw=2.0, label="raw norm / peak")

    mean_act = {int(k): float(v) for k, v in results.get("mean_act_norm", {}).items()}
    if mean_act:
        act_norm = [raw_i / mean_act.get(l, 1.0) for raw_i, l in zip(raw, layers, strict=False)]
        ax.plot(
            layers,
            _peak_norm(act_norm),
            color=ps.TEAL,
            lw=1.8,
            ls="--",
            label="act-normalised norm / peak",
        )

    peak_layer = int(results.get("sharpness", {}).get("peak_layer", layers[int(np.argmax(raw))]))
    ax.axvline(peak_layer, color=ps.VIOLET, lw=0.9, ls=":", alpha=0.8)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("norm")
    ax.set_title("Delta trajectory: raw and activation-normalised", fontsize=10, pad=4)
    ax.legend(fontsize=8, loc="upper left")


def _plot_layer_cosine(ax, deltas: dict[int, torch.Tensor], layers: list[int]) -> None:
    mat = np.full((len(layers), len(layers)), np.nan, dtype=float)
    for i, li in enumerate(layers):
        if li not in deltas:
            continue
        ai = deltas[li].float().unsqueeze(0)
        for j, lj in enumerate(layers):
            if lj not in deltas:
                continue
            mat[i, j] = F.cosine_similarity(ai, deltas[lj].float().unsqueeze(0)).item()

    im = ax.imshow(
        mat,
        origin="lower",
        aspect="auto",
        cmap=ps.CMAP_DIV,
        vmin=-1,
        vmax=1,
        extent=[min(layers) - 0.5, max(layers) + 0.5, min(layers) - 0.5, max(layers) + 0.5],
    )
    cb = plt.colorbar(im, ax=ax, pad=0.01, fraction=0.025)
    cb.set_label("cos", fontsize=8)
    cb.ax.tick_params(labelsize=7, length=0)
    ax.set_ylabel("layer")
    ax.set_title("Layer cosine similarity: cos(delta_i, delta_j)", fontsize=10, pad=4)


def _plot_null(ax, null: dict | None, layers: list[int]) -> None:
    if not null:
        ax.text(0.5, 0.5, "No null permutation results", ha="center", va="center", transform=ax.transAxes)
        ax.set_ylabel("null")
        ax.set_title("Null hypothesis", fontsize=10, pad=4)
        return

    nlayers = [int(x) for x in null.get("layers", layers)]
    real = np.asarray(null.get("real_norms", []), dtype=float)
    nulls = np.asarray(null.get("null_norms", []), dtype=float)
    if real.size == 0 or nulls.size == 0:
        ax.text(0.5, 0.5, "Null results incomplete", ha="center", va="center", transform=ax.transAxes)
        return

    mean = nulls.mean(axis=0)
    std = nulls.std(axis=0)
    p95 = np.percentile(nulls, 95, axis=0)
    for row in nulls:
        ax.plot(nlayers, row, color=ps.GRAY, lw=0.45, alpha=0.18)
    ax.fill_between(nlayers, mean - std, mean + std, color=ps.GRAY, alpha=0.20, label="null mean +/- sd")
    ax.plot(nlayers, p95, color=ps.GRAY, lw=1.0, ls=":", label="null p95")
    ax.plot(nlayers, mean, color=ps.GRAY, lw=1.3)
    ax.plot(nlayers, real, color=ps.VIOLET, lw=2.1, label="real")
    ax.set_ylim(bottom=0)
    ax.set_ylabel("norm")
    ax.set_title("Null hypothesis: within-class permutation", fontsize=10, pad=4)
    ax.legend(fontsize=8, loc="upper left")


def _plot_causal(ax, results: dict, deltas: dict[int, torch.Tensor], template: str, layers: list[int]) -> None:
    causal = results.get("causal")
    if not causal:
        ax.text(0.5, 0.5, "No causal results", ha="center", va="center", transform=ax.transAxes)
        ax.set_ylabel("causal")
        ax.set_title("Causal overlay", fontsize=10, pad=4)
        return

    cs = causal.get(template) or causal.get("all")
    if not cs:
        ax.text(0.5, 0.5, f"No causal results for {template}", ha="center", va="center", transform=ax.transAxes)
        return

    patch = [float(cs.get("patching_mean", {}).get(str(l), 0.0)) for l in layers]
    grad = [float(cs.get("grad_dot_delta_mean", {}).get(str(l), 0.0)) for l in layers]
    delta = [float(deltas[l].norm().item()) if l in deltas else 0.0 for l in layers]

    ax.plot(layers, _peak_norm(delta), color=ps.NAVY, lw=1.8, label="delta norm / peak")
    ax.plot(layers, _peak_norm(patch), color=ps.VIOLET, lw=2.0, label="patching / peak")
    ax.plot(layers, _peak_norm(grad), color=ps.TEAL, lw=2.0, label="grad dot delta / peak")
    ax.axhline(0, color=ps.GRAY, lw=0.8, ls="--")
    ax.set_ylabel("signal")
    ax.set_title(f"Causal overlay: {template}", fontsize=10, pad=4)
    ax.legend(fontsize=8, loc="upper left")


def plot_anchor_layer_summary(anchor_dir: Path, template: str = "T0", out_path: Path | None = None) -> Path:
    results = _load_json(anchor_dir / "results.json")
    if results is None:
        raise FileNotFoundError(anchor_dir / "results.json")
    deltas = _load_deltas(anchor_dir, template)
    if not deltas:
        raise ValueError(f"No deltas found for template={template} in {anchor_dir / 'deltas.pt'}")

    layers = sorted(int(l) for l in deltas.keys())
    ticks = _layer_ticks(layers)
    null = _load_json(anchor_dir / "null" / "null_permutation.json")
    out_path = out_path or (anchor_dir / f"anchor_layer_summary_{template}.png")

    ps.apply()
    fig, axes = plt.subplots(
        5,
        1,
        figsize=(15, 16),
        sharex=False,
        gridspec_kw={"hspace": 0.28},
    )

    _plot_feature_projection(axes[0], results, layers, int(results.get("config", {}).get("top_k", 15)))
    _plot_delta_trajectory(axes[1], deltas, results, layers)
    _plot_layer_cosine(axes[2], deltas, layers)
    _plot_null(axes[3], null, layers)
    _plot_causal(axes[4], results, deltas, template, layers)

    for ax in axes:
        ax.set_xlim(min(layers) - 0.5, max(layers) + 0.5)
        ax.set_xticks(ticks)
        ax.set_xlabel("layer")
        ax.grid(axis="x", color=ps.GRAY, alpha=0.18, lw=0.6)
        ax.tick_params(axis="x", labelsize=8, length=0)
        ax.tick_params(axis="y", labelsize=8, length=0)

    cfg = results.get("config", {})
    fig.suptitle(
        f"{cfg.get('concept', anchor_dir.parent.name)} {template} | "
        f"anchor pos {cfg.get('anchor_pos', '?')} token {cfg.get('anchor_token', '?')!r}",
        fontsize=13,
        y=0.995,
    )
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--anchor_dir", required=True, type=Path)
    parser.add_argument("--template", default="T0")
    parser.add_argument("--out", default=None, type=Path)
    args = parser.parse_args()
    out = plot_anchor_layer_summary(args.anchor_dir, template=args.template, out_path=args.out)
    print(f"Saved anchor layer summary -> {out}")


if __name__ == "__main__":
    main()
