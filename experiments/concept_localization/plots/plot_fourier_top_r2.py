"""Plot top-R² features from a fourier_sweep JSON.

Usage:
    python -m experiments.concept_localization.plots.plot_fourier_top_r2 \
        --json runs/.../fourier_sweep_top30_bal30.json \
        [--top_n 12]
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--json", required=True, type=Path)
    p.add_argument("--top_n", type=int, default=12)
    p.add_argument("--per_page", type=int, default=6)
    p.add_argument("--min_range", type=float, default=0.05,
                   help="Minimum (max-min) of mean activations; filters nearly-flat features")
    p.add_argument("--min_r2", type=float, default=0.85,
                   help="Minimum dominant R² to include")
    args = p.parse_args()

    data = json.loads(args.json.read_text())
    features = sorted(data["features"], key=lambda r: r["dominant_r2"], reverse=True)
    before = len(features)
    features = [r for r in features
                if (max(r["v"]) - min(r["v"])) >= args.min_range
                and r["dominant_r2"] >= args.min_r2]
    if len(features) < before:
        print(f"  filtered {before - len(features)} features (range < {args.min_range} or R² < {args.min_r2})")
    features = features[: args.top_n]
    if not features:
        print("  no features survive filters — skipping")
        return

    out = args.json.with_name(args.json.stem + "_top_r2.pdf")
    chunks = [features[i: i + args.per_page] for i in range(0, len(features), args.per_page)]

    with PdfPages(out) as pdf:
        for chunk in chunks:
            ncols = len(chunk)
            fig, axes = plt.subplots(1, ncols, figsize=(ncols * 2.8, 3.2),
                                     constrained_layout=True)
            if ncols == 1:
                axes = [axes]

            for ax, rec in zip(axes, chunk):
                modulus = rec["modulus"]
                v = np.array(rec["v"])
                k = rec["dominant_k"]
                h = rec["harmonics"][str(k)]
                amp = h["amplitude"]
                phi = np.radians(h["phase_deg"])
                dc = rec["dc"]

                r_vals = np.arange(modulus)
                r_fine = np.linspace(0, modulus - 1, 300)
                curve = dc + amp * np.cos(2 * np.pi * k * r_fine / modulus + phi)

                color = "#8B7CB8"
                ax.bar(r_vals, v, color=color, alpha=0.7, width=0.7)
                ax.plot(r_fine, curve, color="#C0444A", lw=2.0,
                        label=rf"$k={k}$, $R^2={rec['dominant_r2']:.2f}$")
                ax.axhline(dc, color="#888", lw=0.8, ls="--", alpha=0.6)
                ax.set_xticks(r_vals)
                ax.set_xlabel(r"$a \,\mathrm{mod}\, p$", fontsize=8)
                ax.set_ylabel("mean act.", fontsize=8)
                side = rec.get("side", "")
                score = rec.get("score") or 0.0
                ax.set_title(
                    f"{rec['feature']}  ({side}, {score:+.3f})\n"
                    rf"$k={k}$, $R^2={rec['dominant_r2']:.3f}$",
                    fontsize=7.5,
                )
                ax.legend(fontsize=6, loc="upper right")
                ax.tick_params(labelsize=7)
                ax.grid(axis="y", alpha=0.25)

            fig.suptitle(f"Top-R² Fourier features  (mod {features[0]['modulus']})", fontsize=9)
            pdf.savefig(fig)
            plt.close(fig)

    print(f"saved {out}")


if __name__ == "__main__":
    main()
