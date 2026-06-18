"""Generate a GIF showing how a concept direction emerges token by token.

For each token position the delta norm per layer is plotted as a curve. Frames
are assembled into an animated GIF so the viewer can watch the concept
crystallise as the model reads each successive token.

Extraction is done by calling extract_layer_deltas_generic with an explicit
integer anchor_mode for each position — no separate extraction module needed.

Usage
-----
    python -m experiments.concept_localization.concept_emergence_gif.make_gif
    python -m experiments.concept_localization.concept_emergence_gif.make_gif \
        --concept carry --n 50 --out runs/concept_localization/carry/emergence.gif
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.concept_localization.extract_deltas_generic import (
    _find_delimiter_anchor,
    extract_layer_deltas_generic,
)
from experiments.concept_localization.pipeline.run_concept import _load_concept
from experiments.concept_localization.plots.plot_emergence_per_anchor import (
    plot_anchor_layer_grid,
    plot_emergence_per_anchor,
)
from experiments.plot_style import GRAY, TEAL, VIOLET, apply

_GREEN = "#27ae60"
_STEP_COLOR = "#D4A017"   # amber — steepest-step markers

from mechinterp_qwen3.attribution_model import AttributionModel
from mechinterp_qwen3.utils.hf_utils import load_transcoder_from_hub
from mechinterp_qwen3.utils.model_utils import get_default_device, parse_dtype

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("make_gif")

_MODEL = "Qwen/Qwen3-4B"
_TRANSCODER_SET = "mwhanna/qwen3-4b-transcoders"


def _decode_tokens(tokenizer, ids: list[int]) -> list[str]:
    return [
        tokenizer.convert_tokens_to_string([tokenizer.convert_ids_to_tokens(i)])
        for i in ids
    ]


def _detect_step_layers(normed: np.ndarray, n_top: int = 2, smooth: int = 2) -> list[int]:
    """Layer indices (transition i → i+1) of the n_top steepest local gradient peaks."""
    grad = np.diff(normed)
    if smooth > 1:
        kernel = np.ones(smooth) / smooth
        grad = np.convolve(grad, kernel, mode="same")
    # Local maxima of the gradient (peaks)
    peaks = [
        (grad[i], i)
        for i in range(1, len(grad) - 1)
        if grad[i] >= grad[i - 1] and grad[i] > grad[i + 1] and grad[i] > 0
    ]
    if not peaks:
        top = sorted(np.argsort(grad)[-n_top:].tolist())
        return top
    peaks.sort(reverse=True)
    return sorted(i for _, i in peaks[:n_top])


def _detect_phases(
    normed: np.ndarray, min_frac: float = 0.28, merge_gap: int = 4
) -> list[int]:
    """Return layer indices of significant gradient peaks (phase boundaries).

    Each returned index i means the steepest rise is at the transition i → i+1.
    Nearby peaks within merge_gap layers are merged to their centroid.
    """
    grad = np.diff(normed)
    threshold = grad.max() * min_frac
    peaks = [
        i
        for i in range(1, len(grad) - 1)
        if grad[i] >= grad[i - 1] and grad[i] > grad[i + 1] and grad[i] > threshold
    ]
    if not peaks:
        return []
    merged: list[int] = [peaks[0]]
    for p in peaks[1:]:
        if p - merged[-1] <= merge_gap:
            merged[-1] = (merged[-1] + p) // 2
        else:
            merged.append(p)
    return merged


def _render_frame(
    ax: plt.Axes,
    layers: list[int],
    norms_at_pos: np.ndarray,
    act_norms_at_pos: np.ndarray,
    current_pos: int,
    n_pairs: int,
    token_labels_pos: list[str],
    token_labels_neg: list[str],
    concept: str,
    n_layers: int,
    template: str | None = None,
) -> None:
    ax.cla()

    for p in range(current_pos):
        alpha = 0.12 + 0.25 * (p / max(current_pos, 1))
        ax.plot(layers, norms_at_pos[p], color=GRAY, lw=0.8, alpha=alpha)

    purple = norms_at_pos[current_pos]
    ax.plot(layers, purple, color=VIOLET, lw=2.4, zorder=5,
            label=r"$\|\delta_l\| / \max_l(\|\delta_l\|)$  (mean over pairs)")
    ax.plot(layers, act_norms_at_pos[current_pos], color=_GREEN, lw=1.8,
            ls="--", zorder=6,
            label=r"$(\|\delta_l\| / \mathbb{E}\|\mathbf{h}_l\|) / \max_l(\|\delta_l\| / \mathbb{E}\|\mathbf{h}_l\|)$ (mean over pairs)")

    # --- Steepest-step and phase annotations (purple line only) ---
    if purple.max() > 1e-10:
        top_steps = _detect_step_layers(purple, n_top=2)
        phase_bounds = _detect_phases(purple)
        n_phases = len(phase_bounds) + 1

        # Shade phases with alternating very-light fills
        phase_edges = [0] + [b + 1 for b in phase_bounds] + [n_layers]
        phase_alphas = [0.06, 0.0, 0.06, 0.0, 0.06]
        for k in range(len(phase_edges) - 1):
            lo, hi = phase_edges[k], phase_edges[k + 1]
            ax.axvspan(lo - 0.5, hi - 0.5, color=VIOLET,
                       alpha=phase_alphas[k % len(phase_alphas)], lw=0, zorder=0)

        # Mark top-2 steepest transitions
        grad = np.diff(purple)
        for rank, step_l in enumerate(top_steps):
            step_size = float(grad[step_l])
            x_mid = step_l + 0.5
            y_mid = float((purple[step_l] + purple[step_l + 1]) / 2)
            ax.axvline(x_mid, color=_STEP_COLOR, lw=1.4, ls="--", alpha=0.85, zorder=4)
            label_y = min(y_mid + 0.08, 1.05)
            ax.text(x_mid, label_y,
                    f"#{rank + 1}  +{step_size:.3f}",
                    ha="center", va="bottom", fontsize=7, color=_STEP_COLOR,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                              edgecolor=_STEP_COLOR, alpha=0.85, lw=0.6))

        # Phase count in the bottom-right corner
        ax.text(0.98, 0.03, f"{n_phases} phase{'s' if n_phases != 1 else ''}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, color=GRAY, style="italic")

    tok_str = token_labels_pos[current_pos]
    consumed_pos = "".join(token_labels_pos[: current_pos + 1])
    consumed_neg = "".join(token_labels_neg[: current_pos + 1])
    tmpl_str = f"template {template}" if template else "all templates"
    ax.set_title(
        f"{concept} — anchor at token {current_pos}: {tok_str!r}  "
        f"({n_pairs} pairs, {tmpl_str})\n"
        f'pos (example): "{consumed_pos}"    neg (example): "{consumed_neg}"',
        fontsize=9,
        pad=6,
    )
    ax.set_xlabel("Layer", fontsize=10)
    ax.set_ylabel("Normalised to [0, 1]", fontsize=10)
    ax.set_xlim(-0.5, n_layers - 0.5)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xticks(range(0, n_layers, 5))
    ax.legend(fontsize=8, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def make_emergence_gif(
    concept: str,
    model_name: str,
    transcoder_set: str,
    n_per_template: int,
    out_path: Path,
    dtype_str: str = "bfloat16",
    seed: int = 42,
    fps: int = 4,
    template: str | None = "T0",
    max_pairs: int | None = None,
) -> None:
    # The GIF iterates over fixed integer token positions (0 … delimiter_pos), which
    # requires every pair to share the same tokenization length.  Because templates
    # differ in prefix length, mixing templates would force aggressive length-filtering
    # that discards most pairs.  A single template is therefore used by default (T0).
    # To compare across templates, run the GIF separately per template.
    device = get_default_device()
    dtype = parse_dtype(dtype_str)

    log.info("Loading pairs for concept '%s'", concept)
    all_pairs = _load_concept(concept, n_per_template, seed)
    if template is not None:
        all_pairs = [p for p in all_pairs if p.template == template]
        log.info("Filtered to template %s: %d pairs", template, len(all_pairs))
    pairs = all_pairs[:max_pairs] if max_pairs else all_pairs

    log.info("Loading model %s", model_name)
    transcoder_set_obj, _ = load_transcoder_from_hub(
        transcoder_set, dtype=dtype, lazy_encoder=True, lazy_decoder=True
    )
    model = AttributionModel.from_pretrained_and_transcoders(
        model_name, transcoder_set_obj, dtype=dtype, device=device
    )
    model.eval()

    n_layers = model.cfg.n_layers
    layers = list(range(n_layers))

    example_ids = model.tokenizer(pairs[0].prompt_pos, add_special_tokens=False).input_ids
    example_ids_neg = model.tokenizer(pairs[0].prompt_neg, add_special_tokens=False).input_ids
    seq_len = len(example_ids)
    token_labels_pos = _decode_tokens(model.tokenizer, example_ids)
    token_labels_neg = _decode_tokens(model.tokenizer, example_ids_neg)

    # Stop at the last structural delimiter ('=', ':', '?', ')') — the natural anchor
    delimiter_pos = _find_delimiter_anchor(example_ids, model.tokenizer)
    log.info("Delimiter anchor at position %d: %r", delimiter_pos, token_labels_pos[delimiter_pos])

    # Keep only pairs whose tokenization length matches pairs[0].  Within a single
    # template, operands of different digit-counts still tokenize differently, so
    # this further restricts to one digit-size class.  The result is a homogeneous
    # set where every fixed integer position refers to the same token.
    pairs = [
        p for p in pairs
        if len(model.tokenizer(p.prompt_pos, add_special_tokens=False).input_ids) == seq_len
    ]
    n_template_total = len(all_pairs)
    log.info(
        "Token sequence (%d tokens): %s — using %d/%d length-matched pairs (template=%s)",
        seq_len, token_labels_pos, len(pairs), n_template_total, template,
    )

    n_frames = delimiter_pos + 1
    norms_raw = np.zeros((n_frames, n_layers))
    act_norms_raw = np.zeros((n_frames, n_layers))
    mean_cos = np.zeros(n_frames)   # mean pairwise cosine across layers, per anchor pos
    for pos in range(n_frames):
        log.info(
            "Extracting deltas at position %d/%d (%r)...",
            pos, seq_len - 1, token_labels_pos[pos],
        )
        results = extract_layer_deltas_generic(
            model, pairs, layers, device, dtype,
            per_template=False,
            anchor_mode=str(pos),
        )
        ld = results["all"]
        vecs = []
        for li, l in enumerate(layers):
            if l in ld.delta:
                v = ld.delta[l].float()
                raw = v.norm().item()
                norms_raw[pos, li] = raw
                scale = ld.mean_act_norm.get(l, 1.0) if ld.mean_act_norm else 1.0
                act_norms_raw[pos, li] = raw / scale if scale > 0 else raw
                if raw > 1e-8:
                    vecs.append(torch.nn.functional.normalize(v.unsqueeze(0), dim=-1))
        if len(vecs) >= 2:
            mat = torch.cat(vecs, dim=0)          # (L, d)
            cos_mat = mat @ mat.T                  # (L, L)
            n = cos_mat.shape[0]
            idx = torch.triu_indices(n, n, offset=1)
            mean_cos[pos] = cos_mat[idx[0], idx[1]].mean().item()

    row_max = norms_raw.max(axis=1, keepdims=True).clip(min=1e-8)
    norms = norms_raw / row_max

    row_max_act = act_norms_raw.max(axis=1, keepdims=True).clip(min=1e-8)
    act_norms = act_norms_raw / row_max_act

    apply()
    frames: list[Image.Image] = []
    fig, ax = plt.subplots(figsize=(5, 5))

    for pos in range(n_frames):
        _render_frame(ax, layers, norms, act_norms, pos, len(pairs), token_labels_pos, token_labels_neg, concept, n_layers, template)
        fig.tight_layout()
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        frames.append(Image.fromarray(buf[:, :, :3]))

    plt.close(fig)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save raw norms so downstream tests can compare against concept localization results
    # without reloading the model.
    norms_data = {
        "norms_raw": norms_raw,        # (n_frames, n_layers) — raw ‖δ_l‖ per anchor pos
        "act_norms_raw": act_norms_raw,
        "mean_cos": mean_cos,          # (n_frames,) — mean pairwise cosine across layers
        "delimiter_pos": delimiter_pos,
        "n_pairs": len(pairs),
        "layers": layers,
        "token_labels_pos": token_labels_pos,
        "token_labels_neg": token_labels_neg,
        "concept": concept,
        "template": template,
    }
    np.save(out_path.with_suffix(".npy"), norms_data)
    log.info("Saved emergence norms → %s", out_path.with_suffix(".npy"))

    duration_ms = int(1000 / fps)
    durations = [duration_ms] * len(frames)
    durations[-1] = 7000  # hold delimiter frame for 7 s before looping
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=durations,
    )
    log.info("Saved GIF (%d frames, %d fps, 7 s hold at delimiter) → %s", len(frames), fps, out_path)

    log.info("Plotting emergence PDFs for concept=%s template=%s", concept, template)
    plot_emergence_per_anchor(concept)
    plot_anchor_layer_grid(concept, template=template or "T0")
    log.info("Emergence PDFs saved.")


def regen_gif_from_npy(npy_path: Path, out_path: Path, fps: int = 2) -> None:
    """Regenerate the GIF from a saved .npy file — no model loading required."""
    data = np.load(npy_path, allow_pickle=True).item()
    norms_raw = data["norms_raw"]
    act_norms_raw = data["act_norms_raw"]
    delimiter_pos = data["delimiter_pos"]
    n_pairs = data["n_pairs"]
    layers = data["layers"]
    token_labels_pos = data.get("token_labels_pos", [str(i) for i in range(norms_raw.shape[0] + 2)])
    token_labels_neg = data.get("token_labels_neg", token_labels_pos)
    concept = data.get("concept", npy_path.parent.name)
    template = data.get("template", None)

    n_layers = len(layers)
    n_frames = norms_raw.shape[0]

    row_max = norms_raw.max(axis=1, keepdims=True).clip(min=1e-8)
    norms = norms_raw / row_max
    row_max_act = act_norms_raw.max(axis=1, keepdims=True).clip(min=1e-8)
    act_norms = act_norms_raw / row_max_act

    apply()
    frames: list[Image.Image] = []
    fig, ax = plt.subplots(figsize=(5, 5))
    for pos in range(n_frames):
        _render_frame(ax, layers, norms, act_norms, pos, n_pairs,
                      token_labels_pos, token_labels_neg, concept, n_layers, template)
        fig.tight_layout()
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        frames.append(Image.fromarray(buf[:, :, :3]))
    plt.close(fig)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = int(1000 / fps)
    durations = [duration_ms] * len(frames)
    durations[-1] = 7000
    frames[0].save(out_path, save_all=True, append_images=frames[1:], loop=0, duration=durations)
    log.info("Regenerated GIF (%d frames) → %s", len(frames), out_path)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--concept", default="carry")
    parser.add_argument("--model", default=_MODEL)
    parser.add_argument("--transcoder_set", default=_TRANSCODER_SET)
    parser.add_argument("--n", type=int, default=50, help="Pairs per template")
    parser.add_argument("--template", default="T0",
                        help="Template to use (default T0). Pass '' to use all templates "
                             "(triggers aggressive length filtering — most pairs will be dropped).")
    parser.add_argument("--max_pairs", type=int, default=None)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        default=None,
        help="Output GIF path (default: runs/concept_localization/<concept>/emergence.gif)",
    )
    parser.add_argument(
        "--from_npy",
        default=None,
        metavar="PATH",
        help="Regenerate GIF from an existing .npy file without reloading the model.",
    )
    args = parser.parse_args()

    out = Path(args.out or f"runs/concept_localization/{args.concept}/emergence.gif")

    if args.from_npy:
        regen_gif_from_npy(Path(args.from_npy), out, fps=args.fps)
        return

    make_emergence_gif(
        concept=args.concept,
        model_name=args.model,
        transcoder_set=args.transcoder_set,
        n_per_template=args.n,
        out_path=out,
        dtype_str=args.dtype,
        seed=args.seed,
        fps=args.fps,
        template=args.template or None,
        max_pairs=args.max_pairs,
    )


if __name__ == "__main__":
    main()
