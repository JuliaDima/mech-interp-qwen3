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

from experiments.concept_localization.extract_deltas_generic import extract_layer_deltas_generic
from experiments.concept_localization.run_concept import _load_concept
from experiments.plot_style import GRAY, VIOLET, apply
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


def _render_frame(
    ax: plt.Axes,
    layers: list[int],
    norms_at_pos: np.ndarray,
    current_pos: int,
    token_labels: list[str],
    concept: str,
    n_layers: int,
) -> None:
    ax.cla()

    for p in range(current_pos):
        alpha = 0.12 + 0.25 * (p / max(current_pos, 1))
        ax.plot(layers, norms_at_pos[p], color=GRAY, lw=0.8, alpha=alpha)

    ax.plot(layers, norms_at_pos[current_pos], color=VIOLET, lw=2.4, zorder=5)

    tok_str = token_labels[current_pos]
    consumed = "".join(token_labels[: current_pos + 1])
    ax.set_title(
        f"{concept} — anchor at token {current_pos}: {tok_str!r}\n"
        f'consumed: "{consumed}"',
        fontsize=10,
        pad=6,
    )
    ax.set_xlabel("Layer", fontsize=10)
    ax.set_ylabel("‖δ‖ / max(‖δ‖)", fontsize=10)
    ax.set_xlim(-0.5, n_layers - 0.5)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xticks(range(0, n_layers, 5))
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
    template: str | None = None,
    max_pairs: int | None = None,
) -> None:
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
    seq_len = len(example_ids)
    token_labels = _decode_tokens(model.tokenizer, example_ids)
    log.info("Token sequence (%d tokens): %s", seq_len, token_labels)

    norms = np.zeros((seq_len, n_layers))
    for pos in range(seq_len):
        log.info(
            "Extracting deltas at position %d/%d (%r)...",
            pos, seq_len - 1, token_labels[pos],
        )
        results = extract_layer_deltas_generic(
            model, pairs, layers, device, dtype,
            per_template=False,
            anchor_mode=str(pos),
        )
        ld = results["all"]
        for li, l in enumerate(layers):
            if l in ld.delta:
                norms[pos, li] = ld.delta[l].norm().item()

    peak = norms.max()
    if peak > 1e-8:
        norms = norms / peak

    apply()
    frames: list[Image.Image] = []
    fig, ax = plt.subplots(figsize=(10, 4))

    for pos in range(seq_len):
        _render_frame(ax, layers, norms, pos, token_labels, concept, n_layers)
        fig.tight_layout()
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        frames.append(Image.fromarray(buf[:, :, :3]))

    plt.close(fig)

    frames += [frames[-1]] * fps

    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = int(1000 / fps)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=duration_ms,
    )
    log.info("Saved GIF (%d frames, %d fps) → %s", len(frames), fps, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--concept", default="carry")
    parser.add_argument("--model", default=_MODEL)
    parser.add_argument("--transcoder_set", default=_TRANSCODER_SET)
    parser.add_argument("--n", type=int, default=50, help="Pairs per template")
    parser.add_argument("--template", default=None, help="Restrict to one template, e.g. T0")
    parser.add_argument("--max_pairs", type=int, default=None)
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        default=None,
        help="Output GIF path (default: runs/concept_localization/<concept>/emergence.gif)",
    )
    args = parser.parse_args()

    out = Path(args.out or f"runs/concept_localization/{args.concept}/emergence.gif")

    make_emergence_gif(
        concept=args.concept,
        model_name=args.model,
        transcoder_set=args.transcoder_set,
        n_per_template=args.n,
        out_path=out,
        dtype_str=args.dtype,
        seed=args.seed,
        fps=args.fps,
        template=args.template,
        max_pairs=args.max_pairs,
    )


if __name__ == "__main__":
    main()
