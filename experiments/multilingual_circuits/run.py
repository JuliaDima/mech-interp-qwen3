"""Multilingual circuits intervention experiment.

Reproduces Anthropic's Biology of LLMs §5 antonym-completion experiments on
Qwen3-4B with transcoders, testing three independent components:

  1. Operation swap  — antonym features replaced by synonym features
  2. Operand swap    — "small" size features replaced by "hot" temperature features
  3. Language swap   — output-language features swapped across EN / FR / ZH

For each experiment the script sweeps an injection strength alpha from 0 to a
maximum value and records next-token probabilities for a set of tracked tokens.

Usage:
    python -m experiments.multilingual_circuits.run
    python -m experiments.multilingual_circuits.run --dtype bfloat16 --top_k 20
    python -m experiments.multilingual_circuits.run --no_operand --no_language
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm

# Use a CJK-capable font for legend labels if available, so Chinese characters
# render correctly alongside Latin text.
_CJK_FONTS = ["WenQuanYi Micro Hei", "Droid Sans Fallback", "Noto Sans CJK SC"]
_cjk_font = next(
    (f for f in _CJK_FONTS if any(x.name == f for x in _fm.fontManager.ttflist)),
    None,
)
if _cjk_font:
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = [_cjk_font, "DejaVu Sans"] + list(
        matplotlib.rcParams.get("font.sans-serif", [])
    )

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.plot_style import apply as _apply_style_base, NAVY, VIOLET, TEAL, MAUVE, GRAY, RED


def _apply_style() -> None:
    """Apply the shared thesis style, then restore CJK font if available."""
    _apply_style_base()
    if _cjk_font:
        matplotlib.rcParams["font.family"] = "sans-serif"
        matplotlib.rcParams["font.sans-serif"] = [_cjk_font, "DejaVu Sans"]
from mechinterp_qwen3.attribution_model import AttributionModel
from mechinterp_qwen3.utils.hf_utils import load_transcoder_from_hub
from mechinterp_qwen3.utils.model_utils import get_default_device, parse_dtype
from mechinterp_qwen3.utils.token_utils import tokenize_qwen_input

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("multilingual_circuits")

_MODEL = "Qwen/Qwen3-4B"
_TRANSCODER_SET = "mwhanna/qwen3-4b-transcoders"

# ── Prompts ────────────────────────────────────────────────────────────────────

ANTONYM_PROMPTS: dict[str, str] = {
    "en": 'The opposite of "small" is "',
    "fr": 'Le contraire de "petit" est "',
    "zh": '"小"的反义词是"',
}

SYNONYM_PROMPT_EN: str = 'A synonym of "small" is "'

HOT_ANTONYM_PROMPTS: dict[str, str] = {
    "en": 'The opposite of "hot" is "',
    "fr": 'Le contraire de "chaud" est "',
    "zh": '"热"的反义词是"',
}

# Language-swap pairs: (source_lang, target_lang)
LANG_SWAP_PAIRS: list[tuple[str, str]] = [("en", "zh"), ("zh", "fr"), ("fr", "en")]

# Token surface forms to track per experiment.
# All prompts end with an open-quote character, so the model predicts the next
# token without a leading space — surface forms must match that convention.

# Antonyms of "small" in each language (original predictions to track)
ANTONYM_TOKENS: dict[str, list[str]] = {
    "en": ["large", "big"],
    "fr": ["grand"],
    "zh": ["大"],
}

# Synonyms of "small" in each language (intervention targets for op-swap)
SYNONYM_TOKENS: dict[str, list[str]] = {
    "en": ["tiny", "little", "small"],
    "fr": ["petit", "minuscule"],
    "zh": ["小", "微", "矮"],
}

# Antonyms of "hot" = cold in each language (intervention targets for operand-swap).
# French "froid" tokenises as "f" + "roid"; the first predicted token is "f".
COLD_TOKENS: dict[str, list[str]] = {
    "en": ["cold"],
    "fr": ["f"],          # first subtoken of "froid"
    "zh": ["冷"],
}

LANG_LABELS: dict[str, str] = {"en": "English", "fr": "French", "zh": "Chinese"}

# ── Tokenization helpers ───────────────────────────────────────────────────────


def _tokenize(model: AttributionModel, prompt: str) -> torch.Tensor:
    """Tokenize with the attention-sink prefix; returns a 1-D token tensor."""
    return tokenize_qwen_input(prompt, model.tokenizer, model.cfg.device)


def _resolve_ids(tokenizer, surface_forms: list[str]) -> list[int]:
    """Map surface forms to their first token ID, deduplicating."""
    seen: dict[int, None] = {}
    for s in surface_forms:
        ids = tokenizer(s, add_special_tokens=False).input_ids
        if ids:
            seen.setdefault(ids[0], None)
    return list(seen)


def _find_divergence_pos(ids_a: list[int], ids_b: list[int]) -> int:
    """Return the index of the first differing token between two sequences."""
    for i in range(min(len(ids_a), len(ids_b))):
        if ids_a[i] != ids_b[i]:
            return i
    return min(len(ids_a), len(ids_b)) - 1


def _tok_label(tokenizer, tok_id: int) -> str:
    s = tokenizer.decode([tok_id]).strip()
    return s if s else f"id={tok_id}"


# ── Feature identification ─────────────────────────────────────────────────────


@torch.no_grad()
def get_pos_activations(
    model: AttributionModel,
    tokens: torch.Tensor,
    layers: list[int],
    pos: int,
) -> dict[int, torch.Tensor]:
    """Return transcoder feature activations at one token position across layers.

    Args:
        tokens: 1-D token tensor (sink + prompt).
        layers: Layer indices to query.
        pos: Token position (supports negative indexing).

    Returns:
        Mapping from layer index to float activation vector of shape (d_tc,).
    """
    inp = tokens.unsqueeze(0) if tokens.ndim == 1 else tokens
    with torch.inference_mode():
        _, act_cache = model.get_activations(inp)  # (n_layers, n_pos, d_tc)
    n_pos = act_cache.shape[1]
    abs_pos = pos if pos >= 0 else n_pos + pos
    return {layer: act_cache[layer, abs_pos, :].float() for layer in layers}


def find_exclusive_features(
    src_acts: dict[int, torch.Tensor],
    tgt_acts: dict[int, torch.Tensor],
    layers: list[int],
    top_k: int = 15,
    exclusivity: float = 0.5,
) -> tuple[dict[int, list[int]], dict[int, list[int]], dict[int, list[float]]]:
    """Partition top-K features into source-exclusive and target-exclusive sets.

    A feature is source-exclusive if its activation in the target context is below
    `exclusivity * src_val`; conversely for target-exclusive features.  These are
    the features to suppress (source) and inject (target) during intervention.

    Returns:
        suppress_by_layer: source-exclusive feature ids per layer.
        inject_by_layer:   target-exclusive feature ids per layer.
        inject_vals_by_layer: corresponding target activation magnitudes.
    """
    suppress_by_layer: dict[int, list[int]] = {}
    inject_by_layer: dict[int, list[int]] = {}
    inject_vals_by_layer: dict[int, list[float]] = {}

    for layer in layers:
        src = src_acts.get(layer, torch.zeros(1))
        tgt = tgt_acts.get(layer, torch.zeros(1))

        k = min(top_k, int(src.shape[0]))
        src_vals, src_ids = torch.topk(src, k)
        tgt_vals, tgt_ids = torch.topk(tgt, k)

        src_map = {int(f): float(v) for f, v in zip(src_ids, src_vals) if v > 0}
        tgt_map = {int(f): float(v) for f, v in zip(tgt_ids, tgt_vals) if v > 0}

        suppress = [
            f for f, sv in src_map.items()
            if tgt_map.get(f, 0.0) < exclusivity * sv
        ]

        inject, inject_vals = [], []
        for f, tv in tgt_map.items():
            if src_map.get(f, 0.0) < exclusivity * tv:
                inject.append(f)
                inject_vals.append(tv)

        suppress_by_layer[layer] = suppress
        inject_by_layer[layer] = inject
        inject_vals_by_layer[layer] = inject_vals

    return suppress_by_layer, inject_by_layer, inject_vals_by_layer


# ── Intervention hooks ─────────────────────────────────────────────────────────


def _build_swap_hooks(
    model: AttributionModel,
    layer: int,
    suppress_ids: list[int],
    inject_ids: list[int],
    inject_vals: list[float],
    alpha: float,
    target_pos: int,
) -> list[tuple[str, Callable]]:
    """Build a (capture, swap) hook pair for one transcoder layer.

    The capture hook saves the pre-transcoder hidden state; the swap hook
    re-encodes it, zeroes suppressed features, injects target features scaled
    by alpha, then decodes back to residual space — replacing the normal MLP
    output at `target_pos`.

    In torch.no_grad() context the permanent add_skip_connection hook is
    transparent, so our reconstruction is the final MLP contribution.
    """
    tc = model.transcoders[layer]

    def _swap(mlp_out: torch.Tensor, hook) -> torch.Tensor:
        h_in = _swap._mlp_in  # type: ignore[attr-defined]
        if h_in is None:
            return mlp_out
        with torch.no_grad():
            pre = F.linear(h_in.to(tc.W_enc.dtype), tc.W_enc, tc.b_enc)
            acts = tc.activation_function(pre)          # (1, n_pos, d_tc)
            n_pos = acts.shape[1]
            pos = target_pos if target_pos >= 0 else n_pos + target_pos
            if suppress_ids:
                acts[:, pos, suppress_ids] = 0.0
            if inject_ids and alpha > 0.0:
                inj = torch.tensor(inject_vals, device=acts.device, dtype=acts.dtype)
                acts[:, pos, inject_ids] += alpha * inj
            out = acts @ tc.W_dec + tc.b_dec
            if tc.W_skip is not None:
                out = out + h_in.to(out.dtype) @ tc.W_skip.T
        return out.to(mlp_out.dtype)

    _swap._mlp_in = None  # type: ignore[attr-defined]

    def _capture(acts: torch.Tensor, hook) -> torch.Tensor:
        _swap._mlp_in = acts.detach()  # type: ignore[attr-defined]
        return acts

    return [
        (f"blocks.{layer}.{model.feature_input_hook}", _capture),
        (f"blocks.{layer}.{model.original_feature_output_hook}", _swap),
    ]


# ── Intervention sweep ─────────────────────────────────────────────────────────


@torch.no_grad()
def intervention_sweep(
    model: AttributionModel,
    tokens: torch.Tensor,
    suppress_by_layer: dict[int, list[int]],
    inject_by_layer: dict[int, list[int]],
    inject_vals_by_layer: dict[int, list[float]],
    alphas: list[float],
    track_token_ids: list[int],
    target_pos: int = -1,
) -> dict[int, list[float]]:
    """Run one forward pass per alpha value and record final-position softmax probs.

    Args:
        tokens:          1-D or 2-D token tensor.
        suppress_by_layer: Source-exclusive features to zero out, per layer.
        inject_by_layer:   Target-exclusive features to add, per layer.
        inject_vals_by_layer: Activation magnitudes for injected features.
        alphas:          Sequence of injection strength multipliers.
        track_token_ids: Vocabulary indices whose probabilities to record.
        target_pos:      Token position of the intervention (default: last).

    Returns:
        Mapping from token id to list of probabilities (one per alpha value).
    """
    if tokens.ndim == 1:
        tokens = tokens.unsqueeze(0)

    all_layers = set(suppress_by_layer) | set(inject_by_layer)
    result: dict[int, list[float]] = {tid: [] for tid in track_token_ids}

    for alpha in alphas:
        hooks: list[tuple[str, Callable]] = []
        for layer in all_layers:
            hooks.extend(_build_swap_hooks(
                model, layer,
                suppress_by_layer.get(layer, []),
                inject_by_layer.get(layer, []),
                inject_vals_by_layer.get(layer, []),
                alpha, target_pos,
            ))
        logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
        probs = torch.softmax(logits[0, -1].float(), dim=-1)
        for tid in track_token_ids:
            result[tid].append(float(probs[tid]))

    return result


# ── Plotting ───────────────────────────────────────────────────────────────────


def _plot_sweep_panel(
    ax: plt.Axes,
    alphas: list[float],
    probs: dict[int, list[float]],
    orig_ids: list[int],
    int_ids: list[int],
    tokenizer,
    title: str,
) -> None:
    """Draw probability-vs-alpha curves on a single axis."""
    orig_cols = [GRAY, NAVY]
    int_cols = [RED, MAUVE, TEAL, VIOLET]

    for i, tid in enumerate(orig_ids):
        if tid not in probs:
            continue
        col = orig_cols[i % len(orig_cols)]
        label = _tok_label(tokenizer, tid)
        ax.plot(alphas, probs[tid], color=col, linewidth=1.6, label=label, zorder=3)
        ax.scatter([alphas[-1]], [probs[tid][-1]], color=col, s=28, zorder=4)

    for i, tid in enumerate(int_ids):
        if tid not in probs:
            continue
        col = int_cols[i % len(int_cols)]
        label = _tok_label(tokenizer, tid)
        ax.plot(alphas, probs[tid], color=col, linewidth=1.6, label=label, zorder=3)
        ax.scatter([alphas[-1]], [probs[tid][-1]], color=col, s=28, zorder=4)

    ax.axvline(alphas[-1], color=GRAY, linestyle="--", linewidth=0.8, alpha=0.55)
    ax.set_xlim(alphas[0], alphas[-1] * 1.06)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Intervention Strength", fontsize=9)
    tick_vals = [a for a in alphas if a == round(a)]
    ax.set_xticks(tick_vals)
    ax.set_xticklabels([f"{int(a)}×" for a in tick_vals], fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left", ncol=1)


def plot_intervention_grid(
    results_by_lang: dict[str, dict[int, list[float]]],
    alphas: list[float],
    orig_ids_by_lang: dict[str, list[int]],
    int_ids_by_lang: dict[str, list[int]],
    tokenizer,
    suptitle: str,
    out_path: Path,
    lang_order: list[str] | None = None,
) -> None:
    """Produce a 1×3 grid of probability sweep curves, one column per language."""
    _apply_style()
    order = lang_order or ["en", "zh", "fr"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)

    for ax, lang in zip(axes, order):
        _plot_sweep_panel(
            ax, alphas,
            results_by_lang.get(lang, {}),
            orig_ids_by_lang.get(lang, []),
            int_ids_by_lang.get(lang, []),
            tokenizer,
            title=LANG_LABELS.get(lang, lang),
        )

    axes[0].set_ylabel("Next Token Probability", fontsize=9)
    fig.suptitle(suptitle, fontsize=11, fontweight="bold", y=1.03)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out_path)


# ── Main experiment orchestration ──────────────────────────────────────────────


def run_experiment(
    model: AttributionModel,
    out_dir: Path,
    *,
    top_k: int = 15,
    exclusivity: float = 0.5,
    run_operation: bool = True,
    run_operand: bool = True,
    run_language: bool = True,
) -> None:
    """Orchestrate all three intervention experiments and save plots + JSON results."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = model.cfg.n_layers
    tok = model.tokenizer

    # Layer ranges by circuit component
    op_layers   = list(range(n // 3, 2 * n // 3))    # middle third (operation)
    oper_layers = list(range(0, n // 3))               # early third (operand)
    lang_layers = list(range(0, max(1, n // 5)))       # first fifth (language)

    log.info(
        "Layer ranges — operation: %d-%d  operand: 0-%d  language: 0-%d",
        op_layers[0], op_layers[-1], oper_layers[-1], lang_layers[-1],
    )

    # Tokenize all prompts once
    ant = {lang: _tokenize(model, p) for lang, p in ANTONYM_PROMPTS.items()}
    hot = {lang: _tokenize(model, p) for lang, p in HOT_ANTONYM_PROMPTS.items()}
    syn_en = _tokenize(model, SYNONYM_PROMPT_EN)

    # Baseline predictions
    log.info("Baseline predictions:")
    for lang, toks in ant.items():
        with torch.no_grad():
            logits = model(toks.unsqueeze(0))
        pred = tok.decode([int(logits[0, -1].argmax())])
        log.info("  %s: %r", lang.upper(), pred)

    all_results: dict[str, object] = {}

    # ── Experiment 1: Operation swap (antonym → synonym) ─────────────────────
    if run_operation:
        log.info("=== Experiment 1: Operation swap ===")
        alphas_op = [round(i * 0.5, 1) for i in range(13)]  # 0.0 … 6.0

        # Features derived from the English antonym vs. English synonym prompt
        ant_acts_op = get_pos_activations(model, ant["en"], op_layers, pos=-1)
        syn_acts_op = get_pos_activations(model, syn_en,    op_layers, pos=-1)
        sup_op, inj_op, vals_op = find_exclusive_features(
            ant_acts_op, syn_acts_op, op_layers, top_k, exclusivity
        )
        log.info(
            "  Antonym features (suppress): %d  |  Synonym features (inject): %d",
            sum(len(v) for v in sup_op.values()),
            sum(len(v) for v in inj_op.values()),
        )

        res_op: dict[str, dict[int, list[float]]] = {}
        orig_op: dict[str, list[int]] = {}
        int_op:  dict[str, list[int]] = {}

        for lang in ["en", "zh", "fr"]:
            orig_ids = _resolve_ids(tok, ANTONYM_TOKENS[lang])
            int_ids  = _resolve_ids(tok, SYNONYM_TOKENS[lang])
            track    = list(dict.fromkeys(orig_ids + int_ids))
            orig_op[lang] = orig_ids
            int_op[lang]  = int_ids

            log.info("  %s — tracking %d tokens", lang.upper(), len(track))
            res_op[lang] = intervention_sweep(
                model, ant[lang], sup_op, inj_op, vals_op,
                alphas_op, track, target_pos=-1,
            )

        plot_intervention_grid(
            res_op, alphas_op, orig_op, int_op, tok,
            suptitle="Operation Swap: Antonym → Synonym",
            out_path=out_dir / "op_swap.png",
        )
        all_results["operation_swap"] = {
            "alphas": alphas_op,
            "results": {
                lang: {str(tid): probs for tid, probs in lang_res.items()}
                for lang, lang_res in res_op.items()
            },
        }

    # ── Experiment 2: Operand swap (small → hot) ─────────────────────────────
    if run_operand:
        log.info("=== Experiment 2: Operand swap ===")
        alphas_oper = [round(i * 0.1, 1) for i in range(16)]  # 0.0 … 1.5

        # Find divergence position in the English reference pair
        ant_ids_en = ant["en"].tolist()
        hot_ids_en = hot["en"].tolist()
        op_pos_en  = _find_divergence_pos(ant_ids_en, hot_ids_en)
        log.info(
            "  EN operand position: %d = %r",
            op_pos_en, tok.decode([ant_ids_en[op_pos_en]]),
        )

        # Features at the operand token, early layers, from the English pair
        ant_acts_oper = get_pos_activations(model, ant["en"], oper_layers, pos=op_pos_en)
        hot_acts_oper = get_pos_activations(model, hot["en"], oper_layers, pos=op_pos_en)
        sup_oper, inj_oper, vals_oper = find_exclusive_features(
            ant_acts_oper, hot_acts_oper, oper_layers, top_k, exclusivity
        )
        log.info(
            "  Small features (suppress): %d  |  Hot features (inject): %d",
            sum(len(v) for v in sup_oper.values()),
            sum(len(v) for v in inj_oper.values()),
        )

        res_oper: dict[str, dict[int, list[float]]] = {}
        orig_oper: dict[str, list[int]] = {}
        int_oper:  dict[str, list[int]] = {}

        for lang in ["en", "zh", "fr"]:
            # Apply intervention at the language-specific operand position
            lang_ant_ids = ant[lang].tolist()
            lang_hot_ids = hot[lang].tolist()
            lang_op_pos  = _find_divergence_pos(lang_ant_ids, lang_hot_ids)
            log.info(
                "  %s operand position: %d = %r",
                lang.upper(), lang_op_pos, tok.decode([lang_ant_ids[lang_op_pos]]),
            )

            orig_ids = _resolve_ids(tok, ANTONYM_TOKENS[lang])
            int_ids  = _resolve_ids(tok, COLD_TOKENS[lang])
            track    = list(dict.fromkeys(orig_ids + int_ids))
            orig_oper[lang] = orig_ids
            int_oper[lang]  = int_ids

            res_oper[lang] = intervention_sweep(
                model, ant[lang], sup_oper, inj_oper, vals_oper,
                alphas_oper, track, target_pos=lang_op_pos,
            )

        plot_intervention_grid(
            res_oper, alphas_oper, orig_oper, int_oper, tok,
            suptitle="Operand Swap: Small → Hot",
            out_path=out_dir / "operand_swap.png",
        )
        all_results["operand_swap"] = {
            "alphas": alphas_oper,
            "results": {
                lang: {str(tid): probs for tid, probs in lang_res.items()}
                for lang, lang_res in res_oper.items()
            },
        }

    # ── Experiment 3: Language swap ───────────────────────────────────────────
    if run_language:
        log.info("=== Experiment 3: Language swap ===")
        alphas_lang = [round(i * 0.5, 1) for i in range(13)]  # 0.0 … 6.0

        lang_swap_res: dict[tuple[str, str], dict[int, list[float]]] = {}
        lang_orig_ids: dict[tuple[str, str], list[int]] = {}
        lang_int_ids:  dict[tuple[str, str], list[int]] = {}

        for src_lang, tgt_lang in LANG_SWAP_PAIRS:
            log.info("  %s → %s", src_lang.upper(), tgt_lang.upper())

            src_acts = get_pos_activations(model, ant[src_lang], lang_layers, pos=-1)
            tgt_acts = get_pos_activations(model, ant[tgt_lang], lang_layers, pos=-1)
            sup_lang, inj_lang, vals_lang = find_exclusive_features(
                src_acts, tgt_acts, lang_layers, top_k, exclusivity
            )
            log.info(
                "    Src-lang features (suppress): %d  |  Tgt-lang features (inject): %d",
                sum(len(v) for v in sup_lang.values()),
                sum(len(v) for v in inj_lang.values()),
            )

            orig_ids = _resolve_ids(tok, ANTONYM_TOKENS[src_lang])
            int_ids  = _resolve_ids(tok, ANTONYM_TOKENS[tgt_lang])
            track    = list(dict.fromkeys(orig_ids + int_ids))

            probs = intervention_sweep(
                model, ant[src_lang], sup_lang, inj_lang, vals_lang,
                alphas_lang, track, target_pos=-1,
            )
            lang_swap_res[(src_lang, tgt_lang)] = probs
            lang_orig_ids[(src_lang, tgt_lang)] = orig_ids
            lang_int_ids[(src_lang, tgt_lang)]  = int_ids

        # Assemble a 1×3 grid with the three swap pairs in order
        _apply_style()
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
        for ax, (src_lang, tgt_lang) in zip(axes, LANG_SWAP_PAIRS):
            key = (src_lang, tgt_lang)
            _plot_sweep_panel(
                ax, alphas_lang,
                lang_swap_res[key],
                lang_orig_ids[key],
                lang_int_ids[key],
                tok,
                title=f"{src_lang.upper()} → {tgt_lang.upper()}",
            )
        axes[0].set_ylabel("Next Token Probability", fontsize=9)
        fig.suptitle("Language Swap Interventions", fontsize=11, fontweight="bold", y=1.03)
        plt.tight_layout()
        path = out_dir / "lang_swap.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved %s", path)

        all_results["language_swap"] = {
            "alphas": alphas_lang,
            "pairs": [
                {
                    "src": src, "tgt": tgt,
                    "probs": {
                        str(tid): probs_list
                        for tid, probs_list in lang_swap_res[(src, tgt)].items()
                    },
                }
                for src, tgt in LANG_SWAP_PAIRS
            ],
        }

    # Save raw results for post-hoc analysis
    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info("Raw results saved to %s", results_path)


# ── Entry point ────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model",          default=_MODEL)
    parser.add_argument("--transcoder_set", default=_TRANSCODER_SET)
    parser.add_argument("--dtype",          default="bfloat16")
    parser.add_argument(
        "--top_k", type=int, default=15,
        help="Number of top features examined per layer for source/target identification",
    )
    parser.add_argument(
        "--exclusivity", type=float, default=0.5,
        help="Feature exclusivity threshold: tgt < threshold * src to qualify as source-exclusive",
    )
    parser.add_argument("--out_dir", default="runs/multilingual_circuits")
    parser.add_argument("--no_operation", action="store_true", help="Skip operation swap experiment")
    parser.add_argument("--no_operand",   action="store_true", help="Skip operand swap experiment")
    parser.add_argument("--no_language",  action="store_true", help="Skip language swap experiment")
    args = parser.parse_args()

    device = get_default_device()
    dtype  = parse_dtype(args.dtype)
    log.info("Device: %s  dtype: %s", device, dtype)

    log.info("Loading model %s…", args.model)
    transcoder_set, _ = load_transcoder_from_hub(
        args.transcoder_set, dtype=dtype, lazy_encoder=True, lazy_decoder=True
    )
    model = AttributionModel.from_pretrained_and_transcoders(
        args.model, transcoder_set, dtype=dtype, device=device
    )
    model.eval()
    log.info("Model loaded: %d layers", model.cfg.n_layers)

    run_experiment(
        model,
        out_dir=Path(args.out_dir),
        top_k=args.top_k,
        exclusivity=args.exclusivity,
        run_operation=not args.no_operation,
        run_operand=not args.no_operand,
        run_language=not args.no_language,
    )
    log.info("Done.")


if __name__ == "__main__":
    main()
