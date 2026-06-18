"""Soft-prompt / prefix-tuning experiment for addition task.

Trains a small set of learnable prefix vectors (frozen model) that improve
Qwen3-4B's addition accuracy, then analyses whether the improvement is
mechanistically aligned with the carry circuit found by the probe / steering
experiments.

Pipeline stages:

  --train    Train the prefix via CE loss on the addition dataset.
             --mode soft_prompt | prefix_tuning   (default: soft_prompt)

  --eval     Evaluate a trained checkpoint: accuracy with vs. without prefix.

  --analyze  Steering-alignment analysis:
               1. Compute prefix delta at each layer:
                    delta[L] = h_with_prefix[L, eq_pos] - h_base[L, eq_pos]
               2. Recompute contrastive steering vectors (correct/wrong runs)
                  at those layers (same method as steer.py).
               3. Report cosine(delta[L], sv[L]) per layer.
               4. Encode delta[L] through the transcoder → top-k SAE features.

  --all      train → eval → analyze

Usage examples:

    python experiments/soft_prompt/run.py --all
    python experiments/soft_prompt/run.py --train --mode prefix_tuning
    python experiments/soft_prompt/run.py --eval  --mode soft_prompt
    python experiments/soft_prompt/run.py --analyze --mode soft_prompt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in [str(_REPO_ROOT), str(_REPO_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mechinterp_qwen3.attribution_model import AttributionModel  # noqa: E402
from mechinterp_qwen3.utils.hf_utils import load_transcoder_from_hub  # noqa: E402
from mechinterp_qwen3.utils.model_utils import get_default_device, parse_dtype  # noqa: E402

from experiments.soft_prompt.dataset_utils import load_concept_dataset  # noqa: E402
from experiments.soft_prompt.model import PrefixTuning, SoftPrompt, load_prefix  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("soft_prompt.run")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_ids(sample: dict, device: torch.device) -> tuple[torch.Tensor, list[int]]:
    """Return (prompt_token_ids_1d, answer_token_ids) from either dataset format."""
    if "prompt_token_ids" in sample:
        raw_ids = torch.tensor(sample["prompt_token_ids"], dtype=torch.long, device=device)
        answer_ids = sample["answer_token_ids"]
    else:
        raise KeyError(f"Dataset sample missing 'prompt_token_ids'. Keys: {list(sample.keys())}")
    return raw_ids, answer_ids


_target_ids_cache: dict[str, set[int]] = {}


@torch.no_grad()
def _greedy_decode(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    hooks: list | None = None,
    attn_mask: torch.Tensor | None = None,
    eos_token_id: int | None = None,
) -> list[int]:
    """Greedy decode up to max_new_tokens tokens. Works with or without prefix hooks."""
    ids = input_ids  # (1, seq_len)
    mask = attn_mask
    generated = []
    for _ in range(max_new_tokens):
        if hooks:
            logits = model.run_with_hooks(ids, fwd_hooks=hooks, attention_mask=mask)
        else:
            logits = model(ids)
        next_id = int(logits[0, -1, :].argmax())
        generated.append(next_id)
        if eos_token_id is not None and next_id == eos_token_id:
            break
        next_tok = torch.tensor([[next_id]], dtype=torch.long, device=ids.device)
        ids = torch.cat([ids, next_tok], dim=1)
        if mask is not None:
            mask = torch.cat([mask, torch.ones(1, 1, dtype=torch.long, device=ids.device)], dim=1)
    return generated


def _matching_token_ids(tokenizer, target_str: str) -> set[int]:
    """Return all single-token IDs that are case/space variants of target_str.

    Uses candidate generation + tokenization rather than vocab scanning,
    because Qwen stores vocab keys with Ġ (GPT-2 byte encoding) not literal spaces.
    """
    if target_str in _target_ids_cache:
        return _target_ids_cache[target_str]
    base = target_str.strip()
    candidates = {
        base, base.lower(), base.upper(), base.capitalize(),
        " " + base, " " + base.lower(), " " + base.upper(), " " + base.capitalize(),
        "\n" + base, "\n" + base.lower(),
    }
    matches: set[int] = set()
    for cand in candidates:
        ids = tokenizer(cand, add_special_tokens=False).input_ids
        if len(ids) == 1:
            matches.add(ids[0])
    _target_ids_cache[target_str] = matches
    return matches



# ---------------------------------------------------------------------------
# Stage: train
# ---------------------------------------------------------------------------


def run_train(args: argparse.Namespace, device: torch.device) -> None:
    mode = args.mode
    out_dir = Path(args.out_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"prefix_{mode}.pt"
    hist_path = out_dir / f"train_history_{mode}.json"

    log.info("=== Stage: train [%s] ===", mode)
    dtype = parse_dtype(args.dtype)

    log.info("Loading model %s", args.model)
    transcoder_set, _ = load_transcoder_from_hub(
        args.transcoder_set, dtype=dtype, lazy_encoder=True, lazy_decoder=True
    )
    model = AttributionModel.from_pretrained_and_transcoders(
        args.model, transcoder_set, dtype=dtype, device=device
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    tokenizer = model.tokenizer
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    d_model = model.cfg.d_model
    n_layers = model.cfg.n_layers

    # Instantiate prefix module
    if mode == "soft_prompt":
        prefix_module: SoftPrompt | PrefixTuning = SoftPrompt(k=args.prefix_len, d_model=d_model)
    else:
        prefix_module = PrefixTuning(k=args.prefix_len, d_model=d_model, n_layers=n_layers)
    prefix_module = prefix_module.to(device=device, dtype=dtype)

    optimiser = torch.optim.AdamW(prefix_module.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)

    samples = load_concept_dataset(args.concept, tokenizer, template=args.template, n_per_template=args.n_per_template)
    split = int(0.9 * len(samples))
    train_samples = samples[:split]
    val_samples = samples[split:]
    log.info("Dataset: %d train / %d val", len(train_samples), len(val_samples))

    best_val_ce = float("inf")
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        prefix_module.train()
        perm = torch.randperm(len(train_samples)).tolist()
        batches = [perm[i : i + args.batch_size] for i in range(0, len(perm), args.batch_size)]
        epoch_ce = 0.0
        n_batches = 0

        for idx_batch in batches:
            meta: list[dict[str, Any]] = []
            for gi in idx_batch:
                sample = train_samples[gi]
                raw_ids, answer_ids = _sample_ids(sample, device)
                if not answer_ids:
                    continue
                ext_ids, attn = prefix_module.prepare_inputs(raw_ids, pad_id)
                meta.append(
                    {
                        "ids": ext_ids.squeeze(0),  # (k + seq_len,)
                        "attn": attn.squeeze(0),
                        "target_id": answer_ids[0],
                    }
                )

            if not meta:
                continue

            # Pad batch
            max_len = max(d["ids"].shape[0] for d in meta)
            B = len(meta)
            padded = torch.zeros(B, max_len, dtype=torch.long, device=device)
            attn_mask = torch.zeros(B, max_len, dtype=torch.long, device=device)
            last_pos_list = []
            for i, d in enumerate(meta):
                L = d["ids"].shape[0]
                padded[i, :L] = d["ids"]
                attn_mask[i, :L] = d["attn"]
                last_pos_list.append(L - 1)

            targets = torch.tensor([d["target_id"] for d in meta], device=device, dtype=torch.long)
            hooks = prefix_module.hooks(batch_size=B)

            logits = model.run_with_hooks(padded, fwd_hooks=hooks, attention_mask=attn_mask)
            last_logits = torch.stack([logits[i, last_pos_list[i], :] for i in range(B)])
            loss = F.cross_entropy(last_logits, targets)

            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(prefix_module.parameters(), 1.0)
            optimiser.step()

            epoch_ce += loss.item()
            n_batches += 1

        scheduler.step()
        avg_ce = epoch_ce / max(n_batches, 1)
        grad_norm = (
            prefix_module.prefix.grad.norm().item()
            if prefix_module.prefix.grad is not None else 0.0
        )

        # Quick val pass (no grad)
        val_ce, val_prob = _eval_ce(
            prefix_module, model, tokenizer, pad_id, val_samples[:200], args, device, dtype
        )

        if epoch % max(1, args.epochs // 10) == 0 or epoch == args.epochs:
            log.info(
                "Epoch %3d/%d — train_CE=%.4f  val_CE=%.4f  val_p(correct)=%.4f  grad_norm=%.4f",
                epoch, args.epochs, avg_ce, val_ce, val_prob, grad_norm,
            )

        history.append({"epoch": epoch, "train_ce": avg_ce, "val_ce": val_ce, "val_prob_correct": val_prob, "grad_norm": grad_norm})

        if val_ce < best_val_ce:
            best_val_ce = val_ce
            prefix_module.save(ckpt_path)

    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    log.info("[%s] Training done — best val_CE=%.4f — checkpoint: %s", mode, best_val_ce, ckpt_path)


def _eval_ce(
    prefix_module: SoftPrompt | PrefixTuning,
    model: AttributionModel,
    tokenizer,
    pad_id: int,
    samples: list[dict],
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[float, float]:
    """Return (mean_CE, mean_p_correct) over samples."""
    prefix_module.eval()
    total_ce = 0.0
    total_prob = 0.0
    n = 0
    with torch.no_grad():
        for sample in samples:
            raw_ids, answer_ids = _sample_ids(sample, device)
            if not answer_ids:
                continue
            ext_ids, attn = prefix_module.prepare_inputs(raw_ids, pad_id)
            hooks = prefix_module.hooks(batch_size=1)
            logits = model.run_with_hooks(ext_ids, fwd_hooks=hooks, attention_mask=attn)
            last_logit = logits[0, attn.sum() - 1, :]
            target_id = answer_ids[0]
            target = torch.tensor(target_id, device=device)
            total_ce += F.cross_entropy(last_logit.unsqueeze(0), target.unsqueeze(0)).item()
            target_str = sample.get("true_answer_str", tokenizer.decode([target_id]))
            probs = F.softmax(last_logit.float(), dim=-1)
            target_ids_set = _matching_token_ids(tokenizer, target_str)
            total_prob += float(sum(probs[tid].item() for tid in target_ids_set if tid < probs.shape[0]))
            n += 1
    return total_ce / max(n, 1), total_prob / max(n, 1)


# ---------------------------------------------------------------------------
# Stage: eval
# ---------------------------------------------------------------------------


def run_eval(args: argparse.Namespace, device: torch.device) -> None:
    mode = args.mode
    ckpt_path = Path(args.out_root) / f"prefix_{mode}.pt"
    log.info("=== Stage: eval [%s] ===", mode)

    dtype = parse_dtype(args.dtype)
    transcoder_set, _ = load_transcoder_from_hub(
        args.transcoder_set, dtype=dtype, lazy_encoder=True, lazy_decoder=True
    )
    model = AttributionModel.from_pretrained_and_transcoders(
        args.model, transcoder_set, dtype=dtype, device=device
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    tokenizer = model.tokenizer
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    prefix_module = load_prefix(ckpt_path, device=device)
    prefix_module = prefix_module.to(device=device, dtype=dtype)
    prefix_module.eval()

    samples = load_concept_dataset(args.concept, tokenizer, template=args.template, n_per_template=args.n_per_template)
    eval_samples = samples[int(0.9 * len(samples)) :]
    log.info("Evaluating on %d samples", len(eval_samples))

    n_correct_base = 0
    n_correct_prefix = 0
    n_total = 0
    sum_prob_base = 0.0
    sum_prob_pfx = 0.0
    per_sample: list[dict] = []

    with torch.no_grad():
        for sample in tqdm(eval_samples, desc="Eval"):
            raw_ids, answer_ids = _sample_ids(sample, device)
            if not answer_ids:
                continue
            target_id = answer_ids[0]
            target_str = sample.get("true_answer_str", tokenizer.decode([target_id]))
            target_ids_set = _matching_token_ids(tokenizer, target_str)

            # Baseline (no prefix)
            base_input = raw_ids.unsqueeze(0)
            base_logits = model(base_input)
            base_last = base_logits[0, -1, :]
            base_pred = int(base_last.argmax())
            base_probs = F.softmax(base_last.float(), dim=-1)
            base_prob = float(sum(base_probs[tid].item() for tid in target_ids_set if tid < base_probs.shape[0]))

            # With prefix
            ext_ids, attn = prefix_module.prepare_inputs(raw_ids, pad_id)
            hooks = prefix_module.hooks(batch_size=1)
            pfx_logits = model.run_with_hooks(ext_ids, fwd_hooks=hooks, attention_mask=attn)
            pfx_last = pfx_logits[0, attn.sum() - 1, :]
            pfx_pred = int(pfx_last.argmax())
            pfx_probs = F.softmax(pfx_last.float(), dim=-1)
            pfx_prob = float(sum(pfx_probs[tid].item() for tid in target_ids_set if tid < pfx_probs.shape[0]))

            base_ok = int(base_pred in target_ids_set)
            pfx_ok = int(pfx_pred in target_ids_set)
            n_correct_base += base_ok
            n_correct_prefix += pfx_ok
            sum_prob_base += base_prob
            sum_prob_pfx += pfx_prob
            n_total += 1

            # Top-k tokens
            top_k = args.top_k_tokens
            base_topk = base_last.float().softmax(-1).topk(top_k)
            pfx_topk = pfx_last.float().softmax(-1).topk(top_k)
            base_topk_info = [
                {"token": tokenizer.decode([int(i)]), "id": int(i), "prob": float(p)}
                for i, p in zip(base_topk.indices.tolist(), base_topk.values.tolist())
            ]
            pfx_topk_info = [
                {"token": tokenizer.decode([int(i)]), "id": int(i), "prob": float(p)}
                for i, p in zip(pfx_topk.indices.tolist(), pfx_topk.values.tolist())
            ]

            _skip = {"prompt_token_ids", "answer_token_ids"}
            meta = {k: v for k, v in sample.items() if k not in _skip}
            meta.update(
                {
                    "base_correct": bool(base_ok),
                    "pfx_correct": bool(pfx_ok),
                    "base_prob_correct": base_prob,
                    "pfx_prob_correct": pfx_prob,
                    "base_top_tokens": base_topk_info,
                    "pfx_top_tokens": pfx_topk_info,
                }
            )
            per_sample.append(meta)

    acc_base = 100.0 * n_correct_base / max(n_total, 1)
    acc_pfx = 100.0 * n_correct_prefix / max(n_total, 1)
    mean_prob_base = sum_prob_base / max(n_total, 1)
    mean_prob_pfx = sum_prob_pfx / max(n_total, 1)
    log.info("Baseline accuracy:       %.2f%%  (%d/%d)", acc_base, n_correct_base, n_total)
    log.info("With-prefix accuracy:    %.2f%%  (%d/%d)", acc_pfx, n_correct_prefix, n_total)
    log.info("Delta accuracy:          %+.2f%%", acc_pfx - acc_base)
    log.info("Baseline mean p(correct):  %.4f", mean_prob_base)
    log.info("With-prefix mean p(correct): %.4f", mean_prob_pfx)
    log.info("Delta p(correct):          %+.4f", mean_prob_pfx - mean_prob_base)

    # Generate continuations for first n_show samples
    n_show = min(args.n_show, len(eval_samples))
    eos_id = tokenizer.eos_token_id
    if n_show > 0 and args.gen_tokens > 0:
        log.info("Generating %d tokens for first %d samples...", args.gen_tokens, n_show)
        with torch.no_grad():
            for i, sample in enumerate(eval_samples[:n_show]):
                raw_ids, _ = _sample_ids(sample, device)
                base_gen_ids = _greedy_decode(
                    model, raw_ids.unsqueeze(0), args.gen_tokens, eos_token_id=eos_id
                )
                ext_ids, attn = prefix_module.prepare_inputs(raw_ids, pad_id)
                pfx_hooks = prefix_module.hooks(batch_size=1)
                pfx_gen_ids = _greedy_decode(
                    model, ext_ids, args.gen_tokens, hooks=pfx_hooks, attn_mask=attn, eos_token_id=eos_id
                )
                per_sample[i]["base_gen"] = tokenizer.decode(base_gen_ids)
                per_sample[i]["pfx_gen"] = tokenizer.decode(pfx_gen_ids)

    # Print top-k token comparison and generation for a few samples
    n_show = min(args.n_show, len(per_sample))
    if n_show > 0:
        log.info("")
        log.info("Top-%d token predictions and generation for first %d samples:", args.top_k_tokens, n_show)
        _eval_keys = {"base_correct", "pfx_correct", "base_prob_correct", "pfx_prob_correct",
                      "base_top_tokens", "pfx_top_tokens", "base_gen", "pfx_gen",
                      "prompt_token_ids", "answer_token_ids"}
        for s in per_sample[:n_show]:
            base_str = "  ".join(f"{t['token']!r}({t['prob']:.3f})" for t in s["base_top_tokens"])
            pfx_str  = "  ".join(f"{t['token']!r}({t['prob']:.3f})" for t in s["pfx_top_tokens"])
            meta_str = "  ".join(f"{k}={v}" for k, v in s.items() if k not in _eval_keys)
            log.info("  %s", meta_str)
            log.info("    base top: %s", base_str)
            log.info("    pfx  top: %s", pfx_str)
            if "base_gen" in s:
                log.info("    base gen: %r", s["base_gen"])
                log.info("    pfx  gen: %r", s["pfx_gen"])

    results = {
        "mode": mode,
        "n_total": n_total,
        "acc_base": acc_base,
        "acc_prefix": acc_pfx,
        "delta_acc": acc_pfx - acc_base,
        "mean_prob_correct_base": mean_prob_base,
        "mean_prob_correct_prefix": mean_prob_pfx,
        "delta_prob_correct": mean_prob_pfx - mean_prob_base,
    }
    out_path = Path(args.out_root) / f"eval_{mode}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Saved to %s", out_path)

    samples_path = Path(args.out_root) / f"eval_{mode}_samples.json"
    with open(samples_path, "w") as f:
        json.dump(per_sample, f, indent=2)
    log.info("Per-sample results saved to %s", samples_path)


# ---------------------------------------------------------------------------
# Stage: analyze
# ---------------------------------------------------------------------------


def run_analyze(args: argparse.Namespace, device: torch.device) -> None:
    """Steering-alignment analysis.

    For each sweep layer:
      1. Prefix delta: mean over samples of
           h_with_prefix[L, eq_pos+k] − h_base[L, eq_pos]
      2. Contrastive steering vector (correct − wrong baseline runs),
         mirroring the approach in steer.py.
      3. cos(prefix_delta[L], sv[L])
      4. Encode prefix_delta[L] through transcoder → top SAE features.
    """
    mode = args.mode
    ckpt_path = Path(args.out_root) / f"prefix_{mode}.pt"
    out_dir = Path(args.out_root)
    log.info("=== Stage: analyze [%s] ===", mode)

    dtype = parse_dtype(args.dtype)
    transcoder_set, _ = load_transcoder_from_hub(
        args.transcoder_set, dtype=dtype, lazy_encoder=False, lazy_decoder=False
    )
    model = AttributionModel.from_pretrained_and_transcoders(
        args.model, transcoder_set, dtype=dtype, device=device
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    tokenizer = model.tokenizer
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    prefix_module = load_prefix(ckpt_path, device=device)
    prefix_module = prefix_module.to(device=device, dtype=dtype)
    prefix_module.eval()
    k = prefix_module.k

    samples = load_concept_dataset(args.concept, tokenizer, template=args.template, n_per_template=args.n_per_template)
    eval_samples = samples[int(0.9 * len(samples)) :][: args.analyze_n]
    log.info("Analyzing %d samples across layers %s", len(eval_samples), args.analyze_layers)

    # ------------------------------------------------------------------ #
    # 1. Collect prefix deltas and baseline activations per layer         #
    # ------------------------------------------------------------------ #
    sweep_layers: list[int] = args.analyze_layers

    base_acts: dict[int, list[torch.Tensor]] = {L: [] for L in sweep_layers}
    pfx_acts: dict[int, list[torch.Tensor]] = {L: [] for L in sweep_layers}

    # Steering vectors: bucketed by BASELINE correctness (noisy — few correct)
    correct_acts_base: dict[int, list[torch.Tensor]] = {L: [] for L in sweep_layers}
    wrong_acts_base: dict[int, list[torch.Tensor]] = {L: [] for L in sweep_layers}
    # Steering vectors: bucketed by PREFIX correctness (clean — most correct)
    correct_acts_pfx: dict[int, list[torch.Tensor]] = {L: [] for L in sweep_layers}
    wrong_acts_pfx: dict[int, list[torch.Tensor]] = {L: [] for L in sweep_layers}

    with torch.no_grad():
        for sample in tqdm(eval_samples, desc="Collecting deltas"):
            raw_ids, answer_ids = _sample_ids(sample, device)
            if not answer_ids:
                continue
            target_id = answer_ids[0]

            # Last token of the prompt is always the answer position
            last_pos_raw = raw_ids.shape[0] - 1
            base_input = raw_ids.unsqueeze(0)
            ext_ids, attn = prefix_module.prepare_inputs(raw_ids, pad_id)
            last_pos_pfx = attn.sum().item() - 1

            base_cache: dict[int, torch.Tensor] = {}
            pfx_cache: dict[int, torch.Tensor] = {}

            base_hooks = [
                (
                    f"blocks.{L}.hook_resid_post",
                    lambda act, hook, _L=L, _pos=last_pos_raw, cache=base_cache: (
                        cache.update({_L: act[0, _pos, :].detach().clone()}) or act
                    ),
                )
                for L in sweep_layers
            ]
            pfx_inject_hooks = prefix_module.hooks(batch_size=1)
            pfx_capture_hooks = [
                (
                    f"blocks.{L}.hook_resid_post",
                    lambda act, hook, _L=L, _pos=last_pos_pfx, cache=pfx_cache: (
                        cache.update({_L: act[0, _pos, :].detach().clone()}) or act
                    ),
                )
                for L in sweep_layers
            ]

            base_logits = model.run_with_hooks(base_input, fwd_hooks=base_hooks)
            pfx_logits = model.run_with_hooks(
                ext_ids, fwd_hooks=pfx_inject_hooks + pfx_capture_hooks, attention_mask=attn
            )

            for L in sweep_layers:
                if L in base_cache and L in pfx_cache:
                    base_acts[L].append(base_cache[L].to(dtype=torch.float32))
                    pfx_acts[L].append(pfx_cache[L].to(dtype=torch.float32))

            # Bucket baseline acts by baseline correctness
            base_pred = int(base_logits[0, -1, :].argmax())
            b_bucket = correct_acts_base if base_pred == target_id else wrong_acts_base
            for L in sweep_layers:
                if L in base_cache:
                    b_bucket[L].append(base_cache[L].to(dtype=torch.float32))

            # Bucket prefix acts by prefix correctness
            pfx_pred = int(pfx_logits[0, attn.sum() - 1, :].argmax())
            p_bucket = correct_acts_pfx if pfx_pred == target_id else wrong_acts_pfx
            for L in sweep_layers:
                if L in pfx_cache:
                    p_bucket[L].append(pfx_cache[L].to(dtype=torch.float32))

    # ------------------------------------------------------------------ #
    # 2. Compute prefix deltas + two steering vectors                      #
    # ------------------------------------------------------------------ #
    prefix_deltas: dict[int, torch.Tensor] = {}
    sv_base: dict[int, torch.Tensor] = {}  # correct/wrong from baseline runs
    sv_pfx: dict[int, torch.Tensor] = {}  # correct/wrong from prefix runs

    def _make_sv(correct: list, wrong: list, label: str, L: int) -> torch.Tensor | None:
        n_c, n_w = len(correct), len(wrong)
        if n_c == 0 or n_w == 0:
            log.warning("Layer %2d  %s: not enough data (correct=%d wrong=%d)", L, label, n_c, n_w)
            return None
        n = min(n_c, n_w)
        sv = torch.stack(correct[:n]).mean(0) - torch.stack(wrong[:n]).mean(0)
        log.info(
            "Layer %2d  %s sv_norm=%.3f  (%d correct / %d wrong)",
            L,
            label,
            sv.norm().item(),
            n_c,
            n_w,
        )
        return sv

    for L in sweep_layers:
        if base_acts[L] and pfx_acts[L]:
            prefix_deltas[L] = torch.stack(pfx_acts[L]).mean(0) - torch.stack(base_acts[L]).mean(0)
        sv_base[L] = _make_sv(correct_acts_base[L], wrong_acts_base[L], "base_sv", L)
        sv_pfx[L] = _make_sv(correct_acts_pfx[L], wrong_acts_pfx[L], "pfx_sv", L)

    # ------------------------------------------------------------------ #
    # 3. Cosine similarities — delta vs both svs                          #
    # ------------------------------------------------------------------ #
    log.info("")
    log.info("%-6s  %-12s  %-14s  %-14s", "Layer", "delta_norm", "cos(Δ,sv_base)", "cos(Δ,sv_pfx)")
    log.info("-" * 55)

    analysis_rows: list[dict] = []
    for L in sweep_layers:
        row: dict = {"layer": L}
        delta = prefix_deltas.get(L)
        if delta is not None:
            row["delta_norm"] = delta.norm().item()
        for sv_key, sv_dict in [("cos_delta_sv_base", sv_base), ("cos_delta_sv_pfx", sv_pfx)]:
            sv = sv_dict.get(L)
            if delta is not None and sv is not None:
                cos = F.cosine_similarity(delta.unsqueeze(0), sv.unsqueeze(0)).item()
                row[sv_key] = cos
        log.info(
            "%-6d  %-12.4f  %-14s  %-14s",
            L,
            row.get("delta_norm", float("nan")),
            f"{row['cos_delta_sv_base']:.4f}" if "cos_delta_sv_base" in row else "n/a",
            f"{row['cos_delta_sv_pfx']:.4f}" if "cos_delta_sv_pfx" in row else "n/a",
        )
        analysis_rows.append(row)

    # ------------------------------------------------------------------ #
    # 4. SAE feature analysis at every sweep layer + probe cross-ref       #
    # ------------------------------------------------------------------ #

    # Load carry probe top features if checkpoint exists
    probe_top_features: set[int] = set()
    if args.carry_probe_ckpt and Path(args.carry_probe_ckpt).exists():
        try:
            probe_ckpt = torch.load(args.carry_probe_ckpt, map_location="cpu", weights_only=False)
            probe_sd = probe_ckpt["probe"]["state_dict"]
            probe_w = probe_sd["linear.weight"].squeeze(0)  # (d_tc * n_probe_layers,)
            d_tc_probe = probe_ckpt["probe"]["d_transcoder"]
            probe_layers = probe_ckpt["probe"]["layers"]
            # Extract per-layer top features
            for li, _pl in enumerate(probe_layers):
                w_layer = probe_w[li * d_tc_probe : (li + 1) * d_tc_probe]
                topk = w_layer.abs().topk(args.top_k_features)
                probe_top_features.update(topk.indices.tolist())
            log.info(
                "Loaded carry probe (%d layers, %d top features)",
                len(probe_layers),
                len(probe_top_features),
            )
        except Exception as e:
            log.warning("Could not load carry probe: %s", e)

    log.info("")
    log.info("SAE feature analysis of prefix delta per sweep layer:")
    for row in analysis_rows:
        L = row["layer"]
        delta = prefix_deltas.get(L)
        if delta is None:
            continue
        transcoder = transcoder_set.transcoders[L]
        feat_acts = transcoder.encode(
            delta.to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
        ).squeeze()
        topk_vals, topk_idx = feat_acts.abs().topk(args.top_k_features)
        top_feats = topk_idx.tolist()
        overlap = [f for f in top_feats if f in probe_top_features]
        log.info(
            "  Layer %2d  top features: %s  |  probe overlap: %d/%d  %s",
            L,
            top_feats[:5],
            len(overlap),
            args.top_k_features,
            overlap[:5] if overlap else "[]",
        )
        row["top_features"] = [
            {"rank": r + 1, "feature_idx": int(i), "activation": float(feat_acts[i].item())}
            for r, i in enumerate(top_feats)
        ]
        row["probe_overlap"] = overlap

    # Save
    out_path = out_dir / f"analysis_{mode}.json"
    with open(out_path, "w") as f:
        json.dump(analysis_rows, f, indent=2)
    log.info("Analysis saved to %s", out_path)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Soft-prompt / prefix-tuning experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--train", action="store_true")
    p.add_argument("--eval", action="store_true")
    p.add_argument("--analyze", action="store_true")
    p.add_argument("--all", action="store_true", help="Run train → eval → analyze")

    p.add_argument(
        "--mode",
        default="soft_prompt",
        choices=["soft_prompt", "prefix_tuning"],
        help="Which prefix variant to train/eval/analyze",
    )

    # Model
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--transcoder_set", default="mwhanna/qwen3-4b-transcoders")
    p.add_argument("--dtype", default="bfloat16")

    # Data
    p.add_argument("--concept", default="carry",
                   help="Concept dataset name (e.g. carry, gcd, perfect_square)")
    p.add_argument("--template", default="T0",
                   help="Template key within the concept dataset (e.g. T0, T1, T2)")
    p.add_argument("--n_per_template", type=int, default=500,
                   help="Number of pairs per template to generate")

    # Prefix
    p.add_argument("--prefix_len", type=int, default=10, help="Number of prefix tokens (k)")

    # Training
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)

    # Output
    p.add_argument("--out_root", default="runs/soft_prompt")
    p.add_argument("--force", action="store_true")
    p.add_argument("--top_k_tokens", type=int, default=3, help="Top-k tokens to show per sample in eval")
    p.add_argument("--n_show", type=int, default=5, help="Number of sample token comparisons to print in eval")
    p.add_argument("--gen_tokens", type=int, default=50, help="Number of tokens to greedily generate per shown sample (0 to disable)")

    # Analyze
    p.add_argument(
        "--analyze_layers",
        type=int,
        nargs="+",
        default=[8, 12, 16, 20, 24, 28, 32],
        help="Layers to sweep in the analysis stage",
    )
    p.add_argument(
        "--analyze_n", type=int, default=200, help="Number of samples to use in analysis"
    )
    p.add_argument(
        "--top_k_features", type=int, default=20, help="Number of top SAE features to report"
    )
    p.add_argument(
        "--carry_probe_ckpt",
        type=str,
        default=None,
        help="Path to carry probe checkpoint for feature overlap analysis "
        "(e.g. runs/carry_probe/2026-04-08_213359/checkpoints/best_probe.pt)",
    )

    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = build_parser().parse_args()

    if not any([args.train, args.eval, args.analyze, args.all]):
        build_parser().print_help()
        return

    if args.all:
        args.train = args.eval = args.analyze = True

    device = get_default_device()
    log.info("Device: %s  dtype: %s  mode: %s", device, args.dtype, args.mode)

    if args.train:
        run_train(args, device)
    if args.eval:
        run_eval(args, device)
    if args.analyze:
        run_analyze(args, device)


if __name__ == "__main__":
    main()
