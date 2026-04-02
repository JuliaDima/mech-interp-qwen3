"""Feature-level knowledge editing experiment.

Injects the small addition model's representations into Qwen3-4B via a learned
bottleneck θ that aligns with the big model's SAE feature geometry.  Two injection
modes are compared:

  Replace  — hook_mlp_out ← w_out(θ(f_s))
  Add      — hook_mlp_out ← hook_mlp_out + w_out(θ(f_s))

Pipeline stages (each can be run independently):

  --setup   Collect (f_s, decoded_f_B) pairs from both models; compute PCA
            alignment projection P; save to out_root/setup/.

  --train   Train FeatureAlignmentModule with L_CE + λ·L_align.
            --mode replace | add   (default: both, sequentially)

  --eval    Evaluate a trained checkpoint on the test split.

  --compare Load both replace/add checkpoints and print a comparison table.

  --all     Run setup → train (both modes) → compare.

Usage examples:

    python experiments/knowledge_editing/run.py --all
    python experiments/knowledge_editing/run.py --setup
    python experiments/knowledge_editing/run.py --train --mode replace
    python experiments/knowledge_editing/run.py --eval --mode replace
    python experiments/knowledge_editing/run.py --compare
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
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Top-level package imports
from mechinterp_qwen3.attribution_model import AttributionModel  # noqa: E402
from mechinterp_qwen3.transcoder.single_layer_transcoder import SingleLayerTranscoder  # noqa: E402
from mechinterp_qwen3.utils.hf_utils import load_transcoder_from_hub  # noqa: E402
from mechinterp_qwen3.utils.model_utils import get_default_device, parse_dtype  # noqa: E402
from mechinterp_qwen3.utils.token_utils import tokenize_qwen_input  # noqa: E402

# Stitching utilities (dataset loader + small model definition)
_STITCH_DIR = str(Path(__file__).parent.parent / "stitching")
if _STITCH_DIR not in sys.path:
    sys.path.insert(0, _STITCH_DIR)

from run import SmallAdditionTransformer, _load_small_model, _load_small_sae  # noqa: E402
from utils import get_small_model_tokenizer, load_addition_dataset  # noqa: E402

from experiments.knowledge_editing.model import FeatureAlignmentModule  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("knowledge_editing.run")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_LARGE_MODEL = "Qwen/Qwen3-4B"
_TRANSCODER_SET = "mwhanna/qwen3-4b-transcoders"
_INJECT_LAYER = 30  # Qwen3-4B layer to patch (0-indexed)
_D_MID = 256  # Bottleneck dimension
_LAMBDA_ALIGN = 0.1  # Weight for alignment loss
_LR = 3e-4
_EPOCHS = 30
_BATCH_SIZE = 64
_MAX_SAMPLES = 5_000  # Samples used for setup / training
_MAX_NEW_TOKENS = 6  # Greedy decode budget for evaluation

# ---------------------------------------------------------------------------
# Collection helpers
# ---------------------------------------------------------------------------


def _find_eq_pos_large(token_ids: torch.Tensor, tokenizer) -> int | None:
    """Return the index of the '=' token in Qwen3-4B token sequence."""
    eq_ids = tokenizer("=", add_special_tokens=False).input_ids
    if not eq_ids:
        return None
    eq_id = eq_ids[-1]
    positions = (token_ids == eq_id).nonzero(as_tuple=True)[0]
    return int(positions[-1]) if len(positions) > 0 else None


def collect_features(
    small_model: SmallAdditionTransformer,
    small_sae: SingleLayerTranscoder,
    large_model: AttributionModel,
    large_transcoder: SingleLayerTranscoder,
    samples: list[dict[str, Any]],
    inject_layer: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect (f_s, decoded_f_B) pairs for every sample.

    f_s         — small model SAE feature activations at '=' position (last layer),
                  shape (N, small_sae.d_transcoder).  Using SAE features rather than
                  the raw residual stream gives an interpretable, sparse representation
                  of the carry circuit — matching what the stitching pipeline produces.
    decoded_f_B — big model SAE decoded output at '=' position, shape (N, d_model_large)
                  computed as  feat_acts @ W_dec  (lazy decoder loaded on first access)
    """
    small_tokenize = get_small_model_tokenizer(small_model)
    tokenizer = large_model.tokenizer
    sae_layer = small_model.n_layers - 1  # last layer — where carry is most active

    f_s_list: list[torch.Tensor] = []
    decoded_fB_list: list[torch.Tensor] = []

    small_model.model.eval()
    small_sae.eval()
    large_model.eval()
    large_transcoder.eval()

    with torch.no_grad():
        for sample in tqdm(samples, desc="Collecting features", leave=False):
            prompt: str = sample["prompt"]

            # Build a small-model prompt using only {digits, +, =} — safe for any
            # small vocab (QuantaMaths 15-token or scratch 16-token) regardless of
            # the dataset template format ("calc: a+b= " would contain spaces/letters
            # that are out-of-vocabulary for a 15-token model).
            a, b = sample.get("a", 0), sample.get("b", 0)
            answer_str: str = sample.get("answer", str(a + b))
            small_prompt = f"{a}+{b}={answer_str}"
            eq_pos_small = small_prompt.index("=")

            # ---- Small model: encode resid_mid through small SAE → f_s ----
            small_ids = torch.tensor(
                [small_tokenize(small_prompt)], device=device, dtype=torch.long
            )
            resid_cache: dict[str, torch.Tensor] = {}

            def _hook_resid_mid(act: torch.Tensor, hook, _pos=eq_pos_small, cache=resid_cache):
                cache["resid_mid"] = act[:, _pos, :].detach().clone()
                return act

            with small_model.model.hooks(
                fwd_hooks=[(f"blocks.{sae_layer}.hook_resid_mid", _hook_resid_mid)]
            ):
                small_model.model(small_ids)

            if "resid_mid" not in resid_cache:
                continue
            # Encode through small SAE → sparse feature activations
            f_s = small_sae.encode(resid_cache["resid_mid"]).squeeze(0).cpu()  # (d_tc_small,)

            # ---- Large model ----
            large_ids = tokenize_qwen_input(prompt, tokenizer, device=device).unsqueeze(0)
            eq_pos_large = _find_eq_pos_large(large_ids.squeeze(0), tokenizer)
            if eq_pos_large is None:
                continue

            large_cache: dict[str, torch.Tensor] = {}

            def _hook_large_resid_mid(
                act: torch.Tensor, hook, _pos=eq_pos_large, cache=large_cache
            ):
                cache["resid_mid"] = act[:, _pos, :].detach().clone()
                return act

            large_model.run_with_hooks(
                large_ids,
                fwd_hooks=[(f"blocks.{inject_layer}.hook_resid_mid", _hook_large_resid_mid)],
            )

            if "resid_mid" not in large_cache:
                continue

            large_resid_mid = large_cache["resid_mid"].squeeze(0)  # (d_model_large,)

            # Encode through big SAE → decode to get the SAE's reconstructed MLP output
            # W_dec is loaded lazily on first access via __getattr__
            large_feat_acts = large_transcoder.encode(large_resid_mid.unsqueeze(0))  # (1, d_tc)
            decoded_fB = (large_feat_acts @ large_transcoder.W_dec).squeeze(0)  # (d_model_large,)

            f_s_list.append(f_s)
            decoded_fB_list.append(decoded_fB.cpu())

    f_s_all = torch.stack(f_s_list)  # (N, d_s)
    decoded_fB_all = torch.stack(decoded_fB_list)  # (N, d_model_large)
    return f_s_all, decoded_fB_all


# ---------------------------------------------------------------------------
# Stage: setup
# ---------------------------------------------------------------------------


def run_setup(args: argparse.Namespace, device: torch.device) -> None:
    """Collect features and compute PCA alignment projection."""
    out_dir = Path(args.out_root) / "setup"
    out_dir.mkdir(parents=True, exist_ok=True)

    fs_path = out_dir / "f_s.pt"
    fb_path = out_dir / "decoded_fB.pt"
    proj_path = out_dir / "align_proj.pt"
    meta_path = out_dir / "setup_meta.json"

    if fs_path.exists() and fb_path.exists() and proj_path.exists() and not args.force:
        log.info("Setup already done (use --force to redo). Loading from %s", out_dir)
        return

    log.info("=== Stage: setup ===")
    dtype = parse_dtype(args.dtype)

    # Load models
    log.info("Loading large model %s", args.model)
    transcoder_set, _ = load_transcoder_from_hub(
        args.transcoder_set, dtype=dtype, lazy_encoder=False, lazy_decoder=True
    )
    large_model = AttributionModel.from_pretrained_and_transcoders(
        args.model, transcoder_set, dtype=dtype, device=device
    )
    large_model.eval()

    # Get transcoder for inject_layer
    large_transcoder: SingleLayerTranscoder = transcoder_set.transcoders[args.inject_layer]
    d_model_large = large_transcoder.d_model

    log.info("Loading small model from %s", args.small_model_path)
    small_model = _load_small_model(args, Path(args.small_model_path), device)

    log.info("Loading small SAE from %s", args.small_sae_path)
    small_sae = _load_small_sae(args, Path(args.small_sae_path), device)
    d_s = small_sae.d_transcoder  # 4096 — SAE feature dim, not raw d_model

    # Load dataset
    samples = load_addition_dataset(args.dataset_path, max_samples=args.max_samples)
    split = int(0.9 * len(samples))
    train_samples = samples[:split]

    log.info(
        "Collecting features from %d training samples (d_s=%d [SAE], d_model_large=%d)",
        len(train_samples),
        d_s,
        d_model_large,
    )
    f_s_all, decoded_fB_all = collect_features(
        small_model,
        small_sae,
        large_model,
        large_transcoder,
        train_samples,
        args.inject_layer,
        device,
    )

    log.info(
        "Collected %d pairs — f_s: %s, decoded_fB: %s",
        len(f_s_all),
        tuple(f_s_all.shape),
        tuple(decoded_fB_all.shape),
    )

    torch.save(f_s_all, fs_path)
    torch.save(decoded_fB_all, fb_path)

    # Alignment projection (PCA or probe-derived)
    labels = None
    if args.align_proj_method == "probe":
        # Carry count = number of digit positions where (a//10^k + b//10^k) % 10 >= 10
        def _carry_count(s: dict) -> int:
            a, b = s.get("a", 0), s.get("b", 0)
            count, k = 0, 0
            while a > 0 or b > 0:
                if (a % 10) + (b % 10) >= 10:
                    count += 1
                a //= 10
                b //= 10
                k += 1
            return count

        # Only keep labels for samples that made it into f_s_all
        # collect_features skips samples that fail; we rebuild labels in order
        label_list = []
        for sample in train_samples:
            prompt = sample["prompt"]
            a, b = sample.get("a", 0), sample.get("b", 0)
            small_prompt = f"{a}+{b}={sample.get('answer', str(a + b))}"
            if small_prompt.find("=") == -1:
                continue
            large_ids = tokenize_qwen_input(prompt, large_model.tokenizer, device=device)
            if _find_eq_pos_large(large_ids, large_model.tokenizer) is None:
                continue
            label_list.append(_carry_count(sample))
            if len(label_list) >= len(f_s_all):
                break
        labels = torch.tensor(label_list[: len(f_s_all)], dtype=torch.long)
        log.info("Carry-count labels: %s unique values", labels.unique().tolist())

    P = FeatureAlignmentModule.compute_align_proj(
        decoded_fB_all,
        d_mid=args.d_mid,
        method=args.align_proj_method,
        labels=labels,
    )
    torch.save(P, proj_path)

    meta = {
        "n_samples": len(f_s_all),
        "d_s": d_s,
        "d_mid": args.d_mid,
        "d_model_large": d_model_large,
        "inject_layer": args.inject_layer,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    log.info("Setup complete — outputs in %s", out_dir)


# ---------------------------------------------------------------------------
# Stage: train
# ---------------------------------------------------------------------------


def run_train(args: argparse.Namespace, mode: str, device: torch.device) -> None:
    """Train FeatureAlignmentModule for a given injection mode."""
    assert mode in ("replace", "add"), f"Unknown mode: {mode}"
    out_dir = Path(args.out_root)
    setup_dir = out_dir / "setup"
    ckpt_path = out_dir / f"module_{mode}.pt"

    log.info("=== Stage: train [%s] ===", mode)

    # Load precomputed features
    f_s_all: torch.Tensor = torch.load(setup_dir / "f_s.pt", weights_only=True)
    decoded_fB_all: torch.Tensor = torch.load(setup_dir / "decoded_fB.pt", weights_only=True)
    P: torch.Tensor = torch.load(setup_dir / "align_proj.pt", weights_only=True)

    with open(setup_dir / "setup_meta.json") as f:
        meta = json.load(f)

    d_s: int = meta["d_s"]
    d_mid: int = meta["d_mid"]
    d_model_large: int = meta["d_model_large"]
    inject_layer: int = meta["inject_layer"]

    dtype = parse_dtype(args.dtype)

    # Load large model (needed for gradient-based CE loss)
    log.info("Loading large model %s", args.model)
    transcoder_set, _ = load_transcoder_from_hub(
        args.transcoder_set, dtype=dtype, lazy_encoder=False, lazy_decoder=True
    )
    large_model = AttributionModel.from_pretrained_and_transcoders(
        args.model, transcoder_set, dtype=dtype, device=device
    )
    large_model.eval()
    for p in large_model.parameters():
        p.requires_grad_(False)

    tokenizer = large_model.tokenizer

    # Load small model + SAE (frozen; f_s already precomputed, but kept for reference)
    log.info("Loading small model from %s", args.small_model_path)
    small_model = _load_small_model(args, Path(args.small_model_path), device)
    small_model.model.eval()
    for p in small_model.model.parameters():
        p.requires_grad_(False)

    # Instantiate module
    module = FeatureAlignmentModule(d_s=d_s, d_mid=d_mid, d_model_large=d_model_large, align_proj=P)
    module = module.to(device=device, dtype=dtype)

    optimiser = torch.optim.AdamW(module.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)

    # Load dataset
    samples = load_addition_dataset(args.dataset_path, max_samples=args.max_samples)
    split = int(0.9 * len(samples))
    train_samples = samples[:split]

    # Index into precomputed tensors for fast batching
    n = len(f_s_all)

    def _get_batch(indices: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        fs = f_s_all[indices].to(device=device, dtype=dtype)
        fb = decoded_fB_all[indices].to(device=device, dtype=dtype)
        return fs, fb

    log.info(
        "Training [%s] — %d samples, %d epochs, lr=%.0e, λ_align=%.3f",
        mode,
        n,
        args.epochs,
        args.lr,
        args.lambda_align,
    )

    best_val_loss = float("inf")
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        module.train()
        perm = torch.randperm(n).tolist()
        batches = [perm[i : i + args.batch_size] for i in range(0, n, args.batch_size)]
        epoch_ce = 0.0
        epoch_align = 0.0
        n_batches = 0

        for idx_batch in batches:
            fs_batch, fb_batch = _get_batch(idx_batch)  # (B, d_s), (B, d_m)

            # Compute injection vectors for the whole batch at once
            inject_vecs = module(fs_batch)  # (B, d_model_large)

            # --- Gather per-sample metadata, skip invalid ---
            valid: list[dict] = []
            for bi, global_idx in enumerate(idx_batch):
                sample = train_samples[global_idx % len(train_samples)]
                prompt_ids = tokenize_qwen_input(sample["prompt"], tokenizer, device=device)
                answer_ids = tokenizer(sample["answer"], add_special_tokens=False).input_ids
                eq_pos = _find_eq_pos_large(prompt_ids, tokenizer)
                if eq_pos is None or not answer_ids:
                    continue
                valid.append(
                    {
                        "bi": bi,
                        "prompt_ids": prompt_ids,  # 1-D, variable length
                        "eq_pos": eq_pos,
                        "target_id": answer_ids[0],
                    }
                )

            if not valid:
                continue

            # --- Pad to max length (right-pad with 0) ---
            max_len = max(d["prompt_ids"].shape[0] for d in valid)
            B_v = len(valid)
            padded = torch.zeros(B_v, max_len, dtype=torch.long, device=device)
            attn_mask = torch.zeros(B_v, max_len, dtype=torch.long, device=device)
            last_pos: list[int] = []
            for i, d in enumerate(valid):
                L = d["prompt_ids"].shape[0]
                padded[i, :L] = d["prompt_ids"]
                attn_mask[i, :L] = 1
                last_pos.append(L - 1)

            # Inject vectors aligned to the valid subset
            ivecs = inject_vecs[[d["bi"] for d in valid]]  # (B_v, d_model_large)
            eq_pos_list = [d["eq_pos"] for d in valid]

            # --- Single batched forward pass with per-sample hook ---
            if mode == "replace":

                def _patch_batch(act: torch.Tensor, hook, _ivecs=ivecs, _positions=eq_pos_list):
                    act = act.clone()
                    for i, pos in enumerate(_positions):
                        act[i, pos, :] = _ivecs[i]
                    return act
            else:

                def _patch_batch(act: torch.Tensor, hook, _ivecs=ivecs, _positions=eq_pos_list):
                    act = act.clone()
                    for i, pos in enumerate(_positions):
                        act[i, pos, :] = act[i, pos, :] + _ivecs[i]
                    return act

            logits = large_model.run_with_hooks(
                padded,
                fwd_hooks=[(f"blocks.{inject_layer}.hook_mlp_out", _patch_batch)],
                attention_mask=attn_mask,
            )  # (B_v, max_len, vocab)

            # Gather last real token for each sample
            last_logits = torch.stack(
                [logits[i, last_pos[i], :] for i in range(B_v)]
            )  # (B_v, vocab)
            targets = torch.tensor([d["target_id"] for d in valid], device=device, dtype=torch.long)

            loss_ce = F.cross_entropy(last_logits, targets)
            loss_align = module.align_loss(fs_batch, fb_batch)
            loss = loss_ce + args.lambda_align * loss_align

            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
            optimiser.step()

            epoch_ce += loss_ce.item()
            epoch_align += loss_align.item()
            n_batches += 1

        scheduler.step()

        avg_ce = epoch_ce / max(n_batches, 1)
        avg_align = epoch_align / max(n_batches, 1)

        if epoch % max(1, args.epochs // 10) == 0 or epoch == args.epochs:
            log.info(
                "[%s] Epoch %3d/%d — CE=%.4f  ALIGN=%.4f",
                mode,
                epoch,
                args.epochs,
                avg_ce,
                avg_align,
            )

        history.append({"epoch": epoch, "ce": avg_ce, "align": avg_align})

        # Simple validation: save if best
        if avg_ce < best_val_loss:
            best_val_loss = avg_ce
            module.save(ckpt_path)

    # Save training history
    hist_path = out_dir / f"train_history_{mode}.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    log.info("[%s] Training done — best CE=%.4f — checkpoint: %s", mode, best_val_loss, ckpt_path)


# ---------------------------------------------------------------------------
# Stage: eval
# ---------------------------------------------------------------------------


def run_eval(args: argparse.Namespace, mode: str, device: torch.device) -> dict[str, Any]:
    """Evaluate a trained checkpoint on the test split."""
    assert mode in ("replace", "add"), f"Unknown mode: {mode}"
    out_dir = Path(args.out_root)
    ckpt_path = out_dir / f"module_{mode}.pt"
    setup_dir = out_dir / "setup"

    log.info("=== Stage: eval [%s] ===", mode)

    if not ckpt_path.exists():
        log.warning("Checkpoint not found: %s — skipping eval for %s", ckpt_path, mode)
        return {}

    with open(setup_dir / "setup_meta.json") as f:
        meta = json.load(f)
    inject_layer: int = meta["inject_layer"]

    dtype = parse_dtype(args.dtype)
    module = FeatureAlignmentModule.load(ckpt_path, device=device)
    module = module.to(dtype=dtype)
    module.eval()

    log.info("Loading large model for eval")
    transcoder_set, _ = load_transcoder_from_hub(
        args.transcoder_set, dtype=dtype, lazy_encoder=False, lazy_decoder=True
    )
    large_model = AttributionModel.from_pretrained_and_transcoders(
        args.model, transcoder_set, dtype=dtype, device=device
    )
    large_model.eval()
    tokenizer = large_model.tokenizer

    small_model = _load_small_model(args, Path(args.small_model_path), device)
    small_model.model.eval()
    small_tokenize = get_small_model_tokenizer(small_model)

    small_sae = _load_small_sae(args, Path(args.small_sae_path), device)
    small_sae.eval()
    sae_layer = small_model.n_layers - 1  # last layer — carry is most active here

    samples = load_addition_dataset(args.dataset_path, max_samples=args.max_samples)
    split = int(0.9 * len(samples))
    test_samples = samples[split : split + args.num_eval_samples]

    n_correct_before = 0
    n_correct_after = 0
    n_total = 0
    sum_logp_before = 0.0
    sum_logp_after = 0.0

    with torch.no_grad():
        for sample in tqdm(test_samples, desc=f"Eval [{mode}]", leave=False):
            prompt: str = sample["prompt"]
            answer: str = sample["answer"]

            # Small-model prompt: only {digits, +, =} — safe for any small vocab
            a, b = sample.get("a", 0), sample.get("b", 0)
            small_prompt = f"{a}+{b}={answer}"
            eq_pos_small = small_prompt.index("=")

            # f_s: small SAE features at '=' (hook_resid_mid → small_sae.encode)
            small_ids = torch.tensor(
                [small_tokenize(small_prompt)], device=device, dtype=torch.long
            )
            resid_cache: dict[str, torch.Tensor] = {}

            def _hook_resid_mid(act, hook, _pos=eq_pos_small, cache=resid_cache):
                cache["resid_mid"] = act[:, _pos, :].detach().clone()
                return act

            with small_model.model.hooks(
                fwd_hooks=[(f"blocks.{sae_layer}.hook_resid_mid", _hook_resid_mid)]
            ):
                small_model.model(small_ids)

            if "resid_mid" not in resid_cache:
                continue

            f_s = small_sae.encode(resid_cache["resid_mid"]).to(dtype=dtype)  # (1, d_tc_small)
            inject_v = module(f_s).squeeze(0)  # (d_model_large,)
            if n_total < 5:
                log.info(
                    "  [debug] sample=%d  f_s nnz=%d  f_s_norm=%.3f  inject_v_norm=%.3f",
                    n_total,
                    (f_s.abs() > 1e-6).sum().item(),
                    f_s.norm().item(),
                    inject_v.norm().item(),
                )

            # Large model: tokenise prompt only
            prompt_ids = tokenize_qwen_input(prompt, tokenizer, device=device)
            eq_pos_large = _find_eq_pos_large(prompt_ids, tokenizer)
            if eq_pos_large is None:
                continue

            answer_ids = tokenizer(answer, add_special_tokens=False).input_ids
            if not answer_ids:
                continue
            first_answer_tok = answer_ids[0]

            # Before (no injection)
            logits_before = large_model(prompt_ids.unsqueeze(0))
            pred_before = int(logits_before[0, -1, :].argmax())

            # After injection
            if mode == "replace":

                def _patch(act, hook, _v=inject_v, _pos=eq_pos_large):
                    if act.shape[1] <= _pos:
                        return act
                    act = act.clone()
                    act[:, _pos, :] = _v
                    return act
            else:

                def _patch(act, hook, _v=inject_v, _pos=eq_pos_large):
                    if act.shape[1] <= _pos:
                        return act
                    act = act.clone()
                    act[:, _pos, :] = act[:, _pos, :] + _v
                    return act

            logits_after = large_model.run_with_hooks(
                prompt_ids.unsqueeze(0),
                fwd_hooks=[(f"blocks.{inject_layer}.hook_mlp_out", _patch)],
            )
            pred_after = int(logits_after[0, -1, :].argmax())

            tok = torch.tensor([first_answer_tok], device=logits_before.device)
            logp_before = F.log_softmax(logits_before[0, -1, :], dim=-1)[tok].item()
            logp_after = F.log_softmax(logits_after[0, -1, :], dim=-1)[tok].item()

            if pred_before == first_answer_tok:
                n_correct_before += 1
            if pred_after == first_answer_tok:
                n_correct_after += 1
            sum_logp_before += logp_before
            sum_logp_after += logp_after
            n_total += 1

    acc_before = 100.0 * n_correct_before / max(n_total, 1)
    acc_after = 100.0 * n_correct_after / max(n_total, 1)
    delta = acc_after - acc_before
    mean_logp_before = sum_logp_before / max(n_total, 1)
    mean_logp_after = sum_logp_after / max(n_total, 1)
    mean_p_before = float(torch.tensor(mean_logp_before).exp())
    mean_p_after = float(torch.tensor(mean_logp_after).exp())

    results = {
        "mode": mode,
        "n_total": n_total,
        "acc_before": acc_before,
        "acc_after": acc_after,
        "delta": delta,
        "mean_logp_before": mean_logp_before,
        "mean_logp_after": mean_logp_after,
        "mean_p_before": mean_p_before,
        "mean_p_after": mean_p_after,
        "delta_logp": mean_logp_after - mean_logp_before,
    }

    log.info(
        "[%s] acc_before=%.2f%%  acc_after=%.2f%%  delta=%+.2f%%  "
        "P(correct) before=%.4f  after=%.4f  Δ=%+.4f",
        mode,
        acc_before,
        acc_after,
        delta,
        mean_p_before,
        mean_p_after,
        mean_p_after - mean_p_before,
    )

    eval_path = Path(args.out_root) / f"eval_{mode}.json"
    with open(eval_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


# ---------------------------------------------------------------------------
# Stage: compare
# ---------------------------------------------------------------------------


def run_compare(args: argparse.Namespace) -> None:
    """Print a side-by-side comparison of Replace vs Add modes."""
    out_dir = Path(args.out_root)
    results: dict[str, dict] = {}
    for mode in ("replace", "add"):
        p = out_dir / f"eval_{mode}.json"
        if p.exists():
            with open(p) as f:
                results[mode] = json.load(f)
        else:
            log.warning("eval_%s.json not found — run --eval --mode %s first", mode, mode)

    if not results:
        log.error("No eval results found in %s", out_dir)
        return

    COL = 18
    print("\n" + "=" * 68)
    print("  KNOWLEDGE EDITING — Replace vs. Add injection comparison")
    print("=" * 68)
    print(f"  {'Metric':<30s}  {'Replace':>{COL}s}  {'Add':>{COL}s}")
    print("-" * 68)

    def row(label: str, rep_val: str, add_val: str) -> None:
        print(f"  {label:<30s}  {rep_val:>{COL}s}  {add_val:>{COL}s}")

    rep = results.get("replace", {})
    add = results.get("add", {})

    def _fmt(d: dict, key: str, fmt: str, suffix: str = "") -> str:
        return (fmt.format(d[key]) + suffix) if key in d else "N/A"

    row("N samples", str(rep.get("n_total", "N/A")), str(add.get("n_total", "N/A")))
    row(
        "Acc before (baseline)",
        _fmt(rep, "acc_before", "{:.2f}", "%"),
        _fmt(add, "acc_before", "{:.2f}", "%"),
    )
    row(
        "Acc after injection",
        _fmt(rep, "acc_after", "{:.2f}", "%"),
        _fmt(add, "acc_after", "{:.2f}", "%"),
    )
    row("Acc delta", _fmt(rep, "delta", "{:+.2f}", "%"), _fmt(add, "delta", "{:+.2f}", "%"))
    print("-" * 68)
    row(
        "P(correct) before",
        _fmt(rep, "mean_p_before", "{:.4f}"),
        _fmt(add, "mean_p_before", "{:.4f}"),
    )
    row(
        "P(correct) after", _fmt(rep, "mean_p_after", "{:.4f}"), _fmt(add, "mean_p_after", "{:.4f}")
    )
    row(
        "ΔP(correct)",
        (
            _fmt(rep, "mean_p_after", "{:+.4f}")
            if "mean_p_before" not in rep
            else f"{rep['mean_p_after'] - rep['mean_p_before']:+.4f}"
        ),
        (
            _fmt(add, "mean_p_after", "{:+.4f}")
            if "mean_p_before" not in add
            else f"{add['mean_p_after'] - add['mean_p_before']:+.4f}"
        ),
    )
    row("Δlog P(correct)", _fmt(rep, "delta_logp", "{:+.4f}"), _fmt(add, "delta_logp", "{:+.4f}"))
    print("=" * 68)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Feature-level knowledge editing experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Stage flags
    parser.add_argument("--setup", action="store_true", help="Collect features + compute PCA")
    parser.add_argument("--train", action="store_true", help="Train the alignment module")
    parser.add_argument("--eval", action="store_true", help="Evaluate a trained checkpoint")
    parser.add_argument(
        "--compare", action="store_true", help="Compare replace vs add eval results"
    )
    parser.add_argument("--all", action="store_true", help="Run setup → train → eval → compare")

    parser.add_argument(
        "--mode",
        choices=["replace", "add", "both"],
        default="both",
        help="Injection mode for --train / --eval",
    )

    # Paths
    parser.add_argument("--out_root", default="runs/knowledge_editing", help="Output directory")
    parser.add_argument(
        "--dataset_path",
        default="data/addition_dataset.jsonl",
        help="Path to addition dataset JSONL",
    )
    parser.add_argument(
        "--small_model_path",
        default="runs/stitching/rope/small_model.pt",
        help="Path to trained small model checkpoint (.pt)",
    )
    parser.add_argument(
        "--small_sae_path",
        default="runs/stitching/rope/small_sae.safetensors",
        help="Path to small model SAE checkpoint (.safetensors) — trained by stitching pipeline",
    )
    # _load_small_sae reads args.small_sae_layer and args.small_model_layers
    parser.add_argument(
        "--small_sae_layer",
        type=int,
        default=None,
        help="Which small-model layer the SAE was trained on (default: n_layers - 1)",
    )
    parser.add_argument(
        "--small_model_layers",
        type=int,
        default=2,
        help="Number of layers in the small model (used as fallback for SAE layer inference)",
    )

    # Large model
    parser.add_argument("--model", default=_LARGE_MODEL)
    parser.add_argument("--transcoder_set", default=_TRANSCODER_SET)
    parser.add_argument("--inject_layer", type=int, default=_INJECT_LAYER)

    # Alignment projection method
    parser.add_argument(
        "--align_proj_method",
        choices=["pca", "probe"],
        default="pca",
        help=(
            "How to compute the fixed alignment projection P. "
            "'pca': top-d_mid PCA directions of decoded_f_B (unsupervised). "
            "'probe': one-vs-rest logistic probes per carry count, padded with "
            "orthogonal PCA directions to fill d_mid (supervised, needs carry labels)."
        ),
    )

    # Module hyper-params
    parser.add_argument("--d_mid", type=int, default=_D_MID)
    parser.add_argument("--lambda_align", type=float, default=_LAMBDA_ALIGN)
    parser.add_argument("--lr", type=float, default=_LR)
    parser.add_argument("--epochs", type=int, default=_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=_BATCH_SIZE)
    parser.add_argument("--max_samples", type=int, default=_MAX_SAMPLES)
    parser.add_argument("--num_eval_samples", type=int, default=500)

    # Misc
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--force", action="store_true", help="Redo setup even if output exists")

    # Forwarded to _load_small_model (reads args.hub_model and args.small_model_num_digits)
    parser.add_argument("--hub_model", default="", help="Hub model ID (empty = scratch model)")
    parser.add_argument(
        "--small_model_num_digits",
        type=int,
        default=5,
        help="Digit count for small model (only used for hub models)",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not any([args.setup, args.train, args.eval, args.compare, args.all]):
        parser.print_help()
        sys.exit(0)

    device = get_default_device()
    log.info("Device: %s", device)

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    modes = ["replace", "add"] if args.mode == "both" else [args.mode]

    if args.all or args.setup:
        run_setup(args, device)

    if args.all or args.train:
        for mode in modes:
            run_train(args, mode, device)

    if args.all or args.eval:
        for mode in modes:
            run_eval(args, mode, device)

    if args.all or args.compare:
        run_compare(args)


if __name__ == "__main__":
    main()
