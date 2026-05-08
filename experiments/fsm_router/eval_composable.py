"""Evaluate FSM router composability on compound arithmetic expressions.

Tests prompts where two primitives co-occur (e.g. addition + multiplication,
addition + modular) and shows:
  1. Router activation weights per token position
  2. Unsteered vs steered model output
  3. Whether correct answers are recovered

Usage:
    python -m experiments.fsm_router.eval_composable
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.fsm_router.fsm import PrimitiveRouter
from experiments.fsm_router.predicates import N_PREDICATES
from experiments.fsm_router.primitives import FSM_SPECS
from experiments.fsm_router.steer_with_router import build_A_tok, load_svecs, make_steer_hooks
from mechinterp_qwen3.attribution_model import AttributionModel
from mechinterp_qwen3.utils.hf_utils import load_transcoder_from_hub
from mechinterp_qwen3.utils.model_utils import get_default_device

_MODEL = "Qwen/Qwen3-4B"
_ROUTER_PATH = Path("runs/fsm_router/router.pt")
_SVEC_DIR = Path("runs/fsm_router/svecs")
_BEST_LAYER = 11
_ALPHA = 5.0
_MAX_NEW = 30

# ── compound prompts: (prompt, expected_answer, description) ──────────────────
COMPOUND_PROMPTS = [
    # addition + multiplication
    ("calc: 12+3*4= ", str(12 + 3 * 4), "12 + 3×4  (PEMDAS: 24)"),
    ("calc: 5+10*3= ", str(5 + 10 * 3), "5 + 10×3  (PEMDAS: 35)"),
    ("calc: 7+8*9= ", str(7 + 8 * 9), "7 + 8×9   (PEMDAS: 79)"),
    ("calc: 100+25*4= ", str(100 + 25 * 4), "100 + 25×4 (PEMDAS: 200)"),
    # addition + modular
    ("calc: 50+120%7= ", str(50 + 120 % 7), "50 + 120%7  (PEMDAS: 51)"),
    ("calc: 13+200%9= ", str(13 + 200 % 9), "13 + 200%9  (PEMDAS: 22)"),
    ("calc: 100+77%11= ", str(100 + 77 % 11), "100 + 77%11 (PEMDAS: 111)"),
    # subtraction + modular
    ("calc: 200-150%7= ", str(200 - 150 % 7), "200 - 150%7 (PEMDAS: 194)"),
]


def load_svecs(primitives: list[str], dtype=torch.bfloat16):
    svecs = {}
    for prim in primitives:
        path = _SVEC_DIR / f"{prim}.pt"
        if not path.exists():
            print(f"  [warn] SVec not found: {path}")
            continue
        d = torch.load(path, map_location="cpu")
        svecs[prim] = {int(l): v.to(dtype) for l, v in d["svecs"].items()}
    return svecs


def decode_generated(model, prompt_ids, hooks, max_new=_MAX_NEW, device="cpu"):
    generated = []
    with torch.no_grad():
        for _ in range(max_new):
            input_ids = torch.tensor([prompt_ids + generated], dtype=torch.long, device=device)
            logits = model.run_with_hooks(input_ids, fwd_hooks=hooks)[0, -1, :]
            pred_id = int(logits.argmax())
            tok = model.tokenizer.convert_ids_to_tokens([pred_id])[0]
            if pred_id == model.tokenizer.eos_token_id or tok in ("<|endoftext|>", "<|im_end|>"):
                break
            generated.append(pred_id)
    return model.tokenizer.decode(generated).strip()


def show_router_weights(tok_strings, A_tok, prim_names):
    """Print per-token router activations."""
    K = A_tok.shape[1]
    header = f"{'Token':>12}  " + "  ".join(f"{p:>10}" for p in prim_names)
    print(f"  {header}")
    print(f"  {'-' * len(header)}")
    for t, tok in enumerate(tok_strings):
        weights = "  ".join(f"{A_tok[t, k].item():>10.3f}" for k in range(K))
        marker = " ◀" if A_tok[t].max().item() > 0.1 else ""
        print(f"  {tok!r:>12}  {weights}{marker}")


def main():
    device = get_default_device()
    print(f"Device: {device}")

    print("Loading model...")
    transcoder_set, _ = load_transcoder_from_hub(
        "mwhanna/qwen3-4b-transcoders", dtype=torch.bfloat16, lazy_encoder=True, lazy_decoder=True
    )
    model = AttributionModel.from_pretrained_and_transcoders(
        _MODEL, transcoder_set, dtype=torch.bfloat16, device=device
    )
    model.eval()

    print("Loading router...")
    router = PrimitiveRouter(FSM_SPECS, N_PREDICATES)
    router.load_state_dict(torch.load(_ROUTER_PATH, map_location="cpu"))
    router.eval()

    primitives = ["addition", "subtraction", "multiplication", "modular"]
    prim_names = [p[:4] for p in primitives]

    print("Loading SVecs...")
    svecs = load_svecs(primitives)

    results = []
    print("\n" + "=" * 72)

    for prompt, expected, desc in COMPOUND_PROMPTS:
        print(f"\n{desc}")
        print(f"  Prompt:   {prompt!r}")
        print(f"  Expected: {expected!r}")

        # tokenize prompt
        prompt_ids = model.tokenizer.encode(prompt, add_special_tokens=False)
        tok_strings = model.tokenizer.convert_ids_to_tokens(prompt_ids)

        # router weights
        A_tok = build_A_tok(tok_strings, router, device=device)
        print("\n  Router activations:")
        show_router_weights(tok_strings, A_tok, prim_names)

        def _correct(output: str, expected: str) -> bool:
            # "contains" match: model often shows work ("7+72=79. 79 is"), so
            # exact-string match would miss correct answers. We check both.
            return (output.strip() == expected.strip()) or (expected in output)

        # unsteered
        out_base = decode_generated(model, prompt_ids, hooks=[], device=device)
        correct_base = _correct(out_base, expected)

        # steered at best layer
        hooks = make_steer_hooks(
            tok_strings,
            router,
            svecs,
            primitives,
            layers=[_BEST_LAYER],
            alpha=_ALPHA,
            local=True,
            device=device,
        )
        out_steer = decode_generated(model, prompt_ids, hooks=hooks, device=device)
        correct_steer = _correct(out_steer, expected)

        print(f"\n  Unsteered: {out_base!r}  {'✓' if correct_base  else '✗'}")
        print(f"  Steered:   {out_steer!r}  {'✓' if correct_steer else '✗'}")
        print("=" * 72)

        results.append(
            {
                "prompt": prompt,
                "expected": expected,
                "description": desc,
                "unsteered": out_base,
                "steered": out_steer,
                "base_correct": correct_base,
                "steer_correct": correct_steer,
            }
        )

    # summary
    n = len(results)
    n_base = sum(r["base_correct"] for r in results)
    n_steer = sum(r["steer_correct"] for r in results)
    print(f"\nSummary: {n_base}/{n} unsteered  →  {n_steer}/{n} steered")

    out_path = Path("runs/fsm_router/composable_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
