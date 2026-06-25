"""Show router activation weights on compound arithmetic prompts.

No model needed — only loads the router (~KB) and tokenizer.
Demonstrates whether the FSM correctly fires for both primitives in compound expressions.
"""

import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer
from scripts.model_config import default_model

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.fsm_router.fsm import PrimitiveRouter
from experiments.fsm_router.predicates import N_PREDICATES
from experiments.fsm_router.primitives import FSM_SPECS
from experiments.fsm_router.steer_with_router import build_A_tok

_MODEL = default_model()
_ROUTER_PATH = Path("runs/fsm_router/router.pt")

PRIM_NAMES = [name for name, _ in FSM_SPECS]

COMPOUND_PROMPTS = [
    # (prompt, expected_answer, description)
    ("calc: 12+3*4= ", str(12 + 3 * 4), "addition + multiplication"),
    ("calc: 100+25*4= ", str(100 + 25 * 4), "addition + multiplication"),
    ("calc: 7+8*9= ", str(7 + 8 * 9), "addition + multiplication"),
    ("calc: 50+120%7= ", str(50 + 120 % 7), "addition + modular"),
    ("calc: 13+200%9= ", str(13 + 200 % 9), "addition + modular"),
    ("calc: 200-150%7= ", str(200 - 150 % 7), "subtraction + modular"),
    # single-primitive controls
    ("calc: 347+289= ", str(347 + 289), "addition only (control)"),
    ("calc: 7*8= ", str(7 * 8), "multiplication only (control)"),
    ("calc: 120%7= ", str(120 % 7), "modular only (control)"),
]


def show_weights(tok_strings, A_tok, threshold=0.05):
    cols = "  ".join(f"{p[:5]:>6}" for p in PRIM_NAMES)
    print(f"  {'token':>10}  {cols}")
    print(f"  {'-' * 50}")
    for t, tok in enumerate(tok_strings):
        row = A_tok[t]
        if row.max().item() < threshold:
            continue  # skip zero rows
        vals = "  ".join(f"{row[k].item():>6.3f}" for k in range(len(PRIM_NAMES)))
        active = [PRIM_NAMES[k][:5] for k in range(len(PRIM_NAMES)) if row[k].item() > threshold]
        print(f"  {tok!r:>10}  {vals}   ← {', '.join(active)}")


def main():
    print("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(_MODEL, trust_remote_code=True)

    print("Loading router...")
    router = PrimitiveRouter(FSM_SPECS, N_PREDICATES)
    router.load_state_dict(torch.load(_ROUTER_PATH, map_location="cpu"))
    router.eval()

    print(f"\nPrimitives: {PRIM_NAMES}\n")

    for prompt, expected, desc in COMPOUND_PROMPTS:
        print(f"{'─' * 60}")
        print(f"  {desc}")
        print(f"  Prompt: {prompt!r}   expected: {expected!r}")
        ids = tok.encode(prompt, add_special_tokens=False)
        tok_strings = tok.convert_ids_to_tokens(ids)
        A_tok = build_A_tok(tok_strings, router, device=torch.device("cpu"))
        show_weights(tok_strings, A_tok)

        # Which primitives have any activation?
        active_prims = [
            PRIM_NAMES[k] for k in range(len(PRIM_NAMES)) if A_tok[:, k].max().item() > 0.05
        ]
        print(f"  → Active primitives: {active_prims}")


if __name__ == "__main__":
    main()
