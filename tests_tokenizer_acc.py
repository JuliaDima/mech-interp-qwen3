import random
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Fix sys.path for the imports
_REPO_ROOT = Path("/home/eid23/mechinterp-qwen-3B-Instruct/mechinterp-qwen3")
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

# ruff: noqa: E402
from mechinterp_qwen3.utils.token_utils import tokenize_qwen_input

model_name = "Qwen/Qwen3-4B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16
)
model.eval()

# Let's use a mix of difficulties
samples = []
for _ in range(50):
    a = random.randint(0, 30)
    b = random.randint(0, 30)
    samples.append({"a": a, "b": b, "prompt": f"calc: {a}+{b}= ", "ans": str(a + b)})

correct_no_bos = 0
correct_with_bos = 0
correct_t5 = 0

print("Testing accuracy on 50 math prompts...")
for sample in samples:
    prompt = sample["prompt"]
    answer_str = sample["ans"]
    a, b = sample["a"], sample["b"]

    print(f"Prompt: {prompt!r}")
    print(f"  Truth: {answer_str}")

    # --- T0 logic ---
    tokens_no_bos = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(
        "cuda"
    )
    tokens_with_bos = tokenize_qwen_input(prompt, tokenizer, device="cuda").unsqueeze(0)

    # --- T5 logic (ChatML) ---
    prompt_t5 = f"<|im_start|>user\nCalculate {a}+{b}<|im_end|>\n<|im_start|>assistant\n"
    tokens_t5 = tokenize_qwen_input(prompt_t5, tokenizer, device="cuda").unsqueeze(0)

    with torch.no_grad():
        out_no_bos = model(tokens_no_bos).logits[0, -1].argmax().item()
        out_with_bos = model(tokens_with_bos).logits[0, -1].argmax().item()
        out_t5 = model(tokens_t5).logits[0, -1].argmax().item()

    pred_no_bos = tokenizer.decode([out_no_bos]).strip()
    pred_with_bos = tokenizer.decode([out_with_bos]).strip()
    pred_t5 = tokenizer.decode([out_t5]).strip()

    print(f"  Pred (T0 No BOS): {pred_no_bos!r}")
    print(f"  Pred (T0 With BOS): {pred_with_bos!r}")
    print(f"  Pred (T5 ChatML): {pred_t5!r}")

    if answer_str.startswith(pred_no_bos) and pred_no_bos:
        correct_no_bos += 1
    if answer_str.startswith(pred_with_bos) and pred_with_bos:
        correct_with_bos += 1
    if answer_str.startswith(pred_t5) and pred_t5:
        correct_t5 += 1

print("--- RESULTS ---")
print(f"Accuracy T0 NO BOS: {correct_no_bos}/50")
print(f"Accuracy T0 WITH BOS: {correct_with_bos}/50")
print(f"Accuracy T5 ChatML: {correct_t5}/50")
