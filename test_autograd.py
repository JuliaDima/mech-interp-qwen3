import torch
from nnsight import LanguageModel
from transformers import Qwen2Config, Qwen2ForCausalLM

config = Qwen2Config(
    vocab_size=1000,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=2,
    num_key_value_heads=2,
)
model = Qwen2ForCausalLM(config)
model.train()
nn_model = LanguageModel(model, dispatch=True)
input_ids = torch.randint(0, 100, (1, 5))

with nn_model.trace(input_ids) as tracer:
    out = nn_model.model.layers[1].mlp.output.save()
    loss = nn_model.lm_head.output.sum().save()

print("Trace finished")
loss_val = loss
out_val = out[0] if isinstance(out, tuple) else out

print("Out val requires grad:", out_val.requires_grad)
print("Loss val requires grad:", loss_val.requires_grad)

grad = torch.autograd.grad(loss_val, out_val, retain_graph=True, allow_unused=True)[0]
if grad is not None:
    print("Success! Grad shape:", grad.shape)
else:
    print("Failed: Grad is None")
