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
    layer = nn_model.model.layers[1]

    # Check what type layer.mlp.input is. Normally it's a tuple.
    mlp_in = layer.mlp.input[0][0]

    # Save it to see if it works
    mlp_in_save = mlp_in.save()

    # Do a dummy intervention
    layer.mlp.output = layer.mlp.output + 1.0

print("Trace finished!")
print("mlp_in_save value type:", type(mlp_in_save.value))
if isinstance(mlp_in_save.value, torch.Tensor):
    print("mlp_in_save shape:", mlp_in_save.value.shape)
