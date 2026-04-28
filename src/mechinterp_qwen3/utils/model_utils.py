import atexit
import os
import tempfile
from typing import Literal

import torch
from safetensors.torch import load_file, save_file
from torch import nn

_offload_files = set()

_TEMP_PREFIX = "safetensors-offload-mq3-"


@atexit.register
def cleanup_offload_files():
    for f in _offload_files:
        if os.path.exists(f):
            os.remove(f)


def cleanup_all_offload_files():
    temp_dir = tempfile.gettempdir()
    n_removed = 0
    for f in os.listdir(temp_dir):
        if f.startswith(_TEMP_PREFIX):
            os.remove(os.path.join(temp_dir, f))
            n_removed += 1
    return n_removed


def disk_offload_module(module):
    original_device = next(module.parameters()).device
    with tempfile.NamedTemporaryFile(prefix=_TEMP_PREFIX, delete=False) as f:
        save_file(module.state_dict(), f.name)
        _offload_files.add(f.name)

    module.to(device="meta")

    def reload_handle(device=None):
        target_device = str(device or original_device)
        module.load_state_dict(load_file(f.name, device=target_device), assign=True)
        os.remove(f.name)
        _offload_files.remove(f.name)

    return reload_handle


def cpu_offload_module(module):
    original_device = next(module.parameters()).device
    module.to(device="cpu")

    def reload_handle():
        module.to(device=original_device)

    return reload_handle


def offload_modules(
    modules: list | nn.Module | nn.ModuleList | nn.ModuleDict | nn.Sequential,
    offload_type: Literal["cpu", "disk"],
) -> list:
    """Move modules to CPU or disk to free GPU memory, returning reload handles.

    Args:
        modules: A single module, list of modules, or any PyTorch module container
        offload_type: "cpu" moves tensors to RAM; "disk" serializes to a temp file

    Returns:
        List of callables — invoke each to restore the module to its original device
    """
    offload_fn = disk_offload_module if offload_type == "disk" else cpu_offload_module

    if isinstance(modules, nn.ModuleDict):
        mods = modules.values()
    elif isinstance(modules, list | nn.ModuleList | nn.Sequential):
        mods = modules
    else:
        mods = [modules]
    return [offload_fn(module) for module in mods]


@torch.no_grad()
def compute_salient_logits(
    logits: torch.Tensor,
    unembed_proj: torch.Tensor,
    *,
    max_n_logits: int = 10,
    desired_logit_prob: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select the minimal top-k tokens whose cumulative probability meets the threshold.

    Args:
        logits: ``(d_vocab,)`` raw logit vector at a single position.
        unembed_proj: Unembedding matrix, ``(d_model, d_vocab)`` or ``(d_vocab, d_model)``.
        max_n_logits: Upper bound on k.
        desired_logit_prob: Stop adding tokens once cumulative probability exceeds this.

    Returns:
        logit_indices: ``(k,)`` vocabulary ids of selected tokens.
        logit_probs:   ``(k,)`` softmax probabilities.
        demeaned_vecs: ``(k, d_model)`` demeaned unembedding columns.
    """

    probs = torch.softmax(logits, dim=-1)
    top_p, top_idx = torch.topk(probs, max_n_logits)
    cutoff = int(torch.searchsorted(torch.cumsum(top_p, 0), desired_logit_prob)) + 1
    top_p, top_idx = top_p[:cutoff], top_idx[:cutoff]

    if unembed_proj.shape[0] == logits.shape[0]:
        # (d_vocab, d_model) layout — first axis is vocabulary
        cols = unembed_proj[top_idx]
        demean = unembed_proj.mean(dim=0, keepdim=True)
        demeaned_vecs = cols - demean

    else:
        # (d_model, d_vocab) layout — second axis is vocabulary
        cols = unembed_proj[:, top_idx]
        demean = unembed_proj.mean(dim=-1, keepdim=True)
        demeaned_vecs = (cols - demean).T

    return top_idx, top_p, demeaned_vecs


def get_default_device() -> "torch.device":
    """Return a CUDA device if available, otherwise CPU."""
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


DTYPE_MAP = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def parse_dtype(dtype_str: str, default: torch.dtype = torch.bfloat16) -> torch.dtype:
    """Resolve a dtype string to a torch.dtype, using default for unrecognised inputs.

    Examples:
        >>> parse_dtype("float32")
        torch.float32
        >>> parse_dtype("unknown")
        torch.bfloat16
    """
    return DTYPE_MAP.get(dtype_str, default)
