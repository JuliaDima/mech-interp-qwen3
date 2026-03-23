import atexit
import os
import tempfile
from typing import Literal

import torch
from safetensors.torch import load_file, save_file
from torch import nn

_offload_files = set()

_TEMP_PREFIX = "safetensors-offload-YqKRr8m3-"


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
    org_device = next(module.parameters()).device
    with tempfile.NamedTemporaryFile(prefix=_TEMP_PREFIX, delete=False) as f:
        save_file(module.state_dict(), f.name)
        _offload_files.add(f.name)

    module.to(device="meta")

    def reload_handle(device=None):
        target_device = str(device or org_device)
        module.load_state_dict(load_file(f.name, device=target_device), assign=True)
        os.remove(f.name)
        _offload_files.remove(f.name)

    return reload_handle


def cpu_offload_module(module):
    org_device = next(module.parameters()).device
    module.to(device="cpu")

    def reload_handle():
        module.to(device=org_device)

    return reload_handle


def offload_modules(
    modules: list | nn.Module | nn.ModuleList | nn.ModuleDict | nn.Sequential,
    offload_type: Literal["cpu", "disk"],
) -> list:
    """Offload one or more modules to CPU or disk.

    Args:
        modules: A single module, list of modules, or PyTorch module container
                 (ModuleList, ModuleDict, Sequential)
        offload_type: Type of offload - "cpu" or "disk"

    Returns:
        List of reload handles, one per module
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
    """Pick the smallest logit set whose cumulative prob >= *desired_logit_prob*.

    Args:
        logits: ``(d_vocab,)`` vector (single position).
        unembed_proj: ``(d_model, d_vocab)`` unembedding matrix.
        max_n_logits: Hard cap *k*.
        desired_logit_prob: Cumulative probability threshold *p*.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            * logit_indices - ``(k,)`` vocabulary ids.
            * logit_probs   - ``(k,)`` softmax probabilities.
            * demeaned_vecs - ``(k, d_model)`` unembedding columns, demeaned.
    """

    probs = torch.softmax(logits, dim=-1)
    top_p, top_idx = torch.topk(probs, max_n_logits)
    cutoff = int(torch.searchsorted(torch.cumsum(top_p, 0), desired_logit_prob)) + 1
    top_p, top_idx = top_p[:cutoff], top_idx[:cutoff]

    if unembed_proj.shape[0] == logits.shape[0]:
        # Shape is (d_vocab, d_model) – first axis is vocabulary.
        cols = unembed_proj[top_idx]  # (k, d_model)
        demean = unembed_proj.mean(dim=0, keepdim=True)  # (1, d_model)
        demeaned_vecs = cols - demean  # (k, d_model)

    else:
        # Shape is (d_model, d_vocab) – second axis is vocabulary.
        cols = unembed_proj[:, top_idx]  # (d_model, k)
        demean = unembed_proj.mean(dim=-1, keepdim=True)  # (d_model, 1)
        demeaned_vecs = (cols - demean).T  # (k, d_model)

    return top_idx, top_p, demeaned_vecs


def get_default_device() -> "torch.device":
    """Get the default device, preferring CUDA if available."""
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


DTYPE_MAP = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def parse_dtype(dtype_str: str, default: torch.dtype = torch.bfloat16) -> torch.dtype:
    """Parse dtype string to torch.dtype.

    Args:
        dtype_str: String representation of dtype ("float32", "bfloat16", or "float16")
        default: Default dtype to return if dtype_str is not recognized

    Returns:
        Corresponding torch.dtype

    Examples:
        >>> parse_dtype("float32")
        torch.float32
        >>> parse_dtype("bfloat16")
        torch.bfloat16
        >>> parse_dtype("unknown")
        torch.bfloat16
    """
    return DTYPE_MAP.get(dtype_str, default)
