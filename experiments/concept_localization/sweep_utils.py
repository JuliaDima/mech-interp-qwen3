"""Shared utilities for concept-specific transcoder feature sweeps."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mechinterp_qwen3.transcoder.activation_functions import JumpReLU


def apply_transcoder_all(model, layer: int, H_l: np.ndarray) -> np.ndarray:
    """Apply JumpReLU transcoder encoding for all features at a layer.

    H_l: (N, d_model) float32.  Returns (N, d_tc) float32.
    """
    tc = model.transcoders[layer]
    H_t = torch.from_numpy(H_l).to(device=model.cfg.device, dtype=torch.float32)
    W_enc = tc.W_enc.detach().to(H_t.device, H_t.dtype)
    b_enc = tc.b_enc.detach().to(H_t.device, H_t.dtype)
    pre = H_t @ W_enc.T + b_enc
    del W_enc, b_enc
    act_fn = tc.activation_function
    if isinstance(act_fn, JumpReLU):
        thr = act_fn.threshold.detach().to(H_t.device, H_t.dtype)
        acts = pre * (pre > thr)
    else:
        acts = torch.relu(pre)
    del pre
    return acts.float().cpu().numpy()