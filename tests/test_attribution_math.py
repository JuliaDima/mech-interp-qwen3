from unittest.mock import MagicMock

import torch
import torch.nn as nn


class MockTranscoder(nn.Module):
    def __init__(self, d_model, n_features):
        super().__init__()
        self.W_enc = nn.Parameter(torch.randn(n_features, d_model))
        self.W_dec = nn.Parameter(torch.randn(n_features, d_model))
        self.b_enc = nn.Parameter(torch.zeros(n_features))
        self.b_dec = nn.Parameter(torch.randn(d_model))  # Non-zero bias
        self.d_model = d_model

    def forward(self, x):
        return x  # dummy

    def decode(self, features, mlp_act=None):
        return features @ self.W_dec + self.b_dec


def test_attribution_math():
    d_model = 4
    n_features = 8
    seq_len = 5  # Increased length to test start_pos
    start_pos = 1

    # Mock model and tokenizer (Minimal)
    model = MagicMock()
    model.device = "cpu"

    # Inputs (Random data)
    mlp_act = torch.randn(seq_len, d_model)
    dlogit_dmlp = torch.randn(seq_len, d_model)

    transcoder = MockTranscoder(d_model, n_features)

    # Simulate Sparse Features
    # We use dense for simplicity of test, but logic holds
    features = torch.randn(seq_len, n_features)

    # 1. Reconstruction
    reconstruction = transcoder.decode(features)
    error = mlp_act - reconstruction

    # 2. Attributions Calculation (Manual reproduction of compute_attribution LOGIC)

    # Direct Attribution (Sliced!)
    # logic: if mlp_act.shape[0] > start_pos: ... [start_pos:]
    direct_attr = (mlp_act[start_pos:] * dlogit_dmlp[start_pos:]).sum()

    # Feature Attribution
    # logic: mask positions < start_pos
    # Here we just slice the features manually to simulate the filtering
    # feat_attr = sum(feat[pos] * (grad[pos] @ W_dec.T)) for pos >= start_pos

    # Gradients w.r.t features: grad @ W_dec.T
    grad_wrt_feat = dlogit_dmlp @ transcoder.W_dec.T  # [seq, n_features]

    # Total feature attribution per element
    feat_attr_elementwise = features * grad_wrt_feat  # [seq, n_features]

    # Sum only for pos >= start_pos
    feat_attr_sum = feat_attr_elementwise[start_pos:].sum()

    # Error Attribution
    # error * grad
    error_attr = (error * dlogit_dmlp).sum(dim=-1)  # [seq]
    err_attr_sum = error_attr[start_pos:].sum()

    # Bias Attribution
    # b_dec * grad
    # bias is [d_model]. grad is [seq, d_model].
    # bias contribution at pos P is (grad[P] @ b_dec)
    bias_attr_per_pos = dlogit_dmlp @ transcoder.b_dec  # [seq]
    bias_attr_sum = bias_attr_per_pos[start_pos:].sum()

    # 3. Verification
    total_component_attr = feat_attr_sum + err_attr_sum + bias_attr_sum

    print(f"Direct (pos>={start_pos}): {direct_attr.item():.6f}")
    print(f"Sum    (pos>={start_pos}): {total_component_attr.item():.6f}")

    diff = abs(direct_attr - total_component_attr).item()
    print(f"Diff: {diff:.6e}")

    # Verify strict equality (float precision)
    # This proves that IF we exclude data from < start_pos in all terms, math holds.
    assert diff < 1e-4, f"Mismatch in start_pos logic! Diff: {diff}"
    print("Test Passed: Start position exclusion logic is mathematically sound.")


if __name__ == "__main__":
    test_attribution_math()
