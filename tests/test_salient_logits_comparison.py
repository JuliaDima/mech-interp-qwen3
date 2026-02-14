#!/usr/bin/env python3
"""Test that our salient_logits implementation matches circuit_tracer's."""

import sys

sys.path.insert(0, "/home/eid23/mechinterp-qwen-3B-Instruct/circuit_tracer_github")

import torch
from circuit_tracer.utils.salient_logits import compute_salient_logits as ct_compute_salient_logits

from mechinterp_qwen3.salient_logits import compute_salient_logits as our_compute_salient_logits


def test_salient_logits_match():
    """Test that both implementations produce identical results for Qwen orientation."""

    print("=" * 80)
    print("Testing salient_logits implementation")
    print("=" * 80)

    # Create test data
    torch.manual_seed(42)
    vocab_size = 50000
    d_model = 2048

    # Random logits (simulating model output at one position)
    logits = torch.randn(vocab_size)

    # Test Qwen orientation: (d_vocab, d_model)
    print(f"\n{'=' * 80}")
    print(f"Testing Qwen orientation: (d_vocab, d_model) = ({vocab_size}, {d_model})")
    print(f"{'=' * 80}")

    unembed = torch.randn(vocab_size, d_model)

    # Run both implementations
    ct_indices, ct_probs, ct_vecs = ct_compute_salient_logits(
        logits,
        unembed,
        max_n_logits=10,
        desired_logit_prob=0.95,
    )

    our_indices, our_probs, our_vecs = our_compute_salient_logits(
        logits,
        unembed,
        max_n_logits=10,
        desired_logit_prob=0.95,
    )

    # Compare results
    print("\nNumber of selected logits:")
    print(f"  circuit_tracer: {len(ct_indices)}")
    print(f"  ours:           {len(our_indices)}")

    all_pass = True

    # Check if indices match
    if torch.equal(ct_indices, our_indices):
        print("✅ Logit indices MATCH")
    else:
        print("❌ Logit indices DIFFER")
        print(f"  circuit_tracer: {ct_indices}")
        print(f"  ours:           {our_indices}")
        all_pass = False

    # Check if probabilities match
    if torch.allclose(ct_probs, our_probs, rtol=1e-5, atol=1e-7):
        print(f"✅ Probabilities MATCH (max diff: {(ct_probs - our_probs).abs().max():.2e})")
    else:
        max_diff = (ct_probs - our_probs).abs().max()
        print(f"❌ Probabilities DIFFER (max diff: {max_diff:.2e})")
        all_pass = False

    # Check if demeaned vectors match
    if torch.allclose(ct_vecs, our_vecs, rtol=1e-5, atol=1e-7):
        max_diff = (ct_vecs - our_vecs).abs().max()
        print(f"✅ Demeaned vectors MATCH (max diff: {max_diff:.2e})")
    else:
        max_diff = (ct_vecs - our_vecs).abs().max()
        print(f"❌ Demeaned vectors DIFFER (max diff: {max_diff:.2e})")
        all_pass = False

    # Detailed comparison
    print("\nDetailed comparison:")
    print(f"  Demeaned vec shapes: circuit_tracer={ct_vecs.shape}, ours={our_vecs.shape}")
    print(f"  Demeaned vec mean:   circuit_tracer={ct_vecs.mean():.6f}, ours={our_vecs.mean():.6f}")
    print(f"  Demeaned vec std:    circuit_tracer={ct_vecs.std():.6f}, ours={our_vecs.std():.6f}")

    print(f"\n{'=' * 80}")
    print("Summary:")
    print(f"{'=' * 80}")
    if all_pass:
        print("✅ All tests passed! Implementations match for Qwen orientation.")
    else:
        print("❌ Some tests failed!")
        return False

    return True


if __name__ == "__main__":
    success = test_salient_logits_match()
    sys.exit(0 if success else 1)
