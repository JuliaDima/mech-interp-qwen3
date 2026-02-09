"""Analyze actual feature activation distribution to choose optimal threshold."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mechinterp_qwen3.forward_with_sae import forward_linearized_with_sae_features
from mechinterp_qwen3.load_transcoder import load_transcoders_for_layers


def analyze_sparsity():
    """Analyze feature activation distribution."""

    # Simple test prompt
    prompt = """You are solving a simple comparison task.
Two numbers are given: A and B.
Answer with a single character: 'A' if A is larger, otherwise 'B'.

A = 864
B = 394
Answer: """

    print("Loading model...")
    model_name = "Qwen/Qwen3-4B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )
    model.eval()

    print("Loading transcoders...")
    layers = [4, 12, 20]
    transcoders = load_transcoders_for_layers(
        layer_ids=layers,
        transcoder_repo="mwhanna/qwen3-4b-transcoders",
        device=str(model.device),
    )

    print("Running forward pass...")
    result = forward_linearized_with_sae_features(
        model=model,
        tokenizer=tokenizer,
        transcoders=transcoders,
        prompt=prompt,
        layers_to_analyze=layers,
    )

    sae_features = result["sae_features"]

    print("\n" + "=" * 70)
    print("Feature Activation Distribution Analysis")
    print("=" * 70)

    all_activations = []
    for layer_id in layers:
        features = sae_features[layer_id].detach().float()  # [seq_len, n_features]
        all_activations.append(features.flatten())

    all_activations = torch.cat(all_activations)

    # Analyze non-zero activations
    nonzero_mask = all_activations.abs() > 1e-6
    nonzero_acts = all_activations[nonzero_mask]

    print(f"\nTotal activations: {all_activations.numel():,}")
    print(
        f"Non-zero activations: {nonzero_mask.sum().item():,} ({100 * nonzero_mask.sum() / all_activations.numel():.2f}%)"
    )

    if nonzero_acts.numel() > 0:
        print("\nNon-zero activation statistics:")
        print(f"  Min:    {nonzero_acts.min().item():.6f}")
        print(f"  Mean:   {nonzero_acts.mean().item():.6f}")
        print(f"  Median: {nonzero_acts.median().item():.6f}")
        print(f"  Max:    {nonzero_acts.max().item():.6f}")

        # Percentiles
        percentiles = [50, 75, 90, 95, 99]
        pcts = torch.quantile(
            nonzero_acts.abs(),
            torch.tensor([p / 100 for p in percentiles], device=nonzero_acts.device),
        )
        print("\nAbsolute value percentiles (non-zero features):")
        for p, val in zip(percentiles, pcts, strict=False):
            print(f"  p{p:2d}: {val.item():.6f}")

        # Test different thresholds
        print("\n" + "=" * 70)
        print("Sparsity at different thresholds:")
        print("=" * 70)
        thresholds = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
        print(f"{'Threshold':<12} {'Active Features':<20} {'Sparsity %':<15} {'Speedup'}")
        print("-" * 70)

        for thresh in thresholds:
            active = (all_activations.abs() > thresh).sum().item()
            sparsity_pct = 100 * active / all_activations.numel()
            speedup = 100 / sparsity_pct if sparsity_pct > 0 else float("inf")
            print(f"{thresh:<12.3f} {active:<20,} {sparsity_pct:<15.2f} {speedup:.1f}x")

        # Recommend threshold
        print("\n" + "=" * 70)
        print("Recommendations:")
        print("=" * 70)

        # Common SAE practice: keep features that are "meaningfully active"
        # Typically want 1-10% sparsity for interpretability
        median_nonzero = nonzero_acts.abs().median().item()
        mean_nonzero = nonzero_acts.abs().mean().item()

        print("\n1. For high interpretability (1-5% active):")
        print(f"   → Try threshold = {mean_nonzero:.3f} (mean of non-zero)")

        print("\n2. For balanced (5-10% active):")
        print(f"   → Try threshold = {median_nonzero:.3f} (median of non-zero)")

        print("\n3. For comprehensive coverage (10-20% active):")
        print("   → Try threshold = 0.01-0.05")

        print("\n4. Current default (0.01):")
        current_active = (all_activations.abs() > 0.01).sum().item()
        current_pct = 100 * current_active / all_activations.numel()
        print(f"   → {current_pct:.1f}% active features")

        print("\n" + "=" * 70)


if __name__ == "__main__":
    analyze_sparsity()
