"""Quick verification script to test transcoder loading."""

import torch

from mechinterp_qwen3.load_transcoder import get_transcoder_info, load_transcoder

print("Testing transcoder loading...")
print("Loading transcoder for layer 0...")

try:
    transcoder = load_transcoder(layer_id=0, device="cpu")
    info = get_transcoder_info(transcoder)

    print("✓ Transcoder loaded successfully!")
    print(f"  - Features: {info['n_features']}")
    print(f"  - d_model: {info['d_model']}")
    print(f"  - Device: {info['device']}")

    # Test feature extraction with dummy data
    print("\nTesting feature extraction...")
    dummy_acts = torch.randn(10, info["d_model"])  # [seq_len=10, d_model]

    from mechinterp_qwen3.load_transcoder import extract_sae_features

    sae_feats = extract_sae_features(dummy_acts, transcoder)

    print("✓ Feature extraction successful!")
    print(f"  - Input shape: {list(dummy_acts.shape)}")
    print(f"  - Output shape: {list(sae_feats.shape)}")
    print(f"  - Sparsity: {(sae_feats == 0).float().mean().item():.2%}")

    print("\n✅ All tests passed!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
