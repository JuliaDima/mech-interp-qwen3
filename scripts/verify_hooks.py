#!/usr/bin/env python3
"""Verify that hooks correctly capture MLP activations for a real model and transcoder.

This script loads your actual Qwen model and transcoder, installs hooks,
and verifies that mlp_in and mlp_out are captured correctly.
"""

from __future__ import annotations

import argparse
import json

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from mechinterp_qwen3.hooks import MLPHookManager


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify MLP hooks capture activations correctly")
    ap.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Model name or path",
    )
    ap.add_argument(
        "--transcoder_repo",
        type=str,
        default="mwhanna/qwen3-4b-transcoders",
        help="HuggingFace repo containing transcoders",
    )
    ap.add_argument(
        "--layer",
        type=int,
        default=12,
        help="Layer to test (default: 12)",
    )
    ap.add_argument(
        "--prompt",
        type=str,
        default="The capital of France is",
        help="Test prompt",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )
    args = ap.parse_args()

    print("=" * 80)
    print("Hook Verification Script")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Transcoder repo: {args.transcoder_repo}")
    print(f"Layer: {args.layer}")
    print(f"Prompt: {args.prompt}")
    print(f"Device: {args.device}")
    print("=" * 80)

    # Load model and tokenizer
    print("\n[1/5] Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()
    print(f"✓ Model loaded on {args.device}")
    print(f"  Model has {len(model.model.layers)} layers")
    print(f"  Hidden size (d_model): {model.config.hidden_size}")

    # Load transcoder config to check hook points
    print("\n[2/5] Checking transcoder configuration...")
    try:
        config_path = hf_hub_download(
            repo_id=args.transcoder_repo,
            filename=f"layer_{args.layer}/config.json",
        )

        with open(config_path) as f:
            transcoder_config = json.load(f)

        print("✓ Transcoder config loaded")
        print(f"  Expected input hook: {transcoder_config.get('feature_input_hook', 'N/A')}")
        print(f"  Expected output hook: {transcoder_config.get('feature_output_hook', 'N/A')}")

        expected_input_hook = transcoder_config.get("feature_input_hook", "")
        expected_output_hook = transcoder_config.get("feature_output_hook", "")

    except Exception as e:
        print(f"⚠ Could not load transcoder config: {e}")
        print("  Continuing with hook verification only...")
        expected_input_hook = None
        expected_output_hook = None

    # Install our hooks
    print(f"\n[3/5] Installing MLPHookManager on layer {args.layer}...")
    hooker = MLPHookManager(model, layer_ids=[args.layer], detach=True)
    hooker.install()
    print(f"✓ MLPHookManager hooks installed ({len(hooker.handles)} handles)")

    # Also install direct hooks for comparison
    print("\n[4/5] Running forward pass with comparison hooks...")

    # Create dictionaries to store direct hook captures
    mlp_hook_in_capture = {}
    mlp_hook_out_capture = {}

    # Install direct hooks on the MLP to capture what transcoder expects
    layer_module = model.model.layers[args.layer]
    mlp_module = layer_module.mlp

    def capture_hook_in(module, inputs):
        """Capture what mlp.hook_in would see."""
        # inputs is a tuple, inputs[0] is [batch, seq, d_model]
        mlp_hook_in_capture["value"] = inputs[0][0].detach().cpu()  # [seq, d_model]

    def capture_hook_out(module, inputs, output):
        """Capture what mlp.hook_out would see."""
        # Handle both tuple and tensor outputs
        out_tensor = output[0] if isinstance(output, tuple) else output
        mlp_hook_out_capture["value"] = out_tensor[0].detach().cpu()  # [seq, d_model]

    hook_in_handle = mlp_module.register_forward_pre_hook(capture_hook_in)
    hook_out_handle = mlp_module.register_forward_hook(capture_hook_out)

    inputs = tokenizer(args.prompt, return_tensors="pt").to(args.device)
    tokens = [tokenizer.decode([tok_id]) for tok_id in inputs["input_ids"][0]]

    print(f"  Tokenized prompt: {tokens}")
    print(f"  Num tokens: {len(tokens)}")

    with torch.no_grad():
        outputs = model(**inputs)

    # Remove direct hooks
    hook_in_handle.remove()
    hook_out_handle.remove()

    print("✓ Forward pass complete")
    print(f"  Output shape: {outputs.logits.shape}")

    # Verify captured activations
    print("\n[5/5] Verifying captured activations...")
    layer_acts = hooker.cache[args.layer]

    print(f"\n{'=' * 80}")
    print(f"HOOK VERIFICATION RESULTS for Layer {args.layer}")
    print(f"{'=' * 80}")

    # Check mlp_in
    if layer_acts.mlp_in is None:
        print("❌ mlp_in: NOT CAPTURED")
    else:
        print("✓ mlp_in: CAPTURED (our MLPHookManager)")
        print(f"  Shape: {layer_acts.mlp_in.shape}")
        print(f"  Expected: ({len(tokens)}, {model.config.hidden_size})")
        print(f"  Device: {layer_acts.mlp_in.device}")
        print(f"  Dtype: {layer_acts.mlp_in.dtype}")
        print(f"  Mean: {layer_acts.mlp_in.mean():.4f}")
        print(f"  Std: {layer_acts.mlp_in.std():.4f}")

        # Check shape matches expected
        if layer_acts.mlp_in.shape == (len(tokens), model.config.hidden_size):
            print("  ✓ Shape matches expected [seq_len, d_model]")
        else:
            print("  ⚠ Shape mismatch!")

    print()

    # Check mlp_out
    if layer_acts.mlp_out is None:
        print("❌ mlp_out: NOT CAPTURED")
    else:
        print("✓ mlp_out: CAPTURED (our MLPHookManager)")
        print(f"  Shape: {layer_acts.mlp_out.shape}")
        print(f"  Expected: ({len(tokens)}, {model.config.hidden_size})")
        print(f"  Device: {layer_acts.mlp_out.device}")
        print(f"  Dtype: {layer_acts.mlp_out.dtype}")
        print(f"  Mean: {layer_acts.mlp_out.mean():.4f}")
        print(f"  Std: {layer_acts.mlp_out.std():.4f}")

        # Check shape matches expected
        if layer_acts.mlp_out.shape == (len(tokens), model.config.hidden_size):
            print("  ✓ Shape matches expected [seq_len, d_model]")
        else:
            print("  ⚠ Shape mismatch!")

    # Compare our hooks with transcoder's expected hook points
    print(f"\n{'=' * 80}")
    print("HOOK POINT COMPARISON")
    print(f"{'=' * 80}")

    if expected_input_hook:
        print(f"\nTranscoder expects input from: '{expected_input_hook}'")
    if expected_output_hook:
        print(f"Transcoder expects output from: '{expected_output_hook}'")

    # Compare mlp_in with hook_in
    if "value" in mlp_hook_in_capture and layer_acts.mlp_in is not None:
        hook_in_val = mlp_hook_in_capture["value"]
        our_mlp_in = layer_acts.mlp_in

        print("\n✓ Comparing mlp_in (ours) vs mlp.hook_in (direct):")
        print(f"  Our shape: {our_mlp_in.shape}")
        print(f"  Direct hook_in shape: {hook_in_val.shape}")

        # Check if values match
        if torch.allclose(our_mlp_in.float(), hook_in_val.float(), rtol=1e-3, atol=1e-5):
            print("  ✅ VALUES MATCH! Our mlp_in == transcoder's hook_in")
        else:
            max_diff = (our_mlp_in.float() - hook_in_val.float()).abs().max()
            mean_diff = (our_mlp_in.float() - hook_in_val.float()).abs().mean()
            print("  ⚠ Values differ!")
            print(f"    Max difference: {max_diff:.6f}")
            print(f"    Mean difference: {mean_diff:.6f}")

    # Compare mlp_out with hook_out
    if "value" in mlp_hook_out_capture and layer_acts.mlp_out is not None:
        hook_out_val = mlp_hook_out_capture["value"]
        our_mlp_out = layer_acts.mlp_out

        print("\n✓ Comparing mlp_out (ours) vs mlp.hook_out (direct):")
        print(f"  Our shape: {our_mlp_out.shape}")
        print(f"  Direct hook_out shape: {hook_out_val.shape}")

        # Check if values match
        if torch.allclose(our_mlp_out.float(), hook_out_val.float(), rtol=1e-3, atol=1e-5):
            print("  ✅ VALUES MATCH! Our mlp_out == transcoder's hook_out")
        else:
            max_diff = (our_mlp_out.float() - hook_out_val.float()).abs().max()
            mean_diff = (our_mlp_out.float() - hook_out_val.float()).abs().mean()
            print("  ⚠ Values differ!")
            print(f"    Max difference: {max_diff:.6f}")
            print(f"    Mean difference: {mean_diff:.6f}")

    # Clean up
    hooker.remove()
    print(f"\n{'=' * 80}")
    print("✓ Hooks removed")
    print(f"{'=' * 80}")

    # Summary
    print("\nSUMMARY:")
    mlp_in_ok = layer_acts.mlp_in is not None and layer_acts.mlp_in.shape == (
        len(tokens),
        model.config.hidden_size,
    )
    mlp_out_ok = layer_acts.mlp_out is not None and layer_acts.mlp_out.shape == (
        len(tokens),
        model.config.hidden_size,
    )

    # Check if values match
    mlp_in_match = False
    mlp_out_match = False
    if "value" in mlp_hook_in_capture and layer_acts.mlp_in is not None:
        mlp_in_match = torch.allclose(our_mlp_in.float(), hook_in_val.float(), rtol=1e-3, atol=1e-5)
    if "value" in mlp_hook_out_capture and layer_acts.mlp_out is not None:
        mlp_out_match = torch.allclose(
            our_mlp_out.float(), hook_out_val.float(), rtol=1e-3, atol=1e-5
        )

    if mlp_in_ok and mlp_out_ok and mlp_in_match and mlp_out_match:
        print("✅ All hooks working correctly!")
        print("✅ Hook values match transcoder's expected hook points!")
        print("✅ Ready for attribution graph construction")
    elif mlp_in_ok and mlp_out_ok:
        print("✓ Hooks capturing activations correctly")
        if not mlp_in_match or not mlp_out_match:
            print("⚠ But values don't match expected hook points - check details above")
    else:
        print("⚠ Some issues detected - see details above")


if __name__ == "__main__":
    main()
