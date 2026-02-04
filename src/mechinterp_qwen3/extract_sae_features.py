"""Extract SAE features from captured MLP activations using transcoders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .io import write_json
from .load_transcoder import (
    DEFAULT_TRANSCODER_REPO,
    extract_sae_features,
    get_transcoder_info,
    load_transcoders_for_layers,
)


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser(description="Extract SAE features from captured activations")
    ap.add_argument(
        "--run_path", type=str, required=True, help="Path to run directory with activations"
    )
    ap.add_argument(
        "--transcoder",
        type=str,
        default=DEFAULT_TRANSCODER_REPO,
        help="HuggingFace transcoder repo",
    )
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of prompts to process")
    args = ap.parse_args()

    run_path = Path(args.run_path)
    acts_dir = run_path / "activations"

    if not acts_dir.exists():
        raise FileNotFoundError(f"Activations directory not found: {acts_dir}")

    # Find all activation files
    act_files = sorted(acts_dir.glob("*.pt"))
    if args.limit:
        act_files = act_files[: args.limit]

    if not act_files:
        raise ValueError(f"No activation files found in {acts_dir}")

    print(f"Found {len(act_files)} activation files")

    # Load first file to get layer information
    first_data = torch.load(act_files[0])
    layer_ids = list(first_data["per_layer"].keys())
    print(f"Layers in activations: {layer_ids}")

    # Load transcoders for all layers
    print(f"\nLoading transcoders from {args.transcoder}...")
    transcoders = load_transcoders_for_layers(
        layer_ids=layer_ids,
        transcoder_repo=args.transcoder,
        device=args.device,
    )

    # Print transcoder info
    for lid, transcoder in transcoders.items():
        info = get_transcoder_info(transcoder)
        print(f"Layer {lid}: {info['n_features']} features, d_model={info['d_model']}")

    # Create output directory
    sae_dir = run_path / "sae_features"
    sae_dir.mkdir(exist_ok=True)

    # Process each activation file
    processed = 0
    for act_file in act_files:
        # Load activations
        data = torch.load(act_file)
        prompt_id = act_file.stem  # e.g., "gt_0000"

        # Extract SAE features for each layer
        sae_features_per_layer = {}

        for lid in layer_ids:
            layer_acts = data["per_layer"][lid]
            mlp_out = layer_acts["mlp_out"]  # [seq_len, d_model]

            # Extract SAE features
            transcoder = transcoders[lid]
            sae_feats = extract_sae_features(mlp_out, transcoder)

            sae_features_per_layer[lid] = {
                "features": sae_feats,  # [seq_len, n_features]
                "shape": list(sae_feats.shape),
            }

        # Save SAE features
        sae_output = {
            "prompt_id": prompt_id,
            "input_ids": data["input_ids"],
            "seq_len": data["input_ids"].shape[0],
            "layers": layer_ids,
            "per_layer": sae_features_per_layer,
        }

        sae_file = sae_dir / f"{prompt_id}_sae.pt"
        torch.save(sae_output, sae_file)

        # Save metadata
        meta = {
            "prompt_id": prompt_id,
            "seq_len": sae_output["seq_len"],
            "layers": layer_ids,
            "feature_shapes": {lid: sae_features_per_layer[lid]["shape"] for lid in layer_ids},
            "sae_file": str(sae_file),
        }
        write_json(sae_dir / f"{prompt_id}_sae.meta.json", meta)

        processed += 1
        if processed % 10 == 0:
            print(f"Processed {processed}/{len(act_files)} files...")

    # Create summary
    summary = {
        "processed": processed,
        "layers": layer_ids,
        "transcoder_repo": args.transcoder,
        "device": args.device,
        "run_path": str(run_path),
        "transcoder_info": {lid: get_transcoder_info(transcoders[lid]) for lid in layer_ids},
    }

    write_json(run_path / "sae_extraction_summary.json", summary)
    print(f"\n✓ Extracted SAE features for {processed} prompts")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
