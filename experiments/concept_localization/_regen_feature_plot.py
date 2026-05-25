"""Regenerate feature_projections_scatter.png from existing results.json + deltas.pt."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from experiments.concept_localization.analyze import FeatureMatch
from experiments.concept_localization.visualize import plot_feature_projections

RUN_DIR = Path("runs/concept_localization/carry")

deltas = torch.load(RUN_DIR / "deltas.pt", weights_only=True)
delta_norms = {l: v.float().norm().item() for l, v in deltas["all"].items()}

results = json.loads((RUN_DIR / "results.json").read_text())
top_by_layer = results.get("top_features_by_layer", {})

projections: dict[int, list[FeatureMatch]] = {}
for layer_str, feats in top_by_layer.items():
    layer = int(layer_str)
    dn = max(delta_norms.get(layer, 1.0), 1e-8)
    projections[layer] = [
        FeatureMatch(
            feature_id=f["feature_id"],
            projection=f["projection"],
            cos_sim=f["projection"] / dn,
            layer=layer,
        )
        for f in feats
    ]

out = RUN_DIR / "feature_projections_scatter.png"
plot_feature_projections(projections, out, top_k=15, concept="carry")
print(f"Saved {out}")
