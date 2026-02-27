import json
import os

import torch


def add_graph_metadata(graph_metadata, path):
    assert os.path.exists(os.path.dirname(path)), f"Could not find {os.path.dirname(path)}"
    if os.path.isdir(path):
        path = os.path.join(path, "graph-metadata.json")

    if os.path.exists(path):
        with open(path) as f:
            metadata = json.load(f)
    else:
        metadata = {"graphs": []}

    metadata["graphs"] = [g for g in metadata["graphs"] if g["slug"] != graph_metadata["slug"]]
    metadata["graphs"].append(graph_metadata)

    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)


def process_token(token: str) -> str:
    return token.replace("\n", "⏎").replace("\t", "→").replace("\r", "↵")


def get_default_device() -> "torch.device":
    """Get the default device, preferring CUDA if available."""
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
