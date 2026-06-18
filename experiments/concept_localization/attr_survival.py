"""Shared helper: load attribution-graph survival set for pre-filtering features.

Usage::

    from experiments.concept_localization.attr_survival import load_survival_set

    survival = load_survival_set("carry")           # returns set or None
    if survival is not None:
        candidates = [(l, f) for l, f in candidates if (l, f) in survival]
"""

from __future__ import annotations

import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_SURVIVAL_BASE = _REPO_ROOT / "runs" / "concept_localization"


def load_survival_set(
    concept: str,
    min_survival: float = 0.05,
    survival_file: Path | None = None,
    required: bool = True,
) -> set[tuple[int, int]] | None:
    """Return set of (layer, feat_idx) that pass attribution-graph survival threshold.

    By default (required=True) raises FileNotFoundError if no survival file
    exists — the caller must explicitly pass required=False or --no_attr_filter
    to skip the filter.

    Args:
        concept:       concept name, used to find default file path.
        min_survival:  minimum fraction of graphs a feature must appear in.
        survival_file: explicit path override; if None, uses the default location
                       runs/concept_localization/{concept}/feature_survival/survival_stats.json
        required:      if True (default), raise if the file is not found instead
                       of returning None.
    """
    if survival_file is None:
        survival_file = (
            _DEFAULT_SURVIVAL_BASE / concept / "feature_survival" / "survival_stats.json"
        )

    if not survival_file.exists():
        if required:
            raise FileNotFoundError(
                f"Attribution-graph survival file not found for concept '{concept}':\n"
                f"  {survival_file}\n"
                f"Either run attribution_feature_survival.py first, or pass "
                f"--no_attr_filter to disable this check."
            )
        return None

    data = json.loads(survival_file.read_text())
    n_total = data["config"]["n_total_graphs"]
    min_n = max(1, int(min_survival * n_total))
    surviving = {
        (int(f["layer"]), int(f["feat_idx"]))
        for f in data["features"]
        if f["n_graphs"] >= min_n
    }
    print(
        f"  [attr-survival] loaded {len(surviving)} features (≥{min_survival:.0%} of "
        f"{n_total} graphs) from {survival_file.name}"
    )
    return surviving