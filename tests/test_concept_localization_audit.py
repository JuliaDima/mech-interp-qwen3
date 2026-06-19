from __future__ import annotations

import importlib
import json
import pickle
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Keep these tests lightweight: the extraction/scoring modules only need this
# tokenization helper for functions that are not exercised here. Importing the
# full package pulls optional model dependencies such as transformers.
miq = ModuleType("mechinterp_qwen3")
miq_utils = ModuleType("mechinterp_qwen3.utils")
miq_token_utils = ModuleType("mechinterp_qwen3.utils.token_utils")
miq_token_utils.tokenize_qwen_input = lambda ids, tokenizer, device: torch.tensor(ids, device=device)
sys.modules.setdefault("mechinterp_qwen3", miq)
sys.modules.setdefault("mechinterp_qwen3.utils", miq_utils)
sys.modules.setdefault("mechinterp_qwen3.utils.token_utils", miq_token_utils)

miq_attr_model = ModuleType("mechinterp_qwen3.attribution_model")
miq_attr_model.AttributionModel = SimpleNamespace
miq_hf_utils = ModuleType("mechinterp_qwen3.utils.hf_utils")
miq_hf_utils.load_transcoder_from_hub = lambda *args, **kwargs: (None, None)
miq_model_utils = ModuleType("mechinterp_qwen3.utils.model_utils")
miq_model_utils.get_default_device = lambda: torch.device("cpu")
miq_model_utils.parse_dtype = lambda _name: torch.float32
sys.modules.setdefault("mechinterp_qwen3.attribution_model", miq_attr_model)
sys.modules.setdefault("mechinterp_qwen3.utils.hf_utils", miq_hf_utils)
sys.modules.setdefault("mechinterp_qwen3.utils.model_utils", miq_model_utils)

stub_sweep_utils = ModuleType("experiments.concept_localization.sweep_utils")
stub_sweep_utils.apply_transcoder_all = lambda *args, **kwargs: None
sys.modules.setdefault("experiments.concept_localization.sweep_utils", stub_sweep_utils)
stub_run_concept_sweep = ModuleType("experiments.concept_localization.pipeline.run_concept_sweep")
stub_run_concept_sweep._load_concept = lambda *args, **kwargs: []
sys.modules.setdefault(
    "experiments.concept_localization.pipeline.run_concept_sweep", stub_run_concept_sweep
)
stub_attr_survival = ModuleType("experiments.concept_localization.attr_survival")
stub_attr_survival.load_survival_set = lambda *args, **kwargs: set()
sys.modules.setdefault("experiments.concept_localization.attr_survival", stub_attr_survival)
stub_visualize = ModuleType("experiments.concept_localization.plots.visualize")
stub_visualize.plot_feature_heatmap_grid = lambda *args, **kwargs: None
sys.modules.setdefault("experiments.concept_localization.plots.visualize", stub_visualize)

sys.modules.setdefault("tqdm", SimpleNamespace(tqdm=lambda iterable, *args, **kwargs: iterable))

from experiments.concept_localization.analyze import (
    compute_sharpness,
    project_onto_E_dec_model,
)
from experiments.concept_localization.concept_pair import ConceptPair
from experiments.concept_localization.extract_deltas_generic import (
    LayerDeltas,
    _find_delimiter_anchor,
    _resolve_anchor,
)
from experiments.concept_localization.pipeline.delta_feature_projections import (
    _bin_to_heatmap,
    _load_anchor_inputs_and_examples,
    _resolve_survival_set,
    _stable_hash,
    _validate_sweep_cache_metadata,
)


DATASET_DIR = REPO_ROOT / "experiments" / "concept_localization" / "concept_datasets"
CONCEPT_GENERATORS = {
    "carry": ("carry_dataset", "generate_carry_pairs"),
    "gcd": ("gcd_dataset", "generate_gcd_pairs"),
    "residue_class": ("residue_class_dataset", "generate_residue_pairs"),
    "transitive_ordering": ("transitive_ordering_dataset", "generate_ordering_pairs"),
    "conservation": ("conservation_dataset", "generate_conservation_pairs"),
    "causal_direction": ("causal_direction_dataset", "generate_causal_pairs"),
    "negation_scope": ("negation_scope_dataset", "generate_negation_pairs"),
    "balanced_parentheses": ("balanced_parentheses_dataset", "generate_parentheses_pairs"),
    "decimal_termination": ("decimal_termination_dataset", "generate_decimal_pairs"),
    "doppler_shift": ("doppler_shift_dataset", "generate_doppler_pairs"),
    "dot_product_sign": ("dot_product_sign_dataset", "generate_dot_pairs"),
    "geometric_series": ("geometric_series_dataset", "generate_geometric_pairs"),
    "momentum_conservation": ("momentum_conservation_dataset", "generate_momentum_pairs"),
    "perfect_square": ("perfect_square_dataset", "generate_perfect_square_pairs"),
    "syllogism": ("syllogism_dataset", "generate_syllogism_pairs"),
    "triangle_inequality": ("triangle_inequality_dataset", "generate_triangle_pairs"),
    "wave_interference": ("wave_interference_dataset", "generate_wave_pairs"),
}


def _load_concept(concept: str, n_per_template: int, seed: int) -> list[ConceptPair]:
    module_name, function_name = CONCEPT_GENERATORS[concept]
    module = importlib.import_module(f"experiments.concept_localization.concept_datasets.{module_name}")
    return getattr(module, function_name)(n_per_template=n_per_template, seed=seed)


@pytest.mark.parametrize("concept", sorted(CONCEPT_GENERATORS))
def test_registered_concept_generators_return_valid_pairs(concept: str) -> None:
    pairs = _load_concept(concept, n_per_template=8, seed=123)

    assert pairs, f"{concept} generated no examples"
    assert all(isinstance(pair, ConceptPair) for pair in pairs)
    assert {pair.template for pair in pairs}

    seen_prompts = set()
    for pair in pairs:
        assert pair.prompt_pos.strip()
        assert pair.prompt_neg.strip()
        assert pair.prompt_pos != pair.prompt_neg
        assert pair.label_pos != ""
        assert pair.label_neg != ""
        assert pair.predict_pos != ""
        assert pair.predict_neg != ""
        assert pair.template.startswith("T")
        assert isinstance(pair.meta, dict)

        key = (pair.template, pair.prompt_pos, pair.prompt_neg)
        assert key not in seen_prompts
        seen_prompts.add(key)


def test_registered_concepts_have_dataset_files_and_templates() -> None:
    for concept, (module_name, _) in CONCEPT_GENERATORS.items():
        module_path = DATASET_DIR / f"{module_name}.py"
        assert module_path.exists(), f"{concept} is registered without {module_path.name}"

        module = importlib.import_module(f"experiments.concept_localization.concept_datasets.{module_name}")
        assert getattr(module, "TEMPLATES"), f"{concept} has no prompt templates"


def test_reported_carry_dataset_preserves_units_carry_invariant() -> None:
    pairs = _load_concept("carry", n_per_template=120, seed=7)

    assert len(pairs) == 3 * 99
    assert {p.template for p in pairs} == {"T0", "T1", "T2"}

    for pair in pairs:
        meta = pair.meta
        a_pos, b_pos = meta["a_pos"], meta["b_pos"]
        a_neg, b_neg = meta["a_neg"], meta["b_neg"]

        assert (a_pos % 10) + (b_pos % 10) >= 10
        assert (a_neg % 10) + (b_neg % 10) < 10
        assert ((a_pos // 10) % 10) + ((b_pos // 10) % 10) <= 8
        assert ((a_neg // 10) % 10) + ((b_neg // 10) % 10) <= 8
        assert pair.label_pos == "carry"
        assert pair.label_neg == "no_carry"


def test_reported_gcd_dataset_preserves_divisibility_invariant() -> None:
    pairs = _load_concept("gcd", n_per_template=25, seed=7)

    assert {p.template for p in pairs} == {"T0", "T1", "T2"}
    for pair in pairs:
        a_pos = pair.meta["a_pos"]
        a_neg = pair.meta["a_neg"]
        assert a_pos % 7 == 0
        assert a_neg % 7 != 0
        assert pair.label_pos == "7"
        assert pair.label_neg == "1"


def test_reported_residue_dataset_preserves_modular_invariant() -> None:
    pairs = _load_concept("residue_class", n_per_template=25, seed=7)

    assert {p.template for p in pairs} == {"T0", "T1", "T2"}
    for pair in pairs:
        a_pos = pair.meta["a_pos"]
        a_neg = pair.meta["a_neg"]
        r_neg = pair.meta["r_neg"]
        assert a_pos % 7 == 1
        assert a_neg % 7 == r_neg
        assert r_neg in {2, 3, 4, 5, 6}
        assert pair.label_pos == "1"
        assert pair.label_neg == str(r_neg)


class _TinyTokenizer:
    def convert_ids_to_tokens(self, ids):
        return ids


def test_anchor_resolution_raises_instead_of_falling_back() -> None:
    tokenizer = _TinyTokenizer()

    assert _find_delimiter_anchor(["abc", "def:"], tokenizer) == 1
    with pytest.raises(ValueError, match="Could not resolve delimiter anchor"):
        _find_delimiter_anchor(["abc", "def"], tokenizer)
    with pytest.raises(ValueError, match="out of range"):
        _resolve_anchor(["a", "b", "c"], tokenizer, "99", None)
    with pytest.raises(ValueError, match="Unknown anchor_mode"):
        _resolve_anchor(["a", "b"], tokenizer, "not_a_mode", None)


def test_concept_specific_anchor_resolution_raises_instead_of_using_lower_rank() -> None:
    tokenizer = _TinyTokenizer()
    factory = lambda _pair, _tokenizer: {"rank1": 0}

    with pytest.raises(ValueError, match="Known concept-specific anchors"):
        _resolve_anchor(["a", "b"], tokenizer, "rank2", factory, pair=object())


def test_compute_sharpness_uses_peak_window_mass_not_peak_to_average() -> None:
    ld = LayerDeltas(
        delta={
            0: torch.tensor([1.0, 0.0]),
            1: torch.tensor([2.0, 0.0]),
            2: torch.tensor([3.0, 0.0]),
            3: torch.tensor([2.0, 0.0]),
            4: torch.tensor([1.0, 0.0]),
        }
    )

    result = compute_sharpness(ld)

    assert result.peak_layer == 2
    assert result.sharpness_index == pytest.approx((2.0 + 3.0 + 2.0) / 9.0)


def test_project_onto_E_dec_model_ranks_signed_decoder_and_encoder_alignment() -> None:
    tc = SimpleNamespace(
        W_dec=torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
            ]
        ),
        W_enc=torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ),
    )
    model = SimpleNamespace(transcoders=[tc])
    deltas = {0: torch.tensor([1.0, 0.0])}

    matches = project_onto_E_dec_model(model, deltas, top_k=2, score_mode="enc+dec")[0]
    assert [m.feature_id for m in matches] == [0, 1]
    assert matches[0].cos_sim == pytest.approx(1.0)
    assert matches[0].enc_cos_sim == pytest.approx(1.0)

    neg_matches = project_onto_E_dec_model(
        model, deltas, top_k=1, score_mode="dec", direction="neg"
    )[0]
    assert neg_matches[0].feature_id == 2
    assert neg_matches[0].cos_sim == pytest.approx(-1.0)


class _SimpleTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=text.split())


def test_feature_projection_requires_saved_sweep_examples(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Required sweep examples file is missing"):
        _load_anchor_inputs_and_examples(
            tmp_path, SimpleNamespace(tokenizer=_SimpleTokenizer()), "carry", "5", 42, None
        )


def test_survival_set_must_preexist_unless_filter_disabled(tmp_path: Path) -> None:
    args = SimpleNamespace(
        no_attr_filter=False,
        attr_survival_file=tmp_path / "missing.json",
        concept="carry",
        template="T0",
        attr_min_survival=0.05,
    )

    with pytest.raises(FileNotFoundError, match="Generate it explicitly"):
        _resolve_survival_set(args)


def test_bin_to_heatmap_preserves_nan_for_unsampled_cells() -> None:
    examples = [
        {"meta": {"a_pos": 1, "b_pos": 2, "a_neg": 3, "b_neg": 4}},
    ]
    grid = _bin_to_heatmap(np.array([0.5, 0.25], dtype=np.float32), examples)

    assert grid[1, 2] == pytest.approx(0.5)
    assert grid[3, 4] == pytest.approx(0.25)
    assert np.isnan(grid[0, 0])


def test_sweep_cache_metadata_validation_rejects_tampering(tmp_path: Path) -> None:
    anchor_dir = tmp_path / "anchor"
    sweep_dir = anchor_dir / "sweep"
    sweep_dir.mkdir(parents=True)
    payload = {
        "prompts": ["p pos", "p neg"],
        "examples": [{"pair_idx": 0}],
    }
    metadata = {"version": 1, "hash": _stable_hash(payload), "payload": payload}
    (sweep_dir / "sweep_residuals.meta.json").write_text(json.dumps(metadata))
    npz_path = sweep_dir / "sweep_residuals.npz"
    np.savez(npz_path, prompts=np.array(["p pos", "p neg"], dtype=object), layers=np.array([0]))

    npz = np.load(npz_path, allow_pickle=True)
    _validate_sweep_cache_metadata(anchor_dir, [([1], 0), ([2], 0)], [{"pair_idx": 0}], npz)
    with pytest.raises(ValueError, match="current run context"):
        _validate_sweep_cache_metadata(
            anchor_dir,
            [([1], 0), ([2], 0)],
            [{"pair_idx": 0}],
            npz,
            expected={"concept": "carry"},
        )

    metadata["payload"]["prompts"] = ["changed", "p neg"]
    (sweep_dir / "sweep_residuals.meta.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="hash mismatch"):
        _validate_sweep_cache_metadata(anchor_dir, [([1], 0), ([2], 0)], [{"pair_idx": 0}], npz)
