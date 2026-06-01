"""Robustness tests for concept localization scripts.

Catches silent logic errors: wrong anchor positions, template grouping bugs,
incorrect tokenizer usage, mask errors in feature scoring, off-by-one offsets,
factory/dataset inconsistencies, and aggregation mode correctness.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from experiments.concept_localization.concept_pair import ConceptPair
from experiments.concept_localization.extract_deltas import LayerDeltas
from experiments.concept_localization.extract_deltas_generic import (
    _find_delimiter_anchor,
    _resolve_anchor,
    extract_layer_deltas_generic,
    resolve_anchor_token,
)
from mechinterp_qwen3.utils.token_utils import tokenize_qwen_input
from scripts.sweeps.sweep_utils import cluster_top_features, score_and_rank

# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────


class _CharTok:
    """One token per character; IDs = ord(char). Special tokens: pad=0, bos=1, eos=2."""

    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    all_special_ids = [0, 1, 2]

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        ids = [ord(c) for c in text]
        if return_tensors == "pt":
            return SimpleNamespace(input_ids=torch.tensor([ids], dtype=torch.long))
        return SimpleNamespace(input_ids=ids)

    def convert_ids_to_tokens(self, ids):
        # Mimic Qwen: \n (ord 10) → "Ċ", other non-printable → "<N>"
        # Accepts both a single int and a list of ints (like HF tokenizers do)
        def _id_to_tok(i: int) -> str:
            if i == 10:
                return "Ċ"
            return chr(i) if 32 <= i < 128 else f"<{i}>"

        if isinstance(ids, int):
            return _id_to_tok(ids)
        return [_id_to_tok(i) for i in ids]

    def convert_tokens_to_string(self, tokens):
        return "".join(t if not t.startswith("<") else "" for t in tokens)


_CHAR_TOK = _CharTok()


def _make_model(d_model: int = 8) -> MagicMock:
    """Minimal model mock: zero activations, char tokenizer."""

    def _run(input_ids, fwd_hooks):
        acts = torch.zeros(1, input_ids.shape[1], d_model)
        for _, hook_fn in fwd_hooks:
            hook_fn(acts, None)

    model = MagicMock()
    model.cfg.device = "cpu"
    model.cfg.d_model = d_model
    model.tokenizer = _CharTok()
    model.run_with_hooks.side_effect = _run
    model.feature_input_hook = "hook_resid_post"
    return model


def _pairs(n: int, template: str = "T0") -> list[ConceptPair]:
    return [ConceptPair(prompt_pos="ab= ", prompt_neg="ab= ", template=template) for _ in range(n)]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Delimiter anchor detection
# ─────────────────────────────────────────────────────────────────────────────


class TestFindDelimiterAnchor:
    def test_returns_last_delimiter_not_first(self):
        # "calc: 123+456= " has ":" at 4 and "=" at 13 — must return 13, not 4
        text = "calc: 123+456= "
        ids = [ord(c) for c in text]
        pos = _find_delimiter_anchor(ids, _CHAR_TOK)
        assert text[pos] == "=", f"Expected '=' (last delimiter), got {text[pos]!r} at {pos}"

    def test_falls_back_to_last_token_when_no_delimiter(self):
        text = "abcdef"
        ids = [ord(c) for c in text]
        pos = _find_delimiter_anchor(ids, _CHAR_TOK)
        assert pos == len(ids) - 1

    def test_question_mark_is_delimiter(self):
        text = "what is 5+3? "
        ids = [ord(c) for c in text]
        pos = _find_delimiter_anchor(ids, _CHAR_TOK)
        assert text[pos] == "?"

    def test_newline_is_delimiter(self):
        text = "x\ny"
        ids = [ord(c) for c in text]
        pos = _find_delimiter_anchor(ids, _CHAR_TOK)
        assert text[pos] == "\n"

    def test_multiple_colons_returns_last(self):
        # "Yes or No: foo: " — last ":" is at index 14
        text = "Yes or No: foo: "
        ids = [ord(c) for c in text]
        pos = _find_delimiter_anchor(ids, _CHAR_TOK)
        expected = text.rindex(":")
        assert pos == expected, f"Expected last ':' at {expected}, got {pos}"

    def test_position_always_in_range(self):
        for text in ["x= ", "abc", "a:b= ", "\n"]:
            ids = [ord(c) for c in text]
            pos = _find_delimiter_anchor(ids, _CHAR_TOK)
            assert 0 <= pos < len(ids)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Anchor mode resolution — no silent fallbacks
# ─────────────────────────────────────────────────────────────────────────────


class TestResolveAnchor:
    def test_delimiter_mode_finds_last_delimiter(self):
        ids = [ord(c) for c in "calc: 123+456= "]
        pos = _resolve_anchor(ids, _CHAR_TOK, "delimiter", None, None)
        assert "calc: 123+456= "[pos] == "="

    def test_last_mode_returns_final_token(self):
        ids = [ord(c) for c in "abc"]
        assert _resolve_anchor(ids, _CHAR_TOK, "last", None, None) == 2

    def test_integer_mode_in_bounds(self):
        ids = [ord(c) for c in "abcde"]
        assert _resolve_anchor(ids, _CHAR_TOK, "2", None, None) == 2

    def test_integer_mode_clamps_to_last_when_out_of_bounds(self):
        ids = [ord(c) for c in "abc"]
        assert _resolve_anchor(ids, _CHAR_TOK, "999", None, None) == 2

    def test_unknown_mode_raises_value_error_not_silent_fallback(self):
        # Must raise, never silently fall through to delimiter or last
        ids = [ord(c) for c in "abc"]
        with pytest.raises(ValueError, match="Unknown anchor_mode"):
            _resolve_anchor(ids, _CHAR_TOK, "ones_b", None, None)

    def test_factory_missing_mode_raises_value_error(self):
        # Factory exists but doesn't contain the requested key → still raises
        def factory(pair, tokenizer):
            return {"other_mode": 3}

        ids = [ord(c) for c in "abcde"]
        pair = ConceptPair(prompt_pos="x", prompt_neg="y")
        with pytest.raises(ValueError, match="Unknown anchor_mode"):
            _resolve_anchor(ids, _CHAR_TOK, "ones_b", factory, pair)

    def test_factory_mode_returned_when_present(self):
        def factory(pair, tokenizer):
            return {"ones_b": 7}

        ids = list(range(10))
        pair = ConceptPair(prompt_pos="x", prompt_neg="y")
        assert _resolve_anchor(ids, _CHAR_TOK, "ones_b", factory, pair) == 7

    def test_factory_receives_pair_object(self):
        received = {}

        def factory(pair, tokenizer):
            received["pair"] = pair
            return {"mode_x": 2}

        ids = [0, 1, 2, 3]
        pair = ConceptPair(prompt_pos="x", prompt_neg="y", meta={"key": 42})
        _resolve_anchor(ids, _CHAR_TOK, "mode_x", factory, pair)
        assert received["pair"] is pair

    def test_factory_not_called_when_none(self):
        ids = [ord(c) for c in "hello= "]
        pos = _resolve_anchor(ids, _CHAR_TOK, "delimiter", None, None)
        assert "hello= "[pos] == "="


class TestResolveAnchorToken:
    def test_returns_position_and_decoded_token(self):
        prompt = "calc: 5+3= "
        pos, tok_str = resolve_anchor_token(prompt, _CHAR_TOK, "delimiter")
        assert tok_str == "="
        assert prompt[pos] == "="


# ─────────────────────────────────────────────────────────────────────────────
# 3. Attention sink token prepending (tokenize_qwen_input)
# ─────────────────────────────────────────────────────────────────────────────


def _minimal_tok(pad=0, bos=1, eos=2, specials=None):
    tok = MagicMock()
    tok.pad_token_id = pad
    tok.bos_token_id = bos
    tok.eos_token_id = eos
    tok.all_special_ids = (
        specials if specials is not None else [x for x in (pad, bos, eos) if x is not None]
    )
    return tok


class TestTokenizeQwenInput:
    def test_pad_prepended_for_non_special_start(self):
        tok = _minimal_tok(pad=0)
        result = tokenize_qwen_input([100, 101, 102], tok)
        assert result[0].item() == 0
        assert result.shape[0] == 4

    def test_no_sink_if_starts_with_special(self):
        tok = _minimal_tok(pad=0)
        result = tokenize_qwen_input([0, 100, 101], tok)
        assert result.shape[0] == 3
        assert result[0].item() == 0

    def test_pad_preferred_over_bos_and_eos(self):
        tok = _minimal_tok(pad=555, bos=777, eos=888)
        result = tokenize_qwen_input([100, 101], tok)
        assert result[0].item() == 555

    def test_bos_used_when_pad_none(self):
        tok = _minimal_tok(pad=None, bos=777, eos=888, specials=[777, 888])
        result = tokenize_qwen_input([100, 101], tok)
        assert result[0].item() == 777

    def test_eos_used_when_pad_and_bos_none(self):
        tok = _minimal_tok(pad=None, bos=None, eos=888, specials=[888])
        result = tokenize_qwen_input([100, 101], tok)
        assert result[0].item() == 888

    def test_list_input_treated_as_token_ids(self):
        tok = _minimal_tok(pad=0)
        result = tokenize_qwen_input([10, 20, 30], tok)
        assert result.tolist() == [0, 10, 20, 30]

    def test_tensor_input_accepted(self):
        tok = _minimal_tok(pad=0)
        result = tokenize_qwen_input(torch.tensor([10, 20, 30]), tok)
        assert result.shape[0] == 4

    def test_no_special_tokens_warns(self):
        tok = _minimal_tok(pad=None, bos=None, eos=None, specials=[])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = tokenize_qwen_input([10, 20], tok)
        assert any("sink" in str(warning.message).lower() for warning in w)
        assert result.shape[0] == 2  # returned unchanged


# ─────────────────────────────────────────────────────────────────────────────
# 4. Template grouping in extract_layer_deltas_generic
# ─────────────────────────────────────────────────────────────────────────────


class TestTemplateGrouping:
    def test_per_template_false_yields_only_all_key(self):
        model = _make_model()
        pairs = _pairs(4, "T0") + _pairs(4, "T1")
        results = extract_layer_deltas_generic(
            model, pairs, [0], torch.device("cpu"), torch.float32, per_template=False
        )
        assert set(results.keys()) == {"all"}

    def test_per_template_true_yields_all_plus_template_keys(self):
        model = _make_model()
        pairs = _pairs(4, "T0") + _pairs(4, "T1")
        results = extract_layer_deltas_generic(
            model, pairs, [0], torch.device("cpu"), torch.float32, per_template=True
        )
        assert set(results.keys()) == {"all", "T0", "T1"}

    def test_skipped_only_counted_in_all_not_templates(self):
        class _UnevenTok(_CharTok):
            def __init__(self):
                self._n = 0

            def __call__(self, text, add_special_tokens=False, return_tensors=None):
                self._n += 1
                length = 5 if self._n != 2 else 7  # mismatch on 2nd call (neg of pair 0)
                return SimpleNamespace(input_ids=list(range(1, length + 1)))

        model = _make_model()
        model.tokenizer = _UnevenTok()
        pairs = [
            ConceptPair(prompt_pos="abcde", prompt_neg="abcde", template="T0"),
            ConceptPair(prompt_pos="abcde", prompt_neg="abcde", template="T0"),
        ]
        results = extract_layer_deltas_generic(
            model, pairs, [0], torch.device("cpu"), torch.float32, per_template=True
        )
        assert results["all"].skipped == 1
        assert results["T0"].skipped == 0

    def test_n_pairs_tracks_valid_pairs(self):
        model = _make_model()
        pairs = _pairs(8)
        results = extract_layer_deltas_generic(
            model, pairs, [0], torch.device("cpu"), torch.float32, per_template=False
        )
        assert results["all"].n_pairs == 8


# ─────────────────────────────────────────────────────────────────────────────
# 5. Delta aggregation: mean vs cosine_weighted
# ─────────────────────────────────────────────────────────────────────────────


class TestDeltaAggregation:
    def _model_with_pair_acts(self, pair_acts: list[tuple[float, float]], d_model: int = 4):
        """Mock model returning pair_acts[i][0] for pos of pair i, [i][1] for neg."""
        cc = [0]

        def run(input_ids, fwd_hooks):
            c = cc[0]
            cc[0] += 1
            pair_idx, is_pos = c // 2, c % 2 == 0
            val = pair_acts[pair_idx][0] if is_pos else pair_acts[pair_idx][1]
            acts = torch.full((1, input_ids.shape[1], d_model), float(val))
            for _, hook_fn in fwd_hooks:
                hook_fn(acts, None)

        model = MagicMock()
        model.cfg.device = "cpu"
        model.tokenizer = _CharTok()
        model.run_with_hooks.side_effect = run
        return model

    def test_mean_equals_mean_of_pair_deltas(self):
        # 4 pairs, delta = 2.0 each → mean delta = 2.0
        model = self._model_with_pair_acts([(3.0, 1.0)] * 4)
        pairs = _pairs(4)
        res = extract_layer_deltas_generic(
            model, pairs, [0], torch.device("cpu"), torch.float32, per_template=False
        )
        delta = res["all"].delta[0]
        assert torch.allclose(delta, torch.full_like(delta, 2.0), atol=1e-5)

    def test_cosine_weighted_suppresses_anti_aligned_pair(self):
        # 5 pairs with delta = +1; 1 anti-aligned outlier with delta = -3.
        # mean: (5*1 - 3) / 6 = 0.333
        # cosine_weighted: outlier gets weight 0 (anti-aligned with mean) → delta = 1.0
        pair_acts = [(2.0, 1.0)] * 5 + [(0.0, 3.0)]

        results = {}
        for mode in ("mean", "cosine_weighted"):
            model = self._model_with_pair_acts(pair_acts)
            pairs = _pairs(6)
            res = extract_layer_deltas_generic(
                model,
                pairs,
                [0],
                torch.device("cpu"),
                torch.float32,
                per_template=False,
                delta_aggregation=mode,
            )
            results[mode] = res["all"].delta[0].mean().item()

        assert abs(results["mean"] - (2.0 / 6.0)) < 1e-3
        assert abs(results["cosine_weighted"] - 1.0) < 1e-3
        assert results["cosine_weighted"] > results["mean"]

    def test_mean_pair_cos_populated(self):
        model = self._model_with_pair_acts([(3.0, 1.0)] * 4)
        pairs = _pairs(4)
        res = extract_layer_deltas_generic(
            model, pairs, [0], torch.device("cpu"), torch.float32, per_template=False
        )
        assert 0 in res["all"].mean_pair_cos
        assert -1.0 <= res["all"].mean_pair_cos[0] <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 6. Carry anchor factory — position correctness
# ─────────────────────────────────────────────────────────────────────────────


class TestCarryAnchorFactory:
    """With a char tokenizer, every character is one token, so positions are predictable."""

    def test_ones_a_is_last_char_of_a_in_t0(self):
        from data.concept_datasets.carry_dataset import TEMPLATES, make_anchor_positions

        # "calc: {a}+{b}= " → "calc: 123+456= "
        # "calc: 123" = 9 chars → ones_a index = 8
        a, b = 123, 456
        positions = make_anchor_positions(TEMPLATES["T0"][0], a, b, _CHAR_TOK)
        assert positions["ones_a"] == 8

    def test_ones_b_is_last_char_of_b_in_t0(self):
        from data.concept_datasets.carry_dataset import TEMPLATES, make_anchor_positions

        # "calc: 123+456" = 13 chars → ones_b index = 12
        a, b = 123, 456
        positions = make_anchor_positions(TEMPLATES["T0"][0], a, b, _CHAR_TOK)
        assert positions["ones_b"] == 12

    def test_separator_always_equals_ones_a_plus_one(self):
        from data.concept_datasets.carry_dataset import TEMPLATES, make_anchor_positions

        for t in ("T0", "T1", "T2"):
            pos = make_anchor_positions(TEMPLATES[t][0], 234, 567, _CHAR_TOK)
            assert (
                pos["separator"] == pos["ones_a"] + 1
            ), f"{t}: separator={pos['separator']}, ones_a+1={pos['ones_a']+1}"

    def test_factory_uses_meta_not_prompt_text(self):
        from data.concept_datasets.carry_dataset import ANCHOR_FACTORY, TEMPLATES

        pair = ConceptPair(
            prompt_pos=TEMPLATES["T0"][0].format(a=123, b=456),
            prompt_neg=TEMPLATES["T0"][0].format(a=123, b=456),
            template="T0",
            meta={"a_pos": 123, "b_pos": 456},
        )
        positions = ANCHOR_FACTORY(pair, _CHAR_TOK)
        assert set(positions.keys()) == {"ones_a", "ones_b", "separator"}

    def test_factory_wrong_template_raises_key_error(self):
        from data.concept_datasets.carry_dataset import ANCHOR_FACTORY

        pair = ConceptPair(
            prompt_pos="x",
            prompt_neg="y",
            template="T_NONEXISTENT",
            meta={"a_pos": 123, "b_pos": 456},
        )
        with pytest.raises(KeyError):
            ANCHOR_FACTORY(pair, _CHAR_TOK)

    def test_all_anchor_modes_present_in_factory_output(self):
        from data.concept_datasets.carry_dataset import ANCHOR_FACTORY, ANCHOR_MODES, TEMPLATES

        pair = ConceptPair(
            prompt_pos=TEMPLATES["T0"][0].format(a=123, b=456),
            prompt_neg=TEMPLATES["T0"][0].format(a=123, b=456),
            template="T0",
            meta={"a_pos": 123, "b_pos": 456},
        )
        keys = set(ANCHOR_FACTORY(pair, _CHAR_TOK).keys())
        for mode in ANCHOR_MODES:
            assert mode in keys, f"ANCHOR_MODES declares {mode!r} but factory doesn't return it"


# ─────────────────────────────────────────────────────────────────────────────
# 7. GCD anchor factory — digit numbering from right
# ─────────────────────────────────────────────────────────────────────────────


class TestGCDAnchorFactory:
    def test_digit_1_is_ones_digit_2_is_tens_3_is_hundreds(self):
        from data.concept_datasets.gcd_dataset import TEMPLATES, make_anchor_positions

        # "calc: gcd({a},7)= " → "calc: gcd(735,7)= "
        # "calc: gcd(" = 10 chars
        # i=0: prefix "calc: gcd(7" → pos 10 → digit_from_right = 3 → digit_3
        # i=1: prefix "calc: gcd(73" → pos 11 → digit_2
        # i=2: prefix "calc: gcd(735" → pos 12 → digit_1
        positions = make_anchor_positions(TEMPLATES["T0"][0], 735, _CHAR_TOK)
        assert positions["digit_3"] == 10
        assert positions["digit_2"] == 11
        assert positions["digit_1"] == 12

    def test_digit_1_strictly_greater_than_digit_2(self):
        from data.concept_datasets.gcd_dataset import TEMPLATES, make_anchor_positions

        for t in ("T0", "T1", "T2"):
            pos = make_anchor_positions(TEMPLATES[t][0], 700, _CHAR_TOK)
            assert (
                pos["digit_1"] > pos["digit_2"]
            ), f"{t}: digit_1={pos['digit_1']} should be > digit_2={pos['digit_2']}"

    def test_three_digit_a_produces_three_digit_keys(self):
        from data.concept_datasets.gcd_dataset import TEMPLATES, make_anchor_positions

        positions = make_anchor_positions(TEMPLATES["T0"][0], 735, _CHAR_TOK)
        assert set(positions.keys()) == {"digit_1", "digit_2", "digit_3"}

    def test_all_anchor_modes_present_in_factory_output(self):
        from data.concept_datasets.gcd_dataset import ANCHOR_FACTORY, ANCHOR_MODES, TEMPLATES

        pair = ConceptPair(
            prompt_pos=TEMPLATES["T0"][0].format(a=735),
            prompt_neg=TEMPLATES["T0"][0].format(a=735),
            template="T0",
            meta={"a_pos": 735},
        )
        keys = set(ANCHOR_FACTORY(pair, _CHAR_TOK).keys())
        for mode in ANCHOR_MODES:
            assert mode in keys, f"ANCHOR_MODES declares {mode!r} but factory doesn't return it"


# ─────────────────────────────────────────────────────────────────────────────
# 8. score_and_rank: ordering, masks, uniqueness
# ─────────────────────────────────────────────────────────────────────────────


class TestScoreAndRank:
    def test_returns_exactly_top_k_features(self):
        rng = np.random.default_rng(0)
        acts = rng.random((100, 256)).astype(np.float32)
        pos_mask = np.array([True] * 50 + [False] * 50)
        assert len(score_and_rank(acts, pos_mask, top_k=20)) == 20

    def test_sorted_by_jaccard_times_abs_score_descending(self):
        rng = np.random.default_rng(1)
        acts = rng.random((60, 100)).astype(np.float32)
        pos_mask = np.array([True] * 30 + [False] * 30)
        ranked = score_and_rank(acts, pos_mask, top_k=50)
        combined = [abs(s) * j for _, s, j in ranked]
        assert combined == sorted(combined, reverse=True)

    def test_positive_score_feature_uses_pos_jaccard(self):
        # Feature 0 fires on all pos only → score > 0, jaccard = intersection(active, pos) / union
        n = 40
        acts = np.zeros((n, 2), dtype=np.float32)
        acts[: n // 2, 0] = 1.0  # feature 0: fires on all pos
        acts[n // 2 :, 1] = 1.0  # feature 1: fires on all neg
        pos_mask = np.array([True] * (n // 2) + [False] * (n // 2))

        ranked = score_and_rank(acts, pos_mask, top_k=2)
        feat_by_id = {f: (s, j) for f, s, j in ranked}

        # Feature 0: score > 0, jaccard with pos = 1.0
        assert feat_by_id[0][0] > 0
        assert abs(feat_by_id[0][1] - 1.0) < 1e-5

    def test_negative_score_feature_uses_neg_jaccard(self):
        # Feature 0 fires on all neg only → score < 0, jaccard uses neg mask
        n = 40
        acts = np.zeros((n, 1), dtype=np.float32)
        acts[n // 2 :, 0] = 1.0
        pos_mask = np.array([True] * (n // 2) + [False] * (n // 2))

        ranked = score_and_rank(acts, pos_mask, top_k=1)
        _, score, jaccard = ranked[0]
        assert score < 0
        assert abs(jaccard - 1.0) < 1e-5  # fires on all neg examples

    def test_feature_ids_are_unique(self):
        rng = np.random.default_rng(2)
        acts = rng.random((50, 128)).astype(np.float32)
        pos_mask = np.array([True] * 25 + [False] * 25)
        feat_ids = [f for f, _, _ in score_and_rank(acts, pos_mask, top_k=50)]
        assert len(feat_ids) == len(set(feat_ids)), "Duplicate feature IDs in ranking"

    def test_jaccard_values_in_unit_interval(self):
        rng = np.random.default_rng(3)
        acts = (rng.random((60, 80)) > 0.5).astype(np.float32)
        pos_mask = np.array([True] * 30 + [False] * 30)
        for _, _, jac in score_and_rank(acts, pos_mask, top_k=80):
            assert 0.0 <= jac <= 1.0 + 1e-6

    def test_top_k_capped_at_feature_count(self):
        acts = np.ones((10, 5), dtype=np.float32)
        pos_mask = np.array([True] * 5 + [False] * 5)
        assert len(score_and_rank(acts, pos_mask, top_k=1000)) == 5


# ─────────────────────────────────────────────────────────────────────────────
# 9. collect_layer_residuals: shapes, zero-fill, anchor offset
# ─────────────────────────────────────────────────────────────────────────────


class TestCollectLayerResiduals:
    def _make_model(self, d_model: int = 8) -> MagicMock:
        model = MagicMock()
        model.cfg.device = "cpu"
        model.cfg.d_model = d_model
        model.tokenizer = _CharTok()
        model.feature_input_hook = "hook_resid_post"

        def run(input_ids, fwd_hooks):
            acts = torch.ones(1, input_ids.shape[1], d_model)
            for _, fn in fwd_hooks:
                fn(acts, None)

        model.run_with_hooks.side_effect = run
        return model

    def test_output_shape(self):
        from experiments.concept_localization.analyze import collect_layer_residuals

        model = self._make_model(d_model=8)
        prompts_and_anchors = [([10, 20, 30, 40, 50], 2) for _ in range(5)]
        H = collect_layer_residuals(model, prompts_and_anchors, [0, 1, 2])

        assert set(H.keys()) == {0, 1, 2}
        for layer in (0, 1, 2):
            assert H[layer].shape == (5, 8), f"Layer {layer}: {H[layer].shape}"

    def test_out_of_range_anchor_fills_zeros(self):
        from experiments.concept_localization.analyze import collect_layer_residuals

        # raw anchor = 100 → sink_anchor = 101 >> actual seq length → hook condition fails
        model = self._make_model(d_model=6)
        prompts_and_anchors = [([10, 20, 30], 100)]  # 3 tokens + sink = 4 total
        H = collect_layer_residuals(model, prompts_and_anchors, [0])
        assert np.allclose(H[0], 0.0), "Out-of-range anchor should fill with zeros"

    def test_valid_anchor_does_not_fill_zeros(self):
        from experiments.concept_localization.analyze import collect_layer_residuals

        # raw anchor = 2 → sink_anchor = 3; seq of 5 tokens + sink = 6 → 3 < 6, valid
        model = self._make_model(d_model=4)
        prompts_and_anchors = [([10, 20, 30, 40, 50], 2)]
        H = collect_layer_residuals(model, prompts_and_anchors, [0])
        # All-ones activations → non-zero output
        assert not np.allclose(H[0], 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 10. Sharpness analysis
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeSharpness:
    from experiments.concept_localization.analyze import compute_sharpness  # noqa: F811

    def _ld(self, norms: dict[int, float]) -> LayerDeltas:
        ld = LayerDeltas()
        for layer, n in norms.items():
            ld.delta[layer] = torch.ones(4) * n  # norm = 2|n| with d=4
        return ld

    def test_peak_layer_at_maximum_norm(self):
        from experiments.concept_localization.analyze import compute_sharpness

        result = compute_sharpness(self._ld({0: 1.0, 1: 5.0, 2: 2.0, 3: 0.5}))
        assert result.peak_layer == 1

    def test_sharpness_index_in_unit_interval(self):
        from experiments.concept_localization.analyze import compute_sharpness

        result = compute_sharpness(self._ld({0: 1.0, 1: 5.0, 2: 2.0, 3: 0.5}))
        assert 0.0 <= result.sharpness_index <= 1.0

    def test_inter_layer_cos_length_is_n_layers_minus_one(self):
        from experiments.concept_localization.analyze import compute_sharpness

        result = compute_sharpness(self._ld({0: 1.0, 1: 2.0, 2: 3.0, 3: 2.0}))
        assert len(result.inter_layer_cos) == 3
        for c in result.inter_layer_cos:
            assert -1.0 - 1e-5 <= c <= 1.0 + 1e-5

    def test_normalised_flag_true_when_mean_act_norm_provided(self):
        from experiments.concept_localization.analyze import compute_sharpness

        ld = LayerDeltas()
        for l in range(4):
            ld.delta[l] = torch.ones(4)
            ld.mean_act_norm[l] = 2.0
        assert compute_sharpness(ld).normalised is True

    def test_normalised_flag_false_when_mean_act_norm_absent(self):
        from experiments.concept_localization.analyze import compute_sharpness

        ld = LayerDeltas()
        for l in range(4):
            ld.delta[l] = torch.ones(4)
        assert compute_sharpness(ld).normalised is False


# ─────────────────────────────────────────────────────────────────────────────
# 11. Template consistency
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeTemplateConsistency:
    def test_empty_when_only_all_key(self):
        from experiments.concept_localization.analyze import compute_template_consistency

        results = {"all": LayerDeltas(delta={0: torch.ones(4)})}
        assert compute_template_consistency(results) == {}

    def test_empty_when_only_one_template(self):
        from experiments.concept_localization.analyze import compute_template_consistency

        results = {
            "all": LayerDeltas(delta={0: torch.ones(4)}),
            "T0": LayerDeltas(delta={0: torch.ones(4)}),
        }
        assert compute_template_consistency(results) == {}

    def test_two_templates_produce_pairwise_key(self):
        from experiments.concept_localization.analyze import compute_template_consistency

        results = {
            "all": LayerDeltas(delta={0: torch.ones(4)}),
            "T0": LayerDeltas(delta={0: torch.ones(4)}),
            "T1": LayerDeltas(delta={0: torch.ones(4) * 2}),
        }
        consistency = compute_template_consistency(results)
        assert "T0_vs_T1" in consistency[0]

    def test_parallel_vectors_give_cosine_one(self):
        from experiments.concept_localization.analyze import compute_template_consistency

        v = torch.tensor([1.0, 0.0, 0.0, 0.0])
        results = {
            "all": LayerDeltas(delta={5: v}),
            "T0": LayerDeltas(delta={5: v.clone()}),
            "T1": LayerDeltas(delta={5: v * 3.0}),
        }
        cos = compute_template_consistency(results)[5]["T0_vs_T1"]
        assert abs(cos - 1.0) < 1e-4

    def test_orthogonal_vectors_give_cosine_zero(self):
        from experiments.concept_localization.analyze import compute_template_consistency

        results = {
            "all": LayerDeltas(delta={0: torch.tensor([1.0, 0.0])}),
            "T0": LayerDeltas(delta={0: torch.tensor([1.0, 0.0])}),
            "T1": LayerDeltas(delta={0: torch.tensor([0.0, 1.0])}),
        }
        cos = compute_template_consistency(results)[0]["T0_vs_T1"]
        assert abs(cos) < 1e-4


# ─────────────────────────────────────────────────────────────────────────────
# 12. project_onto_features
# ─────────────────────────────────────────────────────────────────────────────


class TestProjectOntoFeatures:
    def _model_with_tc(self, n_features: int = 16, d_model: int = 8) -> MagicMock:
        W_enc = torch.randn(n_features, d_model)
        tc = SimpleNamespace(W_enc=W_enc)
        model = MagicMock()
        model.transcoders = {0: tc}
        return model

    def test_returns_top_k_matches(self):
        from experiments.concept_localization.analyze import project_onto_features

        model = self._model_with_tc(n_features=32, d_model=8)
        ld = LayerDeltas(delta={0: torch.randn(8)})
        result = project_onto_features(model, ld, top_k=10)
        assert len(result[0]) == 10

    def test_top_k_capped_at_feature_count(self):
        from experiments.concept_localization.analyze import project_onto_features

        model = self._model_with_tc(n_features=5, d_model=8)
        ld = LayerDeltas(delta={0: torch.randn(8)})
        result = project_onto_features(model, ld, top_k=100)
        assert len(result[0]) == 5

    def test_zero_norm_delta_layer_skipped(self):
        from experiments.concept_localization.analyze import project_onto_features

        model = self._model_with_tc()
        ld = LayerDeltas(delta={0: torch.zeros(8)})
        result = project_onto_features(model, ld, top_k=5)
        assert 0 not in result

    def test_missing_w_enc_layer_skipped(self):
        from experiments.concept_localization.analyze import project_onto_features

        model = MagicMock()
        model.transcoders = {0: SimpleNamespace()}  # no W_enc
        ld = LayerDeltas(delta={0: torch.randn(8)})
        assert 0 not in project_onto_features(model, ld, top_k=5)

    def test_feature_ids_within_valid_range(self):
        from experiments.concept_localization.analyze import project_onto_features

        n_features = 20
        model = self._model_with_tc(n_features=n_features)
        ld = LayerDeltas(delta={0: torch.randn(8)})
        for fm in project_onto_features(model, ld, top_k=10)[0]:
            assert 0 <= fm.feature_id < n_features

    def test_cos_sim_in_minus_one_to_one(self):
        from experiments.concept_localization.analyze import project_onto_features

        model = self._model_with_tc()
        ld = LayerDeltas(delta={0: torch.randn(8)})
        for fm in project_onto_features(model, ld, top_k=16)[0]:
            assert -1.0 - 1e-5 <= fm.cos_sim <= 1.0 + 1e-5


# ─────────────────────────────────────────────────────────────────────────────
# 13. Dataset generation sanity checks
# ─────────────────────────────────────────────────────────────────────────────


class TestDatasetGeneration:
    def test_carry_pairs_reproducible_with_same_seed(self):
        from data.concept_datasets.carry_dataset import generate_carry_pairs

        p1 = generate_carry_pairs(n_per_template=10, seed=42)
        p2 = generate_carry_pairs(n_per_template=10, seed=42)
        assert len(p1) == len(p2)
        for a, b in zip(p1, p2, strict=False):
            assert a.prompt_pos == b.prompt_pos and a.prompt_neg == b.prompt_neg

    def test_gcd_pairs_reproducible_with_same_seed(self):
        from data.concept_datasets.gcd_dataset import generate_gcd_pairs

        p1 = generate_gcd_pairs(n_per_template=10, seed=0)
        p2 = generate_gcd_pairs(n_per_template=10, seed=0)
        assert all(a.prompt_pos == b.prompt_pos for a, b in zip(p1, p2, strict=False))

    def test_carry_pos_has_carry_neg_does_not(self):
        from data.concept_datasets.carry_dataset import generate_carry_pairs

        for p in generate_carry_pairs(n_per_template=20):
            a_pos, b_pos = p.meta["a_pos"], p.meta["b_pos"]
            a_neg, b_neg = p.meta["a_neg"], p.meta["b_neg"]
            assert (a_pos % 10 + b_pos % 10) >= 10, "Pos pair must carry at units digit"
            assert (a_neg % 10 + b_neg % 10) < 10, "Neg pair must not carry at units digit"

    def test_gcd_pos_divisible_by_7_neg_not(self):
        from data.concept_datasets.gcd_dataset import generate_gcd_pairs

        for p in generate_gcd_pairs(n_per_template=20):
            assert p.meta["a_pos"] % 7 == 0, f"a_pos={p.meta['a_pos']} not divisible by 7"
            assert p.meta["a_neg"] % 7 != 0, f"a_neg={p.meta['a_neg']} is divisible by 7"

    def test_gcd_same_digit_count_for_pos_and_neg(self):
        from data.concept_datasets.gcd_dataset import generate_gcd_pairs

        for p in generate_gcd_pairs(n_per_template=20):
            assert len(str(p.meta["a_pos"])) == len(
                str(p.meta["a_neg"])
            ), f"Digit mismatch: a_pos={p.meta['a_pos']} vs a_neg={p.meta['a_neg']}"

    def test_carry_template_values_are_valid_keys(self):
        from data.concept_datasets.carry_dataset import TEMPLATES, generate_carry_pairs

        for p in generate_carry_pairs(n_per_template=5):
            assert p.template in TEMPLATES

    def test_gcd_template_values_are_valid_keys(self):
        from data.concept_datasets.gcd_dataset import TEMPLATES, generate_gcd_pairs

        for p in generate_gcd_pairs(n_per_template=5):
            assert p.template in TEMPLATES

    def test_carry_meta_has_required_keys(self):
        from data.concept_datasets.carry_dataset import generate_carry_pairs

        for p in generate_carry_pairs(n_per_template=5):
            assert {"a_pos", "b_pos", "a_neg", "b_neg"}.issubset(p.meta.keys())

    def test_gcd_meta_has_required_keys(self):
        from data.concept_datasets.gcd_dataset import generate_gcd_pairs

        for p in generate_gcd_pairs(n_per_template=5):
            assert {"a_pos", "a_neg"}.issubset(p.meta.keys())


# ─────────────────────────────────────────────────────────────────────────────
# 14. cluster_top_features: smoke tests
# ─────────────────────────────────────────────────────────────────────────────


class TestClusterTopFeatures:
    def test_output_types_and_cluster_id_range(self):
        rng = np.random.default_rng(0)
        acts = rng.random((100, 200)).astype(np.float32)
        pos_mask = np.array([True] * 50 + [False] * 50)
        result = cluster_top_features(acts, pos_mask, top_frac=0.15, n_clusters=5)
        assert len(result) > 0
        for feat_id, score, cluster_id in result:
            assert isinstance(feat_id, int)
            assert isinstance(score, float)
            assert 0 <= cluster_id < 5

    def test_sorted_by_cluster_id_first(self):
        rng = np.random.default_rng(1)
        acts = rng.random((80, 100)).astype(np.float32)
        pos_mask = np.array([True] * 40 + [False] * 40)
        result = cluster_top_features(acts, pos_mask, top_frac=0.2, n_clusters=3)
        cluster_ids = [c for _, _, c in result]
        assert cluster_ids == sorted(cluster_ids)
