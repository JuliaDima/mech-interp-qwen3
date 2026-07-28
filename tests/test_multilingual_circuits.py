"""Tests for the multilingual circuits intervention experiment.

Most tests run on CPU with no network access and cover the pure-logic
components that are most likely to harbour silent bugs: token-ID resolution,
feature partitioning, divergence detection, and the attention-sink
tokenisation helper. The intervention-sweep/swap-hook tests additionally need
the `tiny_model` fixture, which loads a real (if tiny) tokenizer and so
requires either a local HF cache or network access; they skip automatically
when neither is available.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformerConfig

from experiments.multilingual_circuits.run import (
    ANTONYM_TOKENS,
    COLD_TOKENS,
    SYNONYM_TOKENS,
    _build_swap_hooks,
    _find_divergence_pos,
    _find_differing_positions,
    find_exclusive_features,
    intervention_sweep,
    _resolve_ids,
    _resolve_next_token_ids,
)
from mechinterp_qwen3.attribution_model import AttributionModel
from mechinterp_qwen3.transcoder.single_layer_transcoder import SingleLayerTranscoder, TranscoderSet
from mechinterp_qwen3.utils.token_utils import tokenize_qwen_input
from tests.conftest import tokenizer_reachable


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_tokenizer(vocab: dict[str, int], special_ids: list[int] | None = None) -> MagicMock:
    """Return a minimal mock tokenizer backed by a fixed vocabulary.

    ``vocab`` maps surface string to token-id.  The mock implements
    ``__call__`` so that ``tokenizer(text, add_special_tokens=False).input_ids``
    returns ``[vocab[text]]`` when the text is a known key, or an empty list
    otherwise.
    """
    tok = MagicMock()
    tok.pad_token_id = 0
    tok.bos_token_id = 1
    tok.eos_token_id = 2
    tok.all_special_ids = special_ids if special_ids is not None else [0, 1, 2]

    def _call(text, *, add_special_tokens=True, return_tensors=None, **kw):
        ids = [vocab[text]] if text in vocab else []
        result = MagicMock()
        result.input_ids = ids
        if return_tensors == "pt":
            result.input_ids = torch.tensor([ids]) if ids else torch.zeros((1, 0), dtype=torch.long)
        return result

    tok.side_effect = _call
    tok.__call__ = _call
    return tok


@pytest.fixture
def tiny_cfg():
    return HookedTransformerConfig(
        n_layers=4,
        d_model=16,
        n_ctx=12,
        d_head=4,
        n_heads=4,
        d_mlp=32,
        d_vocab=200,
        act_fn="relu",
        tokenizer_name="gpt2",
    )


@pytest.fixture
def tiny_transcoder_set(tiny_cfg):
    transcoders = {
        i: SingleLayerTranscoder(
            d_model=tiny_cfg.d_model,
            d_transcoder=64,
            activation_function=F.relu,
            layer_idx=i,
            dtype=torch.float32,
        )
        for i in range(tiny_cfg.n_layers)
    }
    return TranscoderSet(
        transcoders=transcoders,
        feature_input_hook="mlp.hook_in",
        feature_output_hook="mlp.hook_out",
        scan="test_scan",
    )


@pytest.fixture
def tiny_model(tiny_cfg, tiny_transcoder_set):
    if not tokenizer_reachable("gpt2"):
        pytest.skip("gpt2 tokenizer not reachable (no HF cache, no network)")
    return AttributionModel.from_config(tiny_cfg, tiny_transcoder_set)


# ---------------------------------------------------------------------------
# _resolve_ids
# ---------------------------------------------------------------------------


def test_resolve_ids_returns_first_subtoken():
    """Each surface form maps to its *first* subtoken id."""
    vocab = {"large": 10, "big": 20, "small": 30}
    tok = _make_tokenizer(vocab)
    ids = _resolve_ids(tok, ["large", "big"])
    assert set(ids) == {10, 20}


def test_resolve_ids_deduplicates_same_first_token():
    """Two surface forms that share the same first subtoken produce one entry."""
    vocab = {"small": 30, "smaller": 30}
    tok = _make_tokenizer(vocab)
    ids = _resolve_ids(tok, ["small", "smaller"])
    assert ids == [30]


def test_resolve_ids_skips_empty_tokenisation():
    """Surface forms that tokenise to nothing are silently ignored."""
    vocab = {"large": 10}
    tok = _make_tokenizer(vocab)
    ids = _resolve_ids(tok, ["large", "unknown_form_xyz"])
    assert ids == [10]


def test_resolve_ids_empty_input():
    tok = _make_tokenizer({})
    assert _resolve_ids(tok, []) == []


def test_resolve_ids_preserves_order():
    """IDs are returned in insertion order (first occurrence wins for duplicates)."""
    vocab = {"a": 1, "b": 2, "c": 3}
    tok = _make_tokenizer(vocab)
    ids = _resolve_ids(tok, ["c", "a", "b"])
    assert ids == [3, 1, 2]


def test_resolve_next_token_ids_uses_prompt_context():
    """Tracked next-token ids must match the prompt continuation, not standalone text."""
    tok = MagicMock()
    vocab = {
        'The opposite of "small" is "': [10, 20, 30],
        'The opposite of "small" is "large': [10, 20, 30, 100],
        " large": [200],
        "large": [100],
    }

    def _call(text, *, add_special_tokens=True, return_tensors=None, **kw):
        result = MagicMock()
        result.input_ids = vocab[text]
        return result

    tok.side_effect = _call
    tok.__call__ = _call

    prompt = 'The opposite of "small" is "'
    assert _resolve_ids(tok, [" large"]) == [200]
    assert _resolve_next_token_ids(tok, prompt, ["large"]) == [100]


# ---------------------------------------------------------------------------
# _find_divergence_pos
# ---------------------------------------------------------------------------


def test_divergence_pos_known_index():
    a = [1, 2, 3, 99, 5]
    b = [1, 2, 3, 77, 5]
    assert _find_divergence_pos(a, b) == 3


def test_divergence_pos_first_token():
    a = [99, 2, 3]
    b = [1, 2, 3]
    assert _find_divergence_pos(a, b) == 0


def test_divergence_pos_identical_sequences():
    """Identical sequences return the last shared index."""
    a = [1, 2, 3]
    assert _find_divergence_pos(a, a) == len(a) - 1


def test_divergence_pos_different_lengths_diverge_early():
    a = [1, 2, 77, 4, 5]
    b = [1, 2, 3]
    assert _find_divergence_pos(a, b) == 2


def test_divergence_pos_prefix_match_shorter_b():
    """When b is a prefix of a, return len(b)-1."""
    a = [1, 2, 3, 4]
    b = [1, 2, 3]
    assert _find_divergence_pos(a, b) == 2


def test_differing_positions_returns_full_operand_span():
    """Multi-subtoken operands should edit every differing subtoken."""
    assert _find_differing_positions([1, 2, 3, 4], [1, 9, 8, 4]) == [1, 2]


def test_differing_positions_handles_length_mismatch():
    assert _find_differing_positions([1, 2, 3], [1, 2, 3, 4]) == [3]


# ---------------------------------------------------------------------------
# find_exclusive_features
# ---------------------------------------------------------------------------


def _make_acts(d: int, vals: dict[int, float]) -> torch.Tensor:
    t = torch.zeros(d)
    for idx, v in vals.items():
        t[idx] = v
    return t


def test_exclusive_features_correct_partition():
    """Features active only in src go to suppress; only in tgt go to inject."""
    d = 20
    src = _make_acts(d, {0: 10.0, 1: 0.0, 2: 0.5, 3: 8.0})
    tgt = _make_acts(d, {0: 0.0, 1: 9.0, 2: 0.5, 3: 0.0})

    sup, inj, vals = find_exclusive_features(src_acts={0: src}, tgt_acts={0: tgt}, layers=[0], top_k=d)

    assert 0 in sup[0], "Feature 0 should be suppressed (active only in src)"
    assert 1 in inj[0], "Feature 1 should be injected (active only in tgt)"
    assert 3 in sup[0], "Feature 3 should be suppressed (active only in src)"
    assert 1 not in sup[0], "Feature 1 must not be in suppress set"
    assert 0 not in inj[0], "Feature 0 must not be in inject set"


def test_exclusive_features_shared_feature_excluded():
    """A feature active in both contexts at similar magnitude is excluded from both sets."""
    d = 10
    src = _make_acts(d, {5: 8.0})
    tgt = _make_acts(d, {5: 8.0})

    sup, inj, _ = find_exclusive_features(src_acts={0: src}, tgt_acts={0: tgt}, layers=[0], top_k=d)

    assert 5 not in sup[0]
    assert 5 not in inj[0]


def test_exclusive_features_threshold_boundary():
    """A target activation exactly at the exclusivity boundary is not exclusive."""
    exclusivity = 0.5
    d = 10
    src = _make_acts(d, {3: 10.0})
    tgt_boundary = _make_acts(d, {3: 5.0})  # tgt == 0.5 * src → not strictly below

    sup, _, _ = find_exclusive_features(
        src_acts={0: src}, tgt_acts={0: tgt_boundary}, layers=[0], top_k=d, exclusivity=exclusivity
    )
    assert 3 not in sup[0], "At boundary (tgt == exclusivity*src), feature should not be suppressed"

    tgt_below = _make_acts(d, {3: 4.9})  # strictly below threshold
    sup2, _, _ = find_exclusive_features(
        src_acts={0: src}, tgt_acts={0: tgt_below}, layers=[0], top_k=d, exclusivity=exclusivity
    )
    assert 3 in sup2[0], "Below boundary (tgt < exclusivity*src), feature should be suppressed"


def test_exclusive_features_suppress_inject_disjoint():
    """The suppress and inject sets for a layer must not overlap."""
    d = 30
    torch.manual_seed(0)
    src = torch.rand(d) * 5
    tgt = torch.rand(d) * 5

    sup, inj, _ = find_exclusive_features(src_acts={0: src}, tgt_acts={0: tgt}, layers=[0], top_k=d)

    assert set(sup[0]).isdisjoint(set(inj[0])), "Suppress and inject must be disjoint"


def test_exclusive_features_all_zero_activations():
    """All-zero activation tensors should yield empty suppress and inject sets."""
    d = 16
    zero = torch.zeros(d)

    sup, inj, vals = find_exclusive_features(
        src_acts={0: zero}, tgt_acts={0: zero}, layers=[0], top_k=d
    )
    assert sup[0] == []
    assert inj[0] == []
    assert vals[0] == []


def test_exclusive_features_inject_values_match_target():
    """Injection values must correspond to the target activation magnitudes."""
    d = 10
    src = _make_acts(d, {7: 0.0})
    tgt = _make_acts(d, {7: 6.5})

    _, inj, vals = find_exclusive_features(src_acts={0: src}, tgt_acts={0: tgt}, layers=[0], top_k=d)

    assert 7 in inj[0]
    idx = inj[0].index(7)
    assert abs(vals[0][idx] - 6.5) < 1e-4


# ---------------------------------------------------------------------------
# intervention_sweep  (uses tiny model, CPU only)
# ---------------------------------------------------------------------------


def test_intervention_sweep_output_shape(tiny_model):
    """Output dict maps each tracked token id to a list of the same length as alphas."""
    tokens = torch.randint(0, 100, (5,))
    track = [0, 1, 2]
    alphas = [0.0, 0.5, 1.0]

    result = intervention_sweep(
        tiny_model, tokens,
        suppress_by_layer={}, inject_by_layer={}, inject_vals_by_layer={},
        alphas=alphas, track_token_ids=track,
    )

    assert set(result.keys()) == set(track)
    for tid in track:
        assert len(result[tid]) == len(alphas)
        for p in result[tid]:
            assert 0.0 <= p <= 1.0


def test_intervention_sweep_reproducible(tiny_model):
    """The same alpha sweep called twice gives identical results."""
    tokens = torch.randint(0, 100, (5,))
    track = [10, 20]
    alphas = [0.0, 1.0]

    r1 = intervention_sweep(
        tiny_model, tokens, {}, {}, {}, alphas, track
    )
    r2 = intervention_sweep(
        tiny_model, tokens, {}, {}, {}, alphas, track
    )

    for tid in track:
        assert r1[tid] == r2[tid]


def test_swap_hook_injection_adds_feature_delta(tiny_model):
    """Injecting a feature should add only the decoded feature delta to MLP output."""
    layer = 0
    tc = tiny_model.transcoders[layer]
    tc.W_enc.data.zero_()
    tc.b_enc.data.zero_()
    tc.W_dec.data.zero_()
    tc.W_dec.data[0, 0] = 1.0
    tc.b_dec.data.zero_()
    tc.W_skip = None

    mlp_in = torch.zeros(1, 2, tiny_model.cfg.d_model)
    mlp_out = torch.zeros_like(mlp_in)

    (_, capture), (_, swap) = _build_swap_hooks(
        tiny_model, layer, suppress_ids=[], inject_ids=[0], inject_vals=[10.0],
        alpha=2.0, target_pos=-1,
    )

    capture(mlp_in, None)
    out = swap(mlp_out, None)

    assert out[0, 0, 0].item() == pytest.approx(0.0)
    assert out[0, -1, 0].item() == pytest.approx(20.0)


def test_swap_hook_applies_to_multiple_positions(tiny_model):
    """A span edit should touch every requested position and leave others unchanged."""
    layer = 0
    tc = tiny_model.transcoders[layer]
    tc.W_enc.data.zero_()
    tc.b_enc.data.zero_()
    tc.W_dec.data.zero_()
    tc.W_dec.data[0, 0] = 1.0
    tc.b_dec.data.zero_()
    tc.W_skip = None

    mlp_in = torch.zeros(1, 4, tiny_model.cfg.d_model)
    mlp_out = torch.zeros_like(mlp_in)

    (_, capture), (_, swap) = _build_swap_hooks(
        tiny_model, layer, suppress_ids=[], inject_ids=[0], inject_vals=[3.0],
        alpha=1.0, target_pos=[1, 2],
    )

    capture(mlp_in, None)
    out = swap(mlp_out, None)

    assert out[0, 0, 0].item() == pytest.approx(0.0)
    assert out[0, 1, 0].item() == pytest.approx(3.0)
    assert out[0, 2, 0].item() == pytest.approx(3.0)
    assert out[0, 3, 0].item() == pytest.approx(0.0)


def test_intervention_sweep_alpha_zero_is_noop(tiny_model):
    """Alpha zero should report the unmodified baseline, not ablated source features."""
    torch.manual_seed(123)
    tokens = torch.randint(0, 100, (6,))
    track = [5, 6, 7]

    r_noop = intervention_sweep(
        tiny_model, tokens, {}, {}, {},
        alphas=[0.0], track_token_ids=track,
    )
    r_alpha_zero = intervention_sweep(
        tiny_model, tokens,
        suppress_by_layer={0: list(range(64))},
        inject_by_layer={0: [0]},
        inject_vals_by_layer={0: [50.0]},
        alphas=[0.0],
        track_token_ids=track,
    )

    assert r_alpha_zero == r_noop


def test_swap_hook_suppression_is_continuous(tiny_model):
    """Nonzero alpha should attenuate source features gradually, not hard-zero them."""
    layer = 0
    tc = tiny_model.transcoders[layer]
    tc.W_enc.data.zero_()
    tc.b_enc.data.zero_()
    tc.b_enc.data[0] = 10.0
    tc.W_dec.data.zero_()
    tc.W_dec.data[0, 0] = 1.0
    tc.b_dec.data.zero_()
    tc.W_skip = None

    mlp_in = torch.zeros(1, 2, tiny_model.cfg.d_model)
    mlp_out = torch.zeros_like(mlp_in)

    (_, capture), (_, swap) = _build_swap_hooks(
        tiny_model, layer, suppress_ids=[0], inject_ids=[], inject_vals=[],
        alpha=0.25, target_pos=-1,
    )

    capture(mlp_in, None)
    out = swap(mlp_out, None)

    assert out[0, 0, 0].item() == pytest.approx(0.0)
    assert out[0, -1, 0].item() == pytest.approx(-2.5)


# ---------------------------------------------------------------------------
# tokenize_qwen_input
# ---------------------------------------------------------------------------


def _sink_tokenizer(pad_id: int = 0, bos_id: int = 1) -> MagicMock:
    tok = MagicMock()
    tok.pad_token_id = pad_id
    tok.bos_token_id = bos_id
    tok.eos_token_id = 2
    tok.all_special_ids = [pad_id, bos_id, 2]

    def _call(text, *, add_special_tokens=True, return_tensors=None, **kw):
        ids = [ord(c) % 100 + 10 for c in text[:4]]  # deterministic fake ids
        result = MagicMock()
        result.input_ids = torch.tensor([ids])
        return result

    tok.side_effect = _call
    tok.__call__ = _call
    return tok


def test_tokenize_prepends_sink_when_absent():
    """A sink token is prepended when the sequence does not start with one."""
    tok = _sink_tokenizer(pad_id=0)
    tokens = tokenize_qwen_input("hello", tok)
    assert tokens[0].item() == 0, "First token should be the PAD/sink token"
    assert len(tokens) > 1


def test_tokenize_no_double_sink():
    """If the first token is already a special token, no second sink is prepended."""
    tok = _sink_tokenizer(pad_id=0)
    # Feed a tensor that already starts with the sink token id.
    pre_sunk = torch.tensor([0, 10, 20, 30])
    tokens = tokenize_qwen_input(pre_sunk, tok)
    assert tokens[0].item() == 0
    assert tokens[1].item() == 10


def test_tokenize_tensor_passthrough_shape():
    """A 1-D tensor input returns a 1-D tensor (possibly with sink prepended)."""
    tok = _sink_tokenizer(pad_id=0)
    inp = torch.tensor([50, 60, 70])
    tokens = tokenize_qwen_input(inp, tok)
    assert tokens.ndim == 1


# ---------------------------------------------------------------------------
# Layer range sanity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_layers", [4, 8, 12, 36])
def test_layer_ranges_nonempty_and_in_bounds(n_layers):
    """All three layer groups must be non-empty and index valid layers."""
    op_layers   = list(range(n_layers // 3, 2 * n_layers // 3))
    oper_layers = list(range(0, n_layers // 3))
    lang_layers = list(range(0, max(1, n_layers // 5)))

    for name, layers in [("op", op_layers), ("oper", oper_layers), ("lang", lang_layers)]:
        assert len(layers) >= 1, f"{name}_layers must be non-empty for n_layers={n_layers}"
        assert all(0 <= l < n_layers for l in layers), (
            f"{name}_layers contains out-of-bounds index for n_layers={n_layers}"
        )


def test_op_and_oper_layers_are_disjoint():
    """Operation and operand layer groups must not overlap for any realistic model size."""
    for n_layers in [12, 36]:
        op_layers   = set(range(n_layers // 3, 2 * n_layers // 3))
        oper_layers = set(range(0, n_layers // 3))
        assert op_layers.isdisjoint(oper_layers), (
            f"op_layers and oper_layers overlap for n_layers={n_layers}"
        )


# ---------------------------------------------------------------------------
# Token constant sanity checks
# ---------------------------------------------------------------------------


def test_antonym_token_constants_nonempty():
    """ANTONYM_TOKENS, SYNONYM_TOKENS, and COLD_TOKENS must have entries for en/fr/zh."""
    for lang in ("en", "fr", "zh"):
        assert lang in ANTONYM_TOKENS and len(ANTONYM_TOKENS[lang]) >= 1
        assert lang in SYNONYM_TOKENS and len(SYNONYM_TOKENS[lang]) >= 1
        assert lang in COLD_TOKENS and len(COLD_TOKENS[lang]) >= 1


def test_antonym_and_synonym_tokens_are_disjoint_for_english():
    """English antonym tokens (large/big) must not overlap with synonym tokens (small/tiny/little).

    An overlap would mean the same token is being tracked as both original and
    intervention target, making it impossible to observe a directional shift.
    """
    ant = set(ANTONYM_TOKENS["en"])
    syn = set(SYNONYM_TOKENS["en"])
    assert ant.isdisjoint(syn), (
        f"English antonym and synonym surface forms overlap: {ant & syn}"
    )
