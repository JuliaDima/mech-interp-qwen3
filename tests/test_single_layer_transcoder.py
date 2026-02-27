import os
import tempfile

import pytest
import torch
import torch.nn.functional as F

from mechinterp_qwen3.transcoder.single_layer_transcoder import (
    SingleLayerTranscoder,
    TranscoderSet,
    load_relu_transcoder,
)


def test_single_layer_transcoder_basic():
    d_model = 16
    d_transcoder = 64
    layer_idx = 0
    device = torch.device("cpu")

    transcoder = SingleLayerTranscoder(
        d_model=d_model,
        d_transcoder=d_transcoder,
        activation_function=F.relu,
        layer_idx=layer_idx,
        device=device,
        dtype=torch.float32,  # explicit float32 for testing
    )

    assert transcoder.W_enc.shape == (d_transcoder, d_model)
    assert transcoder.W_dec.shape == (d_transcoder, d_model)
    assert transcoder.b_enc.shape == (d_transcoder,)
    assert transcoder.b_dec.shape == (d_model,)

    # Test forward
    input_acts = torch.randn(2, 5, d_model)
    output = transcoder(input_acts)

    assert output.shape == input_acts.shape
    assert output.dtype == torch.float32


def test_single_layer_transcoder_with_skip():
    d_model = 16
    d_transcoder = 64
    transcoder = SingleLayerTranscoder(
        d_model=d_model,
        d_transcoder=d_transcoder,
        activation_function=F.relu,
        layer_idx=0,
        skip_connection=True,
        dtype=torch.float32,
    )

    assert transcoder.W_skip is not None
    assert transcoder.W_skip.shape == (d_model, d_model)

    input_acts = torch.randn(2, d_model)
    output = transcoder(input_acts)
    assert output.shape == input_acts.shape


def test_single_layer_transcoder_encode_decode():
    d_model = 8
    d_transcoder = 32
    transcoder = SingleLayerTranscoder(
        d_model=d_model,
        d_transcoder=d_transcoder,
        activation_function=F.relu,
        layer_idx=1,
        dtype=torch.float32,
    )

    input_acts = torch.randn(4, d_model)
    # Encode without activation
    pre_acts = transcoder.encode(input_acts, apply_activation_function=False)
    assert pre_acts.shape == (4, d_transcoder)

    # Encode with activation
    acts = transcoder.encode(input_acts, apply_activation_function=True)
    assert torch.all(acts >= 0)

    # Decode
    reconstructed = transcoder.decode(acts)
    assert reconstructed.shape == (4, d_model)


def test_transcoder_set_basic():
    d_model = 16
    d_transcoder = 64
    t1 = SingleLayerTranscoder(d_model, d_transcoder, F.relu, 0)
    t2 = SingleLayerTranscoder(d_model, d_transcoder, F.relu, 1)

    transcoders = {0: t1, 1: t2}
    tset = TranscoderSet(
        transcoders=transcoders,
        feature_input_hook="hook_in",
        feature_output_hook="hook_out",
        scan="test_scan",
    )

    assert len(tset) == 2
    assert tset[0] == t1
    assert tset[1] == t2
    assert tset.feature_input_hook == "hook_in"
    assert tset.feature_output_hook == "hook_out"


@pytest.fixture
def mock_safetensors_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "transcoder.safetensors")
        d_model = 16
        d_transcoder = 64

        # Create dummy weights
        W_enc = torch.randn(d_transcoder, d_model)
        W_dec = torch.randn(d_transcoder, d_model)
        b_enc = torch.randn(d_transcoder)
        b_dec = torch.randn(d_model)

        weights = {"W_enc": W_enc, "W_dec": W_dec, "b_enc": b_enc, "b_dec": b_dec}

        from safetensors.torch import save_file

        save_file(weights, path)
        yield path


def test_load_relu_transcoder(mock_safetensors_file):
    transcoder = load_relu_transcoder(
        mock_safetensors_file, layer=0, lazy_encoder=False, lazy_decoder=False
    )

    assert isinstance(transcoder, SingleLayerTranscoder)
    assert transcoder.d_model == 16
    assert transcoder.d_transcoder == 64
    assert transcoder.layer_idx == 0


def test_lazy_loading(mock_safetensors_file):
    transcoder = load_relu_transcoder(
        mock_safetensors_file, layer=0, lazy_encoder=True, lazy_decoder=True
    )

    # W_enc and W_dec should trigger __getattr__
    assert transcoder.W_enc.shape == (64, 16)
    assert transcoder.W_dec.shape == (64, 16)
