import os
import tempfile

import pytest
import torch

from mechinterp_qwen3.transcoder.cross_layer_transcoder import (
    CrossLayerTranscoder,
    load_clt,
)


def test_cross_layer_transcoder_basic():
    n_layers = 4
    d_transcoder = 64
    d_model = 16

    clt = CrossLayerTranscoder(
        n_layers=n_layers,
        d_transcoder=d_transcoder,
        d_model=d_model,
        activation_function="relu",
        dtype=torch.float32,
    )

    # clt.forward expects (n_layers, batch, d_model) based on encode implementation
    input_acts = torch.randn(n_layers, 2, d_model)
    output = clt(input_acts)
    assert output.shape == input_acts.shape


def test_cross_layer_transcoder_encode_layer():
    n_layers = 2
    d_transcoder = 32
    d_model = 8
    clt = CrossLayerTranscoder(n_layers, d_transcoder, d_model, dtype=torch.float32)

    x = torch.randn(5, d_model)
    # Encode layer 0
    acts = clt.encode_layer(x, layer_id=0)
    assert acts.shape == (5, d_transcoder)
    assert torch.all(acts >= 0)


@pytest.fixture
def mock_clt_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        n_layers = 2
        d_transcoder = 32
        d_model = 8

        for i in range(n_layers):
            weights = {
                f"W_enc_{i}": torch.randn(d_transcoder, d_model),
                f"W_dec_{i}": torch.randn(d_transcoder, d_model),
                f"b_enc_{i}": torch.randn(d_transcoder),
                f"b_dec_{i}": torch.randn(d_model),
            }
            from safetensors.torch import save_file

            save_file(weights, os.path.join(tmpdir, f"W_enc_{i}.safetensors"))
            # For simplicity, we put W_dec in same file for testing if expected by _load_state_dict
            # Actually _load_state_dict expects W_dec_{i}.safetensors for decoders if not lazy
            save_file(
                {f"W_dec_{i}": weights[f"W_dec_{i}"]},
                os.path.join(tmpdir, f"W_dec_{i}.safetensors"),
            )

        yield tmpdir


def test_load_clt(mock_clt_dir):
    clt = load_clt(
        clt_path=mock_clt_dir,
        feature_input_hook="hook_in",
        feature_output_hook="hook_out",
        lazy_decoder=True,
        lazy_encoder=False,
    )

    assert isinstance(clt, CrossLayerTranscoder)
    assert clt.n_layers == 2
    assert clt.d_transcoder == 32
    assert clt.d_model == 8
