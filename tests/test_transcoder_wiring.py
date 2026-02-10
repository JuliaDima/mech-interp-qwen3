import unittest
from unittest.mock import MagicMock, patch

import torch

from mechinterp_qwen3.forward_with_sae import (
    forward_with_sae_features_grad,
)


class TestTranscoderWiring(unittest.TestCase):
    def setUp(self):
        self.d_model = 4
        self.seq_len = 5
        self.layer_id = 0

        # Mock Model
        self.model = MagicMock()
        self.model.device = "cpu"
        self.model.model.layers = [MagicMock()]
        self.model.model.embed_tokens = MagicMock()
        self.model.model.norm = MagicMock()

        # Mock Tokenizer
        self.tokenizer = MagicMock()
        # Create a mock object that behaves like BatchEncoding
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs  # .to() returns itself
        mock_inputs.__getitem__.return_value = torch.tensor([[1, 2, 3]])  # input_ids
        # Dictionary access
        mock_inputs.__contains__.side_effect = lambda k: k == "input_ids"

        self.tokenizer.return_value = mock_inputs
        self.tokenizer.decode.return_value = "tok"
        self.tokenizer.all_special_ids = [0]
        self.tokenizer.bos_token_id = 0

        # Mock Transcoder
        self.transcoder = MagicMock()
        # Return tensor with gradients enabled for retain_grad()
        self.transcoder.encode.return_value = torch.randn(1, self.seq_len, 8, requires_grad=True)
        self.transcoders = {self.layer_id: self.transcoder}

    @patch("mechinterp_qwen3.forward_with_sae.MLPHookManager")
    def test_forward_with_sae_features_grad_uses_mlp_in(self, MockHookManager):
        # Setup Mock Hook Manager
        mock_hooker = MockHookManager.return_value

        # Distinct tensors for Input vs Output to verify wiring
        mlp_in_tensor = torch.full((self.seq_len, self.d_model), 1.0, requires_grad=True)
        mlp_out_tensor = torch.full((self.seq_len, self.d_model), 2.0, requires_grad=True)

        # Configure cache
        layer_acts = MagicMock()
        layer_acts.mlp_in = mlp_in_tensor
        layer_acts.mlp_out = mlp_out_tensor
        mock_hooker.cache = {self.layer_id: layer_acts}

        # Run function
        forward_with_sae_features_grad(
            self.model, self.tokenizer, self.transcoders, "test prompt", [self.layer_id]
        )

        # Verify transcoder.encode was called with mlp_in (tensor of 1s) not mlp_out (tensor of 2s)
        # Note: forward_with_sae unsqueezes to [1, seq, d_model]
        args, _ = self.transcoder.encode.call_args
        encoded_tensor = args[0]

        # Check values
        self.assertTrue(
            torch.allclose(encoded_tensor.squeeze(0), mlp_in_tensor),
            "Transcoder encoded mlp_out instead of mlp_in!",
        )
        self.assertFalse(
            torch.allclose(encoded_tensor.squeeze(0), mlp_out_tensor),
            "Transcoder encoded mlp_out! Regression detected.",
        )


if __name__ == "__main__":
    unittest.main()
