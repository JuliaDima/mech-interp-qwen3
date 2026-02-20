"""Integration tests for end-to-end workflows."""

from __future__ import annotations

import torch

from mechinterp_qwen3.build_prompts import generate_examples
from mechinterp_qwen3.io import read_jsonl, write_jsonl
from mechinterp_qwen3.utils_seed import SeedConfig, set_all_seeds


class TestPromptGenerationWorkflow:
    """Test end-to-end prompt generation workflow."""

    def test_generate_and_save_prompts(self, temp_dir):
        """Test generating prompts and saving to JSONL."""
        # Generate examples
        examples = generate_examples(n=10, seed=42, low=0, high=999)

        # Convert to JSONL format
        rows = []
        for ex in examples:
            rows.append(
                {
                    "prompt_id": ex.prompt_id,
                    "behaviour": "greater_than",
                    "a": ex.a,
                    "b": ex.b,
                    "prompt": ex.prompt,
                    "expected": ex.expected,
                }
            )

        # Save to file
        output_path = temp_dir / "prompts.jsonl"
        write_jsonl(output_path, rows)

        # Read back and verify
        loaded = read_jsonl(output_path)
        assert len(loaded) == 10
        assert loaded == rows

    def test_deterministic_prompt_generation(self, temp_dir):
        """Test that prompt generation is deterministic across runs."""
        seed = 123
        n = 20

        # First run
        examples1 = generate_examples(n=n, seed=seed, low=0, high=999)
        path1 = temp_dir / "run1.jsonl"
        rows1 = [{"a": ex.a, "b": ex.b, "expected": ex.expected} for ex in examples1]
        write_jsonl(path1, rows1)

        # Second run
        examples2 = generate_examples(n=n, seed=seed, low=0, high=999)
        path2 = temp_dir / "run2.jsonl"
        rows2 = [{"a": ex.a, "b": ex.b, "expected": ex.expected} for ex in examples2]
        write_jsonl(path2, rows2)

        # Should be identical
        assert read_jsonl(path1) == read_jsonl(path2)


class TestActivationCaptureWorkflow:
    """Test end-to-end activation capture workflow."""

    def test_capture_activations_for_prompts(self, mock_model, temp_dir):
        """Test capturing activations for a set of prompts."""
        layer_ids = [0, 1, 2]

        # Generate some prompts
        examples = generate_examples(n=3, seed=0, low=0, high=999)

        activations_data = []

        for ex in examples:
            seq_len = len(ex.prompt) // 10 + 5  # Mock sequence length
            input_ids = torch.randint(0, 100, (1, seq_len))

            # Capture mlp outputs via simple hooks
            captured: dict[int, torch.Tensor] = {}
            hooks = []
            for lid in layer_ids:

                def make_hook(layer_id, cache):
                    def hook_fn(module, inp, activation):
                        cache[layer_id] = activation.detach()

                    return hook_fn

                h = mock_model.model.layers[lid].mlp.register_forward_hook(make_hook(lid, captured))
                hooks.append(h)

            with torch.no_grad():
                mock_model(input_ids)

            for h in hooks:
                h.remove()

            acts_record = {
                "prompt_id": ex.prompt_id,
                "seq_len": seq_len,
                "layers": {},
            }
            for lid in layer_ids:
                out = captured[lid]
                acts_record["layers"][lid] = {
                    "mlp_in_shape": list(out.shape),
                    "mlp_out_shape": list(out.shape),
                }

            activations_data.append(acts_record)

        # Verify we captured activations for all prompts
        assert len(activations_data) == 3

        for record in activations_data:
            assert "prompt_id" in record
            assert "seq_len" in record
            assert len(record["layers"]) == len(layer_ids)

            for lid in layer_ids:
                assert record["layers"][lid]["mlp_in_shape"][-1] == mock_model.d_model
                assert record["layers"][lid]["mlp_out_shape"][-1] == mock_model.d_model

    def test_activation_capture_with_different_sequence_lengths(self, mock_model):
        """Test that activation capture handles variable sequence lengths."""
        seq_lengths = [5, 10, 15, 20]

        for seq_len in seq_lengths:
            input_ids = torch.randint(0, 100, (1, seq_len))

            captured: list[torch.Tensor] = []

            def make_seq_hook(cache: list) -> torch.nn.modules.module._Hook:
                def hook(module, inp, out):
                    cache.append(out.detach())

                return hook

            h = mock_model.model.layers[0].mlp.register_forward_hook(make_seq_hook(captured))

            with torch.no_grad():
                mock_model(input_ids)

            h.remove()

            assert captured[0].shape[1] == seq_len


class TestModelArchitectureCompliance:
    """Test that model architecture meets expected structure."""

    def test_model_has_layers_attribute(self, mock_model):
        """Test that model has model.layers structure."""
        assert hasattr(mock_model, "model")
        assert hasattr(mock_model.model, "layers")

    def test_layers_have_mlp_modules(self, mock_model):
        """Test that each layer has an MLP module."""
        layers = list(mock_model.model.layers)

        for i, layer in enumerate(layers):
            assert hasattr(layer, "mlp"), f"Layer {i} missing MLP module"

    def test_layer_count(self, mock_model):
        """Test that model has expected number of layers."""
        layers = list(mock_model.model.layers)
        assert len(layers) == mock_model.n_layers

    def test_mlp_forward_pass(self, mock_model):
        """Test that MLP modules can perform forward pass."""
        layers = list(mock_model.model.layers)

        # Test first layer's MLP
        mlp = layers[0].mlp
        test_input = torch.randn(1, 10, mock_model.d_model)

        with torch.no_grad():
            output = mlp(test_input)

        assert output.shape == test_input.shape


class TestEndToEndDeterminism:
    """Test end-to-end determinism of the pipeline."""

    def test_full_pipeline_determinism(self, mock_model, temp_dir):
        """Test that entire pipeline is deterministic with same seed."""
        seed = 42

        def run_pipeline():
            set_all_seeds(SeedConfig(seed=seed))

            # Generate prompts
            examples = generate_examples(n=5, seed=seed, low=0, high=999)

            results = []
            for ex in examples:
                seq_len = 10
                input_ids = torch.randint(0, 100, (1, seq_len))

                # Capture MLP output via simple hook
                captured_out: list[torch.Tensor] = []

                def make_det_hook(cache: list) -> torch.nn.modules.module._Hook:
                    def hook_fn(module, inp, out):
                        cache.append(out.detach())

                    return hook_fn

                h = mock_model.model.layers[0].mlp.register_forward_hook(
                    make_det_hook(captured_out)
                )

                with torch.no_grad():
                    mock_model(input_ids)

                h.remove()

                out_val = captured_out[0]

                # Store activation checksums
                result = {
                    "prompt_id": ex.prompt_id,
                    "a": ex.a,
                    "b": ex.b,
                    "mlp_in_sum": out_val.sum().item(),
                }
                results.append(result)

            return results

        # Run pipeline twice
        results1 = run_pipeline()
        results2 = run_pipeline()

        # Compare results
        assert len(results1) == len(results2)
        for r1, r2 in zip(results1, results2, strict=False):
            assert r1["prompt_id"] == r2["prompt_id"]
            assert r1["a"] == r2["a"]
            assert r1["b"] == r2["b"]
            # Activations should be identical due to determinism
            assert abs(r1["mlp_in_sum"] - r2["mlp_in_sum"]) < 1e-5
