from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from mechinterp_qwen3.probe import (
    CarryProbe,
    ProbeDataset,
    ProbeTrainer,
    binary_cross_entropy_loss,
    compute_carry_label,
    compute_metrics,
    generate_addition_examples,
)
from mechinterp_qwen3.probe.dataset_utils import pool_activations


def test_compute_carry_label():
    """Test carry detection logic."""
    # Basic cases
    assert compute_carry_label(5, 3) == 0  # 8
    assert compute_carry_label(5, 5) == 1  # 10
    assert compute_carry_label(9, 1) == 1  # 10
    assert compute_carry_label(0, 0) == 0  # 0

    # Multiple digits
    assert compute_carry_label(15, 13) == 0  # 28
    assert compute_carry_label(15, 27) == 1  # 42
    assert compute_carry_label(99, 1) == 1  # 100
    assert compute_carry_label(123, 456) == 0  # 579
    assert compute_carry_label(123, 876) == 0  # 999
    assert compute_carry_label(123, 877) == 1  # 1000

    # Different lengths
    assert compute_carry_label(5, 95) == 1  # 100
    assert compute_carry_label(95, 5) == 1  # 100
    assert compute_carry_label(1, 9) == 1  # 10


def test_generate_addition_examples():
    """Test dataset generation strategies."""
    # Grid strategy
    max_val = 9
    a, b, labels = generate_addition_examples(max_value=max_val, strategy="grid")
    assert len(a) == (max_val + 1) ** 2
    assert len(b) == len(a)
    assert len(labels) == len(a)
    for i in range(len(a)):
        assert labels[i] == compute_carry_label(a[i], b[i])

    # Balanced strategy
    n_samples = 20
    a, b, labels = generate_addition_examples(
        max_value=99, n_samples=n_samples, strategy="balanced"
    )
    assert len(a) == n_samples
    assert sum(labels) == n_samples // 2  # Exactly half carry

    # Random strategy
    a, b, labels = generate_addition_examples(max_value=99, n_samples=n_samples, strategy="random")
    assert len(a) == n_samples


def test_carry_probe_init():
    """Test CarryProbe initialization."""
    # Specific layers
    probe = CarryProbe(layers=[5, 10], d_transcoder=100)
    assert probe.layers == [5, 10]
    assert probe.d_transcoder == 100
    assert probe.n_layers == 2
    assert probe.linear.in_features == 200

    # Default layers
    probe = CarryProbe(n_layers=3, d_transcoder=50)
    assert probe.layers == [0, 1, 2]
    assert probe.linear.in_features == 150

    with pytest.raises(ValueError):
        CarryProbe(layers=None, n_layers=None)


def test_carry_probe_forward():
    """Test CarryProbe forward pass."""
    d_transcoder = 10
    layers = [0, 1]
    probe = CarryProbe(layers=layers, d_transcoder=d_transcoder)
    batch_size = 4

    # Input as dict
    activations_dict = {
        0: torch.randn(batch_size, d_transcoder),
        1: torch.randn(batch_size, d_transcoder),
    }
    output = probe(activations_dict)
    assert output.shape == (batch_size,)
    assert torch.all(output >= 0) and torch.all(output <= 1)

    # Input as tensor
    activations_tensor = torch.randn(batch_size, d_transcoder * 2)
    output = probe(activations_tensor)
    assert output.shape == (batch_size,)

    # Return logits
    logits = probe(activations_tensor, return_logits=True)
    assert logits.shape == (batch_size,)

    # Mismatched input error
    with pytest.raises(ValueError):
        probe(torch.randn(batch_size, d_transcoder))


def test_carry_probe_weights():
    """Test weight management in CarryProbe."""
    d_transcoder = 5
    layers = [10, 20]
    probe = CarryProbe(layers=layers, d_transcoder=d_transcoder)

    # Set known weights
    with torch.no_grad():
        probe.linear.weight.copy_(torch.arange(10).float().unsqueeze(0))

    # Get all weights
    all_w = probe.get_layer_weights()
    assert all_w.shape == (10,)
    assert torch.equal(all_w, torch.arange(10).float())

    # Get layer-specific weights
    w10 = probe.get_layer_weights(10)
    assert w10.shape == (5,)
    assert torch.equal(w10, torch.arange(5).float())

    w20 = probe.get_layer_weights(20)
    assert w20.shape == (5,)
    assert torch.equal(w20, torch.arange(5, 10).float())

    # Invalid layer
    with pytest.raises(ValueError):
        probe.get_layer_weights(5)


def test_carry_probe_serialization():
    """Test saving and loading CarryProbe with metadata."""
    probe = CarryProbe(layers=[1, 2, 3], d_transcoder=10)
    checkpoint = probe.state_dict_with_metadata()

    assert "state_dict" in checkpoint
    assert checkpoint["layers"] == [1, 2, 3]
    assert checkpoint["d_transcoder"] == 10

    new_probe = CarryProbe.from_state_dict_with_metadata(checkpoint)
    assert new_probe.layers == [1, 2, 3]
    assert new_probe.d_transcoder == 10

    # Check weights match
    for p1, p2 in zip(probe.parameters(), new_probe.parameters(), strict=False):
        assert torch.equal(p1, p2)


def test_compute_metrics():
    """Test evaluation metrics computation."""
    labels = torch.tensor([0, 1, 0, 1]).float()
    preds = torch.tensor([0, 1, 1, 0]).float()  # 2/4 correct
    probs = torch.tensor([0.1, 0.9, 0.8, 0.2]).float()

    metrics = compute_metrics(preds, labels, probabilities=probs)

    assert metrics.accuracy == 0.5
    assert metrics.n_samples == 4
    assert metrics.n_positive == 2
    assert metrics.n_negative == 2
    assert not np.isnan(metrics.roc_auc)
    assert not np.isnan(metrics.loss)
    assert metrics.loss > 0

    # Mismatched shapes
    with pytest.raises(ValueError):
        compute_metrics(preds, torch.tensor([0, 1]).float())


def test_bce_loss():
    """Test binary cross-entropy loss function."""
    probs = torch.tensor([0.9, 0.1]).float()
    labels = torch.tensor([1.0, 0.0]).float()

    loss_mean = binary_cross_entropy_loss(probs, labels, reduction="mean")
    assert loss_mean.dim() == 0
    assert loss_mean < 0.2  # Should be low for correct predictions

    loss_none = binary_cross_entropy_loss(probs, labels, reduction="none")
    assert loss_none.shape == (2,)

    with pytest.raises(ValueError):
        binary_cross_entropy_loss(probs, labels, reduction="invalid")


def test_pool_activations():
    """Test pooling strategies for activations."""
    batch, seq, dim = 2, 3, 4
    acts = torch.arange(batch * seq * dim).view(batch, seq, dim).float()

    # Final pooling
    pooled_final = pool_activations(acts, strategy="final")
    assert pooled_final.shape == (batch, dim)
    assert torch.equal(pooled_final, acts[:, -1, :])

    # Mean pooling
    pooled_mean = pool_activations(acts, strategy="mean")
    assert pooled_mean.shape == (batch, dim)
    assert torch.allclose(pooled_mean, acts.mean(dim=1))

    # Max pooling
    pooled_max = pool_activations(acts, strategy="max")
    assert pooled_max.shape == (batch, dim)
    assert torch.equal(pooled_max, acts.max(dim=1)[0])


def test_probe_dataset():
    """Test ProbeDataset batching and caching (mocked model)."""
    prompts = ["1+1=", "9+9="]
    labels = [0, 1]
    layers = [5]
    d_transcoder = 10

    # Mock model
    mock_model = MagicMock()
    mock_model.cfg.n_layers = 10
    mock_model.cfg.device = "cpu"

    # Mock tokenizer output so tokenize_qwen_input works
    mock_tokenizer_out = MagicMock()
    mock_tokenizer_out.input_ids = torch.tensor([[1, 2, 3]])
    mock_model.tokenizer.return_value = mock_tokenizer_out
    mock_model.tokenizer.all_special_ids = [0]
    mock_model.tokenizer.pad_token_id = 0
    mock_model.tokenizer.bos_token_id = None
    mock_model.tokenizer.eos_token_id = None
    mock_model.tokenizer.convert_ids_to_tokens.return_value = ["1", "+", "1", "="]

    # Mock get_activations
    mock_act = torch.randn(10, 3, d_transcoder)  # [n_layers, seq_len, d_transcoder]
    mock_logits = torch.randn(1, 3, 100)
    mock_model.get_activations.return_value = (mock_logits, mock_act)

    # Test dataset without caching
    dataset = ProbeDataset(prompts, labels, mock_model, layers=layers, token_position="final")
    assert len(dataset) == 2

    # Test get_batch
    batch_acts, batch_labels = dataset.get_batch([0, 1])
    assert 5 in batch_acts
    assert batch_acts[5].shape == (2, d_transcoder)
    assert torch.equal(batch_labels, torch.tensor([0, 1]).float())

    # Test with all tokens
    dataset_all = ProbeDataset(prompts, labels, mock_model, layers=layers, token_position="all")
    batch_acts_all, _ = dataset_all.get_batch([0, 1])
    assert batch_acts_all[5].ndim == 3  # [batch, seq_len, d_transcoder]


def test_probe_trainer_loss():
    """Test loss computation in ProbeTrainer including L1."""
    d_transcoder = 10
    probe = CarryProbe(layers=[0], d_transcoder=d_transcoder)
    trainer = ProbeTrainer(probe, l1_penalty=0.1)

    activations = {0: torch.randn(4, d_transcoder)}
    labels = torch.tensor([0, 1, 0, 1]).float()

    total_loss, base_loss, probs = trainer.compute_loss(activations, labels)

    # L1 should make total loss > base loss
    assert total_loss > base_loss
    assert probs.shape == (4,)


def test_probe_trainer_fit():
    """Test trainer fit loop with mock data."""
    d_transcoder = 4
    probe = CarryProbe(layers=[0], d_transcoder=d_transcoder)
    trainer = ProbeTrainer(probe)

    # Mock dataset
    mock_dataset = MagicMock(spec=ProbeDataset)
    mock_dataset.__len__.return_value = 16

    def get_batch_side_effect(indices):
        batch_size = len(indices)
        return {0: torch.randn(batch_size, d_transcoder)}, torch.randint(
            0, 2, (batch_size,)
        ).float()

    mock_dataset.get_batch.side_effect = get_batch_side_effect

    history = trainer.fit(mock_dataset, n_epochs=2, batch_size=4, verbose=False)

    assert len(history["train_loss"]) == 2
    assert len(history["train_metrics"]) == 2
    assert "accuracy" in history["train_metrics"][0]
