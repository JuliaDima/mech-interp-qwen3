import torch

from mechinterp_qwen3.transcoder.activation_functions import TopK, rectangle


def test_rectangle():
    # Test values
    x = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
    expected = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0])

    output = rectangle(x)

    assert torch.equal(output, expected)
    assert output.shape == x.shape
    assert output.dtype == x.dtype


def test_topk_basic():
    k = 2
    topk = TopK(k=k)
    x = torch.tensor([[1.0, 3.0, 2.0], [5.0, 4.0, 6.0]])

    # Expected: keep top 2 values, zero out the rest
    expected = torch.tensor([[0.0, 3.0, 2.0], [5.0, 0.0, 6.0]])

    output = topk(x)

    assert torch.equal(output, expected)
    assert output.shape == x.shape
    assert output.dtype == x.dtype


def test_topk_all_zero():
    k = 1
    topk = TopK(k=k)
    x = torch.zeros((2, 5))
    output = topk(x)
    assert torch.equal(output, x)
    assert output.shape == x.shape


def test_topk_negative_values():
    k = 1
    topk = TopK(k=k)
    x = torch.tensor([-5.0, -1.0, -10.0])
    # Top 1 is -1.0
    expected = torch.tensor([0.0, -1.0, 0.0])
    output = topk(x)
    assert torch.equal(output, expected)


def test_topk_batch_dim():
    k = 5
    topk = TopK(k=k)
    batch_size = 8
    d_model = 128
    x = torch.randn(batch_size, d_model)
    output = topk(x)

    assert output.shape == x.shape
    # Check that for each batch, exactly k elements are non-zero (or k largest kept)
    # Actually, if there are ties, topk might be tricky, but random randn is fine.
    for i in range(batch_size):
        non_zero_count = torch.count_nonzero(output[i]).item()
        assert non_zero_count == k
