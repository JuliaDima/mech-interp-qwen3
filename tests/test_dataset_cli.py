import sys
from unittest.mock import patch

from mechinterp_qwen3.__main__ import main


def test_cli_generate_dataset(mocker):
    # Mock run_dataset_generation to avoid running the heavy logic
    mock_run = mocker.patch("mechinterp_qwen3.__main__.run_dataset_generation")

    test_args = [
        "miq",
        "generate-dataset",
        "--model",
        "test-model",
        "--output_path",
        "test.jsonl",
        "--templates",
        "T0",
        "T1",
    ]
    with patch.object(sys, "argv", test_args):
        main()

    assert mock_run.called
    args, _ = mock_run.call_args
    assert args[0].model == "test-model"
    assert args[0].output_path == "test.jsonl"
    assert args[0].templates == ["T0", "T1"]


def test_cli_visualize_dataset(mocker):
    # Mock run_dataset_visualization
    mock_run = mocker.patch("mechinterp_qwen3.__main__.run_dataset_visualization")

    test_args = [
        "miq",
        "visualize-dataset",
        "test.jsonl",
        "--output_dir",
        "viz_out",
        "--template",
        "T2",
    ]
    with patch.object(sys, "argv", test_args):
        main()

    assert mock_run.called
    args, _ = mock_run.call_args
    assert args[0].dataset_path == "test.jsonl"
    assert args[0].output_dir == "viz_out"
    assert args[0].template == "T2"
