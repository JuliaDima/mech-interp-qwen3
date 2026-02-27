import sys
from unittest.mock import MagicMock, patch

import pytest

from mechinterp_qwen3.__main__ import main


def test_cli_attribute_basic(mocker):
    # Mock run_attribution to avoid loading models/transcoders
    mock_run = mocker.patch("mechinterp_qwen3.__main__.run_attribution")

    # Simulate: miq attribute --prompt "test" --model "test-model" --transcoder_set "test-ts"
    test_args = [
        "miq",
        "attribute",
        "--prompt",
        "test",
        "--model",
        "test-model",
        "--transcoder_set",
        "test-ts",
        "--graph_output_path",
        "test.pt",
    ]
    with patch.object(sys, "argv", test_args):
        main()

    assert mock_run.called
    args, _ = mock_run.call_args
    assert args[0].prompt == "test"
    assert args[0].model == "test-model"


def test_cli_attribute_with_stats(mocker):
    # Here we might want to let run_attribution be called but mock its internals,
    # OR mock run_attribution and check if it handles stats_file.
    # Actually, stats_file is handled in run_attribution (the function in __main__.py).

    # Patch the actual implementation of attribute
    mock_attr_func = mocker.patch("mechinterp_qwen3.run_attribution.attribute")
    mock_graph = MagicMock()
    mock_attr_func.return_value = mock_graph

    mock_save_stats = mocker.patch("mechinterp_qwen3.utils.graph_viz.save_graph_stats")

    # Mock the loaders called inside run_attribution - patch the ORIGINAL location
    mocker.patch(
        "mechinterp_qwen3.utils.hf_utils.load_transcoder_from_hub",
        return_value=(MagicMock(), {}),
    )
    mocker.patch(
        "mechinterp_qwen3.attribution_model.AttributionModel.from_pretrained_and_transcoders",
        return_value=MagicMock(),
    )

    test_args = [
        "miq",
        "attribute",
        "--prompt",
        "test",
        "--model",
        "test-model",
        "--transcoder_set",
        "test-ts",
        "--graph_output_path",
        "test.pt",
        "--stats_file",
        "stats.txt",
    ]
    with patch.object(sys, "argv", test_args):
        main()

    assert mock_save_stats.called
    # Check that it was called with the returned graph and the stats path
    assert mock_save_stats.call_args[0][0] == mock_graph
    assert mock_save_stats.call_args[0][1] == "stats.txt"


def test_cli_help(mocker):
    # Just ensure --help works and exits with 0
    test_args = ["miq", "--help"]
    with patch.object(sys, "argv", test_args):
        with pytest.raises(SystemExit) as e:
            main()
        assert e.value.code == 0
