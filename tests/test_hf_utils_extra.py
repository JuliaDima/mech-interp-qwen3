from mechinterp_qwen3.utils.hf_utils import HfUri, download_hf_uri, iter_transcoder_paths


def test_iter_transcoder_paths_with_list(mocker):
    # Test the branch where config has "transcoders" list
    config = {"transcoders": ["hf://user/repo/layer_0.pt", "hf://user/repo/layer_1.pt"]}

    mocker.patch(
        "mechinterp_qwen3.utils.hf_utils.download_hf_uri", side_effect=["/tmp/0.pt", "/tmp/1.pt"]
    )

    paths = list(iter_transcoder_paths(config))
    assert len(paths) == 2
    assert paths[0] == (0, "/tmp/0.pt")
    assert paths[1] == (1, "/tmp/1.pt")


def test_iter_transcoder_paths_snapshot(mocker):
    # Test the branch where it using snapshot_download
    config = {"repo_id": "user/repo", "subfolder": "sub"}

    mocker.patch("mechinterp_qwen3.utils.hf_utils.snapshot_download", return_value="/tmp/snapshot")
    # Mock glob to find files
    mocker.patch(
        "mechinterp_qwen3.utils.hf_utils.glob.glob",
        return_value=["/tmp/snapshot/sub/layer_0.safetensors"],
    )

    paths = list(iter_transcoder_paths(config))
    assert len(paths) == 1
    assert "layer_0.safetensors" in paths[0][1]


def test_download_hf_uri_success(mocker):
    mock_get = mocker.patch("mechinterp_qwen3.utils.hf_utils.hf_hub_download")
    mock_get.return_value = "/local/path"

    path = download_hf_uri("hf://user/repo/file.txt")
    assert path == "/local/path"
    mock_get.assert_called_once()


def test_hf_uri_parsing():
    uri = "hf://org/repo/path/to/file?revision=v1"
    info = HfUri.from_str(uri)
    assert info.repo_id == "org/repo"
    assert info.file_path == "path/to/file"
    assert info.revision == "v1"
