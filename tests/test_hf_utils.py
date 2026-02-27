from mechinterp_qwen3.utils.hf_utils import HfUri, parse_hf_uri


def test_parse_hf_uri():
    # Implementation expects org/repo/file
    uri = "hf://org/repo/path/to/file?revision=main"
    parsed = parse_hf_uri(uri)
    assert isinstance(parsed, HfUri)
    assert parsed.repo_id == "org/repo"
    assert parsed.file_path == "path/to/file"
    assert parsed.revision == "main"

    # Try a simpler one that should work if implementation allows
    uri_simple = "hf://org/repo/file"
    parsed2 = parse_hf_uri(uri_simple)
    assert parsed2.repo_id == "org/repo"
    assert parsed2.file_path == "file"


def test_hf_uri_from_str():
    hf_ref = "org/repo/path@rev"
    parsed = HfUri.from_str(hf_ref)
    assert parsed.repo_id == "org/repo"
    assert parsed.file_path == "path"
    assert parsed.revision == "rev"
