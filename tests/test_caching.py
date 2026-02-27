import os
import tempfile
from pathlib import Path

from mechinterp_qwen3.utils.hf_utils import empty_cache, get_cache_dir, get_cached_path, is_cached


def test_cache_dir_logic():
    default_dir = get_cache_dir()
    assert ".cache/mechinterp_qwen3" in str(default_dir)

    with tempfile.TemporaryDirectory() as tmpdir:
        res = get_cache_dir(tmpdir)
        assert str(res) == tmpdir


def test_cached_path_and_is_cached():
    hf_ref = "test/repo"
    with tempfile.TemporaryDirectory() as tmpdir:
        cached_path = get_cached_path(hf_ref, cache_dir=tmpdir)
        # For non-hf:// refs, it currently just appends the ref as subfolders
        assert str(Path(tmpdir) / "test" / "repo") == str(cached_path)

        assert not is_cached(hf_ref, cache_dir=tmpdir)

        os.makedirs(cached_path, exist_ok=True)
        with open(os.path.join(cached_path, "config.yaml"), "w") as f:
            f.write("test: data")

        assert is_cached(hf_ref, cache_dir=tmpdir)

        empty_cache(hf_ref, cache_dir=tmpdir)
        assert not is_cached(hf_ref, cache_dir=tmpdir)
        assert not os.path.exists(cached_path)
