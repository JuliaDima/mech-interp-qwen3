import os
import tempfile
from pathlib import Path

from mechinterp_qwen3.utils.hf_utils import empty_cache, get_cached_path, is_cached


def test_cached_path_and_is_cached(mocker):
    hf_ref = "test/repo"
    with tempfile.TemporaryDirectory() as tmpdir:
        # Patch get_cache_dir to return our temporary directory
        mocker.patch("mechinterp_qwen3.utils.hf_utils.get_cache_dir", return_value=Path(tmpdir))

        cached_path = get_cached_path(hf_ref)
        # For non-hf:// refs, it currently just appends the ref as subfolders
        assert str(Path(tmpdir) / "test" / "repo") == str(cached_path)

        assert not is_cached(hf_ref)

        os.makedirs(cached_path, exist_ok=True)
        with open(os.path.join(cached_path, "config.yaml"), "w") as f:
            f.write("test: data")

        assert is_cached(hf_ref)

        empty_cache(hf_ref)
        assert not is_cached(hf_ref)
        assert not os.path.exists(cached_path)
