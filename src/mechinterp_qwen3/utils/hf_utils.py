from __future__ import annotations

import glob
import os
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple
from urllib.parse import parse_qs, urlparse

import torch
import yaml
from huggingface_hub import get_token, hf_api, hf_hub_download, snapshot_download
try:
    from huggingface_hub.constants import HF_HUB_ENABLE_HF_TRANSFER
except ImportError:
    import os as _os
    HF_HUB_ENABLE_HF_TRANSFER = _os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "0") == "1"
from huggingface_hub.utils.tqdm import tqdm as hf_tqdm
from tqdm.contrib.concurrent import thread_map


class HfUri(NamedTuple):
    """Parsed components of a HuggingFace repository reference."""

    repo_id: str
    file_path: str | None
    revision: str | None

    @classmethod
    def from_str(cls, hf_ref: str):
        if hf_ref.startswith("hf://"):
            return parse_hf_uri(hf_ref)

        parts = hf_ref.split("@", 1)
        path_part = parts[0]
        revision = parts[1] if len(parts) > 1 else None

        path_components = path_part.split("/")
        if len(path_components) >= 2:
            repo_id = "/".join(path_components[:2])
            file_path = "/".join(path_components[2:]) if len(path_components) > 2 else None
        else:
            repo_id = path_part
            file_path = None

        return cls(repo_id, file_path, revision)


def load_transcoder_from_hub(
    hf_ref: str,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    lazy_encoder: bool = False,
    lazy_decoder: bool = True,
):
    """Download and instantiate a transcoder set or CLT from the HuggingFace hub.

    Checks the local cache first and falls back to downloading if not found.
    """
    if is_cached(hf_ref):
        return load_transcoders_from_cache(
            hf_ref,
            device=device,
            dtype=dtype,
            lazy_encoder=lazy_encoder,
            lazy_decoder=lazy_decoder,
        )

    hf_uri = HfUri.from_str(hf_ref)

    # Fetch the config to determine transcoder kind and file layout
    config_path = hf_hub_download(
        repo_id=hf_uri.repo_id,
        filename="config.yaml",
        revision=hf_uri.revision,
        subfolder=hf_uri.file_path,
    )

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Annotate config with provenance metadata
    config["repo_id"] = hf_uri.repo_id
    config["revision"] = hf_uri.revision
    config["subfolder"] = hf_uri.file_path
    repo_info = (
        hf_uri.repo_id if hf_uri.file_path is None else hf_uri.repo_id + "//" + hf_uri.file_path
    )
    config["scan"] = f"{repo_info}@{hf_uri.revision}" if hf_uri.revision else repo_info

    model_kind = config["model_kind"]

    from ..transcoder.cross_layer_transcoder import load_clt
    from ..transcoder.single_layer_transcoder import load_transcoder_set

    if model_kind == "transcoder_set":
        transcoder_paths = {}
        for layer_idx, local_path in iter_transcoder_paths(config):
            transcoder_paths[layer_idx] = local_path

        transcoder = load_transcoder_set(
            transcoder_paths,
            scan=config["scan"],
            feature_input_hook=config["feature_input_hook"],
            feature_output_hook=config["feature_output_hook"],
            device=device,
            dtype=dtype,
            lazy_encoder=lazy_encoder,
            lazy_decoder=lazy_decoder,
        )

    elif model_kind == "cross_layer_transcoder":
        # CLT requires the full folder of safetensors shards
        subfolder = config.get("subfolder")
        allow_patterns = [f"{subfolder}/*.safetensors"] if subfolder else ["*.safetensors"]

        local_path = snapshot_download(
            config["repo_id"],
            revision=config.get("revision", "main"),
            allow_patterns=allow_patterns,
        )

        if subfolder:
            local_path = os.path.join(local_path, subfolder)

        transcoder = load_clt(
            local_path,
            feature_input_hook=config["feature_input_hook"],
            feature_output_hook=config["feature_output_hook"],
            scan=config["scan"],
            device=device,
            dtype=dtype,
            lazy_decoder=lazy_decoder,
            lazy_encoder=lazy_encoder,
        )
    else:
        raise ValueError(f"Unknown model_kind: {model_kind}")

    return transcoder, config


def get_cache_dir() -> Path:
    """Return the root directory used for caching downloaded transcoders."""
    return Path.home() / ".cache" / "mechinterp_qwen3"


def _normalize_hf_ref(hf_ref: str) -> str:
    """Convert an hf_ref string to a filesystem-safe path component."""
    if hf_ref.startswith("hf://"):
        uri = parse_hf_uri(hf_ref)
        normalized = uri.repo_id
        if uri.file_path:
            normalized = f"{normalized}/{uri.file_path}"
        if uri.revision:
            normalized = f"{normalized}@{uri.revision}"
        return normalized

    return hf_ref


def get_cached_path(hf_ref: str) -> Path:
    """Return the local cache directory for a given hf_ref."""
    cache_base = get_cache_dir()
    normalized = _normalize_hf_ref(hf_ref)
    return cache_base / normalized


def is_cached(hf_ref: str) -> bool:
    """Return True if a complete local cache exists for the given hf_ref."""
    cache_path = get_cached_path(hf_ref)
    config_path = cache_path / "config.yaml"
    return config_path.exists()


def empty_cache(hf_ref: str | None = None):
    """Remove cached transcoders — a specific ref or the entire cache if None."""
    cache_base = get_cache_dir()
    if hf_ref is not None:
        cache_path = get_cached_path(hf_ref)
        if cache_path.exists():
            shutil.rmtree(cache_path)
            print(f"removed cache for {hf_ref!r} at {cache_path}")
    else:
        if cache_base.exists():
            shutil.rmtree(cache_base)
            print(f"removed full cache at {cache_base}")


def _delete_hf_cache(path: str | Path):
    """Remove a HuggingFace cache entry, following symlinks to delete the blob."""
    path = Path(path)
    if path.is_symlink():
        blob_path = path.resolve()
        path.unlink()
        if blob_path.exists():
            blob_path.unlink()
    elif path.exists():
        path.unlink()


def save_transcoders_to_cache(
    hf_ref: str,
    sequential: bool = True,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    delete_hf_cache: bool = True,
) -> Path:
    """Download transcoders and persist them in the local cache directory."""
    if device is None:
        device = torch.device("cpu")

    hf_uri = HfUri.from_str(hf_ref)

    config_path = hf_hub_download(
        repo_id=hf_uri.repo_id,
        revision=hf_uri.revision,
        filename="config.yaml",
        subfolder=hf_uri.file_path,
    )

    with open(config_path) as f:
        config = yaml.safe_load(f)

    config["repo_id"] = hf_uri.repo_id
    config["revision"] = hf_uri.revision
    config["subfolder"] = hf_uri.file_path
    repo_info = (
        hf_uri.repo_id if hf_uri.file_path is None else hf_uri.repo_id + "//" + hf_uri.file_path
    )
    config["scan"] = f"{repo_info}@{hf_uri.revision}" if hf_uri.revision else repo_info

    model_kind = config["model_kind"]
    cache_path = get_cached_path(hf_ref)
    cache_path.mkdir(parents=True, exist_ok=True)

    if model_kind == "transcoder_set":
        _save_transcoder_set_to_cache(
            config=config,
            cache_path=cache_path,
            sequential=sequential,
            device=device,
            dtype=dtype,
            delete_hf_cache=delete_hf_cache,
        )
    elif model_kind == "cross_layer_transcoder":
        _save_clt_to_cache(
            config,
            cache_path,
            sequential,
            device,
            dtype,
            delete_hf_cache,
        )
    else:
        raise ValueError(f"Unknown model kind: {model_kind}")

    simplified_config = {
        k: v for k, v in config.items() if k not in ("transcoders", "repo_id", "subfolder")
    }
    with open(cache_path / "config.yaml", "w") as f:
        yaml.dump(simplified_config, f)

    print(f"transcoders cached to {cache_path}")
    return cache_path


def _save_transcoder_set_to_cache(
    config: dict,
    cache_path: Path,
    sequential: bool,
    device: torch.device,
    dtype: torch.dtype,
    delete_hf_cache: bool,
):
    from ..transcoder.single_layer_transcoder import load_relu_transcoder

    if "transcoders" not in config:
        for layer_idx, local_path in iter_transcoder_paths(config):
            transcoder = load_relu_transcoder(
                local_path,
                layer_idx,
                device=device,
                dtype=dtype,
                lazy_encoder=False,
                lazy_decoder=False,
            )
            save_path = cache_path / f"layer_{layer_idx}.safetensors"
            transcoder.to_safetensors(str(save_path))
            print(f"  layer {layer_idx} → {save_path}")
        return

    if sequential:
        for layer_idx, hf_path in enumerate(config["transcoders"]):
            local_path = download_hf_uri(hf_path) if hf_path.startswith("hf://") else hf_path
            transcoder = load_relu_transcoder(
                local_path,
                layer_idx,
                device=device,
                dtype=dtype,
                lazy_encoder=False,
                lazy_decoder=False,
            )
            save_path = cache_path / f"layer_{layer_idx}.safetensors"
            transcoder.to_safetensors(str(save_path))
            print(f"  layer {layer_idx} → {save_path}")
            if delete_hf_cache and hf_path.startswith("hf://"):
                _delete_hf_cache(local_path)
    else:
        hf_paths = [p for p in config["transcoders"] if p.startswith("hf://")]
        local_map = download_hf_uris(hf_paths)
        transcoder_paths: dict[int, str] = {}
        for i, path in enumerate(config["transcoders"]):
            transcoder_paths[i] = local_map.get(path) or path

        for layer_idx, local_path in transcoder_paths.items():
            transcoder = load_relu_transcoder(
                local_path,
                layer_idx,
                device=device,
                dtype=dtype,
                lazy_encoder=False,
                lazy_decoder=False,
            )
            save_path = cache_path / f"layer_{layer_idx}.safetensors"
            transcoder.to_safetensors(str(save_path))
            print(f"  layer {layer_idx} → {save_path}")

        if delete_hf_cache:
            for i, hf_path in enumerate(config["transcoders"]):
                if hf_path.startswith("hf://"):
                    _delete_hf_cache(transcoder_paths[i])


def _save_clt_to_cache(
    config: dict,
    cache_path: Path,
    sequential: bool,
    device: torch.device,
    dtype: torch.dtype,
    delete_hf_cache: bool,
):
    from ..transcoder.cross_layer_transcoder import load_clt

    subfolder = config.get("subfolder")
    allow_patterns = [f"{subfolder}/*.safetensors"] if subfolder else ["*.safetensors"]

    local_path = snapshot_download(
        config["repo_id"],
        revision=config.get("revision", "main"),
        allow_patterns=allow_patterns,
    )

    if subfolder:
        local_path = os.path.join(local_path, subfolder)

    clt = load_clt(
        local_path,
        feature_input_hook=config["feature_input_hook"],
        feature_output_hook=config["feature_output_hook"],
        scan=config.get("scan"),
        device=device,
        dtype=dtype,
        lazy_decoder=False,
        lazy_encoder=False,
    )

    clt.to_safetensors(str(cache_path))


def load_transcoders_from_cache(
    hf_ref: str,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    lazy_encoder: bool = False,
    lazy_decoder: bool = True,
):
    cache_path = get_cached_path(hf_ref)
    config_path = cache_path / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Cache not found for {hf_ref} at {cache_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_kind = config["model_kind"]

    from ..transcoder.cross_layer_transcoder import load_clt
    from ..transcoder.single_layer_transcoder import load_transcoder_set

    if model_kind == "transcoder_set":
        layer_files = sorted(cache_path.glob("layer_*.safetensors"))
        transcoder_paths = {int(f.stem.split("_")[1]): str(f) for f in layer_files}

        transcoder = load_transcoder_set(
            transcoder_paths,
            scan=config.get("scan", str(cache_path)),
            feature_input_hook=config["feature_input_hook"],
            feature_output_hook=config["feature_output_hook"],
            device=device,
            dtype=dtype,
            lazy_encoder=lazy_encoder,
            lazy_decoder=lazy_decoder,
        )
    elif model_kind == "cross_layer_transcoder":
        transcoder = load_clt(
            str(cache_path),
            feature_input_hook=config["feature_input_hook"],
            feature_output_hook=config["feature_output_hook"],
            scan=config.get("scan", str(cache_path)),
            device=device,
            dtype=dtype,
            lazy_decoder=lazy_decoder,
            lazy_encoder=lazy_encoder,
        )
    else:
        raise ValueError(f"Unknown model kind: {model_kind}")

    return transcoder, config


def iter_transcoder_paths(config: dict) -> Iterable[tuple[int, str]]:
    """Yield (layer_index, local_path) pairs, downloading each file on demand."""
    if "transcoders" in config:
        for i, path in enumerate(config["transcoders"]):
            local_path = download_hf_uri(path) if path.startswith("hf://") else path
            yield i, local_path
    else:
        subfolder = config.get("subfolder")
        if subfolder:
            allow_patterns = [f"{subfolder}/layer_*.safetensors"]
        else:
            allow_patterns = ["layer_*.safetensors"]

        local_path = snapshot_download(
            config["repo_id"],
            revision=config.get("revision", "main"),
            allow_patterns=allow_patterns,
        )

        if subfolder:
            local_path = os.path.join(local_path, subfolder)

        layer_files = glob.glob(os.path.join(local_path, "layer_*.safetensors"))
        for i in range(len(layer_files)):
            yield i, os.path.join(local_path, f"layer_{i}.safetensors")


def parse_hf_uri(uri: str) -> HfUri:
    """Decompose an hf:// URI into repo id, file path, and revision."""
    parsed = urlparse(uri)
    if parsed.scheme != "hf":
        raise ValueError(f"Not a huggingface URI: {uri}")
    path = parsed.path.lstrip("/")
    repo_parts = path.split("/", 1)
    if len(repo_parts) != 2:
        raise ValueError(f"Invalid huggingface URI: {uri}")
    repo_id = f"{parsed.netloc}/{repo_parts[0]}"
    file_path = repo_parts[1]
    revision = parse_qs(parsed.query).get("revision", [None])[0] or None
    return HfUri(repo_id, file_path, revision)


def download_hf_uri(uri: str) -> str:
    """Resolve an hf:// URI to a local file path, downloading if necessary."""
    parsed = parse_hf_uri(uri)
    assert parsed.file_path is not None, "File path is not set"
    return hf_hub_download(
        repo_id=parsed.repo_id,
        filename=parsed.file_path,
        revision=parsed.revision,
        force_download=False,
    )


def download_hf_uris(uris: Iterable[str], max_workers: int = 16) -> dict[str, str]:
    """Download a batch of hf:// URIs concurrently, with an auth pre-check."""
    if not uris:
        return {}

    uri_list = list(uris)
    if not uri_list:
        return {}
    parsed_map = {uri: parse_hf_uri(uri) for uri in uri_list}

    print(f"checking access for {len({info.repo_id for info in parsed_map.values()})} repo(s)...")
    unique_repos = {info.repo_id for info in parsed_map.values()}
    token = get_token()

    for repo_id in unique_repos:
        if hf_api.repo_info(repo_id=repo_id, token=token).gated is not False and token is None:
            raise PermissionError("Cannot access a gated repo without a hf token.")

    print("access confirmed, starting downloads...")

    def _download(uri: str) -> str:
        info = parsed_map[uri]
        assert info.file_path is not None, "File path is not set"

        return hf_hub_download(
            repo_id=info.repo_id,
            filename=info.file_path,
            revision=info.revision,
            token=token,
            force_download=False,
        )

    if HF_HUB_ENABLE_HF_TRANSFER:
        results = [_download(uri) for uri in uri_list]
        return dict(zip(uri_list, results, strict=False))

    results = thread_map(
        _download,
        uri_list,
        desc=f"Fetching {len(parsed_map)} files",
        max_workers=max_workers,
        tqdm_class=hf_tqdm,
    )
    return dict(zip(uri_list, results, strict=False))
