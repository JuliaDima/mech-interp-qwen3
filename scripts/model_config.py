"""Shared model/transcoder configuration for runnable scripts."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_MODEL_CONFIG = Path(__file__).with_name("model_config.yaml")
MODEL_CONFIG_ENV = "MIQ_MODEL_CONFIG"


@dataclass(frozen=True)
class ModelConfig:
    model: str
    transcoder_set: str


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _profile(config: dict) -> dict:
    active = config.get("active_profile")
    profiles = config.get("profiles", {})
    if active:
        if active not in profiles:
            raise ValueError(f"Unknown active_profile {active!r} in model config")
        return profiles[active] or {}
    return config


def load_model_config(path: str | Path | None = None) -> ModelConfig:
    config_path = Path(path or os.environ.get(MODEL_CONFIG_ENV, DEFAULT_MODEL_CONFIG))
    config = _load_yaml(config_path)
    profile = _profile(config)
    model = profile.get("model") or config.get("model")
    transcoder_set = profile.get("transcoder_set") or config.get("transcoder_set")
    if not model or not transcoder_set:
        raise ValueError(
            f"{config_path} must define model and transcoder_set, either at top level "
            "or in the active profile."
        )
    return ModelConfig(model=str(model), transcoder_set=str(transcoder_set))


def add_model_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model_config",
        default=None,
        help=f"YAML file with model/transcoder defaults (env: {MODEL_CONFIG_ENV})",
    )


def resolve_model_args(args: argparse.Namespace) -> ModelConfig:
    defaults = load_model_config(getattr(args, "model_config", None))
    if getattr(args, "model", None) is None:
        args.model = defaults.model
    if getattr(args, "transcoder_set", None) is None:
        args.transcoder_set = defaults.transcoder_set
    return ModelConfig(model=args.model, transcoder_set=args.transcoder_set)


def default_model() -> str:
    return load_model_config().model


def default_transcoder_set() -> str:
    return load_model_config().transcoder_set


def transcoder_snapshot_dir(transcoder_set: str | None = None) -> Path:
    """Return the local cached snapshot directory for a transcoder repo."""
    from huggingface_hub import snapshot_download

    repo_id = transcoder_set or default_transcoder_set()
    return Path(snapshot_download(repo_id=repo_id, local_files_only=True))
