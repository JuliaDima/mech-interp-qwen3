import argparse
import logging
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)


def get_project_root() -> Path:
    """Return the root directory of the project."""
    # This file is in src/mechinterp_qwen3/utils/config_utils.py
    return Path(__file__).resolve().parent.parent.parent.parent


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file, returning a dictionary."""
    if not path.exists():
        return {}
    with open(path) as f:
        try:
            return yaml.safe_load(f) or {}
        except Exception as e:
            log.warning(f"Failed to load config from {path}: {e}")
            return {}


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(
    config_path: str | None = None,
    use_root_default: bool = True,
) -> dict[str, Any]:
    """
    Load and merge configurations from multiple sources.

    Priority:
    1. Root config.yaml (if use_root_default is True)
    2. Explicitly provided config_path

    Returns:
        A dictionary containing the merged configuration.
    """
    config = {}

    # 1. Root config.yaml
    if use_root_default:
        root_config_path = get_project_root() / "config.yaml"
        if root_config_path.exists():
            config = load_yaml(root_config_path)

    # 2. Explicit config path
    if config_path:
        explicit_path = Path(config_path)
        if explicit_path.exists():
            explicit_config = load_yaml(explicit_path)
            config = merge_configs(config, explicit_config)
        else:
            log.warning(f"Config file not found: {config_path}")

    return config


def add_config_args(parser: argparse.ArgumentParser):
    """Add standard configuration arguments to an ArgumentParser."""
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML configuration file to override defaults.",
    )


def apply_config_to_args(
    args: argparse.Namespace, config: dict[str, Any], section: str | None = None
):
    """
    Apply configuration values to an argparse.Namespace.
    If a section is provided, it only applies values from that section.
    Does NOT override values that were explicitly set on the command line
    (this is tricky with argparse, usually we do it before parsing).
    """
    # This is a bit complex to do AFTER parsing because we don't know what was default.
    # Better approach is to set_defaults on the parser BEFORE parsing.
    pass


def set_parser_defaults_from_config(
    parser: argparse.ArgumentParser, config: dict[str, Any], section: str | None = None
):
    """Set defaults on an ArgumentParser from a configuration dictionary."""
    relevant_config = config
    if section and section in config:
        # Merge global defaults with section-specific ones
        # This allows a section to inherit from the top-level
        relevant_config = merge_configs(config, config[section])

    # Flatten the config for argparse (only one level deep for now, except for the section)
    # Most argparse args are top-level
    defaults = {}
    for k, v in relevant_config.items():
        if not isinstance(v, dict):
            defaults[k] = v

    parser.set_defaults(**defaults)
