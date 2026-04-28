import argparse
import logging
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)


def get_project_root() -> Path:
    """Return the project root (four directories above this file)."""
    return Path(__file__).resolve().parent.parent.parent.parent


def load_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file into a dict, returning an empty dict on missing file or parse error."""
    if not path.exists():
        return {}
    with open(path) as f:
        try:
            return yaml.safe_load(f) or {}
        except Exception as e:
            log.warning(f"Failed to load config from {path}: {e}")
            return {}


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge two dicts; override takes precedence on conflicts."""
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
    """Load and merge configuration from root config.yaml and an optional explicit path.

    Load order (later entries win):
    1. Root config.yaml (when use_root_default is True)
    2. Explicitly provided config_path

    Returns:
        Merged configuration dictionary.
    """
    config = {}

    if use_root_default:
        root_config_path = get_project_root() / "config.yaml"
        if root_config_path.exists():
            config = load_yaml(root_config_path)

    if config_path:
        explicit_path = Path(config_path)
        if explicit_path.exists():
            explicit_config = load_yaml(explicit_path)
            config = merge_configs(config, explicit_config)
        else:
            log.warning(f"Config file not found: {config_path}")

    return config


def add_config_args(parser: argparse.ArgumentParser):
    """Attach a --config argument for YAML overrides to the given parser."""
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML configuration file to override defaults.",
    )


def apply_config_to_args(
    args: argparse.Namespace, config: dict[str, Any], section: str | None = None
):
    """Apply config to an already-parsed Namespace (tricky post-parse; prefer set_defaults)."""
    pass


def set_parser_defaults_from_config(
    parser: argparse.ArgumentParser, config: dict[str, Any], section: str | None = None
):
    """Inject config values as parser defaults, optionally scoped to a named section."""
    relevant_config = config
    if section and section in config:
        # Section values override top-level defaults
        relevant_config = merge_configs(config, config[section])

    # Only flat (non-nested) keys map directly to argparse flags
    defaults = {}
    for k, v in relevant_config.items():
        if not isinstance(v, dict):
            defaults[k] = v

    parser.set_defaults(**defaults)


def print_config(args: argparse.Namespace, title: str = "Effective Run Configuration"):
    """Print argument namespace as a formatted key-value table."""
    print("=" * 60)
    print(f"{title}:")
    args_dict = vars(args)
    for key in sorted(args_dict.keys()):
        if key == "command":
            continue
        value = args_dict[key]
        print(f"  {key:<25} : {value}")
    print("=" * 60)
