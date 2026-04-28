import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

from mechinterp_qwen3.utils.config_utils import (  # noqa: E402
    add_config_args,
    load_config,
    print_config,
    set_parser_defaults_from_config,
)

log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="CLI for attribution and graph file creation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.required = True

    # Attach --config to the root parser so it can be pre-parsed before subcommand dispatch
    add_config_args(parser)

    # Dataset generation subcommand
    gen_parser = subparsers.add_parser(
        "generate-dataset", help="Generate addition dataset with model stats"
    )
    # Model configuration
    gen_parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-4B",
        help="HuggingFace model name (default: Qwen/Qwen3-4B)",
    )
    gen_parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). If not specified, uses CUDA if available.",
    )
    gen_parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype (default: float32)",
    )

    # Output configuration
    gen_parser.add_argument(
        "--output_path",
        type=str,
        help="Path to output JSONL file (required if not in config)",
    )

    # Template configuration
    gen_parser.add_argument(
        "--templates",
        nargs="+",
        type=str,
        default=["T0"],
        choices=["T0", "T1", "T2"],
        help="Templates to use (default: T0). Can specify multiple.",
    )

    # Sampling configuration
    gen_parser.add_argument(
        "--sampling_strategy",
        type=str,
        default="grid",
        choices=["grid", "stratified", "random"],
        help="Sampling strategy for (a,b) pairs (default: grid)",
    )
    gen_parser.add_argument(
        "--max_value",
        type=int,
        default=20,
        help="Maximum value for a and b (default: 20)",
    )
    gen_parser.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="Number of samples (only for random strategy)",
    )
    gen_parser.add_argument(
        "--stratified_n_per_category",
        type=int,
        default=100,
        help="Number of samples per carry category for stratified sampling (default: 100)",
    )
    gen_parser.add_argument(
        "--stratified_uniform_remainder",
        type=int,
        default=100,
        help="Number of uniform random samples for stratified sampling (default: 100)",
    )

    # Statistics configuration
    gen_parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of top-k tokens to store per position (default: 10)",
    )

    # Generation configuration
    gen_parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for generation (default: 32)",
    )
    gen_parser.add_argument(
        "--max_gen_tokens",
        type=int,
        default=10,
        help="Maximum tokens to generate in greedy mode (default: 10)",
    )

    # Reproducibility
    gen_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    # Visualization subcommand
    viz_parser = subparsers.add_parser("visualize-dataset", help="Visualize addition dataset")
    viz_parser.add_argument("dataset_path", help="Path to JSONL dataset")
    viz_parser.add_argument(
        "--output_dir",
        default="visualizations",
        help="Output directory for plots (default: visualizations)",
    )
    viz_parser.add_argument("--template", default="T0", help="Template to visualize (default: T0)")

    # Attribution subcommand
    attr_parser = subparsers.add_parser("attribute", help="Run attribution analysis on a prompt")

    # Arguments from attribute_batch.py
    attr_parser.add_argument(
        "-m",
        "--model",
        type=str,
        help=("Model architecture to use for attribution. Can be inferred from transcoder config."),
    )
    attr_parser.add_argument(
        "-t",
        "--transcoder_set",
        help=(
            "HuggingFace repository ID containing transcoders "
            "(e.g. username/repo-name, username/repo-name@revision). "
            "Required if not in config."
        ),
    )
    attr_parser.add_argument(
        "-p", "--prompt", help="Input prompt text to analyze. Required if not in config."
    )
    attr_parser.add_argument(
        "-o",
        "--graph_output_path",
        help=(
            "Path where to save the attribution graph (.pt file). Required if not "
            "creating graph files."
        ),
    )
    attr_parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "bfloat16", "float16", "fp32", "bf16", "fp16"],
        default="float32",
        help="Data type for model weights (default: float32).",
    )
    attr_parser.add_argument(
        "--max_n_logits", type=int, default=10, help="Maximum number of logit nodes."
    )
    attr_parser.add_argument(
        "--desired_logit_prob",
        type=float,
        default=0.95,
        help="Cumulative probability threshold for top logits.",
    )
    attr_parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size for backward passes."
    )
    attr_parser.add_argument(
        "--offload",
        choices=["cpu", "disk", None],
        default=None,
        help="Offload model parameters to save memory.",
    )
    attr_parser.add_argument(
        "--max_feature_nodes",
        type=int,
        default=7500,
        help="Maximum number of feature nodes.",
    )
    attr_parser.add_argument("--verbose", action="store_true", help="Display progress information.")
    attr_parser.add_argument(
        "--lazy-encoder",
        action="store_true",
        help="Enable lazy loading for encoder weights to save memory.",
    )
    attr_parser.add_argument(
        "--lazy-decoder",
        action="store_true",
        default=True,
        help="Enable lazy loading for decoder weights to save memory (default: True).",
    )

    # Arguments for graph creation
    attr_parser.add_argument(
        "--slug",
        type=str,
        help=(
            "Slug for the model metadata (used for graph files). Required if creating graph files."
        ),
    )
    attr_parser.add_argument(
        "--graph_file_dir",
        type=str,
        help=("Path to save the output JSON graph files. Required if creating graph files."),
    )
    attr_parser.add_argument(
        "--node_threshold",
        type=float,
        default=0.8,
        help="Node threshold for pruning graph files.",
    )
    attr_parser.add_argument(
        "--edge_threshold",
        type=float,
        default=0.98,
        help="Edge threshold for pruning graph files.",
    )
    attr_parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    attr_parser.add_argument(
        "--nondeterministic",
        action="store_true",
        help="Disable deterministic algorithms (may improve performance).",
    )
    attr_parser.add_argument(
        "--stats_file",
        type=str,
        help="Path to save graph statistics (nodes, edges, layers, etc.).",
    )

    # Intervention subcommand
    int_parser = subparsers.add_parser(
        "intervene",
        help="Run constrained patching on a saved attribution graph (addition experiment)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    int_parser.add_argument(
        "-g",
        "--graph_path",
        help="Path to a .pt graph file saved by `miq attribute`. Required if not in config.",
    )
    int_parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=None,
        help="HuggingFace model name (inferred from transcoder config if omitted).",
    )
    int_parser.add_argument(
        "-t",
        "--transcoder_set",
        help="HuggingFace repo id for the transcoder set. Required if not in config.",
    )
    int_parser.add_argument(
        "-o",
        "--out_dir",
        default="runs/addition/interventions",
        help="Output directory for intervention results.",
    )
    int_parser.add_argument(
        "-p",
        "--prompt",
        default=None,
        help="Clean prompt (default: FOCUS_PROMPT from addition experiment).",
    )
    int_parser.add_argument(
        "--perturbed_prompt",
        default=None,
        help="Perturbed prompt for constrained patching (default: calc: 36+60= ).",
    )
    int_parser.add_argument(
        "--node_threshold",
        type=float,
        default=0.8,
        help="Node pruning threshold.",
    )
    int_parser.add_argument(
        "--edge_threshold",
        type=float,
        default=0.98,
        help="Edge pruning threshold.",
    )
    int_parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="Feature scale factor (0.0 = full inhibition).",
    )
    int_parser.add_argument(
        "--top_n_groups",
        type=int,
        default=4,
        help="Number of supernode groups to test.",
    )
    int_parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "bfloat16", "float16"],
        default="float32",
        help="Model dtype.",
    )
    int_parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    int_parser.add_argument(
        "--nondeterministic",
        action="store_true",
        help="Disable deterministic algorithms.",
    )
    int_parser.add_argument("--verbose", action="store_true", help="Display progress information.")

    # Plot interventions subcommand
    plot_int_parser = subparsers.add_parser(
        "plot-interventions",
        help="Generate standard visualizations for intervention results JSON",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    plot_int_parser.add_argument("input_json", help="Path to intervention_results.json")
    plot_int_parser.add_argument("--out_dir", default="plots", help="Output directory for figures")

    # --- Config Loading ---
    pre_args, _ = parser.parse_known_args()
    pos_config = None
    if len(sys.argv) > 1 and sys.argv[1].endswith(".yaml") and not sys.argv[1].startswith("-"):
        pos_config = sys.argv[1]

    config_file = pre_args.config or pos_config
    config = load_config(config_file)

    if config:
        log.info("Configuration loaded from %s", config_file or "root config.yaml")
        # Alias graph_output_path → graph_path so intervene can find it
        if "graph_path" not in config and "graph_output_path" in config:
            config["graph_path"] = config["graph_output_path"]
        if "out_dir" not in config and "addition_experiment" in config:
            pass

    # Push config values into each subparser as defaults so CLI args override them
    if "generate-dataset" in subparsers.choices:
        set_parser_defaults_from_config(
            subparsers.choices["generate-dataset"], config, section="generate_dataset"
        )
    if "attribute" in subparsers.choices:
        set_parser_defaults_from_config(
            subparsers.choices["attribute"], config, section="attribution"
        )
    if "intervene" in subparsers.choices:
        set_parser_defaults_from_config(
            subparsers.choices["intervene"], config, section="addition_experiment"
        )
        set_parser_defaults_from_config(
            subparsers.choices["intervene"], config, section="intervention"
        )
    if "visualize-dataset" in subparsers.choices:
        set_parser_defaults_from_config(
            subparsers.choices["visualize-dataset"], config, section="visualize_dataset"
        )

    if pos_config and sys.argv[1] == pos_config:
        sys.argv.pop(1)

    args = parser.parse_args()

    # Validate required arguments that couldn't use required=True (config defaults bypass it)
    if args.command == "attribute":
        if not args.transcoder_set:
            attr_parser.error(
                "the following arguments are required: -t/--transcoder_set (or define in config.yaml)"
            )
        if not args.prompt:
            attr_parser.error("the following arguments are required: -p/--prompt")
    elif args.command == "generate-dataset":
        if not args.output_path:
            gen_parser.error(
                "the following arguments are required: --output_path (or define in config.yaml)"
            )
    elif args.command == "intervene":
        if not args.graph_path:
            int_parser.error(
                "the following arguments are required: -g/--graph_path (or define in config.yaml)"
            )
        if not args.transcoder_set:
            int_parser.error(
                "the following arguments are required: -t/--transcoder_set (or define in config.yaml)"
            )

    print_config(args, title=f"{args.command.title()} configuration")

    if args.command == "attribute":
        run_attribution(args, attr_parser)
    elif args.command == "generate-dataset":
        run_dataset_generation(args)
    elif args.command == "visualize-dataset":
        run_dataset_visualization(args)
    elif args.command == "intervene":
        run_intervene(args, int_parser)
    elif args.command == "plot-interventions":
        run_plot_interventions(args)


def run_plot_interventions(args):
    import json
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg")

    from .plot_interventions import (
        plot_causal_importance,
        plot_layer_locations,
        plot_leakage_comparison,
        plot_probability_impact,
    )

    in_path = Path(args.input_json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(in_path) as f:
        data = json.load(f)

    results = data.get("results", data)
    results = sorted(results, key=lambda x: x.get("intervention_layer", 0))

    print(f"Loaded {len(results)} groups. Generating plots in {out_dir}/ ...")

    plot_causal_importance(results, out_dir)
    plot_probability_impact(results, out_dir)
    plot_leakage_comparison(results, out_dir)
    plot_layer_locations(results, out_dir)

    print("Done.")


def run_dataset_generation(args):
    """Dispatch to the dataset generation pipeline."""
    repo_root = str(Path(__file__).resolve().parent.parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from experiments.addition.dataset_generation import (
        DatasetConfig,
        SamplingStrategy,
        TemplateID,
        generate_dataset,
        write_dataset,
    )

    templates = [TemplateID(t) for t in args.templates]

    config = DatasetConfig(
        model=args.model,
        output_path=Path(args.output_path),
        templates=templates,
        sampling_strategy=SamplingStrategy(args.sampling_strategy),
        max_value=args.max_value,
        n_samples=args.n_samples,
        stratified_n_per_category=args.stratified_n_per_category,
        stratified_uniform_remainder=args.stratified_uniform_remainder,
        top_k=args.top_k,
        batch_size=args.batch_size,
        max_gen_tokens=args.max_gen_tokens,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
    )

    if config.sampling_strategy == SamplingStrategy.RANDOM and config.n_samples is None:
        raise ValueError("--n_samples is required when using random sampling strategy")

    records, summary = generate_dataset(config)
    write_dataset(records, summary, config.output_path)

    print("\nDataset generation complete!")


def run_dataset_visualization(args):
    """Dispatch to the dataset visualisation pipeline."""
    repo_root = str(Path(__file__).resolve().parent.parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from experiments.addition.dataset_generation.visualize_dataset import (
        create_comprehensive_report,
    )

    create_comprehensive_report(
        jsonl_path=Path(args.dataset_path),
        output_dir=Path(args.output_dir),
        template_id=args.template,
    )


def run_attribution(args, parser):
    if bool(args.slug) != bool(args.graph_file_dir):
        which_one = "slug" if args.slug else "graph_file_dir"
        missing_one = "graph_file_dir" if args.slug else "slug"
        warnings.warn(
            (
                f"You provided --{which_one} but not --{missing_one}. Both are required "
                f"for creating graph files (check your config.yaml or CLI flags)."
            ),
            UserWarning,
            stacklevel=2,
        )

    create_graph_files_enabled = args.slug is not None and args.graph_file_dir is not None

    if not create_graph_files_enabled and not args.graph_output_path:
        parser.error(
            "--graph_output_path is required when not creating graph files "
            "(--slug and --graph_file_dir)"
        )

    if create_graph_files_enabled:
        os.makedirs(args.graph_file_dir, exist_ok=True)

    import torch

    dtype = args.dtype
    dtype_mapping = {
        "fp32": "float32",
        "bf16": "bfloat16",
        "fp16": "float16",
    }
    if dtype in dtype_mapping:
        dtype = dtype_mapping[dtype]
    dtype = getattr(torch, dtype)

    # Run attribution
    print(f"model:  {args.model}")
    print(f"dtype:  {dtype}")
    print(f'prompt: "{args.prompt}"')
    if args.graph_output_path:
        print(f"output: {args.graph_output_path}")
    print(f"logit threshold: cumulative p >= {args.desired_logit_prob} (max {args.max_n_logits})")
    print(f"backward batch size: {args.batch_size}")

    from .attribution_model import AttributionModel
    from .run_attribution import attribute
    from .utils.graph_viz import create_graph_files
    from .utils.hf_utils import load_transcoder_from_hub
    from .utils_seed import SeedConfig, set_all_seeds

    set_all_seeds(SeedConfig(seed=args.seed, deterministic=not args.nondeterministic))

    transcoder, config = load_transcoder_from_hub(
        args.transcoder_set,
        dtype=dtype,
        lazy_encoder=args.lazy_encoder,
        lazy_decoder=args.lazy_decoder,
    )
    args.model = args.model or config.get("model_name", None)
    if not args.model:
        parser.error("--model must be specified when not provided in transcoder config")

    model_instance = AttributionModel.from_pretrained_and_transcoders(
        args.model, transcoder, dtype=dtype
    )

    print("running attribution...")
    graph = attribute(
        prompt=args.prompt,
        model=model_instance,  # type:ignore
        max_n_logits=args.max_n_logits,
        desired_logit_prob=args.desired_logit_prob,
        batch_size=args.batch_size,
        verbose=args.verbose,
        offload=args.offload,
        max_feature_nodes=args.max_feature_nodes,
    )

    if args.graph_output_path:
        print(f"saving graph → {args.graph_output_path}")
        graph.to_pt(args.graph_output_path)

    # Save stats if requested
    if args.stats_file:
        from .utils.graph_viz import save_graph_stats

        save_graph_stats(graph, args.stats_file)

    if create_graph_files_enabled:
        print(f"building graph files (slug={args.slug!r})")
        create_graph_files(
            graph_or_path=graph,
            slug=args.slug,
            scan=None,
            output_path=args.graph_file_dir,
            node_threshold=args.node_threshold,
            edge_threshold=args.edge_threshold,
        )
        print(f"graph JSON files written to {args.graph_file_dir}")


def run_intervene(args, parser):
    """Load a saved graph and run constrained-patching interventions."""
    import torch

    repo_root = str(Path(__file__).resolve().parent.parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from .attribution_model import AttributionModel
    from .graph import Graph
    from .utils.hf_utils import load_transcoder_from_hub
    from .utils_seed import SeedConfig, set_all_seeds

    set_all_seeds(SeedConfig(seed=args.seed, deterministic=not args.nondeterministic))

    dtype = getattr(torch, args.dtype)

    graph_path = Path(args.graph_path)
    if not graph_path.exists():
        parser.error(f"Graph file not found: {graph_path}")
    print(f"loading graph from {graph_path}")
    graph = Graph.from_pt(str(graph_path))

    print(f"loading transcoders from {args.transcoder_set!r}")
    transcoder, config = load_transcoder_from_hub(
        args.transcoder_set,
        dtype=dtype,
        lazy_encoder=False,
        lazy_decoder=True,
    )
    model_name = args.model or config.get("model") or "Qwen/Qwen3-4B"
    print(f"loading model {model_name!r}  dtype={args.dtype}")
    model = AttributionModel.from_pretrained_and_transcoders(model_name, transcoder, dtype=dtype)

    # Fall back to addition experiment defaults if prompts not supplied
    from experiments.addition.dataset_generation.generate_dataset_with_predictions import (
        TemplateID,
        build_prompt,
    )
    from experiments.addition.prompts import FOCUS_PROMPT

    clean_prompt = args.prompt or FOCUS_PROMPT
    perturbed_prompt = args.perturbed_prompt or build_prompt(TemplateID.T0, 36, 60)

    from .interventions import run_interventions

    out_dir = Path(args.out_dir)
    results = run_interventions(
        model,
        graph,
        out_dir=out_dir,
        prompt=clean_prompt,
        perturbed_prompt=perturbed_prompt,
        node_threshold=args.node_threshold,
        edge_threshold=args.edge_threshold,
        alpha=args.alpha,
        top_n_groups=args.top_n_groups,
    )
    print(f"\n✓  intervene complete → {out_dir}  ({len(results)} groups tested)")


def main_generate_dataset():
    """CLI entry point for the miq-generate-dataset standalone script."""
    sys.argv.insert(1, "generate-dataset")
    main()


if __name__ == "__main__":
    main()
