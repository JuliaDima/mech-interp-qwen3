#!/usr/bin/env python3
"""Generate/submit attribution graph jobs for matched concept prompts."""

from __future__ import annotations

import argparse
import os
import random
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.concept_localization.pipeline.run_concept import CONCEPTS, _load_concept
from scripts.model_config import add_model_config_arg, resolve_model_args


def _q(value: object) -> str:
    return shlex.quote(str(value))


def _write_chunk(path: Path, commands: list[str], scratch_base: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write("set -euo pipefail\n")
        f.write(f"export MIQ_SCRATCH_BASE=${{MIQ_SCRATCH_BASE:-{_q(scratch_base)}}}\n")
        f.write('export MIQ_CACHE_DIR="${MIQ_SCRATCH_BASE}/cache"\n')
        f.write('export MIQ_RUNS_DIR="${MIQ_SCRATCH_BASE}/runs"\n')
        f.write('export MIQ_TMP_DIR="${MIQ_SCRATCH_BASE}/tmp"\n')
        f.write('export HF_HOME="${MIQ_CACHE_DIR}/huggingface"\n')
        f.write('export TRANSFORMERS_CACHE="${HF_HOME}/transformers"\n')
        f.write('mkdir -p "${MIQ_CACHE_DIR}" "${MIQ_RUNS_DIR}/attribution" "${MIQ_TMP_DIR}" "${MIQ_SCRATCH_BASE}/graphs"\n')
        for command in commands:
            f.write(command)
            f.write("\n")
    path.chmod(0o755)


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--concept", required=True, choices=CONCEPTS)
    parser.add_argument("--template", default="T0")
    parser.add_argument("--n-per-class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scratch-base", default=os.environ.get("MIQ_SCRATCH_BASE", f"/rds/user/{os.environ.get('USER', '$USER')}/hpc-work/p28"))
    parser.add_argument("--chunk-size", type=int, default=10, help="Prompts per chunk script")
    add_model_config_arg(parser)
    parser.add_argument("--model", default=None)
    parser.add_argument("--transcoder-set", dest="transcoder_set", default=None)
    parser.add_argument("--node-threshold", type=float, default=0.9)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--submit", action="store_true", help="Submit generated chunks with sbatch")
    parser.add_argument("--dry-run", action="store_true", help="Print sbatch commands without submitting")
    args = parser.parse_args()
    resolve_model_args(args)

    pairs = [p for p in _load_concept(args.concept, args.n_per_class, args.seed) if p.template == args.template]
    if not pairs:
        raise SystemExit(f"No pairs found for {args.concept}/{args.template}")
    if len(pairs) < args.n_per_class:
        print(f"Warning: only found {len(pairs)} pairs (requested {args.n_per_class}), using all")
    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    pairs = pairs[: args.n_per_class]

    scratch = Path(args.scratch_base)
    run_dir = scratch / "runs" / "attribution"
    graph_base = scratch / "graphs"
    chunk_dir = run_dir / f"{args.concept}_{args.template}_sbatch_chunks"

    commands: list[str] = []
    for i, pair in enumerate(pairs):
        for cls, prompt in (("pos", pair.prompt_pos), ("neg", pair.prompt_neg)):
            slug = f"{args.concept}_{args.template}_{cls}_{i:03d}"
            pt_file = run_dir / f"{slug}.pt"
            graph_dir = graph_base / slug
            commands.append(
                "python -m mechinterp_qwen3 attribute "
                f"-m {_q(args.model)} "
                f"-t {_q(args.transcoder_set)} "
                f"-p {_q(prompt)} "
                f"-o {_q(pt_file)} "
                f"--slug {_q(slug)} "
                f"--graph_file_dir {_q(graph_dir)} "
                f"--node_threshold {_q(args.node_threshold)} "
                f"--dtype {_q(args.dtype)} "
                "--verbose"
            )
            commands.append(f"rm -f {_q(pt_file)}")

    chunk_paths: list[Path] = []
    per_chunk_lines = args.chunk_size * 2
    for chunk_idx, start in enumerate(range(0, len(commands), per_chunk_lines)):
        chunk_path = chunk_dir / f"chunk_{chunk_idx:03d}.sh"
        _write_chunk(chunk_path, commands[start : start + per_chunk_lines], args.scratch_base)
        chunk_paths.append(chunk_path)

    submit_script = run_dir / f"submit_{args.concept}_{args.template}_graphs.sh"
    with submit_script.open("w") as f:
        f.write("#!/usr/bin/env bash\nset -euo pipefail\n")
        for chunk in chunk_paths:
            f.write(f"sbatch scripts/sbatch_run.sh bash {_q(chunk)}\n")
    submit_script.chmod(0o755)

    print(f"wrote {len(chunk_paths)} chunks for {2 * len(pairs)} graphs")
    print(f"chunks: {chunk_dir}")
    print(f"submit: {submit_script}")

    if args.submit or args.dry_run:
        for chunk in chunk_paths:
            cmd = ["sbatch", "scripts/sbatch_run.sh", "bash", str(chunk)]
            if args.dry_run:
                print("DRY:", " ".join(_q(x) for x in cmd))
            else:
                subprocess.run(cmd, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
