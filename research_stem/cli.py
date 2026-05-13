"""Command-line interface."""

from __future__ import annotations

import argparse
from pathlib import Path

from .evaluation import run_evaluation_from_files
from .evolution import evolve
from .io import resolve_path


DEFAULT_DOMAIN = "data/domains/research_agent_failures.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Research Stem Agent prototype")
    subparsers = parser.add_subparsers(dest="command", required=True)

    eval_parser = subparsers.add_parser("eval", help="Run deterministic evaluation")
    eval_parser.add_argument("--mode", choices=["baseline", "frozen"], default="baseline")
    eval_parser.add_argument("--split", choices=["train", "validation", "holdout"], default="holdout")
    eval_parser.add_argument("--domain", default=DEFAULT_DOMAIN)
    eval_parser.add_argument("--artifact-dir", default="artifacts")

    evolve_parser = subparsers.add_parser("evolve", help="Adapt and freeze an architecture config")
    evolve_parser.add_argument("--domain", default=DEFAULT_DOMAIN)
    evolve_parser.add_argument("--baseline", default="configs/baseline.json")
    evolve_parser.add_argument("--artifact-dir", default="artifacts")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "eval":
        config_path = "configs/baseline.json" if args.mode == "baseline" else str(Path(args.artifact_dir) / "frozen_architecture.json")
        run = run_evaluation_from_files(
            domain_path=resolve_path(args.domain),
            config_path=resolve_path(config_path),
            mode=args.mode,
            split=args.split,
            artifact_dir=args.artifact_dir,
        )
        print(f"Evaluation: {run.id}")
        print(f"Config: {run.config_name}")
        print(f"Aggregate metrics: {run.aggregate_metrics}")
        print(f"Trace: {run.trace_path}")
        return 0

    if args.command == "evolve":
        config = evolve(
            domain_path=resolve_path(args.domain),
            baseline_path=args.baseline,
            artifact_dir=args.artifact_dir,
        )
        print(f"Frozen config written: {Path(args.artifact_dir) / 'frozen_architecture.json'}")
        print(f"Selected: {config.name} frozen={config.frozen}")
        return 0

    parser.error("unknown command")
    return 2
