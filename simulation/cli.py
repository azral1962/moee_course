"""Command-line entry point for the MOEE simulation."""

from __future__ import annotations

import argparse
import json

from .engine import build_demo_simulation
from .providers import ProviderError, ProviderFactory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the MOEE/TISE campus-food simulation")
    parser.add_argument("--architecture", choices=("baseline", "sota", "tise"), default="tise")
    parser.add_argument(
        "--reasoning",
        choices=("historical", "probabilistic", "agent", "hybrid"),
        default="hybrid",
    )
    parser.add_argument(
        "--provider",
        choices=("heuristic", "ollama", "openai", "deepseek"),
        default=None,
        help="defaults to MOEE_LLM_PROVIDER or heuristic",
    )
    parser.add_argument("--model", default=None, help="override the selected provider's model")
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--normal", action="store_true", help="disable the disruption profile")
    parser.add_argument("--allow-agent-fallback", action="store_true")
    parser.add_argument("--output", help="optional step-level CSV output path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        provider = ProviderFactory.create(args.provider, args.model)
        simulation = build_demo_simulation(
            architecture=args.architecture,
            reasoning=args.reasoning,
            provider=provider,
            steps=args.steps,
            seed=args.seed,
            disruptive=not args.normal,
            allow_agent_fallback=args.allow_agent_fallback,
        )
        result = simulation.run()
    except (ProviderError, ValueError) as exc:
        print(f"error: {exc}")
        return 2

    if args.output:
        path = result.to_csv(args.output)
        print(f"records: {path}")
    print(json.dumps(result.summary(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
