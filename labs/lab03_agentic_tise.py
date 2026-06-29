"""Lab 3: use a selected provider for stakeholder and PUDAL agent decisions."""

import argparse

from simulation.engine import build_demo_simulation
from simulation.providers import ProviderFactory


parser = argparse.ArgumentParser()
parser.add_argument("--provider", choices=("heuristic", "ollama", "openai", "deepseek"), default="heuristic")
parser.add_argument("--model")
parser.add_argument("--steps", type=int, default=6)
args = parser.parse_args()

provider = ProviderFactory.create(args.provider, args.model)
simulation = build_demo_simulation(
    architecture="tise",
    reasoning="agent",
    provider=provider,
    steps=args.steps,
    disruptive=True,
    allow_agent_fallback=args.provider != "heuristic",
)
result = simulation.run()
print(result.summary())
result.to_csv("results/course_lab03_agentic_tise.csv")
