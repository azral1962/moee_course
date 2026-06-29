"""Lab 1: validate a baseline ecosystem using historical MIG reasoning."""

from simulation.engine import build_demo_simulation
from simulation.providers import HeuristicProvider


simulation = build_demo_simulation(
    architecture="baseline",
    reasoning="historical",
    provider=HeuristicProvider(),
    steps=12,
    disruptive=False,
)
result = simulation.run()
print(result.summary())
result.to_csv("results/course_lab01_baseline.csv")
