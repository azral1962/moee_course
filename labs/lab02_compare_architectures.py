"""Lab 2: compare setup modes while holding the MIG reasoner constant."""

from simulation.engine import build_demo_simulation
from simulation.providers import HeuristicProvider


for architecture in ("baseline", "sota", "tise"):
    simulation = build_demo_simulation(
        architecture=architecture,
        reasoning="probabilistic",
        provider=HeuristicProvider(),
        steps=24,
        seed=42,
        disruptive=True,
    )
    result = simulation.run()
    result.to_csv(f"results/course_lab02_{architecture}.csv")
    print(result.summary())
