"""Lab 4: transfer MOEE to a compact elderly home-care case."""

from simulation.coordination import AdaptivePUDALCoordinator
from simulation.engine import MOEESimulation, SimulationConfig
from simulation.models import Mission, Scenario, Stakeholder, StakeholderRole, VOMRAccount
from simulation.providers import HeuristicProvider
from simulation.reasoning import HistoricalReasoner, ProbabilisticReasoner


provider = HeuristicProvider()
stakeholders = [
    Stakeholder(
        "older_adults",
        "Older adults",
        StakeholderRole.USER,
        Mission("Safe independent living", 20, value_per_output=1.3),
        30,
        VOMRAccount(resources=50),
        HistoricalReasoner((0.5, 0.6, 0.8, 0.7)),
        resource_replenishment=15,
    ),
    Stakeholder(
        "care_workers",
        "Home-care workers",
        StakeholderRole.SOURCE,
        Mission("Reliable care visits", 24, value_per_output=1.2),
        36,
        VOMRAccount(resources=60),
        ProbabilisticReasoner(mean=0.65, stddev=0.06, seed=7),
        resource_replenishment=18,
    ),
    Stakeholder(
        "care_authority",
        "Social-care authority",
        StakeholderRole.REGULATOR,
        Mission("Safety and continuity assurance", 12, value_per_output=0.9),
        20,
        VOMRAccount(resources=35),
        HistoricalReasoner((0.4, 0.5, 0.7, 0.6)),
        resource_replenishment=10,
    ),
    Stakeholder(
        "scheduler",
        "Home-care scheduler",
        StakeholderRole.OPERATOR,
        Mission("Timely visit coordination", 22, value_per_output=1.0),
        32,
        VOMRAccount(resources=55),
        ProbabilisticReasoner(mean=0.6, stddev=0.05, seed=8),
        resource_replenishment=16,
    ),
]

scenario = Scenario(
    "care_worker_shortage",
    demand_profile=(0.55, 0.65, 0.85, 0.75),
    disruption_profile=(0.0, 0.0, 0.75, 0.35),
    description="Care demand rises while workforce availability is disrupted.",
)

simulation = MOEESimulation(
    stakeholders=stakeholders,
    scenario=scenario,
    coordinator=AdaptivePUDALCoordinator(provider),
    config=SimulationConfig(steps=12, seed=7),
)
result = simulation.run()
print(result.summary())
result.to_csv("results/course_lab04_elderly_care.csv")
