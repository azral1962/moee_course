"""Simulation loop and a campus-food demonstration factory."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean

from .coordination import (
    AdaptivePUDALCoordinator,
    CoordinationContext,
    IndependentCoordinator,
    MIGCoordinator,
    SmartContractCoordinator,
    SmartContractTerms,
)
from .exchange import PSKVEExchangeEngine
from .markets import TripleMarketSystem
from .models import (
    Mission,
    PSKVEKind,
    Scenario,
    Stakeholder,
    StakeholderRole,
    StepRecord,
    TriuneIntelligence,
    VOMRAccount,
    clamp,
)
from .providers import JSONAgentProvider
from .reasoning import AgentReasoner, HistoricalReasoner, HybridReasoner, ProbabilisticReasoner


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    steps: int = 24
    seed: int = 42

    def __post_init__(self) -> None:
        if self.steps <= 0:
            raise ValueError("steps must be positive")


@dataclass(slots=True)
class SimulationResult:
    records: list[StepRecord]
    architecture: str
    scenario: str

    @property
    def total_value(self) -> float:
        latest: dict[str, float] = {}
        for record in self.records:
            latest[record.stakeholder_id] = record.total_value
        return sum(latest.values())

    @property
    def average_mission_achievement(self) -> float:
        return fmean(record.mission_achievement for record in self.records) if self.records else 0.0

    def summary(self) -> dict[str, float | str | int]:
        disrupted = [record for record in self.records if record.disruption > 0]
        return {
            "architecture": self.architecture,
            "scenario": self.scenario,
            "steps": max((record.step for record in self.records), default=-1) + 1,
            "total_value": round(self.total_value, 4),
            "average_mission_achievement": round(self.average_mission_achievement, 4),
            "disrupted_average_achievement": round(
                fmean(record.mission_achievement for record in disrupted) if disrupted else 0.0,
                4,
            ),
        }

    def to_csv(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(self.records[0].as_dict()) if self.records else []
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(record.as_dict() for record in self.records)
        return output


@dataclass(slots=True)
class MOEESimulation:
    stakeholders: list[Stakeholder]
    scenario: Scenario
    coordinator: MIGCoordinator
    config: SimulationConfig = field(default_factory=SimulationConfig)
    exchange_engine: PSKVEExchangeEngine = field(default_factory=PSKVEExchangeEngine)
    markets: TripleMarketSystem = field(default_factory=TripleMarketSystem)

    def run(self) -> SimulationResult:
        records: list[StepRecord] = []
        for step in range(self.config.steps):
            demand = self.scenario.demand_at(step)
            disruption = self.scenario.disruption_at(step)
            market_signals = self.markets.discover(demand=demand, disruption=disruption)
            proposals = {}
            contexts = {}
            for stakeholder in self.stakeholders:
                stakeholder.accounts.replenish(stakeholder.resource_replenishment)
                context = stakeholder.context(step=step, demand=demand, disruption=disruption)
                contexts[stakeholder.stakeholder_id] = context
                proposals[stakeholder.stakeholder_id] = stakeholder.propose_mig(context)

            triune = TriuneIntelligence(
                human_natural={"safety_floor": 0.35, "ecological_stress": disruption},
                crowd_cultural_collective={"observed_demand": demand, "cooperation": 0.75},
                artificial={"forecast_demand": clamp(0.9 * demand + 0.1), "anomaly": disruption},
            )
            coordination_context = CoordinationContext(
                step=step,
                demand=demand,
                disruption=disruption,
                resource_ratios={
                    stakeholder.stakeholder_id: clamp(
                        stakeholder.accounts.resources
                        / max(stakeholder.capacity * stakeholder.mission.resource_per_output, 1e-9)
                    )
                    for stakeholder in self.stakeholders
                },
                roles={stakeholder.stakeholder_id: stakeholder.role.value for stakeholder in self.stakeholders},
                triune=triune,
            )
            coordinated = self.coordinator.coordinate(proposals, coordination_context)

            for stakeholder in self.stakeholders:
                stakeholder_id = stakeholder.stakeholder_id
                exchange = self.exchange_engine.execute(
                    stakeholder,
                    coordinated[stakeholder_id],
                    disruption,
                    market_signals,
                )
                records.append(
                    StepRecord(
                        step=step,
                        stakeholder_id=stakeholder_id,
                        role=stakeholder.role.value,
                        proposed_intensity=proposals[stakeholder_id].intensity,
                        coordinated_intensity=coordinated[stakeholder_id].intensity,
                        resource_used=exchange.resource_used,
                        output_created=exchange.output_created,
                        value_change=exchange.value_change,
                        total_value=stakeholder.accounts.value,
                        mission_achievement=exchange.mission_achievement,
                        disruption=disruption,
                        reasoning_mode=proposals[stakeholder_id].reasoning_mode,
                        coordination_mode=self.coordinator.name,
                    )
                )
        return SimulationResult(records, self.coordinator.name, self.scenario.name)


def _make_reasoner(
    mode: str,
    *,
    provider: JSONAgentProvider,
    seed: int,
    offset: int,
    allow_agent_fallback: bool,
):
    historical = HistoricalReasoner((0.35, 0.45, 0.65, 0.8, 0.7, 0.5))
    probabilistic = ProbabilisticReasoner(mean=0.58, stddev=0.08, seed=seed + offset)
    agent = AgentReasoner(provider, allow_fallback=allow_agent_fallback)
    normalized = mode.lower()
    if normalized == "historical":
        return historical
    if normalized == "probabilistic":
        return probabilistic
    if normalized == "agent":
        return agent
    if normalized == "hybrid":
        return HybridReasoner(((0.3, historical), (0.25, probabilistic), (0.45, agent)))
    raise ValueError(f"unsupported reasoning mode: {mode}")


def build_demo_simulation(
    *,
    architecture: str = "tise",
    reasoning: str = "hybrid",
    provider: JSONAgentProvider,
    steps: int = 24,
    seed: int = 42,
    disruptive: bool = True,
    allow_agent_fallback: bool = False,
) -> MOEESimulation:
    specs = (
        ("users", "Student users", StakeholderRole.USER, "Affordable nutritious access", 65.0, 34.0),
        ("sources", "Food vendors", StakeholderRole.SOURCE, "Reliable meal production", 90.0, 55.0),
        ("regulators", "Campus regulator", StakeholderRole.REGULATOR, "Safety and sustainability", 45.0, 25.0),
        ("operators", "Food service operator", StakeholderRole.OPERATOR, "Timely ecosystem operation", 80.0, 48.0),
    )
    stakeholders = []
    for offset, (stakeholder_id, name, role, mission_name, capacity, target) in enumerate(specs):
        reasoner = _make_reasoner(
            reasoning,
            provider=provider,
            seed=seed,
            offset=offset,
            allow_agent_fallback=allow_agent_fallback,
        )
        stakeholders.append(
            Stakeholder(
                stakeholder_id=stakeholder_id,
                name=name,
                role=role,
                mission=Mission(
                    name=mission_name,
                    target_output_per_step=target,
                    resource_per_output=1.0,
                    value_per_output=1.2 if role in {StakeholderRole.USER, StakeholderRole.SOURCE} else 0.9,
                    resource_unit_cost=0.3,
                    pskve_focus=(PSKVEKind.SERVICE, PSKVEKind.VALUE, PSKVEKind.ENVIRONMENT),
                ),
                capacity=capacity,
                accounts=VOMRAccount(resources=capacity * 1.5),
                reasoner=reasoner,
                resource_replenishment=capacity * 0.55,
            )
        )

    architecture_name = architecture.lower()
    if architecture_name == "baseline":
        coordinator: MIGCoordinator = IndependentCoordinator()
    elif architecture_name == "sota":
        coordinator = SmartContractCoordinator(
            SmartContractTerms(
                role_multipliers={"source": 1.05, "operator": 1.08, "regulator": 0.95},
                max_total_intensity=3.2,
            )
        )
    elif architecture_name == "tise":
        coordinator = AdaptivePUDALCoordinator(provider, allow_fallback=allow_agent_fallback)
    else:
        raise ValueError(f"unsupported architecture: {architecture}")

    demand = (0.35, 0.45, 0.65, 0.9, 0.8, 0.55)
    disruption = (0.0, 0.0, 0.0, 0.7, 0.45, 0.1) if disruptive else (0.0,)
    scenario = Scenario(
        name="campus_food_disruption" if disruptive else "campus_food_normal",
        demand_profile=demand,
        disruption_profile=disruption,
        description="Recurring demand cycle with an optional supply/operations shock.",
    )
    return MOEESimulation(
        stakeholders=stakeholders,
        scenario=scenario,
        coordinator=coordinator,
        config=SimulationConfig(steps=steps, seed=seed),
    )
