"""Core MOEE domain objects with no provider or UI dependencies."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .reasoning import MIGReasoner


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


class StakeholderRole(str, Enum):
    USER = "user"
    SOURCE = "source"
    REGULATOR = "regulator"
    OPERATOR = "operator"


class PSKVEKind(str, Enum):
    PRODUCT = "product"
    SERVICE = "service"
    KNOWLEDGE = "knowledge"
    VALUE = "value"
    ENVIRONMENT = "environment"


@dataclass(slots=True)
class VOMRAccount:
    """Value, Output, Mission, and Resource accounts for one stakeholder."""

    value: float = 0.0
    output: float = 0.0
    mission: float = 0.0
    resources: float = 0.0

    def replenish(self, amount: float) -> None:
        self.resources += max(0.0, amount)

    def post_transformation(
        self,
        *,
        resource_used: float,
        output_created: float,
        value_change: float,
        mission_achievement: float,
    ) -> None:
        if resource_used > self.resources + 1e-9:
            raise ValueError("resource posting would overdraw the VOMR resource account")
        self.resources -= max(0.0, resource_used)
        self.output += max(0.0, output_created)
        self.value += value_change
        self.mission += clamp(mission_achievement)


@dataclass(frozen=True, slots=True)
class Mission:
    name: str
    target_output_per_step: float
    resource_per_output: float = 1.0
    value_per_output: float = 1.0
    resource_unit_cost: float = 0.25
    pskve_focus: tuple[PSKVEKind, ...] = (PSKVEKind.SERVICE, PSKVEKind.VALUE)

    def __post_init__(self) -> None:
        if self.target_output_per_step <= 0:
            raise ValueError("target_output_per_step must be positive")
        if self.resource_per_output <= 0:
            raise ValueError("resource_per_output must be positive")


@dataclass(frozen=True, slots=True)
class StakeholderContext:
    step: int
    demand: float
    disruption: float
    available_resources: float
    capacity: float
    mission_name: str
    collective_signal: float = 0.5
    policy_signal: float = 0.5

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "demand": self.demand,
            "disruption": self.disruption,
            "available_resources": self.available_resources,
            "capacity": self.capacity,
            "mission_name": self.mission_name,
            "collective_signal": self.collective_signal,
            "policy_signal": self.policy_signal,
        }


@dataclass(frozen=True, slots=True)
class MIGProposal:
    stakeholder_id: str
    intensity: float
    rationale: str
    confidence: float = 1.0
    reasoning_mode: str = "unspecified"

    def __post_init__(self) -> None:
        object.__setattr__(self, "intensity", clamp(self.intensity))
        object.__setattr__(self, "confidence", clamp(self.confidence))

    def revised(self, intensity: float, rationale: str) -> "MIGProposal":
        return MIGProposal(
            stakeholder_id=self.stakeholder_id,
            intensity=intensity,
            rationale=rationale,
            confidence=self.confidence,
            reasoning_mode=self.reasoning_mode,
        )


@dataclass(slots=True)
class Stakeholder:
    stakeholder_id: str
    name: str
    role: StakeholderRole
    mission: Mission
    capacity: float
    accounts: VOMRAccount
    reasoner: "MIGReasoner"
    resource_replenishment: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("stakeholder capacity must be positive")

    def context(self, *, step: int, demand: float, disruption: float) -> StakeholderContext:
        return StakeholderContext(
            step=step,
            demand=clamp(demand),
            disruption=clamp(disruption),
            available_resources=self.accounts.resources,
            capacity=self.capacity,
            mission_name=self.mission.name,
            collective_signal=clamp(demand * (1.0 - 0.25 * disruption)),
            policy_signal=clamp(float(self.metadata.get("policy_signal", 0.5))),
        )

    def propose_mig(self, context: StakeholderContext) -> MIGProposal:
        return self.reasoner.propose(self, context)


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    demand_profile: tuple[float, ...]
    disruption_profile: tuple[float, ...]
    description: str = ""

    def __post_init__(self) -> None:
        if not self.demand_profile:
            raise ValueError("scenario requires a demand profile")
        if not self.disruption_profile:
            raise ValueError("scenario requires a disruption profile")

    def demand_at(self, step: int) -> float:
        return clamp(self.demand_profile[step % len(self.demand_profile)])

    def disruption_at(self, step: int) -> float:
        return clamp(self.disruption_profile[step % len(self.disruption_profile)])


@dataclass(frozen=True, slots=True)
class TriuneIntelligence:
    human_natural: dict[str, float]
    crowd_cultural_collective: dict[str, float]
    artificial: dict[str, float]

    def as_dict(self) -> dict[str, dict[str, float]]:
        return {
            "human_natural": self.human_natural,
            "crowd_cultural_collective": self.crowd_cultural_collective,
            "artificial": self.artificial,
        }


@dataclass(frozen=True, slots=True)
class StepRecord:
    step: int
    stakeholder_id: str
    role: str
    proposed_intensity: float
    coordinated_intensity: float
    resource_used: float
    output_created: float
    value_change: float
    total_value: float
    mission_achievement: float
    disruption: float
    reasoning_mode: str
    coordination_mode: str

    def as_dict(self) -> dict[str, Any]:
        return {
            field_name: getattr(self, field_name)
            for field_name in self.__dataclass_fields__
        }
