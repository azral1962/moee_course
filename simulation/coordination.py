"""MIG setup modes: independent, smart contract, and adaptive PUDAL."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .models import MIGProposal, TriuneIntelligence, clamp
from .providers import HeuristicProvider, JSONAgentProvider, ProviderError


@dataclass(frozen=True, slots=True)
class CoordinationContext:
    step: int
    demand: float
    disruption: float
    resource_ratios: dict[str, float]
    roles: dict[str, str]
    triune: TriuneIntelligence


class MIGCoordinator(ABC):
    name = "abstract"

    @abstractmethod
    def coordinate(
        self,
        proposals: dict[str, MIGProposal],
        context: CoordinationContext,
    ) -> dict[str, MIGProposal]:
        raise NotImplementedError


class IndependentCoordinator(MIGCoordinator):
    name = "independent"

    def coordinate(self, proposals: dict[str, MIGProposal], context: CoordinationContext) -> dict[str, MIGProposal]:
        return dict(proposals)


@dataclass(slots=True)
class SmartContractTerms:
    intensity_caps: dict[str, float] = field(default_factory=dict)
    role_multipliers: dict[str, float] = field(default_factory=dict)
    max_total_intensity: float | None = None


class SmartContractCoordinator(MIGCoordinator):
    """Fixed, prenegotiated coordination with no within-run policy changes."""

    name = "smart_contract"

    def __init__(self, terms: SmartContractTerms | None = None):
        self.terms = terms or SmartContractTerms(
            role_multipliers={"source": 1.05, "operator": 1.05, "regulator": 0.9}
        )

    def coordinate(self, proposals: dict[str, MIGProposal], context: CoordinationContext) -> dict[str, MIGProposal]:
        revised: dict[str, MIGProposal] = {}
        for stakeholder_id, proposal in proposals.items():
            role = context.roles[stakeholder_id]
            multiplier = self.terms.role_multipliers.get(role, 1.0)
            cap = self.terms.intensity_caps.get(stakeholder_id, 1.0)
            intensity = min(cap, proposal.intensity * multiplier)
            revised[stakeholder_id] = proposal.revised(
                intensity,
                f"prenegotiated smart contract; original={proposal.rationale}",
            )
        if self.terms.max_total_intensity is not None:
            total = sum(item.intensity for item in revised.values())
            if total > self.terms.max_total_intensity and total > 0:
                scale = self.terms.max_total_intensity / total
                revised = {
                    key: value.revised(value.intensity * scale, value.rationale + "; contract total cap")
                    for key, value in revised.items()
                }
        return revised


PUDAL_SCHEMA = {
    "type": "object",
    "properties": {
        "intensities": {
            "type": "object",
            "additionalProperties": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "rationale": {"type": "string"},
    },
    "required": ["intensities", "rationale"],
}


class AdaptivePUDALCoordinator(MIGCoordinator):
    """Adaptive cooperative coordinator driven by Triune Intelligence."""

    name = "adaptive_pudal"

    def __init__(
        self,
        provider: JSONAgentProvider | None = None,
        *,
        allow_fallback: bool = True,
    ):
        self.provider = provider or HeuristicProvider()
        self.allow_fallback = allow_fallback

    @staticmethod
    def _heuristic_coordinate(
        proposals: dict[str, MIGProposal], context: CoordinationContext
    ) -> dict[str, MIGProposal]:
        revised: dict[str, MIGProposal] = {}
        for stakeholder_id, proposal in proposals.items():
            role = context.roles[stakeholder_id]
            resource_ratio = clamp(context.resource_ratios.get(stakeholder_id, 1.0))
            resilience_boost = 0.12 * context.disruption if role in {"source", "operator"} else 0.0
            safety_boost = 0.08 * context.disruption if role == "regulator" else 0.0
            demand_target = 0.2 + 0.75 * context.demand
            intensity = min(resource_ratio, 0.55 * proposal.intensity + 0.45 * demand_target)
            intensity = min(resource_ratio, clamp(intensity + resilience_boost + safety_boost))
            revised[stakeholder_id] = proposal.revised(
                intensity,
                f"adaptive PUDAL heuristic; original={proposal.rationale}",
            )
        return revised

    def coordinate(self, proposals: dict[str, MIGProposal], context: CoordinationContext) -> dict[str, MIGProposal]:
        if isinstance(self.provider, HeuristicProvider):
            return self._heuristic_coordinate(proposals, context)
        state: dict[str, Any] = {
            "step": context.step,
            "demand": context.demand,
            "disruption": context.disruption,
            "resource_ratios": context.resource_ratios,
            "roles": context.roles,
            "proposals": {key: value.intensity for key, value in proposals.items()},
            "triune_intelligence": context.triune.as_dict(),
        }
        system = (
            "You are the cooperative PUDAL coordinator. Revise stakeholder MIG proposals only when "
            "needed for resource feasibility, mission alignment, and disruption resilience. Preserve "
            "stakeholder autonomy and return every stakeholder ID. Return JSON only."
        )
        try:
            decision = self.provider.generate_json(
                system_prompt=system,
                user_prompt=json.dumps(state, sort_keys=True),
                schema=PUDAL_SCHEMA,
            )
            intensities = decision["intensities"]
            if set(intensities) != set(proposals):
                raise ProviderError("PUDAL response must contain exactly all stakeholder IDs")
            rationale = str(decision.get("rationale", "adaptive cooperative coordination"))
            return {
                stakeholder_id: proposal.revised(
                    min(clamp(float(intensities[stakeholder_id])), clamp(context.resource_ratios[stakeholder_id])),
                    f"{self.provider.name} PUDAL: {rationale}; original={proposal.rationale}",
                )
                for stakeholder_id, proposal in proposals.items()
            }
        except (ProviderError, KeyError, TypeError, ValueError):
            if not self.allow_fallback:
                raise
            return self._heuristic_coordinate(proposals, context)
