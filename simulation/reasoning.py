"""Pluggable stakeholder reasoning strategies for creating MIG proposals."""

from __future__ import annotations

import json
import random
from abc import ABC, abstractmethod
from collections.abc import Sequence

from .models import MIGProposal, Stakeholder, StakeholderContext, clamp
from .providers import HeuristicProvider, JSONAgentProvider, ProviderError


MIG_SCHEMA = {
    "type": "object",
    "properties": {
        "intensity": {"type": "number", "minimum": 0, "maximum": 1},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "rationale": {"type": "string"},
    },
    "required": ["intensity", "confidence", "rationale"],
}


class MIGReasoner(ABC):
    name = "abstract"

    @abstractmethod
    def propose(self, stakeholder: Stakeholder, context: StakeholderContext) -> MIGProposal:
        raise NotImplementedError


class HistoricalReasoner(MIGReasoner):
    name = "historical"

    def __init__(self, schedule: Sequence[float]):
        if not schedule:
            raise ValueError("historical schedule cannot be empty")
        self.schedule = tuple(clamp(value) for value in schedule)

    def propose(self, stakeholder: Stakeholder, context: StakeholderContext) -> MIGProposal:
        intensity = self.schedule[context.step % len(self.schedule)]
        return MIGProposal(
            stakeholder_id=stakeholder.stakeholder_id,
            intensity=intensity,
            confidence=1.0,
            rationale="fixed/historical schedule",
            reasoning_mode=self.name,
        )


class ProbabilisticReasoner(MIGReasoner):
    name = "probabilistic"

    def __init__(self, mean: float = 0.55, stddev: float = 0.12, seed: int | None = None):
        if stddev < 0:
            raise ValueError("stddev cannot be negative")
        self.mean = clamp(mean)
        self.stddev = stddev
        self.random = random.Random(seed)

    def propose(self, stakeholder: Stakeholder, context: StakeholderContext) -> MIGProposal:
        conditional_mean = clamp(0.45 * self.mean + 0.55 * context.demand - 0.15 * context.disruption)
        intensity = clamp(self.random.gauss(conditional_mean, self.stddev))
        return MIGProposal(
            stakeholder_id=stakeholder.stakeholder_id,
            intensity=intensity,
            confidence=clamp(1.0 - 2.0 * self.stddev),
            rationale="sampled from a demand- and disruption-conditioned Gaussian model",
            reasoning_mode=self.name,
        )


class AgentReasoner(MIGReasoner):
    name = "agent"

    def __init__(
        self,
        provider: JSONAgentProvider,
        *,
        allow_fallback: bool = False,
        fallback: JSONAgentProvider | None = None,
    ):
        self.provider = provider
        self.allow_fallback = allow_fallback
        self.fallback = fallback or HeuristicProvider()

    def propose(self, stakeholder: Stakeholder, context: StakeholderContext) -> MIGProposal:
        state = context.as_dict() | {
            "stakeholder_id": stakeholder.stakeholder_id,
            "stakeholder_name": stakeholder.name,
            "role": stakeholder.role.value,
            "target_output": stakeholder.mission.target_output_per_step,
            "current_value": stakeholder.accounts.value,
        }
        system = (
            "You are an auditable MOEE stakeholder agent. Choose one mission intensity in [0,1]. "
            "Respect resource feasibility, respond to demand and disruption, and do not invent facts. "
            "Return JSON only with intensity, confidence, and a short rationale."
        )
        prompt = json.dumps(state, sort_keys=True)
        provider = self.provider
        try:
            decision = provider.generate_json(system_prompt=system, user_prompt=prompt, schema=MIG_SCHEMA)
        except ProviderError:
            if not self.allow_fallback:
                raise
            provider = self.fallback
            decision = provider.generate_json(system_prompt=system, user_prompt=prompt, schema=MIG_SCHEMA)
        try:
            return MIGProposal(
                stakeholder_id=stakeholder.stakeholder_id,
                intensity=float(decision["intensity"]),
                confidence=float(decision.get("confidence", 0.5)),
                rationale=f"{provider.name}: {decision.get('rationale', 'agent decision')}",
                reasoning_mode=f"agent:{provider.name}",
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProviderError(f"agent provider returned an invalid MIG decision: {decision}") from exc


class HybridReasoner(MIGReasoner):
    name = "hybrid"

    def __init__(self, weighted_reasoners: Sequence[tuple[float, MIGReasoner]]):
        positive = [(float(weight), reasoner) for weight, reasoner in weighted_reasoners if weight > 0]
        if not positive:
            raise ValueError("hybrid reasoner requires at least one positive weight")
        total = sum(weight for weight, _ in positive)
        self.weighted_reasoners = tuple((weight / total, reasoner) for weight, reasoner in positive)

    def propose(self, stakeholder: Stakeholder, context: StakeholderContext) -> MIGProposal:
        proposals = [
            (weight, reasoner.propose(stakeholder, context))
            for weight, reasoner in self.weighted_reasoners
        ]
        intensity = sum(weight * proposal.intensity for weight, proposal in proposals)
        confidence = sum(weight * proposal.confidence for weight, proposal in proposals)
        modes = ",".join(proposal.reasoning_mode for _, proposal in proposals)
        return MIGProposal(
            stakeholder_id=stakeholder.stakeholder_id,
            intensity=intensity,
            confidence=confidence,
            rationale=f"weighted hybrid of {modes}",
            reasoning_mode=self.name,
        )
