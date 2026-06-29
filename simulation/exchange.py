"""PSKVE exchange execution and VOMR postings."""

from __future__ import annotations

from dataclasses import dataclass

from .markets import TripleMarketSignals
from .models import MIGProposal, Stakeholder, clamp


@dataclass(frozen=True, slots=True)
class ExchangeResult:
    stakeholder_id: str
    resource_used: float
    output_created: float
    value_change: float
    mission_achievement: float


class PSKVEExchangeEngine:
    """Minimal transformation engine that can be replaced by domain-specific subclasses."""

    def execute(
        self,
        stakeholder: Stakeholder,
        mig: MIGProposal,
        disruption: float,
        markets: TripleMarketSignals,
    ) -> ExchangeResult:
        mission = stakeholder.mission
        effective_capacity = (
            stakeholder.capacity
            * (1.0 - 0.35 * clamp(disruption))
            * markets.stock.availability
            * markets.financial.availability
        )
        desired_output = effective_capacity * mig.intensity
        available_resources = stakeholder.accounts.resources * markets.resource_output.availability
        resource_limited_output = available_resources / mission.resource_per_output
        output_created = max(0.0, min(desired_output, resource_limited_output))
        resource_used = output_created * mission.resource_per_output
        output_price_factor = 0.75 + 0.25 * markets.resource_output.price_multiplier
        resource_price_factor = markets.resource_output.price_multiplier
        financing_cost_factor = markets.financial.price_multiplier
        revenue = output_created * mission.value_per_output * output_price_factor
        cost = resource_used * mission.resource_unit_cost * resource_price_factor * financing_cost_factor
        value_change = revenue - cost
        achievement = output_created / mission.target_output_per_step
        stakeholder.accounts.post_transformation(
            resource_used=resource_used,
            output_created=output_created,
            value_change=value_change,
            mission_achievement=achievement,
        )
        return ExchangeResult(
            stakeholder_id=stakeholder.stakeholder_id,
            resource_used=resource_used,
            output_created=output_created,
            value_change=value_change,
            mission_achievement=clamp(achievement),
        )
