"""Simple Triple-Market signals used by the PSKVE transformation engine."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .models import clamp


@dataclass(frozen=True, slots=True)
class MarketSignal:
    name: str
    price_multiplier: float
    availability: float


class Market(ABC):
    name = "abstract"

    @abstractmethod
    def discover(self, *, demand: float, disruption: float) -> MarketSignal:
        raise NotImplementedError


class StockMarket(Market):
    """Represents capitalization and availability of fixed mission assets."""

    name = "stock"

    def discover(self, *, demand: float, disruption: float) -> MarketSignal:
        return MarketSignal(
            name=self.name,
            price_multiplier=max(0.1, 1.0 + 0.1 * demand - 0.08 * disruption),
            availability=clamp(1.0 - 0.12 * disruption),
        )


class FinancialMarket(Market):
    """Represents short-term liquidity available for mission execution."""

    name = "financial"

    def discover(self, *, demand: float, disruption: float) -> MarketSignal:
        return MarketSignal(
            name=self.name,
            price_multiplier=max(0.1, 1.0 + 0.2 * disruption),
            availability=clamp(1.0 - 0.3 * disruption + 0.05 * demand),
        )


class ResourceOutputMarket(Market):
    """Represents operational R-to-O price discovery and clearing conditions."""

    name = "resource_output"

    def discover(self, *, demand: float, disruption: float) -> MarketSignal:
        return MarketSignal(
            name=self.name,
            price_multiplier=max(0.1, 0.85 + 0.35 * demand + 0.25 * disruption),
            availability=clamp(1.0 - 0.4 * disruption),
        )


@dataclass(frozen=True, slots=True)
class TripleMarketSignals:
    stock: MarketSignal
    financial: MarketSignal
    resource_output: MarketSignal


class TripleMarketSystem:
    def __init__(
        self,
        stock: Market | None = None,
        financial: Market | None = None,
        resource_output: Market | None = None,
    ):
        self.stock = stock or StockMarket()
        self.financial = financial or FinancialMarket()
        self.resource_output = resource_output or ResourceOutputMarket()

    def discover(self, *, demand: float, disruption: float) -> TripleMarketSignals:
        return TripleMarketSignals(
            stock=self.stock.discover(demand=demand, disruption=disruption),
            financial=self.financial.discover(demand=demand, disruption=disruption),
            resource_output=self.resource_output.discover(demand=demand, disruption=disruption),
        )
