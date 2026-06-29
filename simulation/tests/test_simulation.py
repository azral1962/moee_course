from __future__ import annotations

import unittest
from unittest.mock import patch

from simulation.coordination import (
    AdaptivePUDALCoordinator,
    CoordinationContext,
    IndependentCoordinator,
    SmartContractCoordinator,
    SmartContractTerms,
)
from simulation.engine import build_demo_simulation
from simulation.markets import TripleMarketSystem
from simulation.models import MIGProposal, TriuneIntelligence
from simulation.providers import (
    DeepSeekProvider,
    FakeProvider,
    HeuristicProvider,
    OllamaProvider,
    OpenAIProvider,
)
from simulation.reasoning import AgentReasoner


def coordination_context() -> CoordinationContext:
    return CoordinationContext(
        step=0,
        demand=0.8,
        disruption=0.4,
        resource_ratios={"a": 1.0, "b": 0.5},
        roles={"a": "source", "b": "operator"},
        triune=TriuneIntelligence(
            human_natural={"safety": 0.7},
            crowd_cultural_collective={"demand": 0.8},
            artificial={"anomaly": 0.4},
        ),
    )


class CoordinationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.proposals = {
            "a": MIGProposal("a", 0.6, "local", reasoning_mode="historical"),
            "b": MIGProposal("b", 0.8, "local", reasoning_mode="historical"),
        }

    def test_independent_preserves_proposals(self) -> None:
        result = IndependentCoordinator().coordinate(self.proposals, coordination_context())
        self.assertEqual(result, self.proposals)

    def test_smart_contract_applies_fixed_cap(self) -> None:
        coordinator = SmartContractCoordinator(
            SmartContractTerms(intensity_caps={"a": 0.4}, role_multipliers={})
        )
        result = coordinator.coordinate(self.proposals, coordination_context())
        self.assertEqual(result["a"].intensity, 0.4)

    def test_adaptive_pudal_respects_resource_ratio(self) -> None:
        result = AdaptivePUDALCoordinator().coordinate(self.proposals, coordination_context())
        self.assertLessEqual(result["b"].intensity, 0.5)


class AgentReasonerTests(unittest.TestCase):
    def test_fake_provider_creates_bounded_proposal(self) -> None:
        provider = FakeProvider({"intensity": 1.4, "confidence": 0.8, "rationale": "test"})
        simulation = build_demo_simulation(
            architecture="baseline",
            reasoning="agent",
            provider=provider,
            steps=1,
        )
        result = simulation.run()
        self.assertTrue(all(record.proposed_intensity == 1.0 for record in result.records))
        self.assertEqual(provider.calls, 4)


class ProviderAdapterTests(unittest.TestCase):
    schema = {"type": "object"}

    def test_ollama_response_parsing(self) -> None:
        provider = OllamaProvider(base_url="http://unused", model="test")
        with patch.object(
            provider,
            "_post",
            return_value={"message": {"content": '{"intensity": 0.4}'}},
        ):
            result = provider.generate_json(system_prompt="s", user_prompt="u", schema=self.schema)
        self.assertEqual(result["intensity"], 0.4)

    def test_openai_response_parsing(self) -> None:
        provider = OpenAIProvider(base_url="http://unused", model="test", api_key="test")
        response = {
            "output": [
                {"content": [{"type": "output_text", "text": '{"intensity": 0.5}'}]}
            ]
        }
        with patch.object(provider, "_post", return_value=response):
            result = provider.generate_json(system_prompt="s", user_prompt="u", schema=self.schema)
        self.assertEqual(result["intensity"], 0.5)

    def test_deepseek_response_parsing(self) -> None:
        provider = DeepSeekProvider(base_url="http://unused", model="test", api_key="test")
        response = {"choices": [{"message": {"content": '{"intensity": 0.6}'}}]}
        with patch.object(provider, "_post", return_value=response):
            result = provider.generate_json(system_prompt="s", user_prompt="u", schema=self.schema)
        self.assertEqual(result["intensity"], 0.6)


class MarketTests(unittest.TestCase):
    def test_disruption_reduces_market_availability(self) -> None:
        markets = TripleMarketSystem()
        normal = markets.discover(demand=0.6, disruption=0.0)
        disrupted = markets.discover(demand=0.6, disruption=0.8)
        self.assertLess(disrupted.stock.availability, normal.stock.availability)
        self.assertLess(disrupted.financial.availability, normal.financial.availability)
        self.assertLess(disrupted.resource_output.availability, normal.resource_output.availability)


class EndToEndTests(unittest.TestCase):
    def test_all_architectures_run_offline(self) -> None:
        for architecture in ("baseline", "sota", "tise"):
            with self.subTest(architecture=architecture):
                simulation = build_demo_simulation(
                    architecture=architecture,
                    reasoning="hybrid",
                    provider=HeuristicProvider(),
                    steps=6,
                )
                result = simulation.run()
                self.assertEqual(len(result.records), 24)
                self.assertGreater(result.total_value, 0)
                self.assertGreaterEqual(result.average_mission_achievement, 0)


if __name__ == "__main__":
    unittest.main()
