"""Tests for llmtrace pricing registry."""

from decimal import Decimal

from llmtrace.models import Cost, TokenUsage
from llmtrace.pricing import ModelPricing, PricingRegistry


class TestPricingRegistry:
    """Tests for PricingRegistry."""

    def test_exact_model_lookup(self) -> None:
        registry = PricingRegistry()
        pricing = registry.get("anthropic", "claude-sonnet-4-20250514")
        assert pricing is not None
        assert pricing.input_per_million == Decimal("3.00")
        assert pricing.output_per_million == Decimal("15.00")

    def test_prefix_match_short_query(self) -> None:
        """Stored 'claude-sonnet-4-20250514' matches query 'claude-sonnet-4'."""
        registry = PricingRegistry()
        pricing = registry.get("anthropic", "claude-sonnet-4")
        assert pricing is not None
        assert pricing.input_per_million == Decimal("3.00")

    def test_prefix_match_long_query(self) -> None:
        """Query 'claude-sonnet-4-20250514' matches when stored as shorter prefix."""
        registry = PricingRegistry()
        registry.register(
            "anthropic",
            "claude-test",
            ModelPricing(
                input_per_million=Decimal("1.00"),
                output_per_million=Decimal("5.00"),
            ),
        )
        pricing = registry.get("anthropic", "claude-test-20250514")
        assert pricing is not None
        assert pricing.input_per_million == Decimal("1.00")

    def test_unknown_model_returns_none(self) -> None:
        registry = PricingRegistry()
        assert registry.get("anthropic", "nonexistent-model") is None

    def test_register_adds_new_model(self) -> None:
        registry = PricingRegistry()
        new_pricing = ModelPricing(
            input_per_million=Decimal("2.00"),
            output_per_million=Decimal("8.00"),
        )
        registry.register("newprovider", "new-model", new_pricing)
        result = registry.get("newprovider", "new-model")
        assert result is not None
        assert result.input_per_million == Decimal("2.00")
        assert result.output_per_million == Decimal("8.00")

    def test_register_overrides_existing(self) -> None:
        registry = PricingRegistry()
        updated = ModelPricing(
            input_per_million=Decimal("99.00"),
            output_per_million=Decimal("199.00"),
        )
        registry.register("anthropic", "claude-sonnet-4-20250514", updated)
        result = registry.get("anthropic", "claude-sonnet-4-20250514")
        assert result is not None
        assert result.input_per_million == Decimal("99.00")
        assert result.output_per_million == Decimal("199.00")

    def test_compute_cost_returns_correct_values(self) -> None:
        registry = PricingRegistry()
        usage = TokenUsage(prompt_tokens=1000, completion_tokens=500)
        cost = registry.compute_cost("anthropic", "claude-sonnet-4-20250514", usage)
        assert cost is not None
        assert isinstance(cost, Cost)
        # input: 1000 * 3.00 / 1_000_000 = 0.003
        assert cost.input_cost == Decimal("0.003")
        # output: 500 * 15.00 / 1_000_000 = 0.0075
        assert cost.output_cost == Decimal("0.0075")
        assert cost.total_cost == Decimal("0.0105")

    def test_compute_cost_unknown_model_returns_none(self) -> None:
        registry = PricingRegistry()
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50)
        assert registry.compute_cost("unknown", "model", usage) is None

    def test_compute_cost_zero_tokens(self) -> None:
        registry = PricingRegistry()
        usage = TokenUsage(prompt_tokens=0, completion_tokens=0)
        cost = registry.compute_cost("openai", "gpt-4o", usage)
        assert cost is not None
        assert cost.input_cost == Decimal("0")
        assert cost.output_cost == Decimal("0")
        assert cost.total_cost == Decimal("0")

    def test_compute_cost_with_cache_tokens(self) -> None:
        registry = PricingRegistry()
        registry.register(
            "anthropic",
            "claude-cached",
            ModelPricing(
                input_per_million=Decimal("3.00"),
                output_per_million=Decimal("15.00"),
                cache_read_per_million=Decimal("0.30"),
                cache_write_per_million=Decimal("3.75"),
            ),
        )
        usage = TokenUsage(
            prompt_tokens=1000,
            completion_tokens=500,
            cache_read_tokens=2000,
            cache_write_tokens=500,
        )
        cost = registry.compute_cost("anthropic", "claude-cached", usage)
        assert cost is not None
        # input: 1000 * 3.00 / 1M = 0.003
        # cache_read: 2000 * 0.30 / 1M = 0.0006
        # cache_write: 500 * 3.75 / 1M = 0.001875
        expected_input = Decimal("0.003") + Decimal("0.0006") + Decimal("0.001875")
        assert cost.input_cost == expected_input
        # output: 500 * 15.00 / 1M = 0.0075
        assert cost.output_cost == Decimal("0.0075")

    def test_list_models_returns_all(self) -> None:
        registry = PricingRegistry()
        models = registry.list_models()
        assert len(models) == 9
        assert ("anthropic", "claude-sonnet-4-20250514") in models
        assert ("openai", "gpt-4o") in models
        assert ("google", "gemini-2.0-flash") in models

    def test_list_models_filters_by_provider(self) -> None:
        registry = PricingRegistry()
        anthropic_models = registry.list_models(provider="anthropic")
        assert len(anthropic_models) == 3
        assert all(p == "anthropic" for p, _ in anthropic_models)
        openai_models = registry.list_models(provider="openai")
        assert len(openai_models) == 4
