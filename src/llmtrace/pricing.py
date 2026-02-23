"""Pricing registry for LLM model cost computation."""

from decimal import Decimal

from pydantic import BaseModel, ConfigDict

from llmtrace.models import Cost, TokenUsage

_MILLION = Decimal("1000000")


class ModelPricing(BaseModel):
    """Pricing rates for a single LLM model.

    All rates are per million tokens, stored as Decimal for exact arithmetic.
    """

    input_per_million: Decimal
    output_per_million: Decimal
    cache_read_per_million: Decimal | None = None
    cache_write_per_million: Decimal | None = None

    model_config = ConfigDict(frozen=True)


class PricingRegistry:
    """Registry of model pricing data with lookup and cost computation.

    Supports exact and prefix-based model matching. Default pricing data
    is loaded on initialization.
    """

    def __init__(self) -> None:
        self._pricing: dict[tuple[str, str], ModelPricing] = {}
        self._load_defaults()

    def _load_defaults(self) -> None:
        """Load default pricing data for known models."""
        defaults: list[tuple[str, str, ModelPricing]] = [
            (
                "anthropic",
                "claude-sonnet-4-20250514",
                ModelPricing(
                    input_per_million=Decimal("3.00"),
                    output_per_million=Decimal("15.00"),
                ),
            ),
            (
                "anthropic",
                "claude-haiku-3-5-20241022",
                ModelPricing(
                    input_per_million=Decimal("0.80"),
                    output_per_million=Decimal("4.00"),
                ),
            ),
            (
                "anthropic",
                "claude-opus-4-20250514",
                ModelPricing(
                    input_per_million=Decimal("15.00"),
                    output_per_million=Decimal("75.00"),
                ),
            ),
            (
                "openai",
                "gpt-4o",
                ModelPricing(
                    input_per_million=Decimal("2.50"),
                    output_per_million=Decimal("10.00"),
                ),
            ),
            (
                "openai",
                "gpt-4o-mini",
                ModelPricing(
                    input_per_million=Decimal("0.15"),
                    output_per_million=Decimal("0.60"),
                ),
            ),
            (
                "openai",
                "o1",
                ModelPricing(
                    input_per_million=Decimal("15.00"),
                    output_per_million=Decimal("60.00"),
                ),
            ),
            (
                "openai",
                "o3-mini",
                ModelPricing(
                    input_per_million=Decimal("1.10"),
                    output_per_million=Decimal("4.40"),
                ),
            ),
            (
                "google",
                "gemini-2.0-flash",
                ModelPricing(
                    input_per_million=Decimal("0.10"),
                    output_per_million=Decimal("0.40"),
                ),
            ),
            (
                "google",
                "gemini-2.0-pro",
                ModelPricing(
                    input_per_million=Decimal("1.25"),
                    output_per_million=Decimal("10.00"),
                ),
            ),
        ]
        for provider, model, pricing in defaults:
            self._pricing[(provider, model)] = pricing

    def get(self, provider: str, model: str) -> ModelPricing | None:
        """Look up pricing for a model.

        Tries exact match first, then prefix matching where either the stored
        model name starts with the requested model, or the requested model
        starts with the stored model name.
        """
        key = (provider, model)
        if key in self._pricing:
            return self._pricing[key]

        for (stored_provider, stored_model), pricing in self._pricing.items():
            if stored_provider != provider:
                continue
            if stored_model.startswith(model) or model.startswith(stored_model):
                return pricing

        return None

    def register(self, provider: str, model: str, pricing: ModelPricing) -> None:
        """Register or update pricing for a model."""
        self._pricing[(provider, model)] = pricing

    def compute_cost(self, provider: str, model: str, usage: TokenUsage) -> Cost | None:
        """Compute cost for a model given token usage.

        Returns None if the model is not found in the registry.
        All arithmetic uses Decimal for exact results.
        """
        pricing = self.get(provider, model)
        if pricing is None:
            return None

        input_cost = Decimal(usage.prompt_tokens) * pricing.input_per_million / _MILLION
        output_cost = Decimal(usage.completion_tokens) * pricing.output_per_million / _MILLION

        if usage.cache_read_tokens and pricing.cache_read_per_million is not None:
            input_cost += (
                Decimal(usage.cache_read_tokens) * pricing.cache_read_per_million / _MILLION
            )

        if usage.cache_write_tokens and pricing.cache_write_per_million is not None:
            input_cost += (
                Decimal(usage.cache_write_tokens) * pricing.cache_write_per_million / _MILLION
            )

        return Cost(
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=input_cost + output_cost,
        )

    def list_models(self, provider: str | None = None) -> list[tuple[str, str]]:
        """List all registered (provider, model) pairs.

        Optionally filtered by provider name.
        """
        if provider is None:
            return list(self._pricing.keys())
        return [(p, m) for (p, m) in self._pricing if p == provider]
