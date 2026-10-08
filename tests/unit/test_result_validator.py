"""Unit tests for the LLM-based result validator."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from pg_mcp.config.settings import OpenAIConfig, ValidationConfig
from pg_mcp.models.errors import LLMError
from pg_mcp.services.result_validator import ResultValidator


def build_validator(config: ValidationConfig | None = None) -> tuple[ResultValidator, MagicMock]:
    """Build a validator with a mocked OpenAI client."""
    validator = ResultValidator(
        openai_config=OpenAIConfig(api_key="sk-test"),
        validation_config=config or ValidationConfig(enabled=True),
    )
    mock_client = MagicMock()
    validator.client = mock_client
    return validator, mock_client


def _completion(content: str, tokens: int | None = 50) -> MagicMock:
    """Build a fake ChatCompletion response."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    if tokens is None:
        response.usage = None
    else:
        response.usage = MagicMock()
        response.usage.total_tokens = tokens
    response.model_dump.return_value = {"id": "fake"}
    return response


class TestResultValidator:
    """ResultValidator behavior with a mocked LLM client."""

    @pytest.mark.asyncio
    async def test_disabled_returns_full_confidence(self) -> None:
        """Disabled validation short-circuits without calling the LLM."""
        validator, mock_client = build_validator(ValidationConfig(enabled=False))
        result = await validator.validate("q", "SELECT 1", [{"x": 1}], 1)
        assert result.confidence == 100
        assert result.is_acceptable is True
        mock_client.chat.completions.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_successful_validation(self) -> None:
        """A valid LLM JSON response maps to the result model."""
        validator, mock_client = build_validator(
            ValidationConfig(enabled=True, confidence_threshold=70)
        )
        mock_client.chat.completions.create = AsyncMock(
            return_value=_completion(
                json.dumps(
                    {
                        "confidence": 85,
                        "explanation": "matches question",
                        "suggestion": None,
                    }
                )
            )
        )

        result = await validator.validate("q", "SELECT 1", [{"x": 1}], 1)

        assert result.confidence == 85
        assert result.is_acceptable is True
        assert validator.last_tokens == 50

    @pytest.mark.asyncio
    async def test_low_confidence_not_acceptable(self) -> None:
        """Confidence below the threshold marks the result unacceptable."""
        validator, mock_client = build_validator(
            ValidationConfig(enabled=True, confidence_threshold=70)
        )
        mock_client.chat.completions.create = AsyncMock(
            return_value=_completion(json.dumps({"confidence": 30, "explanation": "weak"}))
        )

        result = await validator.validate("q", "SELECT 1", [{"x": 1}], 1)

        assert result.confidence == 30
        assert result.is_acceptable is False

    @pytest.mark.asyncio
    async def test_invalid_json_falls_back(self) -> None:
        """Unparseable LLM output yields the documented fallback result."""
        validator, mock_client = build_validator()
        mock_client.chat.completions.create = AsyncMock(return_value=_completion("not json at all"))

        result = await validator.validate("q", "SELECT 1", [{"x": 1}], 1)

        assert result.confidence == 60
        assert result.is_acceptable is False

    @pytest.mark.asyncio
    async def test_out_of_range_confidence_clamped(self) -> None:
        """Non-integer or out-of-range confidence is normalized."""
        validator, mock_client = build_validator()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_completion(json.dumps({"confidence": 250, "explanation": "x"}))
        )

        result = await validator.validate("q", "SELECT 1", [{"x": 1}], 1)

        assert 0 <= result.confidence <= 100

    @pytest.mark.asyncio
    async def test_no_usage_means_no_tokens(self) -> None:
        """Missing usage information leaves last_tokens untouched."""
        validator, mock_client = build_validator()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_completion(
                json.dumps({"confidence": 80, "explanation": "ok"}), tokens=None
            )
        )

        await validator.validate("q", "SELECT 1", [{"x": 1}], 1)

        assert validator.last_tokens is None

    @pytest.mark.asyncio
    async def test_llm_error_propagates(self) -> None:
        """API failures raise LLMError."""
        validator, mock_client = build_validator()
        mock_client.chat.completions.create = AsyncMock(side_effect=RuntimeError("api down"))

        with pytest.raises(LLMError):
            await validator.validate("q", "SELECT 1", [{"x": 1}], 1)
