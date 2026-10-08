"""Unit tests for the newly wired pipeline features.

Covers multi-database executor routing, query/LLM rate limiting, retry
backoff, metrics recording, token accounting, question length enforcement
and request tracing context propagation.
"""

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from pg_mcp.config.settings import ResilienceConfig, ValidationConfig
from pg_mcp.models.errors import SecurityViolationError, SQLParseError
from pg_mcp.models.query import QueryRequest, ResultValidationResult, ReturnType
from pg_mcp.models.schema import DatabaseSchema
from pg_mcp.resilience.rate_limiter import MultiRateLimiter
from pg_mcp.services.orchestrator import QueryOrchestrator


def build_orchestrator(
    pools: dict[str, object] | None = None,
    rate_limiter: object | None = None,
    metrics: object | None = None,
    validation_config: ValidationConfig | None = None,
    resilience_config: ResilienceConfig | None = None,
) -> tuple[QueryOrchestrator, dict[str, object]]:
    """Build an orchestrator with fully mocked pipeline components."""
    mock_schema = DatabaseSchema(database_name="test_db", tables=[], version="15.0")

    mock_generator = AsyncMock()
    mock_generator.last_tokens = 100
    mock_generator.generate.return_value = "SELECT 1;"

    mock_validator = MagicMock()  # sql validator
    mock_validator.validate_or_raise.return_value = None

    pools = pools if pools is not None else {"test_db": MagicMock()}
    mock_executors = {name: AsyncMock() for name in pools}
    for executor in mock_executors.values():
        executor.execute.return_value = ([{"count": 1}], 1)

    mock_cache = MagicMock()
    mock_cache.get.return_value = mock_schema

    mock_result_validator = AsyncMock()
    mock_result_validator.last_tokens = 30
    mock_result_validator.validate.return_value = ResultValidationResult(
        confidence=90,
        explanation="ok",
        suggestion=None,
        is_acceptable=True,
    )

    orchestrator = QueryOrchestrator(
        sql_generator=mock_generator,
        sql_validator=mock_validator,
        sql_executors=mock_executors,
        result_validator=mock_result_validator,
        schema_cache=mock_cache,
        pools=pools,
        resilience_config=resilience_config or ResilienceConfig(),
        validation_config=validation_config or ValidationConfig(enabled=False),
        rate_limiter=rate_limiter,
        metrics=metrics,
    )
    mocks = {
        "generator": mock_generator,
        "validator": mock_validator,
        "executors": mock_executors,
        "cache": mock_cache,
        "result_validator": mock_result_validator,
    }
    return orchestrator, mocks


class TestMultiDatabaseRouting:
    """Executor routing across multiple configured databases."""

    @pytest.mark.asyncio
    async def test_routes_executor_per_database(self) -> None:
        """SQL execution goes to the executor of the requested database."""
        pools = {"db1": MagicMock(), "db2": MagicMock()}
        orchestrator, mocks = build_orchestrator(pools=pools)

        response = await orchestrator.execute_query(
            QueryRequest(question="Count", database="db2", return_type=ReturnType.RESULT)
        )

        assert response.success is True
        mocks["executors"]["db2"].execute.assert_called_once()
        mocks["executors"]["db1"].execute.assert_not_called()
        mocks["cache"].get.assert_called_with("db2")

    @pytest.mark.asyncio
    async def test_missing_executor_raises_database_error(self) -> None:
        """A database with a pool but no executor produces a database error."""
        pools = {"db1": MagicMock(), "db2": MagicMock()}
        orchestrator, mocks = build_orchestrator(pools=pools)
        # Simulate registration gap: db2 has a pool but no executor
        orchestrator.sql_executors = {"db1": mocks["executors"]["db1"]}

        response = await orchestrator.execute_query(
            QueryRequest(question="Count", database="db2", return_type=ReturnType.RESULT)
        )

        assert response.success is False
        assert response.error is not None
        assert response.error.code == "database_error"


class TestRateLimiting:
    """Query and LLM concurrency slot enforcement."""

    @pytest.mark.asyncio
    async def test_query_rate_limit_rejects(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When query slots are exhausted the request is rate limited."""
        import pg_mcp.services.orchestrator as orch_module

        monkeypatch.setattr(orch_module, "QUERY_SLOT_TIMEOUT_S", 0.05)
        limiter = MultiRateLimiter(query_limit=1, llm_limit=5)
        assert await limiter.query_limiter.acquire(timeout=0.01)  # exhaust the slot

        orchestrator, _mocks = build_orchestrator(rate_limiter=limiter)
        response = await orchestrator.execute_query(
            QueryRequest(question="Count", database="test_db", return_type=ReturnType.RESULT)
        )

        assert response.success is False
        assert response.error is not None
        assert response.error.code == "rate_limit_exceeded"

    @pytest.mark.asyncio
    async def test_llm_rate_limit_rejects(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When LLM slots are exhausted generation is rate limited."""
        import pg_mcp.services.orchestrator as orch_module

        monkeypatch.setattr(orch_module, "LLM_SLOT_TIMEOUT_S", 0.05)
        limiter = MultiRateLimiter(query_limit=5, llm_limit=1)
        assert await limiter.llm_limiter.acquire(timeout=0.01)  # exhaust the slot

        orchestrator, _mocks = build_orchestrator(rate_limiter=limiter)
        response = await orchestrator.execute_query(
            QueryRequest(question="Count", database="test_db", return_type=ReturnType.RESULT)
        )

        assert response.success is False
        assert response.error is not None
        assert response.error.code == "rate_limit_exceeded"

    @pytest.mark.asyncio
    async def test_slot_released_after_request(self) -> None:
        """The query slot is released when the pipeline finishes."""
        limiter = MultiRateLimiter(query_limit=1, llm_limit=5)
        orchestrator, _mocks = build_orchestrator(rate_limiter=limiter)

        for _ in range(3):
            response = await orchestrator.execute_query(
                QueryRequest(question="Count", database="test_db", return_type=ReturnType.RESULT)
            )
            assert response.success is True

        # The single slot must be free again: acquiring it succeeds immediately.
        assert await limiter.query_limiter.acquire(timeout=0.01)
        limiter.query_limiter.release()


class TestRequestValidation:
    """Pre-pipeline request checks."""

    @pytest.mark.asyncio
    async def test_question_too_long_rejected(self) -> None:
        """Questions over the configured max length are rejected."""
        config = ValidationConfig(max_question_length=10)
        orchestrator, _mocks = build_orchestrator(validation_config=config)

        response = await orchestrator.execute_query(
            QueryRequest(question="a" * 11, database="test_db", return_type=ReturnType.RESULT)
        )

        assert response.success is False
        assert response.error is not None
        assert response.error.code == "question_too_long"


class TestRetryBackoff:
    """Exponential backoff between SQL generation retries."""

    @pytest.mark.asyncio
    async def test_backoff_between_retries(self) -> None:
        """Retry waits retry_delay * backoff_factor ** attempt between attempts."""
        resilience = ResilienceConfig(max_retries=2, retry_delay=0.1, backoff_factor=2.0)
        orchestrator, mocks = build_orchestrator(resilience_config=resilience)
        # Fail validation twice, succeed on third attempt
        mocks["validator"].validate_or_raise.side_effect = [
            SQLParseError("attempt 1 bad"),
            SQLParseError("attempt 2 bad"),
            None,
        ]

        start = time.monotonic()
        response = await orchestrator.execute_query(
            QueryRequest(question="Count", database="test_db", return_type=ReturnType.RESULT)
        )
        elapsed = time.monotonic() - start

        assert response.success is True
        # sleeps: 0.1 (after attempt 0) + 0.2 (after attempt 1) = 0.3s minimum
        assert elapsed >= 0.3
        assert mocks["generator"].generate.call_count == 3


class TestMetricsRecording:
    """Prometheus metrics emitted from the request pipeline."""

    @pytest.mark.asyncio
    async def test_metrics_recorded_on_success(self) -> None:
        """A successful pipeline records query, LLM and duration metrics."""
        metrics = MagicMock()
        orchestrator, _mocks = build_orchestrator(
            metrics=metrics, validation_config=ValidationConfig(enabled=True)
        )

        response = await orchestrator.execute_query(
            QueryRequest(question="Count", database="test_db", return_type=ReturnType.RESULT)
        )
        assert response.success is True

        metrics.increment_query_request.assert_called_once_with("success", "test_db")
        metrics.observe_query_duration.assert_called_once()
        metrics.increment_llm_call.assert_any_call("generate_sql")
        metrics.increment_llm_call.assert_any_call("validate_result")
        metrics.observe_db_query_duration.assert_called_once()
        metrics.increment_llm_tokens.assert_any_call("generate_sql", 100)
        metrics.increment_llm_tokens.assert_any_call("validate_result", 30)

    @pytest.mark.asyncio
    async def test_metrics_recorded_on_security_violation(self) -> None:
        """Security violations record the rejection metric and error status."""
        metrics = MagicMock()
        orchestrator, mocks = build_orchestrator(metrics=metrics)
        mocks["validator"].validate_or_raise.side_effect = SecurityViolationError(
            "DELETE not allowed"
        )
        orchestrator.resilience_config = ResilienceConfig(max_retries=0, retry_delay=0.1)

        response = await orchestrator.execute_query(
            QueryRequest(question="Delete", database="test_db", return_type=ReturnType.RESULT)
        )

        assert response.success is False
        metrics.increment_sql_rejected.assert_called_once_with(reason="security_violation")
        metrics.increment_query_request.assert_called_once_with("error", "test_db")


class TestTokenAccounting:
    """LLM token usage propagation."""

    @pytest.mark.asyncio
    async def test_tokens_used_accumulated(self) -> None:
        """tokens_used sums generation and validation token usage."""
        orchestrator, _mocks = build_orchestrator(validation_config=ValidationConfig(enabled=True))

        response = await orchestrator.execute_query(
            QueryRequest(question="Count", database="test_db", return_type=ReturnType.RESULT)
        )

        assert response.tokens_used == 130  # 100 generation + 30 validation


class TestTracingContext:
    """Request ID propagation through the pipeline."""

    @pytest.mark.asyncio
    async def test_request_context_propagated_to_llm_calls(self) -> None:
        """LLM calls execute inside the request tracing context."""
        from pg_mcp.observability.tracing import get_request_id

        captured: list[str | None] = []

        async def fake_generate(**kwargs: object) -> str:
            captured.append(get_request_id())
            return "SELECT 1;"

        orchestrator, mocks = build_orchestrator()
        mocks["generator"].generate = AsyncMock(side_effect=fake_generate)

        response = await orchestrator.execute_query(
            QueryRequest(question="Count", database="test_db", return_type=ReturnType.RESULT)
        )

        assert response.success is True
        assert captured and captured[0] is not None and len(captured[0]) == 36  # uuid4
