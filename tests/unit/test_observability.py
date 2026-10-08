"""Unit tests for observability modules: metrics, tracing and logging."""

import logging

import pytest

from pg_mcp.observability.logging import (
    JSONFormatter,
    SensitiveDataFilter,
    configure_logging,
    get_logger,
)
from pg_mcp.observability.metrics import MetricsCollector
from pg_mcp.observability.tracing import (
    TracingLogger,
    clear_request_id,
    generate_request_id,
    get_request_id,
    request_context,
    set_request_id,
    trace_async,
)


class TestMetricsCollector:
    """MetricsCollector helper methods."""

    def test_singleton(self) -> None:
        """MetricsCollector returns the same instance."""
        assert MetricsCollector() is MetricsCollector()

    def test_increment_query_request(self) -> None:
        """Query request counter increments without error."""
        collector = MetricsCollector()
        collector.increment_query_request("success", "somedb")

    def test_observe_query_duration(self) -> None:
        """Query duration observation records without error."""
        collector = MetricsCollector()
        collector.observe_query_duration(0.42)

    def test_llm_metrics(self) -> None:
        """LLM call, latency and token helpers record without error."""
        collector = MetricsCollector()
        collector.increment_llm_call("generate_sql")
        collector.observe_llm_latency("generate_sql", 1.5)
        collector.increment_llm_tokens("generate_sql", 120)

    def test_security_and_db_metrics(self) -> None:
        """Rejection, connection and duration helpers record without error."""
        collector = MetricsCollector()
        collector.increment_sql_rejected(reason="blocked_function")
        collector.set_db_connections_active("db", 3)
        collector.observe_db_query_duration(0.05)
        collector.set_schema_cache_age("db", 60.0)


class TestTracing:
    """Request tracing context helpers."""

    def test_generate_request_id_unique(self) -> None:
        """Generated request IDs are unique uuid4 strings."""
        first = generate_request_id()
        second = generate_request_id()
        assert first != second
        assert len(first) == 36

    def test_set_get_clear_request_id(self) -> None:
        """Request ID can be set, read and cleared."""
        clear_request_id()
        assert get_request_id() is None
        set_request_id("abc-123")
        assert get_request_id() == "abc-123"
        clear_request_id()
        assert get_request_id() is None

    @pytest.mark.asyncio
    async def test_request_context_sets_and_restores(self) -> None:
        """request_context propagates the ID and restores the previous value."""
        set_request_id("outer")
        async with request_context("inner") as rid:
            assert rid == "inner"
            assert get_request_id() == "inner"
        assert get_request_id() == "outer"
        clear_request_id()

    @pytest.mark.asyncio
    async def test_request_context_generates_id_when_missing(self) -> None:
        """request_context generates an ID when none is provided."""
        async with request_context() as rid:
            assert rid is not None
            assert get_request_id() == rid
        assert get_request_id() is None

    @pytest.mark.asyncio
    async def test_trace_async_decorator(self) -> None:
        """trace_async runs the wrapped coroutine and restores state."""

        @trace_async(operation="test_op")
        async def do_work() -> str:
            return "done"

        async with request_context("req-1"):
            assert await do_work() == "done"

    def test_tracing_logger_adds_request_id(self, caplog: pytest.LogCaptureFixture) -> None:
        """TracingLogger injects the current request ID into records."""
        logger = TracingLogger("pg_mcp.test_tracing")
        with caplog.at_level(logging.INFO, logger="pg_mcp.test_tracing"):
            set_request_id("req-42")
            logger.info("hello")
            clear_request_id()

        assert len(caplog.records) == 1
        assert caplog.records[0].request_id == "req-42"


class TestLogging:
    """Structured logging configuration."""

    def test_configure_logging_text(self) -> None:
        """configure_logging accepts text format."""
        configure_logging(level="DEBUG", log_format="text", enable_sensitive_filter=True)

    def test_configure_logging_json(self) -> None:
        """configure_logging accepts json format."""
        configure_logging(level="INFO", log_format="json", enable_sensitive_filter=False)

    def test_get_logger(self) -> None:
        """get_logger returns a logger for the module."""
        assert get_logger("pg_mcp.test") is logging.getLogger("pg_mcp.test")

    def test_sensitive_data_filter_masks_passwords(self) -> None:
        """SensitiveDataFilter masks password values in args and extra fields."""
        log_filter = SensitiveDataFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="connecting db",
            args=(),
            exc_info=None,
        )
        record.extra_data = {"password": "hunter2", "host": "db1"}
        log_filter.filter(record)
        assert record.extra_data["password"] == "***REDACTED***"
        assert record.extra_data["host"] == "db1"

    def test_sensitive_data_filter_masks_args(self) -> None:
        """SensitiveDataFilter redacts sensitive keys inside dict args."""
        log_filter = SensitiveDataFilter()
        # A single mapping arg is unwrapped by LogRecord into record.args itself
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="conn info",
            args=({"password": "hunter2"},),
            exc_info=None,
        )
        log_filter.filter(record)
        assert record.args["password"] == "***REDACTED***"

    def test_json_formatter_serializes_extra(self) -> None:
        """JSONFormatter produces JSON with core fields."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="hello",
            args=(),
            exc_info=None,
        )
        import json

        payload = json.loads(formatter.format(record))
        assert payload["level"] == "INFO"
        assert payload["message"] == "hello"
