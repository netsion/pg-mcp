"""Integration tests for security enforcement, multi-database routing,
metrics output and rate limiting against a real PostgreSQL server.

The LLM generator is stubbed with canned SQL so the tests are fast and
deterministic; every other component (pools, executors, validator,
orchestrator, metrics, rate limiter) is real.

Required environment (provided by tests/integration/conftest.py via .env):
DATABASE_* pointing at the primary database, and DEMO2_DATABASE_NAME=demo2
for the secondary-database routing test (set automatically below).
"""

import asyncio
from unittest.mock import AsyncMock

import pytest
from prometheus_client import REGISTRY

from pg_mcp.cache.schema_cache import SchemaCache
from pg_mcp.config.settings import ResilienceConfig, Settings, load_extra_databases
from pg_mcp.db.pool import create_pools
from pg_mcp.models.query import QueryRequest, ReturnType
from pg_mcp.observability.metrics import MetricsCollector
from pg_mcp.resilience.rate_limiter import MultiRateLimiter
from pg_mcp.services.orchestrator import QueryOrchestrator
from pg_mcp.services.result_validator import ResultValidator
from pg_mcp.services.sql_executor import SQLExecutor
from pg_mcp.services.sql_validator import SQLValidator

pytestmark = pytest.mark.integration

COUNT_TABLES_SQL = (
    "SELECT COUNT(*) AS table_count FROM information_schema.tables WHERE table_schema = 'public';"
)


def metric_value(name: str, **labels: str) -> float:
    """Read a labeled Prometheus sample from the default registry."""
    return REGISTRY.get_sample_value(name, labels) or 0.0


@pytest.fixture
def integration_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure a secondary database plus a strict security policy."""
    monkeypatch.setenv("DEMO2_DATABASE_NAME", "demo2")
    monkeypatch.setenv("SECURITY_BLOCKED_TABLES", "user_sessions")
    monkeypatch.setenv("SECURITY_BLOCKED_COLUMNS", "password,email")
    monkeypatch.setenv("SECURITY_ALLOW_EXPLAIN", "true")


class StubGenerator:
    """Deterministic stand-in for the LLM generator.

    Returns canned SQL keyed by a marker contained in the question.
    """

    def __init__(self, sql_by_marker: dict[str, str]) -> None:
        self.sql_by_marker = sql_by_marker
        self.last_tokens = None

    async def generate(self, question: str, **_kwargs: object) -> str:
        for marker, sql in self.sql_by_marker.items():
            if marker in question:
                return sql
        raise AssertionError(f"no canned SQL for question: {question}")


async def build_real_orchestrator(
    generator: object | None = None,
    rate_limiter: object | None = None,
    metrics: object | None = None,
) -> tuple[QueryOrchestrator, Settings]:
    """Build an orchestrator with real pools/executors/validator."""
    settings = Settings()
    extras = load_extra_databases(settings.database)
    pools = await create_pools([settings.database, *extras])

    cache = SchemaCache(settings.cache)
    for name, pool in pools.items():
        await cache.load(name, pool)

    configs = [settings.database, *extras]
    executors = {
        config.name: SQLExecutor(
            pool=pools[config.name], security_config=settings.security, db_config=config
        )
        for config in configs
    }

    orchestrator = QueryOrchestrator(
        sql_generator=generator or StubGenerator({"count": COUNT_TABLES_SQL}),
        sql_validator=SQLValidator(
            config=settings.security,
            blocked_tables=settings.security.blocked_tables,
            blocked_columns=settings.security.blocked_columns,
            allow_explain=settings.security.allow_explain,
        ),
        sql_executors=executors,
        result_validator=ResultValidator(settings.openai, settings.validation),
        schema_cache=cache,
        pools=pools,
        resilience_config=settings.resilience,
        validation_config=settings.validation,
        rate_limiter=rate_limiter,
        metrics=metrics,
    )
    return orchestrator, settings


class TestSecurityEnforcementReal:
    """Security policy enforced through the full orchestrator pipeline."""

    @pytest.mark.asyncio
    async def test_wildcard_blocked_column_rejected(
        self, integration_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SELECT * must not leak columns protected by SECURITY_BLOCKED_COLUMNS."""
        generator = StubGenerator({"stars": "SELECT * FROM users;"})
        orchestrator, _settings = await build_real_orchestrator(generator=generator)
        orchestrator.resilience_config = ResilienceConfig(max_retries=1, retry_delay=0.1)

        response = await orchestrator.execute_query(
            QueryRequest(question="stars", database="demo2", return_type=ReturnType.RESULT)
        )

        assert response.success is False
        assert response.error is not None
        assert response.error.code == "security_violation"
        assert "*" in response.error.message

    @pytest.mark.asyncio
    async def test_explain_bypass_rejected(self, integration_env: None) -> None:
        """EXPLAIN ANALYZE over writes / blocked functions must be rejected."""
        generator = StubGenerator(
            {
                "analyze-delete": "EXPLAIN ANALYZE DELETE FROM users;",
                "analyze-sleep": "EXPLAIN ANALYZE SELECT pg_sleep(10);",
            }
        )
        orchestrator, _settings = await build_real_orchestrator(generator=generator)
        orchestrator.resilience_config = ResilienceConfig(max_retries=1, retry_delay=0.1)

        for marker in ("analyze-delete", "analyze-sleep"):
            response = await orchestrator.execute_query(
                QueryRequest(question=marker, database="demo2", return_type=ReturnType.RESULT)
            )
            assert response.success is False, f"{marker} should be rejected"
            assert response.error is not None
            assert response.error.code in ("security_violation", "sql_parse_error")

    @pytest.mark.asyncio
    async def test_blocked_table_rejected(self, integration_env: None) -> None:
        """Blocked tables are rejected through the pipeline."""
        generator = StubGenerator({"sessions": "SELECT COUNT(*) FROM user_sessions;"})
        orchestrator, _settings = await build_real_orchestrator(generator=generator)
        orchestrator.resilience_config = ResilienceConfig(max_retries=1, retry_delay=0.1)

        response = await orchestrator.execute_query(
            QueryRequest(question="sessions", database="demo2", return_type=ReturnType.RESULT)
        )
        assert response.success is False
        assert response.error is not None
        assert response.error.code == "security_violation"


class TestMultiDatabaseRoutingReal:
    """Queries route to and execute on the requested database."""

    @pytest.mark.asyncio
    async def test_query_routes_to_secondary_database(self, integration_env: None) -> None:
        """The same question returns different data on primary vs demo2."""
        orchestrator, settings = await build_real_orchestrator()

        results: dict[str, int] = {}
        for db in (settings.database.name, "demo2"):
            response = await orchestrator.execute_query(
                QueryRequest(question="count tables", database=db, return_type=ReturnType.RESULT)
            )
            assert response.success is True, f"{db}: {response.error}"
            results[db] = response.data.rows[0]["table_count"]

        # blog_small has 10 public tables/views, demo2 is empty: if both
        # numbers differ, execution really happened on two databases.
        assert results[settings.database.name] > 0
        assert results["demo2"] == 0

    @pytest.mark.asyncio
    async def test_unknown_database_lists_available(self, integration_env: None) -> None:
        """An unknown database name reports the configured databases."""
        orchestrator, _settings = await build_real_orchestrator()

        response = await orchestrator.execute_query(
            QueryRequest(question="count tables", database="nope", return_type=ReturnType.RESULT)
        )
        assert response.success is False
        assert response.error is not None
        assert response.error.code == "database_error"
        assert "demo2" in str(response.error.details.get("available_databases"))


class TestMetricsAndRateLimitReal:
    """Metrics emission and concurrency limiting through real components."""

    @pytest.mark.asyncio
    async def test_prometheus_metrics_recorded(self, integration_env: None) -> None:
        """Successful requests update labeled Prometheus counters."""
        from dotenv import load_dotenv

        load_dotenv()
        metrics = MetricsCollector()
        before = metric_value("pg_mcp_query_requests_total", status="success", database="demo2")

        orchestrator, _settings = await build_real_orchestrator(metrics=metrics)
        response = await orchestrator.execute_query(
            QueryRequest(question="count tables", database="demo2", return_type=ReturnType.RESULT)
        )
        assert response.success is True

        after = metric_value("pg_mcp_query_requests_total", status="success", database="demo2")
        assert after == before + 1

    @pytest.mark.asyncio
    async def test_rate_limit_under_concurrency(
        self, integration_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With one query slot, concurrent requests beyond it are rejected."""
        import pg_mcp.services.orchestrator as orch_module

        monkeypatch.setattr(orch_module, "QUERY_SLOT_TIMEOUT_S", 0.1)

        slow_executor = AsyncMock()

        async def slow_execute(_sql: str) -> tuple[list[dict], int]:
            await asyncio.sleep(0.5)
            return ([{"count": 1}], 1)

        slow_executor.execute = AsyncMock(side_effect=slow_execute)

        orchestrator, _settings = await build_real_orchestrator(
            rate_limiter=MultiRateLimiter(query_limit=1, llm_limit=5)
        )
        # Replace executors with the slow stub for this test only
        orchestrator.sql_executors = dict.fromkeys(orchestrator.sql_executors, slow_executor)

        responses = await asyncio.gather(
            *[
                orchestrator.execute_query(
                    QueryRequest(
                        question="count tables",
                        database="demo2",
                        return_type=ReturnType.RESULT,
                    )
                )
                for _ in range(3)
            ]
        )

        outcomes = sorted(
            "ok" if r.success else (r.error.code if r.error else "?") for r in responses
        )
        assert outcomes.count("ok") == 1
        assert outcomes.count("rate_limit_exceeded") == 2


class TestMainEntryDelegation:
    """main.py must start the real server, not an unrelated demo."""

    def test_main_module_delegates(self) -> None:
        """main.py imports the real server entry point."""
        import main as main_module

        assert hasattr(main_module, "main")
        assert main_module.main.__module__ == "pg_mcp.__main__"
