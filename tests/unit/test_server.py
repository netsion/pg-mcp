"""Unit tests for the MCP server query tool."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from pg_mcp import server
from pg_mcp.models.query import (
    ErrorDetail,
    QueryRequest,
    QueryResponse,
    QueryResult,
)


def _success_response() -> QueryResponse:
    """Build a typical successful orchestrator response."""
    return QueryResponse(
        success=True,
        generated_sql="SELECT 1;",
        validation=None,
        data=QueryResult(
            columns=["?column?"],
            rows=[{"?column?": 1}],
            row_count=1,
            execution_time_ms=1.5,
        ),
        error=None,
        confidence=95,
        tokens_used=None,
    )


@pytest.fixture
def mock_orchestrator(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch the server-level orchestrator global with a mock."""
    orchestrator = MagicMock()
    orchestrator.execute_query = AsyncMock(return_value=_success_response())
    monkeypatch.setattr(server, "_orchestrator", orchestrator)
    return orchestrator


class TestQueryTool:
    """The MCP query tool wrapper."""

    @pytest.mark.asyncio
    async def test_query_success(self, mock_orchestrator: MagicMock) -> None:
        """Successful responses are returned as plain dicts with tokens_used."""
        result = await server.query(question="How many users?", database="db")

        assert result["success"] is True
        assert result["generated_sql"] == "SELECT 1;"
        assert result["tokens_used"] == 0  # coerced from None
        assert result["confidence"] == 95

        request: QueryRequest = mock_orchestrator.execute_query.await_args.args[0]
        assert request.question == "How many users?"
        assert request.database == "db"
        assert request.return_type == "result"

    @pytest.mark.asyncio
    async def test_query_not_initialized(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Before lifespan initialization the tool reports SERVER_NOT_INITIALIZED."""
        monkeypatch.setattr(server, "_orchestrator", None)
        result = await server.query(question="anything")

        assert result["success"] is False
        assert result["error"]["code"] == "SERVER_NOT_INITIALIZED"

    @pytest.mark.asyncio
    async def test_query_invalid_return_type(self, mock_orchestrator: MagicMock) -> None:
        """Unknown return_type values are rejected before execution."""
        result = await server.query(question="q", return_type="csv")

        assert result["success"] is False
        assert result["error"]["code"] == "INVALID_PARAMETER"
        mock_orchestrator.execute_query.assert_not_called()

    @pytest.mark.asyncio
    async def test_query_invalid_question(self, mock_orchestrator: MagicMock) -> None:
        """Empty questions fail QueryRequest validation with INVALID_REQUEST."""
        result = await server.query(question="   ")

        assert result["success"] is False
        assert result["error"]["code"] == "INVALID_REQUEST"
        mock_orchestrator.execute_query.assert_not_called()

    @pytest.mark.asyncio
    async def test_query_error_response(
        self, mock_orchestrator: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Failed orchestrator responses serialize the error detail."""
        error_response = QueryResponse(
            success=False,
            generated_sql=None,
            validation=None,
            data=None,
            error=ErrorDetail(code="database_error", message="boom", details={"db": "x"}),
            confidence=0,
            tokens_used=None,
        )
        mock_orchestrator.execute_query = AsyncMock(return_value=error_response)

        result = await server.query(question="q")

        assert result["success"] is False
        assert result["error"]["code"] == "database_error"
        assert result["error"]["message"] == "boom"

    @pytest.mark.asyncio
    async def test_query_orchestrator_exception(self, mock_orchestrator: MagicMock) -> None:
        """Unexpected exceptions surface as INTERNAL_ERROR without raising."""
        mock_orchestrator.execute_query = AsyncMock(side_effect=RuntimeError("kaboom"))

        result = await server.query(question="q")

        assert result["success"] is False
        assert result["error"]["code"] == "INTERNAL_ERROR"


class TestLifespan:
    """Server lifespan wiring (mocked external resources)."""

    @pytest.mark.asyncio
    async def test_lifespan_initializes_and_shuts_down(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Lifespan builds pools, executors and orchestrator, then cleans up."""
        mock_pool = MagicMock()

        async def fake_create_pools(configs):
            assert len(configs) >= 1
            return {configs[0].name: mock_pool}

        closed: list[bool] = []

        async def fake_close_pools(pools, **_kwargs):
            closed.append(True)

        schema_stub = MagicMock()
        schema_stub.tables = [MagicMock()]

        cache_instance = MagicMock()
        cache_instance.load = AsyncMock(return_value=schema_stub)

        monkeypatch.setattr(server, "create_pools", fake_create_pools)
        monkeypatch.setattr(server, "close_pools", fake_close_pools)
        monkeypatch.setattr(server, "SchemaCache", MagicMock(return_value=cache_instance))
        monkeypatch.setattr(server, "MetricsCollector", MagicMock())
        monkeypatch.setenv("OBSERVABILITY_METRICS_ENABLED", "false")
        monkeypatch.setenv("DATABASE_NAME", "lifespan_db")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-lifespan-test")

        async def run() -> None:
            async with server.lifespan(MagicMock()):
                assert server._orchestrator is not None
                assert server._pools is not None
                assert "lifespan_db" in server._pools

        await run()

        assert closed == [True]  # shutdown closed the pools


class TestReturnTypeImport:
    """Guards against accidental import breakage of the tool module."""

    def test_module_exports(self) -> None:
        """server module keeps its public surface."""
        assert hasattr(server, "query")
        assert hasattr(server, "lifespan")
        assert hasattr(server, "mcp")
