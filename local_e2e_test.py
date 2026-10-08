"""Local end-to-end test for pg-mcp without an MCP client.

Replicates the server lifespan wiring, then sends a natural-language question
through the full pipeline: LLM SQL generation -> security validation ->
execution on PostgreSQL -> LLM result validation.
"""

import asyncio
import json
import os

from dotenv import load_dotenv

from pg_mcp.cache.schema_cache import SchemaCache
from pg_mcp.config.settings import Settings
from pg_mcp.db.pool import close_pools, create_pool
from pg_mcp.models.query import QueryRequest, ReturnType
from pg_mcp.resilience.rate_limiter import MultiRateLimiter
from pg_mcp.services.orchestrator import QueryOrchestrator
from pg_mcp.services.result_validator import ResultValidator
from pg_mcp.services.sql_executor import SQLExecutor
from pg_mcp.services.sql_generator import SQLGenerator
from pg_mcp.services.sql_validator import SQLValidator

QUESTION = "How many users are there in total, and how many posts have been published?"


async def main() -> None:
    # Standalone script: load the project .env the same way the server entry
    # point does, before Settings is instantiated. Skip the metrics server.
    load_dotenv()
    os.environ.setdefault("OBSERVABILITY_METRICS_ENABLED", "false")

    settings = Settings()
    print(
        f"[1] 配置加载: db={settings.database.name} @ {settings.database.host}, "
        f"model={settings.openai.model}"
    )

    pool = await create_pool(settings.database)
    pools = {settings.database.name: pool}
    print("[2] 数据库连接池已创建")

    schema_cache = SchemaCache(settings.cache)
    schema = await schema_cache.load(settings.database.name, pool)
    print(f"[3] Schema 缓存已加载: {len(schema.tables)} 张表")

    orchestrator = QueryOrchestrator(
        sql_generator=SQLGenerator(settings.openai),
        sql_validator=SQLValidator(
            config=settings.security,
            blocked_tables=settings.security.blocked_tables,
            blocked_columns=settings.security.blocked_columns,
            allow_explain=settings.security.allow_explain,
        ),
        sql_executors={
            settings.database.name: SQLExecutor(
                pool=pool,
                security_config=settings.security,
                db_config=settings.database,
            )
        },
        result_validator=ResultValidator(
            openai_config=settings.openai,
            validation_config=settings.validation,
        ),
        schema_cache=schema_cache,
        pools=pools,
        resilience_config=settings.resilience,
        validation_config=settings.validation,
        rate_limiter=MultiRateLimiter(
            query_limit=settings.resilience.query_concurrency_limit,
            llm_limit=settings.resilience.llm_concurrency_limit,
        ),
    )
    print("[4] Orchestrator ready, querying (calls the LLM)...\n")

    response = await orchestrator.execute_query(
        QueryRequest(
            question=QUESTION,
            database=settings.database.name,
            return_type=ReturnType("result"),
        )
    )
    print(f"问题: {QUESTION}")
    print(json.dumps(response.model_dump(), default=str, indent=2, ensure_ascii=False))

    await close_pools(pools)


if __name__ == "__main__":
    asyncio.run(main())
