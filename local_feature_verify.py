"""One-shot verification of the new assignment features against real infra.

1. Security controls: Settings -> SQLValidator wiring rejects blocked tables
   and EXPLAIN (no LLM involved).
2. Multi-database: SECONDARY_DATABASE_NAME creates a second pool/executor;
   a real LLM query routes to it and executes there.
Run: uv run python local_feature_verify.py
"""

import asyncio
import os

from dotenv import load_dotenv

from pg_mcp.cache.schema_cache import SchemaCache
from pg_mcp.config.settings import Settings, load_extra_databases
from pg_mcp.db.pool import close_pools, create_pools
from pg_mcp.models.errors import SecurityViolationError
from pg_mcp.models.query import QueryRequest, ReturnType
from pg_mcp.resilience.rate_limiter import MultiRateLimiter
from pg_mcp.services.orchestrator import QueryOrchestrator
from pg_mcp.services.result_validator import ResultValidator
from pg_mcp.services.sql_executor import SQLExecutor
from pg_mcp.services.sql_generator import SQLGenerator
from pg_mcp.services.sql_validator import SQLValidator


def check_security() -> None:
    load_dotenv()
    os.environ.setdefault("OBSERVABILITY_METRICS_ENABLED", "false")
    # Security policy for this verification run
    os.environ["SECURITY_BLOCKED_TABLES"] = "user_sessions"
    os.environ["SECURITY_ALLOW_EXPLAIN"] = "false"

    settings = Settings()
    validator = SQLValidator(
        config=settings.security,
        blocked_tables=settings.security.blocked_tables,
        blocked_columns=settings.security.blocked_columns,
        allow_explain=settings.security.allow_explain,
    )
    print(f"[security] blocked_tables={validator.blocked_tables}")

    try:
        validator.validate_or_raise("SELECT * FROM user_sessions LIMIT 1")
        raise SystemExit("FAIL: blocked table query was allowed")
    except SecurityViolationError as e:
        print(f"[security] blocked table rejected: {str(e)[:80]}")

    try:
        validator.validate_or_raise("EXPLAIN SELECT 1")
        raise SystemExit("FAIL: EXPLAIN was allowed")
    except SecurityViolationError as e:
        print(f"[security] EXPLAIN rejected: {str(e)[:80]}")

    validator.validate_or_raise("SELECT COUNT(*) FROM users")
    print("[security] normal query passes")


async def check_multi_db() -> None:
    settings = Settings()
    extras = load_extra_databases(settings.database)
    print(f"[multi-db] extra databases: {[db.name for db in extras]}")

    pools = await create_pools([settings.database, *extras])
    cache = SchemaCache(settings.cache)
    for name, pool in pools.items():
        await cache.load(name, pool)

    orchestrator = QueryOrchestrator(
        sql_generator=SQLGenerator(settings.openai),
        sql_validator=SQLValidator(
            config=settings.security,
            blocked_tables=settings.security.blocked_tables,
            blocked_columns=settings.security.blocked_columns,
            allow_explain=settings.security.allow_explain,
        ),
        sql_executors={
            name: SQLExecutor(pool=pool, security_config=settings.security, db_config=config)
            for config in [settings.database, *extras]
            for name, pool in [(config.name, pools[config.name])]
        },
        result_validator=ResultValidator(settings.openai, settings.validation),
        schema_cache=cache,
        pools=pools,
        resilience_config=settings.resilience,
        validation_config=settings.validation,
        rate_limiter=MultiRateLimiter(query_limit=5, llm_limit=5),
    )

    # Ask a question only answerable per-database; route to the secondary
    secondary = extras[0].name
    response = await orchestrator.execute_query(
        QueryRequest(
            question="How many tables are there in the public schema? Reply with a single count.",
            database=secondary,
            return_type=ReturnType.RESULT,
        )
    )
    print(f"[multi-db] query on '{secondary}': success={response.success}")
    if response.success:
        print(f"[multi-db] sql={response.generated_sql}")
        print(f"[multi-db] rows={response.data.rows} tokens_used={response.tokens_used}")
    else:
        print(f"[multi-db] error={response.error}")

    await close_pools(pools)


async def main() -> None:
    check_security()
    await check_multi_db()


if __name__ == "__main__":
    asyncio.run(main())
