"""Query orchestrator for coordinating the complete query flow.

This module provides the QueryOrchestrator class that coordinates all components
of the query processing pipeline: SQL generation, validation, execution, and result
validation. It implements retry logic with exponential backoff, rate limiting,
circuit breaking, metrics recording, and request tracing.
"""

import asyncio
import functools
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from asyncpg import Pool

from pg_mcp.cache.schema_cache import SchemaCache
from pg_mcp.config.settings import ResilienceConfig, ValidationConfig
from pg_mcp.models.errors import (
    DatabaseError,
    ErrorCode,
    LLMError,
    PgMcpError,
    RateLimitExceededError,
    SchemaLoadError,
    SecurityViolationError,
    SQLParseError,
)
from pg_mcp.models.query import (
    ErrorDetail,
    QueryRequest,
    QueryResponse,
    QueryResult,
    ReturnType,
    ValidationResult,
)
from pg_mcp.observability.metrics import MetricsCollector
from pg_mcp.observability.tracing import request_context
from pg_mcp.resilience.circuit_breaker import CircuitBreaker
from pg_mcp.resilience.rate_limiter import MultiRateLimiter
from pg_mcp.services.result_validator import ResultValidator
from pg_mcp.services.sql_executor import SQLExecutor
from pg_mcp.services.sql_generator import SQLGenerator
from pg_mcp.services.sql_validator import SQLValidator

logger = logging.getLogger(__name__)

# How long a request waits for a concurrency slot before being rejected.
QUERY_SLOT_TIMEOUT_S = 5.0
LLM_SLOT_TIMEOUT_S = 30.0


class QueryOrchestrator:
    """Orchestrates the complete query processing pipeline.

    This class coordinates SQL generation, validation, execution, and result
    validation across multiple databases. It implements retry logic with
    exponential backoff and error feedback, circuit breaker and rate limiting
    for fault tolerance, Prometheus metrics for observability, and request
    tracing for end-to-end diagnostics.

    Example:
        >>> orchestrator = QueryOrchestrator(
        ...     sql_generator=generator,
        ...     sql_validator=validator,
        ...     sql_executors={"mydb": executor},
        ...     result_validator=result_validator,
        ...     schema_cache=cache,
        ...     pools={"mydb": pool},
        ...     resilience_config=resilience_config,
        ...     validation_config=validation_config,
        ...     rate_limiter=MultiRateLimiter(query_limit=10, llm_limit=5),
        ...     metrics=MetricsCollector(),
        ... )
        >>> response = await orchestrator.execute_query(QueryRequest(
        ...     question="How many users?",
        ...     database="mydb"
        ... ))
    """

    def __init__(
        self,
        sql_generator: SQLGenerator,
        sql_validator: SQLValidator,
        sql_executors: dict[str, SQLExecutor],
        result_validator: ResultValidator,
        schema_cache: SchemaCache,
        pools: dict[str, Pool],
        resilience_config: ResilienceConfig,
        validation_config: ValidationConfig,
        rate_limiter: MultiRateLimiter | None = None,
        metrics: MetricsCollector | None = None,
    ) -> None:
        """Initialize query orchestrator.

        Args:
            sql_generator: SQL generation service.
            sql_validator: SQL validation service.
            sql_executors: Mapping of database name to execution service.
            result_validator: Result validation service.
            schema_cache: Schema cache instance.
            pools: Dictionary mapping database names to connection pools.
            resilience_config: Resilience configuration for retries and circuit breaker.
            validation_config: Validation configuration including thresholds.
            rate_limiter: Optional concurrency limiter for queries and LLM calls.
            metrics: Optional metrics collector for observability.
        """
        self.sql_generator = sql_generator
        self.sql_validator = sql_validator
        self.sql_executors = sql_executors
        self.result_validator = result_validator
        self.schema_cache = schema_cache
        self.pools = pools
        self.resilience_config = resilience_config
        self.validation_config = validation_config
        self.rate_limiter = rate_limiter
        self.metrics = metrics

        # Create circuit breaker for LLM calls
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=resilience_config.circuit_breaker_threshold,
            recovery_timeout=resilience_config.circuit_breaker_timeout,
        )

    async def execute_query(self, request: QueryRequest) -> QueryResponse:
        """Execute complete query flow from question to results.

        This method orchestrates the entire pipeline under a query concurrency
        slot and a tracing context:
        1. Validate question length and resolve database name
        2. Acquire a query rate-limit slot
        3. Load schema from cache
        4. Generate and validate SQL with retry + backoff
        5. Execute SQL (if return_type == RESULT)
        6. Validate results (optional)
        7. Record metrics and return structured response

        Args:
            request: Query request containing question and parameters.

        Returns:
            QueryResponse: Complete response with SQL, results, or error information.
        """
        request_id = str(uuid.uuid4())
        start_time_s = time.monotonic()
        database_label = request.database or "unknown"
        status = "error"

        async with request_context(request_id):
            logger.info(
                "Starting query execution",
                extra={"request_id": request_id, "question": request.question[:100]},
            )

            try:
                # Step 0: Enforce configured question length limit
                max_length = self.validation_config.max_question_length
                if len(request.question) > max_length:
                    raise PgMcpError(
                        message=(
                            f"Question exceeds maximum length of {max_length} characters "
                            f"(got {len(request.question)})"
                        ),
                        code=ErrorCode.QUESTION_TOO_LONG,
                        details={
                            "max_length": max_length,
                            "actual_length": len(request.question),
                        },
                    )

                # Step 1: Resolve database name
                database_name = self._resolve_database(request.database)
                database_label = database_name
                logger.debug(
                    "Resolved database",
                    extra={"request_id": request_id, "database": database_name},
                )

                # Step 2: Acquire a query concurrency slot for the pipeline
                if self.rate_limiter is not None:
                    acquired = await self.rate_limiter.query_limiter.acquire(
                        timeout=QUERY_SLOT_TIMEOUT_S
                    )
                    if not acquired:
                        raise RateLimitExceededError(
                            message=(
                                "Too many concurrent queries; "
                                f"no slot available within {QUERY_SLOT_TIMEOUT_S}s"
                            ),
                            details={"timeout": QUERY_SLOT_TIMEOUT_S},
                        )
                try:
                    response = await self._execute_pipeline(request, request_id, database_name)
                finally:
                    if self.rate_limiter is not None:
                        self.rate_limiter.query_limiter.release()

                status = "success" if response.success else "error"
                return response

            except SecurityViolationError as e:
                if self.metrics is not None:
                    self.metrics.increment_sql_rejected(reason=e.code.value)
                return self._build_error_response(request_id, e)
            except PgMcpError as e:
                return self._build_error_response(request_id, e)
            except Exception as e:
                logger.exception(
                    "Query execution failed with unexpected error",
                    extra={"request_id": request_id},
                )
                return self._build_error_response(
                    request_id,
                    PgMcpError(
                        message=f"Internal server error: {e!s}",
                        code=ErrorCode.INTERNAL_ERROR,
                        details={"error_type": type(e).__name__},
                    ),
                )
            finally:
                if self.metrics is not None:
                    self.metrics.increment_query_request(status, database_label)
                    self.metrics.observe_query_duration(time.monotonic() - start_time_s)

    async def _execute_pipeline(
        self,
        request: QueryRequest,
        request_id: str,
        database_name: str,
    ) -> QueryResponse:
        """Run the generation/execution pipeline for an already-resolved database."""
        # Step 3: Get schema from cache
        schema = self.schema_cache.get(database_name)
        if schema is None:
            # Schema not in cache, load it
            pool = self.pools.get(database_name)
            if pool is None:
                raise DatabaseError(
                    message=f"No connection pool available for database '{database_name}'",
                    details={"database": database_name},
                )
            try:
                schema = await self.schema_cache.load(database_name, pool)
            except Exception as e:
                raise SchemaLoadError(
                    message=f"Failed to load schema for database '{database_name}': {e!s}",
                    details={"database": database_name, "error": str(e)},
                ) from e

        logger.debug(
            "Schema loaded",
            extra={
                "request_id": request_id,
                "database": database_name,
                "tables": len(schema.tables),
            },
        )

        # Step 4: Generate and validate SQL with retry logic
        generated_sql, validation_result, tokens_used = await self._generate_sql_with_retry(
            question=request.question,
            schema=schema,
            request_id=request_id,
        )

        # Step 5: If return_type is SQL, return early
        if request.return_type == ReturnType.SQL:
            logger.info(
                "Returning SQL only",
                extra={"request_id": request_id, "sql_length": len(generated_sql)},
            )
            return QueryResponse(
                success=True,
                generated_sql=generated_sql,
                validation=validation_result,
                data=None,
                error=None,
                confidence=100,
                tokens_used=tokens_used,
            )

        # Step 6: Execute SQL against the resolved database's executor
        executor = self._executor_for(database_name)
        logger.debug(
            "Executing SQL",
            extra={"request_id": request_id, "database": database_name},
        )
        start_time = self._get_current_time_ms()

        results, total_count = await executor.execute(generated_sql)

        execution_time_ms = self._get_current_time_ms() - start_time
        if self.metrics is not None:
            self.metrics.observe_db_query_duration(execution_time_ms / 1000.0)
        logger.info(
            "SQL executed successfully",
            extra={
                "request_id": request_id,
                "database": database_name,
                "row_count": total_count,
                "execution_time_ms": execution_time_ms,
            },
        )

        # Step 7: Validate results (non-blocking, failures don't fail the request)
        result_confidence, validation_tokens = await self._validate_results_safely(
            question=request.question,
            sql=generated_sql,
            results=results,
            row_count=total_count,
            request_id=request_id,
        )

        if result_confidence < self.validation_config.min_confidence_score:
            logger.warning(
                "Result validation confidence below configured minimum",
                extra={
                    "request_id": request_id,
                    "confidence": result_confidence,
                    "min_confidence_score": self.validation_config.min_confidence_score,
                },
            )

        total_tokens = tokens_used
        if validation_tokens is not None:
            total_tokens = (
                validation_tokens if total_tokens is None else total_tokens + validation_tokens
            )

        # Step 8: Build successful response
        query_result = QueryResult(
            columns=list(results[0].keys()) if results else [],
            rows=results,
            row_count=len(results),  # Limited row count (after max_rows applied)
            execution_time_ms=execution_time_ms,
        )

        return QueryResponse(
            success=True,
            generated_sql=generated_sql,
            validation=validation_result,
            data=query_result,
            error=None,
            confidence=result_confidence,
            tokens_used=total_tokens,
        )

    def _executor_for(self, database_name: str) -> SQLExecutor:
        """Get the SQL executor registered for a database.

        Args:
            database_name: Resolved database name.

        Returns:
            SQLExecutor: Executor bound to the database's connection pool.

        Raises:
            DatabaseError: If no executor is registered for the database.
        """
        executor = self.sql_executors.get(database_name)
        if executor is None:
            raise DatabaseError(
                message=f"No SQL executor available for database '{database_name}'",
                details={
                    "database": database_name,
                    "configured": list(self.sql_executors.keys()),
                },
            )
        return executor

    async def _run_llm_call(
        self,
        operation: str,
        call: Callable[[], Awaitable[Any]],
    ) -> Any:
        """Run an LLM call under the LLM concurrency slot with metrics.

        Args:
            operation: Operation name for metrics labels (e.g. "generate_sql").
            call: Zero-argument coroutine factory performing the API call.

        Returns:
            The call's result.

        Raises:
            RateLimitExceededError: If no LLM slot becomes free in time.
        """
        if self.metrics is not None:
            self.metrics.increment_llm_call(operation)
        start = time.monotonic()
        try:
            if self.rate_limiter is not None:
                try:
                    async with self.rate_limiter.for_llm(timeout=LLM_SLOT_TIMEOUT_S):
                        return await call()
                except TimeoutError as e:
                    raise RateLimitExceededError(
                        message=(
                            "LLM concurrency limit reached; "
                            f"no slot available within {LLM_SLOT_TIMEOUT_S}s"
                        ),
                        details={"operation": operation, "timeout": LLM_SLOT_TIMEOUT_S},
                    ) from e
            return await call()
        finally:
            if self.metrics is not None:
                self.metrics.observe_llm_latency(operation, time.monotonic() - start)

    def _resolve_database(self, database: str | None) -> str:
        """Resolve database name from request or auto-select.

        If database is specified, validate it exists.
        If not specified and only one database available, auto-select it.

        Args:
            database: Database name from request (optional).

        Returns:
            str: Resolved database name.

        Raises:
            DatabaseError: If database is invalid or cannot be auto-selected.

        Example:
            >>> name = orchestrator._resolve_database("mydb")  # Validates "mydb" exists
            >>> name = orchestrator._resolve_database(None)  # Auto-selects if only one DB
        """
        if database is not None:
            # Validate specified database exists
            if database not in self.pools:
                raise DatabaseError(
                    message=f"Database '{database}' not found",
                    details={
                        "requested_database": database,
                        "available_databases": list(self.pools.keys()),
                    },
                )
            return database

        # Auto-select if only one database available
        available_dbs = list(self.pools.keys())
        if len(available_dbs) == 0:
            raise DatabaseError(
                message="No databases configured",
                details={},
            )
        if len(available_dbs) == 1:
            return available_dbs[0]

        # Multiple databases, must specify
        raise DatabaseError(
            message="Multiple databases available, please specify which to query",
            details={"available_databases": available_dbs},
        )

    async def _generate_sql_with_retry(
        self,
        question: str,
        schema: Any,
        request_id: str,
    ) -> tuple[str, ValidationResult, int | None]:
        """Generate and validate SQL with retry, backoff and circuit breaking.

        This method implements a retry loop that:
        1. Checks circuit breaker state
        2. Generates SQL using LLM (rate-limited, metrics-recorded)
        3. Validates the generated SQL
        4. On SQLParseError, sleeps with exponential backoff
           (retry_delay * backoff_factor ** attempt) and retries with
           error feedback
        5. Records success/failure to circuit breaker

        Security violations fail immediately: they are deterministic policy
        rejections, so retrying would only burn tokens and backoff time (or
        let the model rewrite the query into something unrelated, silently
        swallowing the rejection).

        Args:
            question: User's natural language question.
            schema: Database schema for context.
            request_id: Request ID for tracking.

        Returns:
            tuple: (generated_sql, validation_result, tokens_used)

        Raises:
            LLMError: If circuit breaker is open or generation fails.
            RateLimitExceededError: If the LLM concurrency limit is reached.
            SecurityViolationError: Immediately, if SQL violates policy.
            SQLParseError: If SQL cannot be parsed after all retries.
        """
        # Check circuit breaker
        if not self.circuit_breaker.allow_request():
            raise LLMError(
                message="SQL generation service is temporarily unavailable (circuit breaker open)",
                details={
                    "circuit_state": self.circuit_breaker.state,
                    "failure_count": self.circuit_breaker.failure_count,
                },
            )

        previous_sql: str | None = None
        error_feedback: str | None = None
        max_retries = self.resilience_config.max_retries
        tokens_total: int | None = None

        for attempt in range(max_retries + 1):
            try:
                logger.debug(
                    "Generating SQL",
                    extra={
                        "request_id": request_id,
                        "attempt": attempt + 1,
                        "max_retries": max_retries + 1,
                    },
                )

                # Generate SQL under the LLM concurrency slot. partial binds
                # the current attempt's values at creation time.
                generated_sql = await self._run_llm_call(
                    "generate_sql",
                    functools.partial(
                        self.sql_generator.generate,
                        question=question,
                        schema=schema,
                        previous_attempt=previous_sql,
                        error_feedback=error_feedback,
                    ),
                )

                if self.sql_generator.last_tokens is not None:
                    if self.metrics is not None:
                        self.metrics.increment_llm_tokens(
                            "generate_sql", self.sql_generator.last_tokens
                        )
                    tokens_total = (
                        self.sql_generator.last_tokens
                        if tokens_total is None
                        else tokens_total + self.sql_generator.last_tokens
                    )

                logger.debug(
                    "SQL generated",
                    extra={
                        "request_id": request_id,
                        "sql_length": len(generated_sql),
                    },
                )

                # Validate SQL
                try:
                    self.sql_validator.validate_or_raise(generated_sql)
                except SecurityViolationError as violation_error:
                    # Deterministic policy rejection: retrying cannot help.
                    # The LLM call itself succeeded (so the circuit breaker is
                    # left untouched); a retry would only burn tokens and
                    # backoff time, or let the model "fix" the SQL into an
                    # unrelated query that silently swallows the rejection.
                    logger.warning(
                        "SQL rejected by security policy; failing fast",
                        extra={
                            "request_id": request_id,
                            "attempt": attempt + 1,
                            "error": str(violation_error),
                        },
                    )
                    raise
                except SQLParseError as validation_error:
                    if attempt < max_retries:
                        # Record as failure and retry with feedback after backoff
                        logger.warning(
                            "SQL validation failed, retrying with feedback",
                            extra={
                                "request_id": request_id,
                                "attempt": attempt + 1,
                                "error": str(validation_error),
                            },
                        )
                        previous_sql = generated_sql
                        error_feedback = str(validation_error)
                        delay = self.resilience_config.retry_delay * (
                            self.resilience_config.backoff_factor**attempt
                        )
                        if delay > 0:
                            logger.debug(
                                "Backing off before retry",
                                extra={"request_id": request_id, "delay_seconds": delay},
                            )
                            await asyncio.sleep(delay)
                        continue
                    else:
                        # Out of retries, record failure and raise
                        self.circuit_breaker.record_failure()
                        logger.error(
                            "SQL validation failed after all retries",
                            extra={
                                "request_id": request_id,
                                "attempts": attempt + 1,
                                "error": str(validation_error),
                            },
                        )
                        raise

                # Validation successful
                self.circuit_breaker.record_success()
                logger.info(
                    "SQL generated and validated successfully",
                    extra={
                        "request_id": request_id,
                        "attempts": attempt + 1,
                    },
                )

                # Build validation result
                validation_result = ValidationResult(
                    is_valid=True,
                    is_select=True,
                    allows_data_modification=False,
                    uses_blocked_functions=[],
                    error_message=None,
                )

                return generated_sql, validation_result, tokens_total

            except (LLMError, SecurityViolationError, SQLParseError, RateLimitExceededError):
                # Re-raise known errors
                raise
            except Exception as e:
                # Unexpected error during generation
                self.circuit_breaker.record_failure()
                logger.exception(
                    "Unexpected error during SQL generation",
                    extra={"request_id": request_id},
                )
                raise LLMError(
                    message=f"SQL generation failed unexpectedly: {e!s}",
                    details={"error_type": type(e).__name__},
                ) from e

        # Should not reach here, but just in case
        self.circuit_breaker.record_failure()
        raise LLMError(
            message="SQL generation failed after all retry attempts",
            details={"max_retries": max_retries},
        )

    async def _validate_results_safely(
        self,
        question: str,
        sql: str,
        results: list[dict[str, Any]],
        row_count: int,
        request_id: str,
    ) -> tuple[int, int | None]:
        """Validate query results with error handling (non-blocking).

        This method attempts to validate results using LLM, but failures
        don't cause the overall query to fail. Returns a confidence score
        and the token usage of the validation call.

        Args:
            question: User's original question.
            sql: Generated SQL query.
            results: Query results.
            row_count: Total number of rows.
            request_id: Request ID for tracking.

        Returns:
            tuple: (confidence score 0-100, tokens used or None).
                Confidence is 100 if validation is disabled or fails.
        """
        if not self.validation_config.enabled:
            return 100, None

        try:
            logger.debug(
                "Validating results",
                extra={"request_id": request_id},
            )

            validation_result = await self._run_llm_call(
                "validate_result",
                lambda: self.result_validator.validate(
                    question=question,
                    sql=sql,
                    results=results,
                    row_count=row_count,
                ),
            )

            if self.result_validator.last_tokens is not None and self.metrics is not None:
                self.metrics.increment_llm_tokens(
                    "validate_result", self.result_validator.last_tokens
                )

            logger.info(
                "Result validation completed",
                extra={
                    "request_id": request_id,
                    "confidence": validation_result.confidence,
                    "is_acceptable": validation_result.is_acceptable,
                },
            )

            return validation_result.confidence, self.result_validator.last_tokens

        except Exception as e:
            # Log but don't fail the query
            logger.warning(
                "Result validation failed, continuing with default confidence",
                extra={
                    "request_id": request_id,
                    "error": str(e),
                },
            )
            return 100, None  # Default to high confidence if validation fails

    def _build_error_response(self, request_id: str, error: PgMcpError) -> QueryResponse:
        """Build a failure response for a known application error.

        Args:
            request_id: Request ID for tracking.
            error: The application error to report.

        Returns:
            QueryResponse: Response with success=False and error details.
        """
        logger.warning(
            "Query execution failed with known error",
            extra={
                "request_id": request_id,
                "error_code": error.code,
                "error_message": str(error),
            },
        )
        return QueryResponse(
            success=False,
            generated_sql=None,
            validation=None,
            data=None,
            error=ErrorDetail(
                code=error.code.value,
                message=error.message,
                details=error.details,
            ),
            confidence=0,
            tokens_used=None,
        )

    @staticmethod
    def _get_current_time_ms() -> float:
        """Get current time in milliseconds.

        Returns:
            float: Current time in milliseconds since epoch.
        """
        return time.time() * 1000
