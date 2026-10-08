"""Unit tests for new configuration features.

Covers the configurable security controls (blocked tables/columns, EXPLAIN
policy), multi-database discovery from environment variables, and the
resilience concurrency limits.
"""

from pg_mcp.config.settings import (
    DatabaseConfig,
    ResilienceConfig,
    SecurityConfig,
    load_extra_databases,
)


class TestSecurityControlsConfig:
    """SecurityConfig table/column blocklists and EXPLAIN policy."""

    def test_defaults_empty_blocklists(self) -> None:
        """Blocked tables/columns default to empty and EXPLAIN is off."""
        config = SecurityConfig()
        assert config.blocked_tables == []
        assert config.blocked_columns == []
        assert config.allow_explain is False

    def test_comma_separated_env_parsing_tables(self, monkeypatch) -> None:
        """SECURITY_BLOCKED_TABLES accepts a comma-separated string."""
        monkeypatch.setenv("SECURITY_BLOCKED_TABLES", "users, audit_log , payments")
        config = SecurityConfig()
        assert config.blocked_tables == ["users", "audit_log", "payments"]

    def test_comma_separated_env_parsing_columns(self, monkeypatch) -> None:
        """SECURITY_BLOCKED_COLUMNS accepts a comma-separated string."""
        monkeypatch.setenv("SECURITY_BLOCKED_COLUMNS", "ssn,password_hash")
        config = SecurityConfig()
        assert config.blocked_columns == ["ssn", "password_hash"]

    def test_allow_explain_env(self, monkeypatch) -> None:
        """SECURITY_ALLOW_EXPLAIN toggles the EXPLAIN policy."""
        monkeypatch.setenv("SECURITY_ALLOW_EXPLAIN", "true")
        assert SecurityConfig().allow_explain is True

    def test_blocklists_accept_plain_lists(self) -> None:
        """Programmatic construction accepts real lists."""
        config = SecurityConfig(blocked_tables=["a", "b"], blocked_columns=["c"])
        assert config.blocked_tables == ["a", "b"]
        assert config.blocked_columns == ["c"]


class TestResilienceConcurrency:
    """ResilienceConfig concurrency limits."""

    def test_defaults(self) -> None:
        """Default concurrency limits match the documented values."""
        config = ResilienceConfig()
        assert config.query_concurrency_limit == 10
        assert config.llm_concurrency_limit == 5

    def test_env_override(self, monkeypatch) -> None:
        """RESILIENCE_*_CONCURRENCY_LIMIT env vars are honored."""
        monkeypatch.setenv("RESILIENCE_QUERY_CONCURRENCY_LIMIT", "3")
        monkeypatch.setenv("RESILIENCE_LLM_CONCURRENCY_LIMIT", "2")
        config = ResilienceConfig()
        assert config.query_concurrency_limit == 3
        assert config.llm_concurrency_limit == 2


class TestLoadExtraDatabases:
    """Environment-based multi-database discovery."""

    def test_no_extra_databases(self, monkeypatch) -> None:
        """Only DATABASE_NAME defined -> no extra databases."""
        monkeypatch.setenv("DATABASE_NAME", "primary_db")
        assert load_extra_databases() == []

    def test_minimal_secondary_inherits_primary(self, monkeypatch) -> None:
        """A secondary with only NAME inherits primary connection values."""
        monkeypatch.setenv("DATABASE_NAME", "primary_db")
        monkeypatch.setenv("DATABASE_HOST", "10.1.1.1")
        monkeypatch.setenv("DATABASE_USER", "admin")
        monkeypatch.setenv("DATABASE_PASSWORD", "secret")
        monkeypatch.setenv("SECONDARY_DATABASE_NAME", "analytics")

        primary = DatabaseConfig()
        extras = load_extra_databases(primary)

        assert len(extras) == 1
        assert extras[0].name == "analytics"
        assert extras[0].host == "10.1.1.1"
        assert extras[0].user == "admin"
        assert extras[0].password == "secret"

    def test_fully_specified_secondary(self, monkeypatch) -> None:
        """A secondary can override every connection parameter."""
        monkeypatch.setenv("WAREHOUSE_DATABASE_NAME", "warehouse")
        monkeypatch.setenv("WAREHOUSE_DATABASE_HOST", "10.2.2.2")
        monkeypatch.setenv("WAREHOUSE_DATABASE_PORT", "5433")
        monkeypatch.setenv("WAREHOUSE_DATABASE_USER", "etl")
        monkeypatch.setenv("WAREHOUSE_DATABASE_PASSWORD", "whsecret")

        extras = load_extra_databases(DatabaseConfig(host="primary.example"))

        assert len(extras) == 1
        extra = extras[0]
        assert extra.name == "warehouse"
        assert extra.host == "10.2.2.2"
        assert extra.port == 5433
        assert extra.user == "etl"
        assert extra.password == "whsecret"

    def test_multiple_secondaries_sorted(self, monkeypatch) -> None:
        """Multiple prefixes are discovered and sorted by name."""
        monkeypatch.setenv("ZETA_DATABASE_NAME", "zzz_db")
        monkeypatch.setenv("ALPHA_DATABASE_NAME", "aaa_db")

        extras = load_extra_databases()

        assert [db.name for db in extras] == ["aaa_db", "zzz_db"]

    def test_empty_name_ignored(self, monkeypatch) -> None:
        """A <PREFIX>_DATABASE_NAME that is empty/whitespace is ignored."""
        monkeypatch.setenv("SECONDARY_DATABASE_NAME", "   ")
        assert load_extra_databases() == []
