"""Unit test specific configuration.

Unit tests must not depend on the developer's shell environment or a real
.env file: default-value assertions would break on any machine that exports
DATABASE_*/OPENAI_* (or runs the server first). Strip all pg-mcp env vars
for the duration of each test; monkeypatch restores them afterwards. Tests
that exercise env-var loading set their own variables inside the test body,
after this fixture has run.
"""

import os

import pytest

ENV_PREFIXES = (
    "DATABASE_",
    "OPENAI_",
    "SECURITY_",
    "VALIDATION_",
    "CACHE_",
    "RESILIENCE_",
    "OBSERVABILITY_",
)


@pytest.fixture(autouse=True)
def clean_pg_mcp_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove pg-mcp related environment variables for each unit test."""
    for key in list(os.environ):
        if key.startswith(ENV_PREFIXES):
            monkeypatch.delenv(key, raising=False)
