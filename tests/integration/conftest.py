"""Integration test configuration.

Integration tests run the real server lifespan against real PostgreSQL and
LLM endpoints (see the `integration` pytest marker). Load the project .env
so credentials come from the same place the server reads them. Unit tests
stay hermetic: they strip these variables in tests/unit/conftest.py.
"""

from dotenv import load_dotenv

load_dotenv()
