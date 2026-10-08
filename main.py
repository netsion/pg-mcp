"""PostgreSQL MCP server entry point.

Delegates to pg_mcp.__main__.main so `python main.py` and `python -m pg_mcp`
start the same server; see the README for configuration.
"""

from pg_mcp.__main__ import main

if __name__ == "__main__":
    main()
