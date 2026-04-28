# MAO - Multi Agent Orchestration

<div align="center">
  <p>
    <a href="https://github.com/tiangolo/fastapi"><img src="https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi" alt="FastAPI"></a>
    <a href="https://github.com/duckdb/duckdb"><img src="https://img.shields.io/badge/DuckDB-FFF000?style=for-the-badge&logo=duckdb" alt="DuckDB"></a>
    <a href="https://github.com/langchain-ai/langchain"><img src="https://img.shields.io/badge/LangChain-2C39BD?style=for-the-badge&logo=langchain" alt="LangChain"></a>
  </p>
  <p>
    <a href="https://github.com/anthropics/anthropic-sdk-python"><img src="https://img.shields.io/badge/Anthropic-0B0D10?style=for-the-badge&logo=anthropic" alt="Anthropic"></a>
    <a href="https://github.com/openai/openai-python"><img src="https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai" alt="OpenAI"></a>
    <a href="https://github.com/ollama/ollama"><img src="https://img.shields.io/badge/Ollama-000000?style=for-the-badge&logo=ollama" alt="Ollama"></a>
    <a href="https://github.com/mcp-foundation/mcp"><img src="https://img.shields.io/badge/MCP-5A45FF?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0iI2ZmZiIgZD0iTTEyIDJMMiA3djEwbDEwIDUgMTAtNVY3eiIvPjwvc3ZnPg==" alt="MCP"></a>
  </p>
</div>

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/agentic-dev-io/mao)

A modern framework for orchestrating AI agents. Self-contained — no external services required for vector storage or embeddings.

## Features

- **Agent Orchestration** — Multi-agent workflows with LangGraph
- **Vector-based Memory** — DuckDB-powered vector storage with SentenceTransformers embeddings (no external DB needed)
- **MCP Integration** — Model Context Protocol for agent-tool communication
- **Multi-LLM Support** — OpenAI, Anthropic, Ollama
- **Knowledge & Experience** — Automatic vector-based memory per agent
- **Local Skills** — Claude/Codex-style `SKILL.md` discovery from project and home directories
- **Team Management** — Organize agents into collaborative teams with supervisors
- **FastAPI** — REST API for agent management

## Installation

```bash
pip install mao-agents
```

Or for development:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

## Quick Start

```python
from mao import create_agent

agent = await create_agent(
    provider="anthropic",
    model_name="claude-sonnet-4-20250514",
    agent_name="assistant",
    system_prompt="You are a helpful data analyst.",
)

response = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "Analyze the latest data"}]}
)
```

## Environment Variables

```
# LLM API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-...

# Vector Storage (optional — defaults to mao_vectors.duckdb)
VECTOR_DB_PATH=./data/mao_vectors.duckdb

# LangGraph checkpoints (durable short-term memory)
MAO_CHECKPOINT_DB_PATH=./data/mao_checkpoints.duckdb

# DuckDB Configuration
MCP_DB_PATH=./data/mcp_config.duckdb

# MCP / Ollama
MCP_CONFIG_PATH=./.mcp.json
OLLAMA_HOST=http://localhost:11434

# Optional skill discovery overrides (`;` on Windows, `:` on Unix)
MAO_SKILL_PATHS=./.codex/skills;./.claude/skills

# HITL for selected tool names (comma-separated)
MAO_HITL_TOOLS=send_email,delete_record

# LangSmith tracing / observability
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=mao-agents
LANGSMITH_TRACING=true
LANGSMITH_TRACING_SAMPLING_RATE=0.25
LANGCHAIN_TRACING_V2=true
```

## API

```bash
uv run uvicorn src.mao.api.api:api --host 0.0.0.0 --port 8000 --reload
```

Endpoints: `/agents`, `/teams`, `/mcp`, `/config`, `/health`
Docs: `/docs` (Swagger), `/redoc`

Runtime notes:
- Agent and supervisor checkpoint state is persisted via `MAO_CHECKPOINT_DB_PATH`
- `/agents/{id}/chat` and `/teams/{id}/chat` accept optional `response_schema`
  for structured output
- The same chat endpoints accept optional `approval_decisions` to resume
  human-in-the-loop tool approvals
- Agents support optional `skills` and `skill_paths` fields in create/update payloads
- Discoverable skills are listed via `/config/skills`
- Skill discovery checks project and home roots such as `.codex/skills`,
  `.claude/skills`, and `.agents/skills`
- Workspace guidance from `AGENTS.md` and `CLAUDE.md` is injected into the
  agent system prompt when found
- Team members support typed communication policies via `params`
  with fields such as:
  - `allow_supervisor_delegation`
  - `routing_keywords`
  - `blocked_keywords`
  - `can_receive_direct`
  - `allow_peer_messages`
  - `can_message_roles`
  - `can_message_agents`
  - `accept_messages_from_roles`
- `/teams/{id}/chat` supports direct member routing via `direct_to_agent_id`
- Team chat traces now include structured events for:
  - supervisor delegations
  - direct user-to-member routing
  - peer member-to-member messages

### Team Policy Example

```json
{
  "agent_id": "agent_writer_1",
  "role": "writer",
  "params": {
    "allow_supervisor_delegation": true,
    "routing_keywords": ["draft", "write", "summary"],
    "blocked_keywords": ["delete", "shutdown"],
    "can_receive_direct": true,
    "allow_peer_messages": true,
    "accept_messages_from_roles": ["researcher", "editor"]
  }
}
```

## Docker

```bash
docker compose up -d
```

## License

MIT — see [LICENSE](LICENSE).
