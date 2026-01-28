# MAO - MCP Agent Orchestra

<div align="center">
  <p>
    <a href="https://github.com/tiangolo/fastapi"><img src="https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi" alt="FastAPI"></a>
    <a href="https://github.com/qdrant/qdrant"><img src="https://img.shields.io/badge/Qdrant-FF4582?style=for-the-badge&logo=qdrant" alt="Qdrant"></a>
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

MAO is a modern framework for orchestrating AI agents. It combines the power of vector databases, LLMs, and the Model Context Protocol (MCP) to enable robust and scalable agent workflows.

## Features

- 🤖 **Agent Orchestration** - Manage complex multi-agent workflows
- 🧠 **Vector-based Memory** - Store and retrieve context information
- 🔄 **MCP Integration** - Seamless communication between agents and tools
- 🛠️ **Extensible Tools** - Easy integration of new capabilities
- 📊 **DuckDB Analytics** - Powerful data analysis and processing
- 🔍 **Semantic Search** - Find relevant information across agent memories
- 🤝 **Team Management** - Organize agents into collaborative teams
- 🔒 **Secure Configuration** - Centralized management of API keys and settings
- 📤 **Import/Export** - Backup and restore system configurations
- 🔄 **Supervisor Agents** - Coordinate team workflows with supervisor agents
- 📚 **Knowledge & Experience Trees** - Structured storage for agent knowledge
- 🌐 **Multi-LLM Support** - Works with OpenAI, Anthropic, and Ollama models

## API Endpoints

The MAO API provides the following main endpoints:

- `/agents` - Agent creation, management, and interaction
- `/teams` - Team creation and management
- `/teams/supervisors` - Supervisor management for agent teams
- `/mcp` - MCP server and tool management
- `/config` - Global configuration settings
- `/export`, `/import` - Configuration import/export utilities
- `/health` - API health check endpoint

API documentation is available at:
- Swagger UI: `/docs`
- ReDoc: `/redoc`

## Requirements

- Python 3.11+
- Qdrant vector database (accessible via HTTP)
- DuckDB for configuration storage
- LLM provider API keys (OpenAI, Anthropic, or local Ollama instance)

## Installation

```bash
# With uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh
# Review the install script or use your package manager if you prefer.
uv sync

```

## Quick Start

```python
from mao.agents import create_agent
from mao.storage import KnowledgeTree, ExperienceTree

# Initialize storage
knowledge_tree = await KnowledgeTree.create(collection_name="agent-memory")
experience_tree = await ExperienceTree.create(collection_name="agent-experience")

# Create an agent
agent_app = await create_agent(
    provider="anthropic",
    model_name="claude-3-opus-20240229",
    agent_name="assistant",
    knowledge_tree=knowledge_tree,
    experience_tree=experience_tree,
)

# Execute a query
response = await agent_app.ainvoke(
    {"messages": [{"role": "user", "content": "Analyze the latest economic data"}]}
)
if hasattr(response, "content"):
    print(response.content)
elif isinstance(response, dict) and response.get("messages"):
    print(response["messages"][-1].content)
else:
    print(response)
```

## Environment Variables

The following environment variables are supported:

```
# LLM API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-...

# Vector Database
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your-qdrant-api-key
EMBEDDING_MODEL=text-embedding-3-small

# DuckDB Configuration
MCP_DB_PATH=/path/to/mcp_config.duckdb

# MCP Configuration
MCP_CONFIG_PATH=/path/to/mcp.json
OLLAMA_HOST=http://localhost:11434

# Server
PORT=8000
```

## Docker

```bash
# Build and start the services with Docker Compose
docker compose up -d

# Or build the Docker image manually
docker build -t mao-api -f docker/Dockerfile.api .

# Start the container
docker run -p 8000:8000 -v ./data:/data -v ./.env:/app/.env mao-api
```

For development, you can use the following commands:

```bash
# Build with BuildKit enabled for better caching
DOCKER_BUILDKIT=1 docker build -t mao-api -f docker/Dockerfile.api .

# Run with mounted source directory for development
docker run -p 8000:8000 -v ./data:/data -v ./.env:/app/.env mao-api

# Pass environment variables directly
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=sk-... \
  -e ANTHROPIC_API_KEY=sk-... \
  -e QDRANT_URL=http://localhost:6333 \
  mao-api

# Or use the --env-file option
docker run -p 8000:8000 --env-file .env mao-api
```

### Docker Compose with Environment Variables

You can also use Docker Compose to manage environment variables:

```yaml
services:
  api:
    build:
      context: .
      dockerfile: docker/Dockerfile.api
    ports:
      - "8000:8000"
    volumes:
      - ./data:/data
    env_file:
      - .env
```

## API Example

```python
import httpx

async with httpx.AsyncClient() as client:
    # Create a new agent
    response = await client.post(
        "http://localhost:8000/agents",
        json={
            "name": "research_assistant",
            "provider": "anthropic",
            "model_name": "claude-3-opus-20240229",
            "system_prompt": "You are a research assistant."
        }
    )
    agent_id = response.json()["id"]
    
    # Send a message to the agent
    response = await client.post(
        f"http://localhost:8000/agents/{agent_id}/chat",
        json={"content": "Summarize the latest developments in AI."}
    )
    print(response.json()["response"])
```

## Team Workflow Example

```python
# Create a team with supervisor
team_id = "team_research"
supervisor_id = "supervisor_research_team"

# Add agents to the team
await client.post(
    f"http://localhost:8000/teams/{team_id}/members",
    json={
        "agent_id": "agent_researcher",
        "role": "researcher",
        "order_index": 1
    }
)

await client.post(
    f"http://localhost:8000/teams/{team_id}/members",
    json={
        "agent_id": "agent_writer",
        "role": "writer",
        "order_index": 2
    }
)

# Start the team
await client.post(f"http://localhost:8000/teams/{team_id}/start")

# Send a task to the team
response = await client.post(
    f"http://localhost:8000/teams/{team_id}/chat",
    json={"message": "Research quantum computing advancements and write a report"}
)
```

## AI Agent Capabilities

<div align="center">
  <p>
    <img src="https://img.shields.io/badge/🧠_Memory-4D4D4D?style=for-the-badge" alt="Memory">
    <img src="https://img.shields.io/badge/🔄_Planning-4D4D4D?style=for-the-badge" alt="Planning">
    <img src="https://img.shields.io/badge/🔍_Research-4D4D4D?style=for-the-badge" alt="Research">
  </p>
  <p>
    <img src="https://img.shields.io/badge/🛠️_Tools-4D4D4D?style=for-the-badge" alt="Tools">
    <img src="https://img.shields.io/badge/🤝_Collaboration-4D4D4D?style=for-the-badge" alt="Collaboration">
    <img src="https://img.shields.io/badge/📊_Analytics-4D4D4D?style=for-the-badge" alt="Analytics">
  </p>
  <p>
    <img src="https://img.shields.io/badge/🔒_Security-4D4D4D?style=for-the-badge" alt="Security">
    <img src="https://img.shields.io/badge/⚡_Performance-4D4D4D?style=for-the-badge" alt="Performance">
    <img src="https://img.shields.io/badge/🔌_Integration-4D4D4D?style=for-the-badge" alt="Integration">
  </p>
</div>

## CI/CD with GitHub Actions

This project uses GitHub Actions for continuous integration and deployment:

### Workflows

- **Test and Lint** - Runs tests, linting, and type checking on every push and pull request.
- **Docker Build** - Builds and publishes Docker images on pushes to the main branch and tags.
- **Docker Multi-Platform Build** - Creates Docker images for multiple platforms (amd64, arm64).
- **Dependency Updates** - Automatically updates project dependencies weekly.
- **Package Publishing** - Publishes the package to PyPI on new releases.

### Environment Variables and Secrets

To use environment variables in GitHub Actions workflows, you need to add them as GitHub Secrets:

1. Go to your GitHub repository
2. Navigate to Settings > Secrets and variables > Actions
3. Click on "New repository secret"
4. Add each environment variable from your `.env` file:
   - `OPENAI_API_KEY`
   - `ANTHROPIC_API_KEY`
   - `QDRANT_URL`
   - `EMBEDDING_MODEL`
   - `MCP_DB_PATH`
   - `MCP_CONFIG_PATH`
   - `OLLAMA_HOST`

These secrets are then passed to the Docker build process as build arguments and set as environment variables in the container.

### Workflow Execution

```bash
# Manually run the dependency update workflow
gh workflow run dependency-update.yml

# Manually publish a version
gh workflow run publish.yml -f version=0.2.0

# Manually run multi-platform Docker build
gh workflow run docker-multi-platform.yml -f platforms=linux/amd64,linux/arm64,linux/arm/v7
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
