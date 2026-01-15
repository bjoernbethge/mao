# MAO - MCP Agent Orchestra

## Project Overview

MAO (MCP Agent Orchestra) is a modern framework for orchestrating AI agents using the Model Context Protocol (MCP). It provides infrastructure for multi-agent coordination, vector-based knowledge storage, and LLM integration for building sophisticated AI systems.

**Core Purpose:** Enable seamless orchestration of multiple AI agents with MCP tool integration, vector memory, and flexible LLM backends.

## Tech Stack

### Core Dependencies
- **Python 3.11+** - Required for modern async features and type hints
- **uv** - Modern Python package manager (NOT pip)
- **FastAPI** - Async web framework for API endpoints
- **Pydantic** - Data validation and settings management

### AI/ML Stack
- **LangChain** - LLM orchestration and chains
- **LangGraph** - Agent workflow graphs and state machines
- **LangChain-MCP-Adapters** - MCP protocol integration
- **OpenAI/Anthropic** - LLM API clients
- **Ollama** - Local LLM support

### Data & Storage
- **Qdrant** - Vector database for agent memory and knowledge
- **DuckDB** - Embedded analytics database
- **httpx** - Async HTTP client

## Architecture Principles

### Modular Design (CRITICAL)
- **No god classes** - Break functionality into focused modules
- **DRY principle** - Avoid code duplication, create reusable components
- **Folder-based organization** - Related functionality in dedicated folders with multiple files

### Core Concepts
- **Agent**: LLM-powered autonomous entity with tools and memory
- **MCP Server**: External tool/resource provider following Model Context Protocol
- **Orchestrator**: Coordinates multiple agents using LangGraph workflows
- **Vector Store**: Knowledge base using Qdrant for semantic search
- **RAG System**: Retrieval-Augmented Generation for contextual responses

## Code Style

### Python Standards
- **Type hints**: REQUIRED for all function signatures
- **Async/await**: Use async patterns for I/O operations
- **Formatting**: Black (line length 88) - run with `uv run black .`
- **Linting**: Ruff for fast linting - run with `uv run ruff check .`
- **Type checking**: mypy (configured in pyproject.toml)
- **Docstrings**: Google-style for public APIs

### Package Management
- **Always use `uv`** for dependency management
- Install: `uv pip install <package>`
- Add dependencies: `uv add <package>`
- Dev dependencies: `uv add --dev <package>`
- Sync environment: `uv sync`
- Never use pip directly

### Naming Conventions
- `snake_case` - Functions, variables, modules
- `PascalCase` - Classes
- `UPPER_SNAKE_CASE` - Constants
- `_private` - Internal methods/attributes

## Key Components

### 1. Agent System (`mao/agents.py`)
- LangChain-based agent implementations
- Tool integration and function calling
- Agent memory and context management
- Conversation handling and history

### 2. MCP Integration (`mao/mcp.py`)
- Model Context Protocol server management
- MCP tool discovery and registration
- Dynamic tool loading from MCP servers
- Protocol compliance and validation

### 3. RAG System (`mao/rag-system.py`)
- Retrieval-Augmented Generation pipeline
- Document chunking and embedding
- Vector similarity search
- Context-aware response generation

### 4. Storage Layer (`mao/storage.py`)
- Qdrant vector database operations
- DuckDB analytics queries
- Memory persistence and retrieval
- Collection management

### 5. API Layer (`mao/api/`)
- FastAPI routes and endpoints
- Request/response validation
- WebSocket support for streaming
- Health checks and monitoring

## Development Guidelines

### Adding New Agents
```python
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent

async def create_agent(tools: list) -> Agent:
    llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
    agent = create_react_agent(llm, tools)
    return agent
```

### Integrating MCP Servers
```python
from mao.mcp import MCPServerManager

async def add_mcp_server(server_config: dict):
    manager = MCPServerManager()
    await manager.add_server(server_config)
    tools = await manager.get_tools()
    return tools
```

### Working with Vector Store
```python
from mao.storage import QdrantStore

async def store_documents(docs: list[str]):
    store = QdrantStore()
    await store.add_documents(docs)
    results = await store.search(query="...", limit=5)
    return results
```

### Building API Endpoints
```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

class AgentRequest(BaseModel):
    query: str
    context: dict | None = None

@router.post("/agent/query")
async def query_agent(request: AgentRequest):
    # Implement endpoint logic
    pass
```

## Testing Strategy

### Unit Tests
- Test individual agent behaviors
- Mock LLM responses for deterministic tests
- Test MCP tool integration
- Verify vector store operations

### Integration Tests
- Full agent workflows
- MCP server communication
- API endpoint testing
- LLM provider integration

### Test Requirements
- Use pytest with async support (`pytest-asyncio`)
- Run tests: `uv run pytest`
- Coverage: `uv run pytest --cov=mao`
- Mark async tests: `@pytest.mark.asyncio`

## Configuration Management

### Environment Variables
Required environment variables:
```bash
OPENAI_API_KEY=...           # OpenAI API key
ANTHROPIC_API_KEY=...        # Anthropic API key
QDRANT_URL=localhost:6333    # Qdrant server URL
OLLAMA_URL=localhost:11434   # Ollama server URL
```

### Settings Files
- Use Pydantic settings for configuration
- Support .env files with python-dotenv
- Validate all settings on startup
- Never commit secrets to git

## MCP Protocol

### What is MCP?
Model Context Protocol enables LLMs to:
- Access external tools and resources
- Query databases and APIs
- Execute code and commands
- Integrate with external services

### MCP Server Types
- **Stdio servers**: Communicate via stdin/stdout
- **SSE servers**: Server-Sent Events for streaming
- **HTTP servers**: REST API endpoints

### Adding MCP Servers
1. Define server configuration (connection, capabilities)
2. Register with MCPServerManager
3. Discover available tools
4. Provide tools to agents

## LangGraph Workflows

### Agent Orchestration
```python
from langgraph.graph import StateGraph, END

workflow = StateGraph(AgentState)
workflow.add_node("agent1", agent1_node)
workflow.add_node("agent2", agent2_node)
workflow.add_edge("agent1", "agent2")
workflow.add_edge("agent2", END)
```

### State Management
- Define state schemas with TypedDict
- Pass state between nodes
- Handle conditional routing
- Implement error recovery

## Performance Considerations

### Async Operations
- Use async/await for all I/O operations
- Avoid blocking operations in async context
- Use asyncio.gather() for parallel tasks
- Handle timeouts with asyncio.wait_for()

### Vector Store Optimization
- Batch document insertions
- Use appropriate embedding models
- Configure collection parameters
- Implement caching for frequent queries

### LLM Optimization
- Cache LLM responses when appropriate
- Use streaming for long responses
- Implement retry logic with tenacity
- Monitor token usage and costs

## Common Tasks

### Start Development Server
```bash
uv run uvicorn mao.api.main:app --reload
```

### Run Tests
```bash
uv run pytest
uv run pytest --cov=mao --cov-report=html
```

### Add Dependency
```bash
uv add package-name
uv add --dev pytest-package
```

### Format and Lint
```bash
uv run black .
uv run ruff check .
uv run mypy mao/
```

## Security Notes
- Validate all LLM inputs and outputs
- Sanitize tool parameters before execution
- Use environment variables for API keys
- Implement rate limiting for API endpoints
- Never log sensitive information
- Validate MCP server configurations

## Troubleshooting

### Common Issues
1. **Import errors**: Run `uv sync` to install dependencies
2. **Type errors**: Check mypy configuration in pyproject.toml
3. **Async errors**: Ensure all I/O operations use async/await
4. **MCP connection**: Verify server URL and protocol
5. **Vector store**: Check Qdrant is running and accessible

## Resources
- [Model Context Protocol Spec](https://modelcontextprotocol.io/)
- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Guide](https://langchain-ai.github.io/langgraph/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
