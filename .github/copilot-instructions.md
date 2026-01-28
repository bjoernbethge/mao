# MAO - MCP Agent Orchestra: Developer Guide

## Quick Repository Overview

**MAO** (MCP Agent Orchestra) is a Python 3.11+ FastAPI framework for orchestrating AI agents using the Model Context Protocol (MCP), LangChain, and vector databases. The codebase is **~1.8MB** with **26 Python files** across **~4,400 lines of code**.

**Tech Stack**: FastAPI, LangChain, LangGraph, Qdrant (vectors), DuckDB (analytics), OpenAI/Anthropic/Ollama (LLMs)

**Project Structure**:
```
/home/runner/work/mao/mao/
├── src/mao/              # Main package (agents, storage, MCP, API)
│   ├── agents.py         # Agent creation and management
│   ├── storage.py        # Vector stores (KnowledgeTree, ExperienceTree)
│   ├── mcp.py           # MCP protocol client integration
│   ├── rag-system.py    # RAG pipeline implementation
│   ├── tools.py         # Agent tools
│   └── api/             # FastAPI routes (agents, teams, MCP, storage)
├── tests/               # Unit and integration tests
├── pyproject.toml       # Dependencies and tool configs
├── docker-compose.yml   # Services: api, qdrant, ollama
└── .github/workflows/   # CI/CD pipelines
```

## Critical Setup & Build Instructions

### Prerequisites
- **Python 3.11** (required - see `.python-version`)
- **uv package manager** (NOT pip - this is mandatory)
- **Docker** (for services: Qdrant, Ollama)

### Bootstrap Process (ALWAYS follow this sequence)

**Step 1: Install uv**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

**Step 2: Install dependencies**
```bash
# ALWAYS run this first before any other command
uv sync
```
- Installs all dependencies from `pyproject.toml`
- Creates `.venv/` virtual environment
- Takes ~30-60 seconds
- **CRITICAL**: Never use `pip install` - always use `uv`

**Step 3: Create data directory**
```bash
mkdir -p data
```

### Required Environment Variables

Create `.env` from `.env.example`:
```bash
# LLM API Keys (at least one required for integration tests)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Vector Database
QDRANT_URL=http://localhost:6333
EMBEDDING_MODEL=text-embedding-3-small

# DuckDB & MCP
MCP_DB_PATH=./data/mcp_config.duckdb
MCP_CONFIG_PATH=./mcp.json
OLLAMA_HOST=http://localhost:11434

# MCP Server API Keys (optional)
CONTEXT7_API_KEY=your-context7-api-key  # For up-to-date code docs
```

**Note**: context7 MCP server is configured in `mcp.json` and provides AI agents with access to current library documentation. Get an API key from [context7.com](https://context7.com) if needed.

### Docker Services (Required for Integration Tests)

**Start services:**
```bash
docker compose up -d
```

Services started:
- Qdrant (ports 6333, 6334)
- Ollama (port 11434)
- API (port 8000)

**Wait for services to be ready** (IMPORTANT - services take time to start):
```bash
# Wait for Qdrant (up to 60 seconds)
until curl -s http://localhost:6333/health > /dev/null; do sleep 1; done

# Wait for Ollama (up to 60 seconds)
until curl -s http://localhost:11434/api/version > /dev/null; do sleep 2; done
```

## Linting, Formatting, and Type Checking

**ALWAYS run these before committing code:**

### 1. Format code with Black
```bash
uv run black .
```
- Line length: 88 (configured in pyproject.toml)
- Modifies files in place
- Takes ~5 seconds

### 2. Lint with Ruff
```bash
uv run ruff check .
```
- Fast linting (~2 seconds)
- Auto-fix: `uv run ruff check --fix .`
- Config in `pyproject.toml`

### 3. Type check with mypy
```bash
uv run mypy .
```
- Takes ~10-15 seconds
- Config in `[tool.mypy]` section of `pyproject.toml`
- Some modules have `ignore_errors = true` (e.g., `mao.agents`)

**Pre-commit hooks**: Configured in `.pre-commit-config.yaml` (black, ruff, mypy)

## Testing

### Unit Tests (Fast - No External Services)
```bash
# Run API unit tests only (~15-30 seconds)
uv run pytest tests/api/test_api.py tests/api/test_agents_api.py tests/api/test_mcp_api.py tests/api/test_teams_api.py -v --timeout=60
```
- Uses mocks, no Docker services required
- Set `QDRANT_URL`, `MCP_DB_PATH`, `MCP_CONFIG_PATH` env vars (see unit-tests.yml)

### Integration Tests (Slow - Requires Docker Services)
```bash
# Start Docker services FIRST (see above)
uv run pytest tests/test_storage.py tests/test_agents.py tests/test_mcp.py tests/api/test_live_api.py -v --timeout=120
```
- Requires: Qdrant, Ollama running
- Requires: `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` in environment
- Takes ~2-5 minutes
- Timeout: 120 seconds per test

### Run All Tests with Coverage
```bash
uv run pytest --cov=src --cov-report=html -v
```

### Test Structure
- `tests/` - Unit tests for core modules
- `tests/api/` - API endpoint tests
- `tests/conftest.py` - Shared fixtures
- Mark async tests with `@pytest.mark.asyncio`

## Running the Application

### Development Server
```bash
# From project root
uv run uvicorn src.mao.api.api:api --host 0.0.0.0 --port 8000 --reload
```
- API docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Health check: http://localhost:8000/health

### Production (Docker)
```bash
docker compose up -d
```

## GitHub Actions CI/CD

### Workflows (in `.github/workflows/`)

1. **test-lint.yml** - Runs on ALL PRs
   - Linting (ruff), formatting (black), type checking (mypy)
   - Timeout: 10 minutes
   - NO external services needed

2. **unit-tests.yml** - Runs on ALL PRs
   - Fast API tests with mocks
   - Timeout: 15 minutes
   - NO Docker services needed

3. **integration-tests.yml** - Runs on main OR with 'test-integration' label
   - Full tests with Qdrant + Ollama
   - Timeout: 30 minutes
   - Requires secrets: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`

4. **docker-build.yml** - Builds Docker images
   - Multi-platform: linux/amd64, linux/arm64 (for tags)
   - Pushes to ghcr.io
   - Timeout: 30 minutes

### To run integration tests on your PR:
```bash
# Add label to PR
gh pr edit <PR-NUMBER> --add-label "test-integration"
```

## Key Architecture Elements

### Core Modules
- **`src/mao/agents.py`** (Agent, Supervisor, create_agent) - LangChain-based agent creation
- **`src/mao/storage.py`** (KnowledgeTree, ExperienceTree) - Qdrant vector stores
- **`src/mao/mcp.py`** (MCPClient) - MCP protocol integration
- **`src/mao/rag-system.py`** - RAG pipeline for document retrieval
- **`src/mao/tools.py`** - Agent tool definitions

### API Endpoints (`src/mao/api/`)
- **`api.py`** - Main FastAPI app, middleware, exception handlers
- **`agents.py`** - Agent CRUD and chat endpoints
- **`teams.py`** - Team management and supervisor coordination
- **`mcp.py`** - MCP server and tool management
- **`storage.py`** - Vector store operations
- **`db.py`** - DuckDB configuration storage
- **`models.py`** - Pydantic request/response models

### Configuration Files
- **`pyproject.toml`** - Dependencies, tool configs (black, mypy, pytest)
- **`mcp.json`** - MCP server configurations
- **`docker-compose.yml`** - Service definitions
- **`.pre-commit-config.yaml`** - Git hooks (black, ruff, mypy)

## Code Style Requirements

- **Type hints**: Required for all function signatures
- **Async/await**: Required for I/O operations
- **Docstrings**: Google-style for public APIs
- **Naming**: `snake_case` (functions/vars), `PascalCase` (classes), `UPPER_SNAKE_CASE` (constants)
- **Line length**: 88 characters (Black default)

## Package Management with uv

**Add dependency**:
```bash
uv add <package>
```

**Add dev dependency**:
```bash
uv add --dev <package>
```

**Update dependencies**:
```bash
uv sync --upgrade
```

**NEVER use pip directly** - always use `uv`

## Common Issues & Workarounds

### Issue: Import errors after checkout
**Solution**: Always run `uv sync` first

### Issue: Tests fail with "Connection refused" to Qdrant
**Solution**: Ensure Docker services are running and ready (see wait commands above)

### Issue: Integration tests timeout
**Solution**: Integration tests can take 2-5 minutes. Use `--timeout=120` flag

### Issue: uv not found
**Solution**: Install uv and add to PATH: `export PATH="$HOME/.local/bin:$PATH"`

### Issue: Type checking fails on mao.agents
**Expected**: `mao.agents` has `ignore_errors = true` in mypy config (complex LangChain types)

### Issue: Docker build fails
**Solution**: Ensure BuildKit is enabled: `export DOCKER_BUILDKIT=1`

## Agent-Specific Instructions

See `.github/instructions/` for specialized guidelines:
- `test-files.instructions.md` - Pytest best practices
- `api-endpoints.instructions.md` - FastAPI endpoint patterns
- `mcp-integration.instructions.md` - MCP protocol guidelines
- `rag-system.instructions.md` - RAG pipeline patterns

## Trust These Instructions

These instructions are validated against CI/CD workflows and reflect the actual working commands. Only search for additional information if:
1. These instructions are incomplete for your specific task
2. You encounter an error not documented here
3. You need details about internal module implementation

For any build/test/lint command, **trust and use the exact commands above** - they are proven to work in CI/CD.
