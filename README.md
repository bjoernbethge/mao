# MAO - MCP Agent Orchestra

A modern framework for orchestrating AI agents using the Model Context Protocol (MCP).

## Overview

MAO (MCP Agent Orchestra) is a comprehensive infrastructure for creating, managing, and orchestrating AI agents. It provides a robust platform that integrates various Large Language Models (LLMs) with tools through the Model Context Protocol, enabling sophisticated agent interactions and knowledge management.

## Key Features

- **Flexible Agent Framework**: Create agents powered by various LLM providers (OpenAI, Anthropic, Ollama)
- **Team Orchestration**: Organize agents into teams with supervisor coordination
- **Knowledge Management**: Store and retrieve agent knowledge and experiences using vector databases
- **Tool Integration**: Dynamically load and manage tools via the MCP protocol
- **REST API**: Complete management interface for all system resources

## Architecture

MAO consists of several core components:

1. **Agent Framework** (`agents.py`):
   - LLM integration with OpenAI, Anthropic, and Ollama
   - Advanced features like streaming, tool integration, and memory
   - Support for supervisor agents to coordinate teams

2. **MCP Integration** (`mcp.py`):
   - Client for the Model Context Protocol
   - Management of connections to various MCP servers
   - Dynamic loading and management of tools

3. **Storage System** (`storage.py`):
   - Qdrant vector database for semantic search
   - KnowledgeTree and ExperienceTree for agent knowledge and experience
   - Fully asynchronous API for database operations

4. **REST API** (`api/` folder):
   - FastAPI-based REST interface for managing agents, teams, and tools
   - CRUD operations for all resources
   - DuckDB for lightweight configuration storage

5. **Configuration Management** (`db.py`):
   - Configuration management for agents, teams, servers, and tools
   - Asynchronous database interface
   - Singleton pattern for database connections

## Technical Details

MAO follows modern Python best practices:
- Complete typing with Type Hints
- Asynchronous programming with asyncio
- Error handling and retry logic
- Modular design with clear responsibilities

## Getting Started

1. Install dependencies:
   ```bash
   uv venv
   uv pip install -e .
   ```

2. Configure your environment:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

3. Run the API server:
   ```bash
   uvicorn src.mcp_agents.api.api:api --reload
   ```

4. Access the API documentation at `http://localhost:8000/docs`

## License

[License information]
