# GitHub Copilot Configuration for MAO

This repository is configured with comprehensive instructions for GitHub Copilot coding agent to ensure high-quality, consistent code contributions.

## Configuration Files

### 1. Repository-Wide Instructions (`.github/copilot-instructions.md`)
The main instruction file provides:
- **Project Overview**: Purpose, tech stack, and architecture principles
- **Code Style**: Python standards, formatting, linting, and naming conventions
- **Key Components**: Detailed explanations of agent system, MCP integration, RAG system, storage layer, and API layer
- **Development Guidelines**: Examples for adding agents, integrating MCP servers, working with vector stores, and building endpoints
- **Testing Strategy**: Unit and integration test requirements with pytest
- **Configuration Management**: Environment variables and settings
- **Common Tasks**: Commands for development, testing, and deployment
- **Security Notes**: Best practices for secure development

### 2. Path-Specific Instructions (`.github/instructions/*.instructions.md`)
Specialized guidelines that apply automatically when working on specific file patterns:

#### API Endpoints (`api-endpoints.instructions.md`)
- **Applies to**: `**/api/**/*.py`
- **Covers**: Request/response models, endpoint design, error handling, async operations, testing, and WebSocket endpoints
- **Focus**: FastAPI best practices and Pydantic validation

#### Test Files (`test-files.instructions.md`)
- **Applies to**: `**/tests/**/*.py`
- **Covers**: Test structure, async testing, mocking, API testing, fixtures, test organization, coverage, and integration tests
- **Focus**: pytest with async support and comprehensive testing strategies

#### MCP Integration (`mcp-integration.instructions.md`)
- **Applies to**: `**/mcp.py`
- **Covers**: Protocol compliance, server management, tool discovery, configuration, error handling, tool execution, LangChain integration, and security
- **Focus**: Model Context Protocol implementation and integration

#### RAG System (`rag-system.instructions.md`)
- **Applies to**: `**/rag-system.py`
- **Covers**: Document processing, embeddings, vector storage, retrieval strategies, context assembly, response generation, optimization, and evaluation
- **Focus**: Retrieval-Augmented Generation pipeline development

### 3. Development Environment Setup (`.github/setup.yml`)
Pre-installation steps for Copilot's development environment:
- Install `uv` package manager
- Verify Python 3.11+ requirement
- Install project and development dependencies
- Setup environment file from `.env.example`
- Display project structure

### 4. Custom Agents (`.github/copilot/AGENTS.md`)
Specialized agent definitions for domain-specific tasks:
- **api-architect**: FastAPI endpoints and API design
- **mcp-integration-expert**: MCP protocol and server integration
- **agent-orchestrator**: Multi-agent coordination with LangGraph
- **llm-integration-specialist**: LLM provider integration and prompt engineering
- **vector-store-engineer**: Qdrant operations and embeddings
- **rag-developer**: RAG pipeline development
- **async-python-expert**: Async/await patterns and Python best practices
- **database-specialist**: DuckDB analytics and data modeling
- **test-engineer**: Test strategy and pytest implementation

## How Copilot Uses These Instructions

1. **Repository-wide instructions** apply to all code changes across the repository
2. **Path-specific instructions** are automatically applied when Copilot works on files matching the patterns
3. **Setup steps** ensure Copilot can build, test, and validate changes in its environment
4. **Custom agents** provide specialized expertise for specific development tasks

## Benefits

- ✅ **Consistent Code Quality**: Copilot follows project-specific standards and conventions
- ✅ **Faster Onboarding**: New contributors and Copilot understand the codebase structure immediately
- ✅ **Better Pull Requests**: Copilot can build, test, and validate changes before submitting PRs
- ✅ **Domain Expertise**: Custom agents provide specialized knowledge for complex tasks
- ✅ **Reduced Review Cycles**: Code follows conventions from the start, reducing back-and-forth

## File Structure

```
.github/
├── copilot-instructions.md          # Main repository-wide instructions
├── setup.yml                         # Development environment setup
├── copilot/
│   └── AGENTS.md                    # Custom agent definitions
├── instructions/
│   ├── api-endpoints.instructions.md
│   ├── mcp-integration.instructions.md
│   ├── rag-system.instructions.md
│   └── test-files.instructions.md
├── workflows/                        # GitHub Actions workflows
└── dependabot.yml                    # Dependency management
```

## Best Practices for Updates

When updating instructions:

1. **Be specific**: Provide clear, actionable guidance that Copilot can follow
2. **Include examples**: Show concrete code examples for patterns and practices
3. **Keep it focused**: Each instruction file should target a specific area
4. **Use YAML frontmatter**: Path-specific instructions must include `applyTo` patterns
5. **Test changes**: Verify that instruction updates improve Copilot's code quality
6. **Keep under 1000 lines**: AI models work best with focused, digestible content

## Using Custom Agents

To leverage custom agents in your work:

```
@api-architect Create a new endpoint for agent management
@test-engineer Add comprehensive tests for the RAG pipeline
@mcp-integration-expert Integrate a new filesystem MCP server
```

## Resources

- [GitHub Copilot Best Practices](https://docs.github.com/en/copilot/tutorials/coding-agent/get-the-best-results)
- [Adding Repository Custom Instructions](https://docs.github.com/en/copilot/how-tos/configure-custom-instructions/add-repository-instructions)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Guide](https://langchain-ai.github.io/langgraph/)

## Maintaining Instructions

Instructions should be updated when:
- Architecture or design patterns change
- New major dependencies are added
- Coding standards evolve
- New best practices are discovered
- Team workflows change

Always test instruction changes by observing Copilot's behavior on real tasks.
