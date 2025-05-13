# src/mcp_agents/__init__.py

"""
MCP Agents: Modern agent framework with KnowledgeTree, ExperienceTree, and LangChain integration.
"""

__version__ = "1.0.0"

# Modern typing marker
# from typing import TYPE_CHECKING

# Expose the main factory function and primary classes
from .agents import create_agent, Agent, Supervisor

# Direct API access to storage, using async variants when available
from .storage import (
    KnowledgeTree, ExperienceTree, 
    VectorStoreBase, QdrantOperationError, 
    SearchResult
)

# MCP integration
from .mcp import MCPClient, ToolConfig, ServerConfig

__all__ = [
    # Core agent classes
    "create_agent",
    "Agent",
    "Supervisor",
    
    # Vector store classes
    "KnowledgeTree", 
    "ExperienceTree",
    "VectorStoreBase",
    "QdrantOperationError",
    "SearchResult",
    
    # MCP integration
    "MCPClient",
    "ToolConfig",
    "ServerConfig",
    
    # Version
    "__version__",
]
