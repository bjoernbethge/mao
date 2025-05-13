"""
MCP Agents API Application.
Provides a REST API for managing and interacting with MCP agents.
"""

# Import and re-export the API instance and class
from .api import api, MCPAgentsAPI

# Export important API components
__all__ = ["api", "MCPAgentsAPI"]
