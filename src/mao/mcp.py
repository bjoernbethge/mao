"""
MCPClient and MCP integration for agent framework.
Production-ready, typed, and DRY. All docstrings in English.
"""

from typing import Optional, Dict, Any, List, TypedDict, Union, Callable, TypeVar, Awaitable
import os
import json
import asyncio
import httpx
from urllib.parse import urljoin
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
import logging
import pathlib
from contextlib import asynccontextmanager

class ToolConfig(TypedDict):
    """Configuration for a tool"""
    enabled: bool
    server: Optional[str]

class ServerConfig(TypedDict):
    """Configuration for an MCP server"""
    transport: str
    command: Optional[str]
    args: Optional[List[str]]
    url: Optional[str]
    headers: Optional[Dict[str, str]]
    timeout: Optional[int]
    sse_read_timeout: Optional[int]
    env: Optional[Dict[str, str]]
    cwd: Optional[str]
    errlog: Optional[str]

class MCPClient(MultiServerMCPClient):
    """
    Manages MCP server connections and tool enablement states.
    
    Configuration is loaded from 'mcp.json' located in the project's root directory,
    or from a specified environment variable MCP_CONFIG_PATH.
    
    This client uses the 'mcpServers' configuration key and supports:
    - Async operations for all methods
    - Server management (enable/disable)
    - Tool states tracking
    """
    def __init__(self, 
                 config: Optional[dict] = None, 
                 initial_tool_states: Optional[Dict[str, bool]] = None,
                 config_path: Optional[str] = None):
        self.project_root = pathlib.Path(__file__).resolve().parent.parent.parent 
        
        # Use config_path parameter, environment variable, or default path
        self.config_file_path = (
            pathlib.Path(config_path) if config_path 
            else pathlib.Path(os.environ.get("MCP_CONFIG_PATH", "")) if os.environ.get("MCP_CONFIG_PATH") 
            else self.project_root / "mcp.json"
        )
        
        self.config = config if config is not None else self._load_config()
        self.tool_states = initial_tool_states or {} 
        connections = self._build_connections()
        super().__init__(connections=connections)
        
        # Track active servers to handle dynamic enabling/disabling
        self._active_servers = set(self.list_servers())

    def _load_config(self) -> dict:
        """
        Load MCP config from mcp.json located in the project root (self.config_file_path).
        Raises FileNotFoundError if the file is not found, or ValueError if JSON is malformed or 'mcpServers' key is missing.
        """
        config_path = self.config_file_path
        
        logging.info(f"Attempting to load MCP config from: {config_path}")

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                loaded_config = json.load(f)
            logging.info(f"Successfully loaded MCP config from {config_path}")

            if "mcpServers" not in loaded_config:
                msg = f"'mcpServers' key is mandatory but was not found in {config_path}."
                logging.error(msg)
                raise ValueError(msg)
            return loaded_config
        except FileNotFoundError:
            logging.error(f"MCP configuration file not found: {config_path}")
            raise
        except json.JSONDecodeError as e:
            logging.error(f"Failed to decode JSON from {config_path}: {e}")
            raise ValueError(f"Invalid JSON format in {config_path}: {e}") from e

    def _build_connections(self) -> Dict[str, Any]:
        """
        Constructs the connection dictionary for MultiServerMCPClient 
        based on the 'mcpServers' section of the loaded configuration.
        
        Supports 'stdio', 'sse', 'streamable_http', and 'websocket' transports.
        """
        if not self.config: 
            logging.warning("Attempting to build connections but self.config is not yet populated. Trying to load now.")
            self.config = self._load_config()

        servers = self.config.get("mcpServers", {})
        if not servers:
            logging.warning(f"No 'mcpServers' defined in the configuration from {self.config_file_path}. No connections will be built.")

        connections = {}
        for name, server_config in servers.items():
            transport = server_config.get("transport", "stdio")
            conn_details = {"transport": transport}

            if transport in ["sse", "streamable_http", "websocket"]:
                if "url" not in server_config:
                    logging.warning(f"Server '{name}' with transport '{transport}' is missing required 'url'. Skipping.")
                    continue
                conn_details["url"] = server_config["url"]
                if "headers" in server_config:
                    conn_details["headers"] = server_config["headers"]
                if "timeout" in server_config: 
                    conn_details["timeout"] = server_config["timeout"]
                if transport in ["sse", "streamable_http"] and "sse_read_timeout" in server_config: 
                    conn_details["sse_read_timeout"] = server_config["sse_read_timeout"]
            elif transport == "stdio":
                if "command" not in server_config or "args" not in server_config:
                    logging.warning(f"Server '{name}' with transport 'stdio' is missing required 'command' or 'args'. Skipping.")
                    continue
                conn_details["command"] = server_config["command"]
                conn_details["args"] = server_config["args"]
                if "env" in server_config:
                    conn_details["env"] = server_config["env"]
                if "cwd" in server_config:
                    conn_details["cwd"] = server_config["cwd"]
                if "errlog" in server_config:
                    conn_details["errlog"] = server_config["errlog"]
            else:
                logging.warning(f"Unsupported transport '{transport}' for server '{name}'. Skipping.")
                continue
            
            connections[name] = conn_details
        return connections

    def list_servers(self) -> List[str]:
        """Return a list of all configured server names."""
        return list(self.config.get("mcpServers", {}).keys())

    def list_active_servers(self) -> List[str]:
        """Return a list of currently active server names."""
        return list(self._active_servers)

    def enable_server(self, name: str) -> None:
        """Enable a specific MCP server by name."""
        if name not in self.list_servers():
            raise ValueError(f"Server {name} does not exist.")
        self._active_servers.add(name)

    def disable_server(self, name: str) -> None:
        """Disable a specific MCP server by name."""
        self._active_servers.discard(name)

    def set_tool_enabled(self, tool_name: str, enabled: bool) -> None:
        """Sets the enablement state of a specific tool."""
        self.tool_states[tool_name] = enabled

    def is_tool_enabled(self, tool_name: str) -> bool:
        """Checks if a specific tool is enabled. Returns False if not set."""
        return self.tool_states.get(tool_name, False)

    def reload(self) -> None:
        """Reloads the configuration from the mcp.json file located in the project root."""
        logging.info(f"Reloading MCP configuration from {self.config_file_path}")
        self.config = self._load_config() 
        self.connections = self._build_connections()
        
        # Reset active servers to all configured servers after reload
        self._active_servers = set(self.list_servers())
        
    async def reload_async(self) -> None:
        """Asynchronously reloads the configuration from the mcp.json file located in the project root."""
        self.reload()
        # If there are any async operations needed in the future, they would go here

    async def health_check(self) -> Dict[str, bool]:
        """
        Check health of all active servers.
        
        Returns:
            Dict mapping server names to health status
        """
        results: Dict[str, bool] = {}
        active_servers = self.list_active_servers()
        
        for server_name in active_servers:
            try:
                # Try to ping the server
                if hasattr(self, 'ping_server'):
                    await self.ping_server(server_name)
                    results[server_name] = True
                else:
                    # Fall back to checking if the server is in connections
                    results[server_name] = server_name in self.connections
            except Exception as e:
                logging.warning(f"Health check failed for server {server_name}: {e}")
                results[server_name] = False
                
        return results
        
    def load_tools(self, server_name: str) -> List[Dict[str, Any]]:
        """
        Load tools from a specific server.
        
        Args:
            server_name: Name of the server to load tools from
            
        Returns:
            List of tool definitions
        
        Raises:
            ValueError: If the server does not exist
        """
        if server_name not in self.list_servers():
            raise ValueError(f"Server {server_name} does not exist.")
            
        return load_mcp_tools(self, server_name)

    @asynccontextmanager
    async def session(self):
        """Context manager for an MCP session with proper cleanup."""
        try:
            async with self:
                yield self
        finally:
            # Ensure cleanup happens
            if hasattr(self, 'async_shutdown'):
                await self.async_shutdown()

# Type variable for generic function
T = TypeVar('T')

async def _fetch_config_from_api(
    api_base_url: str,
    enabled_only: bool,
    timeout: float,
    fetch_func: Callable[[str, float], Awaitable[Dict[str, Any]]]
) -> Dict[str, Any]:
    """
    Helper function to fetch configuration from API.
    
    Args:
        api_base_url: Base URL of the MCP Agents API
        enabled_only: Whether to only include enabled servers
        timeout: Request timeout in seconds
        fetch_func: Function to use for fetching (sync or async)
        
    Returns:
        API configuration
    """
    config_url = urljoin(api_base_url, f"/mcp/config?enabled_only={str(enabled_only).lower()}")
    
    try:
        config = await fetch_func(config_url, timeout)
        
        if "mcpServers" not in config:
            raise ValueError("API response missing required 'mcpServers' key")
            
        return config
    except httpx.HTTPError as e:
        logging.error(f"Failed to fetch MCP configuration from API: {e}")
        raise
    except (ValueError, json.JSONDecodeError) as e:
        logging.error(f"Invalid API response format: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error fetching MCP configuration from API: {e}")
        raise

def create_mcp_client_from_api(
    api_base_url: str = "http://localhost:8000",
    enabled_only: bool = True,
    timeout: float = 10.0
) -> MCPClient:
    """
    Creates an MCPClient instance by fetching configuration from the API.
    
    Args:
        api_base_url: Base URL of the MCP Agents API
        enabled_only: Whether to only include enabled servers
        timeout: Request timeout in seconds
        
    Returns:
        Configured MCPClient instance
    
    Raises:
        httpx.HTTPError: If the API request fails
        ValueError: If the API response is invalid or missing required data
    """
    logging.info(f"Creating MCPClient from API at {api_base_url}")
    
    async def fetch_sync(url: str, timeout_val: float) -> Dict[str, Any]:
        with httpx.Client(timeout=timeout_val) as client:
            response = client.get(url)
            response.raise_for_status()
            return response.json()
    
    # Use asyncio to run the async helper function
    config = asyncio.run(_fetch_config_from_api(api_base_url, enabled_only, timeout, fetch_sync))
    
    # Initialize MCPClient with fetched configuration
    client = MCPClient(config=config)
    logging.info(f"Successfully created MCPClient with {len(client.list_servers())} servers from API")
    
    return client

async def create_mcp_client_from_api_async(
    api_base_url: str = "http://localhost:8000",
    enabled_only: bool = True,
    timeout: float = 10.0
) -> MCPClient:
    """
    Asynchronously creates an MCPClient instance by fetching configuration from the API.
    Uses native async HTTP requests with httpx.
    
    Args:
        api_base_url: Base URL of the MCP Agents API
        enabled_only: Whether to only include enabled servers
        timeout: Request timeout in seconds
        
    Returns:
        Configured MCPClient instance
    """
    logging.info(f"Asynchronously creating MCPClient from API at {api_base_url}")
    
    async def fetch_async(url: str, timeout_val: float) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=timeout_val) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.json()
    
    config = await _fetch_config_from_api(api_base_url, enabled_only, timeout, fetch_async)
    
    # Initialize MCPClient with fetched configuration
    client = MCPClient(config=config)
    logging.info(f"Successfully created MCPClient with {len(client.list_servers())} servers from API")
    
    return client
