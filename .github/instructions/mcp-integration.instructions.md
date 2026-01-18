---
applyTo: "**/mcp.py"
description: "Guidelines for Model Context Protocol integration"
---

# MCP Integration Guidelines

When working with MCP (Model Context Protocol) integration in `mao/mcp.py`, follow these guidelines:

## Protocol Compliance

1. **Follow MCP specification** strictly for all server communication
2. **Support all MCP server types**: stdio, SSE, and HTTP
3. **Implement proper error handling** for protocol violations
4. **Validate server responses** against MCP schema

## Server Management

1. **Use MCPServerManager** for centralized server lifecycle management
2. **Track server state** (connecting, connected, error, disconnected)
3. **Handle reconnection logic** for network failures
4. **Clean up resources** when servers disconnect

Example:
```python
from langchain_mcp_adapters import MCPServerManager

class ServerManager:
    async def add_server(self, server_config: dict):
        """
        Add and initialize an MCP server.
        
        Args:
            server_config: Configuration dict with type, url/command, capabilities
        """
        server_type = server_config.get("type")
        if server_type == "stdio":
            await self._init_stdio_server(server_config)
        elif server_type == "sse":
            await self._init_sse_server(server_config)
        elif server_type == "http":
            await self._init_http_server(server_config)
```

## Tool Discovery

1. **Automatically discover tools** from MCP servers on connection
2. **Register tools** with LangChain integration
3. **Handle tool schema validation**
4. **Support dynamic tool loading**

Example:
```python
async def discover_tools(self, server_id: str) -> list[Tool]:
    """
    Discover and validate tools from an MCP server.
    
    Returns:
        List of LangChain-compatible tool objects
    """
    server = self.servers.get(server_id)
    if not server:
        raise ValueError(f"Server {server_id} not found")
    
    # Discover tools via MCP protocol
    tools_response = await server.list_tools()
    
    # Validate and convert to LangChain tools
    tools = []
    for tool_def in tools_response:
        validated_tool = await self._validate_tool_schema(tool_def)
        langchain_tool = self._convert_to_langchain_tool(validated_tool)
        tools.append(langchain_tool)
    
    return tools
```

## Configuration Management

1. **Load server configs** from `mcp.json` or environment
2. **Support dynamic configuration** updates
3. **Validate configuration** before server initialization
4. **Provide configuration templates** for common server types

Example server configuration:
```json
{
  "servers": [
    {
      "id": "filesystem",
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/workspace"],
      "capabilities": ["tools"]
    },
    {
      "id": "github",
      "type": "http",
      "url": "https://api.github.com/mcp",
      "headers": {
        "Authorization": "Bearer ${GITHUB_TOKEN}"
      },
      "capabilities": ["tools", "resources"]
    }
  ]
}
```

## Error Handling

1. **Handle connection failures** gracefully
2. **Implement retry logic** with exponential backoff
3. **Log protocol violations** for debugging
4. **Provide fallback behavior** when servers are unavailable

Example:
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
async def connect_server(self, server_config: dict):
    """Connect to MCP server with retry logic."""
    try:
        if server_config["type"] == "stdio":
            return await self._connect_stdio(server_config)
        # ... other types
    except ConnectionError as e:
        logger.error(f"Failed to connect to MCP server: {e}")
        raise
```

## Tool Execution

1. **Validate tool parameters** before execution
2. **Handle tool execution timeouts**
3. **Sanitize tool inputs** to prevent injection attacks
4. **Log tool usage** for monitoring

Example:
```python
async def execute_tool(
    self,
    tool_name: str,
    parameters: dict,
    timeout: int = 30
) -> dict:
    """
    Execute a tool from an MCP server.
    
    Args:
        tool_name: Name of the tool to execute
        parameters: Tool parameters (validated)
        timeout: Execution timeout in seconds
    
    Returns:
        Tool execution result
    """
    # Validate parameters against tool schema
    tool = self.tools.get(tool_name)
    if not tool:
        raise ValueError(f"Tool {tool_name} not found")
    
    validated_params = tool.validate_parameters(parameters)
    
    # Execute with timeout
    try:
        result = await asyncio.wait_for(
            tool.execute(validated_params),
            timeout=timeout
        )
        return result
    except asyncio.TimeoutError:
        raise TimeoutError(f"Tool {tool_name} execution timed out")
```

## Integration with LangChain

1. **Use `langchain-mcp-adapters`** for seamless integration
2. **Convert MCP tools** to LangChain tool format
3. **Handle tool results** in LangChain chains
4. **Support streaming responses** when available

Example:
```python
from langchain_mcp_adapters import MCPServerTool

def convert_to_langchain_tool(self, mcp_tool: dict) -> Tool:
    """Convert MCP tool definition to LangChain tool."""
    return MCPServerTool(
        name=mcp_tool["name"],
        description=mcp_tool["description"],
        args_schema=mcp_tool.get("inputSchema"),
        func=lambda **kwargs: self.execute_tool(mcp_tool["name"], kwargs)
    )
```

## Resource Management

1. **Support MCP resources** (files, databases, APIs)
2. **Implement resource caching** when appropriate
3. **Handle resource permissions** and access control
4. **Clean up resources** on server disconnect

## Testing

1. **Mock MCP servers** for unit tests
2. **Test each server type** (stdio, SSE, HTTP)
3. **Test connection failures** and retry logic
4. **Test tool discovery and execution**
5. **Use integration tests** with real MCP servers when possible

Example:
```python
@pytest.mark.asyncio
async def test_mcp_server_connection():
    manager = MCPServerManager()
    
    server_config = {
        "type": "stdio",
        "command": "mock-mcp-server",
        "args": []
    }
    
    # Test connection
    await manager.add_server(server_config)
    assert manager.is_connected(server_config["id"])
    
    # Test tool discovery
    tools = await manager.discover_tools(server_config["id"])
    assert len(tools) > 0
```

## Security

1. **Validate server configurations** before loading
2. **Sanitize all inputs** to MCP servers
3. **Use environment variables** for sensitive credentials
4. **Never log sensitive data** (tokens, credentials)
5. **Implement rate limiting** for tool executions
