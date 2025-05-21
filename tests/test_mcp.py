"""
Tests for MCPClient.
Production-ready, async, and robust implementation.
"""

import pytest
import asyncio
import logging
from unittest.mock import patch, MagicMock, mock_open
from mao.mcp import (
    MCPClient,
    create_mcp_client_from_api,
    create_mcp_client_from_api_async,
)

# Import pytest_asyncio if available
try:
    import pytest_asyncio

    ASYNCIO_FIXTURE = pytest_asyncio.fixture
except ImportError:
    ASYNCIO_FIXTURE = pytest.fixture  # type: ignore


@pytest.mark.asyncio
async def test_mcp_client_tool_states(mcp_client):
    """Test MCPClient tool enablement logic."""
    # Set initial tool states
    tools_to_test = ["test_tool1", "test_tool2"]

    # Enable tools and verify
    for tool in tools_to_test:
        await asyncio.to_thread(mcp_client.set_tool_enabled, tool, True)
        assert await asyncio.to_thread(
            mcp_client.is_tool_enabled, tool
        ), f"Tool {tool} should be enabled"

    # Disable tools and verify
    for tool in tools_to_test:
        await asyncio.to_thread(mcp_client.set_tool_enabled, tool, False)
        assert not await asyncio.to_thread(
            mcp_client.is_tool_enabled, tool
        ), f"Tool {tool} should be disabled"

    # Test with non-existent tool
    assert not await asyncio.to_thread(
        mcp_client.is_tool_enabled, "non_existent_tool"
    ), "Non-existent tool should report as disabled"


@pytest.mark.asyncio
async def test_mcp_client_reload(mcp_client):
    """Test configuration reloading."""
    # Get initial state
    initial_servers = await asyncio.to_thread(mcp_client.list_servers)

    # Set some tool states before reload
    test_tools = ["tool_before_reload1", "tool_before_reload2"]
    for tool in test_tools:
        await asyncio.to_thread(mcp_client.set_tool_enabled, tool, True)

    # Reload configuration
    await asyncio.to_thread(mcp_client.reload)

    # Verify servers maintained after reload
    current_servers = await asyncio.to_thread(mcp_client.list_servers)
    assert (
        current_servers == initial_servers
    ), "Server list should be maintained after reload"

    # Verify tool states maintained
    for tool in test_tools:
        assert await asyncio.to_thread(
            mcp_client.is_tool_enabled, tool
        ), f"Tool {tool} should remain enabled after reload"


@pytest.mark.asyncio
async def test_mcp_client_get_tools(mcp_client):
    """Test getting tools from MCP servers."""
    # Test getting tools in a context
    async with mcp_client.session() as client_in_context:
        # get_tools is a synchronous method
        tools = client_in_context.get_tools()
        assert tools is not None, "No tools found from MCP servers"
        assert isinstance(tools, list), "get_tools should return a list"

        # Log detailed information about found tools
        logging.info(f"Found: {len(tools)} tools from MCP servers")

        for i, tool in enumerate(tools):
            tool_info = {
                "name": getattr(tool, "name", "Unknown"),
                "description": getattr(tool, "description", "No description"),
                "type": type(tool).__name__,
            }
            # Additional properties if available
            if hasattr(tool, "args_schema"):
                tool_info["args_schema"] = str(tool.args_schema)

            logging.info(f"Tool {i+1}: {tool_info}")

    # Test tool activation for specific tools if available
    if tools:
        test_tool = tools[0].name
        await asyncio.to_thread(mcp_client.set_tool_enabled, test_tool, True)
        assert await asyncio.to_thread(mcp_client.is_tool_enabled, test_tool)
        await asyncio.to_thread(mcp_client.set_tool_enabled, test_tool, False)
        assert not await asyncio.to_thread(mcp_client.is_tool_enabled, test_tool)


@pytest.mark.asyncio
async def test_mcp_client_health_check():
    """Test MCPClient health check functionality."""
    client = MCPClient()

    # Test health check
    health_status = await client.health_check()

    # Verify health status is a dictionary
    assert isinstance(health_status, dict)

    # Each server should have a boolean status
    for server, status in health_status.items():
        assert isinstance(status, bool)


def test_mcp_client_server_management():
    """Test MCPClient server management functionality."""
    client = MCPClient()

    # Get all servers
    all_servers = client.list_servers()

    if all_servers:
        test_server = all_servers[0]

        # Test disabling a server
        client.disable_server(test_server)
        assert test_server not in client.list_active_servers()

        # Test enabling a server
        client.enable_server(test_server)
        assert test_server in client.list_active_servers()

        # Test error on enabling non-existent server
        with pytest.raises(ValueError):
            client.enable_server("non_existent_server")


@pytest.mark.asyncio
async def test_mcp_client_session():
    """Test MCPClient session context manager."""
    client = MCPClient()

    # Test using the session context manager
    async with client.session() as session_client:
        assert session_client is client


@patch("httpx.Client.get")
def test_create_mcp_client_from_api(mock_get):
    """Test creating MCPClient from API."""
    # Mock API response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "mcpServers": {
            "test_server": {
                "transport": "stdio",
                "command": "python",
                "args": ["-m", "test_server"],
            }
        }
    }
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    # Test creating client from API
    client = create_mcp_client_from_api("http://test-api")

    # Verify client was created with the mock config
    assert "test_server" in client.list_servers()

    # Verify API was called correctly
    mock_get.assert_called_once_with("http://test-api/mcp/config?enabled_only=true")


@pytest.mark.asyncio
@patch("httpx.AsyncClient.get")
async def test_create_mcp_client_from_api_async(mock_get):
    """Test creating MCPClient from API asynchronously."""
    # Mock API response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "mcpServers": {
            "test_server": {
                "transport": "stdio",
                "command": "python",
                "args": ["-m", "test_server"],
            }
        }
    }
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    # Test creating client from API asynchronously
    client = await create_mcp_client_from_api_async("http://test-api")

    # Verify client was created with the mock config
    assert "test_server" in client.list_servers()

    # Verify API was called correctly
    mock_get.assert_called_once_with("http://test-api/mcp/config?enabled_only=true")


@patch(
    "builtins.open",
    new_callable=mock_open,
    read_data='{"mcpServers": {"test_server": {"transport": "stdio", "command": "python", "args": ["-m", "test_server"]}}}',
)
def test_mcp_client_load_config(mock_file):
    """Test MCPClient loading configuration from file."""
    # Create client with mocked file
    client = MCPClient()

    # Verify config was loaded correctly
    assert "test_server" in client.config["mcpServers"]
    assert client.config["mcpServers"]["test_server"]["transport"] == "stdio"


@patch("builtins.open", side_effect=FileNotFoundError)
def test_mcp_client_load_config_file_not_found(mock_file):
    """Test MCPClient handling file not found error."""
    # Expect FileNotFoundError when config file doesn't exist
    with pytest.raises(FileNotFoundError):
        MCPClient()


@patch("builtins.open", new_callable=mock_open, read_data='{"invalid_json":')
def test_mcp_client_load_config_invalid_json(mock_file):
    """Test MCPClient handling invalid JSON in config file."""
    # Expect ValueError for invalid JSON
    with pytest.raises(ValueError):
        MCPClient()


@patch("builtins.open", new_callable=mock_open, read_data='{"not_mcpServers": {}}')
def test_mcp_client_load_config_missing_key(mock_file):
    """Test MCPClient handling missing mcpServers key in config file."""
    # Expect ValueError for missing mcpServers key
    with pytest.raises(ValueError):
        MCPClient()


def test_mcp_client_custom_config():
    """Test MCPClient with custom config."""
    # Create client with custom config
    custom_config = {
        "mcpServers": {
            "custom_server": {
                "transport": "stdio",
                "command": "python",
                "args": ["-m", "custom_server"],
            }
        }
    }

    client = MCPClient(config=custom_config)

    # Verify custom config was used
    assert "custom_server" in client.list_servers()
    assert "custom_server" in client.list_active_servers()


def test_mcp_client_custom_tool_states():
    """Test MCPClient with custom initial tool states."""
    # Create client with custom tool states
    custom_tool_states = {"tool1": True, "tool2": False}

    client = MCPClient(
        config={"mcpServers": {}},  # Empty config to avoid file loading
        initial_tool_states=custom_tool_states,
    )

    # Verify custom tool states were used
    assert client.is_tool_enabled("tool1") is True
    assert client.is_tool_enabled("tool2") is False
