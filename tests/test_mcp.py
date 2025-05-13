"""
Tests für MCPClient.
Produktionsreif, asynchron, robust. Alle Docstrings auf Englisch.
"""

import pytest
import asyncio
import os
import json
from mao.mcp import MCPClient
import logging

# Import pytest_asyncio if available for the fixture decorator, otherwise use pytest.fixture
try:
    import pytest_asyncio
    ASYNCIO_FIXTURE = pytest_asyncio.fixture
except ImportError:
    ASYNCIO_FIXTURE = pytest.fixture # type: ignore

@ASYNCIO_FIXTURE # Use @pytest_asyncio.fixture if available, else @pytest.fixture
async def mcp_client(): # Async fixture
    """
    Fixture to create an MCPClient and ensure its resources are released.
    It will load mcp.json from the project root.
    Assumes a valid mcp.json exists at the project root for these tests to pass.
    """
    client = None
    try:
        client = MCPClient(initial_tool_states={"initial_test_tool": True})
        # MCPClient is now fully initialized here, including loading mcp.json
        # and building connection blueprints. Actual connections/subprocesses 
        # are typically established when __aenter__ is called (e.g. via async with)
        # or a method like get_tools() is invoked that requires a session.
        yield client 
    finally:
        if client and hasattr(client, 'async_shutdown'):
            logging.info("MCPClient fixture: Attempting to shut down MCPClient...")
            try:
                await client.async_shutdown()
                logging.info("MCPClient fixture: MCPClient shutdown successful.")
            except Exception as e:
                logging.error(f"MCPClient fixture: Error during async_shutdown: {e}", exc_info=True)
        elif client:
            logging.warning("MCPClient fixture: MCPClient instance does not have async_shutdown method or it failed prior to finally.")

@pytest.mark.asyncio
async def test_mcp_client_tool_states(mcp_client):
    """
    Test MCPClient tool enablement logic.
    """
    # Set initial tool states
    tools_to_test = ["test_tool1", "test_tool2"]
    
    # Enable tools and verify
    for tool in tools_to_test:
        await asyncio.to_thread(mcp_client.set_tool_enabled, tool, True)
        assert await asyncio.to_thread(mcp_client.is_tool_enabled, tool), f"Tool {tool} should be enabled"
    
    # Disable tools and verify
    for tool in tools_to_test:
        await asyncio.to_thread(mcp_client.set_tool_enabled, tool, False)
        assert not await asyncio.to_thread(mcp_client.is_tool_enabled, tool), f"Tool {tool} should be disabled"
    
    # Test with non-existent tool
    assert not await asyncio.to_thread(mcp_client.is_tool_enabled, "non_existent_tool"), "Non-existent tool should report as disabled"


@pytest.mark.asyncio
async def test_mcp_client_server_operations(mcp_client):
    """
    Test server listing, activation, and deactivation.
    """
    # List all configured servers
    all_servers = await asyncio.to_thread(mcp_client.list_servers)
    assert isinstance(all_servers, list), "list_servers() should return a list"
    
    # Skip test if no servers are configured
    if not all_servers:
        logging.warning("No servers found in mcp_config.json, skipping server operations test")
        return
        
    # Initially all servers should be active
    active_servers = await asyncio.to_thread(mcp_client.list_active_servers)
    assert isinstance(active_servers, list), "list_active_servers() should return a list"
    assert set(active_servers) == set(all_servers), "All servers should initially be active"
    
    # Test disabling a server
    test_server = all_servers[0]
    await asyncio.to_thread(mcp_client.disable_server, test_server)
    
    active_after_disable = await asyncio.to_thread(mcp_client.list_active_servers)
    assert test_server not in active_after_disable, f"Server {test_server} should be inactive after disable_server()"
    
    # Test re-enabling the server
    await asyncio.to_thread(mcp_client.enable_server, test_server)
    
    active_after_enable = await asyncio.to_thread(mcp_client.list_active_servers)
    assert test_server in active_after_enable, f"Server {test_server} should be active after enable_server()"
    assert set(active_after_enable) == set(all_servers), "All servers should be active again"
    
    # Test error on non-existent server
    with pytest.raises(ValueError):
        await asyncio.to_thread(mcp_client.enable_server, "this_server_does_not_exist")


@pytest.mark.asyncio
async def test_mcp_client_reload(mcp_client):
    """
    Test configuration reloading.
    """
    # Get initial state
    initial_servers = await asyncio.to_thread(mcp_client.list_servers)
    initial_connections = mcp_client.connections.copy() if hasattr(mcp_client, 'connections') else {}
    
    # Set some tool states before reload
    test_tools = ["tool_before_reload1", "tool_before_reload2"]
    for tool in test_tools:
        await asyncio.to_thread(mcp_client.set_tool_enabled, tool, True)
        
    # Reload configuration
    await asyncio.to_thread(mcp_client.reload)
    
    # Verify servers maintained after reload
    current_servers = await asyncio.to_thread(mcp_client.list_servers)
    assert current_servers == initial_servers, "Server list should be maintained after reload"
    
    # Verify tool states maintained
    for tool in test_tools:
        assert await asyncio.to_thread(mcp_client.is_tool_enabled, tool), f"Tool {tool} should remain enabled after reload"


@pytest.mark.asyncio
async def test_mcp_client_get_tools(mcp_client):
    """
    Test that verifies the integration with MCP servers using get_tools.
    This test requires real MCP servers to be configured and available.
    """
    # Prüfe aktive Server
    all_servers = await asyncio.to_thread(mcp_client.list_servers)
    assert all_servers, "Dieser Test erfordert konfigurierte MCP-Server. Keine Server gefunden."
    
    active_servers = await asyncio.to_thread(mcp_client.list_active_servers)
    logging.info(f"Verfügbare Server: {active_servers}")
    assert active_servers, "Keine aktiven Server gefunden. Dieser Test erfordert funktionierende MCP-Server."
    
    # Testen mit echten MCP-Servern
    async with mcp_client as client_in_context:
        # get_tools ist eine synchrone Methode in MultiServerMCPClient
        tools = client_in_context.get_tools()
        # Der Test muss fehlschlagen, wenn keine Tools gefunden wurden
        assert tools is not None, "Keine Tools von MCP-Servern gefunden"
        assert isinstance(tools, list), "get_tools sollte eine Liste zurückgeben"
        
        # Detaillierte Informationen über die gefundenen Tools loggen
        logging.info(f"Gefunden: {len(tools)} Tools von MCP-Servern")
        print(f"\n=== Gefundene MCP-Tools ({len(tools)}) ===")
        for i, tool in enumerate(tools):
            tool_info = {
                "name": getattr(tool, "name", "Unbekannt"),
                "description": getattr(tool, "description", "Keine Beschreibung"),
                "type": type(tool).__name__,
            }
            # Weitere Eigenschaften, falls vorhanden
            if hasattr(tool, "args_schema"):
                tool_info["args_schema"] = str(tool.args_schema)
            
            print(f"{i+1}. {tool_info['name']} ({tool_info['type']})")
            print(f"   Beschreibung: {tool_info['description']}")
            if "args_schema" in tool_info:
                print(f"   Schema: {tool_info['args_schema']}")
            print()
            
            # Auch ins Log schreiben
            logging.info(f"Tool {i+1}: {tool_info}")
    
    # Optional: Tool-Aktivierung testen für spezifische Tools, falls vorhanden
    if tools:
        test_tool = tools[0].name
        await asyncio.to_thread(mcp_client.set_tool_enabled, test_tool, True)
        assert await asyncio.to_thread(mcp_client.is_tool_enabled, test_tool)
        await asyncio.to_thread(mcp_client.set_tool_enabled, test_tool, False)
        assert not await asyncio.to_thread(mcp_client.is_tool_enabled, test_tool) 