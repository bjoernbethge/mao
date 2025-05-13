"""
Tests for the MCP API endpoints.
Test CRUD operations on servers, tools, and related functionality.
"""

import random
import string
import httpx


def generate_unique_name(prefix="Test"):
    """Generate a unique name with a random suffix."""
    random_suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
    return f"{prefix}_{random_suffix}"


def test_create_server(api_test_client):
    """Test creating a new server."""
    client, _ = api_test_client

    # Testdaten für die Server-Erstellung mit eindeutigem Namen
    server_name = generate_unique_name("TestServer")
    server_data = {
        "name": server_name,
        "transport": "stdio",
        "enabled": True,
        "command": "python",
        "args": ["-m", "mcp_agents.server"],
        "env_vars": {"TEST_ENV": "test_value"},
    }

    response = client.post("/mcp/servers", json=server_data)

    assert response.status_code == 201
    data = response.json()
    assert data["name"] == server_data["name"]
    assert data["transport"] == server_data["transport"]
    assert data["enabled"] == server_data["enabled"]
    assert data["command"] == server_data["command"]
    assert data["args"] == server_data["args"]
    assert data["env_vars"] == server_data["env_vars"]
    assert "id" in data
    assert data["id"].startswith("server_")


def test_list_servers(api_test_client):
    """Test listing all servers."""
    client, _ = api_test_client

    # Erstelle erst einen Server mit eindeutigem Namen
    server_name = generate_unique_name("ServerList")
    server_data = {
        "name": server_name,
        "transport": "websocket",
        "enabled": True,
        "url": "ws://localhost:8080",
    }

    create_response = client.post("/mcp/servers", json=server_data)
    assert create_response.status_code == 201

    # Hole die Liste aller Server
    list_response = client.get("/mcp/servers")
    assert list_response.status_code == 200

    servers = list_response.json()
    assert isinstance(servers, list)
    assert len(servers) >= 1

    # Finde den erstellten Server in der Liste
    found = False
    for server in servers:
        if server["name"] == server_data["name"]:
            found = True
            break

    assert found, "Der erstellte Server wurde nicht in der Liste gefunden"


def test_create_tool(api_test_client):
    """Test creating a new tool."""
    client, _ = api_test_client

    # Erstelle erst einen Server für den Tool mit eindeutigem Namen
    server_name = generate_unique_name("ServerForTool")
    server_data = {"name": server_name, "transport": "stdio", "enabled": True}

    server_response = client.post("/mcp/servers", json=server_data)
    assert server_response.status_code == 201
    server_id = server_response.json()["id"]

    # Testdaten für die Tool-Erstellung
    tool_data = {
        "name": "Test Tool",
        "enabled": True,
        "server_id": server_id,
        "description": "A test tool for API testing",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"],
        },
    }

    response = client.post("/mcp/tools", json=tool_data)

    assert response.status_code == 201
    data = response.json()
    assert data["name"] == tool_data["name"]
    assert data["enabled"] == tool_data["enabled"]
    assert data["server_id"] == server_id
    assert data["description"] == tool_data["description"]
    assert "id" in data
    assert data["id"].startswith("tool_")


def test_list_tools(api_test_client):
    """Test listing all tools."""
    client, _ = api_test_client

    # Erstelle erst einen Server mit eindeutigem Namen
    server_name = generate_unique_name("ToolListServer")
    server_data = {"name": server_name, "transport": "stdio", "enabled": True}

    server_response = client.post("/mcp/servers", json=server_data)
    assert server_response.status_code == 201
    server_id = server_response.json()["id"]

    # Erstelle ein Tool
    tool_data = {"name": "Test Tool for List", "enabled": True, "server_id": server_id}

    create_response = client.post("/mcp/tools", json=tool_data)
    assert create_response.status_code == 201

    # Hole die Liste aller Tools
    list_response = client.get("/mcp/tools")
    assert list_response.status_code == 200

    tools = list_response.json()
    assert isinstance(tools, list)
    assert len(tools) >= 1

    # Finde das erstellte Tool in der Liste
    found = False
    for tool in tools:
        if tool["name"] == tool_data["name"]:
            found = True
            break

    assert found, "Das erstellte Tool wurde nicht in der Liste gefunden"

    # Teste das Filtern nach Server-ID
    filtered_response = client.get(f"/mcp/tools?server_id={server_id}")
    assert filtered_response.status_code == 200

    filtered_tools = filtered_response.json()
    assert len(filtered_tools) >= 1
    assert all(tool["server_id"] == server_id for tool in filtered_tools)


def test_get_mcp_config(api_test_client):
    """Test getting MCP configuration."""
    client, _ = api_test_client

    # Erstelle einen Server für die Konfiguration mit eindeutigem Namen
    server_name = generate_unique_name("ConfigServer")
    server_data = {
        "name": server_name,
        "transport": "stdio",
        "command": "python",
        "args": ["-m", "server"],
        "enabled": True,
    }

    server_response = client.post("/mcp/servers", json=server_data)
    assert server_response.status_code == 201

    # Teste den Konfigurations-Endpunkt
    config_response = client.get("/mcp/config")
    assert config_response.status_code == 200

    config = config_response.json()
    assert "mcpServers" in config
    assert isinstance(config["mcpServers"], dict)


def test_assign_tool_to_agent(api_test_client):
    """Test assigning a tool to an agent."""
    client, _ = api_test_client

    # Erstelle einen Agenten
    agent_data = {
        "name": "Tool Assignment Test Agent",
        "provider": "anthropic",
        "model_name": "claude-3-sonnet-20240229",
    }

    agent_response = client.post("/agents", json=agent_data)
    assert agent_response.status_code == 201
    agent_id = agent_response.json()["id"]

    # Erstelle einen Server mit eindeutigem Namen
    server_name = generate_unique_name("ToolAssignmentServer")
    server_data = {"name": server_name, "transport": "stdio", "enabled": True}

    server_response = client.post("/mcp/servers", json=server_data)
    assert server_response.status_code == 201
    server_id = server_response.json()["id"]

    # Erstelle ein Tool
    tool_data = {
        "name": "Tool for Assignment Test",
        "enabled": True,
        "server_id": server_id,
    }

    tool_response = client.post("/mcp/tools", json=tool_data)
    assert tool_response.status_code == 201
    tool_id = tool_response.json()["id"]

    # Weise das Tool dem Agenten zu
    assignment_data = {"enabled": True}

    assignment_response = client.post(
        f"/mcp/tools/agent/{agent_id}/tool/{tool_id}", json=assignment_data
    )
    assert assignment_response.status_code == 204

    # Prüfe, ob das Tool in der Liste der Tools des Agenten ist
    agent_tools_response = client.get(f"/agents/{agent_id}/tools")
    assert agent_tools_response.status_code == 200

    agent_tools = agent_tools_response.json()
    assert isinstance(agent_tools, list)
    assert len(agent_tools) >= 1

    # Finde das zugewiesene Tool in der Liste
    found = False
    for tool in agent_tools:
        if tool["id"] == tool_id:
            found = True
            assert tool["enabled"] == assignment_data["enabled"]
            break

    assert (
        found
    ), "Das zugewiesene Tool wurde nicht in der Liste der Tools des Agenten gefunden"

    # Entferne das Tool vom Agenten
    remove_response = client.delete(f"/mcp/tools/agent/{agent_id}/tool/{tool_id}")
    assert remove_response.status_code == 204

    # Prüfe, ob das Tool nicht mehr in der Liste der Tools des Agenten ist
    agent_tools_response_after = client.get(f"/agents/{agent_id}/tools")
    assert agent_tools_response_after.status_code == 200

    agent_tools_after = agent_tools_response_after.json()
    not_found = True
    for tool in agent_tools_after:
        if tool["id"] == tool_id:
            not_found = False
            break

    assert not_found, "Das Tool wurde nicht erfolgreich vom Agenten entfernt"


def test_server_tool_lifecycle_with_live_server(live_api_server):
    """Test complete server and tool lifecycle with a live server."""
    with httpx.Client(base_url=live_api_server, timeout=10.0) as client:
        # 1. Erstelle einen Server
        server_name = generate_unique_name("LiveServer")
        server_data = {
            "name": server_name,
            "transport": "websocket",
            "enabled": True,
            "url": "ws://localhost:9000/ws",
            "headers": {"X-API-KEY": "test-key"},
        }

        server_response = client.post("/mcp/servers", json=server_data)
        assert server_response.status_code == 201
        server = server_response.json()
        server_id = server["id"]

        # 2. Erstelle ein Tool für den Server
        tool_name = generate_unique_name("LiveTool")
        tool_data = {
            "name": tool_name,
            "enabled": True,
            "server_id": server_id,
            "description": "A test tool for live server testing",
            "parameters": {
                "type": "object",
                "properties": {
                    "input": {"type": "string", "description": "Tool input"}
                },
            },
        }

        tool_response = client.post("/mcp/tools", json=tool_data)
        assert tool_response.status_code == 201
        tool = tool_response.json()
        tool_id = tool["id"]

        # 3. Erstelle einen Agenten
        agent_data = {
            "name": generate_unique_name("LiveAgent"),
            "provider": "openai",
            "model_name": "gpt-4",
        }

        agent_response = client.post("/agents", json=agent_data)
        assert agent_response.status_code == 201
        agent = agent_response.json()
        agent_id = agent["id"]

        # 4. Weise das Tool dem Agenten zu
        assignment_data = {"enabled": True}
        assignment_response = client.post(
            f"/mcp/tools/agent/{agent_id}/tool/{tool_id}", json=assignment_data
        )
        assert assignment_response.status_code == 204

        # 5. Hole die Tool-Liste des Agenten
        agent_tools_response = client.get(f"/agents/{agent_id}/tools")
        assert agent_tools_response.status_code == 200
        tools = agent_tools_response.json()
        assert any(t["id"] == tool_id for t in tools)

        # 6. Hole die MCP-Konfiguration
        config_response = client.get("/mcp/config")
        assert config_response.status_code == 200
        config = config_response.json()
        assert "mcpServers" in config
        assert server_name in config["mcpServers"]

        # 7. Lösche die Ressourcen
        # Entferne Tool-Zuweisung
        remove_tool_response = client.delete(
            f"/mcp/tools/agent/{agent_id}/tool/{tool_id}"
        )
        assert remove_tool_response.status_code == 204

        # Lösche Tool
        delete_tool_response = client.delete(f"/mcp/tools/{tool_id}")
        assert delete_tool_response.status_code == 204

        # Lösche Server
        delete_server_response = client.delete(f"/mcp/servers/{server_id}")
        assert delete_server_response.status_code == 204

        # Lösche Agent
        delete_agent_response = client.delete(f"/agents/{agent_id}")
        assert delete_agent_response.status_code == 204
