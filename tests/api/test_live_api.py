"""
Tests für die MCP Agents API mit einem echten laufenden Server.
Diese Tests verwenden einen realen HTTP-Server für eine realistischere Testumgebung.
"""

import pytest
import uuid
import httpx
import random
import string


def generate_unique_name(prefix="LiveTest"):
    """Generate a unique name with a random suffix."""
    random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    return f"{prefix}_{random_suffix}"


def test_health_endpoint_live(live_api_server):
    """Test health endpoint with a live server."""
    with httpx.Client(base_url=live_api_server, timeout=10.0) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data


def test_create_and_get_agent_live(live_api_server):
    """Test creating and retrieving an agent with a live server."""
    with httpx.Client(base_url=live_api_server, timeout=10.0) as client:
        # Erstelle einen Agenten
        agent_name = generate_unique_name("LiveAgent")
        agent_data = {
            "name": agent_name,
            "provider": "anthropic",
            "model_name": "claude-3-sonnet-20240229",
            "system_prompt": "You are a helpful testing assistant.",
        }
        
        create_response = client.post("/agents", json=agent_data)
        assert create_response.status_code == 201
        
        created_agent = create_response.json()
        assert created_agent["name"] == agent_data["name"]
        assert created_agent["provider"] == agent_data["provider"]
        assert "id" in created_agent
        
        agent_id = created_agent["id"]
        
        # Hole den erstellten Agenten
        get_response = client.get(f"/agents/{agent_id}")
        assert get_response.status_code == 200
        
        retrieved_agent = get_response.json()
        assert retrieved_agent["id"] == agent_id
        assert retrieved_agent["name"] == agent_data["name"]


def test_server_crud_operations_live(live_api_server):
    """Test CRUD operations for servers with a live server."""
    with httpx.Client(base_url=live_api_server, timeout=10.0) as client:
        # Erstelle einen Server
        server_name = generate_unique_name("LiveServer")
        server_data = {
            "name": server_name,
            "transport": "stdio",
            "enabled": True,
            "command": "python",
            "args": ["-m", "mcp_agents.server"]
        }
        
        create_response = client.post("/mcp/servers", json=server_data)
        assert create_response.status_code == 201
        
        created_server = create_response.json()
        server_id = created_server["id"]
        
        # Update den Server
        update_data = {
            "name": f"{server_name}_updated",
            "enabled": False
        }
        
        update_response = client.put(f"/mcp/servers/{server_id}", json=update_data)
        assert update_response.status_code == 200
        
        updated_server = update_response.json()
        assert updated_server["name"] == update_data["name"]
        assert updated_server["enabled"] == update_data["enabled"]
        
        # Liste alle Server auf
        list_response = client.get("/mcp/servers")
        assert list_response.status_code == 200
        
        servers = list_response.json()
        found = any(s["id"] == server_id for s in servers)
        assert found, "Der Server wurde nicht in der Liste gefunden"
        
        # Lösche den Server
        delete_response = client.delete(f"/mcp/servers/{server_id}")
        assert delete_response.status_code == 204
        
        # Versuche, den gelöschten Server abzurufen
        get_response = client.get(f"/mcp/servers/{server_id}")
        assert get_response.status_code == 404


def test_mcp_config_endpoint_live(live_api_server):
    """Test the MCP configuration endpoint with a live server."""
    with httpx.Client(base_url=live_api_server, timeout=10.0) as client:
        # Erstelle einen Server für die Konfiguration
        server_name = generate_unique_name("ConfigServer")
        server_data = {
            "name": server_name,
            "transport": "sse",
            "enabled": True,
            "url": "http://example.com/events"
        }
        
        client.post("/mcp/servers", json=server_data)
        
        # Hole die MCP-Konfiguration
        config_response = client.get("/mcp/config")
        assert config_response.status_code == 200
        
        config = config_response.json()
        assert "mcpServers" in config
        assert server_name in config["mcpServers"]
        assert config["mcpServers"][server_name]["transport"] == "sse" 