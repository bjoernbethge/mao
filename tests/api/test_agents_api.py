"""
Tests for the Agents API endpoints.
Test agent creation, retrieval, update, and deletion.
"""

import pytest
import json
import uuid
import httpx


def test_create_agent(api_test_client):
    """Test creating a new agent."""
    client, _ = api_test_client
    
    # Testdaten für die Agent-Erstellung
    agent_data = {
        "name": "Test Agent",
        "provider": "anthropic",
        "model_name": "claude-3-sonnet-20240229",
        "system_prompt": "You are a helpful AI assistant.",
        "use_react_agent": True,
        "max_tokens_trimmed": 4000,
        "llm_specific_kwargs": {"temperature": 0.5}
    }
    
    response = client.post("/agents", json=agent_data)
    
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == agent_data["name"]
    assert data["provider"] == agent_data["provider"]
    assert data["model_name"] == agent_data["model_name"]
    assert data["system_prompt"] == agent_data["system_prompt"]
    assert data["use_react_agent"] == agent_data["use_react_agent"]
    assert data["max_tokens_trimmed"] == agent_data["max_tokens_trimmed"]
    assert data["llm_specific_kwargs"] == agent_data["llm_specific_kwargs"]
    assert "id" in data
    assert data["id"].startswith("agent_")


def test_list_agents(api_test_client):
    """Test listing all agents."""
    client, _ = api_test_client
    
    # Erstelle erst einen Agent
    agent_data = {
        "name": "Test Agent for List",
        "provider": "anthropic",
        "model_name": "claude-3-sonnet-20240229"
    }
    
    create_response = client.post("/agents", json=agent_data)
    assert create_response.status_code == 201
    
    # Hole die Liste aller Agenten
    list_response = client.get("/agents")
    assert list_response.status_code == 200
    
    agents = list_response.json()
    assert isinstance(agents, list)
    assert len(agents) >= 1
    
    # Finde den erstellten Agenten in der Liste
    found = False
    for agent in agents:
        if agent["name"] == agent_data["name"]:
            found = True
            break
    
    assert found, "Der erstellte Agent wurde nicht in der Liste gefunden"


def test_get_agent(api_test_client):
    """Test getting an agent by ID."""
    client, _ = api_test_client
    
    # Erstelle erst einen Agent
    agent_data = {
        "name": "Test Agent for Get",
        "provider": "openai",
        "model_name": "gpt-4"
    }
    
    create_response = client.post("/agents", json=agent_data)
    assert create_response.status_code == 201
    created_agent = create_response.json()
    agent_id = created_agent["id"]
    
    # Hole den Agenten mit seiner ID
    get_response = client.get(f"/agents/{agent_id}")
    assert get_response.status_code == 200
    
    agent = get_response.json()
    assert agent["id"] == agent_id
    assert agent["name"] == agent_data["name"]
    assert agent["provider"] == agent_data["provider"]
    assert agent["model_name"] == agent_data["model_name"]


def test_update_agent(api_test_client):
    """Test updating an agent."""
    client, _ = api_test_client
    
    # Erstelle erst einen Agent
    agent_data = {
        "name": "Agent before update",
        "provider": "anthropic",
        "model_name": "claude-3-haiku-20240307"
    }
    
    create_response = client.post("/agents", json=agent_data)
    assert create_response.status_code == 201
    created_agent = create_response.json()
    agent_id = created_agent["id"]
    
    # Aktualisiere den Agenten
    update_data = {
        "name": "Agent after update",
        "system_prompt": "This is an updated system prompt."
    }
    
    update_response = client.put(f"/agents/{agent_id}", json=update_data)
    assert update_response.status_code == 200
    
    updated_agent = update_response.json()
    assert updated_agent["id"] == agent_id
    assert updated_agent["name"] == update_data["name"]
    assert updated_agent["system_prompt"] == update_data["system_prompt"]
    # Die nicht aktualisierten Felder sollten unverändert bleiben
    assert updated_agent["provider"] == agent_data["provider"]
    assert updated_agent["model_name"] == agent_data["model_name"]


def test_delete_agent(api_test_client):
    """Test deleting an agent."""
    client, _ = api_test_client
    
    # Erstelle erst einen Agent
    agent_data = {
        "name": "Agent to delete",
        "provider": "openai",
        "model_name": "gpt-3.5-turbo"
    }
    
    create_response = client.post("/agents", json=agent_data)
    assert create_response.status_code == 201
    created_agent = create_response.json()
    agent_id = created_agent["id"]
    
    # Lösche den Agenten
    delete_response = client.delete(f"/agents/{agent_id}")
    assert delete_response.status_code == 204
    
    # Versuche, den gelöschten Agenten abzurufen
    get_response = client.get(f"/agents/{agent_id}")
    assert get_response.status_code == 404


def test_agent_not_found(api_test_client):
    """Test responses for non-existing agent ID."""
    client, _ = api_test_client
    
    non_existent_id = f"agent_{uuid.uuid4().hex[:8]}"
    
    # GET
    get_response = client.get(f"/agents/{non_existent_id}")
    assert get_response.status_code == 404
    
    # PUT
    update_data = {"name": "Updated Name"}
    put_response = client.put(f"/agents/{non_existent_id}", json=update_data)
    assert put_response.status_code == 404
    
    # DELETE
    delete_response = client.delete(f"/agents/{non_existent_id}")
    assert delete_response.status_code == 404


def test_list_running_agents(api_test_client):
    """Test listing running agents."""
    client, _ = api_test_client
    
    response = client.get("/agents/running")
    assert response.status_code == 200
    
    data = response.json()
    assert "count" in data
    assert "agents" in data
    assert isinstance(data["agents"], list)


def test_delete_agent_with_dependencies(api_test_client):
    """Tests that deleting an agent cleans up all dependent relationships"""
    client, _ = api_test_client
    
    # 1. Create an agent
    agent_data = {
        "name": "Agent to delete",
        "provider": "anthropic",
        "model_name": "claude-3-haiku-20240307"
    }
    agent_response = client.post("/agents", json=agent_data)
    assert agent_response.status_code == 201
    agent_id = agent_response.json()["id"]
    
    # 2. Create a supervisor using the agent
    supervisor_data = {
        "agent_id": agent_id,
        "strategy": "team_manager"
    }
    supervisor_response = client.post("/teams/supervisors", json=supervisor_data)
    assert supervisor_response.status_code == 201
    supervisor_id = supervisor_response.json()["id"]
    
    # 3. Create a team with the supervisor
    team_data = {
        "name": "Test Team",
        "workflow_type": "sequential",
        "supervisor_id": supervisor_id
    }
    team_response = client.post("/teams", json=team_data)
    assert team_response.status_code == 201
    team_id = team_response.json()["id"]
    assert team_response.json()["supervisor_id"] == supervisor_id
    
    # 4. Add agent as team member
    member_data = {
        "agent_id": agent_id,
        "role": "assistant",
        "order_index": 1
    }
    member_response = client.post(f"/teams/{team_id}/members", json=member_data)
    assert member_response.status_code == 201
    
    # 5. Delete the agent
    delete_response = client.delete(f"/agents/{agent_id}")
    assert delete_response.status_code == 204
    
    # 6. Verify the agent is deleted
    get_agent_response = client.get(f"/agents/{agent_id}")
    assert get_agent_response.status_code == 404
    
    # 7. Verify the supervisor is deleted
    get_supervisor_response = client.get(f"/teams/supervisors/{supervisor_id}")
    assert get_supervisor_response.status_code == 404
    
    # 8. Verify the team no longer has a supervisor
    get_team_response = client.get(f"/teams/{team_id}")
    assert get_team_response.status_code == 200
    assert get_team_response.json()["supervisor_id"] is None
    
    # 9. Verify the agent is no longer a team member
    get_members_response = client.get(f"/teams/{team_id}/members")
    assert get_members_response.status_code == 200
    members = get_members_response.json()
    assert len(members) == 0 