"""
Tests for the Teams API endpoints.
Test team operations, supervisor management, and team member functionality.
"""

import pytest
import json
import uuid
import httpx


def test_create_team(api_test_client):
    """Test creating a new team."""
    client, _ = api_test_client
    
    # Testdaten für die Team-Erstellung
    team_data = {
        "name": "Test Research Team",
        "description": "A team for testing team functionality",
        "workflow_type": "sequential"
    }
    
    response = client.post("/teams", json=team_data)
    
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == team_data["name"]
    assert data["description"] == team_data["description"]
    assert data["workflow_type"] == team_data["workflow_type"]
    assert "id" in data
    assert data["id"].startswith("team_")


def test_create_and_get_supervisor(api_test_client):
    """Test creating and retrieving a supervisor."""
    client, _ = api_test_client
    
    # First create an agent to be the supervisor
    agent_data = {
        "name": "Supervisor Agent",
        "provider": "anthropic",
        "model_name": "claude-3-opus-20240229"
    }
    
    agent_response = client.post("/agents", json=agent_data)
    assert agent_response.status_code == 201
    agent_id = agent_response.json()["id"]
    
    # Now create a supervisor
    supervisor_data = {
        "agent_id": agent_id,
        "strategy": "team_manager",
        "system_prompt": "You are a supervisor managing a team of specialized agents."
    }
    
    supervisor_response = client.post("/teams/supervisors", json=supervisor_data)
    assert supervisor_response.status_code == 201
    
    supervisor = supervisor_response.json()
    supervisor_id = supervisor["id"]
    
    # Get the supervisor by ID
    get_response = client.get(f"/teams/supervisors/{supervisor_id}")
    assert get_response.status_code == 200
    
    retrieved_supervisor = get_response.json()
    assert retrieved_supervisor["id"] == supervisor_id
    assert retrieved_supervisor["agent_id"] == agent_id
    assert retrieved_supervisor["system_prompt"] == supervisor_data["system_prompt"]


def test_team_with_supervisor_and_members(api_test_client):
    """Test creating a team with supervisor and members."""
    client, _ = api_test_client
    
    # Create a supervisor agent
    supervisor_agent_data = {
        "name": "Team Supervisor",
        "provider": "anthropic",
        "model_name": "claude-3-opus-20240229"
    }
    
    supervisor_agent_response = client.post("/agents", json=supervisor_agent_data)
    assert supervisor_agent_response.status_code == 201
    supervisor_agent_id = supervisor_agent_response.json()["id"]
    
    # Create a supervisor
    supervisor_data = {
        "agent_id": supervisor_agent_id,
        "strategy": "team_manager"
    }
    
    supervisor_response = client.post("/teams/supervisors", json=supervisor_data)
    assert supervisor_response.status_code == 201
    supervisor_id = supervisor_response.json()["id"]
    
    # Create team with the supervisor
    team_data = {
        "name": "Full Test Team",
        "description": "A team with supervisor and members",
        "workflow_type": "sequential",
        "supervisor_id": supervisor_id
    }
    
    team_response = client.post("/teams", json=team_data)
    assert team_response.status_code == 201
    team_id = team_response.json()["id"]
    
    # Create member agents
    member_agents = []
    for i, role in enumerate([
        {"name": "Researcher Agent", "role": "researcher"},
        {"name": "Writer Agent", "role": "writer"},
        {"name": "Reviewer Agent", "role": "reviewer"}
    ]):
        agent_data = {
            "name": role["name"],
            "provider": "anthropic",
            "model_name": "claude-3-haiku-20240307"
        }
        
        agent_response = client.post("/agents", json=agent_data)
        assert agent_response.status_code == 201
        agent_id = agent_response.json()["id"]
        
        # Add agent to team
        member_data = {
            "agent_id": agent_id,
            "role": role["role"],
            "order_index": i + 1,
            "is_active": True
        }
        
        member_response = client.post(f"/teams/{team_id}/members", json=member_data)
        assert member_response.status_code == 201
        
        member_agents.append({
            "id": agent_id,
            "role": role["role"],
            "order_index": i + 1
        })
    
    # Get team members
    members_response = client.get(f"/teams/{team_id}/members")
    assert members_response.status_code == 200
    
    members = members_response.json()
    assert len(members) == 3
    
    # Verify member roles and order
    found_roles = {m["role"]: m["order_index"] for m in members}
    assert "researcher" in found_roles
    assert "writer" in found_roles
    assert "reviewer" in found_roles
    
    # Verify the ordering
    assert found_roles["researcher"] < found_roles["writer"] < found_roles["reviewer"]


def test_team_update_and_delete(api_test_client):
    """Test updating and deleting a team."""
    client, _ = api_test_client
    
    # Create a team
    team_data = {
        "name": "Team to Update",
        "workflow_type": "parallel"
    }
    
    create_response = client.post("/teams", json=team_data)
    assert create_response.status_code == 201
    team_id = create_response.json()["id"]
    
    # Update the team
    update_data = {
        "name": "Updated Team Name",
        "description": "This team has been updated",
        "workflow_type": "custom",
        "is_active": False
    }
    
    update_response = client.put(f"/teams/{team_id}", json=update_data)
    assert update_response.status_code == 200
    
    updated_team = update_response.json()
    assert updated_team["name"] == update_data["name"]
    assert updated_team["description"] == update_data["description"]
    assert updated_team["workflow_type"] == update_data["workflow_type"]
    assert updated_team["is_active"] == update_data["is_active"]
    
    # Delete the team
    delete_response = client.delete(f"/teams/{team_id}")
    assert delete_response.status_code == 204
    
    # Verify the team is deleted
    get_response = client.get(f"/teams/{team_id}")
    assert get_response.status_code == 404


def test_team_member_update_and_remove(api_test_client):
    """Test updating and removing team members."""
    client, _ = api_test_client
    
    # Create a team
    team_response = client.post("/teams", json={"name": "Member Test Team", "workflow_type": "sequential"})
    assert team_response.status_code == 201
    team_id = team_response.json()["id"]
    
    # Create an agent
    agent_response = client.post("/agents", json={
        "name": "Team Member", 
        "provider": "anthropic",
        "model_name": "claude-3-haiku-20240307"
    })
    assert agent_response.status_code == 201
    agent_id = agent_response.json()["id"]
    
    # Add agent to team
    member_data = {
        "agent_id": agent_id,
        "role": "assistant",
        "order_index": 1
    }
    
    add_response = client.post(f"/teams/{team_id}/members", json=member_data)
    assert add_response.status_code == 201
    
    # Update the member
    update_data = {
        "role": "senior_assistant",
        "order_index": 2,
        "params": {"priority": "high"}
    }
    
    update_response = client.put(f"/teams/{team_id}/members/{agent_id}", json=update_data)
    assert update_response.status_code == 200
    
    updated_member = update_response.json()
    assert updated_member["role"] == update_data["role"]
    assert updated_member["order_index"] == update_data["order_index"]
    assert updated_member["params"] == update_data["params"]
    
    # Remove the member
    remove_response = client.delete(f"/teams/{team_id}/members/{agent_id}")
    assert remove_response.status_code == 204
    
    # Verify the member is removed
    members_response = client.get(f"/teams/{team_id}/members")
    assert members_response.status_code == 200
    members = members_response.json()
    assert len(members) == 0


def test_supervisor_lifecycle(api_test_client):
    """Test the full lifecycle of a supervisor."""
    client, _ = api_test_client
    
    # Create an agent
    agent_response = client.post("/agents", json={
        "name": "Lifecycle Supervisor", 
        "provider": "anthropic",
        "model_name": "claude-3-opus-20240229"
    })
    assert agent_response.status_code == 201
    agent_id = agent_response.json()["id"]
    
    # Create supervisor
    supervisor_data = {
        "agent_id": agent_id,
        "strategy": "orchestrator",
        "system_prompt": "You orchestrate a team of agents.",
        "add_handoff_back_messages": True,
        "parallel_tool_calls": False,
        "config": {"max_iterations": 10}
    }
    
    create_response = client.post("/teams/supervisors", json=supervisor_data)
    assert create_response.status_code == 201
    supervisor = create_response.json()
    supervisor_id = supervisor["id"]
    
    # Update supervisor
    update_data = {
        "strategy": "team_manager",
        "parallel_tool_calls": True,
        "config": {"max_iterations": 5, "timeout": 30}
    }
    
    update_response = client.put(f"/teams/supervisors/{supervisor_id}", json=update_data)
    assert update_response.status_code == 200
    
    updated_supervisor = update_response.json()
    assert updated_supervisor["strategy"] == update_data["strategy"]
    assert updated_supervisor["parallel_tool_calls"] == update_data["parallel_tool_calls"]
    assert updated_supervisor["config"] == update_data["config"]
    
    # Add to a team
    team_response = client.post("/teams", json={
        "name": "Team With Supervisor",
        "workflow_type": "sequential",
        "supervisor_id": supervisor_id
    })
    assert team_response.status_code == 201
    team_id = team_response.json()["id"]
    
    # Verify team has the supervisor
    team_get_response = client.get(f"/teams/{team_id}")
    assert team_get_response.status_code == 200
    team = team_get_response.json()
    assert team["supervisor_id"] == supervisor_id
    
    # Delete supervisor and verify team supervisor is set to null
    delete_response = client.delete(f"/teams/supervisors/{supervisor_id}")
    assert delete_response.status_code == 204
    
    # Verify team supervisor is null
    team_get_response = client.get(f"/teams/{team_id}")
    assert team_get_response.status_code == 200
    team = team_get_response.json()
    assert team["supervisor_id"] is None 