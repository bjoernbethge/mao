"""
Tests für die Teams API Endpoints.
"""

import uuid


def test_create_team(api_test_client):
    """Test creating a team."""
    client, _ = api_test_client
    team_data = {"name": "Test Team", "description": "A test team"}

    response = client.post("/teams/", json=team_data)
    assert response.status_code == 201

    team = response.json()
    assert team["name"] == team_data["name"]
    assert team["description"] == team_data["description"]
    assert "id" in team
    assert "created_at" in team


def test_create_team_with_supervisor(api_test_client):
    """Test creating a team with a supervisor."""
    client, _ = api_test_client

    # Zuerst einen Agenten erstellen, der als Supervisor dienen wird
    agent_data = {
        "name": "Supervisor Agent",
        "provider": "openai",
        "model_name": "gpt-4",
        "system_prompt": "You are a supervisor agent",
    }
    agent_response = client.post("/agents/", json=agent_data)
    assert agent_response.status_code == 201
    agent_id = agent_response.json()["id"]

    # Supervisor erstellen
    supervisor_data = {
        "agent_id": agent_id,
        "system_prompt": "Supervisor prompt",
        "strategy": "team_manager",
        "add_handoff_back_messages": True,
        "parallel_tool_calls": True,
    }
    supervisor_response = client.post("/teams/supervisors", json=supervisor_data)
    assert supervisor_response.status_code == 201
    supervisor_id = supervisor_response.json()["id"]

    # Team mit Supervisor erstellen
    team_data = {
        "name": "Team with Supervisor",
        "description": "For supervisor test",
        "supervisor_id": supervisor_id,
    }
    team_response = client.post("/teams/", json=team_data)
    assert team_response.status_code == 201
    team_id = team_response.json()["id"]

    # Überprüfen, dass der Supervisor korrekt zugewiesen wurde
    get_response = client.get(f"/teams/{team_id}")
    assert get_response.status_code == 200
    team = get_response.json()
    assert team["supervisor_id"] == supervisor_id


def test_team_with_members(api_test_client):
    """Test creating a team with members."""
    client, _ = api_test_client

    # Create a team
    team_data = {"name": "Team with Members", "description": "Team with members test"}
    team_response = client.post("/teams/", json=team_data)
    assert team_response.status_code == 201
    team_id = team_response.json()["id"]

    # Create agents
    agent1_data = {
        "name": "Team Member 1",
        "description": "First team member",
        "provider": "openai",
        "model_name": "gpt-3.5-turbo",
    }

    agent2_data = {
        "name": "Team Member 2",
        "description": "Second team member",
        "provider": "openai",
        "model_name": "gpt-3.5-turbo",
    }

    agent1_response = client.post("/agents/", json=agent1_data)
    agent2_response = client.post("/agents/", json=agent2_data)

    agent1_id = agent1_response.json()["id"]
    agent2_id = agent2_response.json()["id"]

    # Add agents to team
    member1_data = {
        "agent_id": agent1_id,
        "team_id": team_id,
        "role": "assistant",
    }

    member2_data = {
        "agent_id": agent2_id,
        "team_id": team_id,
        "role": "researcher",
    }

    member1_response = client.post(f"/teams/{team_id}/members", json=member1_data)
    member2_response = client.post(f"/teams/{team_id}/members", json=member2_data)

    assert member1_response.status_code == 201
    assert member2_response.status_code == 201

    # Get team members
    members_response = client.get(f"/teams/{team_id}/members")
    assert members_response.status_code == 200
    members = members_response.json()

    # Check team has correct members
    assert len(members) == 2
    member_agent_ids = [m["agent_id"] for m in members]
    assert agent1_id in member_agent_ids
    assert agent2_id in member_agent_ids


def test_team_with_supervisor_and_members(api_test_client):
    """Test creating a team with a supervisor and members."""
    client, _ = api_test_client

    # Zuerst einen Agenten erstellen, der als Supervisor dienen wird
    supervisor_agent_data = {
        "name": "Team Supervisor Agent",
        "provider": "openai",
        "model_name": "gpt-4",
        "system_prompt": "You are a supervisor agent",
    }
    supervisor_agent_response = client.post("/agents/", json=supervisor_agent_data)
    assert supervisor_agent_response.status_code == 201
    supervisor_agent_id = supervisor_agent_response.json()["id"]

    # Supervisor erstellen
    supervisor_data = {
        "agent_id": supervisor_agent_id,
        "system_prompt": "Supervisor prompt",
        "strategy": "team_manager",
        "add_handoff_back_messages": True,
        "parallel_tool_calls": True,
    }
    supervisor_response = client.post("/teams/supervisors", json=supervisor_data)
    assert supervisor_response.status_code == 201
    supervisor_id = supervisor_response.json()["id"]

    # Create a team with supervisor directly assigned
    team_data = {
        "name": "Full Team",
        "description": "Team with supervisor and members",
        "supervisor_id": supervisor_id,
    }
    team_response = client.post("/teams/", json=team_data)
    assert team_response.status_code == 201
    team_id = team_response.json()["id"]

    # Create agents
    agent1_data = {
        "name": "Team Member 1",
        "description": "First team member",
        "provider": "openai",
        "model_name": "gpt-3.5-turbo",
    }

    agent2_data = {
        "name": "Team Member 2",
        "description": "Second team member",
        "provider": "openai",
        "model_name": "gpt-3.5-turbo",
    }

    agent1_response = client.post("/agents/", json=agent1_data)
    agent2_response = client.post("/agents/", json=agent2_data)

    agent1_id = agent1_response.json()["id"]
    agent2_id = agent2_response.json()["id"]

    # Add agents to team
    member1_data = {
        "agent_id": agent1_id,
        "team_id": team_id,
        "role": "assistant",
    }

    member2_data = {
        "agent_id": agent2_id,
        "team_id": team_id,
        "role": "researcher",
    }

    member1_response = client.post(f"/teams/{team_id}/members", json=member1_data)
    member2_response = client.post(f"/teams/{team_id}/members", json=member2_data)

    assert member1_response.status_code == 201
    assert member2_response.status_code == 201

    # Get team details
    team_response = client.get(f"/teams/{team_id}")
    team = team_response.json()

    # Check team has correct supervisor
    assert team["supervisor_id"] == supervisor_id

    # Get team members
    members_response = client.get(f"/teams/{team_id}/members")
    assert members_response.status_code == 200
    members = members_response.json()

    # Check team has correct members
    assert len(members) == 2
    member_agent_ids = [m["agent_id"] for m in members]
    assert agent1_id in member_agent_ids
    assert agent2_id in member_agent_ids


def test_team_update_and_delete(api_test_client):
    """Test updating and deleting a team."""
    client, _ = api_test_client

    # Create a team
    team_data = {"name": "Original Team", "description": "Original description"}
    team_response = client.post("/teams/", json=team_data)
    team_id = team_response.json()["id"]

    # Update the team
    update_data = {"name": "Updated Team", "description": "Updated description"}
    update_response = client.put(f"/teams/{team_id}", json=update_data)

    assert update_response.status_code == 200
    updated_team = update_response.json()
    assert updated_team["name"] == update_data["name"]
    assert updated_team["description"] == update_data["description"]

    # Delete the team
    delete_response = client.delete(f"/teams/{team_id}")
    assert delete_response.status_code == 204

    # Verify team is deleted
    get_response = client.get(f"/teams/{team_id}")
    assert get_response.status_code == 404


def test_team_member_update_and_remove(api_test_client):
    """Test updating and removing a team member."""
    client, _ = api_test_client

    # Create a team
    team_data = {"name": "Member Test Team", "description": "For member tests"}
    team_response = client.post("/teams/", json=team_data)
    team_id = team_response.json()["id"]

    # Create an agent
    agent_data = {
        "name": "Test Agent",
        "description": "For member test",
        "provider": "openai",
        "model_name": "gpt-3.5-turbo",
    }
    agent_response = client.post("/agents/", json=agent_data)
    agent_id = agent_response.json()["id"]

    # Add agent to team
    member_data = {
        "agent_id": agent_id,
        "team_id": team_id,
        "role": "assistant",
    }
    member_response = client.post(f"/teams/{team_id}/members", json=member_data)
    assert member_response.status_code == 201

    # Überprüfen, dass der Member hinzugefügt wurde
    members_response = client.get(f"/teams/{team_id}/members")
    assert members_response.status_code == 200
    members = members_response.json()
    assert len(members) == 1
    assert members[0]["agent_id"] == agent_id
    assert members[0]["role"] == "assistant"

    # Update member role
    update_data = {"role": "researcher"}
    update_response = client.put(
        f"/teams/{team_id}/members/{agent_id}", json=update_data
    )

    assert update_response.status_code == 200
    updated_member = update_response.json()
    assert updated_member["role"] == update_data["role"]

    # Remove member from team
    delete_response = client.delete(f"/teams/{team_id}/members/{agent_id}")
    assert delete_response.status_code == 204

    # Verify member is removed
    members_response = client.get(f"/teams/{team_id}/members")
    assert members_response.status_code == 200
    members = members_response.json()
    assert len(members) == 0


def test_get_team_not_found(api_test_client):
    """Test getting a non-existent team."""
    client, _ = api_test_client

    random_id = str(uuid.uuid4())
    response = client.get(f"/teams/{random_id}")
    assert response.status_code == 404


def test_update_team_not_found(api_test_client):
    """Test updating a non-existent team."""
    client, _ = api_test_client

    random_id = str(uuid.uuid4())
    update_data = {"name": "Updated Team", "description": "Updated description"}
    response = client.put(f"/teams/{random_id}", json=update_data)
    assert response.status_code == 404


def test_delete_team_not_found(api_test_client):
    """Test deleting a non-existent team."""
    client, _ = api_test_client

    random_id = str(uuid.uuid4())
    response = client.delete(f"/teams/{random_id}")
    assert response.status_code == 404


def test_get_supervisor_not_found(api_test_client):
    """Test getting a non-existent supervisor."""
    client, _ = api_test_client

    random_id = str(uuid.uuid4())
    response = client.get(f"/supervisors/{random_id}")
    assert response.status_code == 404


def test_update_supervisor_not_found(api_test_client):
    """Test updating a non-existent supervisor."""
    client, _ = api_test_client

    random_id = str(uuid.uuid4())
    update_data = {"name": "Updated Supervisor", "prompt": "Updated prompt"}
    response = client.put(f"/supervisors/{random_id}", json=update_data)
    assert response.status_code == 404


def test_delete_supervisor_not_found(api_test_client):
    """Test deleting a non-existent supervisor."""
    client, _ = api_test_client

    random_id = str(uuid.uuid4())
    response = client.delete(f"/supervisors/{random_id}")
    assert response.status_code == 404


def test_list_teams(api_test_client):
    """Test listing all teams."""
    client, _ = api_test_client

    # Create some teams
    team1_data = {"name": "Team 1", "description": "First team"}
    team2_data = {"name": "Team 2", "description": "Second team"}

    client.post("/teams/", json=team1_data)
    client.post("/teams/", json=team2_data)

    # List all teams
    response = client.get("/teams/")
    assert response.status_code == 200

    teams = response.json()
    assert isinstance(teams, list)
    assert len(teams) >= 2
    team_names = [t["name"] for t in teams]
    assert "Team 1" in team_names
    assert "Team 2" in team_names


def test_get_team_members(api_test_client):
    """Test getting members of a specific team."""
    client, _ = api_test_client

    # Create a team
    team_data = {"name": "Member List Team", "description": "For listing members"}
    team_response = client.post("/teams/", json=team_data)
    team_id = team_response.json()["id"]

    # Create agents
    agent1_data = {
        "name": "Member Agent 1",
        "description": "First member agent",
        "provider": "openai",
        "model_name": "gpt-3.5-turbo",
    }
    agent2_data = {
        "name": "Member Agent 2",
        "description": "Second member agent",
        "provider": "openai",
        "model_name": "gpt-3.5-turbo",
    }

    agent1_response = client.post("/agents/", json=agent1_data)
    agent2_response = client.post("/agents/", json=agent2_data)

    agent1_id = agent1_response.json()["id"]
    agent2_id = agent2_response.json()["id"]

    # Add agents to team
    member1_data = {
        "agent_id": agent1_id,
        "team_id": team_id,
        "role": "assistant",
    }

    member2_data = {
        "agent_id": agent2_id,
        "team_id": team_id,
        "role": "researcher",
    }

    client.post(f"/teams/{team_id}/members", json=member1_data)
    client.post(f"/teams/{team_id}/members", json=member2_data)

    # Get team members
    response = client.get(f"/teams/{team_id}/members")
    assert response.status_code == 200

    members = response.json()
    assert isinstance(members, list)
    assert len(members) == 2
    member_agent_ids = [m["agent_id"] for m in members]
    assert agent1_id in member_agent_ids
    assert agent2_id in member_agent_ids
