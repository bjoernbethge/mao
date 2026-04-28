"""
Tests fuer die Teams API Endpoints.
"""

import os
import uuid

from mao.api.agents import active_agents
from mao.api.teams import _peer_message_allowed, active_teams

TEST_LLM_PROVIDER = os.environ.get("TEST_LLM_PROVIDER", "ollama")
TEST_LLM_MODEL = os.environ.get("TEST_LLM_MODEL", "gemma3:4b-cloud")


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


def test_start_team_runtime(api_test_client):
    """Test starting a team with a real runtime path."""
    client, _ = api_test_client
    active_agents.clear()
    active_teams.clear()

    supervisor_agent_data = {
        "name": "Runtime Supervisor Agent",
        "provider": TEST_LLM_PROVIDER,
        "model_name": TEST_LLM_MODEL,
        "system_prompt": "You coordinate a small team.",
    }
    worker_agent_data = {
        "name": "Runtime Worker Agent",
        "provider": TEST_LLM_PROVIDER,
        "model_name": TEST_LLM_MODEL,
        "system_prompt": "You answer directly and concisely.",
    }

    supervisor_agent_id = client.post("/agents/", json=supervisor_agent_data).json()["id"]
    worker_agent_id = client.post("/agents/", json=worker_agent_data).json()["id"]

    supervisor_response = client.post(
        "/teams/supervisors",
        json={"agent_id": supervisor_agent_id, "strategy": "team_manager"},
    )
    assert supervisor_response.status_code == 201
    supervisor_id = supervisor_response.json()["id"]

    team_response = client.post(
        "/teams/",
        json={"name": "Runtime Team", "supervisor_id": supervisor_id},
    )
    assert team_response.status_code == 201
    team_id = team_response.json()["id"]

    member_response = client.post(
        f"/teams/{team_id}/members",
        json={"agent_id": worker_agent_id, "role": "assistant"},
    )
    assert member_response.status_code == 201

    start_response = client.post(f"/teams/{team_id}/start")
    assert start_response.status_code == 200
    assert start_response.json()["status"] == "started"

    running_response = client.get("/teams/running")
    assert running_response.status_code == 200
    assert running_response.json()["count"] >= 1


def test_chat_with_team_runtime(api_test_client):
    """Test chatting with a running team over the real runtime path."""
    client, _ = api_test_client
    active_agents.clear()
    active_teams.clear()

    supervisor_agent_data = {
        "name": "Chat Supervisor Agent",
        "provider": TEST_LLM_PROVIDER,
        "model_name": TEST_LLM_MODEL,
        "system_prompt": "Delegate to the worker and return the final answer.",
    }
    worker_agent_data = {
        "name": "Chat Worker Agent",
        "provider": TEST_LLM_PROVIDER,
        "model_name": TEST_LLM_MODEL,
        "system_prompt": "Reply with a short factual answer.",
    }

    supervisor_agent_id = client.post("/agents/", json=supervisor_agent_data).json()["id"]
    worker_agent_id = client.post("/agents/", json=worker_agent_data).json()["id"]

    supervisor_response = client.post(
        "/teams/supervisors",
        json={
            "agent_id": supervisor_agent_id,
            "system_prompt": "Coordinate the worker agent to answer the user.",
            "strategy": "team_manager",
        },
    )
    assert supervisor_response.status_code == 201
    supervisor_id = supervisor_response.json()["id"]

    team_response = client.post(
        "/teams/",
        json={"name": "Chat Runtime Team", "supervisor_id": supervisor_id},
    )
    assert team_response.status_code == 201
    team_id = team_response.json()["id"]

    member_response = client.post(
        f"/teams/{team_id}/members",
        json={"agent_id": worker_agent_id, "role": "assistant"},
    )
    assert member_response.status_code == 201

    chat_response = client.post(
        f"/teams/{team_id}/chat",
        json={"content": "What is 2 + 2?", "thread_id": f"team_runtime_{uuid.uuid4().hex}"},
    )
    assert chat_response.status_code == 200
    payload = chat_response.json()
    assert payload["response"]
    assert payload["thread_id"]
    assert "details" in payload
    assert "team_metrics" in payload["details"]
    assert payload["details"]["team_metrics"]["chats"] == 1
    assert any(event["route"] == "supervisor" for event in payload["trace"])


def test_team_metrics_endpoint(api_test_client):
    """Test team runtime metrics endpoint before and after chat activity."""
    client, _ = api_test_client
    active_agents.clear()
    active_teams.clear()

    supervisor_agent_id = client.post(
        "/agents/",
        json={
            "name": "Metrics Supervisor",
            "provider": TEST_LLM_PROVIDER,
            "model_name": TEST_LLM_MODEL,
            "system_prompt": "Coordinate workers.",
        },
    ).json()["id"]
    worker_agent_id = client.post(
        "/agents/",
        json={
            "name": "Metrics Worker",
            "provider": TEST_LLM_PROVIDER,
            "model_name": TEST_LLM_MODEL,
            "system_prompt": "Answer briefly.",
        },
    ).json()["id"]
    supervisor_id = client.post(
        "/teams/supervisors",
        json={"agent_id": supervisor_agent_id, "parallel_tool_calls": True},
    ).json()["id"]
    team_id = client.post(
        "/teams/",
        json={"name": "Metrics Team", "supervisor_id": supervisor_id},
    ).json()["id"]
    client.post(
        f"/teams/{team_id}/members",
        json={"agent_id": worker_agent_id, "role": "assistant"},
    )

    inactive_response = client.get(f"/teams/{team_id}/metrics")
    assert inactive_response.status_code == 200
    assert inactive_response.json()["active"] is False

    start_response = client.post(f"/teams/{team_id}/start")
    assert start_response.status_code == 200

    active_response = client.get(f"/teams/{team_id}/metrics")
    assert active_response.status_code == 200
    payload = active_response.json()
    assert payload["active"] is True
    assert payload["metrics"]["starts"] == 1
    assert payload["metrics"]["supervisor"]["parallel_delegations_total"] >= 0


def test_team_runtime_invalidated_on_team_update(api_test_client):
    """Updating a running team should invalidate its cached runtime."""
    client, _ = api_test_client
    active_agents.clear()
    active_teams.clear()

    supervisor_agent_id = client.post(
        "/agents/",
        json={
            "name": "Invalidate Supervisor",
            "provider": TEST_LLM_PROVIDER,
            "model_name": TEST_LLM_MODEL,
            "system_prompt": "Coordinate workers.",
        },
    ).json()["id"]
    worker_agent_id = client.post(
        "/agents/",
        json={
            "name": "Invalidate Worker",
            "provider": TEST_LLM_PROVIDER,
            "model_name": TEST_LLM_MODEL,
            "system_prompt": "Answer briefly.",
        },
    ).json()["id"]
    supervisor_id = client.post(
        "/teams/supervisors",
        json={"agent_id": supervisor_agent_id},
    ).json()["id"]
    team_id = client.post(
        "/teams/",
        json={"name": "Invalidate Team", "supervisor_id": supervisor_id},
    ).json()["id"]
    client.post(
        f"/teams/{team_id}/members",
        json={"agent_id": worker_agent_id, "role": "assistant"},
    )

    start_response = client.post(f"/teams/{team_id}/start")
    assert start_response.status_code == 200
    assert team_id in active_teams

    update_response = client.put(
        f"/teams/{team_id}",
        json={"description": "runtime changed"},
    )
    assert update_response.status_code == 200
    assert team_id not in active_teams


def test_team_runtime_invalidated_on_member_change(api_test_client):
    """Changing team membership should invalidate cached runtime."""
    client, _ = api_test_client
    active_agents.clear()
    active_teams.clear()

    supervisor_agent_id = client.post(
        "/agents/",
        json={
            "name": "Member Invalidate Supervisor",
            "provider": TEST_LLM_PROVIDER,
            "model_name": TEST_LLM_MODEL,
            "system_prompt": "Coordinate workers.",
        },
    ).json()["id"]
    worker_agent_id = client.post(
        "/agents/",
        json={
            "name": "Member Invalidate Worker",
            "provider": TEST_LLM_PROVIDER,
            "model_name": TEST_LLM_MODEL,
            "system_prompt": "Answer briefly.",
        },
    ).json()["id"]
    supervisor_id = client.post(
        "/teams/supervisors",
        json={"agent_id": supervisor_agent_id},
    ).json()["id"]
    team_id = client.post(
        "/teams/",
        json={"name": "Member Invalidate Team", "supervisor_id": supervisor_id},
    ).json()["id"]
    client.post(
        f"/teams/{team_id}/members",
        json={"agent_id": worker_agent_id, "role": "assistant"},
    )

    assert client.post(f"/teams/{team_id}/start").status_code == 200
    assert team_id in active_teams

    update_response = client.put(
        f"/teams/{team_id}/members/{worker_agent_id}",
        json={"role": "researcher"},
    )
    assert update_response.status_code == 200
    assert team_id not in active_teams


def test_team_direct_routing_to_member(api_test_client):
    """A team message can be routed directly to a specific member."""
    client, _ = api_test_client
    active_agents.clear()
    active_teams.clear()

    supervisor_agent_id = client.post(
        "/agents/",
        json={
            "name": "Direct Supervisor",
            "provider": TEST_LLM_PROVIDER,
            "model_name": TEST_LLM_MODEL,
            "system_prompt": "Coordinate workers.",
        },
    ).json()["id"]
    worker_agent_id = client.post(
        "/agents/",
        json={
            "name": "Direct Worker",
            "provider": TEST_LLM_PROVIDER,
            "model_name": TEST_LLM_MODEL,
            "system_prompt": "Reply with a short factual answer.",
        },
    ).json()["id"]
    supervisor_id = client.post(
        "/teams/supervisors",
        json={"agent_id": supervisor_agent_id},
    ).json()["id"]
    team_id = client.post(
        "/teams/",
        json={"name": "Direct Routing Team", "supervisor_id": supervisor_id},
    ).json()["id"]
    client.post(
        f"/teams/{team_id}/members",
        json={"agent_id": worker_agent_id, "role": "assistant"},
    )

    response = client.post(
        f"/teams/{team_id}/chat",
        json={
            "content": "What is 2 + 2?",
            "direct_to_agent_id": worker_agent_id,
            "thread_id": f"team_direct_{uuid.uuid4().hex}",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["responding_agent_id"]
    assert payload["trace"][0]["route"] == "direct"
    assert payload["trace"][0]["event_type"] == "direct_message"
    assert payload["details"]["team_metrics"]["direct_messages"] == 1


def test_team_direct_routing_policy_denied(api_test_client):
    """Direct routing should enforce member policy constraints."""
    client, _ = api_test_client
    active_agents.clear()
    active_teams.clear()

    supervisor_agent_id = client.post(
        "/agents/",
        json={
            "name": "Policy Supervisor",
            "provider": TEST_LLM_PROVIDER,
            "model_name": TEST_LLM_MODEL,
            "system_prompt": "Coordinate workers.",
        },
    ).json()["id"]
    worker_agent_id = client.post(
        "/agents/",
        json={
            "name": "Policy Worker",
            "provider": TEST_LLM_PROVIDER,
            "model_name": TEST_LLM_MODEL,
            "system_prompt": "Reply with a short factual answer.",
        },
    ).json()["id"]
    supervisor_id = client.post(
        "/teams/supervisors",
        json={"agent_id": supervisor_agent_id},
    ).json()["id"]
    team_id = client.post(
        "/teams/",
        json={"name": "Policy Team", "supervisor_id": supervisor_id},
    ).json()["id"]
    client.post(
        f"/teams/{team_id}/members",
        json={
            "agent_id": worker_agent_id,
            "role": "assistant",
            "params": {
                "can_receive_direct": False,
            },
        },
    )

    response = client.post(
        f"/teams/{team_id}/chat",
        json={
            "content": "What is 2 + 2?",
            "direct_to_agent_id": worker_agent_id,
        },
    )
    assert response.status_code == 403


def test_peer_message_role_policy():
    """Peer message policies should enforce sender and receiver constraints."""
    sender = {
        "agent_id": "agent_a",
        "role": "researcher",
        "params": {
            "allow_peer_messages": True,
            "can_message_roles": ["writer"],
        },
    }
    receiver = {
        "agent_id": "agent_b",
        "role": "writer",
        "params": {
            "allow_peer_messages": True,
            "accept_messages_from_roles": ["researcher"],
        },
    }
    allowed, reason = _peer_message_allowed(sender, receiver)
    assert allowed is True
    assert reason is None

    blocked_receiver = {
        "agent_id": "agent_c",
        "role": "critic",
        "params": {"allow_peer_messages": True},
    }
    allowed, reason = _peer_message_allowed(sender, blocked_receiver)
    assert allowed is False
    assert reason == "target role not allowed"


def test_team_member_policy_rejects_unknown_role_reference(api_test_client):
    """Policy role references must point to known team roles."""
    client, _ = api_test_client

    team_id = client.post("/teams/", json={"name": "Policy Ref Team"}).json()["id"]
    agent_id = client.post(
        "/agents/",
        json={
            "name": "Policy Ref Agent",
            "provider": "openai",
            "model_name": "gpt-4",
        },
    ).json()["id"]

    response = client.post(
        f"/teams/{team_id}/members",
        json={
            "agent_id": agent_id,
            "role": "writer",
            "params": {
                "allow_peer_messages": True,
                "can_message_roles": ["reviewer"],
            },
        },
    )
    assert response.status_code == 400
    assert "Unknown team role references" in response.json()["detail"]


def test_team_member_policy_rejects_unknown_agent_reference(api_test_client):
    """Policy agent references must point to known team members."""
    client, _ = api_test_client

    team_id = client.post("/teams/", json={"name": "Policy Agent Ref Team"}).json()["id"]
    agent_id = client.post(
        "/agents/",
        json={
            "name": "Policy Agent Ref",
            "provider": "openai",
            "model_name": "gpt-4",
        },
    ).json()["id"]

    response = client.post(
        f"/teams/{team_id}/members",
        json={
            "agent_id": agent_id,
            "role": "writer",
            "params": {
                "allow_peer_messages": True,
                "can_message_agents": ["agent_missing"],
            },
        },
    )
    assert response.status_code == 400
    assert "Unknown team agent references" in response.json()["detail"]


def test_team_member_policy_respects_team_config_flags(api_test_client):
    """Member policy should be rejected when it conflicts with team config."""
    client, _ = api_test_client

    team_id = client.post(
        "/teams/",
        json={
            "name": "Team Config Policy Guard",
            "config": {"allow_direct_messages": False},
        },
    ).json()["id"]
    agent_id = client.post(
        "/agents/",
        json={
            "name": "Direct Policy Agent",
            "provider": "openai",
            "model_name": "gpt-4",
        },
    ).json()["id"]

    response = client.post(
        f"/teams/{team_id}/members",
        json={
            "agent_id": agent_id,
            "role": "writer",
            "params": {"can_receive_direct": True},
        },
    )
    assert response.status_code == 400
    assert "Team configuration disables direct messages" in response.json()["detail"]


def test_team_member_policy_requires_some_supervisor_delegation(api_test_client):
    """A supervised team must keep at least one active delegable member."""
    client, _ = api_test_client

    supervisor_agent_id = client.post(
        "/agents/",
        json={
            "name": "Delegation Guard Supervisor",
            "provider": "openai",
            "model_name": "gpt-4",
        },
    ).json()["id"]
    worker_agent_id = client.post(
        "/agents/",
        json={
            "name": "Delegation Guard Worker",
            "provider": "openai",
            "model_name": "gpt-4",
        },
    ).json()["id"]
    supervisor_id = client.post(
        "/teams/supervisors",
        json={"agent_id": supervisor_agent_id},
    ).json()["id"]
    team_id = client.post(
        "/teams/",
        json={"name": "Delegation Guard Team", "supervisor_id": supervisor_id},
    ).json()["id"]

    response = client.post(
        f"/teams/{team_id}/members",
        json={
            "agent_id": worker_agent_id,
            "role": "writer",
            "params": {"allow_supervisor_delegation": False},
        },
    )
    assert response.status_code == 400
    assert "At least one active team member must allow supervisor delegation" in response.json()["detail"]
