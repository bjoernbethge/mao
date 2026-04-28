"""
Tests for the Agents API endpoints.
"""

import os
import uuid
from pathlib import Path

import duckdb
import httpx
import pytest
from langchain.agents import create_agent as lc_create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from mao.api.agents import active_agents
from mao.api.teams import active_teams

TEST_LLM_PROVIDER = os.environ.get("TEST_LLM_PROVIDER", "ollama")
TEST_LLM_MODEL = os.environ.get("TEST_LLM_MODEL", "gemma3:4b-cloud")


class ForceToolChoiceMiddleware(AgentMiddleware):
    async def awrap_model_call(self, request: ModelRequest, handler):
        return await handler(request.override(tool_choice="dangerous_add"))


@tool
async def dangerous_add(a: int, b: int) -> str:
    """Add two integers."""
    return str(a + b)


def test_create_agent(api_test_client):
    """Test creating a new agent."""
    client, _ = api_test_client

    # Test data for agent creation
    agent_data = {
        "name": "Test Agent",
        "provider": "anthropic",
        "model_name": "claude-3-sonnet-20240229",
        "system_prompt": "You are a helpful AI assistant.",
    }

    response = client.post("/agents", json=agent_data)

    assert response.status_code == 201
    data = response.json()
    assert data["name"] == agent_data["name"]
    assert data["provider"] == agent_data["provider"]
    assert data["model_name"] == agent_data["model_name"]
    assert data["system_prompt"] == agent_data["system_prompt"]
    assert data["skills"] is None
    assert data["skill_paths"] is None
    assert "use_react_agent" not in data
    assert "max_tokens_trimmed" not in data
    assert "llm_specific_kwargs" not in data
    assert "id" in data
    assert data["id"].startswith("agent_")


def test_list_agents(api_test_client):
    """Test listing all agents."""
    client, _ = api_test_client

    # Create an agent first
    agent_data = {
        "name": "Test Agent for List",
        "provider": "anthropic",
        "model_name": "claude-3-sonnet-20240229",
    }

    create_response = client.post("/agents", json=agent_data)
    assert create_response.status_code == 201

    # Get the list of all agents
    list_response = client.get("/agents")
    assert list_response.status_code == 200

    agents = list_response.json()
    assert isinstance(agents, list)
    assert len(agents) >= 1

    # Find the created agent in the list
    found = False
    for agent in agents:
        if agent["name"] == agent_data["name"]:
            found = True
            break

    assert found, "Created agent was not found in the list"


def test_get_agent(api_test_client):
    """Test getting an agent by ID."""
    client, _ = api_test_client

    # Create an agent first
    agent_data = {
        "name": "Test Agent for Get",
        "provider": "openai",
        "model_name": "gpt-4",
    }

    create_response = client.post("/agents", json=agent_data)
    assert create_response.status_code == 201
    created_agent = create_response.json()
    agent_id = created_agent["id"]

    # Get the agent by ID
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

    # Create an agent first
    agent_data = {
        "name": "Agent before update",
        "provider": "anthropic",
        "model_name": "claude-3-haiku-20240307",
    }

    create_response = client.post("/agents", json=agent_data)
    assert create_response.status_code == 201
    created_agent = create_response.json()
    agent_id = created_agent["id"]

    # Update the agent
    update_data = {
        "name": "Agent after update",
        "system_prompt": "This is an updated system prompt.",
    }

    update_response = client.put(f"/agents/{agent_id}", json=update_data)
    assert update_response.status_code == 200

    updated_agent = update_response.json()
    assert updated_agent["id"] == agent_id
    assert updated_agent["name"] == update_data["name"]
    assert updated_agent["system_prompt"] == update_data["system_prompt"]
    # Fields not updated should remain unchanged
    assert updated_agent["provider"] == agent_data["provider"]
    assert updated_agent["model_name"] == agent_data["model_name"]


def test_delete_agent(api_test_client):
    """Test deleting an agent."""
    client, _ = api_test_client

    # Create an agent first
    agent_data = {
        "name": "Agent to delete",
        "provider": "openai",
        "model_name": "gpt-3.5-turbo",
    }

    create_response = client.post("/agents", json=agent_data)
    assert create_response.status_code == 201
    created_agent = create_response.json()
    agent_id = created_agent["id"]

    # Delete the agent
    delete_response = client.delete(f"/agents/{agent_id}")
    assert delete_response.status_code == 204

    # Try to retrieve the deleted agent
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


def test_agents_schema_does_not_expose_legacy_columns(api_test_client):
    """Test the persisted agents schema matches the simplified API."""
    client, _ = api_test_client

    create_response = client.post(
        "/agents",
        json={
            "name": "Schema Agent",
            "provider": "anthropic",
            "model_name": "claude-3-haiku-20240307",
        },
    )
    assert create_response.status_code == 201

    conn = duckdb.connect(os.environ["MCP_DB_PATH"])
    try:
        columns = {
            row[1] for row in conn.execute("PRAGMA table_info('agents')").fetchall()
        }
    finally:
        conn.close()

    assert "use_react_agent" not in columns
    assert "max_tokens_trimmed" not in columns
    assert "llm_specific_kwargs" not in columns


def test_create_agent_with_skills(api_test_client):
    """Test creating an agent with explicit skills configuration."""
    client, _ = api_test_client
    skill_root = Path.cwd() / ".test_tmp" / f"agent_skills_{uuid.uuid4().hex}"
    skill_dir = skill_root / "reviewer"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: reviewer\ndescription: Review code carefully\n---\n\nCheck for regressions.\n",
        encoding="utf-8",
    )

    create_response = client.post(
        "/agents",
        json={
            "name": "Skilled Agent",
            "provider": "anthropic",
            "model_name": "claude-3-haiku-20240307",
            "skills": ["reviewer"],
            "skill_paths": [str(skill_root)],
        },
    )
    assert create_response.status_code == 201
    payload = create_response.json()
    assert payload["skills"] == ["reviewer"]
    assert payload["skill_paths"] == [str(skill_root)]


def test_start_agent_runtime(api_test_client):
    """Test starting an agent through the real runtime path."""
    client, _ = api_test_client

    create_response = client.post(
        "/agents",
        json={
            "name": "Runtime Agent",
            "provider": TEST_LLM_PROVIDER,
            "model_name": TEST_LLM_MODEL,
            "system_prompt": "You answer briefly.",
        },
    )
    assert create_response.status_code == 201
    agent_id = create_response.json()["id"]

    start_response = client.post(f"/agents/{agent_id}/start")
    assert start_response.status_code == 200
    assert start_response.json()["status"] == "started"

    running_response = client.get("/agents/running")
    assert running_response.status_code == 200
    assert running_response.json()["count"] >= 1


def test_chat_with_agent_runtime(api_test_client):
    """Test chatting with a running agent through the real runtime path."""
    client, _ = api_test_client

    create_response = client.post(
        "/agents",
        json={
            "name": "Chat Runtime Agent",
            "provider": TEST_LLM_PROVIDER,
            "model_name": TEST_LLM_MODEL,
            "system_prompt": "Answer with a short factual reply.",
        },
    )
    assert create_response.status_code == 201
    agent_id = create_response.json()["id"]

    start_response = client.post(f"/agents/{agent_id}/start")
    assert start_response.status_code == 200

    chat_response = client.post(
        f"/agents/{agent_id}/chat",
        json={"content": "What is 2 + 2?", "thread_id": f"agent_runtime_{uuid.uuid4().hex}"},
    )
    assert chat_response.status_code == 200
    payload = chat_response.json()
    assert payload["response"]
    assert payload["thread_id"]
    assert "details" in payload


def test_updating_agent_invalidates_running_team(api_test_client):
    """Updating an agent should invalidate any team runtime that depends on it."""
    client, _ = api_test_client
    active_agents.clear()
    active_teams.clear()

    supervisor_agent_id = client.post(
        "/agents/",
        json={
            "name": "Invalidate Team Supervisor",
            "provider": TEST_LLM_PROVIDER,
            "model_name": TEST_LLM_MODEL,
            "system_prompt": "Coordinate workers.",
        },
    ).json()["id"]
    worker_agent_id = client.post(
        "/agents/",
        json={
            "name": "Invalidate Team Worker",
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
        json={"name": "Agent Update Invalidates Team", "supervisor_id": supervisor_id},
    ).json()["id"]
    client.post(
        f"/teams/{team_id}/members",
        json={"agent_id": worker_agent_id, "role": "assistant"},
    )

    assert client.post(f"/teams/{team_id}/start").status_code == 200
    assert team_id in active_teams

    update_response = client.put(
        f"/agents/{worker_agent_id}",
        json={"system_prompt": "Updated prompt"},
    )
    assert update_response.status_code == 200
    assert team_id not in active_teams


@pytest.mark.asyncio
async def test_chat_with_agent_hitl_resume_real_cloud(
    api_test_client, real_tool_calling_cloud_model
):
    """Test HITL interrupt and resume with a real cloud model."""
    _, test_api = api_test_client
    agent_id = f"agent_hitl_{uuid.uuid4().hex[:8]}"
    thread_id = f"agent_hitl_thread_{uuid.uuid4().hex}"
    transport = httpx.ASGITransport(app=test_api)

    model = init_chat_model(
        real_tool_calling_cloud_model,
        model_provider="ollama",
        temperature=0,
    )
    agent_app = lc_create_agent(
        model=model,
        tools=[dangerous_add],
        system_prompt="Use the dangerous_add tool for addition requests.",
        middleware=[
            ForceToolChoiceMiddleware(),
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "dangerous_add": {
                        "allowed_decisions": ["approve", "edit", "reject"]
                    }
                }
            ),
        ],
        checkpointer=InMemorySaver(),
        name="hitl_runtime_test",
    )
    active_agents[agent_id] = {
        "agent": agent_app,
        "config": {
            "name": "hitl_runtime_test",
            "provider": "ollama",
            "model_name": real_tool_calling_cloud_model,
        },
    }

    try:
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            first_response = await client.post(
                f"/agents/{agent_id}/chat",
                json={
                    "content": "Call dangerous_add with a=2 and b=3.",
                    "thread_id": thread_id,
                },
            )
            assert first_response.status_code == 200
            first_payload = first_response.json()
            assert first_payload["response"] == "Approval required."
            assert first_payload["thread_id"] == thread_id
            assert first_payload["details"]["__interrupt__"]
            tool_calls = first_payload["details"]["messages"][-1]["tool_calls"]
            assert tool_calls[0]["name"] == "dangerous_add"
            assert tool_calls[0]["args"] == {"a": 2, "b": 3}

            second_response = await client.post(
                f"/agents/{agent_id}/chat",
                json={
                    "content": "resume",
                    "thread_id": thread_id,
                    "approval_decisions": [{"type": "approve"}],
                },
            )
            assert second_response.status_code == 200
            second_payload = second_response.json()
            assert second_payload["thread_id"] == thread_id
            assert "5" in second_payload["response"]
            assert "__interrupt__" not in second_payload["details"]
    finally:
        active_agents.pop(agent_id, None)


def test_delete_agent_with_dependencies(api_test_client):
    """Tests that deleting an agent cleans up all dependent relationships"""
    client, _ = api_test_client

    # 1. Create an agent
    agent_data = {
        "name": "Agent to delete",
        "provider": "anthropic",
        "model_name": "claude-3-haiku-20240307",
    }
    agent_response = client.post("/agents", json=agent_data)
    assert agent_response.status_code == 201
    agent_id = agent_response.json()["id"]

    # 2. Create a supervisor using the agent
    supervisor_data = {"agent_id": agent_id, "strategy": "team_manager"}
    supervisor_response = client.post("/teams/supervisors", json=supervisor_data)
    assert supervisor_response.status_code == 201
    supervisor_id = supervisor_response.json()["id"]

    # 3. Create a team with the supervisor
    team_data = {
        "name": "Test Team",
        "workflow_type": "sequential",
        "supervisor_id": supervisor_id,
    }
    team_response = client.post("/teams", json=team_data)
    assert team_response.status_code == 201
    team_id = team_response.json()["id"]
    assert team_response.json()["supervisor_id"] == supervisor_id

    # 4. Add agent as team member
    member_data = {"agent_id": agent_id, "role": "assistant", "order_index": 1}
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
