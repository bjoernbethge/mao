"""
Agent-related API endpoints.
"""

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request, status
from langgraph.types import Command

from .api import active_agents, get_config_db
from .db import ConfigDB
from .helpers import create_and_start_agent, extract_response_text
from .teams import _invalidate_teams_for_agent
from .models import (
    AgentCreate,
    AgentMessage,
    AgentResponse,
    AgentResponseMessage,
    AgentUpdate,
    ToolResponse,
)
from ..agents import _build_invoke_config
from ..observability import langsmith_trace

# Create router
router = APIRouter(prefix="/agents", tags=["agents"])


# Agent management endpoints
@router.post("", response_model=AgentResponse, status_code=status.HTTP_201_CREATED)
async def create_new_agent(agent: AgentCreate, db: ConfigDB = Depends(get_config_db)):
    """Creates a new agent"""
    agent_id = f"agent_{uuid.uuid4().hex[:8]}"

    await db.create_agent(
        agent_id=agent_id,
        name=agent.name,
        provider=agent.provider,
        model_name=agent.model_name,
        system_prompt=agent.system_prompt,
        skills=agent.skills,
        skill_paths=agent.skill_paths,
    )

    return await db.get_agent(agent_id)


@router.get("", response_model=list[AgentResponse])
async def list_agents(
    limit: int | None = Query(50, description="Maximum number of agents to return"),
    offset: int | None = Query(0, description="Number of agents to skip"),
    db: ConfigDB = Depends(get_config_db),
):
    """Lists all configured agents"""
    agents = await db.list_agents(limit=limit, offset=offset)
    return agents


@router.get("/running")
async def list_running_agents():
    """Lists all running agents"""
    return {
        "count": len(active_agents),
        "agents": [
            {
                "id": agent_id,
                "name": info["config"]["name"],
                "provider": info["config"]["provider"],
                "model_name": info["config"]["model_name"],
            }
            for agent_id, info in active_agents.items()
        ],
    }


@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent_by_id(
    agent_id: str = Path(..., description="Agent ID"),
    db: ConfigDB = Depends(get_config_db),
):
    """Gets an agent by its ID"""
    agent = await db.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    return agent


@router.put("/{agent_id}", response_model=AgentResponse)
async def update_agent_by_id(
    agent: AgentUpdate,
    agent_id: str = Path(..., description="Agent ID"),
    db: ConfigDB = Depends(get_config_db),
):
    """Updates an agent"""
    existing = await db.get_agent(agent_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

    update_data = {k: v for k, v in agent.model_dump().items() if v is not None}
    if update_data:
        await db.update_agent(agent_id, **update_data)
        active_agents.pop(agent_id, None)
        _invalidate_teams_for_agent(agent_id)

    return await db.get_agent(agent_id)


@router.delete("/{agent_id}", status_code=204)
async def delete_agent_by_id(
    agent_id: str = Path(..., description="Agent ID"),
    db: ConfigDB = Depends(get_config_db),
):
    """Deletes an agent"""
    existing = await db.get_agent(agent_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

    # Stop running agent if active
    if agent_id in active_agents:
        active_agents.pop(agent_id, None)
    _invalidate_teams_for_agent(agent_id)

    await db.delete_agent(agent_id)
    return None


# Agent runtime endpoints
@router.post("/{agent_id}/start", status_code=200)
async def start_agent(
    agent_id: str = Path(..., description="Agent ID"),
    db: ConfigDB = Depends(get_config_db),
):
    """Starts an agent"""
    if agent_id in active_agents:
        # Agent is already running
        return {"status": "already_running", "agent_id": agent_id}

    agent_config = await db.get_agent(agent_id)
    if not agent_config:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

    try:
        await create_and_start_agent(db, agent_id, agent_config, active_agents)
        return {"status": "started", "agent_id": agent_id}
    except Exception as e:
        logging.exception(f"Failed to start agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start agent: {str(e)}")


@router.post("/{agent_id}/stop", status_code=200)
async def stop_agent(agent_id: str = Path(..., description="Agent ID")):
    """Stops a running agent"""
    if agent_id not in active_agents:
        raise HTTPException(
            status_code=404, detail=f"No running agent with ID {agent_id}"
        )

    active_agents.pop(agent_id, None)
    return {"status": "stopped", "agent_id": agent_id}


@router.post("/{agent_id}/chat", response_model=AgentResponseMessage)
async def chat_with_agent(
    message: AgentMessage,
    request: Request,
    agent_id: str = Path(..., description="Agent ID"),
):
    """Sends a message to a running agent"""
    if agent_id not in active_agents:
        raise HTTPException(
            status_code=404, detail=f"No running agent with ID {agent_id}"
        )

    agent_app = active_agents[agent_id]["agent"]

    # Prepare the message for the agent
    formatted_message = {"role": "user", "content": message.content}

    thread_id = message.thread_id or f"thread_{uuid.uuid4().hex}"

    try:
        config = _build_invoke_config(
            thread_id=thread_id,
            run_name="agent_chat",
            tags=["mao", "agent", agent_id],
            metadata={
                "agent_id": agent_id,
                "agent_name": active_agents[agent_id]["config"]["name"],
                "provider": active_agents[agent_id]["config"]["provider"],
                "model_name": active_agents[agent_id]["config"]["model_name"],
                "skills": active_agents[agent_id]["config"].get("skills") or [],
                "http_path": request.url.path,
            },
        )
        with langsmith_trace(
            run_name="agent_chat",
            tags=config.get("tags"),
            metadata=config.get("metadata"),
        ):
            if message.approval_decisions:
                response = await agent_app.ainvoke(
                    Command(resume={"decisions": message.approval_decisions}),
                    config=config,
                )
            else:
                response = await agent_app.ainvoke(
                    {
                        "messages": [formatted_message],
                        "response_schema": message.response_schema,
                    },
                    config=config,
                )

        # Extract response
        response_message = "No response received."
        if response:
            response_message, _ = extract_response_text(response)
    except Exception as e:
        logging.error(f"Error invoking agent {agent_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to communicate with agent: {str(e)}"
        )

    return {
        "response": response_message,
        "thread_id": thread_id,
        "details": (
            response if isinstance(response, dict) else {"raw_response": str(response)}
        ),
    }


# Tool management endpoints
@router.get("/{agent_id}/tools", response_model=list[ToolResponse])
async def list_agent_tools(
    agent_id: str = Path(..., description="Agent ID"),
    enabled_only: bool = Query(False, description="Only return enabled tools"),
    db: ConfigDB = Depends(get_config_db),
):
    """Lists all tools assigned to an agent"""
    agent = await db.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

    tools = await db.get_agent_tools(agent_id, enabled_only=enabled_only)
    return tools
