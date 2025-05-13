"""
Agent-related API endpoints.
"""

import logging
import uuid
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Query, Path, status

from ..agents import create_agent
from .db import ConfigDB
from ..mcp import MCPClient

from .models import (
    AgentCreate, 
    AgentUpdate, 
    AgentResponse, 
    AgentMessage, 
    AgentResponseMessage,
    ToolResponse
)
from .api import get_config_db, get_active_agents, active_agents

# Create router
router = APIRouter(prefix="/agents", tags=["agents"])

# Agent management endpoints
@router.post("", response_model=AgentResponse, status_code=status.HTTP_201_CREATED)
async def create_new_agent(
    agent: AgentCreate, 
    db: ConfigDB = Depends(get_config_db)
):
    """Creates a new agent"""
    agent_id = f"agent_{uuid.uuid4().hex[:8]}"
    
    await db.create_agent(
        agent_id=agent_id,
        name=agent.name,
        provider=agent.provider,
        model_name=agent.model_name,
        system_prompt=agent.system_prompt,
        use_react_agent=agent.use_react_agent,
        max_tokens_trimmed=agent.max_tokens_trimmed,
        llm_specific_kwargs=agent.llm_specific_kwargs
    )
    
    return await db.get_agent(agent_id)

@router.get("", response_model=List[AgentResponse])
async def list_agents(
    limit: Optional[int] = Query(50, description="Maximum number of agents to return"),
    offset: Optional[int] = Query(0, description="Number of agents to skip"),
    db: ConfigDB = Depends(get_config_db)
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
                "model_name": info["config"]["model_name"]
            }
            for agent_id, info in active_agents.items()
        ]
    }

@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent_by_id(
    agent_id: str = Path(..., description="Agent ID"),
    db: ConfigDB = Depends(get_config_db)
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
    db: ConfigDB = Depends(get_config_db)
):
    """Updates an agent"""
    existing = await db.get_agent(agent_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    update_data = {k: v for k, v in agent.model_dump().items() if v is not None}
    if update_data:
        await db.update_agent(agent_id, **update_data)
    
    return await db.get_agent(agent_id)

@router.delete("/{agent_id}", status_code=204)
async def delete_agent_by_id(
    agent_id: str = Path(..., description="Agent ID"),
    db: ConfigDB = Depends(get_config_db)
):
    """Deletes an agent"""
    existing = await db.get_agent(agent_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    # Stop running agent if active
    if agent_id in active_agents:
        active_agents.pop(agent_id, None)
    
    await db.delete_agent(agent_id)
    return None

# Agent runtime endpoints
@router.post("/{agent_id}/start", status_code=200)
async def start_agent(
    agent_id: str = Path(..., description="Agent ID"),
    db: ConfigDB = Depends(get_config_db)
):
    """Starts an agent"""
    if agent_id in active_agents:
        # Agent is already running
        return {"status": "already_running", "agent_id": agent_id}
    
    agent_config = await db.get_agent(agent_id)
    if not agent_config:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    try:
        # Get tools for the agent
        tools = None
        agent_tools = await db.get_agent_tools(agent_id, enabled_only=True)
        if agent_tools:
            # Import necessary modules for creating tools
            from langchain_mcp_adapters.tools import load_mcp_tools
            try:
                # Convert DB tool objects to LangChain tools
                tool_list = []
                for tool in agent_tools:
                    # Create a simple tool description for use in agents
                    tool_desc = {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", {})
                    }
                    tool_list.append(tool_desc)
                tools = tool_list
            except Exception as e:
                logging.warning(f"Failed to load tools for agent {agent_id}: {e}")
                tools = None
        
        # Create and start the agent
        agent_app = await create_agent(
            provider=agent_config['provider'],
            model_name=agent_config['model_name'],
            agent_name=agent_config['name'],
            system_prompt=agent_config.get('system_prompt'),
            use_react_agent=agent_config.get('use_react_agent', True),
            max_tokens_trimmed=agent_config.get('max_tokens_trimmed', 3000),
            llm_specific_kwargs=agent_config.get('llm_specific_kwargs'),
            tools=tools
        )
        
        # Add agent to active agents
        active_agents[agent_id] = {"agent": agent_app, "config": agent_config}
        
        return {"status": "started", "agent_id": agent_id}
    except Exception as e:
        logging.exception(f"Failed to start agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start agent: {str(e)}")

@router.post("/{agent_id}/stop", status_code=200)
async def stop_agent(agent_id: str = Path(..., description="Agent ID")):
    """Stops a running agent"""
    if agent_id not in active_agents:
        raise HTTPException(status_code=404, detail=f"No running agent with ID {agent_id}")
    
    active_agents.pop(agent_id, None)
    return {"status": "stopped", "agent_id": agent_id}

@router.post("/{agent_id}/chat", response_model=AgentResponseMessage)
async def chat_with_agent(
    message: AgentMessage,
    agent_id: str = Path(..., description="Agent ID")
):
    """Sends a message to a running agent"""
    if agent_id not in active_agents:
        raise HTTPException(status_code=404, detail=f"No running agent with ID {agent_id}")
    
    agent_app = active_agents[agent_id]["agent"]
    
    # Prepare the message for the agent
    formatted_message = {
        "role": "user",
        "content": message.content
    }
    
    thread_id = message.thread_id or f"thread_{uuid.uuid4().hex}"
    
    try:
        # Send message to agent
        response = await agent_app.ainvoke(
            {"messages": [formatted_message]},
            config={"configurable": {"thread_id": thread_id}}
        )
        
        # Extract response
        response_message = "No response received."
        if response:
            if isinstance(response, dict) and "messages" in response and response["messages"]:
                last_message = response["messages"][-1]
                if hasattr(last_message, "content"):
                    response_message = last_message.content
                elif isinstance(last_message, dict) and "content" in last_message:
                    response_message = last_message["content"]
            elif hasattr(response, "content"):
                response_message = response.content
            elif isinstance(response, str):
                response_message = response
    except Exception as e:
        logging.error(f"Error invoking agent {agent_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to communicate with agent: {str(e)}"
        )
    
    return {
        "response": response_message,
        "thread_id": thread_id,
        "details": response if isinstance(response, dict) else {"raw_response": str(response)}
    }

# Tool management endpoints
@router.get("/{agent_id}/tools", response_model=List[ToolResponse])
async def list_agent_tools(
    agent_id: str = Path(..., description="Agent ID"),
    enabled_only: bool = Query(False, description="Only return enabled tools"),
    db: ConfigDB = Depends(get_config_db)
):
    """Lists all tools assigned to an agent"""
    agent = await db.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    tools = await db.get_agent_tools(agent_id, enabled_only=enabled_only)
    return tools 