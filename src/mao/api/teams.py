"""
Team-related API endpoints.
Provides functionality to manage teams, supervisors, and team members.
"""

import logging
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from ..agents import Supervisor, create_agent
from .api import active_agents, get_config_db
from .db import ConfigDB
from .helpers import create_and_start_agent
from .models import (
    SupervisorCreate,
    SupervisorResponse,
    SupervisorUpdate,
    TeamCreate,
    TeamMemberCreate,
    TeamMemberResponse,
    TeamMemberUpdate,
    TeamMessage,
    TeamResponse,
    TeamResponseMessage,
    TeamUpdate,
)

# Create router
router = APIRouter(prefix="/teams", tags=["teams"])

# Global state for active supervisors and teams
active_supervisors: dict[str, dict[str, Any]] = {}
active_teams: dict[str, dict[str, Any]] = {}


# Team management endpoints
@router.post("", response_model=TeamResponse, status_code=201)
async def create_team(team: TeamCreate, db: ConfigDB = Depends(get_config_db)):
    """Creates a new team"""
    team_id = f"team_{uuid.uuid4().hex[:8]}"

    # Validate supervisor if provided
    if team.supervisor_id:
        supervisor = await db.get_supervisor(team.supervisor_id)
        if not supervisor:
            raise HTTPException(
                status_code=404, detail=f"Supervisor {team.supervisor_id} not found"
            )

    # Create team
    await db.create_team(
        team_id=team_id,
        name=team.name,
        description=team.description,
        workflow_type=team.workflow_type,
        supervisor_id=team.supervisor_id,
        config=team.config,
        is_active=team.is_active,
    )

    return await db.get_team(team_id)


@router.get("", response_model=list[TeamResponse])
async def list_teams(
    supervisor_id: str | None = None,
    active_only: bool = False,
    db: ConfigDB = Depends(get_config_db),
):
    """Lists all teams, optionally filtered"""
    teams = await db.list_teams(supervisor_id=supervisor_id, active_only=active_only)
    return teams


@router.get("/running")
async def list_running_teams():
    """Lists all running teams"""
    return {
        "count": len(active_teams),
        "teams": [
            {
                "id": team_id,
                "name": info["config"]["name"],
                "supervisor_id": info["config"].get("supervisor_id"),
            }
            for team_id, info in active_teams.items()
        ],
    }


@router.get("/{team_id}", response_model=TeamResponse)
async def get_team_by_id(team_id: str, db: ConfigDB = Depends(get_config_db)):
    """Gets a team by its ID"""
    team = await db.get_team(team_id)
    if not team:
        raise HTTPException(status_code=404, detail=f"Team {team_id} not found")
    return team


@router.put("/{team_id}", response_model=TeamResponse)
async def update_team_by_id(
    team_id: str, team: TeamUpdate, db: ConfigDB = Depends(get_config_db)
):
    """Updates a team"""
    existing = await db.get_team(team_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Team {team_id} not found")

    # Validate supervisor if provided
    if team.supervisor_id:
        supervisor = await db.get_supervisor(team.supervisor_id)
        if not supervisor:
            raise HTTPException(
                status_code=404, detail=f"Supervisor {team.supervisor_id} not found"
            )

    update_data = {k: v for k, v in team.model_dump().items() if v is not None}
    if isinstance(update_data, dict) and update_data:
        await db.update_team(team_id, **update_data)

    return await db.get_team(team_id)


@router.delete("/{team_id}", status_code=204)
async def delete_team_by_id(team_id: str, db: ConfigDB = Depends(get_config_db)):
    """Deletes a team"""
    existing = await db.get_team(team_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Team {team_id} not found")

    # Stop running team if active
    if team_id in active_teams:
        active_teams.pop(team_id, None)

    await db.delete_team(team_id)
    return None


# Team member management
@router.post("/{team_id}/members", response_model=TeamMemberResponse, status_code=201)
async def add_team_member(
    team_id: str, member: TeamMemberCreate, db: ConfigDB = Depends(get_config_db)
):
    """Adds an agent to a team"""
    # Validate team
    team = await db.get_team(team_id)
    if not team:
        raise HTTPException(status_code=404, detail=f"Team {team_id} not found")

    # Validate agent
    agent = await db.get_agent(member.agent_id)
    if not agent:
        raise HTTPException(
            status_code=404, detail=f"Agent {member.agent_id} not found"
        )

    # Add member data to database
    member_data_dict = member.model_dump(exclude_unset=True)
    member_data_dict["team_id"] = team_id

    # Set default order if not provided
    if "order_index" not in member_data_dict:
        # Get highest order index and add 1
        existing_members = await db.get_team_members(team_id)
        highest_order = max(
            [m.get("order_index", 0) or 0 for m in existing_members], default=0
        )
        member_data_dict["order_index"] = highest_order + 1

    await db.add_team_member(
        team_id=team_id,
        agent_id=member.agent_id,
        role=member.role,
        order_index=member.order_index,
        is_active=member.is_active,
        params=member.params,
    )

    # Return the newly added member
    members = await db.get_team_members(team_id)
    for m in members:
        if m["agent_id"] == member.agent_id:
            return {
                "team_id": team_id,
                "agent_id": m["agent_id"],
                "role": m["role"],
                "order_index": m.get("order_index"),
                "is_active": m["is_active"],
                "params": m.get("params"),
                "created_at": m["created_at"],
                "updated_at": m.get("updated_at"),
            }

    # This should not happen
    raise HTTPException(status_code=500, detail="Failed to retrieve added team member")


@router.get("/{team_id}/members", response_model=list[TeamMemberResponse])
async def get_team_members(
    team_id: str, active_only: bool = False, db: ConfigDB = Depends(get_config_db)
):
    """Gets all members of a team"""
    # Validate team
    team = await db.get_team(team_id)
    if not team:
        raise HTTPException(status_code=404, detail=f"Team {team_id} not found")

    members = await db.get_team_members(team_id, active_only=active_only)
    return members


@router.put("/{team_id}/members/{agent_id}", response_model=TeamMemberResponse)
async def update_team_member(
    team_id: str,
    agent_id: str,
    member: TeamMemberUpdate,
    db: ConfigDB = Depends(get_config_db),
):
    """Updates a team member"""
    # Validate team
    team = await db.get_team(team_id)
    if not team:
        raise HTTPException(status_code=404, detail=f"Team {team_id} not found")

    # Check if member exists
    members = await db.get_team_members(team_id)
    found = False
    for m in members:
        if m["agent_id"] == agent_id:
            found = True
            break

    if not found:
        raise HTTPException(
            status_code=404,
            detail=f"Agent {agent_id} is not a member of team {team_id}",
        )

    # Update member
    update_data = {k: v for k, v in member.model_dump().items() if v is not None}
    if update_data:
        await db.update_team_member(team_id, agent_id, **update_data)

    # Return updated member
    members = await db.get_team_members(team_id)
    for m in members:
        if m["agent_id"] == agent_id:
            return {
                "team_id": team_id,
                "agent_id": m["agent_id"],
                "role": m["role"],
                "order_index": m.get("order_index"),
                "is_active": m["is_active"],
                "params": m.get("params"),
                "created_at": m["created_at"],
                "updated_at": m.get("updated_at"),
            }

    # This should not happen
    raise HTTPException(
        status_code=500, detail="Failed to retrieve updated team member"
    )


@router.delete("/{team_id}/members/{agent_id}", status_code=204)
async def remove_team_member(
    team_id: str, agent_id: str, db: ConfigDB = Depends(get_config_db)
):
    """Removes an agent from a team"""
    # Validate team
    team = await db.get_team(team_id)
    if not team:
        raise HTTPException(status_code=404, detail=f"Team {team_id} not found")

    # Check if member exists
    members = await db.get_team_members(team_id)
    found = False
    for m in members:
        if m["agent_id"] == agent_id:
            found = True
            break

    if not found:
        raise HTTPException(
            status_code=404,
            detail=f"Agent {agent_id} is not a member of team {team_id}",
        )

    # Remove member
    await db.remove_team_member(team_id, agent_id)
    return None


# Supervisor management endpoints
@router.post("/supervisors", response_model=SupervisorResponse, status_code=201)
async def create_supervisor(
    supervisor: SupervisorCreate, db: ConfigDB = Depends(get_config_db)
):
    """Creates a new supervisor"""
    supervisor_id = f"supervisor_{uuid.uuid4().hex[:8]}"

    # Validate agent
    agent = await db.get_agent(supervisor.agent_id)
    if not agent:
        raise HTTPException(
            status_code=404, detail=f"Agent {supervisor.agent_id} not found"
        )

    # Create supervisor
    await db.create_supervisor(
        supervisor_id=supervisor_id,
        agent_id=supervisor.agent_id,
        system_prompt=supervisor.system_prompt,
        strategy=supervisor.strategy,
        add_handoff_back_messages=supervisor.add_handoff_back_messages,
        parallel_tool_calls=supervisor.parallel_tool_calls,
        config=supervisor.config,
    )

    return await db.get_supervisor(supervisor_id)


@router.get("/supervisors", response_model=list[SupervisorResponse])
async def list_supervisors(
    agent_id: str | None = None, db: ConfigDB = Depends(get_config_db)
):
    """Lists all supervisors, optionally filtered by agent_id"""
    supervisors = await db.list_supervisors(agent_id=agent_id)
    return supervisors


@router.get("/supervisors/{supervisor_id}", response_model=SupervisorResponse)
async def get_supervisor_by_id(
    supervisor_id: str, db: ConfigDB = Depends(get_config_db)
):
    """Gets a supervisor by its ID"""
    supervisor = await db.get_supervisor(supervisor_id)
    if not supervisor:
        raise HTTPException(
            status_code=404, detail=f"Supervisor {supervisor_id} not found"
        )
    return supervisor


@router.put("/supervisors/{supervisor_id}", response_model=SupervisorResponse)
async def update_supervisor_by_id(
    supervisor_id: str,
    supervisor: SupervisorUpdate,
    db: ConfigDB = Depends(get_config_db),
):
    """Updates a supervisor"""
    existing = await db.get_supervisor(supervisor_id)
    if not existing:
        raise HTTPException(
            status_code=404, detail=f"Supervisor {supervisor_id} not found"
        )

    # Validate agent if provided
    if supervisor.agent_id:
        agent = await db.get_agent(supervisor.agent_id)
        if not agent:
            raise HTTPException(
                status_code=404, detail=f"Agent {supervisor.agent_id} not found"
            )

    update_data = {k: v for k, v in supervisor.model_dump().items() if v is not None}
    if update_data:
        await db.update_supervisor(supervisor_id, **update_data)

    return await db.get_supervisor(supervisor_id)


@router.delete("/supervisors/{supervisor_id}", status_code=204)
async def delete_supervisor_by_id(
    supervisor_id: str, db: ConfigDB = Depends(get_config_db)
):
    """Deletes a supervisor"""
    existing = await db.get_supervisor(supervisor_id)
    if not existing:
        raise HTTPException(
            status_code=404, detail=f"Supervisor {supervisor_id} not found"
        )

    await db.delete_supervisor(supervisor_id)
    return None


# Team runtime endpoints
@router.post("/{team_id}/start", status_code=200)
async def start_team(team_id: str, db: ConfigDB = Depends(get_config_db)):
    """Starts a team with its supervisor and agents"""
    if team_id in active_teams:
        # Team is already running
        return {"status": "already_running", "team_id": team_id}

    team_config = await db.get_team(team_id)
    if not team_config:
        raise HTTPException(status_code=404, detail=f"Team {team_id} not found")

    # Get team members
    members = await db.get_team_members(team_id, active_only=True)
    if not members:
        raise HTTPException(
            status_code=400, detail=f"Team {team_id} has no active members"
        )

    try:
        # Start all agent team members if they aren't already running
        for member in members:
            agent_id = member["agent_id"]
            if agent_id not in active_agents:
                agent_config = await db.get_agent(agent_id)
                if agent_config:
                    await create_and_start_agent(
                        db, agent_id, agent_config, active_agents
                    )

        # Handle supervisor if configured
        supervisor_app = None
        supervisor_id = team_config.get("supervisor_id")
        supervisor_config = None

        if supervisor_id:
            supervisor_config = await db.get_supervisor(supervisor_id)
            if supervisor_config:
                # Get all agents in this team
                agent_instances = []
                for member in members:
                    agent_id = member["agent_id"]
                    if agent_id in active_agents:
                        agent_instances.append(
                            {
                                "agent": active_agents[agent_id]["agent"],
                                "role": member["role"],
                                "name": active_agents[agent_id]["config"]["name"],
                            }
                        )

                # Get supervisor agent configuration
                supervisor_agent_id = supervisor_config["agent_id"]
                if supervisor_agent_id not in active_agents:
                    supervisor_agent_config = await db.get_agent(supervisor_agent_id)
                    if not supervisor_agent_config:
                        raise HTTPException(
                            status_code=404,
                            detail=f"Supervisor agent {supervisor_agent_id} not found",
                        )

                    # Create the supervisor agent if not already active
                    supervisor_agent_app = await create_agent(
                        provider=supervisor_agent_config["provider"],
                        model_name=supervisor_agent_config["model_name"],
                        agent_name=supervisor_agent_config["name"],
                        system_prompt=supervisor_agent_config.get("system_prompt"),
                        use_react_agent=supervisor_agent_config.get(
                            "use_react_agent", True
                        ),
                        max_tokens_trimmed=supervisor_agent_config.get(
                            "max_tokens_trimmed", 3000
                        ),
                        llm_specific_kwargs=supervisor_agent_config.get(
                            "llm_specific_kwargs"
                        ),
                    )
                    active_agents[supervisor_agent_id] = {
                        "agent": supervisor_agent_app,
                        "config": supervisor_agent_config,
                    }

                # Initialize supervisor
                # Get the actual agent instances from active_agents
                agent_apps = [a["agent"] for a in agent_instances]

                # Extract supervisor-specific parameters from config
                supervisor_params = {
                    "add_handoff_back_messages": supervisor_config.get(
                        "add_handoff_back_messages", True
                    ),
                    "parallel_tool_calls": supervisor_config.get(
                        "parallel_tool_calls", True
                    ),
                }

                # Add any additional configuration from the supervisor_config.config field
                config_value = supervisor_config.get("config")
                if isinstance(config_value, dict):
                    supervisor_params.update(config_value)

                # Create the supervisor
                supervisor = Supervisor(
                    agents=agent_apps,
                    supervisor_provider=active_agents[supervisor_agent_id]["config"][
                        "provider"
                    ],
                    supervisor_model_name=active_agents[supervisor_agent_id]["config"][
                        "model_name"
                    ],
                    supervisor_system_prompt=supervisor_config.get("system_prompt")
                    or "You are a supervisor that coordinates a team of specialized agents.",
                    **supervisor_params,  # Pass all supervisor-specific parameters
                )

                # Initialize the supervisor
                supervisor_app = await supervisor.init_supervisor()

        # Add team to active teams
        active_teams[team_id] = {
            "config": team_config,
            "members": members,
            "supervisor": supervisor_app,
            "supervisor_config": supervisor_config,
        }

        return {"status": "started", "team_id": team_id}
    except Exception as e:
        logging.exception(f"Failed to start team {team_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start team: {str(e)}")


@router.post("/{team_id}/stop", status_code=200)
async def stop_team(team_id: str):
    """Stops a running team"""
    if team_id not in active_teams:
        raise HTTPException(
            status_code=404, detail=f"No running team with ID {team_id}"
        )

    active_teams.pop(team_id, None)

    # Note: We don't stop the individual agents as they could be used by other teams
    # Agents are managed through their own endpoints

    return {"status": "stopped", "team_id": team_id}


@router.post("/{team_id}/chat", response_model=TeamResponseMessage)
async def chat_with_team(
    team_id: str, message: TeamMessage, db: ConfigDB = Depends(get_config_db)
):
    """Sends a message to a team"""
    # Validate team
    team = await db.get_team(team_id)
    if not team:
        raise HTTPException(status_code=404, detail=f"Team {team_id} not found")

    # Check if team has a supervisor
    if not team["supervisor_id"]:
        raise HTTPException(status_code=400, detail="Team does not have a supervisor")

    # Get supervisor
    supervisor_id = team["supervisor_id"]
    supervisor_config = await db.get_supervisor(supervisor_id)
    if not supervisor_config:
        raise HTTPException(
            status_code=404, detail=f"Supervisor {supervisor_id} not found"
        )

    # Get supervisor agent
    agent_id = supervisor_config["agent_id"]

    # Check if supervisor agent is running
    if agent_id not in active_agents:
        # Try to start the agent
        agent_config = await db.get_agent(agent_id)
        if not agent_config:
            raise HTTPException(
                status_code=404, detail=f"Supervisor agent {agent_id} not found"
            )

        try:
            # Create and start the agent
            agent_app = await create_agent(
                provider=agent_config["provider"],
                model_name=agent_config["model_name"],
                agent_name=agent_config["name"],
                system_prompt=supervisor_config.get("system_prompt")
                or agent_config.get("system_prompt"),
                use_react_agent=agent_config.get("use_react_agent", True),
                max_tokens_trimmed=agent_config.get("max_tokens_trimmed", 3000),
                llm_specific_kwargs=agent_config.get("llm_specific_kwargs"),
            )

            # Add agent to active agents
            active_agents[agent_id] = {"agent": agent_app, "config": agent_config}
        except Exception as e:
            logging.exception(f"Failed to start supervisor agent {agent_id}: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to start supervisor agent: {str(e)}"
            )

    # Get team members
    members = await db.get_team_members(team_id, active_only=True)
    if not members:
        raise HTTPException(status_code=400, detail="Team has no active members")

    # Create supervisor
    try:
        supervisor = Supervisor(
            agents=[active_agents[agent_id]["agent"]],
            supervisor_provider=active_agents[agent_id]["config"]["provider"],
            supervisor_model_name=active_agents[agent_id]["config"]["model_name"],
            supervisor_system_prompt=supervisor_config.get("system_prompt")
            or "You are a supervisor that coordinates a team of specialized agents.",
        )

        # Add team members to supervisor
        for member in members:
            # Check if member agent is running
            member_agent_id = member["agent_id"]
            if member_agent_id not in active_agents:
                # Try to start the agent
                member_agent_config = await db.get_agent(member_agent_id)
                if not member_agent_config:
                    continue  # Skip this member

                try:
                    # Create and start the agent
                    member_agent_app = await create_agent(
                        provider=member_agent_config["provider"],
                        model_name=member_agent_config["model_name"],
                        agent_name=member_agent_config["name"],
                        system_prompt=member_agent_config.get("system_prompt"),
                        use_react_agent=member_agent_config.get(
                            "use_react_agent", True
                        ),
                        max_tokens_trimmed=member_agent_config.get(
                            "max_tokens_trimmed", 3000
                        ),
                        llm_specific_kwargs=member_agent_config.get(
                            "llm_specific_kwargs"
                        ),
                    )

                    # Add agent to active agents
                    active_agents[member_agent_id] = {
                        "agent": member_agent_app,
                        "config": member_agent_config,
                    }
                except Exception as e:
                    logging.warning(
                        f"Failed to start team member agent {member_agent_id}: {e}"
                    )
                    continue  # Skip this member

            # Add member to supervisor
            if hasattr(supervisor, "add_agent"):
                supervisor.add_agent(
                    agent=active_agents[member_agent_id]["agent"],
                    role=member["role"],
                    order=member.get("order_index"),
                )

        # Send message to supervisor
        thread_id = message.thread_id or f"team_{team_id}_thread_{uuid.uuid4().hex}"

        if hasattr(supervisor, "run"):
            response = await supervisor.run(message.content, thread_id=thread_id)
        else:
            raise HTTPException(
                status_code=500, detail="Supervisor does not have a run method"
            )

        return {"response": response, "thread_id": thread_id}
    except Exception as e:
        logging.exception(f"Error in team chat: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to process team chat: {str(e)}"
        )
