"""
Team-related API endpoints.
Provides functionality to manage teams, supervisors, and team members.
"""

import logging
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langgraph.types import Command
from pydantic import Field, create_model

from ..agents import Supervisor, _build_invoke_config, create_agent, evaluate_member_policy
from ..observability import langsmith_trace
from .api import active_agents, get_config_db
from .db import ConfigDB
from .helpers import build_agent_app, create_and_start_agent, extract_response_text
from .models import (
    RunningTeamsResponse,
    SupervisorCreate,
    SupervisorResponse,
    SupervisorUpdate,
    TeamMetricsEnvelope,
    TeamCreate,
    TeamMemberCreate,
    TeamMemberResponse,
    TeamMemberUpdate,
    TeamMessage,
    TeamResponse,
    TeamResponseMessage,
    TeamRuntimeActionResponse,
    TeamUpdate,
)

# Create router
router = APIRouter(prefix="/teams", tags=["teams"])

# Global state for active supervisors and teams
active_supervisors: dict[str, dict[str, Any]] = {}
active_teams: dict[str, dict[str, Any]] = {}


def _invalidate_team_runtime(team_id: str) -> None:
    active_teams.pop(team_id, None)


def _invalidate_teams_for_supervisor(supervisor_id: str) -> None:
    to_remove = [
        team_id
        for team_id, runtime in active_teams.items()
        if runtime["config"].get("supervisor_id") == supervisor_id
    ]
    for team_id in to_remove:
        _invalidate_team_runtime(team_id)


def _invalidate_teams_for_agent(agent_id: str) -> None:
    to_remove: list[str] = []
    for team_id, runtime in active_teams.items():
        supervisor_config = runtime.get("supervisor_config") or {}
        if supervisor_config.get("agent_id") == agent_id:
            to_remove.append(team_id)
            continue
        members = runtime.get("members") or []
        if any(member.get("agent_id") == agent_id for member in members):
            to_remove.append(team_id)
    for team_id in to_remove:
        _invalidate_team_runtime(team_id)


def _get_team_runtime_metrics(runtime: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(runtime.get("metrics") or {})
    supervisor = runtime.get("supervisor_instance")
    if supervisor is not None:
        metrics["supervisor"] = supervisor.get_metrics()
    return metrics


def _serialize_supervisor_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "delegations_total": int(metrics.get("delegations_total", 0)),
        "delegation_failures": int(metrics.get("delegation_failures", 0)),
        "parallel_delegations_total": int(metrics.get("parallel_delegations_total", 0)),
        "parallel_tasks_total": int(metrics.get("parallel_tasks_total", 0)),
        "per_agent": dict(metrics.get("per_agent") or {}),
        "recent_delegations": [
            {
                "route": event.get("delegated_via", "supervisor"),
                "event_type": "delegation",
                "agent": event.get("agent"),
                "agent_id": event.get("agent_id"),
                "role": event.get("role"),
                "status": event.get("status"),
                "thread_id": event.get("thread_id"),
                "latency_ms": event.get("latency_ms"),
                "detail": "Supervisor delegation",
            }
            for event in metrics.get("recent_delegations", [])
        ],
    }


def _serialize_team_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    serialized = {
        "starts": int(metrics.get("starts", 0)),
        "chats": int(metrics.get("chats", 0)),
        "direct_messages": int(metrics.get("direct_messages", 0)),
        "peer_messages": int(metrics.get("peer_messages", 0)),
        "last_thread_id": metrics.get("last_thread_id"),
    }
    supervisor_metrics = metrics.get("supervisor")
    serialized["supervisor"] = (
        _serialize_supervisor_metrics(supervisor_metrics) if supervisor_metrics else None
    )
    return serialized


def _append_runtime_trace(runtime: dict[str, Any], event: dict[str, Any]) -> None:
    trace_events = list(runtime.get("trace_events") or [])
    trace_events.append(event)
    runtime["trace_events"] = trace_events[-50:]


def _build_team_trace(runtime: dict[str, Any], response: Any) -> list[dict[str, Any]] | None:
    trace: list[dict[str, Any]] = list(runtime.get("trace_events") or [])
    supervisor = runtime.get("supervisor_instance")
    if supervisor is not None:
        for event in supervisor.get_metrics().get("recent_delegations", []):
            trace.append(
                {
                    "route": event.get("delegated_via", "supervisor"),
                    "event_type": "delegation",
                    "agent": event.get("agent"),
                    "agent_id": event.get("agent_id"),
                    "role": event.get("role"),
                    "status": event.get("status"),
                    "thread_id": event.get("thread_id"),
                    "latency_ms": event.get("latency_ms"),
                    "detail": "Supervisor delegation",
                }
            )
    if isinstance(response, dict) and response.get("messages"):
        for msg in response["messages"]:
            if isinstance(msg, (AIMessage, HumanMessage)):
                trace.append(
                    {
                        "route": "supervisor",
                        "event_type": msg.__class__.__name__,
                        "agent": getattr(msg, "name", None),
                        "content": str(msg.content),
                    }
                )
    return trace or None


def _find_member(runtime: dict[str, Any], agent_id: str) -> dict[str, Any] | None:
    for member in runtime.get("members") or []:
        if member.get("agent_id") == agent_id:
            return member
    return None


def _resolve_team_member(runtime: dict[str, Any], *, agent_ref: str) -> dict[str, Any] | None:
    for agent_id, team_agent in (runtime.get("team_agents") or {}).items():
        if agent_id == agent_ref or team_agent.get("name") == agent_ref:
            return team_agent
    return None


def _can_receive_direct(member: dict[str, Any]) -> bool:
    params = member.get("params") or {}
    return bool(params.get("can_receive_direct", True))


async def _validate_member_policy_refs(
    db: ConfigDB,
    *,
    team_id: str,
    member_role: str,
    member_agent_id: str,
    policy: dict[str, Any] | None,
) -> None:
    if not policy:
        return
    team = await db.get_team(team_id)
    team_config = team.get("config") if team else {}
    members = await db.get_team_members(team_id)
    known_roles = {member.get("role") for member in members if member.get("role")}
    known_roles.add(member_role)
    known_agent_ids = {member.get("agent_id") for member in members if member.get("agent_id")}
    known_agent_ids.add(member_agent_id)

    invalid_roles = sorted(
        role
        for key in ("can_message_roles", "accept_messages_from_roles")
        for role in (policy.get(key) or [])
        if role not in known_roles
    )
    if invalid_roles:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown team role references in policy: {', '.join(invalid_roles)}",
        )

    invalid_agent_ids = sorted(
        agent_id
        for agent_id in (policy.get("can_message_agents") or [])
        if agent_id not in known_agent_ids
    )
    if invalid_agent_ids:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unknown team agent references in policy: "
                + ", ".join(invalid_agent_ids)
            ),
        )

    if team_config and team_config.get("allow_direct_messages") is False and policy.get(
        "can_receive_direct", True
    ):
        raise HTTPException(
            status_code=400,
            detail="Team configuration disables direct messages",
        )

    if team_config and team_config.get("allow_peer_messages") is False and policy.get(
        "allow_peer_messages", False
    ):
        raise HTTPException(
            status_code=400,
            detail="Team configuration disables peer messages",
        )

    if team and team.get("supervisor_id"):
        effective_members = [
            member
            for member in members
            if member.get("is_active", True) and member.get("agent_id") != member_agent_id
        ]
        effective_members.append(
            {
                "agent_id": member_agent_id,
                "role": member_role,
                "is_active": True,
                "params": policy,
            }
        )
        if not any(
            (member.get("params") or {}).get("allow_supervisor_delegation", True)
            for member in effective_members
        ):
            raise HTTPException(
                status_code=400,
                detail="At least one active team member must allow supervisor delegation",
            )


def _peer_message_allowed(sender: dict[str, Any], target: dict[str, Any]) -> tuple[bool, str | None]:
    sender_params = sender.get("params") or {}
    target_params = target.get("params") or {}
    if not sender_params.get("allow_peer_messages", False):
        return False, "sender is not allowed to message peers"
    if not target_params.get("allow_peer_messages", True):
        return False, "target does not accept peer messages"

    allowed_roles = sender_params.get("can_message_roles")
    if allowed_roles and target.get("role") not in allowed_roles:
        return False, "target role not allowed"

    allowed_agents = sender_params.get("can_message_agents")
    if allowed_agents and target.get("agent_id") not in allowed_agents:
        return False, "target agent not allowed"

    accepted_roles = target_params.get("accept_messages_from_roles")
    if accepted_roles and sender.get("role") not in accepted_roles:
        return False, "target rejects sender role"
    return True, None


def _build_peer_tool(
    *,
    team_id: str,
    sender_member: dict[str, Any],
    runtime_agents: dict[str, dict[str, Any]],
    runtime_metrics: dict[str, Any],
) -> StructuredTool:
    input_model = create_model(
        f"{sender_member['agent_id']}_peer_message_input",
        agent_name=(str, Field(description="Target teammate name or agent ID")),
        message=(str, Field(description="Message or task for the teammate")),
    )

    async def message_teammate(
        agent_name: str,
        message: str,
        config: RunnableConfig | None = None,
    ) -> str:
        target = _resolve_team_member({"team_agents": runtime_agents}, agent_ref=agent_name)
        if target is None:
            return f"Teammate '{agent_name}' not found."
        if target["agent_id"] == sender_member["agent_id"]:
            return "Cannot message yourself."

        allowed, reason = _peer_message_allowed(sender_member, target)
        if not allowed:
            return f"Peer message denied ({reason})."

        parent_thread_id = None
        if config:
            parent_thread_id = (config.get("configurable", {}) or {}).get("thread_id")
        child_thread_id = (
            f"{parent_thread_id}:peer:{sender_member['agent_id']}:{target['agent_id']}"
            if parent_thread_id
            else f"peer:{team_id}:{uuid.uuid4().hex}"
        )
        try:
            response = await target["app"].ainvoke(
                {"messages": [{"role": "user", "content": message}]},
                config={"configurable": {"thread_id": child_thread_id}},
            )
            runtime_metrics["peer_messages"] = int(runtime_metrics.get("peer_messages", 0)) + 1
            runtime = active_teams.get(team_id)
            if runtime is not None:
                _append_runtime_trace(
                    runtime,
                    {
                        "route": "peer",
                        "event_type": "peer_message",
                        "agent": sender_member.get("name"),
                        "agent_id": sender_member["agent_id"],
                        "role": sender_member.get("role"),
                        "target_agent": target.get("name"),
                        "target_agent_id": target.get("agent_id"),
                        "target_role": target.get("role"),
                        "status": "ok",
                        "thread_id": child_thread_id,
                        "content": message,
                        "detail": "Direct teammate-to-teammate message",
                    },
                )
            content, _ = extract_response_text(response)
            return content
        except Exception as exc:
            logging.exception(
                "Peer message failed from %s to %s",
                sender_member["agent_id"],
                target["agent_id"],
            )
            return f"Peer message failed: {exc.__class__.__name__}: {exc}"

    return StructuredTool.from_function(
        coroutine=message_teammate,
        name="message_teammate",
        description=(
            "Send a message to another teammate when your role allows direct peer collaboration."
        ),
        args_schema=input_model,
    )

async def _start_team_runtime(
    team_id: str, db: ConfigDB
) -> dict[str, Any]:
    if team_id in active_teams:
        return active_teams[team_id]

    team_config = await db.get_team(team_id)
    if not team_config:
        raise HTTPException(status_code=404, detail=f"Team {team_id} not found")

    members = await db.get_team_members(team_id, active_only=True)
    if not members:
        raise HTTPException(
            status_code=400, detail=f"Team {team_id} has no active members"
        )

    try:
        runtime_metrics = {
            "starts": 1,
            "chats": 0,
            "direct_messages": 0,
            "peer_messages": 0,
            "last_thread_id": None,
        }
        member_configs: dict[str, dict[str, Any]] = {}
        for member in members:
            agent_id = member["agent_id"]
            agent_config = await db.get_agent(agent_id)
            if agent_config:
                member_configs[agent_id] = agent_config
            if agent_id not in active_agents:
                if agent_config:
                    await create_and_start_agent(db, agent_id, agent_config, active_agents)

        team_agents: dict[str, dict[str, Any]] = {}
        for member in members:
            agent_id = member["agent_id"]
            agent_config = member_configs.get(agent_id)
            if not agent_config:
                continue
            peer_tools = [
                _build_peer_tool(
                    team_id=team_id,
                    sender_member=member,
                    runtime_agents=team_agents,
                    runtime_metrics=runtime_metrics,
                )
            ]
            team_app = await build_agent_app(
                db,
                agent_id,
                agent_config,
                extra_tools=peer_tools,
            )
            team_agents[agent_id] = {
                "app": team_app,
                "agent_id": agent_id,
                "name": agent_config["name"],
                "role": member.get("role"),
                "params": member.get("params") or {},
            }

        supervisor_app = None
        supervisor_instance = None
        supervisor_id = team_config.get("supervisor_id")
        supervisor_config = None

        if supervisor_id:
            supervisor_config = await db.get_supervisor(supervisor_id)
            if supervisor_config:
                agent_apps = [
                    {
                        **team_agents[member["agent_id"]],
                        "params": member.get("params") or {},
                    }
                    for member in members
                    if member["agent_id"] in team_agents
                ]

                supervisor_agent_id = supervisor_config["agent_id"]
                if supervisor_agent_id not in active_agents:
                    supervisor_agent_config = await db.get_agent(supervisor_agent_id)
                    if not supervisor_agent_config:
                        raise HTTPException(
                            status_code=404,
                            detail=f"Supervisor agent {supervisor_agent_id} not found",
                        )

                    supervisor_agent_app = await create_agent(
                        provider=supervisor_agent_config["provider"],
                        model_name=supervisor_agent_config["model_name"],
                        agent_name=supervisor_agent_config["name"],
                        system_prompt=supervisor_agent_config.get("system_prompt"),
                        skills=supervisor_agent_config.get("skills"),
                        skill_paths=supervisor_agent_config.get("skill_paths"),
                    )
                    active_agents[supervisor_agent_id] = {
                        "agent": supervisor_agent_app,
                        "config": supervisor_agent_config,
                    }

                supervisor_params = {
                    "add_handoff_back_messages": supervisor_config.get(
                        "add_handoff_back_messages", True
                    ),
                    "parallel_tool_calls": supervisor_config.get(
                        "parallel_tool_calls", True
                    ),
                }
                config_value = supervisor_config.get("config")
                if isinstance(config_value, dict):
                    supervisor_params.update(config_value)

                supervisor_instance = Supervisor(
                    agents=agent_apps,
                    supervisor_provider=active_agents[supervisor_agent_id]["config"][
                        "provider"
                    ],
                    supervisor_model_name=active_agents[supervisor_agent_id]["config"][
                        "model_name"
                    ],
                    supervisor_system_prompt=supervisor_config.get("system_prompt")
                    or "You are a supervisor that coordinates a team of specialized agents.",
                    **supervisor_params,
                )
                supervisor_app = await supervisor_instance.init_supervisor()

        active_teams[team_id] = {
            "config": team_config,
            "members": members,
            "team_agents": team_agents,
            "supervisor": supervisor_app,
            "supervisor_instance": supervisor_instance,
            "supervisor_config": supervisor_config,
            "metrics": runtime_metrics,
            "trace_events": [],
        }
        return active_teams[team_id]
    except HTTPException:
        raise
    except Exception as e:
        logging.exception(f"Failed to start team {team_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start team: {str(e)}")


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
    _invalidate_team_runtime(team_id)

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


@router.get("/running", response_model=RunningTeamsResponse)
async def list_running_teams():
    """Lists all running teams"""
    return {
        "count": len(active_teams),
        "teams": [
            {
                "id": team_id,
                "name": info["config"]["name"],
                "supervisor_id": info["config"].get("supervisor_id"),
                "active": True,
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
        _invalidate_team_runtime(team_id)

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

    await _validate_member_policy_refs(
        db,
        team_id=team_id,
        member_role=member.role,
        member_agent_id=member.agent_id,
        policy=member.params.model_dump() if member.params else None,
    )

    await db.add_team_member(
        team_id=team_id,
        agent_id=member.agent_id,
        role=member.role,
        order_index=member.order_index,
        is_active=member.is_active,
        params=member.params.model_dump() if member.params else None,
    )
    _invalidate_team_runtime(team_id)

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
        current_member = next(m for m in members if m["agent_id"] == agent_id)
        next_role = update_data.get("role", current_member.get("role"))
        raw_policy = update_data.get("params")
        if raw_policy is not None:
            raw_policy = raw_policy.model_dump()
        else:
            raw_policy = current_member.get("params")
        await _validate_member_policy_refs(
            db,
            team_id=team_id,
            member_role=next_role,
            member_agent_id=agent_id,
            policy=raw_policy,
        )
        if "params" in update_data and update_data["params"] is not None:
            update_data["params"] = update_data["params"].model_dump()
        await db.update_team_member(team_id, agent_id, **update_data)
        _invalidate_team_runtime(team_id)

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
    _invalidate_team_runtime(team_id)
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
    _invalidate_teams_for_supervisor(supervisor_id)

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
        _invalidate_teams_for_supervisor(supervisor_id)

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

    _invalidate_teams_for_supervisor(supervisor_id)
    await db.delete_supervisor(supervisor_id)
    return None


# Team runtime endpoints
@router.post("/{team_id}/start", response_model=TeamRuntimeActionResponse, status_code=200)
async def start_team(team_id: str, db: ConfigDB = Depends(get_config_db)):
    """Starts a team with its supervisor and agents"""
    if team_id in active_teams:
        # Team is already running
        return {"status": "already_running", "team_id": team_id}
    await _start_team_runtime(team_id, db)
    return {"status": "started", "team_id": team_id}


@router.post("/{team_id}/stop", response_model=TeamRuntimeActionResponse, status_code=200)
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


@router.get("/{team_id}/metrics", response_model=TeamMetricsEnvelope, status_code=200)
async def get_team_metrics(team_id: str, db: ConfigDB = Depends(get_config_db)):
    """Returns runtime metrics for a team if it is active."""
    team = await db.get_team(team_id)
    if not team:
        raise HTTPException(status_code=404, detail=f"Team {team_id} not found")
    runtime = active_teams.get(team_id)
    if runtime is None:
        return {
            "team_id": team_id,
            "active": False,
            "metrics": None,
        }
    return {
        "team_id": team_id,
        "active": True,
        "metrics": _serialize_team_metrics(_get_team_runtime_metrics(runtime)),
    }


@router.post("/{team_id}/chat", response_model=TeamResponseMessage)
async def chat_with_team(
    team_id: str,
    message: TeamMessage,
    request: Request,
    db: ConfigDB = Depends(get_config_db),
):
    """Sends a message to a team"""
    # Validate team
    team = await db.get_team(team_id)
    if not team:
        raise HTTPException(status_code=404, detail=f"Team {team_id} not found")

    # Check if team has a supervisor
    if not team["supervisor_id"]:
        raise HTTPException(status_code=400, detail="Team does not have a supervisor")

    try:
        runtime = await _start_team_runtime(team_id, db)
        supervisor_app = runtime.get("supervisor")
        if supervisor_app is None:
            raise HTTPException(status_code=400, detail="Team supervisor is not running")

        thread_id = message.thread_id or f"team_{team_id}_thread_{uuid.uuid4().hex}"
        if message.direct_to_agent_id:
            target_member = _find_member(runtime, message.direct_to_agent_id)
            if target_member is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Agent {message.direct_to_agent_id} is not a member of team {team_id}",
                )
            if not _can_receive_direct(target_member):
                raise HTTPException(
                    status_code=403,
                    detail=f"Agent {message.direct_to_agent_id} does not allow direct messages",
                )
            team_agent = runtime.get("team_agents", {}).get(message.direct_to_agent_id)
            if team_agent is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Direct target {message.direct_to_agent_id} is not running",
                )
            allowed, reason = evaluate_member_policy(target_member, message.content)
            if not allowed:
                raise HTTPException(
                    status_code=403,
                    detail=f"Direct routing denied by policy ({reason})",
                )
            direct_config = _build_invoke_config(
                thread_id=thread_id,
                run_name="team_direct_chat",
                tags=["mao", "team", team_id, "direct"],
                metadata={
                    "team_id": team_id,
                    "team_name": team["name"],
                    "target_agent_id": message.direct_to_agent_id,
                    "http_path": request.url.path,
                },
            )
            with langsmith_trace(
                run_name="team_direct_chat",
                tags=direct_config.get("tags"),
                metadata=direct_config.get("metadata"),
            ):
                response = await team_agent["app"].ainvoke(
                    {
                        "messages": [HumanMessage(content=message.content)],
                        "response_schema": message.response_schema,
                    },
                    config=direct_config,
                )
            runtime_metrics = runtime.setdefault("metrics", {})
            runtime_metrics["chats"] = int(runtime_metrics.get("chats", 0)) + 1
            runtime_metrics["direct_messages"] = int(runtime_metrics.get("direct_messages", 0)) + 1
            runtime_metrics["last_thread_id"] = thread_id
            _append_runtime_trace(
                runtime,
                {
                    "route": "direct",
                    "event_type": "direct_message",
                    "agent": team_agent["name"],
                    "agent_id": team_agent["agent_id"],
                    "role": team_agent.get("role"),
                    "status": "ok",
                    "thread_id": thread_id,
                    "content": message.content,
                    "detail": "Direct user-to-member routing",
                },
            )
            response_text, responding_agent_id = extract_response_text(response)
            return {
                "response": response_text,
                "thread_id": thread_id,
                "responding_agent_id": responding_agent_id,
                "trace": _build_team_trace(runtime, response),
                "details": {
                    **(response if isinstance(response, dict) else {"raw_response": str(response)}),
                    "team_metrics": _serialize_team_metrics(_get_team_runtime_metrics(runtime)),
                },
            }
        config = _build_invoke_config(
            thread_id=thread_id,
            run_name="team_chat",
            tags=["mao", "team", team_id],
            metadata={
                "team_id": team_id,
                "team_name": team["name"],
                "http_path": request.url.path,
            },
        )
        with langsmith_trace(
            run_name="team_chat",
            tags=config.get("tags"),
            metadata=config.get("metadata"),
        ):
            if message.approval_decisions:
                response = await supervisor_app.ainvoke(
                    Command(resume={"decisions": message.approval_decisions}),
                    config=config,
                )
            else:
                response = await supervisor_app.ainvoke(
                    {
                        "messages": [HumanMessage(content=message.content)],
                        "response_schema": message.response_schema,
                    },
                    config=config,
                )
        runtime_metrics = runtime.setdefault("metrics", {})
        runtime_metrics["chats"] = int(runtime_metrics.get("chats", 0)) + 1
        runtime_metrics["last_thread_id"] = thread_id
        _append_runtime_trace(
            runtime,
            {
                "route": "supervisor",
                "event_type": "team_chat",
                "status": "ok",
                "thread_id": thread_id,
                "content": message.content,
                "detail": "Team chat routed through supervisor",
            },
        )
        response_text, responding_agent_id = extract_response_text(response)

        return {
            "response": response_text,
            "thread_id": thread_id,
            "responding_agent_id": responding_agent_id,
            "trace": _build_team_trace(runtime, response),
            "details": (
                {
                    **(response if isinstance(response, dict) else {"raw_response": str(response)}),
                    "team_metrics": _serialize_team_metrics(_get_team_runtime_metrics(runtime)),
                }
            ),
        }
    except HTTPException:
        raise
    except Exception as e:
        logging.exception(f"Error in team chat: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to process team chat: {str(e)}"
        )
