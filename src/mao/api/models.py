"""
Shared Pydantic models for API endpoints.
"""

from datetime import datetime
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class AgentCreate(BaseModel):
    name: str = Field(..., description="Display name for the agent")
    provider: str = Field(..., description="LLM provider (openai, anthropic, etc.)")
    model_name: str = Field(..., description="Model name to use")
    system_prompt: str | None = Field(None, description="System prompt for the agent")
    skills: list[str] | None = Field(
        None, description="Optional explicitly enabled skill names"
    )
    skill_paths: list[str] | None = Field(
        None, description="Optional additional skill root directories"
    )


class AgentUpdate(BaseModel):
    name: str | None = Field(None, description="Display name for the agent")
    provider: str | None = Field(None, description="LLM provider (openai, anthropic, etc.)")
    model_name: str | None = Field(None, description="Model name to use")
    system_prompt: str | None = Field(None, description="System prompt for the agent")
    skills: list[str] | None = Field(
        None, description="Optional explicitly enabled skill names"
    )
    skill_paths: list[str] | None = Field(
        None, description="Optional additional skill root directories"
    )


class AgentResponse(BaseModel):
    id: str
    name: str
    provider: str
    model_name: str
    system_prompt: str | None = None
    skills: list[str] | None = None
    skill_paths: list[str] | None = None
    created_at: str | datetime
    updated_at: str | datetime


class PaginatedResponse(BaseModel, Generic[T]):
    items: list[T]
    total: int
    limit: int
    offset: int

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"items": [], "total": 100, "limit": 10, "offset": 0}
        }
    )


class TeamCreate(BaseModel):
    name: str = Field(..., description="Team name")
    description: str | None = Field(None, description="Team description")
    workflow_type: str = Field(
        "sequential", description="Workflow type: sequential, parallel, or custom"
    )
    supervisor_id: str | None = Field(None, description="ID of supervisor agent (if any)")
    config: dict[str, Any] | None = Field(None, description="Team configuration parameters")
    is_active: bool = Field(True, description="Whether the team is active")


class TeamUpdate(BaseModel):
    name: str | None = Field(None, description="Team name")
    description: str | None = Field(None, description="Team description")
    workflow_type: str | None = Field(
        None, description="Workflow type: sequential, parallel, or custom"
    )
    supervisor_id: str | None = Field(None, description="ID of supervisor agent")
    config: dict[str, Any] | None = Field(None, description="Team configuration parameters")
    is_active: bool | None = Field(None, description="Whether the team is active")


class TeamResponse(BaseModel):
    id: str
    name: str
    description: str | None = None
    workflow_type: str
    supervisor_id: str | None = None
    config: dict[str, Any] | None = None
    is_active: bool
    created_at: str | datetime
    updated_at: str | datetime


class TeamMemberPolicy(BaseModel):
    allow_supervisor_delegation: bool = Field(
        True, description="Whether the supervisor may delegate work to this member"
    )
    routing_keywords: list[str] | None = Field(
        None,
        description="Optional keywords that must match before supervisor routing is allowed",
    )
    blocked_keywords: list[str] | None = Field(
        None,
        description="Optional keywords that block supervisor or direct routing to this member",
    )
    can_receive_direct: bool = Field(
        True, description="Whether API callers may address this member directly"
    )
    allow_peer_messages: bool = Field(
        False, description="Whether this member may message teammates directly"
    )
    can_message_roles: list[str] | None = Field(
        None,
        description="Optional teammate roles this member may message directly",
    )
    can_message_agents: list[str] | None = Field(
        None,
        description="Optional teammate agent IDs this member may message directly",
    )
    accept_messages_from_roles: list[str] | None = Field(
        None,
        description="Optional sender roles accepted for direct peer messages",
    )


class TeamMemberCreate(BaseModel):
    agent_id: str = Field(..., description="ID of the agent to add to team")
    role: str = Field(..., description="Agent's role in the team")
    order_index: int | None = Field(None, description="Order in sequential workflows")
    is_active: bool = Field(True, description="Whether this agent is active in the team")
    params: TeamMemberPolicy | None = Field(
        None,
        description="Typed routing and communication policy for this team member",
    )


class TeamMemberUpdate(BaseModel):
    role: str | None = Field(None, description="Agent's role in the team")
    order_index: int | None = Field(None, description="Order in sequential workflows")
    is_active: bool | None = Field(None, description="Whether this agent is active in the team")
    params: TeamMemberPolicy | None = Field(
        None,
        description="Typed routing and communication policy for this team member",
    )


class TeamMemberResponse(BaseModel):
    team_id: str
    agent_id: str
    role: str
    order_index: int | None = None
    is_active: bool
    params: TeamMemberPolicy | None = None
    created_at: str | datetime
    updated_at: str | datetime | None = None


class TeamTraceEvent(BaseModel):
    route: str = Field(description="Routing mode such as supervisor, direct, or peer")
    event_type: str = Field(description="Event type such as delegation, peer_message, or final")
    agent: str | None = Field(None, description="Actor agent name")
    agent_id: str | None = Field(None, description="Actor agent ID")
    role: str | None = Field(None, description="Actor team role")
    target_agent: str | None = Field(None, description="Target teammate name")
    target_agent_id: str | None = Field(None, description="Target teammate ID")
    target_role: str | None = Field(None, description="Target teammate role")
    status: str | None = Field(None, description="Event status")
    thread_id: str | None = Field(None, description="Related thread identifier")
    content: str | None = Field(None, description="Relevant message content")
    detail: str | None = Field(None, description="Additional trace detail")
    latency_ms: float | None = Field(None, description="Observed latency in milliseconds")


class AgentDelegationStats(BaseModel):
    agent_id: str | None = None
    delegations: int
    failures: int
    last_latency_ms: float | None = None


class SupervisorMetricsResponse(BaseModel):
    delegations_total: int
    delegation_failures: int
    parallel_delegations_total: int
    parallel_tasks_total: int
    per_agent: dict[str, AgentDelegationStats]
    recent_delegations: list[TeamTraceEvent]


class TeamRuntimeMetricsResponse(BaseModel):
    starts: int
    chats: int
    direct_messages: int
    peer_messages: int
    last_thread_id: str | None = None
    supervisor: SupervisorMetricsResponse | None = None


class TeamMetricsEnvelope(BaseModel):
    team_id: str
    active: bool
    metrics: TeamRuntimeMetricsResponse | None = None


class RunningTeamStatus(BaseModel):
    id: str
    name: str
    supervisor_id: str | None = None
    active: bool = True


class RunningTeamsResponse(BaseModel):
    count: int
    teams: list[RunningTeamStatus]


class TeamRuntimeActionResponse(BaseModel):
    status: str
    team_id: str


class SupervisorCreate(BaseModel):
    agent_id: str = Field(..., description="ID of the agent to use as supervisor")
    system_prompt: str | None = Field(
        None, description="System prompt override for supervisor"
    )
    strategy: str = Field(
        "team_manager",
        description="Supervisor strategy: team_manager, orchestrator, or custom",
    )
    add_handoff_back_messages: bool = Field(
        True, description="Whether to add handoff back messages"
    )
    parallel_tool_calls: bool = Field(
        True, description="Whether to allow parallel tool calls"
    )
    config: dict[str, Any] | None = Field(
        None, description="Additional supervisor configuration"
    )


class SupervisorUpdate(BaseModel):
    agent_id: str | None = Field(None, description="ID of the agent to use as supervisor")
    system_prompt: str | None = Field(
        None, description="System prompt override for supervisor"
    )
    strategy: str | None = Field(
        None, description="Supervisor strategy: team_manager, orchestrator, or custom"
    )
    add_handoff_back_messages: bool | None = Field(
        None, description="Whether to add handoff back messages"
    )
    parallel_tool_calls: bool | None = Field(
        None, description="Whether to allow parallel tool calls"
    )
    config: dict[str, Any] | None = Field(
        None, description="Additional supervisor configuration"
    )


class SupervisorResponse(BaseModel):
    id: str
    agent_id: str
    system_prompt: str | None = None
    strategy: str
    add_handoff_back_messages: bool
    parallel_tool_calls: bool
    config: dict[str, Any] | None = None
    created_at: str | datetime
    updated_at: str | datetime


class ServerCreate(BaseModel):
    name: str = Field(..., description="Display name for the server")
    transport: str = Field(..., description="Transport type (stdio, sse, websocket, etc.)")
    enabled: bool = Field(True, description="Whether the server is enabled")
    url: str | None = Field(None, description="Server URL (for sse, websocket)")
    command: str | None = Field(None, description="Command to run (for stdio)")
    args: list[str] | None = Field(None, description="Command arguments (for stdio)")
    headers: dict[str, str] | None = Field(
        None, description="HTTP headers (for sse, websocket)"
    )
    env_vars: dict[str, str] | None = Field(
        None, description="Environment variables (for stdio)"
    )
    timeout: int | None = Field(None, description="Connection timeout")


class ServerUpdate(BaseModel):
    name: str | None = Field(None, description="Display name for the server")
    transport: str | None = Field(
        None, description="Transport type (stdio, sse, websocket, etc.)"
    )
    enabled: bool | None = Field(None, description="Whether the server is enabled")
    url: str | None = Field(None, description="Server URL (for sse, websocket)")
    command: str | None = Field(None, description="Command to run (for stdio)")
    args: list[str] | None = Field(None, description="Command arguments (for stdio)")
    headers: dict[str, str] | None = Field(
        None, description="HTTP headers (for sse, websocket)"
    )
    env_vars: dict[str, str] | None = Field(
        None, description="Environment variables (for stdio)"
    )
    timeout: int | None = Field(None, description="Connection timeout")


class ServerResponse(BaseModel):
    id: str
    name: str
    transport: str
    enabled: bool
    url: str | None = None
    command: str | None = None
    args: list[str] | None = None
    headers: dict[str, str] | None = None
    env_vars: dict[str, str] | None = None
    timeout: int | None = None
    created_at: str | datetime
    updated_at: str | datetime


class ToolCreate(BaseModel):
    name: str = Field(..., description="Display name for the tool")
    enabled: bool = Field(True, description="Whether the tool is enabled by default")
    server_id: str | None = Field(None, description="ID of the server providing this tool")
    description: str | None = Field(None, description="Tool description")
    parameters: dict[str, Any] | None = Field(None, description="Tool parameters schema")


class ToolUpdate(BaseModel):
    name: str | None = Field(None, description="Display name for the tool")
    enabled: bool | None = Field(None, description="Whether the tool is enabled by default")
    server_id: str | None = Field(None, description="ID of the server providing this tool")
    description: str | None = Field(None, description="Tool description")
    parameters: dict[str, Any] | None = Field(None, description="Tool parameters schema")


class ToolResponse(BaseModel):
    id: str
    name: str
    enabled: bool
    server_id: str | None = None
    description: str | None = None
    parameters: dict[str, Any] | None = None
    created_at: str | datetime
    updated_at: str | datetime


class AssignToolRequest(BaseModel):
    enabled: bool = Field(True, description="Whether the tool is enabled for this agent")


class Config(BaseModel):
    key: str
    value: Any
    description: str | None = None


class AgentMessage(BaseModel):
    content: str = Field(..., description="Message content")
    thread_id: str | None = Field(None, description="Thread ID for conversation tracking")
    response_schema: dict[str, Any] | None = Field(
        None, description="Optional JSON schema for structured output"
    )
    approval_decisions: list[dict[str, Any]] | None = Field(
        None, description="Optional HITL approval/edit/reject decisions to resume execution"
    )


class AgentResponseMessage(BaseModel):
    response: str = Field(..., description="Agent response")
    thread_id: str | None = Field(None, description="Thread ID for conversation tracking")
    details: dict[str, Any] | None = Field(None, description="Additional response details")


class TeamMessage(BaseModel):
    content: str = Field(..., description="Message content")
    thread_id: str | None = Field(None, description="Thread ID for conversation tracking")
    direct_to_agent_id: str | None = Field(
        None, description="Optional agent ID to send directly to"
    )
    input_file_ids: list[str] | None = Field(None, description="List of file IDs to include")
    metadata: dict[str, Any] | None = Field(
        None, description="Additional metadata for the message"
    )
    response_schema: dict[str, Any] | None = Field(
        None, description="Optional JSON schema for structured output"
    )
    approval_decisions: list[dict[str, Any]] | None = Field(
        None, description="Optional HITL approval/edit/reject decisions to resume execution"
    )


class TeamResponseMessage(BaseModel):
    response: str = Field(..., description="Team response")
    thread_id: str | None = Field(None, description="Thread ID for conversation tracking")
    responding_agent_id: str | None = Field(
        None, description="ID of the agent who provided the final response"
    )
    trace: list[TeamTraceEvent] | None = Field(
        None, description="Trace of internal team communication"
    )
    output_file_ids: list[str] | None = Field(
        None, description="List of file IDs produced by the team"
    )
    details: dict[str, Any] | None = Field(None, description="Additional response details")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "response": "We've analyzed your request and prepared the report.",
                "thread_id": "thread_abc123",
                "responding_agent_id": "agent_writer_1",
                "trace": [
                    {
                        "route": "supervisor",
                        "event_type": "delegation",
                        "agent": "agent_researcher_1",
                        "role": "researcher",
                        "status": "ok",
                        "detail": "Found relevant information",
                    },
                    {
                        "route": "peer",
                        "event_type": "peer_message",
                        "agent": "agent_writer_1",
                        "target_agent": "agent_researcher_1",
                        "status": "ok",
                        "detail": "Requested clarifications",
                    },
                ],
            }
        }
    )
