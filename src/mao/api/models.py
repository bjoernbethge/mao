"""
Shared Pydantic models for API endpoints.
"""

from typing import List, Dict, Any, Optional, Union, Generic, TypeVar
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict

# Generic type for paginated responses
T = TypeVar('T')

# Agent Models
class AgentCreate(BaseModel):
    """Model for creating an agent"""
    name: str = Field(..., description="Display name for the agent")
    provider: str = Field(..., description="LLM provider (openai, anthropic, etc.)")
    model_name: str = Field(..., description="Model name to use")
    system_prompt: Optional[str] = Field(None, description="System prompt for the agent")
    use_react_agent: bool = Field(True, description="Whether to use ReAct agent")
    max_tokens_trimmed: int = Field(3000, description="Maximum tokens to keep in context")
    llm_specific_kwargs: Optional[Dict[str, Any]] = Field(None, description="Provider-specific arguments")

class AgentUpdate(BaseModel):
    """Model for updating an agent"""
    name: Optional[str] = Field(None, description="Display name for the agent")
    provider: Optional[str] = Field(None, description="LLM provider (openai, anthropic, etc.)")
    model_name: Optional[str] = Field(None, description="Model name to use")
    system_prompt: Optional[str] = Field(None, description="System prompt for the agent")
    use_react_agent: Optional[bool] = Field(None, description="Whether to use ReAct agent")
    max_tokens_trimmed: Optional[int] = Field(None, description="Maximum tokens to keep in context")
    llm_specific_kwargs: Optional[Dict[str, Any]] = Field(None, description="Provider-specific arguments")

class AgentResponse(BaseModel):
    """Model for agent response"""
    id: str
    name: str
    provider: str
    model_name: str
    system_prompt: Optional[str] = None
    use_react_agent: bool
    max_tokens_trimmed: int
    llm_specific_kwargs: Optional[Dict[str, Any]] = None
    created_at: Union[str, datetime]
    updated_at: Union[str, datetime]

# Pagination model
class PaginatedResponse(BaseModel, Generic[T]):
    """
    Generic model for paginated responses.
    
    Attributes:
        items: List of items in the current page
        total: Total number of items across all pages
        limit: Maximum number of items per page
        offset: Current offset (number of items skipped)
    """
    items: List[T]
    total: int
    limit: int
    offset: int
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "items": [],
                "total": 100,
                "limit": 10,
                "offset": 0
            }
        }
    )

# Team Models
class TeamCreate(BaseModel):
    """Model for creating a team of agents"""
    name: str = Field(..., description="Team name")
    description: Optional[str] = Field(None, description="Team description")
    workflow_type: str = Field("sequential", description="Workflow type: sequential, parallel, or custom")
    supervisor_id: Optional[str] = Field(None, description="ID of supervisor agent (if any)")
    config: Optional[Dict[str, Any]] = Field(None, description="Team configuration parameters")
    is_active: bool = Field(True, description="Whether the team is active")

class TeamUpdate(BaseModel):
    """Model for updating a team"""
    name: Optional[str] = Field(None, description="Team name")
    description: Optional[str] = Field(None, description="Team description")
    workflow_type: Optional[str] = Field(None, description="Workflow type: sequential, parallel, or custom")
    supervisor_id: Optional[str] = Field(None, description="ID of supervisor agent")
    config: Optional[Dict[str, Any]] = Field(None, description="Team configuration parameters")
    is_active: Optional[bool] = Field(None, description="Whether the team is active")

class TeamResponse(BaseModel):
    """Model for team response"""
    id: str
    name: str
    description: Optional[str] = None
    workflow_type: str
    supervisor_id: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    is_active: bool
    created_at: Union[str, datetime]
    updated_at: Union[str, datetime]

# Team Member Models
class TeamMemberCreate(BaseModel):
    """Model for adding an agent to a team"""
    agent_id: str = Field(..., description="ID of the agent to add to team")
    role: str = Field(..., description="Agent's role in the team")
    order_index: Optional[int] = Field(None, description="Order in sequential workflows")
    is_active: bool = Field(True, description="Whether this agent is active in the team")
    params: Optional[Dict[str, Any]] = Field(None, description="Role-specific parameters")

class TeamMemberUpdate(BaseModel):
    """Model for updating a team member"""
    role: Optional[str] = Field(None, description="Agent's role in the team")
    order_index: Optional[int] = Field(None, description="Order in sequential workflows")
    is_active: Optional[bool] = Field(None, description="Whether this agent is active in the team")
    params: Optional[Dict[str, Any]] = Field(None, description="Role-specific parameters")

class TeamMemberResponse(BaseModel):
    """Model for team member response"""
    team_id: str
    agent_id: str
    role: str
    order_index: Optional[int] = None
    is_active: bool
    params: Optional[Dict[str, Any]] = None
    created_at: Union[str, datetime]
    updated_at: Optional[Union[str, datetime]] = None
    
    model_config = ConfigDict(
        json_encoders={
            dict: lambda v: v
        }
    )

# Supervisor Models
class SupervisorCreate(BaseModel):
    """Model for creating a supervisor"""
    agent_id: str = Field(..., description="ID of the agent to use as supervisor")
    system_prompt: Optional[str] = Field(None, description="System prompt override for supervisor")
    strategy: str = Field("team_manager", description="Supervisor strategy: team_manager, orchestrator, or custom")
    add_handoff_back_messages: bool = Field(True, description="Whether to add handoff back messages")
    parallel_tool_calls: bool = Field(True, description="Whether to allow parallel tool calls")
    config: Optional[Dict[str, Any]] = Field(None, description="Additional supervisor configuration")

class SupervisorUpdate(BaseModel):
    """Model for updating a supervisor"""
    agent_id: Optional[str] = Field(None, description="ID of the agent to use as supervisor")
    system_prompt: Optional[str] = Field(None, description="System prompt override for supervisor")
    strategy: Optional[str] = Field(None, description="Supervisor strategy: team_manager, orchestrator, or custom")
    add_handoff_back_messages: Optional[bool] = Field(None, description="Whether to add handoff back messages")
    parallel_tool_calls: Optional[bool] = Field(None, description="Whether to allow parallel tool calls")
    config: Optional[Dict[str, Any]] = Field(None, description="Additional supervisor configuration")

class SupervisorResponse(BaseModel):
    """Model for supervisor response"""
    id: str
    agent_id: str
    system_prompt: Optional[str] = None
    strategy: str
    add_handoff_back_messages: bool
    parallel_tool_calls: bool
    config: Optional[Dict[str, Any]] = None
    created_at: Union[str, datetime]
    updated_at: Union[str, datetime]
    
    model_config = ConfigDict(
        json_encoders={
            dict: lambda v: v
        }
    )

# Server Models
class ServerCreate(BaseModel):
    """Model for creating a server"""
    name: str = Field(..., description="Display name for the server")
    transport: str = Field(..., description="Transport type (stdio, sse, websocket, etc.)")
    enabled: bool = Field(True, description="Whether the server is enabled")
    url: Optional[str] = Field(None, description="Server URL (for sse, websocket)")
    command: Optional[str] = Field(None, description="Command to run (for stdio)")
    args: Optional[List[str]] = Field(None, description="Command arguments (for stdio)")
    headers: Optional[Dict[str, str]] = Field(None, description="HTTP headers (for sse, websocket)")
    env_vars: Optional[Dict[str, str]] = Field(None, description="Environment variables (for stdio)")
    timeout: Optional[int] = Field(None, description="Connection timeout")

class ServerUpdate(BaseModel):
    """Model for updating a server"""
    name: Optional[str] = Field(None, description="Display name for the server")
    transport: Optional[str] = Field(None, description="Transport type (stdio, sse, websocket, etc.)")
    enabled: Optional[bool] = Field(None, description="Whether the server is enabled")
    url: Optional[str] = Field(None, description="Server URL (for sse, websocket)")
    command: Optional[str] = Field(None, description="Command to run (for stdio)")
    args: Optional[List[str]] = Field(None, description="Command arguments (for stdio)")
    headers: Optional[Dict[str, str]] = Field(None, description="HTTP headers (for sse, websocket)")
    env_vars: Optional[Dict[str, str]] = Field(None, description="Environment variables (for stdio)")
    timeout: Optional[int] = Field(None, description="Connection timeout")

class ServerResponse(BaseModel):
    """Model for server response"""
    id: str
    name: str
    transport: str
    enabled: bool
    url: Optional[str] = None
    command: Optional[str] = None
    args: Optional[List[str]] = None
    headers: Optional[Dict[str, str]] = None
    env_vars: Optional[Dict[str, str]] = None
    timeout: Optional[int] = None
    created_at: Union[str, datetime]
    updated_at: Union[str, datetime]

# Tool Models
class ToolCreate(BaseModel):
    """Model for creating a tool"""
    name: str = Field(..., description="Display name for the tool")
    enabled: bool = Field(True, description="Whether the tool is enabled by default")
    server_id: Optional[str] = Field(None, description="ID of the server providing this tool")
    description: Optional[str] = Field(None, description="Tool description")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Tool parameters schema")

class ToolUpdate(BaseModel):
    """Model for updating a tool"""
    name: Optional[str] = Field(None, description="Display name for the tool")
    enabled: Optional[bool] = Field(None, description="Whether the tool is enabled by default")
    server_id: Optional[str] = Field(None, description="ID of the server providing this tool")
    description: Optional[str] = Field(None, description="Tool description")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Tool parameters schema")

class ToolResponse(BaseModel):
    """Model for tool response"""
    id: str
    name: str
    enabled: bool
    server_id: Optional[str] = None
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    created_at: Union[str, datetime]
    updated_at: Union[str, datetime]

class AssignToolRequest(BaseModel):
    """Model for assigning a tool to an agent"""
    enabled: bool = Field(True, description="Whether the tool is enabled for this agent")

# Config Models
class Config(BaseModel):
    """Model for configuration values"""
    key: str
    value: Any
    description: Optional[str] = None

# Agent Interaction Models
class AgentMessage(BaseModel):
    """Model for messages to send to agents"""
    content: str = Field(..., description="Message content")
    thread_id: Optional[str] = Field(None, description="Thread ID for conversation tracking")

class AgentResponseMessage(BaseModel):
    """Model for responses from agents"""
    response: str = Field(..., description="Agent response")
    thread_id: Optional[str] = Field(None, description="Thread ID for conversation tracking")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional response details")

# Team Interaction Models
class TeamMessage(BaseModel):
    """Model for messages to send to teams"""
    content: str = Field(..., description="Message content")
    thread_id: Optional[str] = Field(None, description="Thread ID for conversation tracking")
    direct_to_agent_id: Optional[str] = Field(None, description="Optional agent ID to send directly to")
    input_file_ids: Optional[List[str]] = Field(None, description="List of file IDs to include")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata for the message")

class TeamResponseMessage(BaseModel):
    """Model for responses from teams"""
    response: str = Field(..., description="Team response")
    thread_id: Optional[str] = Field(None, description="Thread ID for conversation tracking")
    responding_agent_id: Optional[str] = Field(None, description="ID of the agent who provided the final response")
    trace: Optional[List[Dict[str, Any]]] = Field(None, description="Trace of internal team communication")
    output_file_ids: Optional[List[str]] = Field(None, description="List of file IDs produced by the team")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional response details")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "response": "We've analyzed your request and prepared the report.",
                "thread_id": "thread_abc123",
                "responding_agent_id": "agent_writer_1",
                "trace": [
                    {"agent": "agent_researcher_1", "action": "research", "result": "Found relevant information"},
                    {"agent": "agent_writer_1", "action": "write", "result": "Created report"}
                ]
            }
        }
    ) 