"""
MCP-related API endpoints (Servers and Tools).
"""

import json
import os
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from .api import get_config_db
from .db import ConfigDB
from .models import (
    AssignToolRequest,
    ServerCreate,
    ServerResponse,
    ServerUpdate,
    ToolCreate,
    ToolResponse,
    ToolUpdate,
)


router = APIRouter(prefix="/mcp", tags=["mcp"])


@router.post("/servers", response_model=ServerResponse, status_code=201)
async def create_new_server(
    server: ServerCreate, db: ConfigDB = Depends(get_config_db)
):
    """Creates a new MCP server"""
    server_id = f"server_{uuid.uuid4().hex[:8]}"

    await db.create_server(
        server_id=server_id,
        name=server.name,
        transport=server.transport,
        enabled=server.enabled,
        url=server.url,
        command=server.command,
        args=server.args,
        headers=server.headers,
        env_vars=server.env_vars,
        timeout=server.timeout,
    )

    return await db.get_server(server_id)


@router.get("/servers", response_model=list[ServerResponse])
async def list_all_servers(db: ConfigDB = Depends(get_config_db)):
    """Lists all configured servers"""
    return await db.list_servers()


@router.get("/servers/{server_id}", response_model=ServerResponse)
async def get_server_by_id(server_id: str, db: ConfigDB = Depends(get_config_db)):
    """Gets a server by its ID"""
    server = await db.get_server(server_id)
    if not server:
        raise HTTPException(status_code=404, detail=f"Server {server_id} not found")
    return server


@router.put("/servers/{server_id}", response_model=ServerResponse)
async def update_server_by_id(
    server_id: str, server: ServerUpdate, db: ConfigDB = Depends(get_config_db)
):
    """Updates a server"""
    existing = await db.get_server(server_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Server {server_id} not found")

    update_data = {k: v for k, v in server.model_dump().items() if v is not None}
    if update_data:
        await db.update_server(server_id, **update_data)

    return await db.get_server(server_id)


@router.delete("/servers/{server_id}", status_code=204)
async def delete_server_by_id(server_id: str, db: ConfigDB = Depends(get_config_db)):
    """Deletes a server"""
    existing = await db.get_server(server_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Server {server_id} not found")

    await db.delete_server(server_id)
    return None


# Tool endpoints
@router.post("/tools", response_model=ToolResponse, status_code=201)
async def create_new_tool(tool: ToolCreate, db: ConfigDB = Depends(get_config_db)):
    """Creates a new tool"""
    tool_id = f"tool_{uuid.uuid4().hex[:8]}"

    if tool.server_id:
        server = await db.get_server(tool.server_id)
        if not server:
            raise HTTPException(
                status_code=404, detail=f"Server {tool.server_id} not found"
            )

    await db.create_tool(
        tool_id=tool_id,
        name=tool.name,
        enabled=tool.enabled,
        server_id=tool.server_id,
        description=tool.description,
        parameters=tool.parameters,
    )

    return await db.get_tool(tool_id)


@router.get("/tools", response_model=list[ToolResponse])
async def list_all_tools(
    server_id: str | None = None, db: ConfigDB = Depends(get_config_db)
):
    """Lists all configured tools, optionally filtered by server"""
    return await db.list_tools(server_id=server_id)


@router.get("/tools/{tool_id}", response_model=ToolResponse)
async def get_tool_by_id(tool_id: str, db: ConfigDB = Depends(get_config_db)):
    """Gets a tool by its ID"""
    tool = await db.get_tool(tool_id)
    if not tool:
        raise HTTPException(status_code=404, detail=f"Tool {tool_id} not found")
    return tool


@router.put("/tools/{tool_id}", response_model=ToolResponse)
async def update_tool_by_id(
    tool_id: str, tool: ToolUpdate, db: ConfigDB = Depends(get_config_db)
):
    """Updates a tool"""
    existing = await db.get_tool(tool_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Tool {tool_id} not found")

    if tool.server_id:
        server = await db.get_server(tool.server_id)
        if not server:
            raise HTTPException(
                status_code=404, detail=f"Server {tool.server_id} not found"
            )

    update_data = {k: v for k, v in tool.model_dump().items() if v is not None}
    if update_data:
        await db.update_tool(tool_id, **update_data)

    return await db.get_tool(tool_id)


@router.delete("/tools/{tool_id}", status_code=204)
async def delete_tool_by_id(tool_id: str, db: ConfigDB = Depends(get_config_db)):
    """Deletes a tool"""
    existing = await db.get_tool(tool_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Tool {tool_id} not found")

    await db.delete_tool(tool_id)
    return None


# Tool-Agent association endpoints
@router.post("/tools/agent/{agent_id}/tool/{tool_id}", status_code=204)
async def assign_tool_to_agent(
    agent_id: str,
    tool_id: str,
    assignment: AssignToolRequest,
    db: ConfigDB = Depends(get_config_db),
):
    """Assigns a tool to an agent"""
    agent = await db.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

    tool = await db.get_tool(tool_id)
    if not tool:
        raise HTTPException(status_code=404, detail=f"Tool {tool_id} not found")

    await db.assign_tool_to_agent(agent_id, tool_id, assignment.enabled)
    return None


@router.delete("/tools/agent/{agent_id}/tool/{tool_id}", status_code=204)
async def remove_tool_from_agent(
    agent_id: str, tool_id: str, db: ConfigDB = Depends(get_config_db)
):
    """Removes a tool assignment from an agent"""
    agent = await db.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

    tool = await db.get_tool(tool_id)
    if not tool:
        raise HTTPException(status_code=404, detail=f"Tool {tool_id} not found")

    await db.remove_tool_from_agent(agent_id, tool_id)
    return None


# MCP Configuration endpoints
def _generate_mcp_config(servers, enabled_only=True):
    """
    Converts server configurations from database format to MCPClient format

    Args:
        servers: List of server configurations from database
        enabled_only: Whether to include only enabled servers

    Returns:
        Dict with mcpServers configuration
    """
    # Initialize the mcpServers configuration structure
    mcp_config = {"mcpServers": {}}

    for server in servers:
        # Skip disabled servers if enabled_only is True
        if enabled_only and not server.get("enabled", True):
            continue

        server_name = server["name"]
        server_config = {"transport": server["transport"]}

        # Set transport-specific configurations
        if server["transport"] in ["sse", "streamable_http", "websocket"]:
            if server.get("url"):
                server_config["url"] = server["url"]
            if server.get("headers"):
                server_config["headers"] = server["headers"]
            if server.get("timeout"):
                server_config["timeout"] = server["timeout"]

        elif server["transport"] == "stdio":
            if server.get("command"):
                server_config["command"] = server["command"]
            if server.get("args"):
                server_config["args"] = server["args"]
            if server.get("env_vars"):
                server_config["env"] = server["env_vars"]

        # Add the server configuration to mcpServers
        mcp_config["mcpServers"][server_name] = server_config

    return mcp_config


@router.get("/config", status_code=200)
async def get_mcp_config(
    enabled_only: bool = True, db: ConfigDB = Depends(get_config_db)
):
    """
    Returns the MCP configuration as JSON directly for use with MCPClient.

    Args:
        enabled_only: Only include enabled servers in the response

    Returns:
        JSON configuration for MCPClient
    """
    # Get all servers from database
    servers = await db.list_servers()

    # Generate the configuration
    mcp_config = _generate_mcp_config(servers, enabled_only=enabled_only)

    # Return the configuration
    return mcp_config


@router.post("/export-config", status_code=200)
async def export_mcp_config(
    filepath: Path | None = None,
    enabled_only: bool = True,
    db: ConfigDB = Depends(get_config_db),
):
    """
    Exports all server configurations to a mcp.json file format compatible with MCPClient.

    Args:
        filepath: Optional path where to save the file (default: project root / mcp.json)
        enabled_only: Only include enabled servers in the export

    Returns:
        JSON with export status and file location
    """
    # Get project root (3 levels up from current file)
    project_root = Path(__file__).resolve().parent.parent.parent.parent

    # Default filepath is mcp.json in project root
    if not filepath:
        filepath = project_root / "mcp.json"
    else:
        filepath = Path(filepath)

    # Get all servers from database
    servers = await db.list_servers()

    # Generate the configuration
    mcp_config = _generate_mcp_config(servers, enabled_only=enabled_only)

    # Write the configuration to the file
    with open(filepath, "w") as f:
        json.dump(mcp_config, f, indent=2)

    # Return the status and file path
    return {
        "status": "success",
        "filepath": str(filepath),
        "server_count": len(mcp_config["mcpServers"]),
    }


@router.get("/export-config", response_class=FileResponse)
async def get_mcp_config_file(db: ConfigDB = Depends(get_config_db)):
    """
    Exports MCP configuration and returns the file for download
    """
    # Create temporary export
    temp_file = Path("temp_mcp_config.json")

    # Generate the config file
    await export_mcp_config(filepath=temp_file, db=db)

    def cleanup():
        if temp_file.exists():
            os.remove(temp_file)

    return FileResponse(
        path=temp_file,
        filename="mcp.json",
        media_type="application/json",
        background=BackgroundTask(cleanup),
    )
