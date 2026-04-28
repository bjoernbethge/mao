"""
Helper functions for API endpoints to reduce code duplication.
"""

import logging
from typing import Any

from langchain_core.tools import BaseTool

from ..agents import create_agent, load_mcp_tools
from ..mcp import MCPClient
from .db import ConfigDB

logger = logging.getLogger(__name__)


def response_was_interrupted(response: Any) -> bool:
    return isinstance(response, dict) and bool(response.get("__interrupt__"))


def extract_response_text(response: Any) -> tuple[str, str | None]:
    if response_was_interrupted(response):
        return "Approval required.", _extract_responding_agent_id(response)
    if isinstance(response, dict) and response.get("messages"):
        last_message = response["messages"][-1]
        if hasattr(last_message, "content"):
            return str(last_message.content), getattr(last_message, "name", None)
        if isinstance(last_message, dict):
            return str(last_message.get("content", "")), last_message.get("name")
    if hasattr(response, "content"):
        return str(response.content), getattr(response, "name", None)
    return str(response), None


def _extract_responding_agent_id(response: Any) -> str | None:
    if isinstance(response, dict) and response.get("messages"):
        last_message = response["messages"][-1]
        if hasattr(last_message, "name"):
            return getattr(last_message, "name", None)
        if isinstance(last_message, dict):
            return last_message.get("name")
    if hasattr(response, "name"):
        return getattr(response, "name", None)
    return None


def _db_server_to_mcp_config(server: dict[str, Any]) -> dict[str, Any]:
    config: dict[str, Any] = {"transport": server["transport"]}
    for key in ("url", "command", "args", "headers", "timeout"):
        if server.get(key):
            config[key] = server[key]
    if server.get("env_vars"):
        config["env"] = server["env_vars"]
    return config


async def _build_mcp_client_from_db(
    db: ConfigDB, agent_tools: list[dict[str, Any]]
) -> MCPClient | None:
    server_ids = {t.get("server_id") for t in agent_tools if t.get("server_id")}
    if not server_ids:
        skipped = [t.get("name") for t in agent_tools]
        logger.warning("Tools without server_id cannot execute: %s", skipped)
        return None

    mcp_servers_config: dict[str, Any] = {}
    for sid in server_ids:
        server = await db.get_server(sid)
        if server and server.get("enabled"):
            mcp_servers_config[server["name"]] = _db_server_to_mcp_config(server)
        elif server:
            logger.warning("Server '%s' is disabled, skipping", server.get("name"))

    if not mcp_servers_config:
        return None

    return MCPClient(config={"mcpServers": mcp_servers_config})


async def create_and_start_agent(
    db: ConfigDB,
    agent_id: str,
    agent_config: dict[str, Any],
    active_agents: dict[str, dict[str, Any]],
) -> Any:
    agent_app = await build_agent_app(db, agent_id, agent_config)
    active_agents[agent_id] = {"agent": agent_app, "config": agent_config}
    return agent_app


async def build_agent_app(
    db: ConfigDB,
    agent_id: str,
    agent_config: dict[str, Any],
    extra_tools: list[BaseTool] | None = None,
) -> Any:
    agent_tools = await db.get_agent_tools(agent_id, enabled_only=True)
    mcp_client = await _build_mcp_client_from_db(db, agent_tools) if agent_tools else None
    tools: MCPClient | list[dict[str, Any] | BaseTool] | None = mcp_client
    if extra_tools:
        loaded_tools = await load_mcp_tools(mcp_client)
        tools = list(loaded_tools) + list(extra_tools)

    agent_app = await create_agent(
        provider=agent_config["provider"],
        model_name=agent_config["model_name"],
        agent_name=agent_config["name"],
        system_prompt=agent_config.get("system_prompt"),
        tools=tools,
        skills=agent_config.get("skills"),
        skill_paths=agent_config.get("skill_paths"),
    )
    return agent_app
